# src/htm_monitor/orchestration/decision.py

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, List, Mapping, Optional, Set


@dataclass
class Decision:
    score_key: str
    threshold: float
    method: str = "kofn_window"
    k: int = 1
    window_size: int = 1
    per_model_hits: int = 1
    grouping_enabled: bool = False
    group_k: Optional[int] = None
    model_groups: Optional[Dict[str, List[str]]] = None
    min_consecutive_group_steps: int = 1
    min_alert_len: int = 1

    def __post_init__(self) -> None:
        if self.method != "kofn_window":
            raise ValueError(f"Unsupported decision.method: {self.method}")
        if not isinstance(self.score_key, str) or not self.score_key:
            raise ValueError("Decision.score_key must be a non-empty string")
        if self.window_size < 1:
            raise ValueError("Decision.window_size must be >= 1")
        if self.per_model_hits < 1:
            raise ValueError("Decision.per_model_hits must be >= 1")
        if self.k < 1:
            raise ValueError("Decision.k must be >= 1")
        if self.min_consecutive_group_steps < 1:
            raise ValueError("Decision.min_consecutive_group_steps must be >= 1")
        if self.min_alert_len < 1:
            raise ValueError("Decision.min_alert_len must be >= 1")
        if self.group_k is not None and int(self.group_k) < 1:
            raise ValueError("Decision.group_k must be >= 1 if provided")
 
        self.grouping_enabled = bool(self.grouping_enabled)
        self.group_k = int(self.group_k) if self.group_k is not None else int(self.k)

        raw_groups = self.model_groups or {}
        if not isinstance(raw_groups, dict):
            raise ValueError("Decision.model_groups must be a mapping if provided")

        normalized_groups: Dict[str, List[str]] = {}
        seen_models: Set[str] = set()
        for gname, members in raw_groups.items():
            if not isinstance(gname, str) or not gname:
                raise ValueError("Decision.model_groups keys must be non-empty strings")
            if not isinstance(members, list) or not members:
                raise ValueError(
                    f"Decision.model_groups['{gname}'] must be a non-empty list[str]"
                )

            out_members: List[str] = []
            for m in members:
                if not isinstance(m, str) or not m:
                    raise ValueError(
                        f"Decision.model_groups['{gname}'] entries must be non-empty strings"
                    )
                if m in seen_models:
                    raise ValueError(
                        f"Model '{m}' appears in more than one decision group"
                    )
                seen_models.add(m)
                out_members.append(str(m))
            normalized_groups[str(gname)] = out_members

        self.model_groups = normalized_groups
        self._buf: Dict[str, Deque[int]] = {}
        self._count: Dict[str, int] = {}
        self._group_consecutive_count: int = 0
        self._candidate_episode_len: int = 0

    def _effective_groups(self, model_names: List[str]) -> Dict[str, List[str]]:
        """
        Return decision groups for the models present at this timestep.
        """
        if not self.grouping_enabled:
            return {}

        groups: Dict[str, List[str]] = {}
        assigned: Set[str] = set()

        for gname, members in (self.model_groups or {}).items():
            present = [m for m in members if m in model_names]
            if present:
                groups[gname] = present
                assigned.update(present)

        missing = [m for m in model_names if m not in assigned]
        if missing:
            raise ValueError(
                "Grouping is enabled but some models are not assigned to any group: "
                f"{missing}"
            )

        return groups

    def step(self, model_outputs: Mapping[str, Mapping]) -> Dict:
        if not model_outputs:
            raise ValueError("Decision.step received empty model_outputs")
        
        # lazily initialize per-model buffers
        for m in model_outputs.keys():
            if m not in self._buf:
                self._buf[m] = deque(maxlen=self.window_size)
                self._count[m] = 0

        hot_by_model: Dict[str, int] = {}
        window_hot_by_model: Dict[str, int] = {}

        for m, out in model_outputs.items():
            score = out.get(self.score_key)
            hot = 1 if (isinstance(score, (int, float)) and score >= self.threshold) else 0

            buf = self._buf[m]
            cnt = self._count[m]

            # pop left if deque is full (we track count explicitly)
            if len(buf) == buf.maxlen:
                old = buf[0]
                cnt -= old

            buf.append(hot)
            cnt += hot
            self._count[m] = cnt

            hot_by_model[m] = hot
            window_hot_by_model[m] = 1 if cnt >= self.per_model_hits else 0

        window_hot_by_group: Dict[str, int] = {}
        num_window_hot_groups = 0
        group_alert_eligible_now = 0
        
        if self.grouping_enabled:
            groups = self._effective_groups(list(model_outputs.keys()))

            if self.group_k > len(groups):
                raise ValueError(
                    f"Decision.group_k={self.group_k} exceeds number of active groups={len(groups)}"
                )

            for gname, members in groups.items():
                window_hot_by_group[gname] = 1 if any(
                    window_hot_by_model.get(m, 0) for m in members
                ) else 0

            num_window_hot_groups = int(sum(window_hot_by_group.values()))

            # 🔴 NEW: instantaneous eligibility
            group_alert_eligible_now = 1 if num_window_hot_groups >= self.group_k else 0

            # 🔴 NEW: consecutive enforcement
            if group_alert_eligible_now:
                self._group_consecutive_count += 1
            else:
                self._group_consecutive_count = 0

            # Candidate alert (passes short-term stabilizer)
            candidate_alert = 1 if self._group_consecutive_count >= self.min_consecutive_group_steps else 0

            # Track sustained episode length
            if candidate_alert:
                self._candidate_episode_len += 1
            else:
                self._candidate_episode_len = 0

            # FINAL alert (reportable, sustained consensus)
            alert = 1 if self._candidate_episode_len >= self.min_alert_len else 0

            system_score = num_window_hot_groups / max(1, len(groups))

        else:
            num_window_hot = sum(window_hot_by_model.values())
            alert = 1 if num_window_hot >= self.k else 0
            system_score = num_window_hot / max(1, len(window_hot_by_model))

        out = {
             "system_score": float(system_score),
             "alert": int(alert),
             "hot_by_model": hot_by_model,
             "window_hot_by_model": window_hot_by_model,  # optional but useful
         }
        if self.grouping_enabled:
            out["active_groups"] = list(groups.keys())
            out["window_hot_by_group"] = window_hot_by_group
            out["num_window_hot_groups"] = int(num_window_hot_groups)
            out["group_alert_eligible_now"] = int(group_alert_eligible_now)
            out["group_consecutive_count"] = int(self._group_consecutive_count)
            out["candidate_episode_len"] = int(self._candidate_episode_len)

        return out
