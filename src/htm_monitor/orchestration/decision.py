from __future__ import annotations

from collections import deque
from typing import Dict, Any, List


class GroupedConsensusDecision:
    """
    Phase 2 decision logic:

    - Per-model:
        instant_hot (threshold)
        warmth (rolling mean of instant_hot)

    - Per-group:
        group_hot if:
            (instant support) OR (warmth support)

    - System:
        system_hot if >= group_k groups hot

    - Alert:
        alert if system_hot persists for min_system_len steps

    This replaces:
        - window_hot
        - per_model_hits
        - grouping consecutive logic
        - candidate episodes
    """

    def __init__(self, config: Dict[str, Any]):
        decision = config.get("decision") or {}
        if not isinstance(decision, dict):
            raise ValueError("config.decision must be a mapping")

        # --- Core params ---
        self.threshold: float = float(decision.get("threshold", 0.99))
        self.score_key: str = str(decision.get("score_key", "p"))

        # --- Warmth ---
        model_warmth_cfg = decision.get("model_warmth") or {}
        self.warmth_window_size: int = int(model_warmth_cfg.get("window_size", 8))
        if self.warmth_window_size <= 0:
            raise ValueError("model_warmth.window_size must be > 0")

        # --- System ---
        system_cfg = decision.get("system") or {}
        self.group_k: int = int(system_cfg.get("group_k", 1))
        self.min_system_len: int = int(system_cfg.get("min_system_len", 1))

        if self.group_k <= 0:
            raise ValueError("system.group_k must be > 0")
        if self.min_system_len <= 0:
            raise ValueError("system.min_system_len must be > 0")

        # --- Groups ---
        groups_cfg = decision.get("groups") or {}
        if not isinstance(groups_cfg, dict) or not groups_cfg:
            raise ValueError("decision.groups must be a non-empty mapping")

        self.groups: Dict[str, Dict[str, Any]] = {}
        self.all_models: List[str] = []

        for gname, gspec in groups_cfg.items():
            if not isinstance(gspec, dict):
                raise ValueError(f"group '{gname}' must be a mapping")

            members = gspec.get("members") or []
            if not isinstance(members, list) or not members:
                raise ValueError(f"group '{gname}' must have non-empty members list")

            min_instant_members = int(gspec.get("min_instant_members", 1))
            min_group_warmth = float(gspec.get("min_group_warmth", 0.5))

            if min_instant_members <= 0:
                raise ValueError(f"group '{gname}': min_instant_members must be > 0")
            if not (0.0 <= min_group_warmth <= 1.0):
                raise ValueError(f"group '{gname}': min_group_warmth must be in [0,1]")

            self.groups[gname] = {
                "members": list(members),
                "min_instant_members": min_instant_members,
                "min_group_warmth": min_group_warmth,
            }

            for m in members:
                if m not in self.all_models:
                    self.all_models.append(m)

        # --- Model buffers ---
        self.model_buffers: Dict[str, deque] = {
            m: deque(maxlen=self.warmth_window_size) for m in self.all_models
        }

        # --- System state ---
        self.system_hot_streak: int = 0

    # ---------------------------------------------------------
    # Core step
    # ---------------------------------------------------------
    def step(self, model_outputs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        model_outputs:
            {
                "model_name": {
                    "p": float,
                    ...
                }
            }
        """

        # -------------------------
        # 1. Instant hot
        # -------------------------
        instant_hot: Dict[str, int] = {}

        for m in self.all_models:
            out = model_outputs.get(m)
            if out is None:
                instant_hot[m] = 0
                continue

            score = float(out.get(self.score_key, 0.0))
            instant_hot[m] = 1 if score >= self.threshold else 0

        # -------------------------
        # 2. Update buffers
        # -------------------------
        for m in self.all_models:
            self.model_buffers[m].append(instant_hot[m])

        # -------------------------
        # 3. Model warmth
        # -------------------------
        model_warmth: Dict[str, float] = {}

        for m, buf in self.model_buffers.items():
            if len(buf) == 0:
                model_warmth[m] = 0.0
            else:
                model_warmth[m] = sum(buf) / float(len(buf))

        # -------------------------
        # 4. Group stats
        # -------------------------
        group_instant_count: Dict[str, int] = {}
        group_warmth: Dict[str, float] = {}
        group_hot: Dict[str, int] = {}

        for gname, gspec in self.groups.items():
            members = gspec["members"]

            inst_count = sum(instant_hot[m] for m in members)
            group_instant_count[gname] = inst_count

            if members:
                g_warmth = sum(model_warmth[m] for m in members) / float(len(members))
            else:
                g_warmth = 0.0

            group_warmth[gname] = g_warmth

            is_hot = (
                inst_count >= gspec["min_instant_members"]
                or g_warmth >= gspec["min_group_warmth"]
            )

            group_hot[gname] = 1 if is_hot else 0

        # -------------------------
        # 5. System
        # -------------------------
        system_hot_count = sum(group_hot.values())
        system_hot = 1 if system_hot_count >= self.group_k else 0

        if system_hot:
            self.system_hot_streak += 1
        else:
            self.system_hot_streak = 0

        alert = 1 if self.system_hot_streak >= self.min_system_len else 0

        # -------------------------
        # Return payload
        # -------------------------
        return {
            "alert": alert,
            "system_hot": system_hot,
            "system_hot_count": system_hot_count,
            "system_hot_streak": self.system_hot_streak,

            "instant_hot_by_model": dict(instant_hot),
            "model_warmth_by_model": dict(model_warmth),

            "group_instant_count": dict(group_instant_count),
            "group_warmth": dict(group_warmth),
            "group_hot": dict(group_hot),
        }
