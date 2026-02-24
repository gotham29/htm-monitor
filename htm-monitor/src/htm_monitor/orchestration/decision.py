# src/htm_monitor/orchestration/decision.py

from __future__ import annotations

from collections import deque
import numbers
from typing import Deque, Dict, Optional, Set


class Decision:
    def __init__(
        self,
        threshold: float = 0.999,
        method: str = "max",
        k: Optional[int] = None,
        window_size: int = 1,
        per_model_hits: int = 1,
        score_key: str = "likelihood",
    ):
        # ---- Basic validation / normalization (airtight) ----
        if not isinstance(threshold, numbers.Real):
            raise ValueError("threshold must be a real number")
        self.threshold = float(threshold)
        if not isinstance(score_key, str) or not score_key:
            raise ValueError("score_key must be a non-empty string")
        self.score_key = score_key

        self.method = str(method).lower()
        allowed = {"max", "mean", "kofn", "kofn_window"}
        if self.method not in allowed:
            raise ValueError(
                f"Unknown Decision.method: {method}. Must be one of: {sorted(allowed)}"
            )

        if k is not None:
            if not isinstance(k, numbers.Integral):
                raise ValueError("k must be an int when provided")
            if int(k) <= 0:
                raise ValueError("k must be > 0")
        self.k = int(k) if k is not None else None
        self.window_size = int(window_size)
        self.per_model_hits = int(per_model_hits)

        if self.window_size <= 0:
            raise ValueError("window_size must be > 0")
        if self.per_model_hits <= 0:
            raise ValueError("per_model_hits must be > 0")
        if self.per_model_hits > self.window_size:
            raise ValueError("per_model_hits must be <= window_size")

        # Rolling hit buffers per model for kofn_window.
        # Each deque stores booleans: (likelihood >= threshold) for recent timesteps.
        self._hit_windows: Dict[str, Deque[bool]] = {}
        self._seen_models: Set[str] = set()

    def _ensure_model(self, model_name: str) -> None:
        if model_name in self._hit_windows:
            return
        self._hit_windows[model_name] = deque(maxlen=self.window_size)
        self._seen_models.add(model_name)

    def _kofn_window(self, model_outputs: Dict[str, Dict[str, float]]) -> Dict:
        """
        Windowed k-of-n decision:
          - hit := (likelihood >= threshold) at this timestep
          - model is "hot" if (#hits in last W steps) >= per_model_hits
          - system alert if (#hot models) >= k
        """
        if self.k is None:
            raise ValueError("Decision(method='kofn_window') requires k")

        if not model_outputs and not self._seen_models:
            return {"system_score": 0.0, "alert": False, "hot_by_model": {}}

        hot = 0
        hot_by_model: Dict[str, bool] = {}
        # Important: models may be missing on a timestep (e.g., Engine skipped them).
        # We treat missing as a MISS (append False) so hotness cannot "stick" silently.
        all_names = set(self._seen_models) | set(model_outputs.keys())
        for name in sorted(all_names):
            self._ensure_model(name)
            out = model_outputs.get(name) or {}
            lik = out.get(self.score_key)
            is_hit = bool(isinstance(lik, numbers.Real) and lik >= self.threshold)
            self._hit_windows[name].append(is_hit)

            hits = sum(1 for b in self._hit_windows[name] if b)
            is_hot = hits >= self.per_model_hits
            hot_by_model[name] = is_hot
            if is_hot:
                hot += 1

        denom = float(len(all_names) or 1)
        system_score = float(hot) / denom
        alert = hot >= int(self.k)
        return {"system_score": system_score, "alert": alert, "hot_by_model": hot_by_model}

    def step(self, model_outputs: Dict[str, Dict[str, float]]) -> Dict:
        # kofn_window needs per-model history, so handle it first.
        if self.method == "kofn_window":
            return self._kofn_window(model_outputs)

        vals = [
            m.get(self.score_key)
            for m in model_outputs.values()
            if isinstance(m.get(self.score_key), numbers.Real)
        ]

        if not vals:
            return {"system_score": 0.0, "alert": False}

        if self.method == "mean":
            system_score = sum(vals) / len(vals)
            alert = system_score >= self.threshold
        elif self.method == "kofn":
            if self.k is None:
                raise ValueError("Decision(method='kofn') requires k")
            count = sum(1 for x in vals if x >= self.threshold)
            system_score = float(count) / float(len(vals))
            alert = count >= int(self.k)
        else:  # default max
            system_score = max(vals)

            alert = system_score >= self.threshold

        return {
            "system_score": system_score,
            "alert": alert,
        }
