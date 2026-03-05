# src/htm_monitor/orchestration/decision.py

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, Mapping, Optional


@dataclass
class Decision:
    score_key: str
    threshold: float
    method: str = "kofn_window"
    k: int = 1
    window_size: int = 1
    per_model_hits: int = 1

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

        self._buf: Dict[str, Deque[int]] = {}
        self._count: Dict[str, int] = {}

    def step(self, model_outputs: Mapping[str, Mapping]) -> Dict:
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

        num_window_hot = sum(window_hot_by_model.values())
        alert = 1 if num_window_hot >= self.k else 0

        system_score = num_window_hot / max(1, len(window_hot_by_model))

        return {
            "system_score": float(system_score),
            "alert": int(alert),
            "hot_by_model": hot_by_model,
            "window_hot_by_model": window_hot_by_model,  # optional but useful
        }
