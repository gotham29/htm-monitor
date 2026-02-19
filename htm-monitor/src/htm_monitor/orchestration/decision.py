# src/htm_monitor/orchestration/decision.py

from typing import Dict, Optional


class Decision:
    def __init__(self, threshold: float = 0.999, method: str = "max", k: Optional[int] = None):
        self.threshold = threshold
        self.method = method
        self.k = k

    def step(self, model_outputs: Dict[str, Dict[str, float]]) -> Dict:
        likelihoods = [m["likelihood"] for m in model_outputs.values()]

        if not likelihoods:
            return {"system_score": 0.0, "alert": False}

        if self.method == "mean":
            system_score = sum(likelihoods) / len(likelihoods)
            alert = system_score >= self.threshold
        elif self.method == "kofn":
            if self.k is None:
                raise ValueError("Decision(method='kofn') requires k")
            count = sum(1 for x in likelihoods if x >= self.threshold)
            system_score = float(count) / float(len(likelihoods))
            alert = count >= int(self.k)
        else:  # default max
            system_score = max(likelihoods)

            alert = system_score >= self.threshold

        return {
            "system_score": system_score,
            "alert": alert,
        }
