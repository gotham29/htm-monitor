# src/htm_monitor/orchestration/engine.py

from typing import Dict, Mapping, Optional

from htm_monitor.htm_src.htm_model import HTMmodel


class Engine:
    def __init__(
        self,
        models: Dict[str, HTMmodel],
        model_sources: Dict[str, str],
        on_missing: str = "skip",  # skip|hold_last (hold_last can come later)
    ):
        self.models = models
        self.model_sources = model_sources
        self.on_missing = on_missing

    def step(self, rows_by_source: Mapping[str, Mapping], timestep: int) -> Dict[str, Dict[str, float]]:
        outputs = {}

        for name, model in self.models.items():
            src = self.model_sources[name]
            row = rows_by_source.get(src)
            if row is None:
                if self.on_missing == "skip":
                    continue
                # "hold_last" intentionally not implemented yet (keep it tight)
                raise ValueError(f"on_missing='{self.on_missing}' is not supported")

            raw, likelihood, pcount = model.run(
                features_data=row,
                timestep=timestep,
                learn=True,
            )
            outputs[name] = {
                "raw": raw,
                "likelihood": likelihood,
                "pcount": pcount,
            }

        return outputs
