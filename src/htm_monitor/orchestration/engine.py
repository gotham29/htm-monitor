# src/htm_monitor/orchestration/engine.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from htm_monitor.htm_src.htm_model import HTMmodel
from htm_monitor.diagnostics.run_diagnostics import RunDiagnostics


@dataclass(frozen=True)
class EngineConfig:
    on_missing: str = "skip"  # "skip" | "hold_last"


class Engine:
    def __init__(
        self,
        models: Dict[str, HTMmodel],
        model_sources: Dict[str, Sequence[str]],
        model_features: Dict[str, Sequence[str]],
        *,
        on_missing: str = "skip",
    ):
        on_missing = str(on_missing).lower()
        if on_missing not in ("skip", "hold_last"):
            raise ValueError("Engine.on_missing must be 'skip' or 'hold_last'")

        self.models = models
        self.model_sources = {m: list(srcs) for m, srcs in model_sources.items()}
        self.model_features = {m: list(feats) for m, feats in model_features.items()}
        self.on_missing = on_missing

        # hold_last memory (model -> last merged feature dict)
        self._last_good: Dict[str, Dict[str, Any]] = {}

        # basic sanity: every model has sources + features
        for m in self.models.keys():
            if m not in self.model_sources or not self.model_sources[m]:
                raise ValueError(f"Model '{m}' has no sources configured")
            if m not in self.model_features or not self.model_features[m]:
                raise ValueError(f"Model '{m}' has no features configured")
            # if "timestamp" not in self.model_features[m]:
            #     raise ValueError(f"Model '{m}' features must include 'timestamp'")

    @staticmethod
    def _current_timestamp(rows_by_source: Mapping[str, Mapping[str, Any]]) -> Optional[str]:
        for r in rows_by_source.values():
            if isinstance(r, Mapping):
                ts = r.get("timestamp")
                if isinstance(ts, str) and ts:
                    return ts
        return None

    def _merge_for_model(self, model_name: str, rows_by_source: Mapping[str, Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Returns merged features dict for this model at the current timestep, or None if we should skip.
        """
        ts = self._current_timestamp(rows_by_source)
        if ts is None:
            return None

        feats = self.model_features[model_name]
        srcs = self.model_sources[model_name]

        merged: Dict[str, Any] = {"timestamp": ts}

        # pull feature values from sources in order; first non-None wins
        for f in feats:
            if f == "timestamp":
                continue

            v = None
            for s in srcs:
                r = rows_by_source.get(s)
                if isinstance(r, Mapping) and r.get(f) is not None:
                    v = r.get(f)
                    break

            if v is not None:
                merged[f] = v

        # if any non-timestamp feature is missing, apply on_missing
        missing = [f for f in feats if f != "timestamp" and f not in merged]
        if not missing:
            return merged

        if self.on_missing == "skip":
            return None

        # hold_last: fill missing from last_good if available
        prev = self._last_good.get(model_name)
        if prev is None:
            return None

        filled = dict(merged)
        for f in missing:
            if f in prev and prev[f] is not None:
                filled[f] = prev[f]

        still_missing = [f for f in feats if f != "timestamp" and f not in filled]
        if still_missing:
            return None

        return filled

    def step(
        self,
        rows_by_source: Mapping[str, Mapping[str, Any]],
        timestep: int,
        *,
        diag: Optional[RunDiagnostics] = None,
        learn: bool = True,
    ) -> Dict[str, Dict[str, Any]]:
        outputs: Dict[str, Dict[str, Any]] = {}

        for name, model in self.models.items():
            merged = self._merge_for_model(name, rows_by_source)
            if merged is None:
                continue

            if diag is not None and (diag.encoding_writer is not None or diag.tm_writer is not None):
                raw, likelihood, pcount, diag_payload = model.run_with_diagnostics(
                    features_data=merged,
                    timestep=timestep,
                    learn=bool(learn),
                )
                # per-feature encoding evidence
                if diag.encoding_writer is not None:
                    for feat_name, ed in (diag_payload.get("encodings") or {}).items():
                        diag.record_encoding(
                            t=timestep,
                            model=name,
                            feature=feat_name,
                            value=ed.get("value"),
                            sdr=ed.get("sdr"),
                            resolution=ed.get("resolution"),
                            min_val=ed.get("min_val"),
                            approx_bucket=ed.get("approx_bucket"),
                        )
                # tm evidence
                if diag.tm_writer is not None:
                    tm = diag_payload.get("tm") or {}
                    diag.record_tm(
                        t=timestep,
                        model=name,
                        raw_anomaly=float(raw) if isinstance(raw, (int, float)) else None,
                        pred_cells_prior=tm.get("pred_cells_prior"),
                        active_cells=tm.get("active_cells"),
                        winner_cells=tm.get("winner_cells"),
                        active_cols=tm.get("active_cols"),
                        pred_cols_prior_count=tm.get("pred_cols_prior_count"),
                        pred_col_hit_rate=tm.get("pred_col_hit_rate"),
                        burst_frac=tm.get("burst_frac"),
                    )
            else:
                raw, likelihood, pcount = model.run(
                    features_data=merged,
                    timestep=timestep,
                    learn=bool(learn),
                )

            # These are stable for HTMmodel; for test stubs they can be absent (so: None)
            p = model.last_anomaly_probability() if hasattr(model, "last_anomaly_probability") else None
            ll = model.last_log_likelihood() if hasattr(model, "last_log_likelihood") else None

            out = {
                "raw": raw,
                "likelihood": likelihood,
                "p": float(p) if isinstance(p, (int, float)) else None,
                "anomaly_probability": float(p) if isinstance(p, (int, float)) else None,
                "log_likelihood": float(ll) if isinstance(ll, (int, float)) else None,
                "pcount": pcount,
            }
            outputs[name] = out
            self._last_good[name] = dict(merged)

        return outputs

