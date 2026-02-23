# src/htm_monitor/orchestration/engine.py

from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Union
import numbers
from htm_monitor.htm_src.htm_model import HTMmodel


class Engine:
    def __init__(
        self,
        models: Dict[str, HTMmodel],
        model_sources: Dict[str, Union[str, Sequence[str]]],
        on_missing: str = "skip",  # skip|hold_last
    ):
        self.models = models
        self.model_sources = model_sources
        self.on_missing = str(on_missing).lower()

        # Per-model last good row (for hold_last)
        self._last_good: Dict[str, Dict] = {}
        # NOTE: do NOT validate on_missing here.
        # Tests expect construction to succeed even for unknown values; we reject at runtime in step().

    @staticmethod
    def _first_timestamp(rows: List[Mapping]) -> Optional[str]:
        for r in rows:
            ts = r.get("timestamp") if isinstance(r, Mapping) else None
            if isinstance(ts, str) and ts:
                return ts
        return None

    @staticmethod
    def _all_timestamps(rows: List[Mapping]) -> Set[str]:
        out: Set[str] = set()
        for r in rows:
            ts = r.get("timestamp") if isinstance(r, Mapping) else None
            if isinstance(ts, str) and ts:
                out.add(ts)
        return out

    @staticmethod
    def _required_feature_names(model: Any, rows: List[Mapping]) -> List[str]:
        """
        Determine what keys must be present for model.run(features_data=...).

        Priority (most explicit -> most generic):
          1) model.required_keys (used by tests' _StubModel)
          2) model.feature_names (+ model.feature_timestamp to exclude time in loop)
          3) union of keys present in the incoming rows

        Always includes 'timestamp' first (the rest in sorted order for determinism).
        """
        # 1) test stub convention
        rk = getattr(model, "required_keys", None)
        if isinstance(rk, list) and all(isinstance(x, str) for x in rk):
            # keep order stable but ensure timestamp is first if present
            if "timestamp" in rk:
                return ["timestamp"] + [x for x in rk if x != "timestamp"]
            return list(rk)

        # 2) real HTMmodel convention
        fn = getattr(model, "feature_names", None)
        if isinstance(fn, list) and all(isinstance(x, str) for x in fn) and len(fn) > 0:
            time_name = getattr(model, "feature_timestamp", None)
            non_time = [x for x in fn if x != time_name]
            # feature_names already includes timestamp name; normalize to literal 'timestamp' key used in rows
            if "timestamp" in non_time:
                non_time = [x for x in non_time if x != "timestamp"]
            return ["timestamp"] + non_time

        # 3) generic fallback: infer from rows
        keys: Set[str] = set()
        for r in rows:
            if isinstance(r, Mapping):
                for k in r.keys():
                    if isinstance(k, str) and k:
                        keys.add(k)
        keys.discard("timestamp")
        return ["timestamp"] + sorted(keys)

    @staticmethod
    def _merge_features(
        model: Any,
        rows: List[Mapping],
    ) -> Optional[Dict]:
        """
        Build the features_data dict required by HTMmodel.run():
          - timestamp: must exist; if multiple exist they must match
          - each required feature (non-timestamp): first non-None value across rows

        Returns:
          - None if timestamp missing
          - Dict with timestamp + any found feature values (may be partial)
        """
        ts_set = Engine._all_timestamps(rows)
        if len(ts_set) == 0:
            return None

        if len(ts_set) > 1:
            # Airtight: do not silently merge mismatched timesteps
            raise ValueError(f"Timestamp mismatch across sources: {sorted(ts_set)}")
        ts = next(iter(ts_set))

        merged: Dict = {"timestamp": ts}

        required = Engine._required_feature_names(model, rows)
        for fname in required:
            if fname == "timestamp":
                continue

            val: Any = None
            for r in rows:
                if not isinstance(r, Mapping):
                    continue
                if fname in r:
                    v = r.get(fname)
                    if v is not None:
                        val = v
                        break

            # Only set if found; callers decide how to handle missing required features.
            if val is not None:
                merged[fname] = val

        return merged

    def step(self, rows_by_source: Mapping[str, Mapping], timestep: int) -> Dict[str, Dict[str, float]]:
        if self.on_missing not in ("skip", "hold_last"):
            # Reject at runtime (tests expect __init__ to succeed even for bad values)
            raise ValueError(f"on_missing='{self.on_missing}' is not supported")
        outputs = {}

        for name, model in self.models.items():
            srcs_any = self.model_sources.get(name)
            if srcs_any is None:
                raise ValueError(f"Model '{name}' has no sources configured")

            # normalize str|list into list[str]
            if isinstance(srcs_any, str):
                srcs: List[str] = [srcs_any]
            else:
                srcs = [str(s) for s in list(srcs_any)]
            if len(srcs) == 0:
                raise ValueError(f"Model '{name}' has no sources configured")

            rows: List[Mapping] = []
            for s in srcs:
                r = rows_by_source.get(s)
                if isinstance(r, Mapping):
                    rows.append(r)

            merged = self._merge_features(model, rows)

            if merged is None:
                if self.on_missing == "hold_last":
                    prev = self._last_good.get(name)
                    if prev is None:
                        continue
                    merged = prev
                elif self.on_missing == "skip":
                    continue
                else:
                    # Tests expect unknown modes to be rejected at runtime, not construction time
                    raise ValueError(f"on_missing='{self.on_missing}' is not supported")
            else:
                # merged has a valid timestamp; ensure all required features exist
                required = self._required_feature_names(model, rows)
                missing = [k for k in required if k != "timestamp" and k not in merged]
                if missing:
                    if self.on_missing == "hold_last":
                        prev = self._last_good.get(name)
                        if prev is None:
                            continue
                        # Fill missing from last good, but keep current timestamp
                        filled = dict(merged)
                        for k in missing:
                            if k in prev and prev[k] is not None:
                                filled[k] = prev[k]
                        # If still missing, we can't run.
                        still_missing = [k for k in missing if k not in filled]
                        if still_missing:
                            continue
                        merged = filled
                    elif self.on_missing == "skip":
                        continue
                    else:
                        raise ValueError(f"on_missing='{self.on_missing}' is not supported")

            raw, likelihood, pcount = model.run(
                features_data=merged,
                timestep=timestep,
                learn=True,
            )

            # ---- Optional HTMmodel-only metrics ----
            # Unit tests use _StubModel (run-only). Real HTMmodel exposes:
            #   last_anomaly_probability(), last_log_likelihood()
            p_val: Optional[float] = None
            ll_val: Optional[float] = None
            get_p = getattr(model, "last_anomaly_probability", None)
            if callable(get_p):
                try:
                    pv = get_p()
                    if isinstance(pv, numbers.Real):
                       p_val = float(pv)
                except Exception:
                    p_val = None

            get_ll = getattr(model, "last_log_likelihood", None)
            if callable(get_ll):
                try:
                    lv = get_ll()
                    if isinstance(lv, numbers.Real):
                        ll_val = float(lv)
                except Exception:
                    ll_val = None

            outputs[name] = {
                "raw": raw,
                "likelihood": likelihood,  # legacy: this is computeLogLikelihood(p)
                "p": p_val,  # legacy short name (may be None for stub models)
                "anomaly_probability": p_val,
                "log_likelihood": ll_val,
                "pcount": pcount,
            }

            # update last-good after a successful compute
            self._last_good[name] = dict(merged)

        return outputs
