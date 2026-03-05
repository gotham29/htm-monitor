#src/htm_monitor/diagnostics/run_diagnostics.py

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


def _as_int(x: Any) -> Optional[int]:
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float) and math.isfinite(x):
        return int(x)
    return None


def _as_float(x: Any) -> Optional[float]:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)) and math.isfinite(float(x)):
        return float(x)
    return None


def _sdr_sparse(sdr: Any) -> Tuple[int, ...]:
    """
    Robustly extract sparse indices from an htm.bindings.sdr.SDR.
    We avoid depending on one attribute name.
    """
    if sdr is None:
        return tuple()
    # htm.bindings.sdr.SDR typically exposes .sparse as a list/ndarray
    sp = getattr(sdr, "sparse", None)
    if sp is None:
        raise TypeError("Expected SDR-like object with .sparse")
    return tuple(int(i) for i in sp)


def _set_overlap(a: Iterable[int], b: Iterable[int]) -> Tuple[int, float, int]:
    """
    Returns: (intersection_size, jaccard, hamming_symdiff)
    """
    sa = set(a)
    sb = set(b)
    inter = len(sa & sb)
    union = len(sa | sb)
    j = float(inter) / float(union) if union > 0 else 1.0
    ham = len(sa ^ sb)
    return inter, j, ham


@dataclass
class RunDiagnostics:
    """
    Long-lived evidence logger for:
      - per-feature encoding stability (overlap/jaccard/hamming vs prior step)
      - TM prediction effectiveness (predictive-cells hit-rate)

    If writers are None, the corresponding logging is disabled.
    """

    encoding_writer: Optional[csv.DictWriter] = None
    tm_writer: Optional[csv.DictWriter] = None

    # (model, feature) -> previous sparse tuple / value / approx_bucket
    _prev_sparse: Dict[Tuple[str, str], Tuple[int, ...]] = field(default_factory=dict)
    _prev_value: Dict[Tuple[str, str], float] = field(default_factory=dict)
    _prev_bucket: Dict[Tuple[str, str], int] = field(default_factory=dict)

    def record_encoding(
        self,
        *,
        t: int,
        model: str,
        feature: str,
        value: Optional[float],
        sdr: Any,
        resolution: Optional[float] = None,
        min_val: Optional[float] = None,
        approx_bucket: Optional[int] = None,
    ) -> None:
        if self.encoding_writer is None:
            return

        key = (model, feature)
        sp = _sdr_sparse(sdr)

        prev_sp = self._prev_sparse.get(key)
        prev_v = self._prev_value.get(key)
        prev_b = self._prev_bucket.get(key)

        dv = (float(value) - float(prev_v)) if (value is not None and prev_v is not None) else None

        overlap_prev = None
        jaccard_prev = None
        hamming_prev = None
        if prev_sp is not None:
            ov, j, ham = _set_overlap(sp, prev_sp)
            overlap_prev = ov
            jaccard_prev = j
            hamming_prev = ham

        bucket_jump = None
        if approx_bucket is not None and prev_b is not None:
            bucket_jump = 1 if int(approx_bucket) != int(prev_b) else 0

        row = {
            "t": int(t),
            "model": str(model),
            "feature": str(feature),
            "value": value,
            "dv": dv,
            "active_bits": len(sp),
            "overlap_prev": overlap_prev,
            "jaccard_prev": jaccard_prev,
            "hamming_prev": hamming_prev,
            "resolution": resolution,
            "min_val": min_val,
            "approx_bucket": approx_bucket,
            "bucket_jump": bucket_jump,
        }
        self.encoding_writer.writerow(row)

        # update prev state
        self._prev_sparse[key] = sp
        if value is not None:
            self._prev_value[key] = float(value)
        if approx_bucket is not None:
            self._prev_bucket[key] = int(approx_bucket)

    def record_tm(
        self,
        *,
        t: int,
        model: str,
        raw_anomaly: Optional[float],
        pred_cells_prior: Tuple[int, ...],
        active_cells: Tuple[int, ...],
        winner_cells: Tuple[int, ...],
        active_cols: Optional[int] = None,
        pred_cols_prior_count: Optional[int] = None,
        pred_col_hit_rate: Optional[float] = None,
        burst_frac: Optional[float] = None,

    ) -> None:
        if self.tm_writer is None:
            return

        pred_sp = tuple(pred_cells_prior or ())
        act_sp = tuple(active_cells or ())
        win_sp = tuple(winner_cells or ())

        inter = len(set(pred_sp) & set(act_sp))
        hit_rate = float(inter) / float(len(act_sp)) if len(act_sp) > 0 else 0.0

        # A “density” proxy: predictive cells per winner cell (or per active cell if winner is 0)
        denom = len(win_sp) if len(win_sp) > 0 else len(act_sp)
        pred_density = float(len(pred_sp)) / float(denom) if denom > 0 else 0.0

        row = {
            "t": int(t),
            "model": str(model),
            "raw_anomaly": raw_anomaly,
            "pred_cells_prior": len(pred_sp),
            "active_cells": len(act_sp),
            "winner_cells": len(win_sp),
            "pred_hit": int(inter),
            "pred_hit_rate": hit_rate,
            "pred_density": pred_density,
            # Column-level + bursting diagnostics (more interpretable than cell-only):
            "active_cols": int(active_cols) if isinstance(active_cols, int) else None,
            "pred_cols_prior": int(pred_cols_prior_count) if isinstance(pred_cols_prior_count, int) else None,
            "pred_col_hit_rate": float(pred_col_hit_rate) if isinstance(pred_col_hit_rate, (int, float)) else None,
            "burst_frac": float(burst_frac) if isinstance(burst_frac, (int, float)) else None,
        }
        self.tm_writer.writerow(row)


def open_diag_writers(
    *,
    encoding_path: Optional[Path],
    tm_path: Optional[Path],
) -> Tuple[Optional[csv.DictWriter], Optional[csv.DictWriter], Dict[str, Any]]:
    """
    Opens CSVs + returns DictWriters and handles you should close.
    """
    handles: Dict[str, Any] = {}

    enc_w = None
    if encoding_path is not None:
        encoding_path.parent.mkdir(parents=True, exist_ok=True)
        f = open(encoding_path, "w", newline="")
        handles["encoding"] = f
        enc_w = csv.DictWriter(
            f,
            fieldnames=[
                "t", "model", "feature",
                "value", "dv",
                "active_bits",
                "overlap_prev", "jaccard_prev", "hamming_prev",
                "resolution", "min_val", "approx_bucket", "bucket_jump",
            ],
        )
        enc_w.writeheader()

    tm_w = None
    if tm_path is not None:
        tm_path.parent.mkdir(parents=True, exist_ok=True)
        f = open(tm_path, "w", newline="")
        handles["tm"] = f
        tm_w = csv.DictWriter(
            f,
            fieldnames=[
                "t", "model",
                "raw_anomaly",
                "pred_cells_prior", "active_cells", "winner_cells",
                "pred_hit", "pred_hit_rate",
                "pred_density",
                "active_cols",
                "pred_cols_prior",
                "pred_col_hit_rate",
                "burst_frac",
            ],
        )
        tm_w.writeheader()

    return enc_w, tm_w, handles
