from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _require_columns(df: pd.DataFrame, cols: Sequence[str], ctx: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{ctx} missing required columns: {missing}")


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        xf = float(x)
    except Exception:
        return None
    if np.isnan(xf):
        return None
    return float(xf)


def _mad(arr: np.ndarray) -> float:
    if arr.size == 0:
        return 0.0
    med = float(np.median(arr))
    return float(np.median(np.abs(arr - med)))


def _robust_scale(arr: np.ndarray) -> float:
    """
    Robust sigma-like scale from MAD.
    """
    mad = _mad(arr)
    return float(1.4826 * mad)


def _clip_nonnegative(x: float) -> float:
    return float(x) if x > 0.0 else 0.0


def _severity_from_score(score: Optional[float]) -> str:
    if score is None:
        return "unknown"
    s = float(score)
    if s >= 4.0:
        return "strong"
    if s >= 2.5:
        return "moderate"
    if s >= 1.5:
        return "weak"
    return "none"


def _series_stats(x: np.ndarray) -> Dict[str, Optional[float]]:
    if x.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "mad": None,
            "min": None,
            "max": None,
        }
    return {
        "count": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x)),
        "mad": float(_mad(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def _window_slice(
    df: pd.DataFrame,
    *,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> pd.DataFrame:
    return df[(df["ts"] >= start_ts) & (df["ts"] <= end_ts)].copy()


def _baseline_slice(
    df: pd.DataFrame,
    *,
    start_ts: pd.Timestamp,
    baseline_days: int,
) -> pd.DataFrame:
    baseline_start = start_ts - pd.Timedelta(days=int(baseline_days))
    return df[(df["ts"] >= baseline_start) & (df["ts"] < start_ts)].copy()


def _infer_step_minutes(ts: pd.Series) -> Optional[float]:
    ts = ts.dropna()
    dt = ts.sort_values().diff().dropna()
    dt = dt[dt > pd.Timedelta(0)]
    if dt.empty:
        return None
    minutes = dt.median().total_seconds() / 60.0
    return float(minutes) if minutes > 0 else None


def _same_hour_baseline_values(
    df: pd.DataFrame,
    *,
    value_col: str,
    target_hours: Sequence[int],
) -> np.ndarray:
    if df.empty:
        return np.asarray([], dtype=float)
    sub = df[df["ts"].dt.hour.isin(list(target_hours))].copy()
    vals = pd.to_numeric(sub[value_col], errors="coerce").dropna().to_numpy(dtype=float)
    return vals


def _residualize_against_hourly_median(
    df: pd.DataFrame,
    *,
    value_col: str,
    baseline_df: pd.DataFrame,
) -> np.ndarray:
    if df.empty:
        return np.asarray([], dtype=float)

    hour_medians = (
        baseline_df.assign(_val=pd.to_numeric(baseline_df[value_col], errors="coerce"))
        .dropna(subset=["_val"])
        .groupby(baseline_df["ts"].dt.hour)["_val"]
        .median()
        .to_dict()
    )

    vals: List[float] = []
    for _, row in df.iterrows():
        v = _safe_float(row[value_col])
        if v is None:
            continue
        h = int(row["ts"].hour)
        ref = _safe_float(hour_medians.get(h))
        if ref is None:
            continue
        vals.append(float(v - ref))
    return np.asarray(vals, dtype=float)


def _estimate_cycle_amplitude(x: np.ndarray) -> Optional[float]:
    if x.size < 4:
        return None
    q90 = float(np.quantile(x, 0.90))
    q10 = float(np.quantile(x, 0.10))
    return float(q90 - q10)


def _shape_signature(x: np.ndarray, bins: int = 24) -> Optional[np.ndarray]:
    """
    Compress a window into a fixed-length normalized shape signature.
    This is a simple resampling-based surrogate for waveform shape.
    """
    if x.size < 4:
        return None
    xp = np.linspace(0.0, 1.0, num=x.size)
    fp = x.astype(float)
    grid = np.linspace(0.0, 1.0, num=int(bins))
    sig = np.interp(grid, xp, fp)
    sig = sig - np.mean(sig)
    scale = np.std(sig)
    if scale > 0:
        sig = sig / scale
    return sig.astype(float)


def _corr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if a.size < 3 or b.size < 3 or a.size != b.size:
        return None
    sa = float(np.std(a))
    sb = float(np.std(b))
    if sa <= 0.0 or sb <= 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


@dataclass(frozen=True)
class SignalValidation:
    signal: str
    window_stats: Dict[str, Optional[float]]
    baseline_stats: Dict[str, Optional[float]]
    level_shift_score: Optional[float]
    level_shift_direction: str
    volatility_shift_score: Optional[float]
    residual_burst_score: Optional[float]
    amplitude_shift_score: Optional[float]
    shape_shift_score: Optional[float]
    signatures: Dict[str, str]


def validate_signal_window(
    signal_df: pd.DataFrame,
    *,
    signal_name: str,
    value_col: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    baseline_days: int = 7,
) -> SignalValidation:
    _require_columns(signal_df, ["ts", value_col], f"signal_df[{signal_name}]")

    df = signal_df.copy()
    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts", kind="mergesort").reset_index(drop=True)

    win = _window_slice(df, start_ts=start_ts, end_ts=end_ts)
    base = _baseline_slice(df, start_ts=start_ts, baseline_days=baseline_days)

    win_vals = pd.to_numeric(win[value_col], errors="coerce").dropna().to_numpy(dtype=float)
    base_vals = pd.to_numeric(base[value_col], errors="coerce").dropna().to_numpy(dtype=float)

    window_stats = _series_stats(win_vals)
    baseline_stats = _series_stats(base_vals)

    level_shift_score: Optional[float] = None
    level_shift_direction = "unknown"
    volatility_shift_score: Optional[float] = None
    residual_burst_score: Optional[float] = None
    amplitude_shift_score: Optional[float] = None
    shape_shift_score: Optional[float] = None

    if win_vals.size > 0 and base_vals.size > 0:
        base_loc = float(np.median(base_vals))
        base_scale = _robust_scale(base_vals)
        if base_scale > 0.0:
            delta = float(np.median(win_vals) - base_loc)
            level_shift_score = abs(delta) / base_scale
            level_shift_direction = "up" if delta > 0 else ("down" if delta < 0 else "flat")

        win_scale = _robust_scale(win_vals)
        if base_scale > 0.0:
            volatility_shift_score = abs(win_scale - base_scale) / base_scale

        target_hours = sorted(set(win["ts"].dt.hour.tolist()))
        base_same_hour = _same_hour_baseline_values(base, value_col=value_col, target_hours=target_hours)
        win_resid = _residualize_against_hourly_median(win, value_col=value_col, baseline_df=base)
        if base_same_hour.size > 0 and win_resid.size > 0:
            base_same_hour_scale = _robust_scale(base_same_hour)
            win_resid_scale = _robust_scale(win_resid)
            if base_same_hour_scale > 0.0:
                residual_burst_score = win_resid_scale / base_same_hour_scale

        win_amp = _estimate_cycle_amplitude(win_vals)
        base_amp = _estimate_cycle_amplitude(base_same_hour if base_same_hour.size > 0 else base_vals)
        if win_amp is not None and base_amp is not None and base_amp > 0.0:
            amplitude_shift_score = abs(float(win_amp) - float(base_amp)) / float(base_amp)

        win_sig = _shape_signature(win_vals)
        base_ref_vals = base_same_hour if base_same_hour.size >= 4 else base_vals
        base_sig = _shape_signature(base_ref_vals)
        if win_sig is not None and base_sig is not None and len(win_sig) == len(base_sig):
            c = _corr(win_sig, base_sig)
            if c is not None:
                shape_shift_score = 1.0 - c

    signatures = {
        "level_shift": _severity_from_score(level_shift_score),
        "volatility_shift": _severity_from_score(volatility_shift_score),
        "residual_burst": _severity_from_score(residual_burst_score),
        "amplitude_shift": _severity_from_score(amplitude_shift_score),
        "shape_shift": _severity_from_score(shape_shift_score),
    }

    return SignalValidation(
        signal=signal_name,
        window_stats=window_stats,
        baseline_stats=baseline_stats,
        level_shift_score=level_shift_score,
        level_shift_direction=level_shift_direction,
        volatility_shift_score=volatility_shift_score,
        residual_burst_score=residual_burst_score,
        amplitude_shift_score=amplitude_shift_score,
        shape_shift_score=shape_shift_score,
        signatures=signatures,
    )


def validate_cross_signal_divergence(
    signal_map: Mapping[str, pd.DataFrame],
    *,
    left_signal: str,
    right_signal: str,
    value_col_left: str,
    value_col_right: str,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    baseline_days: int = 7,
) -> Dict[str, Any]:
    if left_signal not in signal_map or right_signal not in signal_map:
        raise ValueError("Both left_signal and right_signal must exist in signal_map")

    lhs = signal_map[left_signal].copy()
    rhs = signal_map[right_signal].copy()

    _require_columns(lhs, ["ts", value_col_left], f"signal_map[{left_signal}]")
    _require_columns(rhs, ["ts", value_col_right], f"signal_map[{right_signal}]")

    lhs["ts"] = pd.to_datetime(lhs["ts"], errors="coerce")
    rhs["ts"] = pd.to_datetime(rhs["ts"], errors="coerce")
    lhs[value_col_left] = pd.to_numeric(lhs[value_col_left], errors="coerce")
    rhs[value_col_right] = pd.to_numeric(rhs[value_col_right], errors="coerce")

    merged = lhs[["ts", value_col_left]].merge(
        rhs[["ts", value_col_right]],
        on="ts",
        how="inner",
    ).dropna().sort_values("ts", kind="mergesort").reset_index(drop=True)

    win = _window_slice(merged, start_ts=start_ts, end_ts=end_ts)
    base = _baseline_slice(merged, start_ts=start_ts, baseline_days=baseline_days)

    if win.empty or base.empty:
        return {
            "pair": [left_signal, right_signal],
            "gap_shift_score": None,
            "correlation_shift_score": None,
            "signatures": {
                "gap_shift": "unknown",
                "correlation_shift": "unknown",
            },
        }

    base_gap = (base[value_col_left] - base[value_col_right]).to_numpy(dtype=float)
    win_gap = (win[value_col_left] - win[value_col_right]).to_numpy(dtype=float)

    gap_shift_score: Optional[float] = None
    correlation_shift_score: Optional[float] = None

    base_gap_scale = _robust_scale(base_gap)
    if base_gap_scale > 0.0:
        gap_shift_score = abs(float(np.median(win_gap)) - float(np.median(base_gap))) / base_gap_scale

    base_corr = _corr(
        base[value_col_left].to_numpy(dtype=float),
        base[value_col_right].to_numpy(dtype=float),
    )
    win_corr = _corr(
        win[value_col_left].to_numpy(dtype=float),
        win[value_col_right].to_numpy(dtype=float),
    )
    if base_corr is not None and win_corr is not None:
        correlation_shift_score = abs(float(win_corr) - float(base_corr))

    return {
        "pair": [left_signal, right_signal],
        "gap_shift_score": gap_shift_score,
        "correlation_shift_score": correlation_shift_score,
        "signatures": {
            "gap_shift": _severity_from_score(gap_shift_score),
            "correlation_shift": _severity_from_score(
                None if correlation_shift_score is None else 4.0 * correlation_shift_score
            ),
        },
    }


def summarize_validation(
    per_signal: Sequence[SignalValidation],
    *,
    cross_signal: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    signal_rows = list(per_signal)
    cross_rows = list(cross_signal) if cross_signal is not None else []

    strong_flags: List[str] = []
    moderate_flags: List[str] = []

    for row in signal_rows:
        for name, sev in row.signatures.items():
            tag = f"{row.signal}:{name}"
            if sev == "strong":
                strong_flags.append(tag)
            elif sev == "moderate":
                moderate_flags.append(tag)

    for row in cross_rows:
        for name, sev in (row.get("signatures") or {}).items():
            pair = row.get("pair") or []
            tag = f"{pair}:{name}"
            if sev == "strong":
                strong_flags.append(tag)
            elif sev == "moderate":
                moderate_flags.append(tag)

    return {
        "strong_flags": strong_flags,
        "moderate_flags": moderate_flags,
        "signal_count": int(len(signal_rows)),
        "cross_signal_count": int(len(cross_rows)),
        "looks_structured_anomaly": bool(len(strong_flags) > 0 or len(moderate_flags) >= 2),
    }


def validate_anomaly_window(
    signal_map: Mapping[str, pd.DataFrame],
    *,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    baseline_days: int = 7,
    cross_signal_pairs: Optional[Sequence[Tuple[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Generic anomaly-window validator.

    Expected signal_map contract:
      {
        "demand": DataFrame with columns ["ts", "demand"],
        "net_generation": DataFrame with columns ["ts", "net_generation"],
        ...
      }

    Each dataframe must contain:
      - ts
      - one non-ts value column matching the key
    """
    if not signal_map:
        raise ValueError("signal_map must be non-empty")

    validated: List[SignalValidation] = []

    for signal_name, df in signal_map.items():
        value_cols = [c for c in df.columns if c != "ts"]
        if len(value_cols) != 1:
            raise ValueError(
                f"signal_map[{signal_name}] must have exactly one value column besides 'ts'; "
                f"got {value_cols}"
            )
        validated.append(
            validate_signal_window(
                df,
                signal_name=str(signal_name),
                value_col=str(value_cols[0]),
                start_ts=start_ts,
                end_ts=end_ts,
                baseline_days=baseline_days,
            )
        )

    cross_rows: List[Dict[str, Any]] = []
    for pair in cross_signal_pairs or []:
        left_signal, right_signal = str(pair[0]), str(pair[1])
        left_cols = [c for c in signal_map[left_signal].columns if c != "ts"]
        right_cols = [c for c in signal_map[right_signal].columns if c != "ts"]
        if len(left_cols) != 1 or len(right_cols) != 1:
            raise ValueError("cross-signal inputs must each have exactly one value column besides 'ts'")
        cross_rows.append(
            validate_cross_signal_divergence(
                signal_map,
                left_signal=left_signal,
                right_signal=right_signal,
                value_col_left=str(left_cols[0]),
                value_col_right=str(right_cols[0]),
                start_ts=start_ts,
                end_ts=end_ts,
                baseline_days=baseline_days,
            )
        )

    summary = summarize_validation(validated, cross_signal=cross_rows)

    return {
        "window": {
            "start_ts": pd.Timestamp(start_ts).isoformat(),
            "end_ts": pd.Timestamp(end_ts).isoformat(),
            "baseline_days": int(baseline_days),
        },
        "signals": [
            {
                "signal": row.signal,
                "window_stats": row.window_stats,
                "baseline_stats": row.baseline_stats,
                "level_shift_score": row.level_shift_score,
                "level_shift_direction": row.level_shift_direction,
                "volatility_shift_score": row.volatility_shift_score,
                "residual_burst_score": row.residual_burst_score,
                "amplitude_shift_score": row.amplitude_shift_score,
                "shape_shift_score": row.shape_shift_score,
                "signatures": row.signatures,
            }
            for row in validated
        ],
        "cross_signal": cross_rows,
        "summary": summary,
    }