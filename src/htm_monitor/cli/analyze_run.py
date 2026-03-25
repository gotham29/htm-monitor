#src/htm_monitor/cli/analyze_run.py

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger("analyze_run")


# -------------------------
# Small utilities
# -------------------------

def parse_hot(x: Any) -> Dict[str, Any]:
    """
    hot_by_model is serialized as JSON dict in run.csv.
    Anything else -> {}.
    """
    if isinstance(x, dict):
        return x
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return {}
    if isinstance(x, str) and x.strip():
        v = json.loads(x)
        if not isinstance(v, dict):
            raise ValueError("hot_by_model must decode to a JSON object")
        return v
    return {}


def parse_str_list(x: Any) -> List[str]:
    """
    Accept JSON list, python list, comma-separated string, or blank -> [].
    Used for grouped-decision fields serialized into run.csv.
    """
    if isinstance(x, list):
        return [str(v) for v in x if str(v).strip()]
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, str):
        raw = x.strip()
        if not raw:
            return []
        if raw.startswith("["):
            v = json.loads(raw)
            if not isinstance(v, list):
                raise ValueError("Expected JSON list")
            return [str(z) for z in v if str(z).strip()]
    return []


def _parse_ts_strict(ts: str, fmt: str) -> datetime:
    return datetime.strptime(ts, fmt)


def _format_stats(vals: List[float]) -> Dict[str, Optional[float]]:
    if not vals:
        return {"count": 0, "mean": None, "median": None, "p90": None, "min": None, "max": None}
    arr = np.asarray(vals, dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.quantile(arr, 0.90)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _fmt_metric(x: Optional[float]) -> str:
    """
    Safe formatter for possibly-missing scalar metrics.
    """
    return f"{float(x):.3f}" if x is not None else "—"


def _episodes_from_boolean(mask: Sequence[bool], t_index: Sequence[int]) -> List[Tuple[int, int]]:
    """
    Contiguous [start_t, end_t] episodes (inclusive).
    Assumes t_index is increasing; adjacency is defined by t == prev_t + 1.
    """
    eps: List[Tuple[int, int]] = []
    start: Optional[int] = None
    prev_t: Optional[int] = None

    for on, t in zip(mask, t_index):
        if on and start is None:
            start = int(t)
            prev_t = int(t)
            continue

        if on and start is not None:
            if prev_t is not None and int(t) == int(prev_t) + 1:
                prev_t = int(t)
                continue
            # gap -> close previous, start new
            eps.append((int(start), int(prev_t if prev_t is not None else start)))
            start = int(t)
            prev_t = int(t)
            continue

        if (not on) and start is not None:
            eps.append((int(start), int(prev_t if prev_t is not None else start)))
            start = None
            prev_t = None

    if start is not None:
        eps.append((int(start), int(prev_t if prev_t is not None else start)))
    return eps


def _onsets_from_mask(mask: Sequence[bool], t_index: Sequence[int]) -> List[int]:
    """
    Return onset t's where mask transitions False->True.
    """
    out: List[int] = []
    prev = False
    for on, t in zip(mask, t_index):
        onb = bool(on)
        if onb and (not prev):
            out.append(int(t))
        prev = onb
    return out


def _safe_bool_series(x: pd.Series) -> pd.Series:
    if x.dtype == object:
        return x.astype(str).str.lower().isin(["true", "1", "yes", "y"])
    return x.astype(bool)


def _infer_step_minutes(ts: pd.Series) -> Optional[float]:
    """
    Infer typical timestep size (minutes) from timestamps in run.csv.
    Uses median positive delta.
    """
    # Drop NaT before diff so we don't manufacture garbage deltas.
    ts = ts.dropna()
    dt = ts.sort_values().diff().dropna()
    dt = dt[dt > pd.Timedelta(0)]
    if dt.empty:
        return None
    minutes = dt.median().total_seconds() / 60.0
    return float(minutes) if minutes > 0 else None


# -------------------------
# Load run + manifest + config
# -------------------------

def load_run(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns:
        raise ValueError("run.csv missing required column 'timestamp'")
    if "t" not in df.columns:
        raise ValueError("run.csv missing required column 't'")
    if "model" not in df.columns:
        raise ValueError("run.csv missing required column 'model'")
    if "alert" not in df.columns:
        raise ValueError("run.csv missing required column 'alert'")

    df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["alert"] = _safe_bool_series(df["alert"])
    df["hot_dict"] = df.get("hot_by_model", "").apply(parse_hot)

    # Optional grouped-decision diagnostics emitted by run_pipeline.
    if "group_hot_by_model" in df.columns:
        df["group_hot_dict"] = df["group_hot_by_model"].apply(parse_hot)
    else:
        df["group_hot_dict"] = [{} for _ in range(len(df))]

    if "window_hot_by_group" in df.columns:
        df["window_hot_group_dict"] = df["window_hot_by_group"].apply(parse_hot)
    else:
        df["window_hot_group_dict"] = [{} for _ in range(len(df))]

    if "active_groups" in df.columns:
        df["active_groups_list"] = df["active_groups"].apply(parse_str_list)
    else:
        df["active_groups_list"] = [[] for _ in range(len(df))]

    # Optional warmup signal emitted by run_pipeline (preferred).
    if "in_warmup" in df.columns:
        df["in_warmup"] = pd.to_numeric(df["in_warmup"], errors="coerce").fillna(0).astype(int).astype(bool)
    else:
        df["in_warmup"] = False

    # Defensive invariant lock: stable ordering
    df = df.sort_values(["t", "model"], kind="mergesort").reset_index(drop=True)
    return df


def _dedup_steps(df: pd.DataFrame) -> pd.DataFrame:
    # One row per timestep (system-level fields duplicated across models)
    one = df.drop_duplicates("t", keep="first").copy()
    one = one.sort_values("t", kind="mergesort").reset_index(drop=True)
    return one


def _config_model_sources(config_path: str) -> Dict[str, List[str]]:
    cfg = yaml.safe_load(Path(config_path).read_text())
    if not isinstance(cfg, dict):
        raise ValueError("config YAML must be a mapping at top-level")
    models = cfg.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("config.models missing or empty")
    out: Dict[str, List[str]] = {}
    for m, spec in models.items():
        if not isinstance(spec, dict):
            continue
        srcs = spec.get("sources") or []
        if not isinstance(srcs, list) or not all(isinstance(s, str) and s for s in srcs):
            raise ValueError(f"config.models.{m}.sources must be a non-empty list[str]")
        out[str(m)] = list(srcs)
    return out


@dataclass(frozen=True)
class DecisionParams:
    method: str
    score_key: str
    threshold: float
    k: int
    window_size: int
    per_model_hits: int


@dataclass(frozen=True)
class SystemGtParams:
    """
    Parameters for deriving *system GT onsets* from per-source GT onset timestamps.

    We default to:
      - k, window_size from decision
      - per_source_hits = 1 (because GT timestamps are onsets, not dense hit streams)
    """
    method: str
    k: int
    window_size: int
    per_source_hits: int


@dataclass(frozen=True)
class PredictiveWarningContract:
    event_type: str
    target_source: str
    warning_start_steps_before_event: int
    warning_end_steps_before_event: int
    too_early_min_steps_before_event: Optional[int]


@dataclass
class PredictiveWarningPerEventRow:
    event_t: int
    event_ts: Optional[str]
    first_alert_t: Optional[int]
    first_alert_ts: Optional[str]
    first_alert_lead_steps: Optional[int]
    first_alert_lead_minutes: Optional[float]
    first_alert_classification: str
    has_any_in_warning_window: bool
    in_window_episode_starts: List[int]
    hot_models_first_alert: Optional[List[str]]


@dataclass
class PredictiveWarningEval:
    event_count: int
    episode_count: int
    matched_events_in_warning_window: int
    first_alert_summary: Dict[str, int]
    episode_summary: Dict[str, int]
    lead_steps_stats_in_warning_window: Dict[str, Optional[float]]
    lead_minutes_stats_in_warning_window: Dict[str, Optional[float]]
    per_event: List[PredictiveWarningPerEventRow]


@dataclass(frozen=True)
class GtWindow:
    name: Optional[str]
    start_ts: str
    end_ts: str


@dataclass
class GtWindowPerEventRow:
    gt_start_t: int
    gt_end_t: int
    gt_start_ts: Optional[str]
    gt_end_ts: Optional[str]
    detected: bool
    detection_t: Optional[int]
    detection_ts: Optional[str]
    episode: Optional[Tuple[int, int]]
    lag_steps: Optional[int]
    lag_minutes: Optional[float]
    hot_models: Optional[List[str]]


@dataclass
class GtWindowEval:
    gt_count: int
    episode_count: int
    matched_gt: int
    matched_episodes: int
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    lag_steps_stats: Dict[str, Optional[float]]
    lag_minutes_stats: Dict[str, Optional[float]]
    misses: List[Tuple[int, int]]
    false_positive_episodes: List[Tuple[int, int]]
    per_event: List[GtWindowPerEventRow]


@dataclass
class GtOnsetWindowPerEventRow:
    gt_start_t: int
    gt_end_t: int
    gt_start_ts: Optional[str]
    gt_end_ts: Optional[str]
    detected: bool
    detection_t: Optional[int]
    detection_ts: Optional[str]
    episode: Optional[Tuple[int, int]]
    lag_steps: Optional[int]
    lag_minutes: Optional[float]
    hot_models: Optional[List[str]]


@dataclass
class GtOnsetWindowEval:
    gt_count: int
    episode_count: int
    matched_gt: int
    matched_episodes: int
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    lag_steps_stats: Dict[str, Optional[float]]
    lag_minutes_stats: Dict[str, Optional[float]]
    misses: List[Tuple[int, int]]
    false_positive_episodes: List[Tuple[int, int]]
    per_event: List[GtOnsetWindowPerEventRow]


def _load_predictive_warning_contract(config_path: str) -> Optional[PredictiveWarningContract]:
    """
    Load optional predictive-warning semantic contract from config:

      evaluation:
        mode: predictive_warning
        predictive_warning:
          event_type: failure_horizon
          target_source: system
          warning_window:
            start_steps_before_event: 30
            end_steps_before_event: 5
          too_early_window:
            min_steps_before_event: 31

    This validates and normalizes the semantic contract.
    Runtime scoring may use this contract when evaluation.mode == predictive_warning.
    """
    cfg = yaml.safe_load(Path(config_path).read_text())
    if not isinstance(cfg, dict):
        raise ValueError("config YAML must be a mapping at top-level")

    evaluation = cfg.get("evaluation") or {}
    if not isinstance(evaluation, dict):
        raise ValueError("config.evaluation must be a mapping if provided")

    pw = evaluation.get("predictive_warning")
    if pw is None:
        return None
    if not isinstance(pw, dict):
        raise ValueError("config.evaluation.predictive_warning must be a mapping if provided")

    event_type = pw.get("event_type")
    target_source = pw.get("target_source")
    warning_window = pw.get("warning_window")
    too_early_window = pw.get("too_early_window") or {}

    if not isinstance(event_type, str) or not event_type.strip():
        raise ValueError("config.evaluation.predictive_warning.event_type must be a non-empty string")
    if not isinstance(target_source, str) or not target_source.strip():
        raise ValueError("config.evaluation.predictive_warning.target_source must be a non-empty string")
    if not isinstance(warning_window, dict):
        raise ValueError("config.evaluation.predictive_warning.warning_window must be a mapping")
    if not isinstance(too_early_window, dict):
        raise ValueError("config.evaluation.predictive_warning.too_early_window must be a mapping if provided")

    start_steps = warning_window.get("start_steps_before_event")
    end_steps = warning_window.get("end_steps_before_event")
    if not isinstance(start_steps, int) or start_steps <= 0:
        raise ValueError(
            "config.evaluation.predictive_warning.warning_window.start_steps_before_event "
            "must be an int > 0"
        )
    if not isinstance(end_steps, int) or end_steps < 0:
        raise ValueError(
            "config.evaluation.predictive_warning.warning_window.end_steps_before_event "
            "must be an int >= 0"
        )
    if start_steps <= end_steps:
        raise ValueError(
            "config.evaluation.predictive_warning.warning_window must satisfy "
            "start_steps_before_event > end_steps_before_event"
        )

    too_early_min = too_early_window.get("min_steps_before_event")
    if too_early_min is not None:
        if not isinstance(too_early_min, int) or too_early_min <= 0:
            raise ValueError(
                "config.evaluation.predictive_warning.too_early_window.min_steps_before_event "
                "must be an int > 0 if provided"
            )

    return PredictiveWarningContract(
        event_type=str(event_type).strip(),
        target_source=str(target_source).strip(),
        warning_start_steps_before_event=int(start_steps),
        warning_end_steps_before_event=int(end_steps),
        too_early_min_steps_before_event=int(too_early_min) if too_early_min is not None else None,
    )


def _load_evaluation_mode(config_path: str) -> str:
    """
    Load semantic evaluation mode from config.

    Currently supported:
      - onset_detection
      - predictive_warning

    Runtime scoring is only implemented for onset_detection at this phase.
    """
    cfg = yaml.safe_load(Path(config_path).read_text())
    if not isinstance(cfg, dict):
        raise ValueError("config YAML must be a mapping at top-level")

    evaluation = cfg.get("evaluation") or {}
    if not isinstance(evaluation, dict):
        raise ValueError("config.evaluation must be a mapping if provided")

    mode = str(evaluation.get("mode") or "onset_detection").strip() or "onset_detection"
    if mode not in {"onset_detection", "predictive_warning"}:
        raise ValueError(f"config.evaluation.mode must be 'onset_detection' or 'predictive_warning'; got '{mode}'")
    return mode


def _load_decision_params(config_path: str) -> DecisionParams:
    cfg = yaml.safe_load(Path(config_path).read_text())
    if not isinstance(cfg, dict):
        raise ValueError("config YAML must be a mapping at top-level")

    decision = cfg.get("decision") or {}
    if not isinstance(decision, dict):
        raise ValueError("config.decision must be a mapping")

    method = str(decision.get("method") or "").strip() or "unknown"
    score_key = str(decision.get("score_key") or "p").strip() or "p"

    thr = decision.get("threshold")
    if thr is None:
        raise ValueError("config.decision.threshold missing")
    threshold = float(thr)

    k = decision.get("k")
    if k is None or not isinstance(k, int) or k <= 0:
        raise ValueError("config.decision.k must be an int > 0")

    win = decision.get("window")
    if not isinstance(win, dict):
        raise ValueError("config.decision.window must be a mapping")
    size = win.get("size")
    if size is None or not isinstance(size, int) or size <= 0:
        raise ValueError("config.decision.window.size must be an int > 0")
    per_model_hits = win.get("per_model_hits", 1)
    if not isinstance(per_model_hits, int) or per_model_hits <= 0:
        raise ValueError("config.decision.window.per_model_hits must be an int > 0")

    return DecisionParams(
        method=method,
        score_key=score_key,
        threshold=threshold,
        k=int(k),
        window_size=int(size),
        per_model_hits=int(per_model_hits),
    )


def _load_system_gt_params(config_path: str, decision: DecisionParams) -> SystemGtParams:
    """
    Optional config section:

      ground_truth:
        system:
          method: kofn_window
          k: 2
          window:
            size: 6
            per_source_hits: 1

    If absent, we default to decision.k, decision.window_size, and per_source_hits=1.
    """
    cfg = yaml.safe_load(Path(config_path).read_text())
    if not isinstance(cfg, dict):
        raise ValueError("config YAML must be a mapping at top-level")

    gt = cfg.get("ground_truth") or {}
    if not isinstance(gt, dict):
        gt = {}
    sys_gt = gt.get("system") or {}
    if not isinstance(sys_gt, dict):
        sys_gt = {}

    method = str(sys_gt.get("method") or "kofn_window").strip() or "kofn_window"
    k = sys_gt.get("k", decision.k)
    k = int(k)
    if k <= 0:
        raise ValueError("ground_truth.system.k must be > 0")

    win = sys_gt.get("window") or {}
    if win is None:
        win = {}
    if not isinstance(win, dict):
        raise ValueError("ground_truth.system.window must be a mapping if provided")
    window_size = int(win.get("size", decision.window_size))
    if window_size <= 0:
        raise ValueError("ground_truth.system.window.size must be an int > 0")

    per_source_hits = win.get("per_source_hits", 1)
    per_source_hits = int(per_source_hits)
    if per_source_hits <= 0:
        raise ValueError("ground_truth.system.window.per_source_hits must be an int > 0")

    return SystemGtParams(method=method, k=k, window_size=window_size, per_source_hits=per_source_hits)


def _config_gt_windows_by_source(config_path: str) -> Dict[str, List[GtWindow]]:
    """
    Canonical GT contract:
      data.sources[*].labels.event_windows:
        - name: aug2020
          start: "2020-08-14 00:00:00"
          end:   "2020-08-15 23:00:00"

    Back-compat:
      data.sources[*].labels.timestamps: [ ... ]
    Legacy timestamps are converted later into contiguous windows in timestep space.
    """
    cfg = yaml.safe_load(Path(config_path).read_text())
    if not isinstance(cfg, dict):
        raise ValueError("config YAML must be a mapping at top-level")

    sources = (cfg.get("data") or {}).get("sources") or []
    if not isinstance(sources, list):
        return {}

    out: Dict[str, List[GtWindow]] = {}
    for s in sources:
        if not isinstance(s, dict):
            continue
        name = s.get("name")
        if not isinstance(name, str) or not name:
            continue
        labels = s.get("labels") or {}
        if not isinstance(labels, dict):
            continue
        event_windows = labels.get("event_windows")
        if event_windows is not None:
            if not isinstance(event_windows, list):
                raise ValueError(f"config.data.sources[{name}].labels.event_windows must be a list")
            wins: List[GtWindow] = []
            for i, ev in enumerate(event_windows):
                if not isinstance(ev, dict):
                    raise ValueError(
                       f"config.data.sources[{name}].labels.event_windows[{i}] must be a mapping"
                    )
                start = ev.get("start")
                end = ev.get("end")
                ev_name = ev.get("name")
                if not isinstance(start, str) or not start.strip():
                    raise ValueError(
                        f"config.data.sources[{name}].labels.event_windows[{i}].start must be a non-empty string"
                    )
                if not isinstance(end, str) or not end.strip():
                    raise ValueError(
                        f"config.data.sources[{name}].labels.event_windows[{i}].end must be a non-empty string"
                    )
                wins.append(
                    GtWindow(
                        name=str(ev_name).strip() if isinstance(ev_name, str) and ev_name.strip() else None,
                        start_ts=start.strip(),
                        end_ts=end.strip(),
                    )
                )
            if wins:
                out[str(name)] = wins
            continue

        ts_list = labels.get("timestamps")
        if ts_list is not None:
            if not isinstance(ts_list, list):
                raise ValueError(f"config.data.sources[{name}].labels.timestamps must be a list of strings")
            vals = [x.strip() for x in ts_list if isinstance(x, str) and x.strip()]
            if vals:
                # Legacy path: store as 1-timestep windows for now; later collapsed after ts->t mapping.
                out[str(name)] = [GtWindow(name=None, start_ts=v, end_ts=v) for v in vals]
    return out


def _collapse_t_points_to_windows(t_points: List[int]) -> List[Tuple[int, int]]:
    """
    Collapse sorted timestep points into contiguous inclusive windows.
    """
    pts = sorted(set(int(x) for x in t_points))
    if not pts:
        return []
    out: List[Tuple[int, int]] = []
    start = pts[0]
    prev = pts[0]
    for t in pts[1:]:
        if int(t) == int(prev) + 1:
            prev = int(t)
            continue
        out.append((int(start), int(prev)))
        start = int(t)
        prev = int(t)
    out.append((int(start), int(prev)))
    return out


def _merge_overlapping_windows(windows: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not windows:
        return []
    ws = sorted((int(s), int(e)) for s, e in windows)
    out: List[Tuple[int, int]] = [ws[0]]
    for s, e in ws[1:]:
        ls, le = out[-1]
        if s <= le + 1:
            out[-1] = (ls, max(le, e))
        else:
            out.append((s, e))
    return out


def _models_consuming_source(model_sources: Dict[str, List[str]], source: str) -> List[str]:
    return [m for m, srcs in model_sources.items() if source in srcs]


def _build_ts_to_t_for_models(df: pd.DataFrame, models: List[str]) -> Dict[str, int]:
    """
    Deterministic mapping timestamp_string -> t for a subset of models.
    Contract: for a fixed timestamp, all those models share the same t.
    """
    if not models:
        return {}
    sub = df[df["model"].isin(models)][["timestamp", "t"]].copy()
    if sub.empty:
        return {}

    g = sub.groupby("timestamp")["t"].nunique(dropna=False)
    bad = g[g > 1]
    if len(bad) > 0:
        sample_ts = bad.index.tolist()[:5]
        raise ValueError(
            "Non-deterministic mapping: same timestamp maps to multiple t values. "
            f"Sample timestamps={sample_ts}"
        )

    ts_to_t = sub.drop_duplicates("timestamp", keep="first").set_index("timestamp")["t"].astype(int).to_dict()
    return {str(k): int(v) for k, v in ts_to_t.items() if isinstance(k, str) and k}


def _map_gt_strings_to_t(
    gt_ts: List[str],
    ts_to_t: Dict[str, int],
    *,
    strict: bool,
    ctx: str,
) -> Tuple[List[int], List[str]]:
    out: List[int] = []
    missing: List[str] = []
    for ts in gt_ts:
        if ts in ts_to_t:
            out.append(int(ts_to_t[ts]))
        else:
            missing.append(ts)
    if missing and strict:
        raise ValueError(
            f"GT onset timestamp(s) missing from run timeline for {ctx}: "
            f"{missing[:10]}{' ...' if len(missing)>10 else ''}"
        )
    return sorted(set(out)), missing


# -------------------------
# System GT derivation (kofn_window)
# -------------------------

def _rolling_hits_to_hot_mask(n: int, hits_idx: List[int], *, window_size: int, per_model_hits: int) -> np.ndarray:
    """
    hits_idx are indices in [0, n-1] where a GT hit occurs.
    hot[i] = 1 if sum(hits in [i-window_size+1, i]) >= per_model_hits
    """
    if n <= 0:
        return np.zeros(0, dtype=np.int32)
    if window_size <= 0 or per_model_hits <= 0:
        return np.zeros(n, dtype=np.int32)

    arr = np.zeros(n, dtype=np.int32)
    for i in hits_idx:
        if 0 <= int(i) < n:
            arr[int(i)] = 1

    if arr.sum() == 0:
        return np.zeros(n, dtype=np.int32)

    c = np.cumsum(arr, dtype=np.int32)
    win = int(window_size)
    sums = c.copy()
    if win < n:
        sums[win:] = c[win:] - c[:-win]
    hot = (sums >= int(per_model_hits)).astype(np.int32)
    return hot


def derive_system_gt_onsets(
    one: pd.DataFrame,
    gt_onsets_t_by_source: Dict[str, List[int]],
    *,
    k: int,
    window_size: int,
    per_model_hits: int,
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Returns (system_gt_onset_timesteps, definition_dict).

    - GT per source is onset *points* in timestep space.
    - Convert each source's hit points into a per-timestep "hot" mask using kofn_window.
    - System-hot mask if >=k sources hot at that timestep.
    - Return ONSETS of system-hot mask.
    """
    t_list = one["t"].astype(int).tolist()
    if not t_list:
        return [], {"k": k, "window_size": window_size, "per_model_hits": per_model_hits}

    t_min = int(t_list[0])
    t_max = int(t_list[-1])
    n = int(t_max - t_min + 1)

    hot_by_src: Dict[str, np.ndarray] = {}
    for src, onsets in (gt_onsets_t_by_source or {}).items():
        hits_idx = [int(t) - t_min for t in (onsets or []) if t is not None]
        hot = _rolling_hits_to_hot_mask(n, hits_idx, window_size=window_size, per_model_hits=per_model_hits)
        if hot.sum() > 0:
            hot_by_src[str(src)] = hot

    if not hot_by_src:
        return [], {"k": k, "window_size": window_size, "per_model_hits": per_model_hits}

    hot_count = np.zeros(n, dtype=np.int32)
    for hot in hot_by_src.values():
        hot_count += hot

    sys_hot = (hot_count >= int(k)).astype(bool)
    sys_t = [t_min + i for i in range(n)]

    sys_onsets = _onsets_from_mask(sys_hot.tolist(), sys_t)
    return sys_onsets, {"k": int(k), "window_size": int(window_size), "per_model_hits": int(per_model_hits)}


def derive_system_gt_windows(
    one: pd.DataFrame,
    gt_windows_t_by_source: Dict[str, List[Tuple[int, int]]],
    *,
    k: int,
) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    """
    Derive system GT ACTIVE windows by requiring >=k sources active at each timestep.
    """
    t_list = one["t"].astype(int).tolist()
    if not t_list:
        return [], {"k": int(k)}

    t_min = int(t_list[0])
    t_max = int(t_list[-1])
    n = int(t_max - t_min + 1)

    active_count = np.zeros(n, dtype=np.int32)
    for _, windows in gt_windows_t_by_source.items():
        mask = np.zeros(n, dtype=np.int32)
        for s, e in windows:
            s_idx = max(0, int(s) - t_min)
            e_idx = min(n - 1, int(e) - t_min)
            if s_idx <= e_idx:
                mask[s_idx:e_idx + 1] = 1
        active_count += mask

    sys_active = (active_count >= int(k)).astype(bool)
    sys_t = [t_min + i for i in range(n)]
    sys_windows = _episodes_from_boolean(sys_active.tolist(), sys_t)
    return sys_windows, {"k": int(k)}


def eval_windows_vs_episodes(
    one: pd.DataFrame,
    gt_windows: List[Tuple[int, int]],
    episodes: List[Tuple[int, int]],
    *,
    max_lag_steps: int,
    step_minutes: Optional[float],
    hot_dict_by_t: Optional[Dict[int, Dict[str, Any]]] = None,
) -> GtWindowEval:
    """
    Window/episode scoring:
      - A GT window [gs, ge] is detected if an alert episode overlaps [gs, ge + max_lag_steps]
      - Greedy one-to-one matching in time order
      - Lag is first alert start minus gt window start, floored at 0 if alert begins inside window
      - FP episode = does not overlap any expanded GT window
    """
    gt_sorted = sorted((int(s), int(e)) for s, e in gt_windows)
    ep_sorted = sorted((int(s), int(e)) for s, e in episodes)

    used_eps: Set[int] = set()
    used_gt: Set[int] = set()
    lags_steps: List[float] = []
    lags_minutes: List[float] = []

    t_to_ts: Dict[int, Any] = dict(zip(one["t"].astype(int).tolist(), one["ts"].tolist()))

    def _ts_str(x: Any) -> Optional[str]:
        if x is None:
            return None
        if isinstance(x, pd.Timestamp):
            if pd.isna(x):
                return None
            return x.isoformat()
        if isinstance(x, datetime):
            return x.isoformat()
        s = str(x).strip()
        return s if s else None

    def _hot_models_at_t(t: int) -> List[str]:
        if not hot_dict_by_t:
            return []
        d = hot_dict_by_t.get(int(t)) or {}
        if not isinstance(d, dict):
            return []
        return sorted(str(k) for k, v in d.items() if bool(v))

    per_event: List[GtWindowPerEventRow] = []

    for gi, (gs, ge) in enumerate(gt_sorted):
        match_idx: Optional[int] = None
        match_ep: Optional[Tuple[int, int]] = None
        match_start: Optional[int] = None
        expanded_end = int(ge) + int(max_lag_steps)

        for ei, (es, ee) in enumerate(ep_sorted):
            if ei in used_eps:
                continue
            overlaps = not (int(ee) < int(gs) or int(es) > int(expanded_end))
            if not overlaps:
                continue
            match_idx = ei
            match_ep = (int(es), int(ee))
            match_start = int(es)
            break

        if match_idx is not None and match_ep is not None and match_start is not None:
            used_gt.add(gi)
            used_eps.add(match_idx)
            lag = max(0, int(match_start) - int(gs))
            lags_steps.append(float(lag))
            lag_min = (float(lag) * float(step_minutes)) if step_minutes is not None else None
            if lag_min is not None:
                lags_minutes.append(float(lag_min))
            per_event.append(
                GtWindowPerEventRow(
                    gt_start_t=int(gs),
                    gt_end_t=int(ge),
                    gt_start_ts=_ts_str(t_to_ts.get(int(gs))),
                    gt_end_ts=_ts_str(t_to_ts.get(int(ge))),
                    detected=True,
                    detection_t=int(match_start),
                    detection_ts=_ts_str(t_to_ts.get(int(match_start))),
                    episode=match_ep,
                    lag_steps=int(lag),
                    lag_minutes=lag_min,
                    hot_models=_hot_models_at_t(int(match_start)),
                )
            )
        else:
            per_event.append(
                GtWindowPerEventRow(
                    gt_start_t=int(gs),
                    gt_end_t=int(ge),
                    gt_start_ts=_ts_str(t_to_ts.get(int(gs))),
                    gt_end_ts=_ts_str(t_to_ts.get(int(ge))),
                    detected=False,
                    detection_t=None,
                    detection_ts=None,
                    episode=None,
                    lag_steps=None,
                    lag_minutes=None,
                    hot_models=None,
                )
            )

    fp_eps: List[Tuple[int, int]] = []
    expanded_gt = [(int(gs), int(ge) + int(max_lag_steps)) for gs, ge in gt_sorted]
    for es, ee in ep_sorted:
        overlaps_any = any(not (int(ee) < gs or int(es) > ge) for gs, ge in expanded_gt)
        if not overlaps_any:
            fp_eps.append((int(es), int(ee)))

    matched_gt = len(used_gt)
    matched_eps = len(used_eps)
    gt_count = len(gt_sorted)
    fp_count = len(fp_eps)
    prec_denom = matched_eps + fp_count
    precision = (matched_eps / prec_denom) if prec_denom > 0 else None
    recall = (matched_gt / gt_count) if gt_count > 0 else None
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)

    misses = [gt_sorted[i] for i in range(gt_count) if i not in used_gt]

    return GtWindowEval(
        gt_count=gt_count,
        episode_count=len(ep_sorted),
        matched_gt=matched_gt,
        matched_episodes=matched_eps,
        precision=precision,
        recall=recall,
        f1=f1,
        lag_steps_stats=_format_stats(lags_steps),
        lag_minutes_stats=_format_stats(lags_minutes),
        misses=misses,
        false_positive_episodes=fp_eps,
        per_event=per_event,
    )

def eval_event_starts_vs_episodes(
    one: pd.DataFrame,
    gt_windows: List[Tuple[int, int]],
    episodes: List[Tuple[int, int]],
    *,
    max_lag_steps: int,
    step_minutes: Optional[float],
    hot_dict_by_t: Optional[Dict[int, Dict[str, Any]]] = None,
) -> GtOnsetWindowEval:
    """
    Onset-detection semantics using GT windows as the canonical label contract.

    A GT event window [gs, ge] is considered detected if there exists an alert
    episode whose START lies in [gs, gs + max_lag_steps].

    Important:
      - detection is keyed to event onset, not full-window overlap
      - GT windows are still preserved for reporting/visualization
      - FP episode = episode start lies outside every detection window
    """
    gt_sorted = sorted((int(s), int(e)) for s, e in gt_windows)
    ep_sorted = sorted((int(s), int(e)) for s, e in episodes)

    used_eps: Set[int] = set()
    used_gt: Set[int] = set()
    lags_steps: List[float] = []
    lags_minutes: List[float] = []

    t_to_ts: Dict[int, Any] = dict(zip(one["t"].astype(int).tolist(), one["ts"].tolist()))

    def _ts_str(x: Any) -> Optional[str]:
        if x is None:
            return None
        if isinstance(x, pd.Timestamp):
            if pd.isna(x):
                return None
            return x.isoformat()
        if isinstance(x, datetime):
            return x.isoformat()
        s = str(x).strip()
        return s if s else None

    def _hot_models_at_t(t: int) -> List[str]:
        if not hot_dict_by_t:
            return []
        d = hot_dict_by_t.get(int(t)) or {}
        if not isinstance(d, dict):
            return []
        return sorted(str(k) for k, v in d.items() if bool(v))

    per_event: List[GtOnsetWindowPerEventRow] = []

    for gi, (gs, ge) in enumerate(gt_sorted):
        match_idx: Optional[int] = None
        match_ep: Optional[Tuple[int, int]] = None
        match_start: Optional[int] = None
        detect_end = int(gs) + int(max_lag_steps)

        for ei, (es, ee) in enumerate(ep_sorted):
            if ei in used_eps:
                continue
            es = int(es)
            if es < int(gs):
                continue
            if es > int(detect_end):
                break
            match_idx = ei
            match_ep = (int(es), int(ee))
            match_start = int(es)
            break

        if match_idx is not None and match_ep is not None and match_start is not None:
            used_gt.add(gi)
            used_eps.add(match_idx)
            lag = int(match_start) - int(gs)
            lags_steps.append(float(lag))
            lag_min = (float(lag) * float(step_minutes)) if step_minutes is not None else None
            if lag_min is not None:
                lags_minutes.append(float(lag_min))
            per_event.append(
                GtOnsetWindowPerEventRow(
                    gt_start_t=int(gs),
                    gt_end_t=int(ge),
                    gt_start_ts=_ts_str(t_to_ts.get(int(gs))),
                    gt_end_ts=_ts_str(t_to_ts.get(int(ge))),
                    detected=True,
                    detection_t=int(match_start),
                    detection_ts=_ts_str(t_to_ts.get(int(match_start))),
                    episode=match_ep,
                    lag_steps=int(lag),
                    lag_minutes=lag_min,
                    hot_models=_hot_models_at_t(int(match_start)),
                )
            )
        else:
            per_event.append(
                GtOnsetWindowPerEventRow(
                    gt_start_t=int(gs),
                    gt_end_t=int(ge),
                    gt_start_ts=_ts_str(t_to_ts.get(int(gs))),
                    gt_end_ts=_ts_str(t_to_ts.get(int(ge))),
                    detected=False,
                    detection_t=None,
                    detection_ts=None,
                    episode=None,
                    lag_steps=None,
                    lag_minutes=None,
                    hot_models=None,
                )
            )

    fp_eps: List[Tuple[int, int]] = []
    detect_windows = [(int(gs), int(gs) + int(max_lag_steps)) for gs, _ in gt_sorted]
    for es, ee in ep_sorted:
        start_in_any = any(g0 <= int(es) <= g1 for g0, g1 in detect_windows)
        if not start_in_any:
            fp_eps.append((int(es), int(ee)))

    matched_gt = len(used_gt)
    matched_eps = len(used_eps)
    gt_count = len(gt_sorted)
    fp_count = len(fp_eps)
    prec_denom = matched_eps + fp_count
    precision = (matched_eps / prec_denom) if prec_denom > 0 else None
    recall = (matched_gt / gt_count) if gt_count > 0 else None
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)

    misses = [gt_sorted[i] for i in range(gt_count) if i not in used_gt]

    return GtOnsetWindowEval(
        gt_count=gt_count,
        episode_count=len(ep_sorted),
        matched_gt=matched_gt,
        matched_episodes=matched_eps,
        precision=precision,
        recall=recall,
        f1=f1,
        lag_steps_stats=_format_stats(lags_steps),
        lag_minutes_stats=_format_stats(lags_minutes),
        misses=misses,
        false_positive_episodes=fp_eps,
        per_event=per_event,
    )

# -------------------------
# Episode-level scoring
# -------------------------

@dataclass
class GtPerEventRow:
    gt_t: int
    gt_ts: Optional[str]
    detected: bool
    detection_t: Optional[int]
    detection_ts: Optional[str]
    episode: Optional[Tuple[int, int]]
    lag_steps: Optional[int]
    lag_minutes: Optional[float]
    hot_models: Optional[List[str]]


@dataclass
class GtEval:
    gt_count: int
    episode_count: int
    matched_gt: int
    matched_episodes: int
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    lag_steps_stats: Dict[str, Optional[float]]
    lag_minutes_stats: Dict[str, Optional[float]]
    misses: List[int]
    false_positive_episodes: List[Tuple[int, int]]
    per_event: List[GtPerEventRow]


def eval_onsets_vs_episodes(
    one: pd.DataFrame,
    gt_onsets_t: List[int],
    episodes: List[Tuple[int, int]],
    *,
    max_lag_steps: int,
    step_minutes: Optional[float],
    hot_dict_by_t: Optional[Dict[int, Dict[str, Any]]] = None,
) -> GtEval:
    """
    Episode-level scoring with your stated semantics:
      - a GT onset at g is detected if there exists an episode with start s in [g, g+max_lag_steps]
      - each episode can match at most one GT and vice versa (greedy in time order)
      - lag is s - g (>=0)
    """
    gt_sorted = sorted(set(int(x) for x in gt_onsets_t))
    ep_sorted = sorted(episodes, key=lambda x: (int(x[0]), int(x[1])))

    used_eps: Set[int] = set()
    used_gt: Set[int] = set()

    lags_steps: List[float] = []
    lags_minutes: List[float] = []

    t_to_ts: Dict[int, Any] = dict(zip(one["t"].astype(int).tolist(), one["ts"].tolist()))

    def _ts_str(x: Any) -> Optional[str]:
        """
        Convert pandas Timestamp / datetime / string into a stable string.
        """
        if x is None:
            return None
        # pandas Timestamp / NaT
        if isinstance(x, pd.Timestamp):
            if pd.isna(x):
                return None
            # Keep it readable + stable (includes timezone if present).
            return x.isoformat()
        # python datetime
        if isinstance(x, datetime):
            return x.isoformat()
        # already a string
        if isinstance(x, str):
            s = x.strip()
            return s if s else None
        # fallback: stringify
        s = str(x).strip()
        return s if s else None

    def _hot_models_at_t(t: int) -> List[str]:
        if not hot_dict_by_t:
            return []
        d = hot_dict_by_t.get(int(t)) or {}
        if not isinstance(d, dict):
            return []
        out = [str(k) for k, v in d.items() if bool(v)]
        return sorted(out)

    per_event: List[GtPerEventRow] = []

    for gi, g in enumerate(gt_sorted):
        match_idx: Optional[int] = None
        match_episode: Optional[Tuple[int, int]] = None
        match_start: Optional[int] = None

        for ei, (s, e) in enumerate(ep_sorted):
            if ei in used_eps:
                continue
            s = int(s)
            if s < int(g):
                continue
            if s > int(g) + int(max_lag_steps):
                break
            match_idx = ei
            match_episode = (int(s), int(e))
            match_start = int(s)
            break

        ts_g = t_to_ts.get(int(g))
        if match_idx is not None and match_start is not None:
            used_gt.add(gi)
            used_eps.add(match_idx)
            lag = int(match_start) - int(g)
            lags_steps.append(float(lag))

            lag_min: Optional[float] = None
            if step_minutes is not None:
                lag_min = float(lag) * float(step_minutes)
                lags_minutes.append(float(lag_min))

            ts_s = t_to_ts.get(int(match_start))
            per_event.append(
                GtPerEventRow(
                    gt_t=int(g),
                    gt_ts=_ts_str(ts_g),
                    detected=True,
                    detection_t=int(match_start),
                    detection_ts=_ts_str(ts_s),
                    episode=match_episode,
                    lag_steps=int(lag),
                    lag_minutes=lag_min,
                    hot_models=_hot_models_at_t(int(match_start)),
                )
            )
        else:
            per_event.append(
                GtPerEventRow(
                    gt_t=int(g),
                    gt_ts=_ts_str(ts_g),
                    detected=False,
                    detection_t=None,
                    detection_ts=None,
                    episode=None,
                    lag_steps=None,
                    lag_minutes=None,
                    hot_models=None,
                )
            )

    matched_gt = len(used_gt)
    matched_eps = len(used_eps)
    ep_count = len(ep_sorted)

    # ---------------------------------------------------------
    # Benchmark-grade FP logic
    #
    # An episode is only a false positive if its START lies
    # outside every GT detection window [g, g + max_lag_steps].
    #
    # This prevents secondary alerts shortly after a real
    # anomaly from being incorrectly counted as FP.
    # ---------------------------------------------------------

    gt_windows = [(int(g), int(g) + int(max_lag_steps)) for g in gt_sorted]

    fp_eps: List[Tuple[int, int]] = []

    for ep in ep_sorted:
        s = int(ep[0])

        inside_window = False

        for g0, g1 in gt_windows:
            if g0 <= s <= g1:
                inside_window = True
                break

        if not inside_window:
            fp_eps.append(ep)

    # ---------------------------------------------------------
    # Precision / Recall using benchmark FP semantics
    #
    # Precision denominator must be:
    #   matched_episodes + false_positive_episodes
    #
    # NOT total episode count, because episodes inside a GT
    # detection window are not considered false positives.
    # ---------------------------------------------------------

    gt_count = len(gt_sorted)

    fp_count = len(fp_eps)
    prec_denom = matched_eps + fp_count

    precision = (matched_eps / prec_denom) if prec_denom > 0 else None
    recall = (matched_gt / gt_count) if gt_count else None

    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)

    misses = [gt_sorted[i] for i in range(gt_count) if i not in used_gt]

    return GtEval(
        gt_count=gt_count,
        episode_count=ep_count,
        matched_gt=matched_gt,
        matched_episodes=matched_eps,
        precision=precision,
        recall=recall,
        f1=f1,
        lag_steps_stats=_format_stats(lags_steps),
        lag_minutes_stats=_format_stats(lags_minutes),
        misses=misses,
        false_positive_episodes=fp_eps,
        per_event=per_event,
    )


def _classify_predictive_warning_lead(
    lead_steps: int,
    *,
    warning_start_steps_before_event: int,
    warning_end_steps_before_event: int,
) -> str:
    """
    lead_steps = event_t - alert_t

    Larger lead => earlier alert.
    Smaller lead => later alert.
    """
    if lead_steps > int(warning_start_steps_before_event):
        return "too_early"
    if lead_steps < int(warning_end_steps_before_event):
        return "too_late"
    return "in_warning_window"


def eval_predictive_warning_vs_episodes(
    one: pd.DataFrame,
    event_ts_t: List[int],
    episodes: List[Tuple[int, int]],
    *,
    contract: PredictiveWarningContract,
    step_minutes: Optional[float],
    hot_dict_by_t: Optional[Dict[int, Dict[str, Any]]] = None,
) -> PredictiveWarningEval:
    """
    Predictive-warning scoring:

    Event-centric view:
      - For each event, inspect alerts after the previous event and up to the current event.
      - first_alert_classification is based on the earliest such alert.
      - has_any_in_warning_window is True if any alert for that event lands in-band.

    Episode-centric view:
      - Every episode is scored against its nearest future event, if one exists.
      - Episodes after the final event are counted as unscored.
    """
    events = sorted(set(int(x) for x in event_ts_t))
    eps = sorted((int(s), int(e)) for (s, e) in episodes)
    episode_starts = [int(s) for s, _ in eps]

    t_to_ts: Dict[int, Any] = dict(zip(one["t"].astype(int).tolist(), one["ts"].tolist()))

    def _ts_str(x: Any) -> Optional[str]:
        if x is None:
            return None
        if isinstance(x, pd.Timestamp):
            if pd.isna(x):
                return None
            return x.isoformat()
        if isinstance(x, datetime):
            return x.isoformat()
        s = str(x).strip()
        return s if s else None

    def _hot_models_at_t(t: int) -> List[str]:
        if not hot_dict_by_t:
            return []
        d = hot_dict_by_t.get(int(t)) or {}
        if not isinstance(d, dict):
            return []
        return sorted(str(k) for k, v in d.items() if bool(v))

    first_alert_summary = {
        "in_warning_window": 0,
        "too_early": 0,
        "too_late": 0,
        "no_alert": 0,
    }
    episode_summary = {
        "in_warning_window": 0,
        "too_early": 0,
        "too_late": 0,
        "unscored": 0,
    }

    in_window_leads_steps: List[float] = []
    in_window_leads_minutes: List[float] = []
    per_event: List[PredictiveWarningPerEventRow] = []

    # Event-centric
    prev_event_t: Optional[int] = None
    for event_t in events:
        candidate_starts = [
            s for s in episode_starts
            if (prev_event_t is None or s > int(prev_event_t)) and s <= int(event_t)
        ]
        in_window_starts: List[int] = []
        for s in candidate_starts:
            lead = int(event_t) - int(s)
            cls = _classify_predictive_warning_lead(
                lead,
                warning_start_steps_before_event=contract.warning_start_steps_before_event,
                warning_end_steps_before_event=contract.warning_end_steps_before_event,
            )
            if cls == "in_warning_window":
                in_window_starts.append(int(s))

        if candidate_starts:
            first_s = int(candidate_starts[0])
            first_lead = int(event_t) - int(first_s)
            first_cls = _classify_predictive_warning_lead(
                first_lead,
                warning_start_steps_before_event=contract.warning_start_steps_before_event,
                warning_end_steps_before_event=contract.warning_end_steps_before_event,
            )
            first_alert_summary[first_cls] += 1
            first_alert_lead_minutes = (float(first_lead) * float(step_minutes)) if step_minutes is not None else None
            per_event.append(
                PredictiveWarningPerEventRow(
                    event_t=int(event_t),
                    event_ts=_ts_str(t_to_ts.get(int(event_t))),
                    first_alert_t=int(first_s),
                    first_alert_ts=_ts_str(t_to_ts.get(int(first_s))),
                    first_alert_lead_steps=int(first_lead),
                    first_alert_lead_minutes=first_alert_lead_minutes,
                    first_alert_classification=first_cls,
                    has_any_in_warning_window=bool(in_window_starts),
                    in_window_episode_starts=list(in_window_starts),
                    hot_models_first_alert=_hot_models_at_t(int(first_s)),
                )
            )
        else:
            first_alert_summary["no_alert"] += 1
            per_event.append(
                PredictiveWarningPerEventRow(
                    event_t=int(event_t),
                    event_ts=_ts_str(t_to_ts.get(int(event_t))),
                    first_alert_t=None,
                    first_alert_ts=None,
                    first_alert_lead_steps=None,
                    first_alert_lead_minutes=None,
                    first_alert_classification="no_alert",
                    has_any_in_warning_window=False,
                    in_window_episode_starts=[],
                    hot_models_first_alert=None,
                )
            )

        prev_event_t = int(event_t)

    # Episode-centric
    for s in episode_starts:
        future_events = [e for e in events if int(e) >= int(s)]
        if not future_events:
            episode_summary["unscored"] += 1
            continue
        e = int(future_events[0])
        lead = int(e) - int(s)
        cls = _classify_predictive_warning_lead(
            lead,
            warning_start_steps_before_event=contract.warning_start_steps_before_event,
            warning_end_steps_before_event=contract.warning_end_steps_before_event,
        )
        episode_summary[cls] += 1
        if cls == "in_warning_window":
            in_window_leads_steps.append(float(lead))
            if step_minutes is not None:
                in_window_leads_minutes.append(float(lead) * float(step_minutes))

    matched_events = sum(1 for r in per_event if r.has_any_in_warning_window)
    return PredictiveWarningEval(
        event_count=int(len(events)),
        episode_count=int(len(eps)),
        matched_events_in_warning_window=int(matched_events),
        first_alert_summary=first_alert_summary,
        episode_summary=episode_summary,
        lead_steps_stats_in_warning_window=_format_stats(in_window_leads_steps),
        lead_minutes_stats_in_warning_window=_format_stats(in_window_leads_minutes),
        per_event=per_event,
    )


# -------------------------
# Episode details + takeaways
# -------------------------

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


def _safe_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def _episode_matched_gt_map(
    sys_eval: Optional[GtOnsetWindowEval],
) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    """
    Map detected system episodes -> the GT windows they satisfied.
    """
    out: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    if sys_eval is None:
        return out

    for row in sys_eval.per_event:
        if not row.detected or row.episode is None:
            continue
        key = (int(row.episode[0]), int(row.episode[1]))
        out.setdefault(key, []).append(
            {
                "gt_start_t": int(row.gt_start_t),
                "gt_end_t": int(row.gt_end_t),
                "gt_start_ts": row.gt_start_ts,
                "gt_end_ts": row.gt_end_ts,
                "detection_t": int(row.detection_t) if row.detection_t is not None else None,
                "detection_ts": row.detection_ts,
                "lag_steps": int(row.lag_steps) if row.lag_steps is not None else None,
                "lag_minutes": float(row.lag_minutes) if row.lag_minutes is not None else None,
                "hot_models": list(row.hot_models) if row.hot_models is not None else [],
            }
        )
    return out


def build_episode_details(
    one: pd.DataFrame,
    df: pd.DataFrame,
    episodes: List[Tuple[int, int]],
    *,
    score_col: str,
    step_minutes: Optional[float],
    hot_dict_by_t: Dict[int, Dict[str, Any]],
    group_hot_dict_by_t: Optional[Dict[int, Dict[str, Any]]],
    active_groups_by_t: Optional[Dict[int, List[str]]],
    gt_windows: List[Tuple[int, int]],
    sys_eval: Optional[GtOnsetWindowEval],
    min_eval_t: int,
) -> List[Dict[str, Any]]:
    """
    Build one JSON-friendly record per *postprocessed system alert episode*.

    For each episode we record:
      - duration / timestamps
      - peak system score
      - GT overlap / GT match
      - per-model hot participation within the episode
      - per-model max score within the episode
    """
    one_eval = one[one["t"].astype(int) >= int(min_eval_t)].copy()
    df_eval = df[df["t"].astype(int) >= int(min_eval_t)].copy()

    t_to_ts: Dict[int, Any] = dict(zip(one_eval["t"].astype(int).tolist(), one_eval["ts"].tolist()))
    models_all = sorted(set(df_eval["model"].astype(str).tolist()))
    matched_gt_map = _episode_matched_gt_map(sys_eval)

    all_groups: Set[str] = set()
    if group_hot_dict_by_t:
        for d in group_hot_dict_by_t.values():
            if isinstance(d, dict):
                all_groups.update(str(k) for k in d.keys())
    if active_groups_by_t:
        for xs in active_groups_by_t.values():
            for g in xs or []:
                all_groups.add(str(g))
    groups_all = sorted(all_groups)

    details: List[Dict[str, Any]] = []

    for i, (s, e) in enumerate(sorted((int(a), int(b)) for a, b in episodes), start=1):
        ep_one = one_eval[(one_eval["t"].astype(int) >= int(s)) & (one_eval["t"].astype(int) <= int(e))].copy()
        ep_df = df_eval[(df_eval["t"].astype(int) >= int(s)) & (df_eval["t"].astype(int) <= int(e))].copy()

        t_vals = sorted(set(ep_one["t"].astype(int).tolist()))
        length_steps = int(e) - int(s) + 1
        length_minutes = (float(length_steps) * float(step_minutes)) if step_minutes is not None else None

        ep_sys_scores = pd.to_numeric(ep_one.get("system_score"), errors="coerce")
        if ep_sys_scores.notna().any():
            peak_idx = ep_sys_scores.idxmax()
            peak_system_score = _safe_float(ep_sys_scores.loc[peak_idx])
            peak_t = _safe_int(ep_one.loc[peak_idx, "t"])
        else:
            peak_system_score = None
            peak_t = None

        gt_overlaps: List[Dict[str, int]] = []
        for gi, (gs, ge) in enumerate(sorted((int(gs), int(ge)) for gs, ge in gt_windows), start=1):
            if not (int(e) < int(gs) or int(s) > int(ge)):
                gt_overlaps.append(
                    {
                        "gt_index": int(gi),
                        "gt_start_t": int(gs),
                        "gt_end_t": int(ge),
                    }
                )

        per_model: List[Dict[str, Any]] = []
        active_union: List[str] = []
        per_group: List[Dict[str, Any]] = []
        active_group_union: List[str] = []

        for model_name in models_all:
            hot_steps = 0
            first_hot_t: Optional[int] = None
            for t in t_vals:
                d = hot_dict_by_t.get(int(t)) or {}
                is_hot = bool(d.get(model_name)) if isinstance(d, dict) else False
                if is_hot:
                    hot_steps += 1
                    if first_hot_t is None:
                        first_hot_t = int(t)

            model_sub = ep_df[ep_df["model"].astype(str) == str(model_name)].copy()
            max_score = None
            mean_score = None
            if not model_sub.empty and score_col in model_sub.columns:
                score_series = pd.to_numeric(model_sub[score_col], errors="coerce")
                if score_series.notna().any():
                    max_score = _safe_float(score_series.max())
                    mean_score = _safe_float(score_series.mean())

            if hot_steps > 0:
                active_union.append(str(model_name))

            per_model.append(
                {
                    "model": str(model_name),
                    "hot_steps": int(hot_steps),
                    "hot_frac": float(hot_steps / length_steps) if length_steps > 0 else 0.0,
                    "first_hot_t": int(first_hot_t) if first_hot_t is not None else None,
                    "first_hot_ts": (
                        str(t_to_ts.get(int(first_hot_t)))
                        if first_hot_t is not None and t_to_ts.get(int(first_hot_t)) is not None
                        else None
                    ),
                    "max_score": max_score,
                    "mean_score": mean_score,
                }
            )

        per_model = sorted(
            per_model,
            key=lambda x: (
                -int(x["hot_steps"]),
                -(x["max_score"] if x["max_score"] is not None else -1.0),
                str(x["model"]),
            ),
        )

        for group_name in groups_all:
            group_hot_steps = 0
            first_group_hot_t: Optional[int] = None
            for t in t_vals:
                gd = (group_hot_dict_by_t or {}).get(int(t)) or {}
                is_group_hot = bool(gd.get(group_name)) if isinstance(gd, dict) else False
                if is_group_hot:
                    group_hot_steps += 1
                    if first_group_hot_t is None:
                        first_group_hot_t = int(t)

            if group_hot_steps > 0:
                active_group_union.append(str(group_name))

            per_group.append(
                {
                    "group": str(group_name),
                    "hot_steps": int(group_hot_steps),
                    "hot_frac": float(group_hot_steps / length_steps) if length_steps > 0 else 0.0,
                    "first_hot_t": int(first_group_hot_t) if first_group_hot_t is not None else None,
                    "first_hot_ts": (
                        str(t_to_ts.get(int(first_group_hot_t)))
                        if first_group_hot_t is not None and t_to_ts.get(int(first_group_hot_t)) is not None
                        else None
                    ),
                }
            )
        per_group.sort(key=lambda x: (-int(x["hot_steps"]), str(x["group"])))

        matched_gt = matched_gt_map.get((int(s), int(e)), [])
        details.append(
            {
                "episode_id": int(i),
                "t_start": int(s),
                "t_end": int(e),
                "ts_start": str(t_to_ts.get(int(s))) if t_to_ts.get(int(s)) is not None else None,
                "ts_end": str(t_to_ts.get(int(e))) if t_to_ts.get(int(e)) is not None else None,
                "length_steps": int(length_steps),
                "length_minutes": float(length_minutes) if length_minutes is not None else None,
                "peak_system_score": peak_system_score,
                "peak_t": int(peak_t) if peak_t is not None else None,
                "peak_ts": str(t_to_ts.get(int(peak_t))) if peak_t is not None and t_to_ts.get(int(peak_t)) is not None else None,
                "gt_matched": bool(len(matched_gt) > 0),
                "matched_gt_windows": matched_gt,
                "gt_overlap_count": int(len(gt_overlaps)),
                "gt_overlaps": gt_overlaps,
                "num_models_active": int(len(active_union)),
                "num_groups_active": int(len(set(active_group_union))),
                "active_groups_union": sorted(set(active_group_union)),
                "groups": per_group,
                "active_models_union": sorted(active_union),
                "models": per_model,
            }
        )

    return details


def summarize_episode_takeaways(
    episode_details: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Small aggregate summary for quick terminal / markdown reading.
    """
    if not episode_details:
        return {
            "episode_count": 0,
            "gt_matched_count": 0,
            "non_gt_count": 0,
            "top_models_gt_matched": [],
            "top_models_all": [],
            "strongest_non_gt_episode": None,
        }

    gt_eps = [ep for ep in episode_details if bool(ep.get("gt_matched"))]
    non_gt_eps = [ep for ep in episode_details if not bool(ep.get("gt_matched"))]

    def _aggregate_models(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        by_model: Dict[str, Dict[str, Any]] = {}
        for ep in rows:
            seen_in_episode: Set[str] = set()
            for m in ep.get("models") or []:
                model = str(m.get("model"))
                hot_steps = int(m.get("hot_steps") or 0)
                if model not in by_model:
                    by_model[model] = {
                        "model": model,
                        "episode_count": 0,
                        "total_hot_steps": 0,
                        "max_hot_frac": 0.0,
                    }
                if hot_steps > 0 and model not in seen_in_episode:
                    by_model[model]["episode_count"] += 1
                    seen_in_episode.add(model)
                by_model[model]["total_hot_steps"] += int(hot_steps)
                by_model[model]["max_hot_frac"] = max(
                    float(by_model[model]["max_hot_frac"]),
                    float(m.get("hot_frac") or 0.0),
                )
        vals = list(by_model.values())
        vals.sort(
            key=lambda x: (
                -int(x["episode_count"]),
                -int(x["total_hot_steps"]),
                -float(x["max_hot_frac"]),
                str(x["model"]),
            )
        )
        return vals

    strongest_non_gt = None

    strongest_gt = None

    if gt_eps:
        strongest_gt = sorted(
            gt_eps,
            key=lambda ep: (
                -(ep.get("peak_system_score") if ep.get("peak_system_score") is not None else -1.0),
                -int(ep.get("num_groups_active") or 0),
                -int(ep.get("num_models_active") or 0),
                -int(ep.get("length_steps") or 0),
            ),
        )[0]
        strongest_gt = {
            "episode_id": int(strongest_gt["episode_id"]),
            "t_start": int(strongest_gt["t_start"]),
            "t_end": int(strongest_gt["t_end"]),
            "ts_start": strongest_gt.get("ts_start"),
            "ts_end": strongest_gt.get("ts_end"),
            "length_steps": int(strongest_gt["length_steps"]),
            "peak_system_score": strongest_gt.get("peak_system_score"),
            "active_groups_union": list(strongest_gt.get("active_groups_union") or []),
            "active_models_union": list(strongest_gt.get("active_models_union") or []),
        }

    def _aggregate_groups(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        by_group: Dict[str, Dict[str, Any]] = {}
        for ep in rows:
            for g in ep.get("groups") or []:
                group = str(g.get("group"))
                hot_steps = int(g.get("hot_steps") or 0)
                if hot_steps <= 0:
                    continue
                if group not in by_group:
                    by_group[group] = {"group": group, "episode_count": 0, "total_hot_steps": 0, "max_hot_frac": 0.0}
                by_group[group]["episode_count"] += 1
                by_group[group]["total_hot_steps"] += int(hot_steps)
                by_group[group]["max_hot_frac"] = max(float(by_group[group]["max_hot_frac"]), float(g.get("hot_frac") or 0.0))
        vals = list(by_group.values())
        vals.sort(key=lambda x: (-int(x["episode_count"]), -int(x["total_hot_steps"]), -float(x["max_hot_frac"]), str(x["group"])))
        return vals

    if non_gt_eps:
        strongest_non_gt = sorted(
            non_gt_eps,
            key=lambda ep: (
                -(ep.get("peak_system_score") if ep.get("peak_system_score") is not None else -1.0),
                -int(ep.get("num_models_active") or 0),
                -int(ep.get("length_steps") or 0),
            ),
        )[0]
        strongest_non_gt = {
            "episode_id": int(strongest_non_gt["episode_id"]),
            "t_start": int(strongest_non_gt["t_start"]),
            "t_end": int(strongest_non_gt["t_end"]),
            "ts_start": strongest_non_gt.get("ts_start"),
            "ts_end": strongest_non_gt.get("ts_end"),
            "length_steps": int(strongest_non_gt["length_steps"]),
            "peak_system_score": strongest_non_gt.get("peak_system_score"),
            "active_models_union": list(strongest_non_gt.get("active_models_union") or []),
        }

    return {
        "episode_count": int(len(episode_details)),
        "gt_matched_count": int(len(gt_eps)),
        "non_gt_count": int(len(non_gt_eps)),
        "top_models_gt_matched": _aggregate_models(gt_eps)[:5],
        "top_models_all": _aggregate_models(episode_details)[:5],
        "strongest_non_gt_episode": strongest_non_gt,
    }


# -------------------------
# Detection episode derivation
# -------------------------

def _resolve_score_column(df: pd.DataFrame, score_key: str) -> str:
    if score_key in df.columns:
        return score_key
    if score_key == "anomaly_probability" and "p" in df.columns:
        return "p"
    if score_key == "p" and "p" in df.columns:
        return "p"
    if "score" in df.columns:
        return "score"
    if "p" in df.columns:
        return "p"
    raise ValueError(
        f"Run CSV missing score column for score_key='{score_key}'. "
        f"Have columns={list(df.columns)}"
    )


def build_model_hot_mask(
    df: pd.DataFrame,
    model_name: str,
    *,
    score_col: str,
    threshold: float,
    prefer_hot_by_model: bool,
    min_t: Optional[int] = None,
) -> Tuple[List[int], List[bool]]:
    """
    Return (t_index, hot_mask) for a specific model.
    prefer_hot_by_model: if True and hot_dict contains a key for this model, use it.
    Else: use (score_col >= threshold).
    """
    sub = df[df["model"] == model_name].copy()
    if min_t is not None:
        sub = sub[sub["t"].astype(int) >= int(min_t)].copy()
    if sub.empty:
        return [], []

    sub = sub.sort_values("t", kind="mergesort")
    t_index = sub["t"].astype(int).tolist()

    if prefer_hot_by_model:
        # If hot_by_model is present, it is system-decision-consistent.
        # We treat missing keys as False.
        hot_mask: List[bool] = []
        for d in sub["hot_dict"].tolist():
            if isinstance(d, dict) and (model_name in d):
                hot_mask.append(bool(d.get(model_name)))
            else:
                hot_mask.append(False)
        # Only use it if it provides ANY signal (otherwise fallback)
        if any(hot_mask):
            return t_index, hot_mask

    s = pd.to_numeric(sub[score_col], errors="coerce")
    hot = (s >= float(threshold)).fillna(False).astype(bool).tolist()
    return t_index, hot


# -------------------------
# Main summarize
# -------------------------

def _load_timebase_params(config_path: str) -> Dict[str, str]:
    cfg = yaml.safe_load(Path(config_path).read_text())
    if not isinstance(cfg, dict):
        raise ValueError("config YAML must be a mapping at top-level")

    tb = ((cfg.get("data") or {}).get("timebase") or {})
    if not isinstance(tb, dict):
        tb = {}

    mode = str(tb.get("mode") or "union").strip() or "union"
    on_missing = str(tb.get("on_missing") or "hold_last").strip() or "hold_last"
    return {
        "mode": mode,
        "on_missing": on_missing,
    }

def summarize(
    df: pd.DataFrame,
    *,
    config_path: str,
    out_dir: Optional[str],
    max_lag_steps: int,
    threshold_override: Optional[float],
    prefer_hot_by_model: bool,
    strict: bool,
) -> Dict[str, Any]:
    one = _dedup_steps(df)

    # Determine evaluation window: exclude warmup (if present).
    warmup_steps = int(one["in_warmup"].sum()) if "in_warmup" in one.columns else 0
    if "in_warmup" in one.columns:
        one_eval = one[~one["in_warmup"]].copy()
    else:
        one_eval = one.copy()

    if one_eval.empty:
        raise ValueError("All timesteps are warmup (or no data). Cannot score.")

    # Basic run stats
    t_min = int(one_eval["t"].min())
    t_max = int(one_eval["t"].max())
    n_steps = int(len(one_eval))
    step_minutes = _infer_step_minutes(one_eval["ts"])

    # hot_dict for “what fired” at system detection times
    hot_dict_by_t: Dict[int, Dict[str, Any]] = dict(zip(one_eval["t"].astype(int).tolist(), one_eval["hot_dict"].tolist()))

    # Load contracts
    model_sources = _config_model_sources(config_path)
    decision = _load_decision_params(config_path)
    predictive_warning_contract = _load_predictive_warning_contract(config_path)
    evaluation_mode = _load_evaluation_mode(config_path)
    timebase = _load_timebase_params(config_path)
    sys_gt_params = _load_system_gt_params(config_path, decision)
    gt_windows_cfg_by_source = _config_gt_windows_by_source(config_path)

    if not gt_windows_cfg_by_source and strict:
        raise ValueError("No GT labels found in config (strict mode). Provide data.sources[*].labels.event_windows.")

    # Alert episodes: runtime alert stream is canonical (no offline postprocess)
    sys_eps = _episodes_from_boolean(
        one_eval["alert"].astype(bool).tolist(),
        one_eval["t"].astype(int).tolist(),
    )

    # Score selection + effective threshold
    score_col = _resolve_score_column(df, decision.score_key)
    threshold = float(threshold_override) if threshold_override is not None else float(decision.threshold)

    # --- Map per-source GT windows -> t windows (source-aware)
    gt_windows_t_by_source: Dict[str, List[Tuple[int, int]]] = {}
    gt_excluded_by_source: Dict[str, List[str]] = {}
    for src, gt_windows_cfg in gt_windows_cfg_by_source.items():
        models_for_src = _models_consuming_source(model_sources, src)
        ts_to_t = _build_ts_to_t_for_models(df, models_for_src)
        mapped_windows: List[Tuple[int, int]] = []
        excluded_ts: List[str] = []

        legacy_singletons: List[int] = []
        for gw in gt_windows_cfg:
            if gw.start_ts == gw.end_ts:
                # legacy timestamp path; collapse later
                if gw.start_ts in ts_to_t:
                    legacy_singletons.append(int(ts_to_t[gw.start_ts]))
                else:
                    excluded_ts.append(gw.start_ts)
                continue

            missing = []
            if gw.start_ts not in ts_to_t:
                missing.append(gw.start_ts)
            if gw.end_ts not in ts_to_t:
                missing.append(gw.end_ts)
            if missing:
                excluded_ts.extend(missing)
                continue

            s_t = int(ts_to_t[gw.start_ts])
            e_t = int(ts_to_t[gw.end_ts])
            if e_t < s_t:
                s_t, e_t = e_t, s_t
            mapped_windows.append((s_t, e_t))

        if legacy_singletons:
            mapped_windows.extend(_collapse_t_points_to_windows(legacy_singletons))

        if warmup_steps > 0:
            clipped: List[Tuple[int, int]] = []
            for s_t, e_t in mapped_windows:
                if int(e_t) < int(t_min):
                    continue
                clipped.append((max(int(s_t), int(t_min)), int(e_t)))
            mapped_windows = clipped

        gt_windows_t_by_source[src] = _merge_overlapping_windows(mapped_windows)
        if excluded_ts:
            gt_excluded_by_source[src] = sorted(set(excluded_ts))

    if strict and (not any(bool(v) for v in gt_windows_t_by_source.values())):
        raise ValueError("All sources have empty GT windows after mapping (strict).")

    if (not strict) and gt_excluded_by_source:
        for src, ts_list in gt_excluded_by_source.items():
            log.warning(
                "coverage-aware eval: excluded %d GT onset(s) for source='%s' because timestamps were not present in run timeline: %s",
                len(ts_list),
                src,
                ts_list[:10],
            )

    # --- Per-model eval (model detection episodes vs that model's source GT onsets)
    by_model: Dict[str, Any] = {}
    for model_name in sorted(set(df["model"].astype(str).tolist())):
        srcs = model_sources.get(model_name) or []
        if not srcs:
            continue

        gt_model_windows = _merge_overlapping_windows(
            [w for s in srcs for w in (gt_windows_t_by_source.get(s) or [])]
        )
        gt_model_onsets = sorted(set(int(s) for s, _ in gt_model_windows))
        if not gt_model_windows:
            continue

        t_idx, hot_mask = build_model_hot_mask(
            df,
            model_name,
            score_col=score_col,
            threshold=threshold,
            prefer_hot_by_model=prefer_hot_by_model,
            min_t=t_min,
        )
        model_eps = _episodes_from_boolean(hot_mask, t_idx)

        ev = eval_event_starts_vs_episodes(
            one_eval,
            gt_model_windows,
            model_eps,
            max_lag_steps=max_lag_steps,
            step_minutes=step_minutes,
            hot_dict_by_t=None,
        )

        by_model[model_name] = {
            "sources": list(srcs),
            "threshold": float(threshold),
            "score_col": score_col,
            "prefer_hot_by_model": bool(prefer_hot_by_model),
            "gt_onsets": gt_model_onsets,
            "gt_windows": gt_model_windows,
            "gt_onsets_derived": gt_model_onsets,
            "episodes": model_eps,
            "gt_count": ev.gt_count,
            "episode_count": ev.episode_count,
            "precision": ev.precision,
            "recall": ev.recall,
            "f1": ev.f1,
            "lag_steps_stats": ev.lag_steps_stats,
            "lag_minutes_stats": ev.lag_minutes_stats,
            "misses": ev.misses,
            "false_positive_episodes": ev.false_positive_episodes,
        }

    predictive_warning_summary: Optional[Dict[str, Any]] = None
    sys_gt_windows: List[Tuple[int, int]] = []
    sys_def: Dict[str, Any] = {}
    if evaluation_mode == "predictive_warning" and predictive_warning_contract is not None:
        if predictive_warning_contract.target_source == "system":
            sys_gt_windows, sys_def = derive_system_gt_windows(
                one_eval,
                gt_windows_t_by_source,
                k=sys_gt_params.k,
            )
    sys_eval: Optional[GtOnsetWindowEval] = None

    if evaluation_mode == "predictive_warning":
        if predictive_warning_contract is None:
            raise ValueError(
                "config.evaluation.mode='predictive_warning' requires "
                "config.evaluation.predictive_warning"
            )

        if predictive_warning_contract.target_source == "system":
            target_event_ts = sorted(set(int(s) for s, _ in sys_gt_windows))
        else:
            target_event_ts = sorted(set(
                int(s) for s, _ in (gt_windows_t_by_source.get(predictive_warning_contract.target_source) or [])
            ))

        if strict and not target_event_ts:
            raise ValueError(
                "Predictive-warning target event stream is empty after GT mapping. "
                f"target_source='{predictive_warning_contract.target_source}'"
            )

        pw_eval = eval_predictive_warning_vs_episodes(
            one_eval,
            target_event_ts,
            sys_eps,
            contract=predictive_warning_contract,
            step_minutes=step_minutes,
            hot_dict_by_t=hot_dict_by_t,
        )
        predictive_warning_summary = {
            "contract": {
                "event_type": predictive_warning_contract.event_type,
                "target_source": predictive_warning_contract.target_source,
                "warning_window": {
                    "start_steps_before_event": predictive_warning_contract.warning_start_steps_before_event,
                    "end_steps_before_event": predictive_warning_contract.warning_end_steps_before_event,
                },
                "too_early_window": {
                    "min_steps_before_event": predictive_warning_contract.too_early_min_steps_before_event,
                },
            },
            "status": "implemented",
            "event_count": pw_eval.event_count,
            "episode_count": pw_eval.episode_count,
            "matched_events_in_warning_window": pw_eval.matched_events_in_warning_window,
            "first_alert_summary": pw_eval.first_alert_summary,
            "episode_summary": pw_eval.episode_summary,
            "lead_steps_stats_in_warning_window": pw_eval.lead_steps_stats_in_warning_window,
            "lead_minutes_stats_in_warning_window": pw_eval.lead_minutes_stats_in_warning_window,
            "per_event": [
                {
                    "event_t": r.event_t,
                    "event_ts": r.event_ts,
                    "first_alert_t": r.first_alert_t,
                    "first_alert_ts": r.first_alert_ts,
                    "first_alert_lead_steps": r.first_alert_lead_steps,
                    "first_alert_lead_minutes": r.first_alert_lead_minutes,
                    "first_alert_classification": r.first_alert_classification,
                    "has_any_in_warning_window": bool(r.has_any_in_warning_window),
                    "in_window_episode_starts": list(r.in_window_episode_starts),
                    "hot_models_first_alert": r.hot_models_first_alert,
                }
                for r in pw_eval.per_event
            ],
        }
    else:
        # --- System GT + system eval (window semantics)
        sys_gt_windows, sys_def = derive_system_gt_windows(
            one_eval,
            gt_windows_t_by_source,
            k=sys_gt_params.k,
        )

        if sys_gt_windows:
            sys_eval = eval_event_starts_vs_episodes(
                one_eval,
                sys_gt_windows,
                sys_eps,
                max_lag_steps=max_lag_steps,
                step_minutes=step_minutes,
                hot_dict_by_t=hot_dict_by_t,
            )

    group_hot_dict_by_t: Dict[int, Dict[str, Any]] = {
        int(t): (d if isinstance(d, dict) else {})
        for t, d in zip(
            one["t"].astype(int).tolist(),
            one["window_hot_group_dict"].tolist(),
        )
    }
    active_groups_by_t: Dict[int, List[str]] = {
        int(t): ([str(g) for g in xs] if isinstance(xs, list) else [])
        for t, xs in zip(
            one["t"].astype(int).tolist(),
            one["active_groups_list"].tolist(),
        )
    }

    episode_details = build_episode_details(
        one,
        df,
        sys_eps,
        score_col=score_col,
        step_minutes=step_minutes,
        hot_dict_by_t=hot_dict_by_t,
        group_hot_dict_by_t=group_hot_dict_by_t,
        active_groups_by_t=active_groups_by_t,
        gt_windows=sys_gt_windows,
        sys_eval=sys_eval,
        min_eval_t=t_min,
    )
    episode_takeaways = summarize_episode_takeaways(episode_details)

    # Emit summary
    summary: Dict[str, Any] = {
        "csv": None,
        "config": config_path,
        "gt_mapping_mode": "strict" if strict else "coverage_aware",
        "use_case_semantics": {
            "evaluation_mode": evaluation_mode,
            "predictive_warning_contract": (
                predictive_warning_contract.__dict__
                if predictive_warning_contract is not None
                else None
            ),
            "implemented_scoring_mode": evaluation_mode,
        },
        "predictive_warning_eval": predictive_warning_summary,
        "warmup_steps": int(warmup_steps),
        "eval_timesteps_excludes_warmup": True,
        "timesteps": {"min": t_min, "max": t_max, "count": n_steps},
        "time_range": {
            "start": str(one_eval["ts"].min()),
            "end": str(one_eval["ts"].max()),
        },
        "timebase": {
            "mode": timebase["mode"],
            "on_missing": timebase["on_missing"],
         },
        "step_minutes_inferred": step_minutes,
        "decision": {
            "method": decision.method,
            "score_key": decision.score_key,
            "score_col": score_col,
            "threshold_effective": float(threshold),
            "k": decision.k,
            "window_size": decision.window_size,
        },
        "alerts": {
            "alert_timesteps": int(one_eval["alert"].sum()),
            "alert_rate": float(one_eval["alert"].mean()) if n_steps else 0.0,
            "episodes": {
                "count": int(len(sys_eps)),
                "count_raw": len(sys_eps),
                "count_merged": len(sys_eps),
                "episodes_raw": [[int(a), int(b)] for a, b in sys_eps],
                "episodes_merged": [[int(a), int(b)] for a, b in sys_eps],
                "episodes_postprocessed": [[int(a), int(b)] for a, b in sys_eps],
                "episodes": sys_eps,
            },
            "episode_takeaways": episode_takeaways,
        },
        "ground_truth": {
            "source": "config.data.sources[*].labels.event_windows",
            "by_source_windows": gt_windows_t_by_source,
            "excluded_by_source": gt_excluded_by_source,
            "excluded_count": int(sum(len(v) for v in gt_excluded_by_source.values())),
            "by_model": by_model,
            "system": {
                "definition": {
                    "type": "event_start_detection",
                    "method": "kofn_active_sources_on_window_labels",
                    **sys_def,
                    "detection_rule": {
                        "episode_start_must_fall_in": f"[event_start, event_start + {int(max_lag_steps)}]"
                    },
                },
                "gt_windows": sys_gt_windows,
                "gt_onsets_derived": [int(s) for s, _ in sys_gt_windows],
                "eval": None,
            },
        },
        "max_lag_steps": int(max_lag_steps),
    }

    if sys_eval is not None:
        summary["ground_truth"]["system"]["eval"] = {
            "gt_count": sys_eval.gt_count,
            "episode_count": sys_eval.episode_count,
            "precision": sys_eval.precision,
            "recall": sys_eval.recall,
            "f1": sys_eval.f1,
            "matched_gt": int(sys_eval.matched_gt),
            "matched_episodes": int(sys_eval.matched_episodes),
            "lag_steps_stats": sys_eval.lag_steps_stats,
            "lag_minutes_stats": sys_eval.lag_minutes_stats,
            "misses": sys_eval.misses,
            "false_positive_episodes": sys_eval.false_positive_episodes,
            "per_event": [
                {
                    "gt_start_t": r.gt_start_t,
                    "gt_end_t": r.gt_end_t,
                    "gt_start_ts": r.gt_start_ts,
                    "gt_end_ts": r.gt_end_ts,
                    "detected": bool(r.detected),
                    "detection_t": r.detection_t,
                    "detection_ts": r.detection_ts,
                    "episode": list(r.episode) if r.episode is not None else None,
                    "lag_steps": r.lag_steps,
                    "lag_minutes": r.lag_minutes,
                    "hot_models": r.hot_models,
                }
                for r in sys_eval.per_event
            ],
        }

    # Write artifacts
    if out_dir:
        outp = Path(out_dir)
        outp.mkdir(parents=True, exist_ok=True)

        (outp / "run_summary.json").write_text(json.dumps(summary, indent=2))
        (outp / "episode_details.json").write_text(json.dumps(episode_details, indent=2))

        # Markdown snapshot
        md: List[str] = []
        md.append("# Run Summary\n\n")
        md.append(f"- Warmup steps (excluded from eval): **{warmup_steps}**\n")
        md.append(f"- Timesteps: **{t_min} → {t_max}** (n={n_steps})\n")
        md.append(f"- Time range: **{one_eval['ts'].min()} → {one_eval['ts'].max()}**\n")
        md.append(f"- GT mapping mode: **{'strict' if strict else 'coverage-aware'}**\n")
        md.append(f"- Use-case semantics: **{evaluation_mode}**\n")
        if predictive_warning_summary is not None:
            md.append(
                f"- Predictive-warning contract present: `{json.dumps(predictive_warning_summary['contract'], sort_keys=True)}`\n"
            )
            md.append(
                f"- Predictive-warning matched events in window: **{predictive_warning_summary['matched_events_in_warning_window']} / {predictive_warning_summary['event_count']}**\n"
            )
            md.append(
                f"- Predictive-warning first alert summary: `{predictive_warning_summary['first_alert_summary']}`\n"
            )
            md.append(
                f"- Predictive-warning episode summary: `{predictive_warning_summary['episode_summary']}`\n"
            )
            md.append(
                f"- Predictive-warning lead steps (in-window): `{predictive_warning_summary['lead_steps_stats_in_warning_window']}`\n"
            )
            md.append(
                f"- Predictive-warning lead minutes (in-window): `{predictive_warning_summary['lead_minutes_stats_in_warning_window']}`\n"
            )
        if step_minutes is not None:
            md.append(f"- Inferred step: **{step_minutes:.3f} min**\n")
        md.append(f"- System alert timesteps: **{int(one_eval['alert'].sum())}** ({float(one_eval['alert'].mean())*100:.2f}% of steps)\n")
        md.append(f"- System alert episodes: **{len(sys_eps)}**\n\n")
        md.append("## Episode takeaways\n\n")
        md.append(f"- GT-matched episodes: **{episode_takeaways['gt_matched_count']} / {episode_takeaways['episode_count']}**\n")
        md.append(f"- Non-GT episodes: **{episode_takeaways['non_gt_count']}**\n")
        if episode_takeaways.get("top_models_gt_matched"):
            md.append("- Top GT-matched models:\n")
            for row in episode_takeaways["top_models_gt_matched"]:
                md.append(
                    f"  - `{row['model']}`: episodes={row['episode_count']}, "
                    f"total_hot_steps={row['total_hot_steps']}, max_hot_frac={row['max_hot_frac']:.3f}\n"
                )
        if episode_takeaways.get("strongest_non_gt_episode"):
            sn = episode_takeaways["strongest_non_gt_episode"]
            md.append(
                f"- Strongest non-GT episode: **#{sn['episode_id']}** "
                f"t={sn['t_start']}..{sn['t_end']} "
                f"(len={sn['length_steps']}, peak_system_score={sn['peak_system_score']}) "
                f"active_models={sn['active_models_union']}\n"
            )
        md.append("\n")
        if gt_excluded_by_source:
            md.append("## Coverage-aware GT exclusions\n\n")
            md.append("The following GT onset timestamps were excluded from scoring because they were not present in the observed run timeline for the corresponding source.\n\n")
            md.append("| Source | Excluded GT count | Example timestamps |\n")
            md.append("|---|---:|---|\n")
            for src, ts_list in sorted(gt_excluded_by_source.items()):
                preview = ", ".join(ts_list[:3])
                md.append(f"| {src} | {len(ts_list)} | {preview} |\n")
            md.append("\n")
        md.append("## Decision\n")
        md.append(f"- method: `{decision.method}`\n")
        md.append(f"- score_col: `{score_col}` (score_key=`{decision.score_key}`)\n")
        md.append(f"- threshold_effective: **{threshold}**\n")
        md.append(f"- kofn_window: k={decision.k}, window_size={decision.window_size}, per_model_hits={decision.per_model_hits}\n\n")

        md.append("## Per-model eval (GT windows → detection episodes)\n\n")
        if by_model:
            md.append("| Model | Sources | GT | Episodes | P | R | F1 | Lag p50 (steps) |\n")
            md.append("|---|---|---:|---:|---:|---:|---:|---:|\n")
            for m, ev in by_model.items():
                p = ev.get("precision"); r = ev.get("recall"); f1 = ev.get("f1")
                lag_med = (ev.get("lag_steps_stats") or {}).get("median")
                md.append(
                    f"| {m} | {','.join(ev.get('sources') or [])} | {ev.get('gt_count',0)} | {ev.get('episode_count',0)} | "
                    f"{(p if p is not None else '—')} | {(r if r is not None else '—')} | {(f1 if f1 is not None else '—')} | "
                    f"{(lag_med if lag_med is not None else '—')} |\n"
                )
        else:
            md.append("_No per-model GT eval rows (missing GT or model_sources mapping)._ \n")

        md.append("\n## System eval\n\n")
        if evaluation_mode == "predictive_warning" and predictive_warning_summary is not None:
            md.append(f"- Target events: **{predictive_warning_summary['event_count']}**\n")
            md.append(f"- Matched in warning window: **{predictive_warning_summary['matched_events_in_warning_window']}**\n")
            md.append(f"- First alert summary: `{predictive_warning_summary['first_alert_summary']}`\n")
            md.append(f"- Episode summary: `{predictive_warning_summary['episode_summary']}`\n")
        else:
            if sys_eval is not None:
                md.append(f"- GT windows: **{sys_eval.gt_count}**\n")
                md.append(f"- Matched events: **{sys_eval.matched_gt}**\n")
                md.append(
                    f"- Precision / Recall / F1: **{_fmt_metric(sys_eval.precision)} / "
                    f"{_fmt_metric(sys_eval.recall)} / {_fmt_metric(sys_eval.f1)}**\n"
                )
                md.append(f"- Lag steps stats: `{sys_eval.lag_steps_stats}`\n")
                md.append(f"- Lag minutes stats: `{sys_eval.lag_minutes_stats}`\n")
            else:
                md.append("_No system GT events available for scoring._\n")

        (outp / "run_summary.md").write_text("".join(md))
        log.info("wrote %s", outp / "run_summary.json")
        log.info("wrote %s", outp / "episode_details.json")
        log.info("wrote %s", outp / "run_summary.md")

    return summary


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--run-dir", default=None, help="Directory containing run.csv + run.manifest.json (canonical)")
    ap.add_argument("--csv", default=None, help="Path to run.csv (optional if --run-dir provided)")
    ap.add_argument("--config", required=True, help="YAML config (for GT + decision params)")

    ap.add_argument("--out-dir", default=None, help="Write run_summary.json/md here (default: <run-dir>/analysis)")
    ap.add_argument("--log-level", default="INFO")

    ap.add_argument("--max-lag-steps", type=int, default=50)
    ap.add_argument("--threshold", type=float, default=None, help="Override decision.threshold for scoring")
    ap.add_argument("--prefer-hot-by-model", action="store_true", help="Prefer hot_by_model for per-model episodes if present")
    ap.add_argument("--non-strict", action="store_true", help="Allow missing/empty GT without raising")

    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    # Resolve paths from run-dir
    if args.run_dir:
        rd = Path(args.run_dir)
        if args.csv is None:
            args.csv = str(rd / "run.csv")
        if args.out_dir is None:
            args.out_dir = str(rd / "analysis")

    if not args.csv:
        ap.error("Need --run-dir OR --csv")

    df = load_run(args.csv)

    summary = summarize(
        df,
        config_path=args.config,
        out_dir=args.out_dir,
        max_lag_steps=int(args.max_lag_steps),
        threshold_override=args.threshold,
        prefer_hot_by_model=bool(args.prefer_hot_by_model),
        strict=(not bool(args.non_strict)),
    )

    # Attach csv path in json (nice for aggregators)
    if args.out_dir:
        p = Path(args.out_dir) / "run_summary.json"
        j = json.loads(p.read_text())
        j["csv"] = args.csv
        p.write_text(json.dumps(j, indent=2))

    # Print a tiny terminal summary
    sys_eval = ((summary.get("ground_truth") or {}).get("system") or {}).get("eval")
    print("\n=== analyze_run ===")
    print(f"run.csv: {args.csv}")
    print(f"timesteps: {summary['timesteps']['min']} → {summary['timesteps']['max']} (n={summary['timesteps']['count']})")
    print(f"alert episodes: {summary['alerts']['episodes']['count']}")
    if isinstance(sys_eval, dict):
        print(
            "system P/R/F1: "
            f"{_fmt_metric(sys_eval.get('precision'))}/"
            f"{_fmt_metric(sys_eval.get('recall'))}/"
            f"{_fmt_metric(sys_eval.get('f1'))}"
        )
    else:
        print("system eval: (no system GT windows or insufficient data)")

    ep_takeaways = ((summary.get("alerts") or {}).get("episode_takeaways") or {})
    if isinstance(ep_takeaways, dict) and ep_takeaways:
        gt_rows = ep_takeaways.get("top_models_gt_matched") or []
        if gt_rows:
            top_txt = ", ".join(
                f"{row['model']}({row['episode_count']} eps, {row['total_hot_steps']} hot)"
                for row in gt_rows[:3]
            )
            print(f"top GT-matched models: {top_txt}")
        strongest_non_gt = ep_takeaways.get("strongest_non_gt_episode")
        if isinstance(strongest_non_gt, dict):
            print(
                "strongest non-GT episode: "
                f"#{strongest_non_gt.get('episode_id')} "
                f"t={strongest_non_gt.get('t_start')}..{strongest_non_gt.get('t_end')} "
                f"peak={strongest_non_gt.get('peak_system_score')} "
                f"models={strongest_non_gt.get('active_models_union')}"
            )

    if args.out_dir:
        print(f"wrote: {Path(args.out_dir) / 'run_summary.json'}")
        print(f"wrote: {Path(args.out_dir) / 'episode_details.json'}")
        print(f"wrote: {Path(args.out_dir) / 'run_summary.md'}")


if __name__ == "__main__":
    main()
