#src/htm_monitor/cli/analyze_run.py

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

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


def _config_gt_onsets_by_source(config_path: str) -> Dict[str, List[str]]:
    """
    Load GT onset timestamp strings from config YAML:
      data.sources[*].labels.timestamps: [ "....", ... ]
    (We treat these as ONSET timestamps per your spec.)
    """
    cfg = yaml.safe_load(Path(config_path).read_text())
    if not isinstance(cfg, dict):
        raise ValueError("config YAML must be a mapping at top-level")

    sources = (cfg.get("data") or {}).get("sources") or []
    if not isinstance(sources, list):
        return {}

    out: Dict[str, List[str]] = {}
    for s in sources:
        if not isinstance(s, dict):
            continue
        name = s.get("name")
        if not isinstance(name, str) or not name:
            continue
        labels = s.get("labels") or {}
        if not isinstance(labels, dict):
            continue
        ts_list = labels.get("timestamps")
        if ts_list is None:
            continue
        if not isinstance(ts_list, list):
            raise ValueError(f"config.data.sources[{name}].labels.timestamps must be a list of strings")
        vals = [x for x in ts_list if isinstance(x, str) and x.strip()]
        if vals:
            out[str(name)] = vals
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

    # Load contracts
    model_sources = _config_model_sources(config_path)
    decision = _load_decision_params(config_path)
    timebase = _load_timebase_params(config_path)
    sys_gt_params = _load_system_gt_params(config_path, decision)
    gt_ts_by_source = _config_gt_onsets_by_source(config_path)

    if not gt_ts_by_source and strict:
        raise ValueError("No GT labels found in config (strict mode). Provide data.sources[*].labels.timestamps.")

    # Score selection + effective threshold
    score_col = _resolve_score_column(df, decision.score_key)
    threshold = float(threshold_override) if threshold_override is not None else float(decision.threshold)

    # Alert episodes (system detection)
    sys_eps = _episodes_from_boolean(one_eval["alert"].astype(bool).tolist(), one_eval["t"].astype(int).tolist())

    # hot_dict for “what fired” at system detection times
    hot_dict_by_t: Dict[int, Dict[str, Any]] = dict(zip(one_eval["t"].astype(int).tolist(), one_eval["hot_dict"].tolist()))

    # --- Map per-source GT onset timestamp strings -> t (source-aware)
    gt_onsets_t_by_source: Dict[str, List[int]] = {}
    gt_excluded_by_source: Dict[str, List[str]] = {}
    for src, gt_ts in gt_ts_by_source.items():
        models_for_src = _models_consuming_source(model_sources, src)
        ts_to_t = _build_ts_to_t_for_models(df, models_for_src)
        mapped_t, excluded_ts = _map_gt_strings_to_t(
            gt_ts,
            ts_to_t,
            strict=strict,
            ctx=f"source='{src}'",
        )
        gt_onsets_t_by_source[src] = mapped_t
        if excluded_ts:
            gt_excluded_by_source[src] = list(excluded_ts)
        # Drop GT that occurs during warmup from evaluation window
        if warmup_steps > 0:
            gt_onsets_t_by_source[src] = [t for t in gt_onsets_t_by_source[src] if int(t) >= int(t_min)]

    if strict and (not any(bool(v) for v in gt_onsets_t_by_source.values())):
        raise ValueError("All sources have empty GT after mapping (strict).")

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

        # union source GT onsets feeding this model
        gt_model_onsets = sorted(set(t for s in srcs for t in (gt_onsets_t_by_source.get(s) or [])))
        if not gt_model_onsets:
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

        ev = eval_onsets_vs_episodes(
            one_eval,
            gt_model_onsets,
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

    # --- System GT + system eval (kofn_window semantics)
    sys_gt_onsets, sys_def = derive_system_gt_onsets(
        one_eval,
        gt_onsets_t_by_source,
        k=sys_gt_params.k,
        window_size=sys_gt_params.window_size,
        per_model_hits=sys_gt_params.per_source_hits,
    )

    sys_eval: Optional[GtEval] = None
    if sys_gt_onsets:
        sys_eval = eval_onsets_vs_episodes(
            one_eval,
            sys_gt_onsets,
            sys_eps,
            max_lag_steps=max_lag_steps,
            step_minutes=step_minutes,
            hot_dict_by_t=hot_dict_by_t,
        )

    # Emit summary
    summary: Dict[str, Any] = {
        "csv": None,
        "config": config_path,
        "evaluation_mode": "strict" if strict else "coverage_aware",
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
            "per_model_hits": decision.per_model_hits,
        },
        "alerts": {
            "alert_timesteps": int(one_eval["alert"].sum()),
            "alert_rate": float(one_eval["alert"].mean()) if n_steps else 0.0,
            "episodes": {
                "count": int(len(sys_eps)),
                "episodes": sys_eps,
            },
        },
        "ground_truth": {
            "source": "config.data.sources[*].labels.timestamps (onsets)",
            "by_source_onsets": gt_onsets_t_by_source,
            "excluded_by_source": gt_excluded_by_source,
            "excluded_count": int(sum(len(v) for v in gt_excluded_by_source.values())),
            "by_model": by_model,
            "system": {
                "definition": {
                    "type": "kofn_window",
                    "method": sys_gt_params.method,
                    **sys_def,
                },
                "gt_onsets": sys_gt_onsets,
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
                    "gt_t": r.gt_t,
                    "gt_ts": r.gt_ts,
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

        # Markdown snapshot
        md: List[str] = []
        md.append("# Run Summary\n\n")
        md.append(f"- Warmup steps (excluded from eval): **{warmup_steps}**\n")
        md.append(f"- Timesteps: **{t_min} → {t_max}** (n={n_steps})\n")
        md.append(f"- Time range: **{one_eval['ts'].min()} → {one_eval['ts'].max()}**\n")
        md.append(f"- Evaluation mode: **{'strict' if strict else 'coverage-aware'}**\n")
        if step_minutes is not None:
            md.append(f"- Inferred step: **{step_minutes:.3f} min**\n")
        md.append(f"- System alert timesteps: **{int(one_eval['alert'].sum())}** ({float(one_eval['alert'].mean())*100:.2f}% of steps)\n")
        md.append(f"- System alert episodes: **{len(sys_eps)}**\n\n")
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

        md.append("## Per-model eval (GT onsets → detection episode starts)\n\n")
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

        md.append("\n## System eval (kofn_window-derived system GT onsets)\n\n")
        md.append(f"- System GT onsets: **{len(sys_gt_onsets)}**\n")
        if sys_eval is not None and sys_eval.precision is not None:
            md.append(f"- Precision / Recall / F1: **{sys_eval.precision:.3f} / {sys_eval.recall:.3f} / {sys_eval.f1:.3f}**\n")
            md.append(f"- Lag steps stats: `{sys_eval.lag_steps_stats}`\n")
            md.append(f"- Lag minutes stats: `{sys_eval.lag_minutes_stats}`\n")
        else:
            md.append("_System GT empty or insufficient data for scoring._\n")

        (outp / "run_summary.md").write_text("".join(md))
        log.info("wrote %s", outp / "run_summary.json")
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
    if isinstance(sys_eval, dict) and sys_eval.get("precision") is not None:
        print(f"system P/R/F1: {sys_eval['precision']:.3f}/{sys_eval['recall']:.3f}/{sys_eval['f1']:.3f}")
    else:
        print("system eval: (no system GT onsets or insufficient data)")
    if args.out_dir:
        print(f"wrote: {Path(args.out_dir) / 'run_summary.json'}")
        print(f"wrote: {Path(args.out_dir) / 'run_summary.md'}")


if __name__ == "__main__":
    main()