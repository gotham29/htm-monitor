import argparse
import json
import yaml
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Set
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


log = logging.getLogger("analyze_run")

def parse_hot(x):
    if isinstance(x, dict):
        return x
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return {}
    if isinstance(x, str) and x.strip():
        # Canonical serialization for CSV is JSON. Anything else is treated as empty.
        try:
            v = json.loads(x)
        except Exception:
            return {}
        return v if isinstance(v, dict) else {}
    return {}


def _load_manifest(manifest_path: str) -> Dict[str, Any]: 
    """
    Load manifest JSON as dict with minimal validation.
    """
    p = Path(manifest_path)
    if not p.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    m = json.loads(p.read_text())
    if not isinstance(m, dict):
        raise ValueError("manifest JSON must be an object")
    return m


def load_run(path: str):
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["timestamp"])
    # If alert comes in as 0/1 numeric, astype(bool) is fine. If it's "True"/"False", coerce.
    if df["alert"].dtype == object:
        df["alert"] = df["alert"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
    else:
        df["alert"] = df["alert"].astype(bool)
    df["hot_dict"] = df.get("hot_by_model", "").apply(parse_hot)
    # Defensive invariant lock: ensure stable ordering for all downstream logic.
    if "t" in df.columns:
        df = df.sort_values(["t", "model"], kind="mergesort").reset_index(drop=True)
    return df


def _dedup_steps(df: pd.DataFrame) -> pd.DataFrame:
    # one row per timestep (system-level fields live duplicated across models)
    df = df.sort_values(["t", "model"], kind="mergesort")
    one = df.drop_duplicates("t", keep="first").copy()
    one = one.sort_values("t", kind="mergesort").reset_index(drop=True)
    return one


def _episodes_from_boolean(mask: Sequence[bool], t_index: Sequence[int]) -> List[Tuple[int, int]]:
    """
    Convert a boolean mask aligned to t_index into contiguous [start_t, end_t] episodes (inclusive).
    Assumes t_index is increasing by 1; if not, still works but episode gaps are based on adjacency in the list.
    """
    eps: List[Tuple[int, int]] = []
    start: Optional[int] = None
    prev_t: Optional[int] = None
    for on, t in zip(mask, t_index):
        if on and start is None:
            start = t
            prev_t = t
            continue
        if on and start is not None:
            # continue episode if adjacent in index
            if prev_t is not None and t == prev_t + 1:
                prev_t = t
                continue
            # break episode if gap
            eps.append((start, prev_t if prev_t is not None else start))
            start = t
            prev_t = t
            continue
        if (not on) and start is not None:
            eps.append((start, prev_t if prev_t is not None else start))
            start = None
            prev_t = None
    if start is not None:
        eps.append((start, prev_t if prev_t is not None else start))
    return eps


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, float) and np.isnan(x):
            return None
        return float(x)
    except Exception:
        return None


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


def _load_decision_params_from_yaml_strict(config_path: str) -> Tuple[int, int, int, str]:
    """
    Return (k, window_size, per_model_hits, score_key) from config YAML decision block.
    Fail-fast if missing/malformed.
    """
    cfg = yaml.safe_load(Path(config_path).read_text())
    if not isinstance(cfg, dict):
        raise ValueError("config YAML must be a mapping at top-level")
    decision = cfg.get("decision")
    if not isinstance(decision, dict):
        raise ValueError("config.decision must be a mapping")
    k = decision.get("k")
    if k is None:
        raise ValueError("config.decision.k missing")
    if not isinstance(k, int):
        raise ValueError("config.decision.k must be an int")
    win = decision.get("window")
    if not isinstance(win, dict):
        raise ValueError("config.decision.window must be a mapping")
    size = win.get("size")
    if size is None:
        raise ValueError("config.decision.window.size missing")
    if not isinstance(size, int):
        raise ValueError("config.decision.window.size must be an int")
    if k <= 0:
        raise ValueError("config.decision.k must be > 0")
    if size <= 0:
        raise ValueError("config.decision.window.size must be > 0")
    per_model_hits = win.get("per_model_hits", 1)
    if not isinstance(per_model_hits, int):
        raise ValueError("config.decision.window.per_model_hits must be an int")
    if per_model_hits <= 0:
        raise ValueError("config.decision.window.per_model_hits must be > 0")
    score_key = decision.get("score_key") or "p"
    if not isinstance(score_key, str) or not score_key.strip():
        raise ValueError("config.decision.score_key must be a non-empty string when provided")
    return int(k), int(size), int(per_model_hits), str(score_key)


def _resolve_score_column(df: pd.DataFrame, score_key: str) -> str:
    """
    Decide which column in the run CSV to use as the per-model score stream.
    Priority:
      1) explicit score column (recommended future-proof schema)
      2) exact score_key column, if present
      3) backwards-compat: anomaly_probability -> p
      4) fallback to p if present
    """
    if "score" in df.columns:
        return "score"
    if score_key in df.columns:
        return score_key
    if score_key == "anomaly_probability" and "p" in df.columns:
        return "p"
    if "p" in df.columns:
        return "p"
    raise ValueError(f"Run CSV missing score column for score_key='{score_key}'. Have columns={list(df.columns)}")


def _manifest_model_sources(manifest: Dict[str, Any]) -> Dict[str, List[str]]:
    models = manifest.get("models")
    if not isinstance(models, dict):
        raise ValueError("manifest.models missing or not an object")
    ms = models.get("model_sources")
    if not isinstance(ms, dict):
        raise ValueError("manifest.models.model_sources missing or not an object")
    out: Dict[str, List[str]] = {}
    for m, srcs in ms.items():
        if isinstance(srcs, list) and all(isinstance(s, str) and s for s in srcs):
            out[str(m)] = list(srcs)
    if not out:
        raise ValueError("manifest.models.model_sources is empty or invalid")
    return out


def _manifest_gt_by_source(manifest: Dict[str, Any]) -> Dict[str, List[str]]:
    data = manifest.get("data")
    if not isinstance(data, dict):
        raise ValueError("manifest.data missing or not an object")
    gt = data.get("ground_truth")
    if not isinstance(gt, dict):
        return {}
    by = gt.get("by_source") or {}
    if not isinstance(by, dict):
        return {}
    out: Dict[str, List[str]] = {}
    for src, lst in by.items():
        if isinstance(lst, list):
            vals = [x for x in lst if isinstance(x, str) and x.strip()]
            if vals:
                out[str(src)] = vals
    return out


def _config_gt_by_source(config_path: str) -> Dict[str, List[str]]:
    """
    Load GT timestamp STRINGS from config YAML (no datetime mapping here).
    Missing labels is allowed per-source.
    """
    cfg = yaml.safe_load(Path(config_path).read_text())
    if not isinstance(cfg, dict):
        raise ValueError("config YAML must be a mapping at top-level")
    sources = (cfg.get("data") or {}).get("sources") or []
    out: Dict[str, List[str]] = {}
    if not isinstance(sources, list):
        return out
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
            out[name] = vals
    return out


def _models_consuming_source(model_sources: Dict[str, List[str]], source: str) -> List[str]:
    return [m for m, srcs in model_sources.items() if source in srcs]


def _build_ts_to_t_for_models(df: pd.DataFrame, models: List[str]) -> Dict[str, int]:
    """
    Deterministic mapping from timestamp_string -> t for a subset of models.
    Contract: (timestamp, model) -> t is unique; for a fixed timestamp the t is identical across those models.
    """
    if not models:
        return {}
    sub = df[df["model"].isin(models)][["timestamp", "t"]].copy()
    if sub.empty:
        return {}
    # Ensure mapping is deterministic: any timestamp maps to exactly one t.
    g = sub.groupby("timestamp")["t"].nunique(dropna=False)
    bad = g[g > 1]
    if len(bad) > 0:
        # show a small sample for debugging
        sample_ts = bad.index.tolist()[:5]
        raise ValueError(f"Non-deterministic mapping: same timestamp maps to multiple t values. Sample timestamps={sample_ts}")
    # use first t per timestamp
    ts_to_t = sub.drop_duplicates("timestamp", keep="first").set_index("timestamp")["t"].astype(int).to_dict()
    return {str(k): int(v) for k, v in ts_to_t.items() if isinstance(k, str) and k}


def _map_gt_strings_to_t(
    gt_ts: List[str],
    ts_to_t: Dict[str, int],
    *,
    strict: bool,
    ctx: str,
) -> List[int]:
    out: List[int] = []
    missing: List[str] = []
    for ts in gt_ts:
        if ts in ts_to_t:
            out.append(int(ts_to_t[ts]))
        else:
            missing.append(ts)
    if missing and strict:
        raise ValueError(f"GT timestamp(s) missing from run timeline for {ctx}: {missing[:10]}{' ...' if len(missing)>10 else ''}")
    return sorted(set(out))


def _episodes_from_series_crossing(piv: pd.DataFrame, model_col: str, thr: float) -> List[Tuple[int, int]]:
    if model_col not in piv.columns:
        return []
    s = pd.to_numeric(piv[model_col], errors="coerce")
    mask = (s >= float(thr)).fillna(False).astype(bool).tolist()
    t_index = piv.index.astype(int).tolist()
    return _episodes_from_boolean(mask, t_index)


def _derive_system_gt_kofn_window(
    one: pd.DataFrame,
    gt_t_by_source: Dict[str, List[int]],
    *,
    k: int,
    window_size: int,
    per_model_hits: int,
) -> List[int]:
    """
    System GT consistent with kofn_window semantics:
      - For each source, mark timesteps where that source has a GT "hit".
      - A source is "hot" at t if (# hits in [t-window_size+1, t]) >= per_model_hits.
      - System GT at t if >= k sources are hot at t.
    """
    if k <= 0:
        return []
    if window_size <= 0:
        return []
    if per_model_hits <= 0:
        return []

    t_list = one["t"].astype(int).tolist()
    if not t_list:
        return []
    t_min = int(t_list[0])
    t_max = int(t_list[-1])
    n = t_max - t_min + 1

    # For each source, build a 0/1 hit vector over the full contiguous t range.
    hot_by_src: Dict[str, np.ndarray] = {}
    for src, hits in (gt_t_by_source or {}).items():
        arr = np.zeros(n, dtype=np.int32)
        for g in hits or []:
            gi = int(g) - t_min
            if 0 <= gi < n:
                arr[gi] = 1
        if arr.sum() == 0:
            continue
        # rolling window sum via cumulative sum
        c = np.cumsum(arr, dtype=np.int32)
        # window sum at i: c[i] - c[i-window_size] (with floor at 0)
        win = window_size
        sums = c.copy()
        if win < n:
            sums[win:] = c[win:] - c[:-win]
        # if win >= n, sums already equals c (hits from start)
        hot = (sums >= int(per_model_hits)).astype(np.int32)
        hot_by_src[str(src)] = hot

    if not hot_by_src:
        return []

    # Count hot sources per timestep
    hot_count = np.zeros(n, dtype=np.int32)
    for hot in hot_by_src.values():
        hot_count += hot
    sys_mask = hot_count >= int(k)
    sys_t = [t_min + i for i, on in enumerate(sys_mask.tolist()) if on]
    return sys_t


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


def _eval_against_gt(one: pd.DataFrame, gt_t: List[int], episodes: List[Tuple[int, int]], max_lag_steps: int) -> GtEval:
    """
    Episode-level scoring:
      - an episode is a detection if its start is within [gt, gt+max_lag_steps]
      - each episode can match at most one GT and vice versa (greedy in time order)
    Lag is measured as episode_start_t - gt_t (>=0).
    """
    gt_t_sorted = sorted(gt_t)
    ep_sorted = sorted(episodes, key=lambda x: x[0])

    used_eps = set()
    used_gt = set()
    lags_steps: List[float] = []
    lags_minutes: List[float] = []

    # timestep -> timestamp mapping (for minutes lag)
    t_to_ts: Dict[int, Any] = dict(zip(one["t"].astype(int).tolist(), one["ts"].tolist()))
    # timestep -> hot_dict mapping (for "what fired")
    t_to_hot: Dict[int, Any] = dict(zip(one["t"].astype(int).tolist(), one.get("hot_dict", pd.Series([{}]*len(one))).tolist()))

    def _ts_str(x: Any) -> Optional[str]:
        if x is None:
            return None
        try:
            return str(pd.Timestamp(x))
        except Exception:
            try:
                return str(x)
            except Exception:
                return None

    def _hot_models_at_t(t: int) -> List[str]:
        d = t_to_hot.get(int(t)) or {}
        if not isinstance(d, dict):
            return []
        out = []
        for k, v in d.items():
            if bool(v):
                out.append(str(k))
        return sorted(out)

    per_event: List[GtPerEventRow] = []

    for gi, g in enumerate(gt_t_sorted):
        best_ep_idx: Optional[int] = None
        best_ep_start: Optional[int] = None
        best_episode: Optional[Tuple[int, int]] = None
        for ei, (s, e) in enumerate(ep_sorted):
            if ei in used_eps:
                continue
            if s < g:
                continue
            if s > g + max_lag_steps:
                break
            best_ep_idx = ei
            best_ep_start = s
            best_episode = (s, e)
            break
        if best_ep_idx is not None and best_ep_start is not None:
            used_gt.add(gi)
            used_eps.add(best_ep_idx)
            lag = best_ep_start - g
            lags_steps.append(float(lag))
            # compute minutes if timestamps exist
            ts_g = t_to_ts.get(int(g))
            ts_s = t_to_ts.get(int(best_ep_start))
            if ts_g is not None and ts_s is not None:
                dt = (pd.Timestamp(ts_s) - pd.Timestamp(ts_g)).total_seconds() / 60.0
                if dt >= 0:
                    lags_minutes.append(float(dt))

            # per-event row
            lag_minutes: Optional[float] = None
            if ts_g is not None and ts_s is not None:
                try:
                    lag_minutes = (pd.Timestamp(ts_s) - pd.Timestamp(ts_g)).total_seconds() / 60.0
                except Exception:
                    lag_minutes = None

            per_event.append(
                GtPerEventRow(
                    gt_t=int(g),
                    gt_ts=_ts_str(ts_g),
                    detected=True,
                    detection_t=int(best_ep_start),
                    detection_ts=_ts_str(ts_s),
                    episode=best_episode,
                    lag_steps=int(lag),
                    lag_minutes=float(lag_minutes) if (lag_minutes is not None and not math.isnan(lag_minutes)) else None,
                    hot_models=_hot_models_at_t(int(best_ep_start)),
                )
            )
        else:
            # missed GT
            ts_g = t_to_ts.get(int(g))
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
    gt_count = len(gt_t_sorted)

    precision = (matched_eps / ep_count) if ep_count else None
    recall = (matched_gt / gt_count) if gt_count else None
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)

    misses = [gt_t_sorted[i] for i in range(gt_count) if i not in used_gt]
    fp_eps = [ep_sorted[i] for i in range(ep_count) if i not in used_eps]

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


def summarize(
    df: pd.DataFrame,
    threshold: float,
    config: Optional[str],
    max_lag_steps: int,
    out_dir: Optional[str],
    manifest_path: Optional[str] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    one = _dedup_steps(df)

    print("\n=== RUN SUMMARY ===")
    t_min = int(one["t"].min())
    t_max = int(one["t"].max())
    print("Timesteps:", t_min, "→", t_max, f"(n={len(one)})")
    print("Time range:", str(one["ts"].min()), "→", str(one["ts"].max()))

    total_alerts = int(one["alert"].sum())
    print("Total alert timesteps:", total_alerts, f"({(total_alerts/len(one))*100:.2f}% of steps)" if len(one) else "")

    alert_rows = one[one["alert"]]
    if len(alert_rows):
        print("\nAlert timesteps:")
        for _, r in alert_rows.iterrows():
            print(f"t={r['t']} ts={r['ts']} hot={r.get('hot_dict')}")
    else:
        print("\nNo alerts fired.")

    # Episodes (so we can talk in “incidents”, not just per-step alerts)
    episodes = _episodes_from_boolean(one["alert"].tolist(), one["t"].astype(int).tolist())
    ep_lens = [e - s + 1 for (s, e) in episodes]
    print("\nAlert episodes:", len(episodes))
    if episodes:
        print("Episode lengths (steps):", f"min={min(ep_lens)} median={np.median(ep_lens):.1f} max={max(ep_lens)}")

    print(f"\n=== Per-model threshold crossings (p>={threshold}) ===")
    score_key = "p"
    if config:
        _k, _w, _h, score_key = _load_decision_params_from_yaml_strict(config)
    score_col = _resolve_score_column(df, score_key)
    piv = df.pivot_table(index="t", columns="model", values=score_col, aggfunc="first").sort_index()

    for col in piv.columns:
        count = int((piv[col] >= threshold).sum())
        print(f"{col}: {count} crossings")

    # Optional GT evaluation:
    # Contract:
    #  - Prefer YAML config GT (by_source) if provided
    #  - Else use manifest GT (by_source) if manifest_path is provided
    #  - Mapping is source-aware and exact-match on timestamp strings, before any dedup.

    gt_eval: Optional[GtEval] = None
    gt_t: List[int] = []
    gt_source: Optional[str] = None
    gt_eval_by_model: Dict[str, Any] = {}
    gt_eval_system: Optional[GtEval] = None
    system_gt_t: List[int] = []
    system_gt_def: Optional[Dict[str, Any]] = None

    gt_ts_by_source: Dict[str, List[str]] = {}
    model_sources: Optional[Dict[str, List[str]]] = None
    if manifest_path:
        manifest = _load_manifest(manifest_path)
        model_sources = _manifest_model_sources(manifest)
        if not config:
            gt_ts_by_source = _manifest_gt_by_source(manifest)
            gt_source = "manifest"

    if config:
        gt_ts_by_source = _config_gt_by_source(config)
        gt_source = "config"

    if gt_ts_by_source:
        if model_sources is None:
            raise ValueError("GT evaluation requires manifest_path to provide model_sources mapping (model -> sources)")

        # --- per-source GT mapped to timesteps (SOURCE-AWARE, EXACT MATCH) ---
        gt_t_by_source: Dict[str, List[int]] = {}
        for src, gt_ts in gt_ts_by_source.items():
            models_for_src = _models_consuming_source(model_sources, src)
            ts_to_t = _build_ts_to_t_for_models(df, models_for_src)
            gt_t_by_source[src] = _map_gt_strings_to_t(gt_ts, ts_to_t, strict=strict, ctx=f"source='{src}'")

        # Strict mode: if ALL sources have empty GT, raise (otherwise evaluation is meaningless).
        any_gt = any(bool(v) for v in gt_t_by_source.values())
        if (not any_gt) and strict:
            raise ValueError("No GT labels found for any source (strict mode). Provide labels.timestamps or run non-strict.")

        # --- per-model eval (diagnostic): p-crossing episodes vs source GT ---
        piv = df.pivot_table(index="t", columns="model", values=score_col, aggfunc="first").sort_index()
        for model_name in piv.columns:
            # source(s) are defined by manifest contract, not string matching
            srcs = model_sources.get(str(model_name), []) if model_sources else []
            if not srcs:
                continue
            # For diagnostic eval, if model has multiple sources, union their GT hits.
            gt_model_t: List[int] = sorted(set(t for s in srcs for t in (gt_t_by_source.get(s) or [])))
            if not gt_model_t:
                continue
            model_eps = _episodes_from_series_crossing(piv, str(model_name), threshold)
            model_eval = _eval_against_gt(one, gt_model_t, model_eps, max_lag_steps=max_lag_steps)
            gt_eval_by_model[str(model_name)] = {
                "sources": list(srcs),
                "gt_count": model_eval.gt_count,
                "episode_count": model_eval.episode_count,
                "precision": model_eval.precision,
                "recall": model_eval.recall,
                "f1": model_eval.f1,
                "lag_steps_stats": model_eval.lag_steps_stats,
                "lag_minutes_stats": model_eval.lag_minutes_stats,
                "misses": model_eval.misses,
                "false_positive_episodes": model_eval.false_positive_episodes,
            }

        # --- system-level eval: system alert episodes vs derived k-of-n overlap GT ---
        k_cfg, win_cfg, hits_cfg, score_key = _load_decision_params_from_yaml_strict(config)
        system_gt_t = _derive_system_gt_kofn_window(
            one,
            gt_t_by_source,
            k=int(k_cfg),
            window_size=int(win_cfg),
            per_model_hits=int(hits_cfg),
        )
        system_gt_def = {"k": int(k_cfg), "window_size": int(win_cfg), "per_model_hits": int(hits_cfg), "source": "config.decision.kofn_window"}
        if system_gt_t:
            gt_eval_system = _eval_against_gt(one, system_gt_t, episodes, max_lag_steps=max_lag_steps)

        # Optional legacy union-of-all debug: union GT over sources in TIMESTEP SPACE (still exact).
        gt_t = sorted(set(t for v in gt_t_by_source.values() for t in (v or [])))
        if gt_t:
            gt_eval = _eval_against_gt(one, gt_t, episodes, max_lag_steps=max_lag_steps)

        if gt_eval_by_model:
            print("\n=== Ground-truth evaluation: per-model (p-crossing episodes vs per-source GT) ===")
            for m, ev in gt_eval_by_model.items():
                p = ev.get("precision"); r = ev.get("recall"); f1 = ev.get("f1")
                if p is not None and r is not None and f1 is not None:
                    print(f"{m}: P/R/F1 = {p:.3f}/{r:.3f}/{f1:.3f}")
                else:
                    print(f"{m}: insufficient data")

        if gt_eval_system is not None:
            print("\n=== Ground-truth evaluation: system-level (system alert vs derived k-of-n overlap GT) ===")
            print("System GT def:", system_gt_def)
            print(f"System GT events: {gt_eval_system.gt_count}")
            if gt_eval_system.precision is not None:
                print(f"Precision: {gt_eval_system.precision:.3f}")
            if gt_eval_system.recall is not None:
                print(f"Recall:    {gt_eval_system.recall:.3f}")
            if gt_eval_system.f1 is not None:
                print(f"F1:        {gt_eval_system.f1:.3f}")

    summary: Dict[str, Any] = {
        "csv": None,
        "timesteps": {"min": t_min, "max": t_max, "count": int(len(one))},
        "time_range": {"start": str(one["ts"].min()), "end": str(one["ts"].max())},
        "alerts": {
            "alert_timesteps": total_alerts,
            "alert_rate": float(total_alerts / len(one)) if len(one) else 0.0,
            "episodes": {
                "count": int(len(episodes)),
                "lengths_steps": ep_lens,
            },
        },
        "threshold": float(threshold),
        "per_model_crossings": {str(c): int((piv[c] >= threshold).sum()) for c in piv.columns},
        "ground_truth": None,
    }
    if gt_eval is not None:
        summary["ground_truth"] = {
            "source": gt_source,
            "legacy_union_of_all": True,
            "gt_event_timesteps": gt_t,
            "max_lag_steps": int(max_lag_steps),
            "precision": gt_eval.precision,
            "recall": gt_eval.recall,
            "f1": gt_eval.f1,
            "matched_gt": int(gt_eval.matched_gt),
            "matched_episodes": int(gt_eval.matched_episodes),
            "lag_steps_stats": gt_eval.lag_steps_stats,
            "lag_minutes_stats": gt_eval.lag_minutes_stats,
            "misses": gt_eval.misses,
            "false_positive_episodes": gt_eval.false_positive_episodes,
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
                for r in gt_eval.per_event
            ],
        }

    # Attach new layers even if legacy union is missing
    if summary.get("ground_truth") is None:
        summary["ground_truth"] = {
            "source": gt_source,
            "legacy_union_of_all": False,
        }
    if gt_eval_by_model:
        summary["ground_truth"]["by_model"] = gt_eval_by_model
    if gt_eval_system is not None:
        summary["ground_truth"]["system"] = {
            "definition": system_gt_def,
            "gt_event_timesteps": system_gt_t,
            "max_lag_steps": int(max_lag_steps),
            "precision": gt_eval_system.precision,
            "recall": gt_eval_system.recall,
            "f1": gt_eval_system.f1,
            "matched_gt": int(gt_eval_system.matched_gt),
            "matched_episodes": int(gt_eval_system.matched_episodes),
            "lag_steps_stats": gt_eval_system.lag_steps_stats,
            "lag_minutes_stats": gt_eval_system.lag_minutes_stats,
            "misses": gt_eval_system.misses,
            "false_positive_episodes": gt_eval_system.false_positive_episodes,
        }

    # Write snapshot artifacts
    if out_dir:
        outp = Path(out_dir)
        outp.mkdir(parents=True, exist_ok=True)
        (outp / "run_summary.json").write_text(json.dumps(summary, indent=2))

        # small Markdown snapshot for README-friendly reporting
        md_lines = []
        md_lines.append("# Run Summary\n")
        md_lines.append(f"- Timesteps: **{t_min} → {t_max}** (n={len(one)})\n")
        md_lines.append(f"- Time range: **{one['ts'].min()} → {one['ts'].max()}**\n")
        md_lines.append(f"- Alert timesteps: **{total_alerts}** ({(total_alerts/len(one))*100:.2f}% of steps)\n" if len(one) else "- Alert timesteps: **0**\n")
        md_lines.append(f"- Alert episodes: **{len(episodes)}**\n")
        if episodes:
            md_lines.append(f"  - Episode length (steps): min **{min(ep_lens)}**, median **{np.median(ep_lens):.1f}**, max **{max(ep_lens)}**\n")
        md_lines.append("\n## Per-model crossings\n")
        for m, c in summary["per_model_crossings"].items():
            md_lines.append(f"- {m}: {c}\n")
        if gt_eval is not None:
            md_lines.append("\n## Ground-truth eval (episode-level)\n")
            md_lines.append(f"- Source: **{gt_source}**\n")
            md_lines.append(f"- GT events: **{gt_eval.gt_count}**\n")
            md_lines.append(f"- Precision / Recall / F1: **{gt_eval.precision:.3f} / {gt_eval.recall:.3f} / {gt_eval.f1:.3f}**\n" if (gt_eval.precision is not None and gt_eval.recall is not None and gt_eval.f1 is not None) else "")
            md_lines.append(f"- Lag steps stats: `{gt_eval.lag_steps_stats}`\n")
            md_lines.append(f"- Lag minutes stats: `{gt_eval.lag_minutes_stats}`\n")
            if gt_eval.misses:
                md_lines.append(f"- Missed GTs (first 20): `{gt_eval.misses[:20]}`\n")
            if gt_eval.false_positive_episodes:
                md_lines.append(f"- False-positive episodes (first 10): `{gt_eval.false_positive_episodes[:10]}`\n")
            # Per-event table for instant understanding
            md_lines.append("\n## Per-event results\n\n")
            md_lines.append("| # | Ground Truth Time | Detected? | Detection Time | Lag (min) | Episode (t) | Hot Models |\n")
            md_lines.append("|---:|---|:---:|---|---:|---|---|\n")
            for i, r in enumerate(gt_eval.per_event, start=1):
                det = "✅" if r.detected else "❌"
                gt_time = r.gt_ts or f"t={r.gt_t}"
                det_time = r.detection_ts or "—"
                lag_min = f"{r.lag_minutes:.1f}" if (r.lag_minutes is not None) else "—"
                ep = f"{r.episode[0]}→{r.episode[1]}" if r.episode is not None else "—"
                hot = ",".join([m.replace('Twitter_volume_', '').replace('_model','') for m in (r.hot_models or [])]) if r.hot_models else "—"
                md_lines.append(f"| {i} | {gt_time} | {det} | {det_time} | {lag_min} | {ep} | {hot} |\n")
        (outp / "run_summary.md").write_text("".join(md_lines))

        print(f"\n[analyze_run] wrote -> {outp / 'run_summary.json'}")
        print(f"[analyze_run] wrote -> {outp / 'run_summary.md'}")

    return summary


def plot_diagnostics(df, threshold=0.997):
    one = df.drop_duplicates("t").set_index("t")
    p_pivot = df.pivot_table(index="t", columns="model", values="p", aggfunc="first")

    plt.figure()
    for col in p_pivot.columns:
        plt.plot(p_pivot.index, p_pivot[col], label=col)
    plt.axhline(threshold, linestyle="--", color="black")
    plt.title("Anomaly Probability (p)")
    plt.xlabel("timestep")
    plt.ylabel("p")
    plt.legend()
    plt.show()

    hot_counts = one["hot_dict"].apply(
        lambda d: sum(1 for v in d.values() if v) if isinstance(d, dict) else 0
    )

    plt.figure()
    plt.plot(one.index, hot_counts, label="hot_count")
    plt.plot(one.index, one["alert"].astype(int), label="alert")
    plt.title("Decision Dynamics")
    plt.xlabel("timestep")
    plt.ylabel("count")
    plt.legend()
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    # Canonical flag
    ap.add_argument("--csv", required=False, help="Run CSV produced by pipeline (canonical flag)")
    ap.add_argument("--run-dir", default=None, help="Canonical: run directory containing run.csv and run.manifest.json")
    # Back-compat aliases (your command uses these)
    ap.add_argument("--in-csv", dest="csv", help="Alias for --csv")
    ap.add_argument("--out-dir", default=None, help="If set, writes run_summary.json and run_summary.md here")
    ap.add_argument("--outdir", dest="out_dir", help="Alias for --out-dir")
    ap.add_argument("--manifest", default=None, help="Back-compat: accepted but not required by this script")
    ap.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    ap.add_argument("--threshold", type=float, default=0.997, help="Per-model p threshold for crossings/plots")
    ap.add_argument("--config", default=None, help="Optional YAML config to load GT labels.timestamps from")
    ap.add_argument("--max-lag-steps", type=int, default=50, help="Max allowed detection lag (steps) after a GT event")
    ap.add_argument("--no-plots", action="store_true", help="Skip plotting (useful for large runs / CI)")
    ap.add_argument("--non-strict", action="store_true", help="Allow missing GT timestamps / empty GT without raising")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))

    # If run-dir is provided, fill defaults from it unless explicitly overridden.
    if args.run_dir:
        rd = Path(args.run_dir)
        if args.csv is None:
            args.csv = str(rd / "run.csv")
        if args.manifest is None:
            m = rd / "run.manifest.json"
            if m.exists():
                args.manifest = str(m)
        if args.out_dir is None:
            args.out_dir = str(rd / "analysis")

    if not args.csv:
        ap.error("the following arguments are required: --csv (or --in-csv) or --run-dir")

    manifest_path = args.manifest

    if (manifest_path is None) and args.csv:
        p = Path(args.csv)
        auto = p.with_suffix(".manifest.json")
        if auto.exists():
            manifest_path = str(auto)

    df = load_run(args.csv)
    summary = summarize(
        df,
        threshold=args.threshold,
        config=args.config,
        max_lag_steps=args.max_lag_steps,
        out_dir=args.out_dir,
        manifest_path=manifest_path,
        strict=(not bool(args.non_strict)),
    )

    # (manifest GT eval is now handled inside summarize(), so JSON/MD are consistent)
    # attach csv path in emitted JSON (nice when you aggregate results later)
    if args.out_dir:
        p = Path(args.out_dir) / "run_summary.json"
        try:
            j = json.loads(p.read_text())
            j["csv"] = args.csv
            j["config"] = args.config
            p.write_text(json.dumps(j, indent=2))
        except Exception:
            pass

    if not args.no_plots:
        plot_diagnostics(df, threshold=args.threshold)


if __name__ == "__main__":
    main()