from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml


def parse_json_dict(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return x
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return {}
    if isinstance(x, str) and x.strip():
        v = json.loads(x)
        if not isinstance(v, dict):
            raise ValueError("Expected JSON object")
        return v
    return {}


def _episodes_from_boolean(mask: Sequence[bool], t_index: Sequence[int]) -> List[Tuple[int, int]]:
    eps: List[Tuple[int, int]] = []
    start: Optional[int] = None
    prev_t: Optional[int] = None

    for on, t in zip(mask, t_index):
        t = int(t)
        if on and start is None:
            start = t
            prev_t = t
            continue

        if on and start is not None:
            if prev_t is not None and t == prev_t + 1:
                prev_t = t
                continue
            eps.append((int(start), int(prev_t if prev_t is not None else start)))
            start = t
            prev_t = t
            continue

        if (not on) and start is not None:
            eps.append((int(start), int(prev_t if prev_t is not None else start)))
            start = None
            prev_t = None

    if start is not None:
        eps.append((int(start), int(prev_t if prev_t is not None else start)))
    return eps


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


def _infer_step_minutes(ts: pd.Series) -> Optional[float]:
    ts = ts.dropna()
    dt = ts.sort_values().diff().dropna()
    dt = dt[dt > pd.Timedelta(0)]
    if dt.empty:
        return None
    minutes = dt.median().total_seconds() / 60.0
    return float(minutes) if minutes > 0 else None


def _fmt_float(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    xf = float(x)
    if np.isnan(xf):
        return None
    return xf


@dataclass(frozen=True)
class GroupSpec:
    name: str
    members: List[str]
    min_instant_members: int
    min_group_warmth: float


@dataclass(frozen=True)
class DecisionSpec:
    score_key: str
    threshold: float
    group_k: int
    min_system_len: int
    groups: Dict[str, GroupSpec]


@dataclass(frozen=True)
class WindowSpec:
    name: Optional[str]
    kind: str
    start_ts: str
    end_ts: str


@dataclass(frozen=True)
class EvalResult:
    episodes: List[Tuple[int, int]]
    matched_gt_count: int
    matched_gt_windows: List[Tuple[int, int]]
    strict_fp_episodes: List[Tuple[int, int]]
    explanatory_matched_episodes: List[Tuple[int, int]]
    unexplained_fp_episodes: List[Tuple[int, int]]
    strict_precision: Optional[float]
    adjusted_precision: Optional[float]
    recall: Optional[float]
    gt_preserved: bool


def load_config(config_path: Path) -> Tuple[DecisionSpec, Dict[str, List[str]], Dict[str, List[WindowSpec]], int]:
    cfg = yaml.safe_load(config_path.read_text())
    if not isinstance(cfg, dict):
        raise ValueError("config must be a top-level mapping")

    models_cfg = cfg.get("models") or {}
    if not isinstance(models_cfg, dict) or not models_cfg:
        raise ValueError("config.models missing or empty")

    model_sources: Dict[str, List[str]] = {}
    for m, spec in models_cfg.items():
        if not isinstance(spec, dict):
            raise ValueError(f"config.models.{m} must be a mapping")
        srcs = spec.get("sources") or []
        if not isinstance(srcs, list) or not srcs:
            raise ValueError(f"config.models.{m}.sources must be a non-empty list[str]")
        model_sources[str(m)] = [str(s) for s in srcs]

    decision = cfg.get("decision") or {}
    if not isinstance(decision, dict):
        raise ValueError("config.decision must be a mapping")

    score_key = str(decision.get("score_key") or "p").strip() or "p"
    threshold = float(decision.get("threshold", 0.99))

    system = decision.get("system") or {}
    if not isinstance(system, dict):
        raise ValueError("config.decision.system must be a mapping")
    group_k = int(system.get("group_k", 1))
    min_system_len = int(system.get("min_system_len", 1))

    groups_cfg = decision.get("groups") or {}
    if not isinstance(groups_cfg, dict) or not groups_cfg:
        raise ValueError("config.decision.groups must be a non-empty mapping")

    groups: Dict[str, GroupSpec] = {}
    for gname, gspec in groups_cfg.items():
        if not isinstance(gspec, dict):
            raise ValueError(f"config.decision.groups.{gname} must be a mapping")
        members = [str(x) for x in (gspec.get("members") or [])]
        if not members:
            raise ValueError(f"config.decision.groups.{gname}.members must be non-empty")
        groups[str(gname)] = GroupSpec(
            name=str(gname),
            members=members,
            min_instant_members=int(gspec.get("min_instant_members", 1)),
            min_group_warmth=float(gspec.get("min_group_warmth", 0.5)),
        )

    sources_cfg = ((cfg.get("data") or {}).get("sources") or [])
    if not isinstance(sources_cfg, list):
        raise ValueError("config.data.sources must be a list")

    windows_by_source: Dict[str, List[WindowSpec]] = {}
    for s in sources_cfg:
        if not isinstance(s, dict):
            continue
        src_name = s.get("name")
        if not isinstance(src_name, str) or not src_name:
            continue
        labels = s.get("labels") or {}
        if not isinstance(labels, dict):
            continue
        event_windows = labels.get("event_windows") or []
        if not isinstance(event_windows, list):
            raise ValueError(f"config.data.sources[{src_name}].labels.event_windows must be a list")
        wins: List[WindowSpec] = []
        for ev in event_windows:
            if not isinstance(ev, dict):
                raise ValueError(f"config.data.sources[{src_name}].labels.event_windows entries must be mappings")
            start = ev.get("start")
            end = ev.get("end")
            if not isinstance(start, str) or not isinstance(end, str):
                raise ValueError(f"config.data.sources[{src_name}] event window start/end must be strings")
            wins.append(
                WindowSpec(
                    name=str(ev.get("name")).strip() if ev.get("name") is not None else None,
                    kind=str(ev.get("kind") or "primary_gt").strip() or "primary_gt",
                    start_ts=start.strip(),
                    end_ts=end.strip(),
                )
            )
        if wins:
            windows_by_source[str(src_name)] = wins

    gt_cfg = cfg.get("ground_truth") or {}
    sys_gt_cfg = gt_cfg.get("system") or {}
    lag_steps = int(sys_gt_cfg.get("max_lag_steps", 48))

    decision_spec = DecisionSpec(
        score_key=score_key,
        threshold=threshold,
        group_k=group_k,
        min_system_len=min_system_len,
        groups=groups,
    )
    return decision_spec, model_sources, windows_by_source, lag_steps


def load_run(run_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(run_csv)
    required = ["timestamp", "t", "model", "alert"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"run.csv missing required column '{col}'")

    df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["alert"] = (
        df["alert"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
        if df["alert"].dtype == object else df["alert"].astype(bool)
    )
    df["instant_hot_dict"] = df.get("instant_hot_by_model", pd.Series(["{}"] * len(df))).apply(parse_json_dict)
    df["model_warmth_dict"] = df.get("model_warmth_by_model", pd.Series(["{}"] * len(df))).apply(parse_json_dict)
    df["group_instant_count_dict"] = df.get("group_instant_count", pd.Series(["{}"] * len(df))).apply(parse_json_dict)
    df["group_warmth_dict"] = df.get("group_warmth", pd.Series(["{}"] * len(df))).apply(parse_json_dict)
    df["group_hot_dict"] = df.get("group_hot", pd.Series(["{}"] * len(df))).apply(parse_json_dict)
    df = df.sort_values(["t", "model"], kind="mergesort").reset_index(drop=True)
    return df


def load_episode_details(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("episode_details.json must be a list")
    return data


def dedup_steps(df: pd.DataFrame) -> pd.DataFrame:
    one = df.drop_duplicates("t", keep="first").copy()
    one = one.sort_values("t", kind="mergesort").reset_index(drop=True)
    return one


def _models_consuming_source(model_sources: Dict[str, List[str]], source: str) -> List[str]:
    return [m for m, srcs in model_sources.items() if source in srcs]


def _build_ts_to_t_for_models(df: pd.DataFrame, models: List[str]) -> Dict[str, int]:
    if not models:
        return {}
    sub = df[df["model"].isin(models)][["timestamp", "t"]].copy()
    if sub.empty:
        return {}
    g = sub.groupby("timestamp")["t"].nunique(dropna=False)
    bad = g[g > 1]
    if len(bad) > 0:
        raise ValueError("same timestamp maps to multiple t values")
    ts_to_t = (
        sub.drop_duplicates("timestamp", keep="first")
        .set_index("timestamp")["t"]
        .astype(int)
        .to_dict()
    )
    return {str(k): int(v) for k, v in ts_to_t.items() if isinstance(k, str) and k}


def _map_boundary_ts(
    target_ts: str,
    ts_to_t: Dict[str, int],
    *,
    is_start: bool,
) -> Optional[int]:
    if target_ts in ts_to_t:
        return int(ts_to_t[target_ts])

    if not ts_to_t:
        return None

    target_dt = pd.to_datetime(target_ts, errors="coerce")
    if pd.isna(target_dt):
        return None

    pairs: List[Tuple[pd.Timestamp, int]] = []
    for ts_str, t in ts_to_t.items():
        dt = pd.to_datetime(ts_str, errors="coerce")
        if pd.isna(dt):
            continue
        pairs.append((dt, int(t)))
    if not pairs:
        return None

    pairs.sort(key=lambda x: x[0])

    if is_start:
        candidates = [t for dt, t in pairs if dt >= target_dt]
        return int(candidates[0]) if candidates else None

    candidates = [t for dt, t in pairs if dt <= target_dt]
    return int(candidates[-1]) if candidates else None


def build_system_windows_from_config(
    df: pd.DataFrame,
    model_sources: Dict[str, List[str]],
    windows_by_source: Dict[str, List[WindowSpec]],
    *,
    k: int,
    kind: str,
) -> List[Tuple[int, int]]:
    by_source_t: Dict[str, List[Tuple[int, int]]] = {}
    one = dedup_steps(df)
    t_list = one["t"].astype(int).tolist()
    if not t_list:
        return []

    for src, wins in windows_by_source.items():
        selected = [w for w in wins if w.kind == kind]
        if not selected:
            continue
        models = _models_consuming_source(model_sources, src)
        ts_to_t = _build_ts_to_t_for_models(df, models)
        src_windows: List[Tuple[int, int]] = []
        for w in selected:
            s = _map_boundary_ts(w.start_ts, ts_to_t, is_start=True)
            e = _map_boundary_ts(w.end_ts, ts_to_t, is_start=False)
            if s is None or e is None:
                continue
            if e < s:
                continue
            src_windows.append((int(s), int(e)))
        if src_windows:
            by_source_t[src] = _merge_overlapping_windows(src_windows)

    if not by_source_t:
        return []

    t_min = int(min(t_list))
    t_max = int(max(t_list))
    n = int(t_max - t_min + 1)
    active_count = np.zeros(n, dtype=np.int32)

    for _, windows in by_source_t.items():
        mask = np.zeros(n, dtype=np.int32)
        for s, e in windows:
            s_idx = max(0, int(s) - t_min)
            e_idx = min(n - 1, int(e) - t_min)
            if s_idx <= e_idx:
                mask[s_idx:e_idx + 1] = 1
        active_count += mask

    sys_active = (active_count >= int(k)).astype(bool)
    sys_t = [t_min + i for i in range(n)]
    return _episodes_from_boolean(sys_active.tolist(), sys_t)


def eval_onset_windows_vs_episodes(
    gt_windows: List[Tuple[int, int]],
    episodes: List[Tuple[int, int]],
    explanatory_windows: List[Tuple[int, int]],
    *,
    max_lag_steps: int,
) -> EvalResult:
    gt_sorted = sorted((int(s), int(e)) for s, e in gt_windows)
    ep_sorted = sorted((int(s), int(e)) for s, e in episodes)

    used_eps: set[int] = set()
    used_gt: set[int] = set()
    matched_gt_windows: List[Tuple[int, int]] = []

    for gi, (gs, ge) in enumerate(gt_sorted):
        detect_end = int(gs) + int(max_lag_steps)
        for ei, (es, ee) in enumerate(ep_sorted):
            if ei in used_eps:
                continue
            es = int(es)
            if es < int(gs):
                continue
            if es > int(detect_end):
                break
            used_gt.add(gi)
            used_eps.add(ei)
            matched_gt_windows.append((int(gs), int(ge)))
            break

    strict_fp: List[Tuple[int, int]] = []
    detect_windows = [(int(gs), int(gs) + int(max_lag_steps)) for gs, _ in gt_sorted]
    for es, ee in ep_sorted:
        start_in_any = any(g0 <= int(es) <= g1 for g0, g1 in detect_windows)
        if not start_in_any:
            strict_fp.append((int(es), int(ee)))

    explanatory_matched = [
        ep for ep in strict_fp
        if any(not (int(ep[1]) < int(ws) or int(ep[0]) > int(we)) for ws, we in explanatory_windows)
    ]
    unexplained_fp = [
        ep for ep in strict_fp
        if ep not in explanatory_matched
    ]

    matched_gt_count = len(used_gt)
    gt_count = len(gt_sorted)

    strict_precision = None
    denom = len(used_eps) + len(strict_fp)
    if denom > 0:
        strict_precision = len(used_eps) / denom

    adjusted_precision = None
    denom_adj = len(used_eps) + len(unexplained_fp)
    if denom_adj > 0:
        adjusted_precision = len(used_eps) / denom_adj

    recall = None
    if gt_count > 0:
        recall = matched_gt_count / gt_count

    return EvalResult(
        episodes=ep_sorted,
        matched_gt_count=matched_gt_count,
        matched_gt_windows=matched_gt_windows,
        strict_fp_episodes=strict_fp,
        explanatory_matched_episodes=explanatory_matched,
        unexplained_fp_episodes=unexplained_fp,
        strict_precision=_fmt_float(strict_precision),
        adjusted_precision=_fmt_float(adjusted_precision),
        recall=_fmt_float(recall),
        gt_preserved=(matched_gt_count == gt_count),
    )


def replay_decision(
    one: pd.DataFrame,
    candidate: Dict[str, Any],
) -> pd.DataFrame:
    group_specs = candidate["groups"]
    group_k = int(candidate["group_k"])
    min_system_len = int(candidate["min_system_len"])

    rows: List[Dict[str, Any]] = []
    streak = 0

    for _, row in one.iterrows():
        t = int(row["t"])
        group_instant = row["group_instant_count_dict"] if isinstance(row["group_instant_count_dict"], dict) else {}
        group_warmth = row["group_warmth_dict"] if isinstance(row["group_warmth_dict"], dict) else {}

        group_hot: Dict[str, bool] = {}
        for gname, gspec in group_specs.items():
            instant_ct = int(float(group_instant.get(gname, 0.0)))
            warmth = float(group_warmth.get(gname, 0.0))
            is_hot = (
                instant_ct >= int(gspec["min_instant_members"])
                or warmth >= float(gspec["min_group_warmth"])
            )
            group_hot[gname] = bool(is_hot)

        system_hot_count = int(sum(1 for v in group_hot.values() if v))
        system_hot = bool(system_hot_count >= group_k)
        streak = streak + 1 if system_hot else 0
        alert = bool(streak >= min_system_len)

        rows.append(
            {
                "t": t,
                "timestamp": row["timestamp"],
                "ts": row["ts"],
                "group_hot": group_hot,
                "system_hot_count": system_hot_count,
                "system_hot": system_hot,
                "system_hot_streak": int(streak),
                "alert": alert,
            }
        )

    return pd.DataFrame(rows)


def build_candidate_from_config(decision: DecisionSpec) -> Dict[str, Any]:
    return {
        "group_k": int(decision.group_k),
        "min_system_len": int(decision.min_system_len),
        "groups": {
            gname: {
                "min_instant_members": int(g.min_instant_members),
                "min_group_warmth": float(g.min_group_warmth),
            }
            for gname, g in decision.groups.items()
        },
    }


def safe_upper_bounds_from_gt_episodes(
    episode_details: List[Dict[str, Any]],
    decision: DecisionSpec,
) -> Dict[str, Any]:
    gt_eps = [ep for ep in episode_details if bool(ep.get("gt_matched"))]
    out: Dict[str, Any] = {
        "group_k_max": None,
        "min_system_len_max": None,
        "groups": {},
    }

    if not gt_eps:
        return out

    out["group_k_max"] = min(int(ep.get("num_groups_active") or 0) for ep in gt_eps)
    out["min_system_len_max"] = min(int(float(ep.get("peak_system_hot_streak") or 0.0)) for ep in gt_eps)

    for gname in decision.groups:
        gt_supporting = []
        for ep in gt_eps:
            group_rows = {str(g["group"]): g for g in (ep.get("groups") or [])}
            if gname in group_rows and int(group_rows[gname].get("hot_steps") or 0) > 0:
                gt_supporting.append(group_rows[gname])
        if gt_supporting:
            out["groups"][gname] = {
                "min_instant_members_max": min(
                    int(float(g.get("max_instant_count") or 0.0)) for g in gt_supporting
                ),
                "min_group_warmth_max_observed": min(
                    float(g.get("max_warmth") or 0.0) for g in gt_supporting
                ),
            }
        else:
            out["groups"][gname] = {
                "min_instant_members_max": None,
                "min_group_warmth_max_observed": None,
            }

    return out


def candidate_grid(
    decision: DecisionSpec,
    one: pd.DataFrame,
    *,
    warmth_step: float,
    exhaustive: bool,
) -> Iterable[Tuple[str, Dict[str, Any]]]:
    base = build_candidate_from_config(decision)

    yield ("current", json.loads(json.dumps(base)))

    for v in range(decision.group_k, len(decision.groups) + 1):
        cand = json.loads(json.dumps(base))
        cand["group_k"] = int(v)
        yield (f"group_k={v}", cand)

    max_streak = int(pd.to_numeric(one.get("system_hot_streak"), errors="coerce").fillna(0).max())
    for v in range(decision.min_system_len, max(1, max_streak) + 1):
        cand = json.loads(json.dumps(base))
        cand["min_system_len"] = int(v)
        yield (f"min_system_len={v}", cand)

    for gname, g in decision.groups.items():
        for v in range(g.min_instant_members, len(g.members) + 1):
            cand = json.loads(json.dumps(base))
            cand["groups"][gname]["min_instant_members"] = int(v)
            yield (f"{gname}.min_instant_members={v}", cand)

    if exhaustive:
        warmth_values_by_group: Dict[str, List[float]] = {}
        for gname, g in decision.groups.items():
            current = float(g.min_group_warmth)
            vals = np.arange(current, 1.0000001 + warmth_step, warmth_step)
            warmth_values_by_group[gname] = sorted(set(round(float(v), 6) for v in vals if 0.0 <= v <= 1.000001))

        gnames = list(decision.groups.keys())
        for gname in gnames:
            for v in warmth_values_by_group[gname]:
                cand = json.loads(json.dumps(base))
                cand["groups"][gname]["min_group_warmth"] = float(v)
                yield (f"{gname}.min_group_warmth={v:.3f}", cand)


def candidate_signature(cand: Dict[str, Any]) -> str:
    return json.dumps(cand, sort_keys=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-csv", required=True)
    ap.add_argument("--episode-details", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--warmth-step", type=float, default=0.05)
    ap.add_argument("--max-lag-steps", type=int, default=None)
    ap.add_argument("--exhaustive", action="store_true")
    args = ap.parse_args()

    run_csv = Path(args.run_csv)
    episode_details_path = Path(args.episode_details)
    config_path = Path(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    decision, model_sources, windows_by_source, cfg_lag_steps = load_config(config_path)
    lag_steps = int(args.max_lag_steps) if args.max_lag_steps is not None else int(cfg_lag_steps)

    df = load_run(run_csv)
    one = dedup_steps(df)
    one["group_instant_count_dict"] = one["group_instant_count_dict"].apply(lambda x: x if isinstance(x, dict) else {})
    one["group_warmth_dict"] = one["group_warmth_dict"].apply(lambda x: x if isinstance(x, dict) else {})

    episode_details = load_episode_details(episode_details_path)
    step_minutes = _infer_step_minutes(one["ts"])

    primary_gt_windows = build_system_windows_from_config(
        df,
        model_sources,
        windows_by_source,
        k=decision.group_k,
        kind="primary_gt",
    )
    explanatory_windows = build_system_windows_from_config(
        df,
        model_sources,
        windows_by_source,
        k=decision.group_k,
        kind="explanatory",
    )

    seen: set[str] = set()
    results: List[Dict[str, Any]] = []

    for label, cand in candidate_grid(
        decision,
        one,
        warmth_step=float(args.warmth_step),
        exhaustive=bool(args.exhaustive),
    ):
        sig = candidate_signature(cand)
        if sig in seen:
            continue
        seen.add(sig)

        replay = replay_decision(one, cand)
        episodes = _episodes_from_boolean(
            replay["alert"].astype(bool).tolist(),
            replay["t"].astype(int).tolist(),
        )
        ev = eval_onset_windows_vs_episodes(
            primary_gt_windows,
            episodes,
            explanatory_windows,
            max_lag_steps=lag_steps,
        )

        results.append(
            {
                "label": label,
                "candidate": cand,
                "episode_count": len(ev.episodes),
                "matched_gt_count": ev.matched_gt_count,
                "strict_fp_count": len(ev.strict_fp_episodes),
                "explanatory_matched_count": len(ev.explanatory_matched_episodes),
                "unexplained_fp_count": len(ev.unexplained_fp_episodes),
                "strict_precision": ev.strict_precision,
                "adjusted_precision": ev.adjusted_precision,
                "recall": ev.recall,
                "gt_preserved": ev.gt_preserved,
                "episodes": ev.episodes,
                "strict_fp_episodes": ev.strict_fp_episodes,
                "explanatory_matched_episodes": ev.explanatory_matched_episodes,
                "unexplained_fp_episodes": ev.unexplained_fp_episodes,
            }
        )

    current = next(r for r in results if r["label"] == "current")
    gt_safe = [r for r in results if bool(r["gt_preserved"])]
    gt_safe_sorted = sorted(
        gt_safe,
        key=lambda r: (
            -(r["adjusted_precision"] if r["adjusted_precision"] is not None else -1.0),
            -(r["strict_precision"] if r["strict_precision"] is not None else -1.0),
            int(r["unexplained_fp_count"]),
            int(r["episode_count"]),
            str(r["label"]),
        ),
    )

    report = {
        "inputs": {
            "run_csv": str(run_csv),
            "episode_details": str(episode_details_path),
            "config": str(config_path),
            "step_minutes": step_minutes,
            "max_lag_steps": int(lag_steps),
        },
        "current": current,
        "primary_gt_windows": primary_gt_windows,
        "explanatory_windows": explanatory_windows,
        "safe_upper_bounds_from_gt_episodes": safe_upper_bounds_from_gt_episodes(
            episode_details,
            decision,
        ),
        "gt_safe_candidate_count": len(gt_safe),
        "best_gt_safe_candidates": gt_safe_sorted[:25],
        "all_candidates": results,
        "separability_assessment": {
            "has_any_gt_safe_improvement": any(
                (r["adjusted_precision"] or -1.0) > (current["adjusted_precision"] or -1.0)
                for r in gt_safe
            ),
            "best_adjusted_precision": gt_safe_sorted[0]["adjusted_precision"] if gt_safe_sorted else None,
            "current_adjusted_precision": current["adjusted_precision"],
            "best_unexplained_fp_count": min((r["unexplained_fp_count"] for r in gt_safe), default=None),
            "current_unexplained_fp_count": current["unexplained_fp_count"],
        },
    }

    (out_dir / "decision_boundary_report.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame(
        [
            {
                "label": r["label"],
                "episode_count": r["episode_count"],
                "matched_gt_count": r["matched_gt_count"],
                "strict_fp_count": r["strict_fp_count"],
                "explanatory_matched_count": r["explanatory_matched_count"],
                "unexplained_fp_count": r["unexplained_fp_count"],
                "strict_precision": r["strict_precision"],
                "adjusted_precision": r["adjusted_precision"],
                "recall": r["recall"],
                "gt_preserved": r["gt_preserved"],
            }
            for r in results
        ]
    ).to_csv(out_dir / "decision_boundary_candidates.csv", index=False)

    print(f"wrote {(out_dir / 'decision_boundary_report.json')}")
    print(f"wrote {(out_dir / 'decision_boundary_candidates.csv')}")
    print(f"current adjusted precision: {current['adjusted_precision']}")
    if gt_safe_sorted:
        best = gt_safe_sorted[0]
        print(f"best gt-safe candidate: {best['label']}")
        print(f"best adjusted precision: {best['adjusted_precision']}")
        print(f"best unexplained fp count: {best['unexplained_fp_count']}")


if __name__ == "__main__":
    main()
