#cli/plot_run_figure.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import yaml


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _load_config(path: Path) -> Dict[str, Any]:
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict):
        raise ValueError("config must parse to a mapping")
    return cfg


def _episodes_from_alert(one: pd.DataFrame) -> List[Tuple[int, int]]:
    eps: List[Tuple[int, int]] = []
    on = one["alert"].astype(bool).tolist()
    ts = one["t"].astype(int).tolist()
    start: Optional[int] = None
    prev_t: Optional[int] = None

    for flag, t in zip(on, ts):
        if flag and start is None:
            start = int(t)
            prev_t = int(t)
            continue
        if flag and start is not None:
            if prev_t is not None and int(t) == prev_t + 1:
                prev_t = int(t)
            else:
                eps.append((int(start), int(prev_t if prev_t is not None else start)))
                start = int(t)
                prev_t = int(t)
            continue
        if (not flag) and start is not None:
            eps.append((int(start), int(prev_t if prev_t is not None else start)))
            start = None
            prev_t = None

    if start is not None:
        eps.append((int(start), int(prev_t if prev_t is not None else start)))
    return eps


def _dedup_steps(df: pd.DataFrame) -> pd.DataFrame:
    one = df.drop_duplicates("t", keep="first").copy()
    one = one.sort_values("t", kind="mergesort").reset_index(drop=True)
    return one


def _choose_models(summary: Dict[str, Any], top_n: int) -> List[str]:
    by_model = ((summary.get("ground_truth") or {}).get("by_model") or {})
    scored: List[Tuple[int, str]] = []
    fallback: List[str] = []

    for model_name, row in by_model.items():
        fallback.append(model_name)
        episodes = row.get("episodes") or []
        if episodes:
            scored.append((int(episodes[0][0]), str(model_name)))

    if scored:
        scored = sorted(scored, key=lambda x: (x[0], x[1]))
        return [m for _, m in scored[:top_n]]

    return sorted(fallback)[:top_n]


def _model_to_feature(cfg: Dict[str, Any], model_name: str) -> Optional[str]:
    models = cfg.get("models") or {}
    spec = models.get(model_name) or {}
    feats = spec.get("features") or []
    feats = [f for f in feats if f != "timestamp"]
    if not feats:
        return None
    return str(feats[0])


def _warning_window_bounds(summary: Dict[str, Any]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    pw = summary.get("predictive_warning_eval") or {}
    contract = pw.get("contract") or {}
    ww = contract.get("warning_window") or {}
    per_event = pw.get("per_event") or []
    if not per_event:
        return None, None, None

    event_t = per_event[0].get("event_t")
    start_before = ww.get("start_steps_before_event")
    end_before = ww.get("end_steps_before_event")
    if event_t is None or start_before is None or end_before is None:
        return None, None, None

    event_t = int(event_t)
    start_t = event_t - int(start_before)
    end_t = event_t - int(end_before)
    return start_t, end_t, event_t


def _first_alert_annotation(summary: Dict[str, Any]) -> Optional[str]:
    pw = summary.get("predictive_warning_eval") or {}
    per_event = pw.get("per_event") or []
    if not per_event:
        return None
    row = per_event[0]
    lead = row.get("first_alert_lead_steps")
    cls = row.get("first_alert_classification")
    if lead is None or cls is None:
        return None
    return f"First system alert: {lead} steps before failure ({cls})"


def _minmax01(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    vmin = s.min()
    vmax = s.max()
    if pd.isna(vmin) or pd.isna(vmax) or float(vmax) <= float(vmin):
        return pd.Series([0.5] * len(s), index=s.index, dtype=float)
    return (s - vmin) / (vmax - vmin)


def _shade_context(
    ax,
    *,
    warm_start_ts,
    warm_end_ts,
    warning_start_ts,
    warning_end_ts,
    event_ts,
) -> None:
    if warm_start_ts is not None and warm_end_ts is not None:
        ax.axvspan(warm_start_ts, warm_end_ts, alpha=0.08, color="gray")
    if warning_start_ts is not None and warning_end_ts is not None:
        ax.axvspan(warning_start_ts, warning_end_ts, alpha=0.12, color="magenta")
    if event_ts is not None:
        ax.axvline(event_ts, color="magenta", linestyle=":", linewidth=2.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--top-models", type=int, default=4)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    analysis_dir = run_dir / "analysis"
    csv_path = run_dir / "run.csv"
    summary_path = analysis_dir / "run_summary.json"
    config_path = Path(args.config)
    out_path = Path(args.out) if args.out else (analysis_dir / "run_figure.png")

    df = pd.read_csv(csv_path)
    df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["alert"] = pd.to_numeric(df["alert"], errors="coerce").fillna(0).astype(int)
    df["system_score"] = pd.to_numeric(df["system_score"], errors="coerce")
    df["p"] = pd.to_numeric(df["p"], errors="coerce")
    if "in_warmup" in df.columns:
        df["in_warmup"] = pd.to_numeric(df["in_warmup"], errors="coerce").fillna(0).astype(int)
    else:
        df["in_warmup"] = 0

    df = df.sort_values(["t", "model"], kind="mergesort").reset_index(drop=True)
    one = _dedup_steps(df)

    summary = _load_json(summary_path)
    cfg = _load_config(config_path)
    subtitle = _first_alert_annotation(summary)

    chosen_models = _choose_models(summary, max(1, int(args.top_models)))
    chosen_features = {m: _model_to_feature(cfg, m) for m in chosen_models}

    start_t, end_t, event_t = _warning_window_bounds(summary)
    alert_eps = _episodes_from_alert(one)

    warm = one[one["in_warmup"] == 1]
    warm_start_ts = warm["ts"].min() if not warm.empty else None
    warm_end_ts = warm["ts"].max() if not warm.empty else None

    warning_start_ts = None
    warning_end_ts = None
    failure_ts = None
    if start_t is not None and end_t is not None and event_t is not None:
        start_ts = one.loc[one["t"] == start_t, "ts"]
        end_ts = one.loc[one["t"] == end_t, "ts"]
        event_ts = one.loc[one["t"] == event_t, "ts"]
        warning_start_ts = start_ts.iloc[0] if not start_ts.empty else None
        warning_end_ts = end_ts.iloc[0] if not end_ts.empty else None
        failure_ts = event_ts.iloc[0] if not event_ts.empty else None

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 2.0, 1.5], hspace=0.18)

    ax_raw = fig.add_subplot(gs[0, 0])
    ax_p = fig.add_subplot(gs[1, 0], sharex=ax_raw)
    ax_sys = fig.add_subplot(gs[2, 0], sharex=ax_raw)

    # Panel 1: normalized sensor traces (shape/drift comparison)
    for model_name in chosen_models:
        feature = chosen_features.get(model_name)
        if not feature:
            continue
        sub = df[df["model"] == model_name].sort_values("t", kind="mergesort")
        if feature not in sub.columns and "raw" not in sub.columns:
            continue
        # raw sensor value is reconstructed from config feature name via score CSV row grouping;
        # run.csv does not store raw sensor values directly, so use source CSV-aligned "score" fallback
        # only if a dedicated feature column is unavailable.
        # For this first version we use score-free axis from the source data path encoded in config.
        source_name = ((cfg.get("models") or {}).get(model_name) or {}).get("sources", [None])[0]
        if source_name is None:
            continue
        src_cfg = None
        for s in ((cfg.get("data") or {}).get("sources") or []):
            if s.get("name") == source_name:
                src_cfg = s
                break
        if src_cfg is None:
            continue
        source_csv = Path(src_cfg["path"])
        src_df = pd.read_csv(source_csv)
        src_df["ts"] = pd.to_datetime(src_df[src_cfg["timestamp_col"]], errors="coerce")
        val_col = list((src_cfg.get("fields") or {}).values())[0]
        src_df[val_col] = pd.to_numeric(src_df[val_col], errors="coerce")
        src_df["value_norm"] = _minmax01(src_df[val_col])
        ax_raw.plot(src_df["ts"], src_df["value_norm"], linewidth=2.0, label=f"{feature}")

    _shade_context(
        ax_raw,
        warm_start_ts=warm_start_ts,
        warm_end_ts=warm_end_ts,
        warning_start_ts=warning_start_ts,
        warning_end_ts=warning_end_ts,
        event_ts=failure_ts,
    )
    ax_raw.set_title("Selected sensor trajectories (normalized 0–1)")
    ax_raw.set_ylim(-0.05, 1.05)

    ax_raw.grid(True, alpha=0.3)
    ax_raw.legend(loc="upper left", ncol=2, frameon=True)

    # Panel 2: anomaly probability traces
    for model_name in chosen_models:
        sub = df[df["model"] == model_name].sort_values("t", kind="mergesort")
        ax_p.plot(sub["ts"], sub["p"], linewidth=1.8, label=model_name.removesuffix("_model"))

    _shade_context(
        ax_p,
        warm_start_ts=warm_start_ts,
        warm_end_ts=warm_end_ts,
        warning_start_ts=warning_start_ts,
        warning_end_ts=warning_end_ts,
        event_ts=failure_ts,
    )

    ax_p.set_title("Per-model anomaly probability")
    ax_p.set_ylim(-0.05, 1.05)
    ax_p.grid(True, alpha=0.3)
    ax_p.legend(loc="upper left", ncol=2, frameon=True)

    # Panel 3: system score + alert episodes
    ax_sys.plot(one["ts"], one["system_score"], linewidth=2.0, label="system_score")
    for s, e in alert_eps:
        s_ts = one.loc[one["t"] == s, "ts"].iloc[0]
        e_ts = one.loc[one["t"] == e, "ts"].iloc[0]
        ax_sys.axvspan(s_ts, e_ts, alpha=0.10, color="tab:red")

    if warm_start_ts is not None and warm_end_ts is not None:
        ax_sys.axvspan(warm_start_ts, warm_end_ts, alpha=0.08, color="gray", label="warmup")
    if warning_start_ts is not None and warning_end_ts is not None:
        ax_sys.axvspan(warning_start_ts, warning_end_ts, alpha=0.12, color="magenta", label="warning window")
    if failure_ts is not None:
        ax_sys.axvline(failure_ts, color="magenta", linestyle=":", linewidth=2.5, label="failure")

    ax_sys.set_title("System score and alert episodes")
    ax_sys.set_ylim(-0.05, 1.05)
    ax_sys.grid(True, alpha=0.3)
    ax_sys.legend(loc="upper left", frameon=True)
    ax_sys.set_xlabel("time")

    title = "HTM-Monitor CMAPSS unit run"
    if subtitle:
        title = f"{title}\n{subtitle}"
    fig.suptitle(title, fontsize=14, y=0.98)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.07)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
