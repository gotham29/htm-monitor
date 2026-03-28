#scripts/render_grid_review_plots.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import yaml


def _load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, dict):
        raise ValueError("Config must parse to a mapping")
    return cfg


def _load_source_series(cfg: dict, source_name: str) -> pd.DataFrame:
    src_cfg = None
    for s in ((cfg.get("data") or {}).get("sources") or []):
        if s.get("name") == source_name:
            src_cfg = s
            break
    if src_cfg is None:
        raise ValueError(f"Could not find source '{source_name}' in config")

    p = Path(src_cfg["path"])
    df = pd.read_csv(p)
    df["ts"] = pd.to_datetime(df[src_cfg["timestamp_col"]], errors="raise")

    fields = dict(src_cfg.get("fields") or {})
    if len(fields) != 1:
        raise ValueError(f"Expected exactly one field mapping for source '{source_name}'")
    canon_name, csv_col = next(iter(fields.items()))
    df[canon_name] = pd.to_numeric(df[csv_col], errors="coerce")

    return df[["ts", canon_name]].copy()


def _modeled_feature_order(cfg: dict) -> List[str]:
    """
    Stable display order based on model order in config.
    Assumes the common case here: one feature per model.
    """
    out: List[str] = []
    for _, mcfg in ((cfg.get("models") or {}).items()):
        feats = list((mcfg.get("features") or []))
        for f in feats:
            if f not in out:
                out.append(f)
    return out


def _load_modeled_feature_series(cfg: dict) -> Dict[str, pd.DataFrame]:
    """
    Load one timestamped series per modeled feature, keyed by canonical feature name.
    """
    out: Dict[str, pd.DataFrame] = {}
    for src in ((cfg.get("data") or {}).get("sources") or []):
        fields = dict(src.get("fields") or {})
        if len(fields) != 1:
            continue
        feat_name = next(iter(fields.keys()))
        out[feat_name] = _load_source_series(cfg, str(src.get("name")))
    return out


def _dedup_steps(run_df: pd.DataFrame) -> pd.DataFrame:
    one = run_df.drop_duplicates("t", keep="first").copy()
    one = one.sort_values("t", kind="mergesort").reset_index(drop=True)
    return one


def _add_recent_hour_baseline_residual(
    df: pd.DataFrame,
    value_col: str,
    *,
    history_days: int = 7,
) -> pd.DataFrame:
    out = df.copy().sort_values("ts", kind="mergesort").reset_index(drop=True)
    out["hour"] = out["ts"].dt.hour

    vals = []
    hist_delta = pd.Timedelta(days=int(history_days))

    for _, row in out.iterrows():
        ts = row["ts"]
        hour = int(row["hour"])
        window_start = ts - hist_delta

        hist = out[
            (out["ts"] < ts)
            & (out["ts"] >= window_start)
            & (out["hour"] == hour)
        ][value_col].dropna()

        if hist.empty:
            vals.append(float("nan"))
        else:
            vals.append(float(row[value_col]) - float(hist.median()))

    out[f"{value_col}_resid"] = vals
    return out


def _format_time_axis(ax) -> None:
    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)


def _episodes_from_alert_series(one_sub: pd.DataFrame) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    alert_rows = one_sub[one_sub["alert"] == 1].copy()
    if alert_rows.empty:
        return []

    eps: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    in_ep = False
    ep_start = None
    prev_ts = None

    for _, row in alert_rows.iterrows():
        ts = row["ts"]
        if not in_ep:
            ep_start = ts
            prev_ts = ts
            in_ep = True
        elif prev_ts is not None and (ts - prev_ts) == pd.Timedelta(hours=1):
            prev_ts = ts
        else:
            eps.append((ep_start, prev_ts))
            ep_start = ts
            prev_ts = ts

    if in_ep and ep_start is not None and prev_ts is not None:
        eps.append((ep_start, prev_ts))
    return eps


def _parse_json_dict(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict):
        return x
    if x is None:
        return {}
    if isinstance(x, float) and pd.isna(x):
        return {}
    if isinstance(x, str):
        raw = x.strip()
        if not raw:
            return {}
        v = json.loads(raw)
        if not isinstance(v, dict):
            raise ValueError("Expected JSON object")
        return v
    return {}


def _sorted_group_names(run_df: pd.DataFrame) -> List[str]:
    names = set()
    for col in ("group_hot", "window_hot_by_group"):
        if col not in run_df.columns:
            continue
        for v in run_df[col].tolist():
            d = _parse_json_dict(v)
            names.update(str(k) for k in d.keys() if str(k).strip())
    return sorted(names)


def _shade_alert_episodes(ax, one_sub: pd.DataFrame) -> None:
    for s, e in _episodes_from_alert_series(one_sub):
        ax.axvspan(s, e, alpha=0.12, color="tab:red")


def _shade_region(ax, start_ts: pd.Timestamp, end_ts: pd.Timestamp, *, color: str = "magenta") -> None:
    ax.axvspan(start_ts, end_ts, alpha=0.10, color=color)
    ax.axvline(start_ts, linestyle="--", linewidth=1.5)
    ax.axvline(end_ts, linestyle="--", linewidth=1.5)


def _feature_label(cfg: dict, feat_name: str) -> str:
    for src in ((cfg.get("data") or {}).get("sources") or []):
        fields = dict(src.get("fields") or {})
        if feat_name in fields:
            unit = str(src.get("unit") or "").strip()
            return f"{feat_name} ({unit})" if unit else feat_name
    return feat_name


def _group_hot_series(one_sub: pd.DataFrame, group_name: str) -> List[int]:
    if "group_hot" in one_sub.columns:
        return [
            int(bool(_parse_json_dict(v).get(group_name, 0)))
            for v in one_sub["group_hot"].tolist()
        ]
    if "window_hot_by_group" in one_sub.columns:
        return [
            int(bool(_parse_json_dict(v).get(group_name, 0)))
            for v in one_sub["window_hot_by_group"].tolist()
        ]
    return [0] * len(one_sub)


def _system_trace(one_sub: pd.DataFrame) -> Tuple[pd.Series, str]:
    if "system_hot_count" in one_sub.columns:
        vals = pd.to_numeric(one_sub["system_hot_count"], errors="coerce")
        return vals, "System hot count"
    if "system_score" in one_sub.columns:
        vals = pd.to_numeric(one_sub["system_score"], errors="coerce")
        return vals, "System score"
    return pd.Series([0.0] * len(one_sub)), "System signal"


def _plot_one_window(
    *,
    run_df: pd.DataFrame,
    one: pd.DataFrame,
    cfg: dict,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    out_path: Path,
    title: str,
    focal_start: Optional[pd.Timestamp] = None,
    focal_end: Optional[pd.Timestamp] = None,
) -> None:    
    group_names = _sorted_group_names(run_df)

    feature_order = _modeled_feature_order(cfg)
    feature_src = _load_modeled_feature_series(cfg)

    if not feature_src:
        raise ValueError("Could not load any modeled feature series from config")

    feature_resid: Dict[str, pd.DataFrame] = {}
    for feat_name, df in feature_src.items():
        feature_resid[feat_name] = _add_recent_hour_baseline_residual(df, feat_name, history_days=7)

    feature_sub: Dict[str, pd.DataFrame] = {}
    for feat_name, df in feature_resid.items():
        feature_sub[feat_name] = df[(df["ts"] >= start_ts) & (df["ts"] <= end_ts)].copy()

    run_sub = run_df[(run_df["ts"] >= start_ts) & (run_df["ts"] <= end_ts)].copy()
    one_sub = one[(one["ts"] >= start_ts) & (one["ts"] <= end_ts)].copy()

    per_model_raw: Dict[str, pd.DataFrame] = {}
    for feat_name in feature_order:
        model_name = f"{feat_name}_model"
        per_model_raw[feat_name] = run_sub[run_sub["model"] == model_name].copy()

    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(5, 1, height_ratios=[2.2, 1.8, 1.3, 1.0, 1.4], hspace=0.18) 

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
    axg = fig.add_subplot(gs[3, 0], sharex=ax0)
    ax3 = fig.add_subplot(gs[4, 0], sharex=ax0)

    # Panel 1: all modeled raw signals
    for feat_name in feature_order:
        sub = feature_sub.get(feat_name)
        if sub is None or sub.empty:
            continue
        ax0.plot(
            sub["ts"],
            sub[feat_name],
            linewidth=1.8,
            alpha=0.9,
            label=_feature_label(cfg, feat_name),
        )
    ax0.set_title("Raw modeled signals")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper left", frameon=True)

    # Panel 2: recent-hour residuals for all modeled signals
    ax1.axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.7)

    for feat_name in feature_order:
        sub = feature_sub.get(feat_name)
        resid_col = f"{feat_name}_resid"
        if sub is None or sub.empty or resid_col not in sub.columns:
            continue
        ax1.plot(
            sub["ts"],
            sub[resid_col],
            linewidth=1.6,
            alpha=0.9,
            label=f"{feat_name} residual",
        )

    ax1.set_title("Recent-hour residuals")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", frameon=True)

    for feat_name in feature_order:
        p = per_model_raw.get(feat_name)
        if p is None or p.empty:
            continue
        ax2.plot(p["ts"], p["raw"], linewidth=2.0, label=f"{feat_name} raw anomaly")

    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title("Per-model raw anomaly")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", frameon=True)

    # Panel 4: grouped decision activation
    if group_names:
        for g in group_names:
            axg.plot(one_sub["ts"], _group_hot_series(one_sub, g), linewidth=2.0, label=g)
        axg.set_ylim(-0.05, 1.05)
        axg.set_title("Group activation")
        axg.grid(True, alpha=0.3)
        axg.legend(loc="upper left", frameon=True)
    else:
        axg.plot([], [])
        axg.set_ylim(-0.05, 1.05)
        axg.set_title("Group activation")
        axg.grid(True, alpha=0.3)

    system_vals, system_label = _system_trace(one_sub)
    ax3.plot(one_sub["ts"], system_vals, linewidth=2.0, label=system_label)

    _shade_alert_episodes(ax3, one_sub)
    if system_label == "System score":
        ax3.set_ylim(-0.05, 1.05)
    ax3.set_title(f"{system_label} and alert episodes")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper left", frameon=True)
    ax3.set_xlabel("Time")

    if focal_start is not None and focal_end is not None:
        for ax in (ax0, ax1, ax2, axg, ax3):
            _shade_region(ax, focal_start, focal_end)

    _format_time_axis(ax3)
    fig.suptitle(title, fontsize=14, y=0.98)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.07)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--pad-steps", type=int, default=336)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default=None)
    ap.add_argument("--event-start", default=None)
    ap.add_argument("--event-end", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    analysis_dir = run_dir / "analysis"
    summary_path = analysis_dir / "run_summary.json"
    run_csv = run_dir / "run.csv"

    cfg = _load_config(Path(args.config))
    summary = json.loads(summary_path.read_text())

    run_df = pd.read_csv(run_csv)
    run_df["ts"] = pd.to_datetime(run_df["timestamp"], errors="coerce")
    run_df["raw"] = pd.to_numeric(run_df["raw"], errors="coerce")
    run_df["p"] = pd.to_numeric(run_df["p"], errors="coerce")
    if "system_score" in run_df.columns:
        run_df["system_score"] = pd.to_numeric(run_df["system_score"], errors="coerce")
    if "system_hot_count" in run_df.columns:
        run_df["system_hot_count"] = pd.to_numeric(run_df["system_hot_count"], errors="coerce")

    run_df["alert"] = pd.to_numeric(run_df["alert"], errors="coerce").fillna(0).astype(int)
    if "group_hot" in run_df.columns:
        run_df["group_hot"] = run_df["group_hot"].apply(_parse_json_dict)
    if "window_hot_by_group" in run_df.columns:
        run_df["window_hot_by_group"] = run_df["window_hot_by_group"].apply(_parse_json_dict)
    run_df = run_df.sort_values(["t", "model"], kind="mergesort").reset_index(drop=True)

    one = _dedup_steps(run_df)

    t_to_ts = dict(zip(one["t"].astype(int).tolist(), one["ts"].tolist()))
    t_min = int(one["t"].min())
    t_max = int(one["t"].max())
    pad_steps = int(args.pad_steps)

    gt_windows = (((summary.get("ground_truth") or {}).get("system") or {}).get("gt_windows") or [])
    alert_eps = (((summary.get("alerts") or {}).get("episodes") or {}).get("episodes") or [])

    gt_dir = analysis_dir / "plots" / "gt_windows"
    alert_dir = analysis_dir / "plots" / "alert_episodes"

    if args.start and args.end and args.out:
        start_ts = pd.Timestamp(args.start)
        end_ts = pd.Timestamp(args.end)
        focal_start = pd.Timestamp(args.event_start) if args.event_start else None
        focal_end = pd.Timestamp(args.event_end) if args.event_end else None
        _plot_one_window(
            run_df=run_df,
            one=one,
            cfg=cfg,
            start_ts=start_ts,
            end_ts=end_ts,
            out_path=Path(args.out),
            title=args.title or f"Review window: {start_ts} to {end_ts}",
            focal_start=focal_start,
            focal_end=focal_end,
        )
        return

    # GT windows
    for i, (gs, ge) in enumerate(gt_windows, start=1):
        gs = int(gs)
        ge = int(ge)
        plot_s = max(t_min, gs - pad_steps)
        plot_e = min(t_max, ge + pad_steps)

        start_ts = t_to_ts[plot_s]
        end_ts = t_to_ts[plot_e]
        focal_start = t_to_ts[gs]
        focal_end = t_to_ts[ge]

        _plot_one_window(
            run_df=run_df,
            one=one,
            cfg=cfg,
            start_ts=start_ts,
            end_ts=end_ts,
            out_path=gt_dir / f"gt_window_{i:02d}_t{gs}_t{ge}.png",
            title=f"GT window {i}: t={gs}..{ge}",
            focal_start=focal_start,
            focal_end=focal_end,
        )

    # Alert episodes
    for i, (es, ee) in enumerate(alert_eps, start=1):
        es = int(es)
        ee = int(ee)
        plot_s = max(t_min, es - pad_steps)
        plot_e = min(t_max, ee + pad_steps)

        start_ts = t_to_ts[plot_s]
        end_ts = t_to_ts[plot_e]
        focal_start = t_to_ts[es]
        focal_end = t_to_ts[ee]

        _plot_one_window(
            run_df=run_df,
            one=one,
            cfg=cfg,
            start_ts=start_ts,
            end_ts=end_ts,
            out_path=alert_dir / f"alert_episode_{i:02d}_t{es}_t{ee}.png",
            title=f"Alert episode {i}: t={es}..{ee}",
            focal_start=focal_start,
            focal_end=focal_end,
        )


if __name__ == "__main__":
    main()
