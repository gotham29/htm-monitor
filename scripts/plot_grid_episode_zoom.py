#scripts/plot_grid_episode_zoom.py

from __future__ import annotations

import argparse
import matplotlib.dates as mdates
from pathlib import Path

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


def _dedup_steps(run_df: pd.DataFrame) -> pd.DataFrame:
    one = run_df.drop_duplicates("t", keep="first").copy()
    one = one.sort_values("t", kind="mergesort").reset_index(drop=True)
    return one


def _add_gap_features(
    demand_df: pd.DataFrame,
    netgen_df: pd.DataFrame,
) -> pd.DataFrame:
    merged = demand_df.merge(netgen_df, on="ts", how="inner")
    merged["gap"] = merged["demand"] - merged["net_generation"]
    return merged[["ts", "demand", "net_generation", "gap"]].copy()


def _load_optional_source_series(cfg: dict, source_name: str) -> pd.DataFrame | None:
    try:
        return _load_source_series(cfg, source_name)
    except ValueError:
        return None


def _format_time_axis(ax) -> None:
    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)


def _shade_alert_episodes(ax, one_sub: pd.DataFrame) -> None:
    alert_rows = one_sub[one_sub["alert"] == 1].copy()
    if alert_rows.empty:
        return
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
            ax.axvspan(ep_start, prev_ts, alpha=0.12, color="tab:red")
            ep_start = ts
            prev_ts = ts
    if in_ep and ep_start is not None and prev_ts is not None:
        ax.axvspan(ep_start, prev_ts, alpha=0.12, color="tab:red")


def _add_recent_hour_baseline_residual(
    df: pd.DataFrame,
    value_col: str,
    *,
    history_days: int = 7,
) -> pd.DataFrame:
    """
    For each timestamp, compute:
      residual = current value - median(value at same hour-of-day over prior N days)

    This makes subtle departures from recently learned diurnal structure much easier
    to see than the raw signal alone.
    """
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


def _shade_event(ax, event_start: pd.Timestamp | None, event_end: pd.Timestamp | None) -> None:
    if event_start is None or event_end is None:
        return
    ax.axvspan(event_start, event_end, alpha=0.10, color="magenta")
    ax.axvline(event_start, linestyle="--", linewidth=1.5)
    ax.axvline(event_end, linestyle="--", linewidth=1.5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--start", required=True, help="Zoom start, e.g. '2020-08-12 00:00:00'")
    ap.add_argument("--end", required=True, help="Zoom end, e.g. '2020-08-15 23:00:00'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default=None)
    ap.add_argument("--event-start", default=None)
    ap.add_argument("--event-end", default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_csv = run_dir / "run.csv"
    cfg = _load_config(Path(args.config))

    start_ts = pd.Timestamp(args.start)
    end_ts = pd.Timestamp(args.end)
    event_start = pd.Timestamp(args.event_start) if args.event_start else None
    event_end = pd.Timestamp(args.event_end) if args.event_end else None

    run_df = pd.read_csv(run_csv)
    run_df["ts"] = pd.to_datetime(run_df["timestamp"], errors="coerce")
    run_df["raw"] = pd.to_numeric(run_df["raw"], errors="coerce")
    run_df["p"] = pd.to_numeric(run_df["p"], errors="coerce")
    run_df["system_score"] = pd.to_numeric(run_df["system_score"], errors="coerce")
    run_df["alert"] = pd.to_numeric(run_df["alert"], errors="coerce").fillna(0).astype(int)
    run_df = run_df.sort_values(["t", "model"], kind="mergesort").reset_index(drop=True)

    one = _dedup_steps(run_df)

    demand_src = _load_source_series(cfg, "demand")
    netgen_src = _load_source_series(cfg, "net_generation")
    imbalance_src = _load_optional_source_series(cfg, "imbalance")

    gap_src = _add_gap_features(demand_src, netgen_src)
    gap_src = _add_recent_hour_baseline_residual(gap_src, "gap", history_days=7)

    demand_src = _add_recent_hour_baseline_residual(demand_src, "demand", history_days=7)
    netgen_src = _add_recent_hour_baseline_residual(netgen_src, "net_generation", history_days=7)

    if imbalance_src is not None:
        imbalance_src = _add_recent_hour_baseline_residual(
            imbalance_src, "imbalance", history_days=7
        )

    demand_sub = demand_src[(demand_src["ts"] >= start_ts) & (demand_src["ts"] <= end_ts)].copy()
    netgen_sub = netgen_src[(netgen_src["ts"] >= start_ts) & (netgen_src["ts"] <= end_ts)].copy()
    gap_sub = gap_src[(gap_src["ts"] >= start_ts) & (gap_src["ts"] <= end_ts)].copy()
    imbalance_sub = (
        imbalance_src[(imbalance_src["ts"] >= start_ts) & (imbalance_src["ts"] <= end_ts)].copy()
        if imbalance_src is not None else None
    )
    run_sub = run_df[(run_df["ts"] >= start_ts) & (run_df["ts"] <= end_ts)].copy()
    one_sub = one[(one["ts"] >= start_ts) & (one["ts"] <= end_ts)].copy()

    demand_p = run_sub[run_sub["model"] == "demand_model"].copy()
    netgen_p = run_sub[run_sub["model"] == "net_generation_model"].copy()
    imbalance_p = run_sub[run_sub["model"] == "imbalance_model"].copy()

    fig = plt.figure(figsize=(15, 13))
    gs = fig.add_gridspec(4, 1, height_ratios=[2.2, 1.8, 1.3, 1.4], hspace=0.18)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
    ax3 = fig.add_subplot(gs[3, 0], sharex=ax0)

    # Panel 1: raw operational signals
    ax0.plot(
        demand_sub["ts"],
        demand_sub["demand"],
        linewidth=2.0,
        label="Demand (MW)",
    )
    ax0.plot(
        netgen_sub["ts"],
        netgen_sub["net_generation"],
        linewidth=2.0,
        label="Net generation (MW)",
    )
    ax0.plot(
        gap_sub["ts"],
        gap_sub["gap"],
        linewidth=1.8,
        alpha=0.8,
        label="Demand - generation gap (MW)",
    )
    if imbalance_sub is not None:
        ax0.plot(
            imbalance_sub["ts"],
            imbalance_sub["imbalance"],
            linewidth=1.6,
            alpha=0.9,
            label="Imbalance (MW)",
        )
    ax0.set_title("Raw grid signals")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper left", frameon=True)

    # Panel 2: strain proxy vs recent same-hour baseline
    ax1.plot(
        gap_sub["ts"],
        gap_sub["gap"],
        linewidth=2.0,
        label="Gap = demand - generation (MW)",
    )
    ax1.plot(
        gap_sub["ts"],
        gap_sub["gap_resid"],
        linewidth=2.0,
        label="Demand residual vs recent same-hour median",
    )
    if imbalance_sub is not None and "imbalance_resid" in imbalance_sub.columns:
        ax1.plot(
            imbalance_sub["ts"],
            imbalance_sub["imbalance_resid"],
            linewidth=1.8,
            label="Imbalance residual vs recent same-hour median",
        )
    ax1.axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.7)
    ax1.set_title("Grid strain proxy")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", frameon=True)

    # Panel 3: raw anomaly scores
    ax2.plot(demand_p["ts"], demand_p["raw"], linewidth=2.0, label="Demand raw anomaly")
    ax2.plot(netgen_p["ts"], netgen_p["raw"], linewidth=2.0, label="Net generation raw anomaly")
    if not imbalance_p.empty:
        ax2.plot(
            imbalance_p["ts"], imbalance_p["raw"], linewidth=2.0, label="Imbalance raw anomaly"
        )
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title("Per-model raw anomaly")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", frameon=True)

    # Panel 4: system score + alert shading
    ax3.plot(one_sub["ts"], one_sub["system_score"], linewidth=2.0, label="System score")
    _shade_alert_episodes(ax3, one_sub)
    ax3.set_ylim(-0.05, 1.05)
    ax3.set_title("System score and alert episodes")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper left", frameon=True)
    ax3.set_xlabel("Time")

    # Event shading
    if event_start is not None and event_end is not None:
        for ax in (ax0, ax1, ax2, ax3):
            _shade_event(ax, event_start, event_end)

    _format_time_axis(ax3)
    fig.suptitle(args.title or f"Grid episode zoom: {start_ts} to {end_ts}", fontsize=14, y=0.98)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.07)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
