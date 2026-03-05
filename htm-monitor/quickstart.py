#!/usr/bin/env python3
"""quickstart.py

One-command demo runner for HTM-Monitor:
  1) (optional) generate synthetic demo data
  2) build a usecase config (non-interactive for synthetic; interactive wizard for real datasets)
  3) run pipeline (includes live plot if enabled in config)
  4) analyze run + render an intuitive system-eval scorecard (✅ detected / ❌ missed)

Examples:

  # Synthetic end-to-end (non-interactive config build):
  python quickstart.py --usecase demo_synth --mode synth --run-id run_001

  # Use the interactive wizard to build config for real data, then run:
  python quickstart.py --usecase nab_foo --mode wizard --run-id run_001

Notes:
- This script intentionally shells out to the existing CLI entrypoints so behavior stays identical
  to what you run manually today.
- For the synthetic path, we generate a config automatically using build_usecase_config().
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml


def sh(cmd: List[str], *, cwd: Optional[Path] = None) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _write_gif_from_frames(frames_dir: Path, out_gif: Path, *, fps: int = 20) -> Optional[Path]:
    """
    Convert frames/frame_00000.png ... into a GIF.
    Requires: imageio (added to requirements.txt).
    """
    import imageio.v2 as imageio

    if not frames_dir.exists():
        print(f"[quickstart] GIF skipped: frames dir missing: {frames_dir}")
        return None

    frames = sorted(frames_dir.glob("frame_*.png"))
    if not frames:
        print(f"[quickstart] GIF skipped: no frames found in: {frames_dir}")
        return None

    out_gif.parent.mkdir(parents=True, exist_ok=True)
    duration = 1.0 / max(1, int(fps))
    with imageio.get_writer(out_gif, mode="I", duration=duration) as w:
        for p in frames:
            w.append_data(imageio.imread(p))

    print(f"[quickstart] wrote GIF -> {out_gif}")
    return out_gif

def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def maybe_open(path: Path) -> None:
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        elif sys.platform.startswith("linux"):
            subprocess.run(["xdg-open", str(path)], check=False)
    except Exception:
        pass


@dataclass(frozen=True)
class SynthSource:
    name: str
    csv_path: Path
    ts_col: str = "timestamp"
    ts_fmt: str = "%Y-%m-%d %H:%M:%S"
    value_col: str = "value"
    feature: str = ""
    gt_timestamps_json: Optional[Path] = None


def build_synth_config(
    *,
    usecase: str,
    out_config: Path,
    sources: Sequence[SynthSource],
    defaults: Dict[str, Any],
    disable_timestamp_feature: bool = True,
) -> None:
    """Build configs/<usecase>.yaml programmatically for synthetic sources."""
    from src.htm_monitor.cli.make_usecase_config import SourceSpec, build_usecase_config

    src_specs: List[SourceSpec] = []
    for s in sources:
        feat = s.feature or s.name
        gt: Optional[List[str]] = None
        if s.gt_timestamps_json and s.gt_timestamps_json.exists():
            blob = json.loads(s.gt_timestamps_json.read_text())
            if isinstance(blob, list):
                gt = [str(x) for x in blob]
            elif isinstance(blob, dict):
                if s.name in blob and isinstance(blob[s.name], list):
                    gt = [str(x) for x in blob[s.name]]
                elif "timestamps" in blob and isinstance(blob["timestamps"], list):
                    gt = [str(x) for x in blob["timestamps"]]
                elif feat in blob and isinstance(blob[feat], list):
                    gt = [str(x) for x in blob[feat]]

        src_specs.append(
            SourceSpec(
                name=s.name,
                path=str(s.csv_path),
                timestamp_col=s.ts_col,
                timestamp_format=s.ts_fmt,
                fields={feat: s.value_col},
                gt_timestamps=gt,
            )
        )

    cfg = build_usecase_config(usecase, src_specs, **defaults)

    if disable_timestamp_feature:
        feats = cfg.get("features") or {}
        if "timestamp" in feats and isinstance(feats["timestamp"], dict):
            feats["timestamp"]["encode"] = False
        cfg["features"] = feats

        # IMPORTANT INVARIANT:
        # If timestamp encoding is disabled, it must not appear in any model's feature list.
        # Otherwise, downstream diagnostics that concatenate per-feature encodings will KeyError.
        models = cfg.get("models") or {}
        if isinstance(models, dict):
            for mname, mcfg in models.items():
                if not isinstance(mcfg, dict):
                    continue
                flist = mcfg.get("features")
                if isinstance(flist, list):
                    mcfg["features"] = [f for f in flist if f != "timestamp"]
        cfg["models"] = models

    write_text_atomic(out_config, yaml.safe_dump(cfg, sort_keys=False))
    print(f"[quickstart] wrote config -> {out_config}")


def render_system_eval_scorecard(run_dir: Path) -> Optional[Path]:
    """Render an intuitive system eval scorecard from analysis/run_summary.json."""
    import matplotlib.pyplot as plt

    summ_path = run_dir / "analysis" / "run_summary.json"
    if not summ_path.exists():
        print(f"[quickstart] no run_summary.json at {summ_path} (skipping scorecard)")
        return None

    summ = read_json(summ_path)
    sys_eval = (((summ.get("ground_truth") or {}).get("system") or {}).get("eval")) or None
    if not isinstance(sys_eval, dict):
        print("[quickstart] system eval missing in run_summary.json (skipping scorecard)")
        return None

    per_event = sys_eval.get("per_event") or []
    if not isinstance(per_event, list) or not per_event:
        print("[quickstart] system eval has no per_event rows (skipping scorecard)")
        return None

    rows: List[str] = []
    for r in per_event:
        detected = bool(r.get("detected"))
        gt_ts = r.get("gt_ts") or str(r.get("gt_t"))
        if detected:
            lag_steps = r.get("lag_steps")
            lag_min = r.get("lag_minutes")
            lag_str = f"lag={lag_steps} steps"
            if isinstance(lag_min, (int, float)):
                lag_str += f" ({lag_min:.0f} min)"
            hot = r.get("hot_models") or []
            hot_str = f"hot={','.join(hot)}" if hot else ""
            rows.append(f"[DETECTED]  {gt_ts}   {lag_str}   {hot_str}".strip())
        else:
            rows.append(f"[MISSED]    {gt_ts}   missed")

    p = sys_eval.get("precision")
    r = sys_eval.get("recall")
    f1 = sys_eval.get("f1")
    hdr = (
        f"System P/R/F1: {p:.3f}/{r:.3f}/{f1:.3f}"
        if all(isinstance(x, (int, float)) for x in [p, r, f1])
        else "System eval"
    )

    h = max(2.5, 0.45 * (len(rows) + 2))
    plt.figure(figsize=(14, h))
    plt.axis("off")
    plt.title(hdr)

    y = 0.95
    dy = 0.90 / (len(rows) + 1)
    plt.text(0.01, y, hdr, fontsize=14, transform=plt.gca().transAxes)
    y -= dy * 1.5

    for line in rows:
        # keep monospace alignment, but avoid unicode glyph issues
        plt.text(0.01, y, line, fontsize=12, family="monospace", transform=plt.gca().transAxes)
        y -= dy

    out_png = run_dir / "analysis" / "system_eval_scorecard.png"
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"[quickstart] wrote scorecard -> {out_png}")
    return out_png


def _load_manifest(run_dir: Path) -> Dict[str, Any]:
    mp = run_dir / "run.manifest.json"
    if not mp.exists():
        raise FileNotFoundError(f"Missing manifest: {mp}")
    return json.loads(mp.read_text())


def _read_run_csv(run_dir: Path):
    import pandas as pd
    rp = run_dir / "run.csv"
    if not rp.exists():
        raise FileNotFoundError(f"Missing run.csv: {rp}")
    df = pd.read_csv(rp)
    # Defensive invariant: always sort by t
    if "t" in df.columns:
        df = df.sort_values(["t", "model"]).reset_index(drop=True)
    return df


def _episode_spans_from_alert(df_run) -> List[tuple]:
    """
    Returns list[(t_start, t_end_excl)] for contiguous alert==1 runs (system-level).
    run.csv has one row per (t, model), so we reduce to max over models.
    """
    import numpy as np
    if "t" not in df_run.columns or "alert" not in df_run.columns:
        return []
    g = df_run.groupby("t")["alert"].max().reset_index()
    ts = g["t"].to_numpy()
    ys = g["alert"].fillna(0).to_numpy().astype(int)
    spans = []
    start = None
    for t, a in zip(ts, ys):
        if a and start is None:
            start = int(t)
        if (not a) and start is not None:
            spans.append((start, int(t)))
            start = None
    if start is not None and len(ts):
        spans.append((start, int(ts[-1]) + 1))
    return spans


def render_run_overview(run_dir: Path) -> Optional[Path]:
    """
    Static, demo-grade plot saved to analysis/run_overview.png.
    Uses run.csv + run.manifest.json + analysis/run_summary.json.
    Works even if live plot was skipped (--no-plot).
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    try:
        manifest = _load_manifest(run_dir)
        df_run = _read_run_csv(run_dir)
    except Exception as e:
        print(f"[quickstart] render_run_overview skipped: {e}")
        return None

    # Pull source files + GT timestamps from manifest
    sources = ((manifest.get("data") or {}).get("sources") or {})
    gt_by_source = (((manifest.get("data") or {}).get("ground_truth") or {}).get("by_source") or {})

    # Build a merged value dataframe on timestamp from the raw CSV sources.
    # This stays general: it reads whatever fields were specified per source.
    frames = []
    for sname, s in sources.items():
        path = s.get("path")
        ts_col = s.get("timestamp_col")
        fields = s.get("fields") or {}
        if not path or not ts_col or not fields:
            continue
        cols = [ts_col] + [c for c in fields.values()]
        try:
            d = pd.read_csv(path, usecols=cols)
        except Exception:
            # fallback: read full, then subset
            d = pd.read_csv(path)
            d = d[[c for c in cols if c in d.columns]]
        d = d.rename(columns={ts_col: "timestamp"})
        # rename each selected field into a stable column name: <source>:<canon>
        ren = {col: f"{sname}:{canon}" for canon, col in fields.items() if col in d.columns}
        d = d.rename(columns=ren)
        frames.append(d)

    if not frames:
        print("[quickstart] render_run_overview: no source frames (skipping)")
        return None

    df_val = frames[0]
    for d in frames[1:]:
        df_val = df_val.merge(d, on="timestamp", how="outer")

    # We use run.csv’s (t -> timestamp) mapping for consistent x-axis.
    # run.csv repeats rows per model; reduce to first timestamp per t.
    tmap = (
        df_run.groupby("t")["timestamp"]
        .first()
        .reset_index()
        .dropna(subset=["timestamp"])
    )
    df_val = df_val.merge(tmap, on="timestamp", how="inner")
    df_val = df_val.sort_values("t").reset_index(drop=True)

    # Score series per model (prefer 'score' column written by run_pipeline)
    score_col = "score" if "score" in df_run.columns else ("p" if "p" in df_run.columns else None)
    if score_col is None:
        print("[quickstart] render_run_overview: no score column found (skipping)")
        return None

    # Pivot: t x model -> score
    df_score = df_run.pivot_table(index="t", columns="model", values=score_col, aggfunc="first").reset_index()

    # System alert series (max over models)
    df_sys = df_run.groupby("t")[["system_score", "alert"]].max().reset_index()

    # Pull GT + detection markers from run_summary.json (if present)
    summ_path = run_dir / "analysis" / "run_summary.json"
    per_event = []
    sys_gt_onsets: List[int] = []

    if summ_path.exists():
        try:
            summ = json.loads(summ_path.read_text())
            sys_eval = (((summ.get("ground_truth") or {}).get("system") or {}).get("eval")) or {}
            per_event = sys_eval.get("per_event") or []
            # Pull canonical system GT onsets
            sys_gt = (((summ.get("ground_truth") or {}).get("system") or {}).get("gt_onsets")) or []
            sys_gt_onsets = [
                int(x)
                for x in sys_gt
                if isinstance(x, (int, float)) or (isinstance(x, str) and str(x).isdigit())
            ]
        except Exception:
            per_event = []

    # Build figure: value panels (one per source-field), anomaly panels (one per model), system panel
    value_cols = [c for c in df_val.columns if c not in ("timestamp", "t")]
    model_cols = [c for c in df_score.columns if c != "t"]
    n_val = len(value_cols)
    n_mod = len(model_cols)
    nrows = n_val + n_mod + 1

    if nrows <= 1:
        print("[quickstart] render_run_overview: nothing to plot (skipping)")
        return None

    fig = plt.figure(figsize=(16, max(6, 1.6 * nrows)))
    gs = fig.add_gridspec(nrows, 1, hspace=0.18)

    axes = []
    first_ax = None
    row_i = 0

    # --- values ---
    for c in value_cols:
        ax = fig.add_subplot(gs[row_i, 0], sharex=first_ax)
        if first_ax is None:
            first_ax = ax
        ax.plot(df_val["t"], df_val[c])
        ax.set_ylabel(c, rotation=0, labelpad=40, va="center")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelbottom=False)
        axes.append(ax)
        row_i += 1

    # --- per-model score ---
    for m in model_cols:
        ax = fig.add_subplot(gs[row_i, 0], sharex=first_ax)
        ax.plot(df_score["t"], df_score[m], label=score_col)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel(m.removesuffix("_model"), rotation=0, labelpad=40, va="center")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelbottom=False)
        axes.append(ax)
        row_i += 1

    # --- system alert ---
    axs = fig.add_subplot(gs[row_i, 0], sharex=first_ax)
    axs.step(df_sys["t"], df_sys["alert"].fillna(0), where="post", label="system alert")
    axs.fill_between(df_sys["t"], 0.0, df_sys["alert"].fillna(0), step="post", alpha=0.25)
    axs.set_ylim(-0.05, 1.05)
    axs.set_title("SYSTEM ALERT", pad=10)
    axs.grid(True, alpha=0.3)
    axs.set_xlabel("t")
    axes.append(axs)

    # --- overlays: alert episode spans ---
    spans = _episode_spans_from_alert(df_run)
    for (t0, t1) in spans:
        for ax in axes:
            ax.axvspan(t0, t1, alpha=0.08, linewidth=0)

    # --- overlays: GT + detection markers ---
    # GT: use manifest ground_truth.by_source timestamps, convert to t using tmap
    ts_to_t = dict(zip(tmap["timestamp"].tolist(), tmap["t"].tolist()))
    gt_ts_all = set()
    for _, ts_list in (gt_by_source or {}).items():
        if isinstance(ts_list, list):
            gt_ts_all.update([str(x) for x in ts_list])

    gt_t_all = sorted([ts_to_t[ts] for ts in gt_ts_all if ts in ts_to_t])
    for t in gt_t_all:
        for ax in axes:
            # draw per-source GT only on value/anomaly panels
            if ax is axs:
                continue
            ax.axvline(t, color="purple", linewidth=2.0, alpha=0.6, linestyle=":")

    # detection markers (from per_event)
    det_t = []
    for r in per_event:
        dt = r.get("detection_t")
        if isinstance(dt, int):
            det_t.append(dt)

    # System GT markers (magenta) — SYSTEM ALERT axis only
    for t in sorted(set(sys_gt_onsets)):
        axs.axvline(
            t,
            color="magenta",
            linewidth=2.5,
            alpha=0.95,
            linestyle=":"
        )

    for t in sorted(set(det_t)):
        for ax in axes:
            ax.axvline(t, color="green", linewidth=2.0, alpha=0.35, linestyle="--")

    fig.suptitle(f"HTM-Monitor overview — {run_dir.name}", y=0.995)

    out_png = run_dir / "analysis" / "run_overview.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"[quickstart] wrote overview -> {out_png}")
    return out_png


def _resolve_data_dir(repo: Path, demo_dir_arg: str, usecase: str) -> Path:
    """
    Accept either:
      --demo-dir data            -> data/<usecase>
      --demo-dir data/demo_synth -> data/demo_synth
    """
    p = (repo / demo_dir_arg).resolve()
    if p.name == usecase:
        return p
    return (p / usecase).resolve()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--usecase", required=True)
    ap.add_argument("--run-id", default="run_001")
    ap.add_argument("--mode", choices=["synth", "wizard"], default="synth")
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--defaults", default="configs/htm_defaults.yaml")
    ap.add_argument("--config-dir", default="configs")
    ap.add_argument("--demo-dir", default="data")
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--make-gif", action="store_true", help="Record live plot frames and build assets/live_demo.gif")
    ap.add_argument("--gif-fps", type=int, default=24, help="FPS for assets/live_demo.gif")
    ap.add_argument("--docs-dir", default="assets", help="Where README assets are written (gif + images)")

    # Synthetic generator knobs (fast repeat + low noise)
    ap.add_argument("--hours", type=int, default=1152)
    ap.add_argument("--freq", default="30min")
    ap.add_argument("--baseline-period-hours", type=float, default=6.0)
    ap.add_argument("--baseline-noise-frac", type=float, default=0.03)
    ap.add_argument("--baseline-noise-phi", type=float, default=0.96)
    ap.add_argument("--inject-jitter-sigma", type=float, default=2.5)
    ap.add_argument("--inject-jitter-phi", type=float, default=0.6)
    ap.add_argument("--settle-hours", type=float, default=576)
    ap.add_argument("--min-cycles-before-anomaly", type=int, default=24)
    ap.add_argument("--ramp-min", type=int, default=60)

    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    config_dir = (repo / args.config_dir).resolve()
    demo_dir = _resolve_data_dir(repo, args.demo_dir, args.usecase)
    defaults_path = (repo / args.defaults).resolve()

    usecase = args.usecase
    run_id = args.run_id

    out_dir = (repo / "outputs" / usecase / run_id).resolve()
    config_path = (config_dir / f"{usecase}.yaml").resolve()
    build_spec_path = (config_dir / f"{usecase}.build.yaml").resolve()
    docs_dir = (repo / args.docs_dir).resolve()
    docs_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "synth":
        # 1) Generate demo data (faster repeat + less noise)
        sh(
            [
                sys.executable, "-m", "src.demo.make_data",
                "--out-dir", str(demo_dir),
                "--baseline-drift", "0.0",
                "--baseline-noise-frac", str(args.baseline_noise_frac),
                "--baseline-noise-phi", str(args.baseline_noise_phi),
                "--inject-jitter-sigma", str(args.inject_jitter_sigma),
                "--inject-jitter-phi", str(args.inject_jitter_phi),
                "--hours", str(args.hours),
                "--freq", str(args.freq),
                "--baseline-period-hours", str(args.baseline_period_hours),
                "--settle-hours", str(args.settle_hours),
                "--min-cycles-before-anomaly", str(args.min_cycles_before_anomaly),
                "--ramp-min", str(args.ramp_min),
                "--envelope", "minmax",
                "--envelope-pad", "0.0",
                # Stronger distribution-preserving anomalies (stay inside baseline envelope):
                # For 30min sampling, 240min => 8 steps shift which *breaks HTM predictions*.
                "--inject-timewarp-max-shift-min", "240",
                "--inject-timewarp-phi", "0.90",
                "--winA", "648,672,A solo",
                "--winB", "684,708,B solo",
                "--winC", "720,744,C solo",
                "--winA", "756,780,A+B overlap (A)",
                "--winB", "756,780,A+B overlap (B)",
                "--winA", "892,916,A+C overlap (A)",
                "--winC", "892,916,A+C overlap (C)",
                "--winB", "1028,1052,B+C overlap (B)",
                "--winC", "1028,1052,B+C overlap (C)",
            ],
            cwd=repo,
        )

        gt_json = demo_dir / "gt_timestamps.json"
        synth_sources = [
            SynthSource(name="sA", csv_path=demo_dir / "sA.csv", feature="sA", gt_timestamps_json=gt_json),
            SynthSource(name="sB", csv_path=demo_dir / "sB.csv", feature="sB", gt_timestamps_json=gt_json),
            SynthSource(name="sC", csv_path=demo_dir / "sC.csv", feature="sC", gt_timestamps_json=gt_json),
        ]

        # Hard defaults for synth (kept explicit so you can reason about them)
        build_defaults: Dict[str, Any] = dict(
            rdse_size=2048,
            rdse_active_bits=40,
            rdse_num_buckets=80,
            seed_base=42,
            low_q=0.01,
            high_q=0.99,
            margin=0.03,
            model_layout="separate",
            decision_score_key="anomaly_probability",
            decision_threshold=0.99,
            decision_method="kofn_window",
            decision_k=2,
            decision_window_size=24,
            decision_per_model_hits=2,
            run_warmup_steps=500,
            run_learn_after_warmup=True,
            plot_enable=(not args.no_plot),
            plot_window=300,
            plot_show_ground_truth=True,
            plot_step_pause_s=0.01,
            plot_show_warmup_span=True,
        )

        # Write repo-relative paths into the generated config/build spec
        # (customer-ready: no absolute /Users/... paths).
        synth_sources = [
            SynthSource(
                name=s.name,
                csv_path=s.csv_path.relative_to(repo) if s.csv_path.is_absolute() else s.csv_path,
                feature=s.feature,
                gt_timestamps_json=(s.gt_timestamps_json.relative_to(repo) if (s.gt_timestamps_json and s.gt_timestamps_json.is_absolute()) else s.gt_timestamps_json),
            )
            for s in synth_sources
        ]

        build_synth_config(
            usecase=usecase,
            out_config=config_path,
            sources=synth_sources,
            defaults=build_defaults,
            disable_timestamp_feature=True,
        )

        # Optional: write a small build spec stub (for reproducibility / future enhancements)
        spec_blob = {
            "usecase": usecase,
            "sources": [
                dict(
                    name=s.name,
                    path=str(s.csv_path),
                    timestamp_col=s.ts_col,
                    timestamp_format=s.ts_fmt,
                    fields={ (s.feature or s.name): s.value_col },
                )
                for s in synth_sources
            ],
            "params": build_defaults,
            "calibration": {"method": "quantile", "low_q": 0.01, "high_q": 0.99, "margin": 0.03},
            "overrides": {"features": {}},
        }
        write_text_atomic(build_spec_path, yaml.safe_dump(spec_blob, sort_keys=False))
        print(f"[quickstart] wrote build spec -> {build_spec_path}")

    else:
        # wizard mode: build config + build spec interactively
        sh(
            [
                sys.executable, "-m", "src.htm_monitor.cli.usecase_wizard",
                "--out-dir", str(config_dir),
                "--spec-out", str(build_spec_path),
            ],
            cwd=repo,
        )

        if not config_path.exists():
            print(
                f"[quickstart] WARNING: expected config at {config_path} but it does not exist. "
                "Did you use a different usecase name inside the wizard?"
            )

    # 2) Run pipeline (canonical run-dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)

    # If making a GIF, we record frames into run_dir/analysis/frames via config override.
    # We do this by writing a small config patch file next to the config and passing it
    # through the normal config merge behavior (your build_from_config already merges defaults+config).
    #
    # Minimal approach: we just edit the generated synth config in-place if present.
    # For real-data runs, users can add plot.record to their YAML manually.
    if args.make_gif and (not args.no_plot) and config_path.exists():
        cfg = yaml.safe_load(config_path.read_text()) or {}
        cfg_plot = cfg.get("plot") or {}
        if not isinstance(cfg_plot, dict):
            cfg_plot = {}
        cfg_plot.setdefault("enable", True)
        cfg_plot.setdefault("window", 300)
        cfg_plot.setdefault("show_ground_truth", True)
        cfg_plot["record"] = {
            "enable": True,
            "dir": str(out_dir / "analysis" / "frames"),
            "every": 1,
            "dpi": 140,
        }
        cfg["plot"] = cfg_plot
        write_text_atomic(config_path, yaml.safe_dump(cfg, sort_keys=False))

    sh(
        [
            sys.executable, "-m", "src.htm_monitor.cli.run_pipeline",
            "--defaults", str(defaults_path),
            "--config", str(config_path),
            "--run-dir", str(out_dir),
        ],
        cwd=repo,
    )

    # 3) Analyze run
    sh(
        [
            sys.executable, "-m", "src.htm_monitor.cli.analyze_run",
            "--run-dir", str(out_dir),
            "--config", str(config_path),
            "--prefer-hot-by-model",
        ],
        cwd=repo,
    )

    # 4) Scorecard
    png = render_system_eval_scorecard(out_dir)
    if png and (not args.no_open):
        maybe_open(png)

    # 5) Static overview plot (always useful, even when --no-plot)
    ov = render_run_overview(out_dir)
    if ov and (not args.no_open):
        maybe_open(ov)

    # 6) README assets: GIF + images into docs/
    if args.make_gif and (not args.no_plot):
        frames_dir = out_dir / "analysis" / "frames"
        out_gif = docs_dir / "live_demo.gif"
        _write_gif_from_frames(frames_dir, out_gif, fps=int(args.gif_fps))

    # Copy scorecard + overview to assets/ for stable README paths
    _copy_if_exists(out_dir / "analysis" / "system_eval_scorecard.png", docs_dir / "system_eval_scorecard.png")
    _copy_if_exists(out_dir / "analysis" / "run_overview.png", docs_dir / "run_overview.png")

    print("\n[quickstart] done")
    print(f"  run dir: {out_dir}")
    print(f"  config : {config_path}")
    print(f"  summary: {out_dir / 'analysis' / 'run_summary.md'}")
    if png:
        print(f"  score  : {png}")
    if ov:
        print(f"  overview: {ov}")
    if args.make_gif and (not args.no_plot):
        print(f"  gif    : {docs_dir / 'live_demo.gif'}")


if __name__ == "__main__":
    main()
