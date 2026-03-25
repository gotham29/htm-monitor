#scripts/run_grid_encoder_sweep.py

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


PAD_FRACS = [0.05, 0.10, 0.15]
NUM_BUCKETS = [40, 60, 80]

THRESHOLD = 0.99
PER_MODEL_HITS = 3
K = 2
WINDOW_SIZE = 12
WARMUP_STEPS = 24 * 14

CAL_START = "2020-01-01 00:00:00"
CAL_END = "2020-07-31 23:00:00"

DATASET_DIR = Path("data/grid/caiso_aug2020_focus")
WIDE_CSV = DATASET_DIR / "ciso_wide.csv"


def _safe_bounds(series: pd.Series, pad_frac: float) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        raise ValueError("Cannot compute bounds on empty series")

    vmin = float(s.min())
    vmax = float(s.max())
    span = vmax - vmin
    if span <= 0:
        pad = max(1.0, abs(vmin) * pad_frac, 1.0)
    else:
        pad = span * pad_frac
    return {
        "observed_min": vmin,
        "observed_max": vmax,
        "minVal": vmin - pad,
        "maxVal": vmax + pad,
    }


def _compute_bounds(pad_frac: float) -> dict:
    df = pd.read_csv(WIDE_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    cal = df[(df["timestamp"] >= pd.Timestamp(CAL_START)) & (df["timestamp"] <= pd.Timestamp(CAL_END))].copy()
    if cal.empty:
        raise ValueError("Calibration slice is empty")

    return {
        "demand": _safe_bounds(cal["demand"], pad_frac),
        "net_generation": _safe_bounds(cal["net_generation"], pad_frac),
    }


def _write_config(*, config_path: Path, bounds: dict, num_buckets: int) -> None:
    label_src = DATASET_DIR / "feature_stats.json"
    stats = json.loads(label_src.read_text())
    event_ts = stats.get("event_timestamps", {}).get("aug2020", [])
    if not event_ts:
        raise ValueError("feature_stats.json missing event_timestamps.aug2020")

    label_lines = "\n".join([f"          - '{t}'" for t in event_ts])

    text = f"""features:
  timestamp:
    type: datetime
    format: '%Y-%m-%d %H:%M:%S'
    timeOfDay: 21
    weekend: 0
    dayOfWeek: 0
    holiday: 0
    season: 0
    encode: false

  demand:
    type: float
    size: 2048
    activeBits: 40
    numBuckets: {num_buckets}
    minVal: {bounds['demand']['minVal']}
    maxVal: {bounds['demand']['maxVal']}
    seed: 101

  net_generation:
    type: float
    size: 2048
    activeBits: 40
    numBuckets: {num_buckets}
    minVal: {bounds['net_generation']['minVal']}
    maxVal: {bounds['net_generation']['maxVal']}
    seed: 103

models:
  demand_model:
    sources: [demand]
    features: [demand]

  net_generation_model:
    sources: [net_generation]
    features: [net_generation]

data:
  timebase:
    mode: union
    on_missing: hold_last
  sources:
    - name: demand
      kind: csv
      path: data/grid/caiso_aug2020_focus/demand.csv
      timestamp_col: timestamp
      timestamp_format: '%Y-%m-%d %H:%M:%S'
      fields:
        demand: value
      labels:
        timestamps:
{label_lines}

    - name: net_generation
      kind: csv
      path: data/grid/caiso_aug2020_focus/net_generation.csv
      timestamp_col: timestamp
      timestamp_format: '%Y-%m-%d %H:%M:%S'
      fields:
        net_generation: value
      labels:
        timestamps:
{label_lines}

run:
  warmup_steps: {WARMUP_STEPS}
  learn_after_warmup: true

decision:
  score_key: anomaly_probability
  threshold: {THRESHOLD}
  method: kofn_window
  k: {K}
  window:
    size: {WINDOW_SIZE}
    per_model_hits: {PER_MODEL_HITS}

plot:
  enable: false
  step_pause_s: 0.0
  window: 500
  show_ground_truth: true
  show_warmup_span: true

ground_truth:
  system:
    method: kofn_window
    k: 2
    window:
      size: 12
      per_source_hits: 1
"""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(text)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    failures: list[tuple[str, int]] = []

    for pad_frac in PAD_FRACS:
        bounds = _compute_bounds(pad_frac)
        pad_tag = str(pad_frac).replace(".", "p")

        for num_buckets in NUM_BUCKETS:
            name = f"grid_aug2020_encpad{pad_tag}_buckets{num_buckets}"
            out_dir = repo_root / "outputs" / name
            summary = out_dir / "analysis" / "run_summary.json"
            cfg_path = repo_root / "configs" / "generated" / f"{name}.yaml"

            if summary.exists():
                print("=" * 68)
                print(f"SKIP {name}")
                print(f"Already completed: {summary}")
                print("=" * 68)
                continue

            _write_config(
                config_path=cfg_path,
                bounds=bounds,
                num_buckets=num_buckets,
            )

            cmd = [
                sys.executable,
                "-m",
                "htm_monitor.cli.run_pipeline",
                "--defaults",
                "configs/htm_defaults.yaml",
                "--config",
                str(cfg_path.relative_to(repo_root)),
                "--run-dir",
                str(out_dir.relative_to(repo_root)),
                "--no-plot",
                "--no-analyze",
            ]

            print("=" * 68)
            print(f"Running {name}")
            print("Command:")
            print(" ".join(cmd))
            print("=" * 68)

            result = subprocess.run(cmd, cwd=repo_root)
            if result.returncode != 0:
                failures.append((name, result.returncode))

    print()
    if failures:
        print("Completed with failures:")
        for name, code in failures:
            print(f"  {name}: exit_code={code}")
    else:
        print("All encoder sweeps completed successfully.")


if __name__ == "__main__":
    main()
