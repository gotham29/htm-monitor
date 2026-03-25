#scripts/run_grid_window_sweep.py

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from textwrap import dedent


SIGNAL_SETS = [
    ["demand", "net_generation"],
    ["demand", "net_generation", "forecast_error"],
]

THRESHOLDS = [0.99, 0.995]
PER_MODEL_HITS = [2, 3]
K = 2
WINDOW_SIZE = 12
WARMUP_STEPS = 24 * 14


def _yaml_list(xs: list[str]) -> str:
    return "[" + ", ".join(xs) + "]"


def _write_config(
    *,
    repo_root: Path,
    config_path: Path,
    stats_path: Path,
    dataset_dir: Path,
    signals: list[str],
    threshold: float,
    per_model_hits: int,
) -> None:
    stats = json.loads(stats_path.read_text())
    event_ts = stats.get("event_timestamps", {}).get("aug2020", [])
    if not event_ts:
        raise ValueError("feature_stats.json missing event_timestamps.aug2020")

    seed_base = {
        "demand": 101,
        "forecast_error": 102,
        "net_generation": 103,
        "total_interchange": 104,
        "margin_proxy": 105,
    }

    feature_blocks: list[str] = []
    model_blocks: list[str] = []
    source_blocks: list[str] = []

    for sig in signals:
        bounds = stats[sig]
        feature_blocks.append(
            "\n".join(
                [
                    f"  {sig}:",
                    "    type: float",
                    "    size: 2048",
                    "    activeBits: 40",
                    "    numBuckets: 80",
                    f"    minVal: {bounds['minVal']}",
                    f"    maxVal: {bounds['maxVal']}",
                    f"    seed: {seed_base[sig]}",
                ]
            )
        )

        model_blocks.append(
            "\n".join(
                [
                    f"  {sig}_model:",
                    f"    sources: [{sig}]",
                    f"    features: [{sig}]",
                ]
            )
        )

        label_lines = "\n".join([f"        - '{t}'" for t in event_ts])
        source_blocks.append(
            "\n".join(
                [
                    f"    - name: {sig}",
                    "      kind: csv",
                    f"      path: {dataset_dir.as_posix()}/{sig}.csv",
                    "      timestamp_col: timestamp",
                    "      timestamp_format: '%Y-%m-%d %H:%M:%S'",
                    "      fields:",
                    f"        {sig}: value",
                    "      labels:",
                    "        timestamps:",
                    label_lines,
                ]
            )
        )

    yaml_text = "\n".join(
        [
            "features:",
            "  timestamp:",
            "    type: datetime",
            "    format: '%Y-%m-%d %H:%M:%S'",
            "    timeOfDay: 21",
            "    weekend: 0",
            "    dayOfWeek: 0",
            "    holiday: 0",
            "    season: 0",
            "    encode: false",
            *feature_blocks,
            "",
            "models:",
            *model_blocks,
            "",
            "data:",
            "  timebase:",
            "    mode: union",
            "    on_missing: hold_last",
            "  sources:",
            *source_blocks,
            "",
            "run:",
            f"  warmup_steps: {WARMUP_STEPS}",
            "  learn_after_warmup: true",
            "",
            "decision:",
            "  score_key: anomaly_probability",
            f"  threshold: {threshold}",
            "  method: kofn_window",
            f"  k: {K}",
            "  window:",
            f"    size: {WINDOW_SIZE}",
            f"    per_model_hits: {per_model_hits}",
            "",
            "plot:",
            "  enable: false",
            "  step_pause_s: 0.0",
            "  window: 500",
            "  show_ground_truth: true",
            "  show_warmup_span: true",
            "",
            "ground_truth:",
            "  system:",
            "    method: kofn_window",
            "    k: 2",
            "    window:",
            "      size: 12",
            "      per_source_hits: 1",
            "",
        ]
    )

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml_text)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    dataset_dir = repo_root / "data" / "grid" / "caiso_aug2020_focus"
    stats_path = dataset_dir / "feature_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing stats file: {stats_path}")

    failures: list[tuple[str, int]] = []

    for signals in SIGNAL_SETS:
        sig_tag = "_".join(signals)
        for threshold in THRESHOLDS:
            thr_tag = str(threshold).replace(".", "p")
            for per_hits in PER_MODEL_HITS:
                name = f"grid_aug2020_{sig_tag}_thr{thr_tag}_hits{per_hits}"
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
                    repo_root=repo_root,
                    config_path=cfg_path,
                    stats_path=stats_path,
                    dataset_dir=Path("data/grid/caiso_aug2020_focus"),
                    signals=signals,
                    threshold=threshold,
                    per_model_hits=per_hits,
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
        print("All grid sweeps completed successfully.")


if __name__ == "__main__":
    main()
