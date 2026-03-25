#scripts/run_cmapss_window_sweep.py

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


DATASETS = ["FD001", "FD002", "FD003", "FD004"]
WINDOWS = [30, 60, 90, 120]


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    failures: list[tuple[str, int, int]] = []

    for dataset in DATASETS:
        for warning_start in WINDOWS:
            out_name = f"cmapss_{dataset.lower()}_fleet_eval_window{warning_start}"
            out_dir = repo_root / "outputs" / out_name
            done_marker = out_dir / "fleet_summary.csv"

            if done_marker.exists():
                print("=" * 68)
                print(f"SKIP dataset={dataset} warning_start={warning_start}")
                print(f"Already completed: {done_marker}")
                print("=" * 68)
                continue

            cmd = [
                sys.executable,
                "-m",
                "htm_monitor.demo.eval_cmapss_fleet",
                "--dataset",
                dataset,
                "--output-name",
                out_name,
                "--warning-start",
                str(warning_start),
                "--skip-figures",
            ]

            print("=" * 68)
            print(f"Running dataset={dataset} warning_start={warning_start}")
            print(f"Output name: {out_name}")
            print("Command:")
            print(" ".join(cmd))
            print("=" * 68)

            result = subprocess.run(cmd, cwd=repo_root)
            if result.returncode != 0:
                print()
                print(f"FAILED on dataset={dataset}, warning_start={warning_start}")
                print(f"Exit code: {result.returncode}")
                failures.append((dataset, warning_start, result.returncode))

    print()
    if failures:
        print("Completed with failures:")
        for dataset, warning_start, code in failures:
            print(f"  dataset={dataset}, warning_start={warning_start}, exit_code={code}")
    else:
        print("All sweeps completed successfully.")


if __name__ == "__main__":
    main()
