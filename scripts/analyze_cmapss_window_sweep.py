from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


DATASETS = ["FD001", "FD002", "FD003", "FD004"]
WINDOWS = [30, 60, 90, 120]


def load_summary(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main():

    repo_root = Path(__file__).resolve().parents[1]
    outputs = repo_root / "outputs"

    rows = []

    for d in DATASETS:
        for w in WINDOWS:

            run_name = f"cmapss_{d.lower()}_fleet_eval_window{w}"
            run_dir = outputs / run_name

            summary_path = run_dir / "fleet_summary.json"
            eligible_path = run_dir / "fleet_summary_eligible_only.json"

            summary = load_summary(summary_path)
            eligible = load_summary(eligible_path)

            if summary is None:
                continue

            cls = summary["first_alert_classification_counts"]

            rows.append({
                "dataset": d,
                "window": w,

                "units_total": summary["unit_count"],
                "units_eligible": summary["eligibility"]["eligible_unit_count"],

                "in_window_units": cls.get("in_warning_window", 0),
                "too_early_units": cls.get("too_early", 0),
                "too_late_units": cls.get("too_late", 0),
                "no_alert_units": cls.get("no_alert", 0),

                "matched_in_window_rate": summary["matched_in_window_rate"],

                "mean_lead": (
                    summary["in_window_first_alert_lead_steps_stats"]["mean"]
                    if summary["in_window_first_alert_lead_steps_stats"]
                    else None
                ),
            })

    df = pd.DataFrame(rows)

    df = df.sort_values(["dataset", "window"])

    out_csv = outputs / "cmapss_window_sweep_summary.csv"
    df.to_csv(out_csv, index=False)

    print("\nSweep summary table\n")
    print(df.to_string(index=False))

    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
