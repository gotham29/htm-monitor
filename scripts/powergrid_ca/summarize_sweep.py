from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def main() -> None:
    outputs = Path("outputs")
    rows = []

    for run_dir in sorted(outputs.glob("grid_aug2020_*")):
        summary_path = run_dir / "analysis" / "run_summary.json"
        if not summary_path.exists():
            continue

        data = json.loads(summary_path.read_text())
        system_eval = (
            data.get("ground_truth", {})
            .get("system", {})
            .get("eval", {})
        )
        alerts = data.get("alerts", {})
        decision = data.get("decision", {})
        per_event = system_eval.get("per_event", [])

        detected = False
        lag_steps = None
        lag_minutes = None
        hot_models = None

        if per_event:
            ev0 = per_event[0]
            detected = bool(ev0.get("detected"))
            lag_steps = ev0.get("lag_steps")
            lag_minutes = ev0.get("lag_minutes")
            hot_models = ",".join(ev0.get("hot_models", []))

        rows.append(
            {
                "run_name": run_dir.name,
                "threshold": decision.get("threshold_effective"),
                "per_model_hits": decision.get("per_model_hits"),
                "alert_timesteps": alerts.get("alert_timesteps"),
                "alert_rate": alerts.get("alert_rate"),
                "episode_count": alerts.get("episodes", {}).get("count"),
                "matched_gt": system_eval.get("matched_gt"),
                "precision": system_eval.get("precision"),
                "recall": system_eval.get("recall"),
                "f1": system_eval.get("f1"),
                "lag_steps": lag_steps,
                "lag_minutes": lag_minutes,
                "detected": detected,
                "false_positive_episodes": len(system_eval.get("false_positive_episodes", [])),
                "hot_models": hot_models,
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        print("No grid sweep summaries found.")
        return

    df = df.sort_values(
        by=[
            "detected",
            "false_positive_episodes",
            "lag_steps",
            "episode_count",
            "alert_timesteps",
        ],
        ascending=[False, True, True, True, True],
        na_position="last",
    )

    out_csv = Path("outputs/grid_aug2020_sweep_summary.csv")
    df.to_csv(out_csv, index=False)

    print(df.to_string(index=False))
    print()
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
