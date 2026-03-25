from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _hot_models_from_runcsv(run_csv: Path, episode_start_t: int) -> str:
    df = pd.read_csv(run_csv)
    df = df[df["t"] == episode_start_t].copy()
    if df.empty or "hot_by_model" not in df.columns:
        return ""
    raw = df["hot_by_model"].dropna().iloc[0] if not df["hot_by_model"].dropna().empty else None
    if raw is None:
        return ""
    try:
        obj = json.loads(raw)
    except Exception:
        return ""
    hot = [k for k, v in obj.items() if int(v) == 1]
    return ",".join(sorted(hot))


def main() -> None:
    outputs = Path("outputs")
    rows = []

    for run_dir in sorted(outputs.glob("grid_aug2020_*")):
        summary_path = run_dir / "analysis" / "run_summary.json"
        run_csv = run_dir / "run.csv"
        if not summary_path.exists() or not run_csv.exists():
            continue

        summary = json.loads(summary_path.read_text())
        system_eval = (((summary.get("ground_truth") or {}).get("system") or {}).get("eval") or {})
        fps = system_eval.get("false_positive_episodes") or []
        time_range = summary.get("time_range") or {}
        warmup_steps = int(summary.get("warmup_steps") or 0)

        df = pd.read_csv(run_csv)
        one = df.drop_duplicates("t", keep="first").sort_values("t").reset_index(drop=True)
        one["ts"] = pd.to_datetime(one["timestamp"], errors="coerce")

        gt_t = None
        per_event = system_eval.get("per_event") or []
        if per_event:
            gt_t = per_event[0].get("gt_t")

        for ep in fps:
            s, e = int(ep[0]), int(ep[1])
            start_row = one.loc[one["t"] == s]
            end_row = one.loc[one["t"] == e]
            start_ts = start_row["ts"].iloc[0] if not start_row.empty else pd.NaT
            end_ts = end_row["ts"].iloc[0] if not end_row.empty else pd.NaT

            rows.append(
                {
                    "run_name": run_dir.name,
                    "start_t": s,
                    "end_t": e,
                    "duration_steps": e - s + 1,
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                    "hours_after_warmup": s - warmup_steps,
                    "steps_before_gt": (gt_t - s) if gt_t is not None else None,
                    "hot_models_at_start": _hot_models_from_runcsv(run_csv, s),
                }
            )

    out = pd.DataFrame(rows).sort_values(["start_ts", "run_name"]).reset_index(drop=True)
    out_csv = Path("outputs/grid_aug2020_fp_episodes.csv")
    out.to_csv(out_csv, index=False)

    print(out.to_string(index=False))
    print()
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
