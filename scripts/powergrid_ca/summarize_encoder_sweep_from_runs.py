from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


GT_START = pd.Timestamp("2020-08-14 00:00:00")
GT_END = pd.Timestamp("2020-08-15 23:00:00")


def _episodes(one: pd.DataFrame) -> list[tuple[int, int]]:
    rows = one.sort_values("t", kind="mergesort")
    eps: list[tuple[int, int]] = []
    in_ep = False
    start = None
    prev_t = None

    for _, row in rows.iterrows():
        t = int(row["t"])
        alert = int(row["alert"])
        if alert == 1:
            if not in_ep:
                start = t
                prev_t = t
                in_ep = True
            elif prev_t is not None and t == prev_t + 1:
                prev_t = t
            else:
                eps.append((int(start), int(prev_t)))
                start = t
                prev_t = t
        else:
            if in_ep:
                eps.append((int(start), int(prev_t)))
                in_ep = False
                start = None
                prev_t = None

    if in_ep and start is not None and prev_t is not None:
        eps.append((int(start), int(prev_t)))

    return eps


def _hot_models_at_t(df: pd.DataFrame, t0: int) -> str:
    sub = df[df["t"] == t0]
    if sub.empty:
        return ""
    raw = sub["hot_by_model"].dropna()
    if raw.empty:
        return ""
    try:
        obj = json.loads(raw.iloc[0])
    except Exception:
        return ""
    return ",".join(sorted([k for k, v in obj.items() if int(v) == 1]))


def main() -> None:
    rows = []

    for run_dir in sorted(Path("outputs").glob("grid_aug2020_encpad*_buckets*")):
        run_csv = run_dir / "run.csv"
        if not run_csv.exists():
            continue

        df = pd.read_csv(run_csv)
        df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["alert"] = pd.to_numeric(df["alert"], errors="coerce").fillna(0).astype(int)
        df["system_score"] = pd.to_numeric(df["system_score"], errors="coerce")

        one = df.drop_duplicates("t", keep="first").sort_values("t", kind="mergesort").reset_index(drop=True)
        eps = _episodes(one)

        # first episode that overlaps GT window
        detected = False
        lag_hours = None
        hot_models = None

        for s, e in eps:
            ep = one[(one["t"] >= s) & (one["t"] <= e)]
            if ep.empty:
                continue
            ep_start_ts = ep["ts"].min()
            ep_end_ts = ep["ts"].max()
            overlaps = not (ep_end_ts < GT_START or ep_start_ts > GT_END)
            if overlaps:
                detected = True
                lag_hours = (ep_start_ts - GT_START).total_seconds() / 3600.0
                hot_models = _hot_models_at_t(df, s)
                break

        # false positives = episodes fully before GT_START or fully after GT_END
        fp_eps = []
        for s, e in eps:
            ep = one[(one["t"] >= s) & (one["t"] <= e)]
            if ep.empty:
                continue
            ep_start_ts = ep["ts"].min()
            ep_end_ts = ep["ts"].max()
            if ep_end_ts < GT_START or ep_start_ts > GT_END:
                fp_eps.append((s, e))

        rows.append(
            {
                "run_name": run_dir.name,
                "alert_timesteps": int(one["alert"].sum()),
                "episode_count": len(eps),
                "false_positive_episodes": len(fp_eps),
                "detected_gt_window": detected,
                "lag_hours_from_gt_start": lag_hours,
                "hot_models_at_first_detection": hot_models,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        print("No encoder sweep runs found.")
        return

    out = out.sort_values(
        by=[
            "detected_gt_window",
            "false_positive_episodes",
            "episode_count",
            "alert_timesteps",
            "lag_hours_from_gt_start",
        ],
        ascending=[False, True, True, True, True],
        na_position="last",
    )

    out_csv = Path("outputs/grid_aug2020_encoder_sweep_summary.csv")
    out.to_csv(out_csv, index=False)
    print(out.to_string(index=False))
    print()
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
