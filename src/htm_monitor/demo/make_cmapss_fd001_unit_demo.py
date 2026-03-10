#demo/make_cmapss_fd001_unit_demo.py

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pandas as pd


UNIT_ID = 34
START_TS = "2000-01-01 00:00:00"
FREQ = "1H"

# First-pass demo subset: strong, interpretable, and still plottable.
SELECTED_SENSORS: List[str] = [
    "sensor_9",
    "sensor_14",
    "sensor_4",
    "sensor_3",
    "sensor_17",
    "sensor_7",
    "sensor_12",
    "sensor_11",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _fmt_ts(series: pd.Series) -> List[str]:
    ts = pd.to_datetime(series)
    return [t.strftime("%Y-%m-%d %H:%M:%S") for t in ts]


def main() -> None:
    repo = _repo_root()

    src_csv = repo / "data" / "cmapss_fd001" / "demo" / "fd001_test_scored.csv"
    out_dir = repo / "data" / "cmapss_fd001" / "unit_demo_34"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not src_csv.exists():
        raise FileNotFoundError(f"Missing scored CMAPSS demo file: {src_csv}")

    df = pd.read_csv(src_csv)
    unit_df = df[df["unit_id"] == UNIT_ID].copy()

    if unit_df.empty:
        raise ValueError(f"No rows found for unit_id={UNIT_ID}")

    unit_df = unit_df.sort_values("cycle", kind="mergesort").reset_index(drop=True)

    # Synthetic timestamp so the existing HTM-Monitor pipeline can consume it directly.
    # One cycle = one hour for demo purposes.
    unit_df["timestamp"] = _fmt_ts(
        pd.date_range(start=START_TS, periods=len(unit_df), freq=FREQ)
    )

    gt_30 = unit_df.loc[unit_df["is_late_life_30"] == 1, "timestamp"].tolist()
    gt_20 = unit_df.loc[unit_df["is_late_life_20"] == 1, "timestamp"].tolist()
    gt_10 = unit_df.loc[unit_df["is_late_life_10"] == 1, "timestamp"].tolist()

    # Write one CSV per selected sensor in the repo's standard demo style:
    # timestamp,value
    for sensor in SELECTED_SENSORS:
        if sensor not in unit_df.columns:
            raise ValueError(f"Missing expected sensor column: {sensor}")

        out = unit_df[["timestamp", sensor]].copy()
        out = out.rename(columns={sensor: "value"})
        out.to_csv(out_dir / f"{sensor}.csv", index=False)

    gt_payload = {
        "late_life_30": gt_30,
        "late_life_20": gt_20,
        "late_life_10": gt_10,
    }
    (out_dir / "gt_timestamps.json").write_text(json.dumps(gt_payload, indent=2))

    manifest = {
        "dataset": "CMAPSS_FD001",
        "unit_id": UNIT_ID,
        "source_csv": str(src_csv),
        "selected_sensors": SELECTED_SENSORS,
        "n_cycles": int(len(unit_df)),
        "cycle_min": int(unit_df["cycle"].min()),
        "cycle_max": int(unit_df["cycle"].max()),
        "synthetic_time": {
            "start": START_TS,
            "freq": FREQ,
        },
        "label_definition": {
            "late_life_30": "rows where rul_at_cycle <= 30",
            "late_life_20": "rows where rul_at_cycle <= 20",
            "late_life_10": "rows where rul_at_cycle <= 10",
        },
        "gt_counts": {
            "late_life_30": int(len(gt_30)),
            "late_life_20": int(len(gt_20)),
            "late_life_10": int(len(gt_10)),
        },
        "rul_summary": {
            "max_rul_at_cycle": int(unit_df["rul_at_cycle"].max()),
            "min_rul_at_cycle": int(unit_df["rul_at_cycle"].min()),
            "failure_cycle": int(unit_df["failure_cycle"].iloc[0]),
            "max_cycle_observed": int(unit_df["max_cycle_observed"].iloc[0]),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("Built CMAPSS FD001 unit demo artifacts:")
    print(f"  {out_dir}")
    print(f"  unit_id={UNIT_ID}")
    print(f"  sensors={SELECTED_SENSORS}")
    print(f"  late_life_30 GT count={len(gt_30)}")


if __name__ == "__main__":
    main()
