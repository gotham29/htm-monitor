#demo/prepare_cmapss.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd


CMAPSS_COLUMNS: List[str] = [
    "unit_id",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_cmapss_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")

    # Some CMAPSS copies can carry extra all-NaN trailing columns from spacing.
    df = df.dropna(axis=1, how="all")

    if df.shape[1] != len(CMAPSS_COLUMNS):
        raise ValueError(
            f"{path} expected {len(CMAPSS_COLUMNS)} columns after cleanup, "
            f"but found {df.shape[1]}"
        )

    df.columns = CMAPSS_COLUMNS
    df["unit_id"] = df["unit_id"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df


def _read_rul(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df = df.dropna(axis=1, how="all")

    if df.shape[1] != 1:
        raise ValueError(f"{path} expected 1 column, found {df.shape[1]}")

    df.columns = ["rul"]
    df["test_unit_id"] = range(1, len(df) + 1)
    df["rul"] = df["rul"].astype(int)
    return df[["test_unit_id", "rul"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["FD001", "FD002", "FD003", "FD004"])
    args = ap.parse_args()

    dataset = str(args.dataset).upper()
    dataset_dir = f"cmapss_{dataset.lower()}"
    repo = _repo_root()
    raw_dir = repo / "data" / dataset_dir / "raw"
    out_dir = repo / "data" / dataset_dir / "prepared"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = raw_dir / f"train_{dataset}.txt"
    test_path = raw_dir / f"test_{dataset}.txt"
    rul_path = raw_dir / f"RUL_{dataset}.txt"

    if not train_path.exists():
        raise FileNotFoundError(train_path)
    if not test_path.exists():
        raise FileNotFoundError(test_path)
    if not rul_path.exists():
        raise FileNotFoundError(rul_path)

    train_df = _read_cmapss_table(train_path)
    test_df = _read_cmapss_table(test_path)
    rul_df = _read_rul(rul_path)

    train_df.to_csv(out_dir / f"train_{dataset.lower()}.csv", index=False)
    test_df.to_csv(out_dir / f"test_{dataset.lower()}.csv", index=False)
    rul_df.to_csv(out_dir / f"rul_{dataset.lower()}.csv", index=False)

    settings = ["setting_1", "setting_2", "setting_3"]
    sensors = [c for c in CMAPSS_COLUMNS if c.startswith("sensor_")]

    metadata = {
        "dataset": f"CMAPSS_{dataset}",
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_units": int(train_df["unit_id"].nunique()),
        "test_units": int(test_df["unit_id"].nunique()),
        "rul_rows": int(len(rul_df)),
        "settings": settings,
        "sensors": sensors,
        "train_cycle_range_by_unit": {
            str(int(k)): {
                "min_cycle": int(v["min"]),
                "max_cycle": int(v["max"]),
            }
            for k, v in train_df.groupby("unit_id")["cycle"].agg(["min", "max"]).to_dict("index").items()
        },
    }

    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"Prepared {dataset} files written to:\n{out_dir}")


if __name__ == "__main__":
    main()
