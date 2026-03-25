#demo/build_cmapss_demo.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


LATE_LIFE_THRESHOLDS = [30, 20, 10]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _required_columns_present(df: pd.DataFrame, required: List[str], *, ctx: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{ctx} missing required columns: {missing}")


def _load_prepared(prepared_dir: Path, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    ds = dataset.lower()
    test_path = prepared_dir / f"test_{ds}.csv"
    rul_path = prepared_dir / f"rul_{ds}.csv"
    meta_path = prepared_dir / "metadata.json"

    if not test_path.exists():
        raise FileNotFoundError(f"Missing prepared test file: {test_path}")
    if not rul_path.exists():
        raise FileNotFoundError(f"Missing prepared RUL file: {rul_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing prepared metadata file: {meta_path}")

    test_df = pd.read_csv(test_path)
    rul_df = pd.read_csv(rul_path)
    meta = json.loads(meta_path.read_text())

    _required_columns_present(test_df, ["unit_id", "cycle"], ctx=test_path.name)
    _required_columns_present(rul_df, ["test_unit_id", "rul"], ctx=rul_path.name)

    return test_df, rul_df, meta


def _build_engine_summary(test_df: pd.DataFrame, rul_df: pd.DataFrame) -> pd.DataFrame:
    max_cycle = (
        test_df.groupby("unit_id", as_index=False)["cycle"]
        .max()
        .rename(columns={"cycle": "max_cycle_observed"})
    )

    engine_df = max_cycle.merge(
        rul_df.rename(columns={"test_unit_id": "unit_id"}),
        on="unit_id",
        how="left",
        validate="one_to_one",
    )

    if engine_df["rul"].isna().any():
        bad_units = engine_df.loc[engine_df["rul"].isna(), "unit_id"].tolist()
        raise ValueError(f"Missing RUL values for test units: {bad_units[:10]}")

    engine_df["rul"] = engine_df["rul"].astype(int)
    engine_df["max_cycle_observed"] = engine_df["max_cycle_observed"].astype(int)
    engine_df["failure_cycle"] = engine_df["max_cycle_observed"] + engine_df["rul"]

    return engine_df.sort_values("unit_id", kind="mergesort").reset_index(drop=True)


def _attach_rul_targets(test_df: pd.DataFrame, engine_df: pd.DataFrame) -> pd.DataFrame:
    out = test_df.merge(
        engine_df[["unit_id", "max_cycle_observed", "rul", "failure_cycle"]],
        on="unit_id",
        how="left",
        validate="many_to_one",
    )

    if out["failure_cycle"].isna().any():
        raise ValueError("Failed to attach failure_cycle to all test rows")

    out["unit_id"] = out["unit_id"].astype(int)
    out["cycle"] = out["cycle"].astype(int)
    out["max_cycle_observed"] = out["max_cycle_observed"].astype(int)
    out["rul"] = out["rul"].astype(int)
    out["failure_cycle"] = out["failure_cycle"].astype(int)

    out["rul_at_cycle"] = out["failure_cycle"] - out["cycle"]

    if (out["rul_at_cycle"] < 0).any():
        bad = out.loc[out["rul_at_cycle"] < 0, ["unit_id", "cycle", "failure_cycle"]].head(10)
        raise ValueError(f"Negative rul_at_cycle encountered:\n{bad}")

    for thr in LATE_LIFE_THRESHOLDS:
        out[f"is_late_life_{thr}"] = (out["rul_at_cycle"] <= thr).astype(int)

    return out.sort_values(["unit_id", "cycle"], kind="mergesort").reset_index(drop=True)


def _sensor_variability_summary(test_df: pd.DataFrame) -> pd.DataFrame:
    sensor_cols = [c for c in test_df.columns if c.startswith("sensor_")]
    if not sensor_cols:
        raise ValueError("No sensor_* columns found in prepared test CSV")

    rows: List[Dict[str, float | str]] = []
    for col in sensor_cols:
        s = pd.to_numeric(test_df[col], errors="coerce")
        rows.append(
            {
                "sensor": col,
                "non_null_count": int(s.notna().sum()),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=0)),
                "min": float(s.min()),
                "max": float(s.max()),
                "n_unique": int(s.nunique(dropna=True)),
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(["std", "n_unique"], ascending=[False, False], kind="mergesort").reset_index(drop=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["FD001", "FD002", "FD003", "FD004"])
    args = ap.parse_args()

    dataset = str(args.dataset).upper()
    dataset_dir = f"cmapss_{dataset.lower()}"
    ds = dataset.lower()
    repo = _repo_root()
    prepared_dir = repo / "data" / dataset_dir / "prepared"
    demo_dir = repo / "data" / dataset_dir / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)

    test_df, rul_df, prep_meta = _load_prepared(prepared_dir, dataset)

    engine_df = _build_engine_summary(test_df, rul_df)
    scored_df = _attach_rul_targets(test_df, engine_df)
    sensor_summary_df = _sensor_variability_summary(test_df)

    scored_path = demo_dir / f"{ds}_test_scored.csv"
    engine_summary_path = demo_dir / f"{ds}_engine_summary.csv"
    sensor_summary_path = demo_dir / f"{ds}_sensor_summary.csv"
    metadata_path = demo_dir / f"{ds}_demo_metadata.json"

    scored_df.to_csv(scored_path, index=False)
    engine_df.to_csv(engine_summary_path, index=False)
    sensor_summary_df.to_csv(sensor_summary_path, index=False)

    metadata = {
        "source_dataset": f"NASA CMAPSS {dataset}",
        "prepared_inputs": {
            "test_csv": str(prepared_dir / f"test_{ds}.csv"),
            "rul_csv": str(prepared_dir / f"rul_{ds}.csv"),
            "metadata_json": str(prepared_dir / "metadata.json"),
        },
        "artifacts": {
            f"{ds}_test_scored_csv": str(scored_path),
            f"{ds}_engine_summary_csv": str(engine_summary_path),
            f"{ds}_sensor_summary_csv": str(sensor_summary_path),
        },
        "row_counts": {
            "test_rows": int(len(test_df)),
            "engine_count": int(engine_df["unit_id"].nunique()),
            "scored_rows": int(len(scored_df)),
        },
        "label_definition": {
            "failure_cycle": "max_cycle_observed + final RUL for each test engine",
            "rul_at_cycle": "failure_cycle - current cycle",
            "late_life_thresholds": LATE_LIFE_THRESHOLDS,
        },
        "top_variable_sensors_by_std": sensor_summary_df["sensor"].head(10).tolist(),
        "prepared_metadata_summary": prep_meta,
    }

    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(f"Built CMAPSS {dataset} demo artifacts:")
    print(f"  {scored_path}")
    print(f"  {engine_summary_path}")
    print(f"  {sensor_summary_path}")
    print(f"  {metadata_path}")


if __name__ == "__main__":
    main()
