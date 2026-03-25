from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TARGET_UNIT = 34
EARLY_LIFE_RUL_MIN = 60
LATE_LIFE_RUL_MAX = 30

DEFAULT_SENSORS: List[str] = [
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


def _load_prepared_fd001(prepared_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    test_csv = prepared_dir / "test_fd001.csv"
    rul_csv = prepared_dir / "rul_fd001.csv"
    metadata_json = prepared_dir / "metadata.json"

    if not test_csv.exists():
        raise FileNotFoundError(f"Missing file: {test_csv}")
    if not rul_csv.exists():
        raise FileNotFoundError(f"Missing file: {rul_csv}")
    if not metadata_json.exists():
        raise FileNotFoundError(f"Missing file: {metadata_json}")

    test_df = pd.read_csv(test_csv)
    rul_df = pd.read_csv(rul_csv)
    metadata = json.loads(metadata_json.read_text())

    return test_df, rul_df, metadata


def _build_scored_test_df(test_df: pd.DataFrame, rul_df: pd.DataFrame) -> pd.DataFrame:
    df = test_df.copy()

    max_cycle = (
        df.groupby("unit_id", as_index=False)["cycle"]
        .max()
        .rename(columns={"cycle": "max_cycle_observed"})
    )

    df = df.merge(max_cycle, on="unit_id", how="left")
    df = df.merge(rul_df.rename(columns={"test_unit_id": "unit_id"}), on="unit_id", how="left")

    if df["rul"].isna().any():
        missing_units = sorted(df.loc[df["rul"].isna(), "unit_id"].unique().tolist())
        raise ValueError(f"Missing RUL labels for test units: {missing_units[:10]}")

    df["failure_cycle"] = df["max_cycle_observed"] + df["rul"]
    df["rul_at_cycle"] = df["failure_cycle"] - df["cycle"]

    df["is_late_life_30"] = (df["rul_at_cycle"] <= 30).astype(int)
    df["is_late_life_20"] = (df["rul_at_cycle"] <= 20).astype(int)
    df["is_late_life_10"] = (df["rul_at_cycle"] <= 10).astype(int)

    return df


def _safe_float(x: float) -> float | None:
    if pd.isna(x):
        return None
    return float(x)


def _cohens_d(a: pd.Series, b: pd.Series) -> float | None:
    a = pd.to_numeric(a, errors="coerce").dropna()
    b = pd.to_numeric(b, errors="coerce").dropna()

    if len(a) < 2 or len(b) < 2:
        return None

    mean_a = float(a.mean())
    mean_b = float(b.mean())
    var_a = float(a.var(ddof=1))
    var_b = float(b.var(ddof=1))

    pooled_num = ((len(a) - 1) * var_a) + ((len(b) - 1) * var_b)
    pooled_den = len(a) + len(b) - 2
    if pooled_den <= 0:
        return None

    pooled_std = np.sqrt(pooled_num / pooled_den)
    if pooled_std <= 0:
        return None

    return float((mean_b - mean_a) / pooled_std)


def _rank_sensors_for_late_life(df: pd.DataFrame, sensors: List[str]) -> pd.DataFrame:
    rows = []

    early = df[df["rul_at_cycle"] >= EARLY_LIFE_RUL_MIN].copy()
    late = df[df["rul_at_cycle"] <= LATE_LIFE_RUL_MAX].copy()

    for sensor in sensors:
        s_all = pd.to_numeric(df[sensor], errors="coerce")
        s_early = pd.to_numeric(early[sensor], errors="coerce")
        s_late = pd.to_numeric(late[sensor], errors="coerce")

        effect = _cohens_d(s_early, s_late)
        corr = pd.DataFrame(
            {"sensor": s_all, "rul_at_cycle": pd.to_numeric(df["rul_at_cycle"], errors="coerce")}
        ).corr(method="spearman").loc["sensor", "rul_at_cycle"]

        rows.append(
            {
                "sensor": sensor,
                "non_null_count": int(s_all.notna().sum()),
                "early_mean_rul_ge_60": _safe_float(s_early.mean()),
                "late_mean_rul_le_30": _safe_float(s_late.mean()),
                "late_minus_early": _safe_float(s_late.mean() - s_early.mean()),
                "cohens_d_late_vs_early": effect,
                "abs_cohens_d": abs(effect) if effect is not None else None,
                "spearman_corr_vs_rul": _safe_float(corr),
                "abs_spearman_corr_vs_rul": abs(float(corr)) if pd.notna(corr) else None,
                "std_all": _safe_float(s_all.std()),
                "min_all": _safe_float(s_all.min()),
                "max_all": _safe_float(s_all.max()),
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(
        by=["abs_cohens_d", "abs_spearman_corr_vs_rul", "std_all"],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    return out


def _unit_vs_fleet_summary(df: pd.DataFrame, unit_id: int, sensors: List[str]) -> pd.DataFrame:
    unit_df = df[df["unit_id"] == unit_id].copy()
    if unit_df.empty:
        raise ValueError(f"No rows found for unit_id={unit_id}")

    rows = []

    first_late_cycle = unit_df.loc[unit_df["is_late_life_30"] == 1, "cycle"].min()
    if pd.isna(first_late_cycle):
        first_late_cycle = None

    for sensor in sensors:
        unit_series = pd.to_numeric(unit_df[sensor], errors="coerce")
        fleet_series = pd.to_numeric(df[sensor], errors="coerce")

        unit_early = pd.to_numeric(unit_df.loc[unit_df["rul_at_cycle"] >= EARLY_LIFE_RUL_MIN, sensor], errors="coerce")
        unit_late = pd.to_numeric(unit_df.loc[unit_df["rul_at_cycle"] <= LATE_LIFE_RUL_MAX, sensor], errors="coerce")

        fleet_early = pd.to_numeric(df.loc[df["rul_at_cycle"] >= EARLY_LIFE_RUL_MIN, sensor], errors="coerce")
        fleet_late = pd.to_numeric(df.loc[df["rul_at_cycle"] <= LATE_LIFE_RUL_MAX, sensor], errors="coerce")

        unit_effect = _cohens_d(unit_early, unit_late)
        fleet_effect = _cohens_d(fleet_early, fleet_late)

        rows.append(
            {
                "unit_id": unit_id,
                "sensor": sensor,
                "unit_non_null_count": int(unit_series.notna().sum()),
                "fleet_non_null_count": int(fleet_series.notna().sum()),
                "unit_cycle_min": int(unit_df["cycle"].min()),
                "unit_cycle_max": int(unit_df["cycle"].max()),
                "unit_first_late_life_30_cycle": int(first_late_cycle) if first_late_cycle is not None else None,
                "unit_mean_all": _safe_float(unit_series.mean()),
                "unit_std_all": _safe_float(unit_series.std()),
                "unit_early_mean_rul_ge_60": _safe_float(unit_early.mean()),
                "unit_late_mean_rul_le_30": _safe_float(unit_late.mean()),
                "unit_late_minus_early": _safe_float(unit_late.mean() - unit_early.mean()),
                "unit_cohens_d_late_vs_early": unit_effect,
                "fleet_early_mean_rul_ge_60": _safe_float(fleet_early.mean()),
                "fleet_late_mean_rul_le_30": _safe_float(fleet_late.mean()),
                "fleet_late_minus_early": _safe_float(fleet_late.mean() - fleet_early.mean()),
                "fleet_cohens_d_late_vs_early": fleet_effect,
            }
        )

    return pd.DataFrame(rows)


def _plot_unit_sensor_panels(
    df: pd.DataFrame,
    unit_id: int,
    sensors: List[str],
    out_png: Path,
) -> None:
    unit_df = df[df["unit_id"] == unit_id].copy()
    if unit_df.empty:
        raise ValueError(f"No rows found for unit_id={unit_id}")

    unit_df = unit_df.sort_values("cycle", kind="mergesort").reset_index(drop=True)

    first_late_cycle = unit_df.loc[unit_df["is_late_life_30"] == 1, "cycle"].min()
    if pd.isna(first_late_cycle):
        first_late_cycle = None

    n = len(sensors)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 3.2 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, sensor in zip(axes_flat, sensors):
        y = pd.to_numeric(unit_df[sensor], errors="coerce")
        x = unit_df["cycle"]

        ax.plot(x, y, linewidth=1.8)
        ax.set_title(sensor)
        ax.set_xlabel("Cycle")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

        if first_late_cycle is not None:
            ax.axvline(first_late_cycle, linestyle="--", linewidth=1.5)
            ax.axvspan(first_late_cycle, int(unit_df["cycle"].max()), alpha=0.10)

    for ax in axes_flat[len(sensors):]:
        ax.set_axis_off()

    title = (
        f"CMAPSS FD001 — Unit {unit_id} sensor trajectories\n"
        f"Late-life zone starts at RUL<=30"
    )
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    repo = _repo_root()

    prepared_dir = repo / "data" / "cmapss_fd001" / "prepared"
    out_dir = repo / "data" / "cmapss_fd001" / "exploration" / f"unit_{TARGET_UNIT}"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_df, rul_df, metadata = _load_prepared_fd001(prepared_dir)
    scored_df = _build_scored_test_df(test_df, rul_df)

    all_sensors = list(metadata.get("sensors", []))
    if not all_sensors:
        all_sensors = [c for c in scored_df.columns if c.startswith("sensor_")]

    chosen_sensors = [s for s in DEFAULT_SENSORS if s in all_sensors]
    if not chosen_sensors:
        raise ValueError("No DEFAULT_SENSORS found in dataset columns")

    sensor_rank_df = _rank_sensors_for_late_life(scored_df, all_sensors)
    unit_vs_fleet_df = _unit_vs_fleet_summary(scored_df, TARGET_UNIT, chosen_sensors)

    out_rank_csv = out_dir / "sensor_rankings_for_late_life.csv"
    out_unit_csv = out_dir / "unit34_vs_fleet_summary.csv"
    out_png = out_dir / "unit34_sensor_panels.png"
    out_meta = out_dir / "exploration_metadata.json"

    sensor_rank_df.to_csv(out_rank_csv, index=False)
    unit_vs_fleet_df.to_csv(out_unit_csv, index=False)
    _plot_unit_sensor_panels(scored_df, TARGET_UNIT, chosen_sensors, out_png)

    meta = {
        "dataset": "CMAPSS_FD001",
        "target_unit": TARGET_UNIT,
        "early_life_definition": f"rul_at_cycle >= {EARLY_LIFE_RUL_MIN}",
        "late_life_definition": f"rul_at_cycle <= {LATE_LIFE_RUL_MAX}",
        "chosen_sensors_for_plotting": chosen_sensors,
        "all_ranked_sensors": all_sensors,
        "artifacts": {
            "sensor_rankings_for_late_life_csv": str(out_rank_csv),
            "unit34_vs_fleet_summary_csv": str(out_unit_csv),
            "unit34_sensor_panels_png": str(out_png),
        },
    }
    out_meta.write_text(json.dumps(meta, indent=2))

    print("Built CMAPSS FD001 exploration artifacts:")
    print(f"  {out_rank_csv}")
    print(f"  {out_unit_csv}")
    print(f"  {out_png}")
    print(f"  {out_meta}")


if __name__ == "__main__":
    main()
