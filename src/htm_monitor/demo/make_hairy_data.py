#demo/make_hairy_data.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


SEED = 123
JITTER_SECONDS = 2


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _fmt_ts(series: pd.Series) -> List[str]:
    ts = pd.to_datetime(series)
    return [t.strftime("%Y-%m-%d %H:%M:%S") for t in ts]


def _read_source_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns or "value" not in df.columns:
        raise ValueError(f"{path} must contain columns: timestamp,value")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _jitter_series(ts: pd.Series, *, rng: np.random.Generator, max_seconds: int) -> pd.Series:
    offsets = rng.integers(-max_seconds, max_seconds + 1, size=len(ts))
    return pd.to_datetime(ts) + pd.to_timedelta(offsets, unit="s")


def _build_ts_map(orig_ts: pd.Series, jittered_ts: pd.Series) -> Dict[str, str]:
    return {
        o.strftime("%Y-%m-%d %H:%M:%S"): j.strftime("%Y-%m-%d %H:%M:%S")
        for o, j in zip(pd.to_datetime(orig_ts), pd.to_datetime(jittered_ts))
    }


def _sparse_nan_rows(n_rows: int, *, every: int, start: int, exclude: set[int]) -> List[int]:
    rows: List[int] = []
    i = int(start)
    while i < n_rows:
        if i not in exclude:
            rows.append(i)
        i += int(every)
    return rows


def main() -> None:
    repo = _repo_root()
    src_dir = repo / "data" / "demo_synth"
    out_dir = repo / "data" / "demo_synth_hairy"
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_src = json.loads((src_dir / "gt_timestamps.json").read_text())

    rng_a = np.random.default_rng(SEED + 0)
    rng_b = np.random.default_rng(SEED + 1)
    rng_c = np.random.default_rng(SEED + 2)

    manifest: Dict[str, object] = {
        "seed": SEED,
        "max_jitter_seconds": JITTER_SECONDS,
        "description": (
            "Combined robustness fixture: "
            "sA=jitter + very light sparse missing, "
            "sB=full coverage + short dropout + jitter, "
            "sC=late partial coverage + short dropout + jitter"
        ),
        "signals": {},
    }
    gt_out: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------
    # sA: full coverage, small timestamp jitter, very light sparse missing
    # ------------------------------------------------------------------
    sA = _read_source_csv(src_dir / "sA.csv")
    sA_jittered_ts = _jitter_series(
        sA["timestamp"], rng=rng_a, max_seconds=JITTER_SECONDS
    )
    sA_map = _build_ts_map(sA["timestamp"], sA_jittered_ts)

    gt_rows_sA = {
        i for i, ts in enumerate(_fmt_ts(sA["timestamp"])) if ts in set(gt_src["sA"])
    }
    sA_missing_rows = _sparse_nan_rows(
        len(sA), every=97, start=31, exclude=gt_rows_sA
    )
    sA.loc[sA_missing_rows, "value"] = np.nan

    sA["timestamp"] = sA_jittered_ts
    sA = sA.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    sA["timestamp"] = _fmt_ts(sA["timestamp"])
    sA.to_csv(out_dir / "sA.csv", index=False)

    gt_out["sA"] = [sA_map[ts] for ts in gt_src["sA"]]

    manifest["signals"]["sA"] = {
        "rows_kept": int(len(sA)),
        "jitter_seconds": JITTER_SECONDS,
        "sparse_missing_rows_pre_sort": [int(i) for i in sA_missing_rows],
        "transform": "full_coverage_jitter_plus_very_light_sparse_missing",
        "gt_timestamps": gt_out["sA"],
    }

    # ------------------------------------------------------------------
    # sB: full coverage, short dropout block, timestamp jitter
    # ------------------------------------------------------------------
    sB_src = _read_source_csv(src_dir / "sB.csv")
    sB = sB_src.copy().reset_index(drop=True)

    # Jitter the kept timestamps.
    sB_jittered_ts = _jitter_series(
        sB["timestamp"], rng=rng_b, max_seconds=JITTER_SECONDS
    )
    sB_map = _build_ts_map(sB["timestamp"], sB_jittered_ts)

    gt_rows_sB = {
        i for i, ts in enumerate(_fmt_ts(sB["timestamp"])) if ts in set(gt_src["sB"])
    }
    dropout_start_b = 690
    dropout_len_b = 10
    dropout_rows_b = [
        i
        for i in range(dropout_start_b, min(dropout_start_b + dropout_len_b, len(sB)))
        if i not in gt_rows_sB
    ]
    sB.loc[dropout_rows_b, "value"] = np.nan
    sB["timestamp"] = sB_jittered_ts
    sB = sB.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    sB["timestamp"] = _fmt_ts(sB["timestamp"])
    sB.to_csv(out_dir / "sB.csv", index=False)

    gt_out["sB"] = [sB_map[ts] for ts in gt_src["sB"]]

    manifest["signals"]["sB"] = {
        "rows_kept": int(len(sB)),
        "rows_dropped_from_source": 0,
        "dropout_rows_pre_sort": [int(i) for i in dropout_rows_b],
        "jitter_seconds": JITTER_SECONDS,
        "transform": "full_coverage_dropout_plus_jitter",
        "gt_timestamps": gt_out["sB"],
    }

    # ------------------------------------------------------------------
    # sC: late partial coverage, short dropout block, then timestamp jitter
    # ------------------------------------------------------------------
    sC_src = _read_source_csv(src_dir / "sC.csv")

    # End coverage early so the last GT falls outside observed coverage.
    # Keep enough data to still include the first two GT events.
    sC_last_row_kept = max(0, len(sC_src) - 72)
    sC = sC_src.iloc[: sC_last_row_kept + 1].copy().reset_index(drop=True)

    # Short dropout block inside observed coverage; avoid GT rows.
    gt_rows_sC = {
        i for i, ts in enumerate(_fmt_ts(sC["timestamp"])) if ts in set(gt_src["sC"])
    }
    dropout_start = 760
    dropout_len = 8
    dropout_rows = [
        i
        for i in range(dropout_start, min(dropout_start + dropout_len, len(sC)))
        if i not in gt_rows_sC
    ]
    sC.loc[dropout_rows, "value"] = np.nan

    sC_jittered_ts = _jitter_series(
        sC["timestamp"], rng=rng_c, max_seconds=JITTER_SECONDS
    )
    sC_map = _build_ts_map(sC["timestamp"], sC_jittered_ts)

    # Only GTs still inside retained coverage should be carried forward.
    retained_gt_sC = [
        ts for ts in gt_src["sC"] if ts in set(_fmt_ts(sC["timestamp"]))
    ]
    gt_out["sC"] = [sC_map[ts] for ts in retained_gt_sC]

    excluded_gt_sC = [ts for ts in gt_src["sC"] if ts not in set(retained_gt_sC)]

    sC["timestamp"] = sC_jittered_ts
    sC = sC.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    sC["timestamp"] = _fmt_ts(sC["timestamp"])
    sC.to_csv(out_dir / "sC.csv", index=False)

    manifest["signals"]["sC"] = {
        "rows_kept": int(len(sC)),
        "first_row_kept": 0,
        "last_row_kept": int(sC_last_row_kept),
        "rows_dropped_from_source": int(len(sC_src) - len(sC)),
        "dropout_rows_pre_sort": [int(i) for i in dropout_rows],
        "jitter_seconds": JITTER_SECONDS,
        "transform": "partial_coverage_plus_dropout_plus_jitter",
        "gt_timestamps": gt_out["sC"],
        "excluded_gt_timestamps_outside_coverage": excluded_gt_sC,
    }

    (out_dir / "gt_timestamps.json").write_text(json.dumps(gt_out, indent=2))
    (out_dir / "hairy_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Hairy dataset written to:\n{out_dir}")


if __name__ == "__main__":
    main()
