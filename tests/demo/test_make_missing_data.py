from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from htm_monitor.demo import make_missing_data


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_make_missing_data_outputs_and_manifest_match() -> None:
    repo = _repo_root()
    src_dir = repo / "data" / "demo_synth"
    out_dir = repo / "data" / "demo_synth_missing"

    # Rebuild deterministically
    make_missing_data.main()

    assert out_dir.exists()
    assert (out_dir / "sA.csv").exists()
    assert (out_dir / "sB.csv").exists()
    assert (out_dir / "sC.csv").exists()
    assert (out_dir / "gt_timestamps.json").exists()
    assert (out_dir / "missing_manifest.json").exists()

    manifest = json.loads((out_dir / "missing_manifest.json").read_text())
    assert manifest["seed"] == 42
    assert set(manifest["signals"].keys()) == {"sA", "sB", "sC"}

    for sig in ("sA", "sB", "sC"):
        src = pd.read_csv(src_dir / f"{sig}.csv")
        out = pd.read_csv(out_dir / f"{sig}.csv")

        # Same row count and same timestamps: missingness test should not alter cadence
        assert len(src) == len(out)
        assert src["timestamp"].tolist() == out["timestamp"].tolist()

    # sA: clean control
    sA = pd.read_csv(out_dir / "sA.csv")
    assert int(sA["value"].isna().sum()) == 0
    assert manifest["signals"]["sA"]["sparse_missing_rows"] == []
    assert manifest["signals"]["sA"]["dropout_rows"] == []

    # sB: sparse missing only
    sB = pd.read_csv(out_dir / "sB.csv")
    sB_missing_rows = [int(i) for i in sB.index[sB["value"].isna()].tolist()]
    assert sB_missing_rows == manifest["signals"]["sB"]["sparse_missing_rows"]
    assert manifest["signals"]["sB"]["dropout_rows"] == []

    # sC: contiguous dropout only
    sC = pd.read_csv(out_dir / "sC.csv")
    sC_missing_rows = [int(i) for i in sC.index[sC["value"].isna()].tolist()]
    assert sC_missing_rows == manifest["signals"]["sC"]["dropout_rows"]
    assert manifest["signals"]["sC"]["sparse_missing_rows"] == []

    # GT copied unchanged
    gt_src = json.loads((src_dir / "gt_timestamps.json").read_text())
    gt_out = json.loads((out_dir / "gt_timestamps.json").read_text())
    assert gt_src == gt_out
