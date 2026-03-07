from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from htm_monitor.demo import make_partial_coverage_data


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_make_partial_coverage_outputs_and_manifest_match() -> None:
    repo = _repo_root()
    src_dir = repo / "data" / "demo_synth"
    out_dir = repo / "data" / "demo_synth_partial_coverage"

    make_partial_coverage_data.main()

    assert out_dir.exists()
    assert (out_dir / "sA.csv").exists()
    assert (out_dir / "sB.csv").exists()
    assert (out_dir / "sC.csv").exists()
    assert (out_dir / "gt_timestamps.json").exists()
    assert (out_dir / "partial_coverage_manifest.json").exists()

    manifest = json.loads((out_dir / "partial_coverage_manifest.json").read_text())
    assert set(manifest["signals"].keys()) == {"sA", "sB", "sC"}

    sA_src = pd.read_csv(src_dir / "sA.csv")
    sB_src = pd.read_csv(src_dir / "sB.csv")
    sC_src = pd.read_csv(src_dir / "sC.csv")

    sA_out = pd.read_csv(out_dir / "sA.csv")
    sB_out = pd.read_csv(out_dir / "sB.csv")
    sC_out = pd.read_csv(out_dir / "sC.csv")

    # sA unchanged
    assert len(sA_out) == len(sA_src)
    assert sA_out["timestamp"].tolist() == sA_src["timestamp"].tolist()
    np.testing.assert_allclose(
        sA_out["value"].to_numpy(),
        sA_src["value"].to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )

    # sB starts late: should match original from first_row_kept onward
    b0 = int(manifest["signals"]["sB"]["first_row_kept"])
    assert b0 > 0
    assert sB_out["timestamp"].tolist() == sB_src.iloc[b0:]["timestamp"].tolist()
    np.testing.assert_allclose(
        sB_out["value"].to_numpy(),
        sB_src.iloc[b0:]["value"].to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )

    # sC ends early: should match original up to last_row_kept inclusive
    c_last = int(manifest["signals"]["sC"]["last_row_kept"])
    assert c_last < (len(sC_src) - 1)
    assert sC_out["timestamp"].tolist() == sC_src.iloc[: c_last + 1]["timestamp"].tolist()
    np.testing.assert_allclose(
        sC_out["value"].to_numpy(),
        sC_src.iloc[: c_last + 1]["value"].to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )

    # Manifest row counts should match reality
    assert manifest["signals"]["sA"]["rows_kept"] == len(sA_out)
    assert manifest["signals"]["sB"]["rows_kept"] == len(sB_out)
    assert manifest["signals"]["sC"]["rows_kept"] == len(sC_out)

    # GT copied through exactly
    gt_src = json.loads((src_dir / "gt_timestamps.json").read_text())
    gt_out = json.loads((out_dir / "gt_timestamps.json").read_text())
    assert gt_out == gt_src
