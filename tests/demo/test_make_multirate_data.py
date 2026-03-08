from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from htm_monitor.demo import make_multirate_data


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_make_multirate_data_outputs_and_manifest_match() -> None:
    repo = _repo_root()
    src_dir = repo / "data" / "demo_synth"
    out_dir = repo / "data" / "demo_synth_multirate"

    make_multirate_data.main()

    assert out_dir.exists()
    assert (out_dir / "sA.csv").exists()
    assert (out_dir / "sB.csv").exists()
    assert (out_dir / "sC.csv").exists()
    assert (out_dir / "gt_timestamps.json").exists()
    assert (out_dir / "multirate_manifest.json").exists()

    manifest = json.loads((out_dir / "multirate_manifest.json").read_text())
    assert set(manifest["signals"].keys()) == {"sA", "sB", "sC"}

    sA_src = pd.read_csv(src_dir / "sA.csv")
    sB_src = pd.read_csv(src_dir / "sB.csv")
    sC_src = pd.read_csv(src_dir / "sC.csv")

    sA_out = pd.read_csv(out_dir / "sA.csv")
    sB_out = pd.read_csv(out_dir / "sB.csv")
    sC_out = pd.read_csv(out_dir / "sC.csv")

    # sA stays full cadence
    assert len(sA_out) == len(sA_src)
    assert sA_out["timestamp"].tolist() == sA_src["timestamp"].tolist()
    assert sA_out["value"].tolist() == sA_src["value"].tolist()

    # sB and sC must be strictly downsampled
    assert 0 < len(sB_out) < len(sB_src)
    assert 0 < len(sC_out) < len(sC_src)

    # Output timestamps must remain ordered and come from original timestamps
    sB_src_ts = set(sB_src["timestamp"].tolist())
    sC_src_ts = set(sC_src["timestamp"].tolist())

    assert sB_out["timestamp"].is_monotonic_increasing
    assert sC_out["timestamp"].is_monotonic_increasing
    assert set(sB_out["timestamp"].tolist()).issubset(sB_src_ts)
    assert set(sC_out["timestamp"].tolist()).issubset(sC_src_ts)

    # Manifest should contain signal definitions
    assert "signals" in manifest
    assert set(manifest["signals"].keys()) == {"sA", "sB", "sC"}

    # CSVs themselves define the true row counts
    assert len(sA_out) == len(sA_src)
    assert 0 < len(sB_out) < len(sB_src)
    assert 0 < len(sC_out) < len(sC_src)

    # Multirate fixture should preserve GT file exactly
    gt_src = json.loads((src_dir / "gt_timestamps.json").read_text())
    gt_out = json.loads((out_dir / "gt_timestamps.json").read_text())
    assert gt_out == gt_src
