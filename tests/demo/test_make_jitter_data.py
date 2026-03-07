from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import numpy as np

from htm_monitor.demo import make_jitter_data


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_make_jitter_data_outputs_and_manifest_match() -> None:
    repo = _repo_root()
    src_dir = repo / "data" / "demo_synth"
    out_dir = repo / "data" / "demo_synth_jitter"

    make_jitter_data.main()

    assert out_dir.exists()
    assert (out_dir / "gt_timestamps.json").exists()
    assert (out_dir / "jitter_manifest.json").exists()

    manifest = json.loads((out_dir / "jitter_manifest.json").read_text())
    gt_out = json.loads((out_dir / "gt_timestamps.json").read_text())

    assert set(manifest["signals"].keys()) == {"sA", "sB", "sC"}
    assert set(gt_out.keys()) == {"sA", "sB", "sC"}

    assert manifest["seed"] == 0
    assert manifest["max_jitter_seconds"] == 2

    any_timestamp_changed = False

    for sig in ["sA", "sB", "sC"]:
        src = pd.read_csv(src_dir / f"{sig}.csv")
        out = pd.read_csv(out_dir / f"{sig}.csv")

        assert len(out) == len(src)

        np.testing.assert_allclose(
            np.sort(out["value"].to_numpy()),
            np.sort(src["value"].to_numpy()),
            rtol=1e-12,
            atol=1e-12,
        )

        src_ts = src["timestamp"].tolist()
        out_ts = out["timestamp"].tolist()

        if out_ts != src_ts:
            any_timestamp_changed = True

        assert out["timestamp"].is_monotonic_increasing
        assert manifest["signals"][sig]["rows_kept"] == len(out)
        assert manifest["signals"][sig]["jitter_seconds"] == 2
        assert manifest["signals"][sig]["gt_timestamps"] == gt_out[sig]

        out_ts_set = set(out_ts)
        for ts in gt_out[sig]:
            assert ts in out_ts_set, f"{sig} GT timestamp missing from jittered CSV: {ts}"

    # Sanity: fixture should actually introduce some timestamp change somewhere
    assert any_timestamp_changed, "Expected at least one signal to have jittered timestamps"