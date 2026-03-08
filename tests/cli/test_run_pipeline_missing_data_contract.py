from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_pipeline(config_name: str, run_dir_name: str) -> Path:
    repo = _repo_root()
    run_dir = repo / "outputs" / run_dir_name

    if run_dir.exists():
        # keep it simple and deterministic for tests
        for p in sorted(run_dir.rglob("*"), reverse=True):
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                p.rmdir()
        run_dir.rmdir()

    cmd = [
        sys.executable,
        "-m",
        "htm_monitor.cli.run_pipeline",
        "--config",
        str(repo / "configs" / config_name),
        "--defaults",
        str(repo / "configs" / "htm_defaults.yaml"),
        "--run-dir",
        str(run_dir),
        "--no-plot",
    ]
    subprocess.run(cmd, check=True, cwd=repo)
    return run_dir


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def test_missing_skip_run_artifacts_and_manifest_contract() -> None:
    run_dir = _run_pipeline(
        config_name="demo_synth_missing_skip.yaml",
        run_dir_name="test_demo_synth_missing_skip",
    )

    assert (run_dir / "run.csv").exists()
    assert (run_dir / "run.manifest.json").exists()
    assert (run_dir / "analysis" / "run_summary.json").exists()
    assert (run_dir / "analysis" / "run_summary.md").exists()

    manifest = _load_json(run_dir / "run.manifest.json")
    summary = _load_json(run_dir / "analysis" / "run_summary.json")

    assert manifest["data"]["timebase_mode"] == "union"
    assert manifest["data"]["on_missing"] == "skip"

    assert summary["timebase"]["mode"] == "union"
    assert summary["timebase"]["on_missing"] == "skip"

    sys_eval = summary["ground_truth"]["system"]["eval"]
    assert sys_eval is not None
    assert summary["ground_truth"]["system"]["gt_onsets"]
    assert sys_eval["recall"] is not None
    assert float(sys_eval["recall"]) >= 0.99


def test_missing_holdlast_run_artifacts_and_manifest_contract() -> None:
    run_dir = _run_pipeline(
        config_name="demo_synth_missing_holdlast.yaml",
        run_dir_name="test_demo_synth_missing_holdlast",
    )

    assert (run_dir / "run.csv").exists()
    assert (run_dir / "run.manifest.json").exists()
    assert (run_dir / "analysis" / "run_summary.json").exists()
    assert (run_dir / "analysis" / "run_summary.md").exists()

    manifest = _load_json(run_dir / "run.manifest.json")
    summary = _load_json(run_dir / "analysis" / "run_summary.json")

    assert manifest["data"]["timebase_mode"] == "union"
    assert manifest["data"]["on_missing"] == "hold_last"

    assert summary["timebase"]["mode"] == "union"
    assert summary["timebase"]["on_missing"] == "hold_last"

    sys_eval = summary["ground_truth"]["system"]["eval"]
    assert sys_eval is not None
    assert summary["ground_truth"]["system"]["gt_onsets"]
    assert sys_eval["recall"] is not None
    assert float(sys_eval["recall"]) >= 0.99
