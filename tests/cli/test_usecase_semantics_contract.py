from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from htm_monitor.cli.analyze_run import load_run, summarize
from htm_monitor.utils.config import build_from_config


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_build_from_config_accepts_semantic_sections() -> None:
    defaults_path = REPO_ROOT / "configs" / "htm_defaults.yaml"
    user_path = REPO_ROOT / "configs" / "demo_synth.yaml"

    cfg, engine, decision, model_sources = build_from_config(
        str(defaults_path),
        str(user_path),
    )

    assert cfg["decision"]["method"] == "kofn_window"
    assert isinstance(model_sources, dict)
    assert "sA_model" in model_sources


def test_summarize_writes_use_case_semantics_for_onset_detection(tmp_path: Path) -> None:
    run_dir = REPO_ROOT / "outputs" / "demo_synth_semantics_check"
    csv_path = run_dir / "run.csv"
    config_path = REPO_ROOT / "configs" / "demo_synth.yaml"
    out_dir = tmp_path / "analysis"

    df = load_run(str(csv_path))
    summary = summarize(
        df,
        config_path=str(config_path),
        out_dir=str(out_dir),
        max_lag_steps=50,
        threshold_override=None,
        prefer_hot_by_model=False,
        strict=True,
    )

    assert summary["gt_mapping_mode"] == "strict"
    assert summary["use_case_semantics"]["evaluation_mode"] == "onset_detection"
    assert summary["use_case_semantics"]["implemented_scoring_mode"] == "onset_detection"
    assert summary["use_case_semantics"]["predictive_warning_contract"] is None

    written = json.loads((out_dir / "run_summary.json").read_text())
    assert written["use_case_semantics"]["evaluation_mode"] == "onset_detection"
    assert written["use_case_semantics"]["implemented_scoring_mode"] == "onset_detection"


def test_summarize_rejects_predictive_warning_mode_until_implemented(tmp_path: Path) -> None:
    run_dir = REPO_ROOT / "outputs" / "demo_synth_semantics_check"
    csv_path = run_dir / "run.csv"
    base_config_path = REPO_ROOT / "configs" / "demo_synth.yaml"
    tmp_config_path = tmp_path / "demo_synth_predictive.yaml"

    cfg = yaml.safe_load(base_config_path.read_text())
    cfg["evaluation"] = {
        "mode": "predictive_warning",
        "predictive_warning": {
            "event_type": "failure_horizon",
            "target_source": "system",
            "warning_window": {
                "start_steps_before_event": 30,
                "end_steps_before_event": 5,
            },
            "too_early_window": {
                "min_steps_before_event": 31,
            },
        },
    }
    tmp_config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    df = load_run(str(csv_path))

    with pytest.raises(NotImplementedError, match="predictive_warning"):
        summarize(
            df,
            config_path=str(tmp_config_path),
            out_dir=str(tmp_path / "analysis"),
            max_lag_steps=50,
            threshold_override=None,
            prefer_hot_by_model=False,
            strict=True,
        )


def test_build_from_config_rejects_invalid_learning_mode(tmp_path: Path) -> None:
    defaults_path = REPO_ROOT / "configs" / "htm_defaults.yaml"
    base_user_path = REPO_ROOT / "configs" / "demo_synth.yaml"
    bad_user_path = tmp_path / "bad_learning.yaml"

    cfg = yaml.safe_load(base_user_path.read_text())
    cfg["learning"] = {"mode": "totally_invalid_mode"}
    bad_user_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    with pytest.raises(ValueError, match="Config.learning.mode"):
        build_from_config(str(defaults_path), str(bad_user_path))


def test_build_from_config_rejects_invalid_evaluation_mode(tmp_path: Path) -> None:
    defaults_path = REPO_ROOT / "configs" / "htm_defaults.yaml"
    base_user_path = REPO_ROOT / "configs" / "demo_synth.yaml"
    bad_user_path = tmp_path / "bad_eval.yaml"

    cfg = yaml.safe_load(base_user_path.read_text())
    cfg["evaluation"] = {"mode": "not_a_real_mode"}
    bad_user_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    with pytest.raises(ValueError, match="Config.evaluation.mode"):
        build_from_config(str(defaults_path), str(bad_user_path))


def test_summarize_carries_predictive_warning_contract_metadata_before_not_implemented(tmp_path: Path) -> None:
    run_dir = REPO_ROOT / "outputs" / "demo_synth_semantics_check"
    csv_path = run_dir / "run.csv"
    base_config_path = REPO_ROOT / "configs" / "demo_synth.yaml"
    tmp_config_path = tmp_path / "predictive_contract.yaml"

    cfg = yaml.safe_load(base_config_path.read_text())
    cfg["evaluation"] = {
        "mode": "predictive_warning",
        "predictive_warning": {
            "event_type": "failure_horizon",
            "target_source": "system",
            "warning_window": {
                "start_steps_before_event": 30,
                "end_steps_before_event": 5,
            },
            "too_early_window": {
                "min_steps_before_event": 31,
            },
        },
    }
    tmp_config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    df = load_run(str(csv_path))

    with pytest.raises(NotImplementedError):
        summarize(
            df,
            config_path=str(tmp_config_path),
            out_dir=str(tmp_path / "analysis"),
            max_lag_steps=50,
            threshold_override=None,
            prefer_hot_by_model=False,
            strict=True,
        )
