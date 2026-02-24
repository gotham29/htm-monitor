# tests/test_calibrate_encoders.py

import pandas as pd
from pathlib import Path
import subprocess
import sys
import yaml

from demo.calibrate_encoders import (
    compute_min_max_from_series,
    calibrate_from_csv,
)


def test_compute_min_max_basic_behavior():
    s = pd.Series([0, 1, 2, 3, 4, 5, 100])  # long tail
    min_val, max_val = compute_min_max_from_series(
        s,
        low_q=0.01,
        high_q=0.99,
        margin_frac=0.03,
    )

    assert min_val < 1  # slightly below robust low
    assert max_val > 5  # slightly above robust high
    assert max_val < 100  # long tail excluded


def test_constant_series_expands_safely():
    s = pd.Series([5.0] * 100)
    min_val, max_val = compute_min_max_from_series(s)

    assert min_val < 5.0
    assert max_val > 5.0
    assert max_val > min_val


def test_calibrate_from_csv(tmp_path: Path):
    df = pd.DataFrame({
        "value": [1, 2, 3, 4, 5, 100]
    })

    csv_path = tmp_path / "test.csv"
    df.to_csv(csv_path, index=False)

    result = calibrate_from_csv(csv_path, "value")

    assert "minVal" in result
    assert "maxVal" in result
    assert result["minVal"] < 2
    assert result["maxVal"] > 4
    assert result["maxVal"] < 100


def test_calibrate_from_config_mode(tmp_path: Path):
    """
    End-to-end CLI test for config-driven mode using fixtures.
    Uses FIRST source defining the feature and fills missing minVal/maxVal.
    """
    repo_root = Path(__file__).resolve().parents[1]
    fixtures = repo_root / "tests" / "fixtures" / "trio_small"

    defaults = {
        "features": {},
        "data": {"sources": []},
        # minimal required keys that may exist in merged config;
        # we don't need models/decision for this tool to run.
    }

    user_cfg = {
        "features": {
            "timestamp": {"type": "datetime", "format": "%Y-%m-%d %H:%M:%S", "timeOfDay": 21},
            "x": {"type": "float", "size": 512, "activeBits": 21},
        },
        "data": {
            "sources": [
                {
                    "name": "s1",
                    "kind": "csv",
                    "path": str(fixtures / "nojump.csv"),
                    "timestamp_col": "timestamp",
                    "timestamp_format": "%Y-%m-%d %H:%M:%S",
                    "fields": {"x": "value"},
                }
            ]
        },
    }

    defaults_path = tmp_path / "defaults.yaml"
    config_path = tmp_path / "usecase.yaml"
    out_config_path = tmp_path / "calibrated.yaml"
    report_path = tmp_path / "report.txt"

    defaults_path.write_text(yaml.safe_dump(defaults, sort_keys=False))
    config_path.write_text(yaml.safe_dump(user_cfg, sort_keys=False))

    cmd = [
        sys.executable,
        "-m",
        "demo.calibrate_encoders",
        "--defaults",
        str(defaults_path),
        "--config",
        str(config_path),
        "--out-config",
        str(out_config_path),
        "--report",
        str(report_path),
        "--low",
        "0.01",
        "--high",
        "0.99",
        "--margin",
        "0.03",
    ]
    subprocess.check_call(cmd, cwd=str(repo_root))

    assert out_config_path.exists()
    out = yaml.safe_load(out_config_path.read_text())
    assert "features" in out
    assert "x" in out["features"]
    assert "minVal" in out["features"]["x"]
    assert "maxVal" in out["features"]["x"]
    assert out["features"]["x"]["maxVal"] > out["features"]["x"]["minVal"]

    assert report_path.exists()
    rep = report_path.read_text()
    assert "Encoder Calibration Report" in rep
    assert "feature: x" in rep


def test_calibrate_from_config_does_not_override_existing_minmax(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    fixtures = repo_root / "tests" / "fixtures" / "trio_small"

    defaults = {"features": {}, "data": {"sources": []}}
    user_cfg = {
        "features": {
            "timestamp": {"type": "datetime", "format": "%Y-%m-%d %H:%M:%S"},
            "x": {"type": "float", "size": 512, "activeBits": 21, "minVal": -1.0, "maxVal": 999.0},
        },
        "data": {"sources": [{
            "name": "s1",
            "kind": "csv",
            "path": str(fixtures / "nojump.csv"),
            "timestamp_col": "timestamp",
            "timestamp_format": "%Y-%m-%d %H:%M:%S",
            "fields": {"x": "value"},
        }]}
    }

    defaults_path = tmp_path / "defaults.yaml"
    config_path = tmp_path / "usecase.yaml"
    out_config_path = tmp_path / "calibrated.yaml"

    defaults_path.write_text(yaml.safe_dump(defaults, sort_keys=False))
    config_path.write_text(yaml.safe_dump(user_cfg, sort_keys=False))

    cmd = [
        sys.executable, "-m", "demo.calibrate_encoders",
        "--defaults", str(defaults_path),
        "--config", str(config_path),
        "--out-config", str(out_config_path),
    ]
    subprocess.check_call(cmd, cwd=str(repo_root))

    out = yaml.safe_load(out_config_path.read_text())
    assert out["features"]["x"]["minVal"] == -1.0
    assert out["features"]["x"]["maxVal"] == 999.0


def test_calibrate_from_config_override_clobbers_existing_minmax(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    fixtures = repo_root / "tests" / "fixtures" / "trio_small"

    defaults = {"features": {}, "data": {"sources": []}}
    user_cfg = {
        "features": {
            "timestamp": {"type": "datetime", "format": "%Y-%m-%d %H:%M:%S"},
            "x": {"type": "float", "size": 512, "activeBits": 21, "minVal": -1.0, "maxVal": 999.0},
        },
        "data": {"sources": [{
            "name": "s1",
            "kind": "csv",
            "path": str(fixtures / "nojump.csv"),
            "timestamp_col": "timestamp",
            "timestamp_format": "%Y-%m-%d %H:%M:%S",
            "fields": {"x": "value"},
        }]}
    }

    defaults_path = tmp_path / "defaults.yaml"
    config_path = tmp_path / "usecase.yaml"
    out_config_path = tmp_path / "calibrated.yaml"

    defaults_path.write_text(yaml.safe_dump(defaults, sort_keys=False))
    config_path.write_text(yaml.safe_dump(user_cfg, sort_keys=False))

    cmd = [
        sys.executable, "-m", "demo.calibrate_encoders",
        "--defaults", str(defaults_path),
        "--config", str(config_path),
        "--out-config", str(out_config_path),
        "--override",
    ]
    subprocess.check_call(cmd, cwd=str(repo_root))

    out = yaml.safe_load(out_config_path.read_text())
    # Should no longer be the sentinel values
    assert out["features"]["x"]["minVal"] != -1.0
    assert out["features"]["x"]["maxVal"] != 999.0
    assert out["features"]["x"]["maxVal"] > out["features"]["x"]["minVal"]