import csv
import os
import subprocess
import sys
from pathlib import Path


def test_run_pipeline_smoke_trio_multisignal(tmp_path: Path) -> None:
    """
    End-to-end smoke test:
      - writes a minimal user config pointing at tiny fixture CSVs
      - runs `python -m demo.run_pipeline ...`
      - asserts it produces a non-empty output CSV

    We keep plot disabled to avoid GUI / backend issues in CI.
    """
    repo_root = Path(__file__).resolve().parents[1]
    defaults = repo_root / "configs" / "htm_defaults.yaml"
    assert defaults.exists(), "Expected configs/htm_defaults.yaml to exist in repo"

    fx = repo_root / "tests" / "fixtures" / "trio_small"
    assert fx.exists(), "Expected tests/fixtures/trio_small to exist"

    # Write a minimal user config that uses 3 sources but 1 model (multi-signal).
    user_cfg = tmp_path / "user_config.yaml"
    user_cfg.write_text(
        f"""
features:
  timestamp:
    type: datetime
    format: "%Y-%m-%d %H:%M:%S"
    timeOfDay: 21
    weekend: 0
    dayOfWeek: 0
    holiday: 0
    season: 0

  jumpsdown:
    type: float
    size: 512
    activeBits: 21
    numBuckets: 100
    minVal: 0.0
    maxVal: 200.0
    seed: 1

  jumpsup:
    type: float
    size: 512
    activeBits: 21
    numBuckets: 100
    minVal: 0.0
    maxVal: 200.0
    seed: 2

  nojump:
    type: float
    size: 512
    activeBits: 21
    numBuckets: 100
    minVal: 0.0
    maxVal: 200.0
    seed: 3

models:
  trio_model:
    sources: ["jumpsdown", "jumpsup", "nojump"]
    features: ["timestamp", "jumpsdown", "jumpsup", "nojump"]

data:
  timebase:
    mode: intersection
    on_missing: skip

  sources:
    - name: jumpsdown
      kind: csv
      path: {str((fx / "jumpsdown.csv").resolve())}
      timestamp_col: timestamp
      timestamp_format: "%Y-%m-%d %H:%M:%S"
      fields:
        jumpsdown: value
      labels:
        timestamps:
          - "2014-04-11 05:00:00"

    - name: jumpsup
      kind: csv
      path: {str((fx / "jumpsup.csv").resolve())}
      timestamp_col: timestamp
      timestamp_format: "%Y-%m-%d %H:%M:%S"
      fields:
        jumpsup: value
      labels:
        timestamps:
          - "2014-04-11 05:00:00"

    - name: nojump
      kind: csv
      path: {str((fx / "nojump.csv").resolve())}
      timestamp_col: timestamp
      timestamp_format: "%Y-%m-%d %H:%M:%S"
      fields:
        nojump: value
      labels:
        timestamps:
          - "2014-04-11 05:00:00"

decision:
  threshold: 0.40
  method: kofn_window
  k: 1
  window:
    size: 3
    per_model_hits: 1

plot:
  enable: false
  show_ground_truth: false
""".lstrip()
    )

    out_csv = tmp_path / "out.csv"

    env = dict(os.environ)
    # Ensures matplotlib (if imported anywhere) won't try to use an interactive backend.
    env["MPLBACKEND"] = "Agg"

    cmd = [
        sys.executable,
        "-m",
        "demo.run_pipeline",
        "--defaults",
        str(defaults),
        "--config",
        str(user_cfg),
        "--out",
        str(out_csv),
    ]

    r = subprocess.run(cmd, cwd=str(repo_root), env=env, capture_output=True, text=True)
    assert r.returncode == 0, f"run_pipeline failed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"

    assert out_csv.exists(), "Expected output CSV to be written"
    rows = list(csv.DictReader(out_csv.open("r", newline="")))
    assert len(rows) >= 5, "Expected at least a few output rows"
    # Basic schema sanity:
    for k in ["t", "timestamp", "model", "raw", "likelihood", "system_score", "alert"]:
        assert k in rows[0], f"Missing expected output column: {k}"