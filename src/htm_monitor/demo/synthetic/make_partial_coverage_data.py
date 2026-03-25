from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]

SRC_DIR = REPO_ROOT / "data" / "demo_synth"
OUT_DIR = REPO_ROOT / "data" / "demo_synth_partial_coverage"

START_LATE_ROWS = 400
END_EARLY_ROWS = 400


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sA = pd.read_csv(SRC_DIR / "sA.csv")
    sB = pd.read_csv(SRC_DIR / "sB.csv")
    sC = pd.read_csv(SRC_DIR / "sC.csv")

    nA = len(sA)
    nB = len(sB)
    nC = len(sC)

    # Phase 3 fixture:
    # - sA full coverage
    # - sB starts late
    # - sC ends early
    sA_out = sA.copy().reset_index(drop=True)
    sB_out = sB.iloc[START_LATE_ROWS:].reset_index(drop=True)
    sC_out = sC.iloc[: nC - END_EARLY_ROWS].reset_index(drop=True)

    sA_out.to_csv(OUT_DIR / "sA.csv", index=False)
    sB_out.to_csv(OUT_DIR / "sB.csv", index=False)
    sC_out.to_csv(OUT_DIR / "sC.csv", index=False)

    gt = json.loads((SRC_DIR / "gt_timestamps.json").read_text())
    (OUT_DIR / "gt_timestamps.json").write_text(json.dumps(gt, indent=2))

    manifest = {
        "type": "partial_coverage_fixture",
        "signals": {
            "sA": {
                "rows_original": nA,
                "rows_kept": len(sA_out),
                "first_row_kept": 0,
                "last_row_kept": nA - 1,
                "coverage": "full",
            },
            "sB": {
                "rows_original": nB,
                "rows_kept": len(sB_out),
                "first_row_kept": START_LATE_ROWS,
                "last_row_kept": nB - 1,
                "coverage": "starts_late",
            },
            "sC": {
                "rows_original": nC,
                "rows_kept": len(sC_out),
                "first_row_kept": 0,
                "last_row_kept": (nC - END_EARLY_ROWS - 1),
                "coverage": "ends_early",
            },
        },
    }

    (OUT_DIR / "partial_coverage_manifest.json").write_text(json.dumps(manifest, indent=2))

    print("Partial-coverage dataset written to:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
