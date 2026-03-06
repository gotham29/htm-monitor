"""
Create deterministic missing-value variants of the synthetic demo dataset.

This script reads the clean demo dataset:

    data/demo_synth/

and writes a corrupted version to:

    data/demo_synth_missing/

Missing values are introduced in two ways:
    - sparse random missing values
    - contiguous dropout windows

Important:
Rows are NOT deleted. Values are replaced with blank cells so that the
timestamp cadence remains identical.

Outputs:
    data/demo_synth_missing/
        sA.csv
        sB.csv
        sC.csv
        gt_timestamps.json
        missing_manifest.json
"""

from pathlib import Path
import json
import random
import shutil
import pandas as pd


# ---------------------------------------------------------------------
# Configuration (first robustness scenario)
# ---------------------------------------------------------------------

SEED = 42

# sparse missing: fraction of rows to blank
SPARSE_MISSING = {
    "sB": 0.02,   # 2%
}

# contiguous dropout windows: start_idx:end_idx
DROPOUT_WINDOWS = {
    "sC": [(700, 712)],  # 12 step outage
}

# signals present
SIGNALS = ["sA", "sB", "sC"]


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]

DATA_IN = REPO_ROOT / "data" / "demo_synth"
DATA_OUT = REPO_ROOT / "data" / "demo_synth_missing"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def inject_sparse_missing(df, frac, rng):
    """Blank random rows in 'value' column."""
    n = len(df)
    k = int(n * frac)

    idx = rng.sample(range(n), k)

    df.loc[idx, "value"] = ""
    return idx


def inject_dropout(df, start, end):
    """Blank value column over a contiguous window."""
    df.loc[start:end-1, "value"] = ""
    return list(range(start, end))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    rng = random.Random(SEED)

    DATA_OUT.mkdir(parents=True, exist_ok=True)

    manifest = {
        "seed": SEED,
        "signals": {}
    }

    for sig in SIGNALS:

        in_path = DATA_IN / f"{sig}.csv"
        out_path = DATA_OUT / f"{sig}.csv"

        df = pd.read_csv(in_path)

        manifest["signals"][sig] = {
            "sparse_missing_rows": [],
            "dropout_rows": []
        }

        # sparse missing
        if sig in SPARSE_MISSING:
            rows = inject_sparse_missing(df, SPARSE_MISSING[sig], rng)
            manifest["signals"][sig]["sparse_missing_rows"] = rows

        # dropout windows
        if sig in DROPOUT_WINDOWS:
            for start, end in DROPOUT_WINDOWS[sig]:
                rows = inject_dropout(df, start, end)
                manifest["signals"][sig]["dropout_rows"].extend(rows)

        df.to_csv(out_path, index=False)

    # copy GT timestamps
    shutil.copy(
        DATA_IN / "gt_timestamps.json",
        DATA_OUT / "gt_timestamps.json"
    )

    # write manifest
    with open(DATA_OUT / "missing_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("Missing-data dataset written to:")
    print(DATA_OUT)


if __name__ == "__main__":
    main()
