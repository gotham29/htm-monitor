from __future__ import annotations

import json
from pathlib import Path
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]

SRC_DIR = REPO_ROOT / "data" / "demo_synth"
OUT_DIR = REPO_ROOT / "data" / "demo_synth_multirate"


def downsample(df: pd.DataFrame, stride: int) -> pd.DataFrame:
    return df.iloc[::stride].reset_index(drop=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sA = pd.read_csv(SRC_DIR / "sA.csv")
    sB = pd.read_csv(SRC_DIR / "sB.csv")
    sC = pd.read_csv(SRC_DIR / "sC.csv")

    sA_out = sA.copy()
    sB_out = downsample(sB, 2)
    sC_out = downsample(sC, 4)

    sA_out.to_csv(OUT_DIR / "sA.csv", index=False)
    sB_out.to_csv(OUT_DIR / "sB.csv", index=False)
    sC_out.to_csv(OUT_DIR / "sC.csv", index=False)

    # copy GT
    gt = json.loads((SRC_DIR / "gt_timestamps.json").read_text())
    (OUT_DIR / "gt_timestamps.json").write_text(json.dumps(gt, indent=2))

    manifest = {
        "type": "multirate_fixture",
        "signals": {
            "sA": {"stride": 1},
            "sB": {"stride": 2},
            "sC": {"stride": 4},
        },
    }

    (OUT_DIR / "multirate_manifest.json").write_text(json.dumps(manifest, indent=2))

    print("Multirate dataset written to:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()