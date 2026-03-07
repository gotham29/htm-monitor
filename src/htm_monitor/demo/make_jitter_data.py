from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


JITTER_SECONDS = 2
SEED = 0


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _jitter_timestamps(ts: pd.Series, *, max_jitter_seconds: int, rng: np.random.Generator) -> pd.Series:
    ts = pd.to_datetime(ts)

    jitter = rng.integers(
        -max_jitter_seconds,
        max_jitter_seconds + 1,
        size=len(ts),
    )

    return ts + pd.to_timedelta(jitter, unit="s")


def main() -> None:
    repo = _repo_root()

    src_dir = repo / "data" / "demo_synth"
    out_dir = repo / "data" / "demo_synth_jitter"
    out_dir.mkdir(parents=True, exist_ok=True)

    signals = ["sA", "sB", "sC"]
    gt_src = json.loads((src_dir / "gt_timestamps.json").read_text())

    manifest: dict[str, object] = {
        "seed": SEED,
        "max_jitter_seconds": JITTER_SECONDS,
        "signals": {},
    }
    gt_out: dict[str, list[str]] = {}

    for i, sig in enumerate(signals):
        rng = np.random.default_rng(SEED + i)

        df = pd.read_csv(src_dir / f"{sig}.csv")

        orig_ts = pd.to_datetime(df["timestamp"])
        jittered_ts = _jitter_timestamps(
            df["timestamp"],
            max_jitter_seconds=JITTER_SECONDS,
            rng=rng,
        )

        ts_map = {
            orig.strftime("%Y-%m-%d %H:%M:%S"): jit.strftime("%Y-%m-%d %H:%M:%S")
            for orig, jit in zip(orig_ts, jittered_ts)
        }

        df["timestamp"] = [t.strftime("%Y-%m-%d %H:%M:%S") for t in jittered_ts]
        df = df.sort_values("timestamp").reset_index(drop=True)
        df.to_csv(out_dir / f"{sig}.csv", index=False)

        src_gt = gt_src.get(sig, [])
        gt_out[sig] = [ts_map[ts] for ts in src_gt]

        manifest["signals"][sig] = {
            "rows_kept": int(len(df)),
            "jitter_seconds": JITTER_SECONDS,
            "gt_timestamps": gt_out[sig],
        }

    (out_dir / "gt_timestamps.json").write_text(json.dumps(gt_out, indent=2))
    (out_dir / "jitter_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Jitter dataset written to:\n{out_dir}")


if __name__ == "__main__":
    main()
