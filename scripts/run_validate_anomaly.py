from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from htm_monitor.diagnostics.validate_anomaly import validate_anomaly_window


def _load_signal_map(config: dict) -> dict:
    """
    Load raw signal data based on config.
    Returns:
        {
            "demand": df,
            ...
        }
    """
    signal_map = {}

    for src in config["data"]["sources"]:
        name = src["name"]
        path = Path(src["path"])

        df = pd.read_csv(path)
        df["ts"] = pd.to_datetime(df[src["timestamp_col"]])

        # map field -> value column
        field_map = src["fields"]
        if len(field_map) != 1:
            raise ValueError(f"{name}: expected 1 field mapping")

        field_name, col = list(field_map.items())[0]
        df = df[["ts", col]].rename(columns={col: field_name})

        signal_map[name] = df

    return signal_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--episode-details", required=True)
    ap.add_argument("--episode-id", type=int, default=None)
    ap.add_argument("--ts-start", default=None)
    ap.add_argument("--ts-end", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    episodes = json.load(open(args.episode_details))

    if args.episode_id is None and args.ts_start is None:
        raise ValueError("Provide either --episode-id or --ts-start")

    ep = None
    if args.episode_id is not None:
        matches = [e for e in episodes if int(e["episode_id"]) == int(args.episode_id)]
        if not matches:
            raise ValueError(f"No episode found for episode_id={args.episode_id}")
        ep = matches[0]

    if args.ts_start is not None:
        ts_matches = [
            e for e in episodes
            if str(e.get("ts_start")) == str(args.ts_start)
            and (args.ts_end is None or str(e.get("ts_end")) == str(args.ts_end))
        ]
        if not ts_matches:
            raise ValueError(
                f"No episode found for ts_start={args.ts_start}"
                + (f", ts_end={args.ts_end}" if args.ts_end is not None else "")
            )
        ep = ts_matches[0]

    start_ts = pd.Timestamp(ep["ts_start"])
    end_ts = pd.Timestamp(ep["ts_end"])

    signal_map = _load_signal_map(config)

    result = validate_anomaly_window(
        signal_map,
        start_ts=start_ts,
        end_ts=end_ts,
        baseline_days=7,
        cross_signal_pairs=[
            ("demand", "net_generation"),
            ("demand", "imbalance"),
        ],
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"validated episode_id={ep['episode_id']} ts={ep['ts_start']}..{ep['ts_end']}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()