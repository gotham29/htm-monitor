import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import pandas as pd
import yaml

from htm_monitor.diagnostics.validate_anomaly import validate_anomaly_window


def _load_signal_map(config: dict) -> dict:
    signal_map = {}

    for src in config["data"]["sources"]:
        name = src["name"]
        path = Path(src["path"])

        df = pd.read_csv(path)
        df["ts"] = pd.to_datetime(df[src["timestamp_col"]])

        field_map = src["fields"]
        if len(field_map) != 1:
            raise ValueError(f"{name}: expected 1 field mapping")

        field_name, col = list(field_map.items())[0]
        df = df[["ts", col]].rename(columns={col: field_name})
        signal_map[name] = df

    return signal_map


def _episode_groups(ep: Mapping[str, Any]) -> List[str]:
    out: List[str] = []
    for g in ep.get("groups", []):
        name = str(g.get("group") or "").strip()
        if name:
            out.append(name)
    return out


def _episode_models(ep: Mapping[str, Any]) -> List[str]:
    out: List[str] = []
    for m in ep.get("models", []):
        name = str(m.get("model") or "").strip()
        if name:
            out.append(name)
    return out


def _top_flags(result: Mapping[str, Any]) -> Tuple[List[str], List[str]]:
    summary = result.get("summary") or {}
    strong = [str(x) for x in (summary.get("strong_flags") or [])]
    moderate = [str(x) for x in (summary.get("moderate_flags") or [])]
    return strong, moderate


def _cross_pairs_for_config(config: Mapping[str, Any]) -> List[Tuple[str, str]]:
    source_names = [str(s["name"]) for s in config["data"]["sources"]]
    pairs: List[Tuple[str, str]] = []

    preferred = [
        ("demand", "net_generation"),
        ("demand", "imbalance"),
        ("net_generation", "imbalance"),
    ]
    for a, b in preferred:
        if a in source_names and b in source_names:
            pairs.append((a, b))
    return pairs


def _flatten_row(ep: Mapping[str, Any], result: Mapping[str, Any]) -> Dict[str, Any]:
    strong_flags, moderate_flags = _top_flags(result)
    summary = result.get("summary") or {}

    return {
        "episode_id": int(ep["episode_id"]),
        "ts_start": str(ep["ts_start"]),
        "ts_end": str(ep["ts_end"]),
        "length_steps": int(ep.get("length_steps") or 0),
        "length_minutes": float(ep.get("length_minutes") or 0.0),
        "gt_matched": bool(ep.get("gt_matched")),
        "explanatory_matched": bool(ep.get("explanatory_matched")),
        "peak_system_hot_count": float(ep.get("peak_system_hot_count") or 0.0),
        "peak_system_hot_streak": float(ep.get("peak_system_hot_streak") or 0.0),
        "num_groups_active": int(ep.get("num_groups_active") or 0),
        "num_models_instant_active": int(ep.get("num_models_instant_active") or 0),
        "active_groups_union": json.dumps(ep.get("active_groups_union") or []),
        "active_models_instant_union": json.dumps(ep.get("active_models_instant_union") or []),
        "looks_structured_anomaly": bool(summary.get("looks_structured_anomaly")),
        "strong_flags": json.dumps(strong_flags),
        "moderate_flags": json.dumps(moderate_flags),
        "promote_explanatory": "",
        "justification": "",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--episode-details", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--baseline-days", type=int, default=7)
    ap.add_argument("--include-gt", action="store_true")
    ap.add_argument("--include-explanatory-matched", action="store_true")
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    episodes = json.load(open(args.episode_details))
    signal_map = _load_signal_map(config)
    cross_pairs = _cross_pairs_for_config(config)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    review_rows: List[Dict[str, Any]] = []
    json_rows: List[Dict[str, Any]] = []

    for ep in episodes:
        if not args.include_gt and bool(ep.get("gt_matched")):
            continue
        if not args.include_explanatory_matched and bool(ep.get("explanatory_matched")):
            continue

        start_ts = pd.Timestamp(ep["ts_start"])
        end_ts = pd.Timestamp(ep["ts_end"])

        result = validate_anomaly_window(
            signal_map,
            start_ts=start_ts,
            end_ts=end_ts,
            baseline_days=int(args.baseline_days),
            cross_signal_pairs=cross_pairs,
        )

        json_rows.append(
            {
                "episode": ep,
                "validation": result,
            }
        )
        review_rows.append(_flatten_row(ep, result))

    json_path = out_dir / "anomaly_candidate_review.json"
    csv_path = out_dir / "anomaly_candidate_review.csv"

    with open(json_path, "w") as f:
        json.dump(json_rows, f, indent=2)

    fieldnames = [
        "episode_id",
        "ts_start",
        "ts_end",
        "length_steps",
        "length_minutes",
        "gt_matched",
        "explanatory_matched",
        "peak_system_hot_count",
        "peak_system_hot_streak",
        "num_groups_active",
        "num_models_instant_active",
        "active_groups_union",
        "active_models_instant_union",
        "looks_structured_anomaly",
        "strong_flags",
        "moderate_flags",
        "promote_explanatory",
        "justification",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in review_rows:
            writer.writerow(row)

    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()