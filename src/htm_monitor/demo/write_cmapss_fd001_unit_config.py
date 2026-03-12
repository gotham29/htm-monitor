from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml


LABEL_KEY = "late_life_60"
THRESHOLD = 0.99
WINDOW_SIZE = 12
PER_MODEL_HITS = 2
K_MODELS = 2


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text())


def _sensor_min_max(csv_path: Path) -> tuple[float, float]:
    df = pd.read_csv(csv_path)
    if "value" not in df.columns:
        raise ValueError(f"{csv_path} missing required column 'value'")
    vals = pd.to_numeric(df["value"], errors="coerce").dropna()
    if vals.empty:
        raise ValueError(f"{csv_path} has no numeric values in 'value'")
    return float(vals.min()), float(vals.max())


def _feature_block_for_sensor(sensor: str, csv_path: Path, seed: int) -> Dict[str, Any]:
    min_val, max_val = _sensor_min_max(csv_path)
    if min_val == max_val:
        # Defensive tiny pad so RDSE span does not collapse.
        min_val -= 1e-6
        max_val += 1e-6

    return {
        "type": "float",
        "size": 2048,
        "activeBits": 40,
        "numBuckets": 80,
        "minVal": min_val,
        "maxVal": max_val,
        "seed": int(seed),
    }


def main() -> None:
    repo = _repo_root()

    unit_dir = repo / "data" / "cmapss_fd001" / "unit_demo_34"
    out_yaml = repo / "configs" / "cmapss_fd001_unit34_demo.auto.yaml"

    manifest_path = unit_dir / "manifest.json"
    gt_path = unit_dir / "gt_timestamps.json"

    manifest = _load_json(manifest_path)
    gt_payload = _load_json(gt_path)

    sensors = manifest.get("selected_sensors")
    if not isinstance(sensors, list) or not sensors:
        raise ValueError("manifest.json missing non-empty selected_sensors")

    label_ts = gt_payload.get(LABEL_KEY)
    if not isinstance(label_ts, list) or not all(isinstance(x, str) for x in label_ts):
        raise ValueError(f"gt_timestamps.json missing valid '{LABEL_KEY}' list[str]")

    if not label_ts:
        raise ValueError(f"gt_timestamps.json has empty '{LABEL_KEY}' list")

    # Demo-facing evaluation policy:
    # Treat CMAPSS "late life" as a regime-entry problem, not repeated onset events.
    # So we use the FIRST late-life timestamp as the GT onset for every source.
    onset_ts = [label_ts[0]]

    # Shared timestamp feature
    features: Dict[str, Any] = {
        "timestamp": {
            "type": "datetime",
            "format": "%Y-%m-%d %H:%M:%S",
            "timeOfDay": 21,
            "weekend": 0,
            "dayOfWeek": 0,
            "holiday": 0,
            "season": 0,
            "encode": False,
        }
    }

    models: Dict[str, Any] = {}
    sources: List[Dict[str, Any]] = []

    for i, sensor in enumerate(sensors):
        csv_path = unit_dir / f"{sensor}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing sensor CSV for selected sensor '{sensor}': {csv_path}")

        features[sensor] = _feature_block_for_sensor(sensor, csv_path, seed=42 + i)

        model_name = f"{sensor}_model"
        models[model_name] = {
            "sources": [sensor],
            "features": [sensor],
        }

        sources.append(
            {
                "name": sensor,
                "kind": "csv",
                "path": str(csv_path.relative_to(repo)),
                "timestamp_col": "timestamp",
                "timestamp_format": "%Y-%m-%d %H:%M:%S",
                "fields": {
                    sensor: "value",
                },
                "labels": {
                    "timestamps": list(onset_ts),
                },
            }
        )

    cfg: Dict[str, Any] = {
        "features": features,
        "models": models,
        "data": {
            "timebase": {
                "mode": "union",
                "on_missing": "hold_last",
            },
            "sources": sources,
        },
        "run": {
            "warmup_steps": 24,
            "learn_after_warmup": True,
        },
        "decision": {
            "score_key": "anomaly_probability",
            "threshold": THRESHOLD,
            "method": "kofn_window",
            "k": K_MODELS,
            "window": {
                "size": WINDOW_SIZE,
                "per_model_hits": PER_MODEL_HITS,
            },
        },
        "plot": {
            "enable": False,
            "step_pause_s": 0.01,
            "window": 300,
            "show_ground_truth": True,
            "show_warmup_span": True,
        },
        "ground_truth": {
            "system": {
                "method": "kofn_window",
                "k": K_MODELS,
                "window": {
                    "size": WINDOW_SIZE,
                    "per_source_hits": 1,
                },
            }
        },
    }

    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False))

    print("Wrote CMAPSS FD001 unit demo config:")
    print(f"  {out_yaml}")
    print(f"  unit_dir={unit_dir}")
    print(f"  label_key={LABEL_KEY}")
    print(f"  sensors={sensors}")
    print(f"  gt_count_raw={len(label_ts)}")
    print(f"  gt_count_written={len(onset_ts)}")
    print(f"  onset_timestamp={onset_ts[0]}")


if __name__ == "__main__":
    main()
