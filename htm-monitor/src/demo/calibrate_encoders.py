from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import yaml


def _load_yaml(path: str) -> Dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text())
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML must be a mapping at top-level: {path}")
    return raw


def _write_yaml(path: str, obj: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(obj, sort_keys=False), encoding="utf-8")


def calibrate_from_config(
    *,
    defaults_path: str,
    config_path: str,
    low_q: float,
    high_q: float,
    margin_frac: float,
    max_rows: Optional[int] = None,
    override: bool = True,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Minimal calibration stub to unblock demo.sweep_thresholds.

    What it does right now:
    - Loads the user config YAML and ensures a top-level `calibration` block exists.
    - Writes the requested low/high/margin into that block.
    - Returns (updated_config_dict, report_lines).

    NOTE:
    - This does NOT compute encoder min/max from data. It only persists the calibration
      parameters so downstream build_from_config can proceed.
    """
    cfg = _load_yaml(config_path)

    cal = cfg.get("calibration") or {}
    if not isinstance(cal, dict):
        cal = {}

    cal["low_q"] = float(low_q)
    cal["high_q"] = float(high_q)
    cal["margin_frac"] = float(margin_frac)

    cfg["calibration"] = cal

    report = [
        "[calibrate_encoders] NOTE: minimal stub calibration (no data-driven min/max).",
        f"[calibrate_encoders] defaults_path={defaults_path}",
        f"[calibrate_encoders] config_path={config_path}",
        f"[calibrate_encoders] low_q={low_q} high_q={high_q} margin_frac={margin_frac}",
    ]

    # Caller is already writing out_cfg elsewhere, but keeping this harmless and explicit
    # is useful if someone runs this module directly later.
    return cfg, report