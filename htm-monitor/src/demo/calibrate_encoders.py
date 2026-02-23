# src/demo/calibrate_encoders.py

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional, Mapping

import pandas as pd
import yaml

from htm_monitor.utils.config import load_yaml, merge_dicts


# ----------------------------
# Core calibration logic
# ----------------------------

def compute_min_max_from_series(
    series: pd.Series,
    low_q: float = 0.01,
    high_q: float = 0.99,
    margin_frac: float = 0.03,
) -> Tuple[float, float]:
    """
    Compute robust min/max from a numeric pandas Series.

    Strategy:
        - Use quantiles [low_q, high_q]
        - Expand by margin_frac * span
    """

    s = pd.to_numeric(series, errors="coerce").dropna()

    if len(s) == 0:
        raise ValueError("Cannot calibrate: series contains no numeric data")

    q_low = float(s.quantile(low_q))
    q_high = float(s.quantile(high_q))

    if q_high < q_low:
        raise ValueError("Invalid quantile ordering")

    span = q_high - q_low

    # Handle degenerate constant signal
    if span == 0:
        eps = max(abs(q_low) * 0.01, 1e-6)
        return q_low - eps, q_high + eps

    margin = span * margin_frac
    return q_low - margin, q_high + margin


def calibrate_from_csv(
    csv_path: Path,
    value_column: str,
    low_q: float = 0.01,
    high_q: float = 0.99,
    margin_frac: float = 0.03,
    max_rows: int = None,
) -> Dict[str, float]:
    """
    Calibrate a single feature from a CSV file.
    Only uses the first source that defines the feature.
    """

    df = pd.read_csv(csv_path)

    if value_column not in df.columns:
        raise ValueError(f"Column '{value_column}' not found in {csv_path}")

    if max_rows is not None:
        df = df.iloc[:max_rows]

    min_val, max_val = compute_min_max_from_series(
        df[value_column],
        low_q=low_q,
        high_q=high_q,
        margin_frac=margin_frac,
    )

    return {
        "minVal": float(min_val),
        "maxVal": float(max_val),
    }


def _is_datetime_feature(fcfg: Mapping[str, Any]) -> bool:
    t = str(fcfg.get("type", "")).lower()
    return t in ("datetime", "date", "time")


def _is_numeric_feature(fcfg: Mapping[str, Any]) -> bool:
    t = str(fcfg.get("type", "")).lower()
    return t in ("float", "int", "numeric", "number")


def _first_source_for_feature(cfg: Mapping[str, Any], feature_name: str) -> Optional[Tuple[str, str]]:
    """
    Return (csv_path, column_name) for the FIRST source that defines this feature.
    Uses cfg["data"]["sources"][*]["fields"] mapping canonical -> column.
    """
    data = cfg.get("data") or {}
    sources = data.get("sources") or []
    if not isinstance(sources, list):
        return None

    for s in sources:
        if not isinstance(s, dict):
            continue
        fields = s.get("fields") or {}
        if not isinstance(fields, dict):
            continue
        if feature_name not in fields:
            continue
        csv_path = s.get("path")
        col = fields.get(feature_name)
        if isinstance(csv_path, str) and csv_path and isinstance(col, str) and col:
            return csv_path, col
    return None


def _safe_read_csv_column(csv_path: str, column: str, max_rows: Optional[int]) -> pd.Series:
    df = pd.read_csv(csv_path)
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in {csv_path}")
    if max_rows is not None:
        df = df.iloc[:max_rows]
    return df[column]


def calibrate_from_config(
    defaults_path: str,
    config_path: str,
    low_q: float = 0.01,
    high_q: float = 0.99,
    margin_frac: float = 0.03,
    max_rows: Optional[int] = None,
    override: bool = False,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Load defaults + config, merge, calibrate numeric features using FIRST defining source.
    Returns (updated_cfg, report_lines).
    """
    defaults = load_yaml(defaults_path)
    user = load_yaml(config_path)
    cfg = merge_dicts(defaults, user)

    feats = cfg.get("features") or {}
    if not isinstance(feats, dict):
        raise ValueError("cfg['features'] must be a mapping")

    report: List[str] = []
    report.append("=== Encoder Calibration Report ===")
    report.append(f"defaults: {defaults_path}")
    report.append(f"config:   {config_path}")
    report.append(f"quantiles: low={low_q} high={high_q}")
    report.append(f"margin_frac: {margin_frac}")
    report.append(f"max_rows: {max_rows}")
    report.append(f"override: {override}")
    report.append("")

    # Work on a deep-ish copy of cfg; we only mutate features entries.
    out_cfg: Dict[str, Any] = dict(cfg)
    out_features: Dict[str, Any] = dict(out_cfg.get("features") or {})
    out_cfg["features"] = out_features

    for fname, fcfg_any in feats.items():
        if not isinstance(fname, str) or not fname:
            continue
        if not isinstance(fcfg_any, dict):
            continue

        if fname == "timestamp" or _is_datetime_feature(fcfg_any):
            continue
        if not _is_numeric_feature(fcfg_any):
            continue

        # Only calibrate if missing min/max unless override.
        cur_min = fcfg_any.get("minVal")
        cur_max = fcfg_any.get("maxVal")
        if not override and cur_min is not None and cur_max is not None:
            report.append(f"feature: {fname}")
            report.append("  action: KEEP (minVal/maxVal already set)")
            report.append(f"  current: minVal={cur_min} maxVal={cur_max}")
            report.append("")
            continue

        loc = _first_source_for_feature(cfg, fname)
        if loc is None:
            report.append(f"feature: {fname}")
            report.append("  action: SKIP (no source defines this feature in data.sources[*].fields)")
            report.append("")
            continue

        csv_path, col = loc
        try:
            series = _safe_read_csv_column(csv_path, col, max_rows=max_rows)
            # capture quantiles for report transparency
            s_num = pd.to_numeric(series, errors="coerce").dropna()
            if len(s_num) == 0:
                raise ValueError("no numeric samples")
            qlow = float(s_num.quantile(low_q))
            qhigh = float(s_num.quantile(high_q))
            min_val, max_val = compute_min_max_from_series(
                series,
                low_q=low_q,
                high_q=high_q,
                margin_frac=margin_frac,
            )
        except Exception as e:
            report.append(f"feature: {fname}")
            report.append(f"  source: {csv_path} col={col}")
            report.append(f"  action: FAIL ({e})")
            report.append("")
            continue

        new_fcfg = dict(fcfg_any)
        if override or cur_min is None:
            new_fcfg["minVal"] = float(min_val)
        if override or cur_max is None:
            new_fcfg["maxVal"] = float(max_val)
        out_features[fname] = new_fcfg

        report.append(f"feature: {fname}")
        report.append(f"  source: {csv_path} col={col}")
        report.append(f"  q_low={qlow} q_high={qhigh}")
        report.append(f"  set: minVal={new_fcfg.get('minVal')} maxVal={new_fcfg.get('maxVal')}")
        report.append("")

    return out_cfg, report


# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Calibrate RDSE min/max from CSV or from a use-case config")

    # Mode A: single CSV
    parser.add_argument("--csv", default=None, help="Path to CSV file")
    parser.add_argument("--column", default=None, help="Numeric column name")
    parser.add_argument("--write-yaml", type=str, default=None,
                        help="(CSV mode) Optional path to write a tiny minVal/maxVal YAML")

    # Mode B: config-driven
    parser.add_argument("--defaults", default=None, help="Defaults YAML path (config mode)")
    parser.add_argument("--config", default=None, help="Use-case YAML path (config mode)")
    parser.add_argument("--out-config", default=None, help="Write calibrated merged config YAML here (config mode)")
    parser.add_argument("--report", default=None, help="Write calibration report text here (config mode)")
    parser.add_argument("--override", action="store_true",
                        help="Override existing minVal/maxVal (config mode). Default: fill missing only.")

    # Shared calibration params
    parser.add_argument("--low", type=float, default=0.01, help="Lower quantile")
    parser.add_argument("--high", type=float, default=0.99, help="Upper quantile")
    parser.add_argument("--margin", type=float, default=0.03, help="Margin fraction")
    parser.add_argument("--max-rows", type=int, default=None)

    args = parser.parse_args()

    # Decide mode
    is_config_mode = args.config is not None or args.defaults is not None or args.out_config is not None

    if is_config_mode:
        if not args.defaults or not args.config:
            raise SystemExit("Config mode requires --defaults and --config")
        if not args.out_config:
            raise SystemExit("Config mode requires --out-config (where to write calibrated YAML)")

        out_cfg, report_lines = calibrate_from_config(
            defaults_path=args.defaults,
            config_path=args.config,
            low_q=args.low,
            high_q=args.high,
            margin_frac=args.margin,
            max_rows=args.max_rows,
            override=bool(args.override),
        )

        out_path = Path(args.out_config)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(out_cfg, f, sort_keys=False)
        print(f"[calibrate] wrote calibrated config -> {out_path}")

        if args.report:
            rep_path = Path(args.report)
            rep_path.parent.mkdir(parents=True, exist_ok=True)
            rep_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
            print(f"[calibrate] wrote report -> {rep_path}")
        else:
            # still print report to stdout for interactive use
            print("\n".join(report_lines))
        return

    # CSV mode
    if not args.csv or not args.column:
        raise SystemExit("CSV mode requires --csv and --column (or use config mode with --defaults/--config)")

    result = calibrate_from_csv(
        Path(args.csv),
        args.column,
        low_q=args.low,
        high_q=args.high,
        margin_frac=args.margin,
        max_rows=args.max_rows,
    )

    print("\n=== Calibration Result ===")
    print(f"minVal: {result['minVal']}")
    print(f"maxVal: {result['maxVal']}")

    if args.write_yaml:
        out = {
            "type": "float",
            "minVal": result["minVal"],
            "maxVal": result["maxVal"],
        }
        with open(args.write_yaml, "w", encoding="utf-8") as f:
            yaml.safe_dump(out, f, sort_keys=False)
        print(f"\nWrote YAML to {args.write_yaml}")


if __name__ == "__main__":
    main()