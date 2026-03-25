from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


CLASS_KEYS = ["in_warning_window", "too_early", "too_late", "no_alert", "missing"]


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON: {path}")
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return obj


def _safe_int(x: Any, default: int = 0) -> int:
    if x is None:
        return default
    return int(x)


def _safe_float(x: Any, default: float | None = None) -> float | None:
    if x is None:
        return default
    return float(x)


def _get_nested(d: Dict[str, Any], *keys: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _dataset_code(dataset_label: str) -> str:
    parts = str(dataset_label).strip().split()
    return parts[-1] if parts else dataset_label


def _row_from_summary(path: Path) -> Dict[str, Any]:
    s = _load_json(path)

    dataset_label = str(s.get("dataset", "")).strip()
    if not dataset_label:
        raise ValueError(f"{path} missing 'dataset'")

    counts = s.get("first_alert_classification_counts") or {}
    lead = s.get("in_window_first_alert_lead_steps_stats") or {}
    elig = s.get("eligibility") or {}
    episode_stats = s.get("episode_count_stats") or {}
    alert_rate_stats = s.get("alert_rate_stats") or {}
    warning = s.get("warning_window") or {}

    row: Dict[str, Any] = {
        "dataset": _dataset_code(dataset_label),
        "dataset_label": dataset_label,
        "source_json": str(path),
        "unit_count": _safe_int(s.get("unit_count")),
        "eligible_unit_count": _safe_int(elig.get("eligible_unit_count")),
        "ineligible_unit_count": _safe_int(elig.get("ineligible_unit_count")),
        "eligible_rate": _safe_float(elig.get("eligible_rate")),
        "warning_start": _safe_int(warning.get("start_steps_before_event")),
        "warning_end": _safe_int(warning.get("end_steps_before_event")),
        "matched_in_window_units": _safe_int(s.get("matched_in_window_units")),
        "matched_in_window_rate": _safe_float(s.get("matched_in_window_rate")),
        "episode_count_mean": _safe_float(episode_stats.get("mean")),
        "episode_count_median": _safe_float(episode_stats.get("median")),
        "alert_rate_mean": _safe_float(alert_rate_stats.get("mean")),
        "alert_rate_median": _safe_float(alert_rate_stats.get("median")),
        "in_window_lead_count": _safe_int(lead.get("count")),
        "in_window_lead_mean": _safe_float(lead.get("mean")),
        "in_window_lead_median": _safe_float(lead.get("median")),
        "in_window_lead_p90": _safe_float(lead.get("p90")),
    }

    for k in CLASS_KEYS:
        row[f"class_count__{k}"] = _safe_int(counts.get(k))
        row[f"class_rate__{k}"] = (
            float(row[f"class_count__{k}"]) / float(row["unit_count"])
            if row["unit_count"] > 0
            else None
        )

    eligible_n = row["eligible_unit_count"]
    for k in CLASS_KEYS:
        c = row[f"class_count__{k}"]
        row[f"class_rate_eligible_denom__{k}"] = (
            float(c) / float(eligible_n) if eligible_n > 0 else None
        )

    return row


def _weighted_mean(rows: List[Dict[str, Any]], value_key: str, weight_key: str) -> float | None:
    num = 0.0
    den = 0.0
    for r in rows:
        v = r.get(value_key)
        w = r.get(weight_key)
        if v is None or w is None:
            continue
        if float(w) <= 0:
            continue
        num += float(v) * float(w)
        den += float(w)
    return (num / den) if den > 0 else None


def _total_row(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_units = sum(int(r["unit_count"]) for r in rows)
    total_eligible = sum(int(r["eligible_unit_count"]) for r in rows)
    total_ineligible = sum(int(r["ineligible_unit_count"]) for r in rows)
    total_matched = sum(int(r["matched_in_window_units"]) for r in rows)

    out: Dict[str, Any] = {
        "dataset": "TOTAL",
        "dataset_label": "NASA CMAPSS TOTAL",
        "source_json": "",
        "unit_count": total_units,
        "eligible_unit_count": total_eligible,
        "ineligible_unit_count": total_ineligible,
        "eligible_rate": (float(total_eligible) / float(total_units)) if total_units > 0 else None,
        "warning_start": rows[0]["warning_start"] if rows else None,
        "warning_end": rows[0]["warning_end"] if rows else None,
        "matched_in_window_units": total_matched,
        "matched_in_window_rate": (float(total_matched) / float(total_units)) if total_units > 0 else None,
        "episode_count_mean": _weighted_mean(rows, "episode_count_mean", "unit_count"),
        "episode_count_median": None,
        "alert_rate_mean": _weighted_mean(rows, "alert_rate_mean", "unit_count"),
        "alert_rate_median": None,
        "in_window_lead_count": sum(int(r["in_window_lead_count"]) for r in rows),
        "in_window_lead_mean": _weighted_mean(rows, "in_window_lead_mean", "in_window_lead_count"),
        "in_window_lead_median": None,
        "in_window_lead_p90": None,
    }

    for k in CLASS_KEYS:
        c = sum(int(r[f"class_count__{k}"]) for r in rows)
        out[f"class_count__{k}"] = c
        out[f"class_rate__{k}"] = (float(c) / float(total_units)) if total_units > 0 else None
        out[f"class_rate_eligible_denom__{k}"] = (float(c) / float(total_eligible)) if total_eligible > 0 else None

    return out


def _pct(x: Any) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{100.0 * float(x):.1f}%"


def _num(x: Any, digits: int = 3) -> str:
    if x is None or pd.isna(x):
        return "—"
    if float(x).is_integer():
        return str(int(float(x)))
    return f"{float(x):.{digits}f}"


def _markdown_table(df: pd.DataFrame) -> str:
    cols = [
        "dataset",
        "unit_count",
        "eligible_unit_count",
        "eligible_rate",
        "class_count__in_warning_window",
        "class_count__too_early",
        "class_count__too_late",
        "class_count__no_alert",
        "matched_in_window_units",
        "matched_in_window_rate",
        "in_window_lead_median",
        "in_window_lead_p90",
        "alert_rate_mean",
        "episode_count_mean",
    ]

    lines: List[str] = []
    lines.append("# CMAPSS HTM baseline summary\n\n")
    lines.append("| Dataset | Units | Eligible | Eligible % | In-window | Too-early | Too-late | No-alert | Matched in-window | Matched % | Median lead | P90 lead | Mean alert rate | Mean episode count |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")

    for _, r in df.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(r["dataset"]),
                    str(int(r["unit_count"])),
                    str(int(r["eligible_unit_count"])),
                    _pct(r["eligible_rate"]),
                    str(int(r["class_count__in_warning_window"])),
                    str(int(r["class_count__too_early"])),
                    str(int(r["class_count__too_late"])),
                    str(int(r["class_count__no_alert"])),
                    str(int(r["matched_in_window_units"])),
                    _pct(r["matched_in_window_rate"]),
                    _num(r["in_window_lead_median"], 1),
                    _num(r["in_window_lead_p90"], 1),
                    _num(r["alert_rate_mean"], 3),
                    _num(r["episode_count_mean"], 3),
                ]
            )
            + " |\n"
        )

    return "".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fleet-summary-jsons",
        nargs="+",
        required=True,
        help="One or more fleet_summary.json files",
    )
    ap.add_argument(
        "--out-dir",
        default="outputs/cmapss_baseline_summary",
        help="Directory to write CSV and Markdown outputs",
    )
    args = ap.parse_args()

    json_paths = [Path(x) for x in args.fleet_summary_jsons]
    rows = [_row_from_summary(p) for p in json_paths]

    order = {"FD001": 1, "FD002": 2, "FD003": 3, "FD004": 4}
    rows = sorted(rows, key=lambda r: order.get(str(r["dataset"]), 999))
    rows.append(_total_row(rows))

    df = pd.DataFrame(rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "cmapss_htm_baseline_summary.csv"
    md_path = out_dir / "cmapss_htm_baseline_summary.md"

    df.to_csv(csv_path, index=False)
    md_path.write_text(_markdown_table(df))

    print(f"wrote: {csv_path}")
    print(f"wrote: {md_path}")


if __name__ == "__main__":
    main()
