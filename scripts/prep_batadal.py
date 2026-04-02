#!/usr/bin/env python3
"""
BATADAL C-Town data preparation script
=======================================
Converts the three raw BATADAL CSV files into the per-signal CSV format
expected by the HTM-Monitor pipeline (configs/batadal_ctown.yaml).

Usage (from repo root):
    python scripts/prep_batadal.py

Or with explicit paths:
    python scripts/prep_batadal.py \\
        --train1 data/batadal_ctown/raw/BATADAL_dataset03.csv \\
        --train2 data/batadal_ctown/raw/BATADAL_dataset04.csv \\
        --test   data/batadal_ctown/raw/BATADAL_test_dataset.csv \\
        --out    data/batadal_ctown

Outputs
-------
For each of the 20 selected signals, creates:
    data/batadal_ctown/<SIGNAL>.csv  with columns: timestamp, value

Also writes:
    data/batadal_ctown/combined.csv  — full merged timeline (all 44 cols + iso timestamp)
    data/batadal_ctown/prep_summary.txt — row counts, date ranges, ATT_FLAG stats

Pipeline strategy
-----------------
Training Dataset 1 (8761 rows, Jan 2014 – Jan 2015): attack-free warmup/learning.
Training Dataset 2 (4177 rows, Jul – Dec 2016):       live evaluation, 7 attacks.
Test Dataset       (2089 rows, Jan – Apr 2017):        live evaluation, 7 attacks.

The 18-month gap between Dataset 1 and Dataset 2 is preserved in the timeline.
HTM models will re-adapt after the gap; this is acceptable because the config
sets warmup_steps=8761 (all of Dataset 1), so the decision layer only activates
from row 8762 onward — well past any re-adaptation transient.

The ATT_FLAG column in the combined output:
    Dataset 1:  0  (confirmed clean)
    Dataset 2: -1  (attacks present but originally unlabeled; kept as -1)
    Test set:   0 / 1 (clean / attack, per the competition labels)
"""

import argparse
import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# ---------------------------------------------------------------------------
# Signals to extract (20 HTM models, 3 groups)
# ---------------------------------------------------------------------------

SIGNALS: List[str] = [
    # gr_tanks
    "L_T1", "L_T2", "L_T3", "L_T4", "L_T5", "L_T6", "L_T7",
    # gr_pumps
    "F_PU1", "F_PU2", "F_PU4", "F_PU7", "F_PU8", "F_PU10", "F_V2",
    # gr_pressure
    "P_J269", "P_J256", "P_J306", "P_J317", "P_J302", "P_J415",
]

# ---------------------------------------------------------------------------
# Default paths (relative to repo root)
# ---------------------------------------------------------------------------

DEFAULT_TRAIN1 = "data/batadal_ctown/raw/BATADAL_dataset03.csv"
DEFAULT_TRAIN2 = "data/batadal_ctown/raw/BATADAL_dataset04.csv"
DEFAULT_TEST   = "data/batadal_ctown/raw/BATADAL_test_dataset.csv"
DEFAULT_OUT    = "data/batadal_ctown"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_datetime(s: str) -> datetime:
    """Parse BATADAL datetime string 'DD/MM/YY HH' -> datetime."""
    s = s.strip()
    return datetime.strptime(s, "%d/%m/%y %H")


def to_iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def normalise_row(row: Dict[str, str]) -> Dict[str, str]:
    """Strip leading/trailing spaces from all keys and values."""
    return {k.strip(): v.strip() for k, v in row.items()}


def load_csv(path: str, att_flag_missing_value: str = "0") -> List[Dict[str, str]]:
    """Load a BATADAL CSV, normalise column names, fill missing ATT_FLAG."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = [normalise_row(r) for r in reader]
    for row in rows:
        if "ATT_FLAG" not in row:
            row["ATT_FLAG"] = att_flag_missing_value
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(train1_path: str, train2_path: str, test_path: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[prep] Loading training dataset 1: {train1_path}")
    train1 = load_csv(train1_path, att_flag_missing_value="0")

    print(f"[prep] Loading training dataset 2: {train2_path}")
    # Dataset 2 has ATT_FLAG=-999 (partially labeled) — recode as -1 for clarity
    train2 = load_csv(train2_path, att_flag_missing_value="-1")
    for row in train2:
        if row.get("ATT_FLAG") in ("-999", ""):
            row["ATT_FLAG"] = "-1"

    print(f"[prep] Loading test dataset:        {test_path}")
    # Test dataset has no ATT_FLAG; ground truth is supplied separately via config
    test = load_csv(test_path, att_flag_missing_value="0")

    # Verify no temporal overlap or reversal between files
    t1_end   = parse_datetime(train1[-1]["DATETIME"])
    t2_start = parse_datetime(train2[0]["DATETIME"])
    t2_end   = parse_datetime(train2[-1]["DATETIME"])
    t3_start = parse_datetime(test[0]["DATETIME"])

    gap1 = (t2_start - t1_end).days
    gap2 = (t3_start - t2_end).days
    print(f"[prep] Timeline: {to_iso(parse_datetime(train1[0]['DATETIME']))} → "
          f"{to_iso(t1_end)} | gap {gap1}d | "
          f"{to_iso(t2_start)} → {to_iso(t2_end)} | gap {gap2}d | "
          f"{to_iso(t3_start)} → {to_iso(parse_datetime(test[-1]['DATETIME']))}")

    all_rows = train1 + train2 + test
    total = len(all_rows)
    print(f"[prep] Total rows: {total:,}  "
          f"(train1={len(train1)}, train2={len(train2)}, test={len(test)})")

    # Convert timestamps and build combined list
    combined: List[Dict] = []
    for raw in all_rows:
        dt  = parse_datetime(raw["DATETIME"])
        iso = to_iso(dt)
        row: Dict[str, str] = {"timestamp": iso}
        for sig in SIGNALS:
            row[sig] = raw.get(sig, "")
        row["ATT_FLAG"] = raw.get("ATT_FLAG", "0")
        combined.append(row)

    # ── Write per-signal CSVs ──────────────────────────────────────
    print(f"[prep] Writing per-signal CSVs to: {out_dir}/")
    for sig in SIGNALS:
        sig_path = out / f"{sig}.csv"
        with open(sig_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "value"])
            for row in combined:
                writer.writerow([row["timestamp"], row[sig]])
        print(f"  Wrote {sig}.csv  ({total:,} rows)")

    # ── Write combined CSV ─────────────────────────────────────────
    combined_path = out / "combined.csv"
    fieldnames = ["timestamp"] + SIGNALS + ["ATT_FLAG"]
    with open(combined_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combined)
    print(f"[prep] Wrote combined.csv ({total:,} rows, {len(fieldnames)} columns)")

    # ── Summary ────────────────────────────────────────────────────
    n_attack = sum(1 for r in combined if r["ATT_FLAG"] == "1")
    n_normal = sum(1 for r in combined if r["ATT_FLAG"] == "0")
    n_unlabeled = sum(1 for r in combined if r["ATT_FLAG"] == "-1")
    summary = (
        f"BATADAL C-Town Prep Summary\n"
        f"===========================\n"
        f"Train1:    {len(train1):,} rows  "
        f"{to_iso(parse_datetime(train1[0]['DATETIME']))} – {to_iso(t1_end)}\n"
        f"Train2:    {len(train2):,} rows  "
        f"{to_iso(t2_start)} – {to_iso(t2_end)}  (gap from train1: {gap1} days)\n"
        f"Test:      {len(test):,} rows   "
        f"{to_iso(t3_start)} – {to_iso(parse_datetime(test[-1]['DATETIME']))}  (gap from train2: {gap2} days)\n"
        f"Total:     {total:,} rows\n"
        f"\n"
        f"ATT_FLAG=0  (clean):      {n_normal:,}\n"
        f"ATT_FLAG=1  (attack):     {n_attack:,}\n"
        f"ATT_FLAG=-1 (unlabeled):  {n_unlabeled:,}\n"
        f"\n"
        f"Signals extracted: {len(SIGNALS)}\n"
        f"  Tanks    (gr_tanks):    L_T1 – L_T7\n"
        f"  Pumps    (gr_pumps):    F_PU1/2/4/7/8/10, F_V2\n"
        f"  Pressure (gr_pressure): P_J269/256/306/317/302/415\n"
        f"\n"
        f"Pipeline warmup_steps = 8761 (all of Train1)\n"
        f"Live evaluation starts at row 8762 (Train2 onset).\n"
    )
    summary_path = out / "prep_summary.txt"
    summary_path.write_text(summary)
    print(f"[prep] Wrote prep_summary.txt")
    print()
    print(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare BATADAL data for HTM-Monitor")
    parser.add_argument("--train1", default=DEFAULT_TRAIN1)
    parser.add_argument("--train2", default=DEFAULT_TRAIN2)
    parser.add_argument("--test",   default=DEFAULT_TEST)
    parser.add_argument("--out",    default=DEFAULT_OUT)
    args = parser.parse_args()
    main(args.train1, args.train2, args.test, args.out)
