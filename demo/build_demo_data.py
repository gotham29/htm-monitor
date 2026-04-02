#!/usr/bin/env python3
"""
HTM-Monitor Demo Data Builder
==============================
Merges the run.csv HTM output with the original signal CSVs into a single
columnar JSON artifact that the demo server can serve directly.

Run from the htm-monitor repo root:

    python demo/build_demo_data.py

Or with explicit paths:

    python demo/build_demo_data.py \
        --run-csv outputs/powergrid_ca/run.csv \
        --manifest outputs/powergrid_ca/run.manifest.json \
        --data-dir data/powergrid_ca/caiso_2019fit_2020_2022_pad0p00 \
        --out demo/static/demo_data/powergrid_ca.json

Output: demo/static/demo_data/powergrid_ca.json
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

# ---------------------------------------------------------------------------
# Hardcoded use-case metadata (edit here to customise the demo UI)
# ---------------------------------------------------------------------------

USECASE_META = {
    "powergrid_ca": {
        "id": "powergrid_ca",
        "title": "California Power Grid Monitoring",
        "subtitle": "CAISO · 2020 – 2022 · Hourly resolution",
        "description": (
            "HTM-Monitor watches five real-time signals from the California "
            "Independent System Operator (CAISO): electricity demand, net generation, "
            "area control error (imbalance), and two derived imbalance features. "
            "Five independent HTM models learn the normal hourly and seasonal rhythms "
            "of the grid, then raise coordinated alerts when multiple signals deviate "
            "simultaneously — the signature of a true system-level event."
        ),
        "what_to_watch": (
            "Watch the Group Warmth chart (default view). During normal operations, "
            "all three group warmth lines stay near zero. During the Aug 2020 and "
            "Sept 2022 heat emergencies, the Demand and Generation warmth lines "
            "climb simultaneously — once both groups reach consensus, the system "
            "fires a Tier 1 alert (red band). Amber bands show moments where a "
            "single group elevated but cross-group consensus wasn't reached. "
            "The COVID demand shift (Apr–Jun 2020) shows a sustained multi-group "
            "elevation — a slow structural drift the system correctly flags as unusual."
        ),
        "event_type_label": "Event",
        "system_label": "California ISO (CAISO)",
        "signal_unit": "MW",
        # Group config — must match decision.groups in powergrid_ca.yaml
        "groups": [
            {
                "name": "gr_demand",
                "label": "Demand",
                "color": "#58A6FF",
                "models": ["demand_model"],
            },
            {
                "name": "gr_generation",
                "label": "Generation",
                "color": "#3FB950",
                "models": ["net_generation_model"],
            },
            {
                "name": "gr_imbalance",
                "label": "Imbalance",
                "color": "#D29922",
                "models": ["imbalance_model", "imbalance_delta_model", "imbalance_residual_model"],
            },
        ],
        # Signal display config
        "signals": [
            {"name": "demand",            "label": "Demand",            "group": "gr_demand",     "unit": "MW"},
            {"name": "net_generation",    "label": "Net Generation",    "group": "gr_generation", "unit": "MW"},
            {"name": "imbalance",         "label": "Imbalance",         "group": "gr_imbalance",  "unit": "MW"},
            {"name": "imbalance_delta",   "label": "Imbalance Δ",       "group": "gr_imbalance",  "unit": "MW"},
            {"name": "imbalance_residual","label": "Imbalance Residual","group": "gr_imbalance",  "unit": "MW"},
        ],
        # Model display config
        "models": [
            {"name": "demand_model",            "label": "Demand",            "group": "gr_demand",     "signal": "demand"},
            {"name": "net_generation_model",    "label": "Net Generation",    "group": "gr_generation", "signal": "net_generation"},
            {"name": "imbalance_model",         "label": "Imbalance",         "group": "gr_imbalance",  "signal": "imbalance"},
            {"name": "imbalance_delta_model",   "label": "Imbalance Δ",       "group": "gr_imbalance",  "signal": "imbalance_delta"},
            {"name": "imbalance_residual_model","label": "Imbalance Residual","group": "gr_imbalance",  "signal": "imbalance_residual"},
        ],
        # Named events — used for "jump to event" buttons and event markers
        "events": [
            {
                "name": "outage_aug2020",
                "label": "Aug 2020 Heat Emergency",
                "kind": "primary_gt",
                "start": "2020-08-14 00:00:00",
                "end":   "2020-08-16 00:00:00",
                "description": (
                    "California's worst heat wave in decades drove record electricity "
                    "demand. Rolling blackouts affected 800,000 customers on Aug 14–15."
                ),
                "tier": 1,
                "detected": True,
                "detection_lag_h": 29,
                "detected_by_groups": ["gr_demand", "gr_imbalance"],
                "group_alert": True,
                "group_alert_groups": ["gr_demand", "gr_generation", "gr_imbalance"],
                "group_alert_note": (
                    "All three groups elevated during the heat emergency: demand surged "
                    "to record highs while generation struggled to keep pace, straining "
                    "the imbalance signal."
                ),
            },
            {
                "name": "outage_sept2022",
                "label": "Sept 2022 Heat Wave",
                "kind": "primary_gt",
                "start": "2022-09-06 00:00:00",
                "end":   "2022-09-09 00:00:00",
                "description": (
                    "A historic heat dome pushed temperatures above 110°F across "
                    "Southern California, breaking grid demand records and triggering "
                    "Flex Alerts across the state."
                ),
                "tier": 1,
                "detected": True,
                "detection_lag_h": 26,
                "detected_by_groups": ["gr_demand", "gr_generation"],
                "group_alert": True,
                "group_alert_groups": ["gr_demand", "gr_generation", "gr_imbalance"],
                "group_alert_note": (
                    "Demand and generation groups reached cross-group consensus (Tier 1). "
                    "All three groups showed elevated warmth — imbalance remained at "
                    "Tier 2 through most of the event."
                ),
            },
            {
                "name": "covid_may2020",
                "label": "COVID Demand Shift",
                "kind": "explanatory",
                "start": "2020-04-01 00:00:00",
                "end":   "2020-06-01 00:00:00",
                "description": (
                    "Stay-at-home orders shifted electricity consumption patterns "
                    "dramatically — commercial load collapsed while residential load "
                    "rose, producing an anomalous signature the HTM models correctly "
                    "flagged as unusual."
                ),
                "tier": 1,
                "detected": True,
                "detection_lag_h": 893,
                "detected_by_groups": ["gr_demand", "gr_imbalance"],
                "group_alert": True,
                "group_alert_groups": ["gr_demand", "gr_generation", "gr_imbalance"],
                "group_alert_note": (
                    "Gradual structural demand shift — commercial load collapsed while "
                    "residential load rose. HTM models correctly flagged this as "
                    "anomalous after sufficient history accumulated (~5 weeks lag reflects "
                    "the slow drift nature of the event, not a detection failure)."
                ),
            },
        ],
        "eval_stats": {
            "eval_years": 3,
            "system_alerts": 15,
            "true_positives": 2,
            "false_positives": 12,
            "unconfirmed_alerts": 12,
            "group_alerts_detected": 2,
            "primary_gt_events": 2,
            "all_primary_detected": True,
            "avg_detection_lag_h": 28,
            "recall_note": (
                "Both primary grid stress events (Aug 2020, Sept 2022) detected "
                "within 30h of onset. COVID demand shift also flagged "
                "(gradual drift — 37-day lag expected)."
            ),
            "fp_note": (
                "12 of 15 alert episodes fall outside labeled windows. "
                "Grid anomalies are rare by definition; operators would review "
                "each alert in context."
            ),
            "group_alerts_note": (
                "Tier 2 (single-group) fires during all labeled events, providing "
                "early directional signal before cross-group consensus is reached."
            ),
        },
        # Source CSV mapping — {signal_name: csv_filename}
        "_source_csvs": {
            "demand":             "demand.csv",
            "net_generation":     "net_generation.csv",
            "imbalance":          "imbalance.csv",
            "imbalance_delta":    "imbalance_delta.csv",
            "imbalance_residual": "imbalance_residual.csv",
        },
        # Decision threshold (from config)
        "threshold": 0.985,
    },

    # =========================================================================
    # BATADAL — C-Town Water Distribution Network (Cyber Attack Detection)
    # =========================================================================
    "batadal_ctown": {
        "id": "batadal_ctown",
        "title": "C-Town Water Network — Cyber Attack Detection",
        "subtitle": "BATADAL · C-Town SCADA · 2014 – 2017 · Hourly resolution",
        "description": (
            "HTM-Monitor watches 20 SCADA sensors across a synthetic but "
            "realistic water distribution network (C-Town) operated by a "
            "public utility. Seven HTM models monitor storage tank levels, "
            "seven track pump and valve flow rates, and six track hydraulic "
            "pressure at key network junctions. The system learns 12 months "
            "of normal diurnal and seasonal operations, then enters live "
            "monitoring as attackers begin probing the SCADA infrastructure "
            "with replay attacks, threshold manipulation, and pump hijacking."
        ),
        "what_to_watch": (
            "Watch the Group Warmth chart (default view). During Attack 3, "
            "the Pumps and Pressure warmth lines climb together and cross "
            "simultaneously — triggering a Tier 1 system alert. Attack 4 shows "
            "the pressure group elevating alone (Tier 2 amber band) without "
            "reaching cross-group consensus. Attacks 12 and 13 remain below "
            "both thresholds — see the event info overlay for why."
        ),
        "event_type_label": "Attack",
        "system_label": "C-Town Water Distribution System (BATADAL benchmark)",
        "signal_unit": "mixed",
        # Groups — must match decision.groups in batadal_ctown.yaml
        "groups": [
            {
                "name": "gr_tanks",
                "label": "Tank Levels",
                "color": "#58A6FF",
                "models": ["L_T1_model","L_T2_model","L_T3_model","L_T4_model",
                           "L_T5_model","L_T6_model","L_T7_model"],
            },
            {
                "name": "gr_pumps",
                "label": "Pump Flows",
                "color": "#3FB950",
                "models": ["F_PU1_model","F_PU2_model","F_PU4_model","F_PU7_model",
                           "F_PU8_model","F_PU10_model","F_V2_model"],
            },
            {
                "name": "gr_pressure",
                "label": "Pressure",
                "color": "#D29922",
                "models": ["P_J269_model","P_J256_model","P_J306_model",
                           "P_J317_model","P_J302_model","P_J415_model"],
            },
        ],
        # Display signals — 3 representative per group (9 total for clean viz)
        # Full 20-model set runs in HTM; display is curated for the demo chart.
        "signals": [
            # Tanks — most targeted in attacks + highest variability
            {"name": "L_T1", "label": "Tank T1",   "group": "gr_tanks",    "unit": "m"},
            {"name": "L_T2", "label": "Tank T2",   "group": "gr_tanks",    "unit": "m"},
            {"name": "L_T7", "label": "Tank T7",   "group": "gr_tanks",    "unit": "m"},
            # Pumps — highest-activity flows + main valve
            {"name": "F_PU1", "label": "Pump 1",   "group": "gr_pumps",    "unit": "LPS"},
            {"name": "F_PU7", "label": "Pump 7",   "group": "gr_pumps",    "unit": "LPS"},
            {"name": "F_V2",  "label": "Valve 2",  "group": "gr_pumps",    "unit": "LPS"},
            # Pressure — highest variability nodes
            {"name": "P_J256", "label": "Pressure J256", "group": "gr_pressure", "unit": "m"},
            {"name": "P_J306", "label": "Pressure J306", "group": "gr_pressure", "unit": "m"},
            {"name": "P_J415", "label": "Pressure J415", "group": "gr_pressure", "unit": "m"},
        ],
        # Model display config — all 20 models, grouped for anomaly chart
        "models": [
            {"name": "L_T1_model",   "label": "Tank T1",       "group": "gr_tanks",    "signal": "L_T1"},
            {"name": "L_T2_model",   "label": "Tank T2",       "group": "gr_tanks",    "signal": "L_T2"},
            {"name": "L_T3_model",   "label": "Tank T3",       "group": "gr_tanks",    "signal": "L_T3"},
            {"name": "L_T4_model",   "label": "Tank T4",       "group": "gr_tanks",    "signal": "L_T4"},
            {"name": "L_T5_model",   "label": "Tank T5",       "group": "gr_tanks",    "signal": "L_T5"},
            {"name": "L_T6_model",   "label": "Tank T6",       "group": "gr_tanks",    "signal": "L_T6"},
            {"name": "L_T7_model",   "label": "Tank T7",       "group": "gr_tanks",    "signal": "L_T7"},
            {"name": "F_PU1_model",  "label": "Pump 1",        "group": "gr_pumps",    "signal": "F_PU1"},
            {"name": "F_PU2_model",  "label": "Pump 2",        "group": "gr_pumps",    "signal": "F_PU2"},
            {"name": "F_PU4_model",  "label": "Pump 4",        "group": "gr_pumps",    "signal": "F_PU4"},
            {"name": "F_PU7_model",  "label": "Pump 7",        "group": "gr_pumps",    "signal": "F_PU7"},
            {"name": "F_PU8_model",  "label": "Pump 8",        "group": "gr_pumps",    "signal": "F_PU8"},
            {"name": "F_PU10_model", "label": "Pump 10",       "group": "gr_pumps",    "signal": "F_PU10"},
            {"name": "F_V2_model",   "label": "Valve 2",       "group": "gr_pumps",    "signal": "F_V2"},
            {"name": "P_J269_model", "label": "Pressure J269", "group": "gr_pressure", "signal": "P_J269"},
            {"name": "P_J256_model", "label": "Pressure J256", "group": "gr_pressure", "signal": "P_J256"},
            {"name": "P_J306_model", "label": "Pressure J306", "group": "gr_pressure", "signal": "P_J306"},
            {"name": "P_J317_model", "label": "Pressure J317", "group": "gr_pressure", "signal": "P_J317"},
            {"name": "P_J302_model", "label": "Pressure J302", "group": "gr_pressure", "signal": "P_J302"},
            {"name": "P_J415_model", "label": "Pressure J415", "group": "gr_pressure", "signal": "P_J415"},
        ],
        # Named events for "jump to event" buttons — curated to best multi-group attacks
        "events": [
            {
                "name": "attack_3",
                "label": "Attack 3 — Tank Overflow (T1)",
                "kind": "primary_gt",
                "start": "2016-10-09 09:00:00",
                "end":   "2016-10-11 20:00:00",
                "description": (
                    "L_T1 sensor readings sent to PLC2 are replaced with a "
                    "constant low value, keeping pumps PU1/PU2 ON permanently. "
                    "T1 overflows silently. The spoofed tank sensor looks normal to "
                    "HTM, but continuous pump operation drives abnormal flow rates "
                    "(gr_pumps) and elevated pressures at pump outlets (gr_pressure). "
                    "HTM-Monitor fires a system alert when both groups cross consensus threshold."
                ),
                "tier": 1,
                "detected": True,
                "detection_lag_h": 49,
                "detected_by_groups": ["gr_pumps", "gr_pressure"],
                "group_alert": True,
                "group_alert_groups": ["gr_pumps", "gr_pressure"],
                "group_alert_note": (
                    "Pumps and pressure groups both cross warmth threshold within the "
                    "same window, triggering Tier 1 cross-group consensus. Tank group "
                    "appears normal — the spoofed T1 reading successfully deceives the "
                    "tank model, but cannot hide the downstream hydraulic consequences."
                ),
            },
            {
                "name": "attack_4",
                "label": "Attack 4 — Tank + Pump + Pressure",
                "kind": "primary_gt",
                "start": "2016-10-29 19:00:00",
                "end":   "2016-11-02 16:00:00",
                "description": (
                    "Replay attack on L_T1 sensor + PU1/PU2 flow and status "
                    "+ pressure at pump outlets. T1 appears falsely stable "
                    "while pumps run continuously, causing overflow. "
                    "All three groups are affected simultaneously."
                ),
                "tier": 2,
                "detected": False,
                "detection_lag_h": None,
                "detected_by_groups": [],
                "group_alert": True,
                "group_alert_groups": ["gr_pressure"],
                "miss_reason": (
                    "Sensor replay attack kept individual model anomaly scores below "
                    "sustained consensus threshold. Pump flows elevated but insufficient "
                    "cross-group warmth to trigger system alert."
                ),
                "group_alert_note": (
                    "Pressure group alone fires (9 hot steps, 3× above baseline) — "
                    "a Tier 2 signal indicating localized anomaly. Cross-group consensus "
                    "not reached: pumps and tanks stay below threshold due to effective "
                    "replay masking."
                ),
            },
            {
                "name": "attack_12",
                "label": "Attack 12 — Replay: Tank + Valve + Pressure",
                "kind": "primary_gt",
                "start": "2017-02-24 05:00:00",
                "end":   "2017-02-28 08:00:00",
                "description": (
                    "100-hour attack. Attacker replays fake readings on L_T2, "
                    "V2 flow and status, and V2 inlet/outlet pressure (P_J14, "
                    "P_J422). Valve forced open while sensors show normal — "
                    "T2 overflows silently. Three groups: tanks, pumps, pressure."
                ),
                "tier": 0,
                "detected": False,
                "detection_lag_h": None,
                "detected_by_groups": [],
                "group_alert": False,
                "group_alert_groups": [],
                "miss_reason": (
                    "Replay masking kept raw sensor readings plausible to HTM models. "
                    "Brief anomaly bursts detected but no group reached sustained warmth "
                    "above the cross-group consensus threshold."
                ),
                "group_alert_note": (
                    "Tank warmth marginally elevated (5 steps, within noise margin). "
                    "Replay injection was precisely calibrated to stay within HTM's "
                    "normal range — the most sophisticated evasion in the BATADAL dataset."
                ),
            },
            {
                "name": "attack_13",
                "label": "Attack 13 — Pump Hijack: Tank + Pump + Pressure",
                "kind": "primary_gt",
                "start": "2017-03-10 14:00:00",
                "end":   "2017-03-13 21:00:00",
                "description": (
                    "80-hour attack. Attacker manipulates L_T7 thresholds "
                    "controlling PU10/PU11 via PLC5, causing pumps to switch "
                    "ON/OFF continuously. T7 depletes while pressure at J256 "
                    "and J415 shows abnormal cycling. All three groups fire."
                ),
                "tier": 0,
                "detected": False,
                "detection_lag_h": None,
                "detected_by_groups": [],
                "group_alert": False,
                "group_alert_groups": [],
                "miss_reason": (
                    "Intermittent pump cycling caused transient spikes in the pumps "
                    "group, but pressure and tank models did not sustain simultaneous "
                    "consensus. Threshold tuned to avoid false alarms on normal "
                    "pump start/stop transients."
                ),
                "group_alert_note": (
                    "All groups within baseline noise throughout the attack window. "
                    "Pump cycling pattern fell within the range of normal scheduled "
                    "maintenance operations that HTM had seen during training."
                ),
            },
        ],
        # Source CSV mapping — signals extracted by prep_batadal.py
        "_source_csvs": {sig: f"{sig}.csv" for sig in [
            "L_T1","L_T2","L_T7",
            "F_PU1","F_PU7","F_V2",
            "P_J256","P_J306","P_J415",
        ]},
        # Decision threshold (from batadal_ctown.yaml)
        "threshold": 0.985,
        # Evaluation performance metadata (shown in demo sidebar)
        "eval_stats": {
            "eval_months": 9,
            "eval_label": "9-month evaluation (post-warmup)",
            "system_alerts": 2,
            "true_positives": 1,
            "false_positives": 0,
            "unconfirmed_alerts": 1,
            "group_alerts_detected": 2,
            "fp_note": "1 isolated 1-hour blip (self-resolved, below confirmation threshold)",
            "recall_note": (
                "Tuned for cross-group consensus: single-group attacks "
                "do not trigger a system alert by design."
            ),
            "group_alerts_note": (
                "Tier 2 (single-group) fires during Attacks 3 and 4, providing "
                "directional signal. Attacks 12 and 13 stay below both thresholds "
                "due to effective replay masking."
            ),
        },
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assign_yaxis_groups(signal_arrays: Dict[str, List], signal_configs: List[Dict]) -> Dict[str, int]:
    """Cluster signals by order-of-magnitude into y-axis groups.

    Signals within 1.0 log10 unit of each other share an axis group.
    Groups are indexed 0, 1, 2, ... in descending scale order.
    The frontend maps group 0 → left y-axis, 1 → right y-axis, etc.
    If more than MAX_YAXES groups exist the frontend falls back to a
    normalised z-score view.
    """
    # Compute median absolute value per signal (ignoring nulls and zeros)
    medians: Dict[str, float] = {}
    for sig in signal_configs:
        name = sig["name"]
        vals = [abs(v) for v in signal_arrays.get(name, []) if v is not None and v != 0]
        if vals:
            vals.sort()
            medians[name] = vals[len(vals) // 2]
        else:
            medians[name] = 1.0

    # Greedy clustering: iterate signals largest→smallest, assign to first
    # existing cluster within 1.0 log-decade, else start a new cluster.
    sorted_sigs = sorted(signal_configs, key=lambda s: medians[s["name"]], reverse=True)
    clusters: List[tuple] = []  # [(representative_log10, [signal_names])]
    for sig in sorted_sigs:
        name = sig["name"]
        log_scale = math.log10(max(medians[name], 1e-10))
        assigned = False
        for rep, members in clusters:
            if abs(log_scale - rep) <= 1.0:
                members.append(name)
                assigned = True
                break
        if not assigned:
            clusters.append((log_scale, [name]))

    result: Dict[str, int] = {}
    for group_idx, (_, members) in enumerate(clusters):
        for name in members:
            result[name] = group_idx
    return result


def _round(v, ndigits=4):
    """Round a float, pass through None."""
    if v is None:
        return None
    try:
        f = float(v)
        return round(f, ndigits)
    except (TypeError, ValueError):
        return None


def _parse_json_dict(s: Optional[str]) -> Dict:
    if not s or s == "None":
        return {}
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def _load_signal_csv(path: str, signal_name: str) -> Dict[str, float]:
    """Load a source signal CSV (timestamp, value) → {timestamp: value}."""
    result: Dict[str, float] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = row.get("timestamp") or row.get("Timestamp") or row.get("time")
            val_raw = row.get("value") or row.get("Value") or row.get(signal_name)
            if ts and val_raw is not None:
                try:
                    result[ts.strip()] = float(val_raw)
                except ValueError:
                    pass
    return result


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build(
    run_csv: str,
    manifest_path: str,
    data_dir: str,
    out_path: str,
    usecase_id: str = "powergrid_ca",
) -> None:
    meta = USECASE_META[usecase_id]
    print(f"[build] Loading manifest: {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)

    warmup_steps: int = manifest.get("run", {}).get("warmup_steps", 1000)
    model_names: List[str] = manifest.get("models", {}).get("names", [])
    group_names: List[str] = [g["name"] for g in meta["groups"]]
    threshold: float = meta["threshold"]

    print(f"[build] Loading signal CSVs from: {data_dir}")
    signal_data: Dict[str, Dict[str, float]] = {}
    for sig_cfg in meta["signals"]:
        sig_name = sig_cfg["name"]
        csv_file = meta["_source_csvs"][sig_name]
        csv_path = os.path.join(data_dir, csv_file)
        if not os.path.exists(csv_path):
            print(f"  [warn] Signal CSV not found: {csv_path} — signal will be null")
            signal_data[sig_name] = {}
        else:
            signal_data[sig_name] = _load_signal_csv(csv_path, sig_name)
            print(f"  Loaded {sig_name}: {len(signal_data[sig_name]):,} rows")

    print(f"[build] Loading run.csv: {run_csv}")
    print("  (this may take a moment for large files...)")

    # run.csv schema: t, timestamp, model, in_warmup, learn,
    #   raw, p, likelihood, score,
    #   system_hot, system_hot_count, system_hot_streak, alert,
    #   instant_hot_by_model, model_warmth_by_model,
    #   group_instant_count, group_warmth, group_hot

    # Collect per-timestep data: t -> {model -> {p, warmth, instant_hot}, system, groups, ts, in_warmup}
    steps: Dict[int, Dict] = {}

    with open(run_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = int(row["t"])
            model = row["model"]

            if t not in steps:
                steps[t] = {
                    "t": t,
                    "timestamp": row.get("timestamp", ""),
                    "in_warmup": row.get("in_warmup", "0") == "1",
                    "models": {},
                    "system": {
                        "hot": int(row.get("system_hot") or 0),
                        "count": int(row.get("system_hot_count") or 0),
                        "streak": int(row.get("system_hot_streak") or 0),
                        "alert": int(row.get("alert") or 0),
                    },
                    "groups_raw": {
                        "warmth": _parse_json_dict(row.get("group_warmth")),
                        "hot": _parse_json_dict(row.get("group_hot")),
                        "instant_count": _parse_json_dict(row.get("group_instant_count")),
                    },
                    "warmth_raw": _parse_json_dict(row.get("model_warmth_by_model")),
                    "instant_hot_raw": _parse_json_dict(row.get("instant_hot_by_model")),
                }

            p_val = _round(row.get("p"), 4)
            warmth_val = _round(
                steps[t]["warmth_raw"].get(model), 4
            )
            instant_hot = int(steps[t]["instant_hot_raw"].get(model) or 0)

            steps[t]["models"][model] = {
                "p": p_val,
                "warmth": warmth_val,
                "instant_hot": instant_hot,
            }

    print(f"  Loaded {len(steps):,} timesteps")

    # Sort timesteps
    sorted_ts = sorted(steps.keys())

    # ---------------------------------------------------------------------------
    # Build columnar output arrays
    # ---------------------------------------------------------------------------
    print("[build] Building columnar arrays...")

    timestamps: List[str] = []
    in_warmup_arr: List[int] = []

    # Signals: {signal_name: [float|None, ...]}
    signal_arrays: Dict[str, List] = {s["name"]: [] for s in meta["signals"]}

    # Models: {model_name: {p: [], warmth: [], instant_hot: []}}
    model_arrays: Dict[str, Dict[str, List]] = {
        m["name"]: {"p": [], "warmth": [], "instant_hot": []}
        for m in meta["models"]
    }

    # Groups: {group_name: {hot: [], warmth: []}}
    group_arrays: Dict[str, Dict[str, List]] = {
        g["name"]: {"hot": [], "warmth": []}
        for g in meta["groups"]
    }

    # System
    system_arrays: Dict[str, List] = {
        "hot": [], "streak": [], "alert": [], "count": []
    }

    for t in sorted_ts:
        step = steps[t]
        ts = step["timestamp"]
        timestamps.append(ts)
        in_warmup_arr.append(1 if step["in_warmup"] else 0)

        # Signal values (join from source CSVs by timestamp)
        for sig_cfg in meta["signals"]:
            sig_name = sig_cfg["name"]
            val = signal_data.get(sig_name, {}).get(ts)
            signal_arrays[sig_name].append(_round(val, 2))

        # Model outputs
        for m_cfg in meta["models"]:
            m_name = m_cfg["name"]
            m_data = step["models"].get(m_name, {})
            model_arrays[m_name]["p"].append(m_data.get("p"))
            model_arrays[m_name]["warmth"].append(m_data.get("warmth"))
            model_arrays[m_name]["instant_hot"].append(m_data.get("instant_hot", 0))

        # Groups
        for g_cfg in meta["groups"]:
            g_name = g_cfg["name"]
            group_arrays[g_name]["hot"].append(
                int(step["groups_raw"]["hot"].get(g_name) or 0)
            )
            group_arrays[g_name]["warmth"].append(
                _round(step["groups_raw"]["warmth"].get(g_name), 4)
            )

        # System
        sys = step["system"]
        system_arrays["hot"].append(sys.get("hot", 0))
        system_arrays["streak"].append(sys.get("streak", 0))
        system_arrays["alert"].append(sys.get("alert", 0))
        system_arrays["count"].append(sys.get("count", 0))

    # ---------------------------------------------------------------------------
    # Compute y-axis scale groups for the frontend chart
    # ---------------------------------------------------------------------------
    yaxis_groups = _assign_yaxis_groups(signal_arrays, meta["signals"])
    n_axes = max(yaxis_groups.values()) + 1 if yaxis_groups else 1
    print(f"[build] Signal y-axis groups: {yaxis_groups}  ({n_axes} scale cluster(s))")

    # ---------------------------------------------------------------------------
    # Assemble output document
    # ---------------------------------------------------------------------------
    print("[build] Assembling output document...")

    # Strip internal keys before embedding meta, then inject yaxis_group per signal
    out_meta = {k: v for k, v in meta.items() if not k.startswith("_")}
    # Embed yaxis_group into each signal entry so the frontend can use it generically
    out_meta["signals"] = [
        {**sig, "yaxis_group": yaxis_groups.get(sig["name"], 0)}
        for sig in meta["signals"]
    ]
    out_meta["warmup_end_t"] = warmup_steps - 1
    out_meta["total_steps"] = len(sorted_ts)
    out_meta["time_range"] = {
        "start": timestamps[0] if timestamps else "",
        "end": timestamps[-1] if timestamps else "",
    }

    output = {
        "meta": out_meta,
        "data": {
            "timestamps": timestamps,
            "in_warmup": in_warmup_arr,
            "signals": signal_arrays,
            "models": model_arrays,
            "groups": group_arrays,
            "system": system_arrays,
        },
    }

    # ---------------------------------------------------------------------------
    # Write output
    # ---------------------------------------------------------------------------
    out_path_obj = Path(out_path)
    out_path_obj.parent.mkdir(parents=True, exist_ok=True)

    print(f"[build] Writing output: {out_path}")
    with open(out_path, "w") as f:
        json.dump(output, f, separators=(",", ":"))  # compact JSON

    size_mb = out_path_obj.stat().st_size / 1_048_576
    print(f"[build] Done. Output size: {size_mb:.1f} MB  ({len(sorted_ts):,} steps)")
    print(f"[build] File ready: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    repo_root = Path(__file__).parent.parent  # demo/ lives one level below repo root

    p = argparse.ArgumentParser(description="Build HTM-Monitor demo data artifact")
    p.add_argument("--usecase", default="powergrid_ca",
                   choices=list(USECASE_META.keys()),
                   help="Use-case ID (default: powergrid_ca)")
    p.add_argument("--run-csv",  default=None, help="Override path to run.csv")
    p.add_argument("--manifest", default=None, help="Override path to run.manifest.json")
    p.add_argument("--data-dir", default=None, help="Override directory containing signal CSVs")
    p.add_argument("--out",      default=None, help="Override output JSON path")
    args = p.parse_args()

    # Per-use-case default paths
    uc = args.usecase
    if uc == "powergrid_ca":
        default_run_csv  = str(repo_root / "outputs" / "powergrid_ca" / "run.csv")
        default_manifest = str(repo_root / "outputs" / "powergrid_ca" / "run.manifest.json")
        default_data_dir = str(repo_root / "data" / "powergrid_ca" / "caiso_2019fit_2020_2022_pad0p00")
    elif uc == "batadal_ctown":
        default_run_csv  = str(repo_root / "outputs" / "batadal_ctown" / "run.csv")
        default_manifest = str(repo_root / "outputs" / "batadal_ctown" / "run.manifest.json")
        default_data_dir = str(repo_root / "data" / "batadal_ctown")
    else:
        # Generic fallback for future use cases
        default_run_csv  = str(repo_root / "outputs" / uc / "run.csv")
        default_manifest = str(repo_root / "outputs" / uc / "run.manifest.json")
        default_data_dir = str(repo_root / "data" / uc)

    default_out = str(Path(__file__).parent / "static" / "demo_data" / f"{uc}.json")

    build(
        run_csv=args.run_csv or default_run_csv,
        manifest_path=args.manifest or default_manifest,
        data_dir=args.data_dir or default_data_dir,
        out_path=args.out or default_out,
        usecase_id=uc,
    )


if __name__ == "__main__":
    main()
