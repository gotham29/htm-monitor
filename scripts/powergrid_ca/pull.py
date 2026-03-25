#scripts/pull_eai.py

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Optional

import pandas as pd
import requests

API_BASE_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
DEFAULT_RESPONDENT = "CISO"
DEFAULT_PAGE_LEN = 5000

def _download_eia_long_csv(
    *,
    api_key: str,
    out_csv: Path,
    respondent: str,
    start: str,
    end: str,
    page_len: int = DEFAULT_PAGE_LEN,
) -> None:
    params = {
        "api_key": api_key,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": respondent,
        "start": start,
        "end": end,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": int(page_len),
    }

    rows = []
    offset = 0

    while True:
        params["offset"] = offset
        r = requests.get(API_BASE_URL, params=params, timeout=60)
        r.raise_for_status()
        payload = r.json()
        data = ((payload or {}).get("response") or {}).get("data") or []

        if not data:
            break

        rows.extend(data)
        offset += int(page_len)
        print(f"Downloaded rows so far: {len(rows)}")

    if not rows:
        raise ValueError(f"No EIA rows returned for respondent={respondent}, start={start}, end={end}")

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved raw EIA CSV: {out_csv}")

TYPE_MAP = {
    "D": "demand",
    "DF": "demand_forecast",
    "NG": "net_generation",
    "TI": "total_interchange",
}

ALL_SIGNAL_COLS = [
    "demand",
    "forecast_error",
    "net_generation",
    "imbalance",
    "total_interchange",
    "margin_proxy",
]


@dataclass(frozen=True)
class EventWindow:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Download and/or prepare CAISO/CISO EIA hourly data for HTM Monitor demos. "
            "Computes encoder bounds from a calibration window and exports "
            "an HTM-ready dataset for a chosen run window."
        )
    )
    ap.add_argument(
        "--download",
        action="store_true",
        help="Download raw EIA data before preparing dataset",
    )
    ap.add_argument(
        "--api-key",
        default=None,
        help="EIA API key (required when --download is used)",
    )
    ap.add_argument(
        "--respondent",
        default=DEFAULT_RESPONDENT,
        help=f"EIA respondent code to download (default: {DEFAULT_RESPONDENT})",
    )
    ap.add_argument(
        "--download-start",
        default=None,
        help="Download start timestamp, e.g. 2019-01-01T00",
    )
    ap.add_argument(
        "--download-end",
        default=None,
        help="Download end timestamp, e.g. 2022-10-31T00",
    )
    ap.add_argument(
        "--raw-csv",
        required=True,
        help="Path to raw EIA long-format CSV",
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for prepared dataset",
    )
    ap.add_argument(
        "--calibration-start",
        required=True,
        help="Calibration window start timestamp, e.g. '2019-01-01 00:00:00'",
    )
    ap.add_argument(
        "--calibration-end",
        required=True,
        help="Calibration window end timestamp, e.g. '2019-12-31 23:00:00'",
    )
    ap.add_argument(
        "--export-start",
        required=True,
        help="Export/run window start timestamp",
    )
    ap.add_argument(
        "--export-end",
        required=True,
        help="Export/run window end timestamp",
    )
    ap.add_argument(
        "--signals",
        nargs="+",
        default=["demand", "net_generation", "imbalance"],
        choices=ALL_SIGNAL_COLS,
        help="Signals to export as separate HTM CSVs",
    )
    ap.add_argument(
        "--event-window",
        action="append",
        default=[],
        help="Named event window in the form 'name|start|end'; can be repeated",
    )
    ap.add_argument(
        "--pad-frac",
        type=float,
        default=0.0,
        help="Fractional padding applied to calibration min/max",
    )
    return ap.parse_args()


def _parse_ts(s: str) -> pd.Timestamp:
    ts = pd.Timestamp(s)
    if pd.isna(ts):
        raise ValueError(f"Invalid timestamp: {s}")
    return ts


def _parse_event_window(spec: str) -> EventWindow:
    parts = [p.strip() for p in str(spec).split("|")]
    if len(parts) != 3:
        raise ValueError(f"--event-window must be 'name|start|end'; got: {spec}")
    name, start_s, end_s = parts
    start = _parse_ts(start_s)
    end = _parse_ts(end_s)
    if end < start:
        raise ValueError(f"Event window end < start: {spec}")
    return EventWindow(name=name, start=start, end=end)


def _safe_bounds(series: pd.Series, pad_frac: float = 0.05) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        raise ValueError("Cannot compute bounds on empty series")

    vmin = float(s.min())
    vmax = float(s.max())
    span = vmax - vmin

    if span <= 0:
        pad = max(1.0, abs(vmin) * pad_frac, 1.0)
    else:
        pad = span * pad_frac

    return {
        "observed_min": vmin,
        "observed_max": vmax,
        "minVal": vmin - pad,
        "maxVal": vmax + pad,
    }


def _load_and_pivot(raw_csv: Path) -> pd.DataFrame:
    if not raw_csv.exists():
        raise FileNotFoundError(f"Raw CSV not found: {raw_csv}")

    df = pd.read_csv(raw_csv)

    required_cols = {"period", "respondent", "type", "value"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.loc[df["respondent"] == "CISO"].copy()
    df = df.loc[df["type"].isin(TYPE_MAP.keys())].copy()
    df["timestamp"] = pd.to_datetime(df["period"], format="%Y-%m-%dT%H", errors="raise")
    df["signal"] = df["type"].map(TYPE_MAP)

    wide = (
        df.pivot_table(
            index="timestamp",
            columns="signal",
            values="value",
           aggfunc="first",
        )
        .sort_index()
        .reset_index()
    )

    expected = ["demand", "demand_forecast", "net_generation", "total_interchange"]
    missing_signals = [c for c in expected if c not in wide.columns]
    if missing_signals:
        raise ValueError(f"Missing expected signals after pivot: {missing_signals}")

    wide["forecast_error"] = wide["demand"] - wide["demand_forecast"]
    wide["imbalance"] = wide["demand"] - wide["net_generation"]
    wide["margin_proxy"] = (
        wide["net_generation"] + wide["total_interchange"] - wide["demand"]
    )

    core_cols = [
        "demand",
        "demand_forecast",
        "net_generation",
        "total_interchange",
        "imbalance",
        "forecast_error",
        "margin_proxy",
    ]
    before = len(wide)
    wide = wide.dropna(subset=core_cols).copy()
    after = len(wide)
    wide.attrs["rows_before_dropna"] = int(before)
    wide.attrs["rows_after_dropna"] = int(after)
    return wide


def _slice_closed(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    out = df.loc[(df["timestamp"] >= start) & (df["timestamp"] <= end)].copy()
    if out.empty:
        raise ValueError(f"Slice is empty for range [{start}, {end}]")
    return out


def _event_timestamp_strings(
    export_df: pd.DataFrame,
    events: Sequence[EventWindow],
) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for ev in events:
        ts = (
            export_df.loc[
                (export_df["timestamp"] >= ev.start) & (export_df["timestamp"] <= ev.end),
                "timestamp",
            ]
            .dt.strftime("%Y-%m-%d %H:%M:%S")
            .tolist()
        )
        out[ev.name] = ts
    return out


def main() -> None:
    args = _parse_args()

    raw_csv = Path(args.raw_csv)

    if args.download:
        if not args.api_key:
            raise ValueError("--api-key is required when using --download")
        if not args.download_start or not args.download_end:
            raise ValueError("--download-start and --download-end are required when using --download")

        _download_eia_long_csv(
            api_key=args.api_key,
            out_csv=raw_csv,
            respondent=args.respondent,
            start=args.download_start,
            end=args.download_end,
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cal_start = _parse_ts(args.calibration_start)
    cal_end = _parse_ts(args.calibration_end)
    exp_start = _parse_ts(args.export_start)
    exp_end = _parse_ts(args.export_end)

    if cal_end < cal_start:
        raise ValueError("calibration-end must be >= calibration-start")
    if exp_end < exp_start:
        raise ValueError("export-end must be >= export-start")

    events = [_parse_event_window(spec) for spec in args.event_window]

    wide = _load_and_pivot(raw_csv)
    calibration_df = _slice_closed(wide, cal_start, cal_end)
    export_df = _slice_closed(wide, exp_start, exp_end)

    wide_out = export_df.copy()
    wide_out["timestamp"] = wide_out["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    wide_csv = out_dir / "ciso_wide.csv"
    wide_out.to_csv(wide_csv, index=False)

    stats: Dict[str, dict] = {}
    for col in args.signals:
        out = export_df[["timestamp", col]].copy()
        out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        out = out.rename(columns={col: "value"})
        out.to_csv(out_dir / f"{col}.csv", index=False)
        stats[col] = _safe_bounds(calibration_df[col], pad_frac=float(args.pad_frac))

    event_ts_by_name = _event_timestamp_strings(export_df, events)

    stats["meta"] = {
        "raw_csv": str(raw_csv),
        "wide_csv": str(wide_csv),
        "rows_before_dropna": int(wide.attrs["rows_before_dropna"]),
        "rows_after_dropna": int(wide.attrs["rows_after_dropna"]),
        "full_start_timestamp": wide["timestamp"].min().strftime("%Y-%m-%d %H:%M:%S"),
        "full_end_timestamp": wide["timestamp"].max().strftime("%Y-%m-%d %H:%M:%S"),
        "calibration_start": calibration_df["timestamp"].min().strftime("%Y-%m-%d %H:%M:%S"),
        "calibration_end": calibration_df["timestamp"].max().strftime("%Y-%m-%d %H:%M:%S"),
        "export_start": export_df["timestamp"].min().strftime("%Y-%m-%d %H:%M:%S"),
        "export_end": export_df["timestamp"].max().strftime("%Y-%m-%d %H:%M:%S"),
        "signals_exported": list(args.signals),
        "pad_frac": float(args.pad_frac),
    }
    stats["event_windows"] = {
        ev.name: {
            "start": ev.start.strftime("%Y-%m-%d %H:%M:%S"),
            "end": ev.end.strftime("%Y-%m-%d %H:%M:%S"),
            "num_timestamps": int(len(event_ts_by_name.get(ev.name, []))),
        }
        for ev in events
    }
    stats["event_timestamps"] = event_ts_by_name

    stats_path = out_dir / "feature_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))

    print(f"Wrote wide table: {wide_csv}")
    for col in args.signals:
        print(f"Wrote signal CSV: {out_dir / f'{col}.csv'}")
    print(f"Wrote stats: {stats_path}")
    print(json.dumps(stats["meta"], indent=2))
    if stats["event_windows"]:
        print(json.dumps(stats["event_windows"], indent=2))


if __name__ == "__main__":
    main()
