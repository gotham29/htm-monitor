from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


FINAL_TIMESTAMP_COL = "timestamp"


def _clean_name(name: object) -> str:
    s = "" if name is None else str(name).strip()
    if not s or s.lower() == "nan":
        return ""
    if s == FINAL_TIMESTAMP_COL:
        return FINAL_TIMESTAMP_COL
    
    # Normalize common SWaT suffixes so different dataset vintages share the
    # same canonical signal names:
    #   LIT101.Pv      -> LIT101
    #   MV101.Status   -> MV101
    #   P101.Status    -> P101
    #   FIT301.Pv      -> FIT301
    #
    # Do this before removing punctuation so we preserve the suffix boundary.
    s = re.sub(r"\.(Pv|Status)$", "", s, flags=re.IGNORECASE)

    s = s.replace(" ", "")
    s = s.replace(".", "")
    s = s.replace("+", "")
    return s


def _binary_map() -> Dict[str, int]:
    return {
        "Active": 1,
        "Inactive": 0,
        "True": 1,
        "False": 0,
        "ON": 1,
        "OFF": 0,
        "Open": 1,
        "Closed": 0,
    }


def _extract_structured_value(cell: object) -> object:
    """
    Handle raw SWaT cells that sometimes appear as serialized dict-like payloads
    such as:

      "{u'IsSystem': False, u'Name': u'Inactive', u'Value': 0}"

    We prefer the numeric Value field when present. If parsing fails, return the
    original object unchanged.
    """
    if cell is None:
        return cell
    if isinstance(cell, (int, float, bool)):
        return cell

    s = str(cell).strip()
    if not s:
        return s

    # Fast path: only attempt heavier parsing when this looks like a dict payload.
    if not (s.startswith("{") and "Value" in s):
        return cell

    s_norm = re.sub(r"\bu(['\"])", r"\1", s)
    try:
        obj = ast.literal_eval(s_norm)
        if isinstance(obj, dict) and "Value" in obj:
            return obj["Value"]
    except Exception:
        pass

    # Fallback: regex-extract the numeric payload after 'Value': ...
    m = re.search(r"['\"]Value['\"]\s*:\s*([-+]?\d+(?:\.\d+)?)", s_norm)
    if m:
        val = m.group(1)
        if "." in val:
            return float(val)
        return int(val)

    return cell


def _parse_delta_cols(arg: str | None) -> List[str]:
    if not arg:
        return []
    return [x.strip() for x in arg.split(",") if x.strip()]


def _parse_feature_cols(arg: str | None) -> List[str]:
    if not arg:
        return []
    return [x.strip() for x in arg.split(",") if x.strip()]


def _parse_keep_cols(arg: str | None) -> List[str]:
    if not arg:
        return []
    return [x.strip() for x in arg.split(",") if x.strip()]


def _load_raw_xlsx(
    path: Path,
    *,
    header_row: int,
    data_start_row: int,
    raw_timestamp_col: str,
) -> pd.DataFrame:
    raw = pd.read_excel(path, header=None)

    if header_row < 0:
        raise ValueError("header_row must be >= 0")
    if data_start_row <= header_row:
        raise ValueError("data_start_row must be > header_row")

    headers = [_clean_name(x) for x in raw.iloc[header_row].tolist()]
    headers = [FINAL_TIMESTAMP_COL if h == _clean_name(raw_timestamp_col) else h for h in headers]
    if FINAL_TIMESTAMP_COL not in headers:
        raise ValueError(
            f"Could not find timestamp column '{raw_timestamp_col}' in header row {header_row}"
        )

    df = raw.iloc[data_start_row:].copy()
    df.columns = headers
    df = df.loc[:, [c for c in df.columns if c]]

    return df.reset_index(drop=True)


def _coerce_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    raw_ts = df[FINAL_TIMESTAMP_COL].astype(str).str.strip()
    ts = pd.to_datetime(
        raw_ts,
        errors="coerce",
        utc=True,
    )

    bad_mask = ts.isna()
    if bad_mask.any():
        bad_examples = raw_ts[bad_mask].head(10).tolist()
        raise ValueError(
            "Failed to parse some timestamps from raw SWaT workbook. "
            f"Examples: {bad_examples}"
        )

    # Floor to second and drop tz info for downstream config compatibility.
    ts = ts.dt.floor("s").dt.tz_localize(None)

    df = df.copy()
    df[FINAL_TIMESTAMP_COL] = ts
    df = df.sort_values(FINAL_TIMESTAMP_COL).reset_index(drop=True)

    if not df[FINAL_TIMESTAMP_COL].is_monotonic_increasing:
        raise ValueError("Timestamp column is not monotonic increasing after parsing")

    dupes = int(df[FINAL_TIMESTAMP_COL].duplicated().sum())
    if dupes:
        duped = df.loc[df[FINAL_TIMESTAMP_COL].duplicated(keep=False), FINAL_TIMESTAMP_COL]
        raise ValueError(
            f"Found {dupes} duplicated timestamps after flooring to second. "
            f"Example duplicates: {duped.head(10).tolist()}"
        )

    return df


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mapper = _binary_map()

    for col in out.columns:
        if col == FINAL_TIMESTAMP_COL:
            continue

        series = out[col]
        if pd.api.types.is_numeric_dtype(series):
            out[col] = pd.to_numeric(series, errors="raise")
            continue

        # Normalize strings like Active/Inactive and structured payload cells.
        series = series.map(_extract_structured_value)
        series = series.astype(str).str.strip()
        series = series.replace(mapper)

        numeric = pd.to_numeric(series, errors="coerce")
        bad_mask = numeric.isna()
        if bad_mask.any():
            bad_examples = series[bad_mask].head(10).tolist()
            raise ValueError(
                f"Unable to coerce column '{col}' to numeric. "
                f"Examples: {bad_examples}"
            )
        out[col] = numeric

    return out


def _add_delta_features(df: pd.DataFrame, delta_cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in delta_cols:
        if col not in out.columns:
            raise KeyError(f"Requested delta column '{col}' not found")
        delta_name = f"{col}_delta"
        out[delta_name] = out[col].diff().fillna(0.0)
    return out


def _add_slope_features(
    df: pd.DataFrame,
    slope_cols: Iterable[str],
    *,
    slope_lag: int,
) -> pd.DataFrame:
    if slope_lag < 1:
        raise ValueError("slope_lag must be >= 1")

    out = df.copy()
    for col in slope_cols:
        if col not in out.columns:
            raise KeyError(f"Requested slope column '{col}' not found")
        feat_name = f"{col}_slope{int(slope_lag)}"
        out[feat_name] = (out[col] - out[col].shift(slope_lag)).fillna(0.0)
    return out


def _add_ema_residual_features(
    df: pd.DataFrame,
    ema_resid_cols: Iterable[str],
    *,
    short_span: int,
    long_span: int,
) -> pd.DataFrame:
    if short_span < 1 or long_span < 1:
        raise ValueError("EMA spans must be >= 1")
    if short_span >= long_span:
        raise ValueError("ema_short_span must be < ema_long_span")

    out = df.copy()
    for col in ema_resid_cols:
        if col not in out.columns:
            raise KeyError(f"Requested EMA residual column '{col}' not found")

        short_ema = out[col].ewm(span=short_span, adjust=False).mean()
        long_ema = out[col].ewm(span=long_span, adjust=False).mean()
        feat_name = f"{col}_emaresid_{int(short_span)}_{int(long_span)}"
        out[feat_name] = (short_ema - long_ema).fillna(0.0)
    return out


def _select_base_columns(
    df: pd.DataFrame,
    *,
    keep_cols: List[str],
) -> pd.DataFrame:
    if not keep_cols:
        raise ValueError("keep_cols must contain at least one signal column")

    cols = [FINAL_TIMESTAMP_COL] + keep_cols
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected columns: {missing}")

    out = df[cols].copy()
    return out


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Keep timestamp first.
    cols = [FINAL_TIMESTAMP_COL] + [c for c in out.columns if c != FINAL_TIMESTAMP_COL]
    out = out[cols]

    # Sanity check for NaN after coercion / feature creation.
    nan_cols = [c for c in out.columns if out[c].isna().any()]
    if nan_cols:
        raise ValueError(f"NaN values remain in columns: {nan_cols}")

    return out


def build_final_swat_table(
    xlsx_path: Path,
    *,
    delta_cols: List[str],
    slope_cols: List[str],
    slope_lag: int,
    ema_resid_cols: List[str],
    ema_short_span: int,
    ema_long_span: int,
    resample_sec: int,
    header_row: int,
    data_start_row: int,
    raw_timestamp_col: str,
    keep_cols: List[str],
) -> pd.DataFrame:
    df = _load_raw_xlsx(
        xlsx_path,
        header_row=header_row,
        data_start_row=data_start_row,
        raw_timestamp_col=raw_timestamp_col,
    )
    df = _coerce_timestamp(df)
    df = _coerce_numeric(df)
    df = _select_base_columns(df, keep_cols=keep_cols)

    analog_cols = list(keep_cols)

    if resample_sec > 1:
        df = df.set_index(FINAL_TIMESTAMP_COL)
        agg = {c: "mean" for c in analog_cols}
        df = df.resample(f"{resample_sec}s").agg(agg)
        df = df.dropna().reset_index()
    df = _add_delta_features(df, delta_cols)
    df = _add_slope_features(df, slope_cols, slope_lag=slope_lag)
    df = _add_ema_residual_features(
        df,
        ema_resid_cols,
        short_span=ema_short_span,
        long_span=ema_long_span,
    )
    df = _finalize(df)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Prepare a SWaT workbook into a model-ready CSV."
    )
    ap.add_argument(
        "--xlsx",
        required=True,
        help="Path to raw SWaT workbook (.xlsx)",
    )
    ap.add_argument(
        "--out-csv",
        required=True,
        help="Path to final model-ready CSV output",
    )
    ap.add_argument(
        "--header-row",
        type=int,
        default=1,
        help="0-based row index containing column headers (default: 1)",
    )
    ap.add_argument(
        "--data-start-row",
        type=int,
        default=3,
        help="0-based first row of actual data (default: 3)",
    )
    ap.add_argument(
        "--raw-timestamp-col",
        default="GMT +0",
        help="Exact raw timestamp column name before cleaning (default: 'GMT +0')",
    )
    ap.add_argument(
        "--keep-cols",
        default="DPIT301,FIT301,LIT301",
        help="Comma-separated base numeric columns to keep before feature engineering. "
             "Example: LIT101,FIT201,AIT202",
    )
    ap.add_argument(
        "--delta-cols",
        default="",
        help="Comma-separated numeric columns to add first-difference features for "
             "(default: none). Example: LIT301,DPIT301",
    )
    ap.add_argument(
        "--slope-cols",
        default="",
        help="Comma-separated numeric columns to add lagged slope features for. "
             "Example: LIT301,FIT301",
    )
    ap.add_argument(
        "--slope-lag",
        type=int,
        default=5,
        help="Lag in rows for slope features (default: 5)",
    )
    ap.add_argument(
        "--ema-resid-cols",
        default="",
        help="Comma-separated numeric columns to add short-long EMA residual features for. "
             "Example: LIT301,FIT301",
    )
    ap.add_argument(
        "--ema-short-span",
        type=int,
        default=5,
        help="Short EMA span for EMA residual features (default: 5)",
    )
    ap.add_argument(
        "--ema-long-span",
        type=int,
        default=30,
        help="Long EMA span for EMA residual features (default: 30)",
    )
    ap.add_argument(
        "--resample-sec",
        type=int,
        default=1,
        help="Resample interval in seconds (e.g., 5 or 10). Default=1 (no resampling)",
    )
    args = ap.parse_args()

    xlsx_path = Path(args.xlsx).expanduser().resolve()
    out_csv = Path(args.out_csv).expanduser().resolve()
    delta_cols = _parse_delta_cols(args.delta_cols)
    slope_cols = _parse_feature_cols(args.slope_cols)
    ema_resid_cols = _parse_feature_cols(args.ema_resid_cols)
    keep_cols = _parse_keep_cols(args.keep_cols)

    df = build_final_swat_table(
        xlsx_path,
        delta_cols=delta_cols,
        slope_cols=slope_cols,
        slope_lag=args.slope_lag,
        ema_resid_cols=ema_resid_cols,
        ema_short_span=args.ema_short_span,
        ema_long_span=args.ema_long_span,
        resample_sec=args.resample_sec,
        header_row=args.header_row,
        data_start_row=args.data_start_row,
        raw_timestamp_col=args.raw_timestamp_col,
        keep_cols=keep_cols,
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, date_format="%Y-%m-%d %H:%M:%S")

    print(f"wrote {out_csv}")
    print(f"rows={len(df)} cols={len(df.columns)}")
    print("timestamp_min=", df[FINAL_TIMESTAMP_COL].min())
    print("timestamp_max=", df[FINAL_TIMESTAMP_COL].max())
    print("kept_base_cols=", keep_cols)
    print("added_delta_cols=", [c for c in df.columns if c.endswith("_delta")])
    print("added_slope_cols=", [c for c in df.columns if "_slope" in c])
    print("added_ema_resid_cols=", [c for c in df.columns if "_emaresid_" in c])


if __name__ == "__main__":
    main()
