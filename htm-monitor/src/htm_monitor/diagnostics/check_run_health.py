from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd


def _load_run(run_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(run_csv)
    if "t" not in df.columns:
        raise ValueError(f"run.csv missing required column: t ({run_csv})")
    df = df.sort_values("t").reset_index(drop=True)
    return df


def _find_model_cols(df: pd.DataFrame, suffix: str) -> List[str]:
    # supports columns like: sA_model.raw_anomaly, sA_model.anomaly_score, etc.
    return [c for c in df.columns if c.endswith(suffix)]


def _is_long_schema(df: pd.DataFrame) -> bool:
    # long/tidy schema example:
    #   t, timestamp, model, raw, p, likelihood, score, system_score, alert, hot_by_model
    return ("model" in df.columns) and ("raw" in df.columns)


def _baseline_slice(df: pd.DataFrame, *, baseline_window: int, baseline_end_t: Optional[int]) -> pd.DataFrame:
    """
    Choose baseline window ending at baseline_end_t (inclusive) if provided,
    otherwise default to end of dataframe.
    """
    bw = int(baseline_window)
    if bw <= 10:
        raise ValueError("--baseline-window too small")

    if baseline_end_t is None:
        if len(df) < bw + 50:
            raise ValueError(f"run too short for baseline-window={bw}: n={len(df)}")
        return df.iloc[-bw:]

    end_t = int(baseline_end_t)
    d0 = df[df["t"] <= end_t]
    if len(d0) < bw + 10:
        raise ValueError(
            f"Not enough rows with t<=baseline_end_t={end_t} for baseline-window={bw}: n={len(d0)}"
        )
    return d0.iloc[-bw:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="outputs/<run>/ directory containing run.csv and analysis/ (optional)")
    ap.add_argument("--baseline-window", type=int, default=400,
                    help="Number of timesteps at end of settle region to evaluate baseline stability.")
    ap.add_argument("--baseline-end-t", type=int, default=None,
                    help="End timestep (t) for baseline window. Use this to anchor baseline before injected anomalies (e.g., end of settle).")
    ap.add_argument("--baseline-mean-max", type=float, default=0.05,
                    help="Max allowed mean raw anomaly in baseline window (per model).")
    ap.add_argument("--spike-quantile", type=float, default=0.99,
                    help="Quantile of raw anomaly used to validate spikes exist.")
    ap.add_argument("--spike-min", type=float, default=0.5,
                    help="Minimum acceptable high-quantile raw anomaly (per model).")

    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    run_csv = run_dir / "run.csv"
    if not run_csv.exists():
        raise ValueError(f"Missing run.csv: {run_csv}")

    df = _load_run(run_csv)

    ok = True
    q = float(args.spike_quantile)

    # ---- LONG schema: (model, raw) ----
    if _is_long_schema(df):
        if df["raw"].isna().all():
            raise ValueError("run.csv has column 'raw' but it is all-NaN.")

        models = sorted(str(m) for m in df["model"].dropna().unique().tolist())
        if not models:
            raise ValueError("run.csv long schema: no models found in column 'model'.")

        print(f"[check] run={run_dir} schema=long models={models} n_rows={len(df)}")

        for m in models:
            d = df[df["model"] == m].sort_values("t").reset_index(drop=True)
            base = _baseline_slice(d, baseline_window=args.baseline_window, baseline_end_t=args.baseline_end_t)

            mean_raw = float(base["raw"].mean())
            if mean_raw > float(args.baseline_mean_max):
                ok = False
                print(f"[FAIL] baseline mean too high: model={m} mean_raw={mean_raw:.6f} > {args.baseline_mean_max}")
            else:
                print(f"[PASS] baseline mean: model={m} mean_raw={mean_raw:.6f}")

            spike_q = float(d["raw"].quantile(q))
            if spike_q < float(args.spike_min):
                ok = False
                print(f"[FAIL] spikes too weak: model={m} q{q:.2f}={spike_q:.6f} < {args.spike_min}")
            else:
                print(f"[PASS] spikes present: model={m} q{q:.2f}={spike_q:.6f}")

    # ---- WIDE schema: per-model columns ----
    else:
        raw_cols = _find_model_cols(df, ".raw_anomaly")
        if not raw_cols:
            raw_cols = _find_model_cols(df, ".anomaly_score")
        if not raw_cols:
            raise ValueError(
                "No raw anomaly columns found.\n"
                "Expected either long schema with columns ['model','raw'] OR wide schema with columns ending in "
                "'.raw_anomaly' (or '.anomaly_score')."
            )

        base = _baseline_slice(df, baseline_window=args.baseline_window, baseline_end_t=args.baseline_end_t)

        print(f"[check] run={run_dir} schema=wide n_rows={len(df)} baseline_window={int(args.baseline_window)}")
        print(f"[check] using raw anomaly columns: {raw_cols}")

        for c in raw_cols:
            mean_raw = float(base[c].mean())
            if mean_raw > float(args.baseline_mean_max):
                ok = False
                print(f"[FAIL] baseline mean too high: col={c} mean={mean_raw:.6f} > {args.baseline_mean_max}")
            else:
                print(f"[PASS] baseline mean: col={c} mean={mean_raw:.6f}")

            spike_q = float(df[c].quantile(q))
            if spike_q < float(args.spike_min):
                ok = False
                print(f"[FAIL] spikes too weak: col={c} q{q:.2f}={spike_q:.6f} < {args.spike_min}")
            else:
                print(f"[PASS] spikes present: col={c} q{q:.2f}={spike_q:.6f}")

    if not ok:
        raise SystemExit(2)
    print("[check] OK")


if __name__ == "__main__":
    main()
