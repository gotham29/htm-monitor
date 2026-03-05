#src/demo/make_proof_data.py

from __future__ import annotations

import argparse
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ===============================
# Core Helpers
# ===============================

@dataclass
class Window:
    start: int
    end: int
    label: str


def _sigmoid_stable(x: np.ndarray) -> np.ndarray:
    """
    Numerically-stable sigmoid for large |x| (avoids exp overflow warnings).
    sigmoid(x) = 1/(1+exp(-x))
    """
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    # for x>=0: 1/(1+exp(-x)) is stable
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    # for x<0: exp(x)/(1+exp(x)) is stable
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out

def ar1_noise(n: int, sigma: float, phi: float, rng: np.random.Generator) -> np.ndarray:
    x = np.zeros(n)
    eps = rng.normal(0.0, sigma, size=n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    return x

def _interp_shifted(y: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """
    Return y evaluated at (t + shift[t]) via linear interpolation.
    shift is in *steps* (can be fractional). Values are clamped to [0, n-1].
    This is the core of the "time-warp" anomaly: it preserves the marginal
    distribution/range while breaking temporal predictability.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    t = np.arange(n, dtype=float)
    tp = np.clip(t + np.asarray(shift, dtype=float), 0.0, float(n - 1))
    return np.interp(tp, t, y)


def sigmoid_ramp(n: int, start: int, end: int, ramp_steps: int) -> np.ndarray:
    if end <= start:
        return np.zeros(n)

    t = np.arange(n)
    rs = max(1, ramp_steps)

    rise = _sigmoid_stable((t - start) / rs)
    fall = _sigmoid_stable((t - end) / rs)

    w = rise * (1 - fall)
    w = (w - w.min()) / (w.max() - w.min() + 1e-12)
    return w

def soft_clip(y: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """
    Smoothly squash into [lo, hi] without hard clipping artifacts.
    """
    mid = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    if half <= 1e-12:
        return np.full_like(y, mid)
    z = (y - mid) / half
    return mid + half * np.tanh(z)


def envelope_bounds(x: np.ndarray, mode: str, pad: float) -> Tuple[float, float]:
    mode = str(mode).lower().strip()
    if mode == "minmax":
        lo, hi = float(np.min(x)), float(np.max(x))
    else:  # "p01p99"
        lo, hi = float(np.quantile(x, 0.01)), float(np.quantile(x, 0.99))
    return lo - float(pad), hi + float(pad)

def gt_timestamps_for_windows(
    ts: pd.DatetimeIndex,
    wins: List[Window],
    mode: str,
) -> List[str]:
    """
    Convert injected anomaly windows into a list of timestamp strings suitable for
    SourceSpec.gt_timestamps / labels.timestamps.

    mode:
      - start:      one timestamp per window (start)
      - start_end:  start + last affected timestamp (end-1)
      - all:        every timestamp inside each window (potentially large)
    """
    mode = str(mode).lower().strip()
    out: List[str] = []
    n = len(ts)
    for w in wins:
        if w.end <= w.start:
            continue
        a = max(0, min(int(w.start), n - 1))
        b = max(0, min(int(w.end), n))  # end exclusive
        if mode == "all":
            out.extend(ts[a:b].strftime("%Y-%m-%d %H:%M:%S").tolist())
        elif mode == "start_end":
            out.append(ts[a].strftime("%Y-%m-%d %H:%M:%S"))
            out.append(ts[max(a, b - 1)].strftime("%Y-%m-%d %H:%M:%S"))
        else:  # "start"
            out.append(ts[a].strftime("%Y-%m-%d %H:%M:%S"))
    # de-dupe, stable order
    seen = set()
    deduped = []
    for x in out:
        if x not in seen:
            seen.add(x)
            deduped.append(x)
    return deduped


def generate_baseline(
    n: int,
    level: float,
    period_steps: int,
    amplitude: float,
    noise_sigma: float,
    rng: np.random.Generator,
    style: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Non-trivial but learnable baseline:
      - dominant seasonal component
      - secondary shorter seasonal component
      - 2nd harmonic (adds asymmetry)
      - tiny linear drift
      - low AR(1) noise

    Still periodic and repeatable, but not a toy sine wave.
    """

    t = np.arange(n, dtype=float)
    s = style or {}

    # dominant daily-like cycle
    main_phase = float(s.get("main_phase", 0.0))
    main = amplitude * np.sin(2 * np.pi * t / period_steps + main_phase)

    # second harmonic (creates sharper peaks / asymmetry)
    harm_w = float(s.get("harm_w", 0.35))
    harm_phase = float(s.get("harm_phase", 0.8))
    harmonic = harm_w * amplitude * np.sin(4 * np.pi * t / period_steps + harm_phase)

    # shorter sub-cycle (different per signal automatically via amplitude scaling)
    sub_ratio = float(s.get("sub_ratio", 3.0))
    sub_w = float(s.get("sub_w", 0.25))
    sub_phase = float(s.get("sub_phase", 1.7))
    sub_period = period_steps / sub_ratio
    sub = sub_w * amplitude * np.sin(2 * np.pi * t / sub_period + sub_phase)

    # tiny drift (keeps it realistic but slow)
    drift_rate = float(s.get("drift", 0.0005))
    drift = drift_rate * t

    # correlated low noise
    noise_phi = float(s.get("noise_phi", 0.92))
    noise = ar1_noise(n, noise_sigma, noise_phi, rng)

    return level + main + harmonic + sub + drift + noise


def inject_timewarp(
    y: np.ndarray,
    win: Window,
    ramp_steps: int,
    max_shift_steps: float,
    warp_phi: float,
    rng: np.random.Generator,
    *,
    warp_smooth: float = 3.0,
    raw_shift_driver: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    "Hard for threshold detectors" anomaly:
      - does NOT push values outside the baseline envelope
      - largely preserves the marginal distribution (it's mostly re-timing)
      - breaks temporal predictability (HTM should notice)

    Mechanism:
      - Create a smooth, bounded shift(t) during the window
      - Evaluate y at (t + shift(t)) using interpolation
      - Blend with a sigmoid ramp so edges are gradual

    max_shift_steps:
      Max absolute time shift in *steps* (e.g., 2.0 means up to +/- 2 timesteps).
    warp_phi:
      AR(1) phi controlling smoothness of the shift trajectory.
    warp_smooth:
      Extra smoothing factor applied via a moving-average window (>=1).
    """
    if win.end <= win.start:
        return y
    y = np.asarray(y, dtype=float)
    n = len(y)
    w = sigmoid_ramp(n, win.start, win.end, ramp_steps)

    # Shift trajectory: allow caller to supply a shared driver so overlaps are correlated.
    if raw_shift_driver is None:
        raw = ar1_noise(n, sigma=1.0, phi=float(warp_phi), rng=rng)
    else:
        raw = np.asarray(raw_shift_driver, dtype=float)
        if raw.shape[0] != n:
            raise ValueError("raw_shift_driver must have length n")

    shift = float(max_shift_steps) * np.tanh(raw / 2.0)
    shift = shift * w  # only active in the window (with ramp)

    yw = _interp_shifted(y, shift)
    # blend: outside window => original y; inside => warped
    return (1.0 - w) * y + w * yw


def _is_overlap(w: Window) -> bool:
    return "overlap" in str(w.label).lower()


def _shared_overlap_driver(n: int, warp_phi: float, rng: np.random.Generator) -> np.ndarray:
    # One driver used for BOTH signals in an overlap window => system gate becomes satisfiable.
    return ar1_noise(n, sigma=1.0, phi=float(warp_phi), rng=rng)


# ===============================
# CLI
# ===============================

def parse_window_arg(s: str, steps_per_hour: int) -> Window:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) < 2:
        raise ValueError("Window must be start,end[,label]")

    start_h = float(parts[0])
    end_h = float(parts[1])
    label = parts[2] if len(parts) >= 3 else f"{start_h}-{end_h}"

    return Window(
        start=int(round(start_h * steps_per_hour)),
        end=int(round(end_h * steps_per_hour)),
        label=label,
    )


# ===============================
# Main
# ===============================

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--out-dir",
        default="data/demo_synth",
        help="Output directory for generated CSVs + gt_timestamps.json (repo-relative recommended).",
    )
    ap.add_argument("--hours", type=int, default=96)
    ap.add_argument("--freq", default="5min", help="Sampling interval (e.g. 5min, 15min, 30min). Larger => fewer timesteps per cycle => baseline repeats faster in step-space.")
    ap.add_argument("--seed", type=int, default=1)
    # Baseline noise control:
    # - If --baseline-noise-frac is set, baseline noise_sigma is computed as:
    #       noise_sigma = amplitude * baseline_noise_frac
    #   This is a defensible, unit-invariant way to set noise.
    # - If --zero-noise is set, baseline noise_sigma is forced to 0 regardless of other args.
    ap.add_argument("--baseline-noise-frac", type=float, default=None,
                    help="Baseline noise sigma as a fraction of the baseline amplitude (e.g. 0.05 = 5%%).")
    ap.add_argument("--zero-noise", action="store_true",
                    help="Make baseline deterministic (noise_sigma=0, drift=0) and injected jitter=0.")
    ap.add_argument("--settle-hours", type=float, default=48.0)
    ap.add_argument("--baseline-period-hours", type=float, default=24.0)
    ap.add_argument("--min-cycles-before-anomaly", type=int, default=2)

    ap.add_argument("--winA", action="append", default=[])
    ap.add_argument("--winB", action="append", default=[])
    ap.add_argument("--winC", action="append", default=[])

    ap.add_argument("--ramp-min", type=float, default=45.0)
    ap.add_argument("--envelope", choices=["minmax", "p01p99"], default="minmax",
                    help="Keep injected signals within baseline envelope (soft-squash).")
    ap.add_argument("--envelope-pad", type=float, default=0.0, help="Extra padding for envelope bounds (units).")
    ap.add_argument("--gt-mode", choices=["start", "start_end", "all"], default="start",
                    help="How to emit GT timestamps from anomaly windows.")
    # --- baseline controllability (lets us do “zero-noise” runs without code edits) ---
    ap.add_argument("--baseline-noise-sigma", type=float, default=None,
                    help="Override baseline AR(1) noise sigma for ALL signals (A/B/C). 0 => deterministic baseline.")
    ap.add_argument("--baseline-noise-phi", type=float, default=None,
                    help="Override baseline AR(1) noise phi for ALL signals (A/B/C). Ignored if sigma=0.")
    ap.add_argument("--baseline-drift", type=float, default=None,
                    help="Override baseline drift rate for ALL signals (A/B/C). 0 => no drift.")

    # --- injection controllability ---
    ap.add_argument("--inject-jitter-sigma", type=float, default=1.2,
                    help="Sigma for injected instability AR(1) jitter. 0 => no injected noise (windows become no-ops).")
    ap.add_argument("--inject-jitter-phi", type=float, default=0.6,
                    help="Phi for injected instability AR(1) jitter.")

    # --- anomaly injection (distribution-preserving, HTM-detectable, stays within envelope) ---
    # Keep flags stable so quickstart doesn’t break, but we intentionally make TIMEWARP the primary mechanism.
    ap.add_argument("--inject-mode", choices=["instability", "timewarp", "hybrid"], default="timewarp",
                    help="Deprecated modes kept for CLI compatibility. Effective injection is timewarp.")
    ap.add_argument("--inject-timewarp-max-shift-min", type=float, default=240.0,
                    help="Max time-warp shift in minutes (converted to steps). For 30min sampling: 240min => 8 steps.")
    ap.add_argument("--inject-timewarp-phi", type=float, default=0.90,
                    help="AR(1) phi for time-warp trajectory. Lower => less predictable => more HTM-detectable.")
    ap.add_argument("--inject-instability-gain", type=float, default=0.0,
                    help="Deprecated; kept for compatibility. Use 0 (default).")

    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    steps_per_hour = int(pd.Timedelta("1h") / pd.Timedelta(args.freq))
    n = args.hours * steps_per_hour
    period_steps = int(round(args.baseline_period_hours * steps_per_hour))
    settle_steps = int(round(args.settle_hours * steps_per_hour))

    # ===============================
    # Enforce baseline learnability
    # ===============================

    required_settle = (
        args.baseline_period_hours
        * args.min_cycles_before_anomaly
    )

    if args.settle_hours < required_settle:
        raise ValueError(
            f"settle-hours={args.settle_hours} too small. "
            f"Need at least {required_settle} hours "
            f"for {args.min_cycles_before_anomaly} full cycles."
        )

    ramp_steps = int(round((args.ramp_min / 60.0) * steps_per_hour))

    # ===============================
    # Generate Baselines
    # ===============================

    print(
        f"[proof] freq={args.freq} => steps_per_hour={steps_per_hour}, "
        f"period_hours={args.baseline_period_hours} => period_steps={period_steps}, "
        f"hours={args.hours} => n={n}, "
        f"settle_hours={args.settle_hours} => settle_steps={settle_steps}"
    )

    # ===============================
    # Baseline noise policy
    # ===============================
    ampA = 15.0
    ampB = 10.0
    ampC = 18.0
    if args.zero_noise:
        noiseA = 0.0
        noiseB = 0.0
        noiseC = 0.0
    elif args.baseline_noise_frac is not None:
        frac = float(args.baseline_noise_frac)
        if frac < 0:
            raise ValueError("--baseline-noise-frac must be >= 0")
        noiseA = ampA * frac
        noiseB = ampB * frac
        noiseC = ampC * frac
    else:
        # current defaults (legacy behavior)
        noiseA = 0.15
        noiseB = 0.12
        noiseC = 0.10

    # Optional override: explicit sigma for ALL signals
    if args.baseline_noise_sigma is not None:
        s = float(args.baseline_noise_sigma)
        if s < 0:
            raise ValueError("--baseline-noise-sigma must be >= 0")
        noiseA = s
        noiseB = s
        noiseC = s

    # Drift defaults (per-signal), with optional override for ALL signals
    drift_A = 0.0 if args.zero_noise else 0.0004
    drift_B = 0.0 if args.zero_noise else -0.0002
    drift_C = 0.0 if args.zero_noise else 0.0001
    if args.baseline_drift is not None:
        d = float(args.baseline_drift)
        drift_A = d
        drift_B = d
        drift_C = d

    # Noise phi defaults, with optional override for ALL signals
    phi_A = 0.92
    phi_B = 0.92
    phi_C = 0.92
    if args.baseline_noise_phi is not None:
        p = float(args.baseline_noise_phi)
        phi_A = p
        phi_B = p
        phi_C = p

    A = generate_baseline(
        n, 120, period_steps, ampA, noiseA, rng,
        style={"sub_ratio": 3.0, "harm_w": 0.35, "sub_w": 0.22, "main_phase": 0.2, "drift": drift_A, "noise_phi": phi_A}
        )
    B = generate_baseline(
        n, 60, period_steps, ampB, noiseB, rng,
        style={"sub_ratio": 4.5, "harm_w": 0.15, "sub_w": 0.35, "main_phase": 1.1, "sub_phase": 0.4, "drift": drift_B, "noise_phi": phi_B}
        )
    C = generate_baseline(
        n, 260, period_steps, ampC, noiseC, rng,
        style={"sub_ratio": 2.2, "harm_w": 0.55, "sub_w": 0.12, "main_phase": 2.0, "harm_phase": 1.6, "drift": drift_C, "noise_phi": phi_C}
        )

    envA = envelope_bounds(A, args.envelope, args.envelope_pad)
    envB = envelope_bounds(B, args.envelope, args.envelope_pad)
    envC = envelope_bounds(C, args.envelope, args.envelope_pad)

    # ===============================
    # Parse Windows
    # ===============================

    winsA = [parse_window_arg(w, steps_per_hour) for w in args.winA]
    winsB = [parse_window_arg(w, steps_per_hour) for w in args.winB]
    winsC = [parse_window_arg(w, steps_per_hour) for w in args.winC]

    for group, name in [(winsA, "A"), (winsB, "B"), (winsC, "C")]:
        for w in group:
            if w.start < settle_steps:
                raise ValueError(
                    f"{name} window starts before settle period."
                )

    # ===============================
    # Inject
    # ===============================

    A2, B2, C2 = A.copy(), B.copy(), C.copy()

    # Effective injection: TIMEWARP only (distribution-preserving + HTM-detectable).
    # We keep old CLI args but do not use instability anymore (it encourages threshold-style anomalies).

    # time-warp controls (minutes -> steps)
    max_shift_steps = float(args.inject_timewarp_max_shift_min) / 60.0 * float(steps_per_hour)
    warp_phi = float(args.inject_timewarp_phi)

    # Smoothing: keep it a “warp” not jitter, but not so smooth that HTM predicts it.
    # Make smoothing scale gently with sampling rate: ~1.5 hours of smoothing in steps.
    warp_smooth = max(1.0, round(1.5 * float(steps_per_hour)))

    # Precompute shared drivers for overlap windows by label (so A+C overlap shares driver)
    overlap_driver_by_label: Dict[str, np.ndarray] = {}
    def get_driver(w: Window) -> Optional[np.ndarray]:
        if not _is_overlap(w):
            return None
        key = str(w.label).strip()
        if key not in overlap_driver_by_label:
            overlap_driver_by_label[key] = _shared_overlap_driver(n, warp_phi, rng)
        return overlap_driver_by_label[key]

    for w in winsA:
        A2 = inject_timewarp(
            A2, w, ramp_steps,
            max_shift_steps=max_shift_steps,
            warp_phi=warp_phi,
            rng=rng,
            warp_smooth=warp_smooth,
            raw_shift_driver=get_driver(w),
        )

    for w in winsB:
        B2 = inject_timewarp(
            B2, w, ramp_steps,
            max_shift_steps=max_shift_steps,
            warp_phi=warp_phi,
            rng=rng,
            warp_smooth=warp_smooth,
            raw_shift_driver=get_driver(w),
        )

    for w in winsC:
        C2 = inject_timewarp(
            C2, w, ramp_steps,
            max_shift_steps=max_shift_steps,
            warp_phi=warp_phi,
            rng=rng,
            warp_smooth=warp_smooth,
            raw_shift_driver=get_driver(w),
        )

    # Keep injected signals inside baseline envelope
    A2 = soft_clip(A2, *envA)
    B2 = soft_clip(B2, *envB)
    C2 = soft_clip(C2, *envC)

    # ===============================
    # Plot
    # ===============================

    ts = pd.date_range("2015-03-01", periods=n, freq=args.freq)

    # ===============================
    # Ground truth timestamps (from windows)
    # ===============================

    gt = {
        "sA": gt_timestamps_for_windows(ts, winsA, args.gt_mode),
        "sB": gt_timestamps_for_windows(ts, winsB, args.gt_mode),
        "sC": gt_timestamps_for_windows(ts, winsC, args.gt_mode),
        "meta": {
            "gt_mode": args.gt_mode,
            "timestamp_format": "%Y-%m-%d %H:%M:%S",
        },
    }

    # ===============================
    # Write CSVs
    # ===============================

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "timestamp": ts,
        "value": A2,
    }).to_csv(out_dir / "sA.csv", index=False)

    pd.DataFrame({
        "timestamp": ts,
        "value": B2,
    }).to_csv(out_dir / "sB.csv", index=False)

    pd.DataFrame({
        "timestamp": ts,
        "value": C2,
    }).to_csv(out_dir / "sC.csv", index=False)

    (out_dir / "gt_timestamps.json").write_text(json.dumps(gt, indent=2))
    print(f"Wrote GT timestamps -> {(out_dir / 'gt_timestamps.json').resolve()}")

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    for ax, name, base, sig, wins in [
        (axes[0], "A", A, A2, winsA),
        (axes[1], "B", B, B2, winsB),
        (axes[2], "C", C, C2, winsC),
    ]:
        ax.plot(ts, base, alpha=0.4)
        ax.plot(ts, sig)
        for w in wins:
            ax.axvspan(ts[w.start], ts[w.end], alpha=0.2)
        ax.set_ylabel(name)

    axes[0].set_title("Proof-mode signals")
    p_out = str((out_dir / "proof.png").resolve())
    plt.tight_layout()
    plt.savefig(p_out, dpi=150)
    plt.close()

    print(f"Wrote CSVs to {out_dir.resolve()}")
    print("Generated proof.png")


if __name__ == "__main__":
    main()