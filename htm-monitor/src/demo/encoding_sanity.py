from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class RangeSummary:
    feature: str
    q_low: float
    q_high: float
    min_val: float
    max_val: float
    clip_low_rate: float
    clip_high_rate: float


def summarize_range(
    s: pd.Series,
    *,
    feature: str,
    low_q: float = 0.01,
    high_q: float = 0.99,
    margin: float = 0.03,
    floor: Optional[float] = None,
    ceil: Optional[float] = None,
) -> RangeSummary:
    """
    Compute a stable encoder range from data via quantiles + margin.
    Also report clip rates for sanity checking.
    """
    if not (0.0 < low_q < high_q < 1.0):
        raise ValueError("low_q/high_q must satisfy 0 < low_q < high_q < 1")
    if margin < 0.0:
        raise ValueError("margin must be >= 0")

    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) == 0:
        raise ValueError(f"[encoding_sanity] '{feature}': no numeric values")

    ql = float(x.quantile(low_q))
    qh = float(x.quantile(high_q))
    span = qh - ql
    if span <= 0:
        eps = max(abs(ql) * 1e-6, 1e-6)
        mn = ql - eps
        mx = qh + eps
    else:
        pad = float(margin) * span
        mn = ql - pad
        mx = qh + pad

    if floor is not None:
        mn = max(mn, float(floor))
    if ceil is not None:
        mx = min(mx, float(ceil))
    if mx <= mn:
        eps2 = max(abs(mn) * 1e-6, 1e-6)
        mn, mx = mn - eps2, mx + eps2

    # clip rates against the proposed encoder range
    arr = x.to_numpy(dtype=float)
    clip_low = float(np.mean(arr < mn))
    clip_high = float(np.mean(arr > mx))

    return RangeSummary(
        feature=str(feature),
        q_low=ql,
        q_high=qh,
        min_val=float(mn),
        max_val=float(mx),
        clip_low_rate=clip_low,
        clip_high_rate=clip_high,
    )


def write_linear_hist_png(
    s: pd.Series,
    *,
    summary: RangeSummary,
    out_png: Path,
    bins: int = 80,
    title_suffix: str = "",
) -> Path:
    """
    Simple matplotlib histogram with range markers. No seaborn, no style games.
    """
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) == 0:
        raise ValueError(f"[encoding_sanity] '{summary.feature}': no numeric values")

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.hist(x.to_numpy(dtype=float), bins=int(bins))
    plt.axvline(summary.q_low, linestyle="--")
    plt.axvline(summary.q_high, linestyle="--")
    plt.axvline(summary.min_val, linestyle="-")
    plt.axvline(summary.max_val, linestyle="-")
    t = f"{summary.feature}"
    if title_suffix:
        t += f" ({title_suffix})"
    plt.title(t)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    return out_png
