#cli/plot_fleet_figure.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


CLASS_ORDER = {
    "in_warning_window": 0,
    "too_early": 1,
    "too_late": 2,
    "no_alert": 3,
    "missing": 4,
}


CLASS_LABELS = {
    "in_warning_window": "In warning window",
    "too_early": "Too early",
    "too_late": "Too late",
    "no_alert": "No alert",
    "missing": "Missing",
}


CLASS_COLORS = {
    "in_warning_window": "#2ca02c",  # green
    "too_early": "#ff7f0e",          # orange
    "too_late": "#d62728",           # red
    "no_alert": "#7f7f7f",           # gray
    "missing": "#9467bd",            # purple
}


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    return pd.read_csv(path)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON: {path}")
    return json.loads(path.read_text())


def _infer_dataset_label(
    *,
    fleet_summary_csv: Path,
    fleet_df: pd.DataFrame,
    explicit_title: str | None,
) -> str:
    """
    Prefer dataset label from sibling fleet_summary.json, then from the first
    run_summary.json, then fall back to the explicit title or a generic label.
    """
    if explicit_title:
        return str(explicit_title)

    sibling_json = fleet_summary_csv.with_suffix(".json")
    if sibling_json.exists():
        try:
            obj = _load_json(sibling_json)
            ds = obj.get("dataset")
            if isinstance(ds, str) and ds.strip():
                return f"HTM-Monitor {ds.strip()} fleet summary"
        except Exception:
            pass

    if not fleet_df.empty and "run_summary_json" in fleet_df.columns:
        try:
            first_summary_path = Path(str(fleet_df.iloc[0]["run_summary_json"]))
            first_summary = _load_json(first_summary_path)
            use_case = first_summary.get("use_case_semantics") or {}
            ds = use_case.get("dataset") or first_summary.get("dataset")
            if isinstance(ds, str) and ds.strip():
                return f"HTM-Monitor {ds.strip()} fleet summary"
        except Exception:
            pass

    return "HTM-Monitor CMAPSS fleet summary"


def _dedup_steps(df: pd.DataFrame) -> pd.DataFrame:
    out = df.drop_duplicates("t", keep="first").copy()
    out = out.sort_values("t", kind="mergesort").reset_index(drop=True)
    return out


def _resolve_run_csv(row: pd.Series) -> Path:
    """
    fleet_summary.csv stores run_summary_json, which should be:
      <run_dir>/analysis/run_summary.json
    So run.csv is:
      <run_dir>/run.csv
    """
    run_summary_json = Path(str(row["run_summary_json"]))
    run_dir = run_summary_json.parent.parent
    run_csv = run_dir / "run.csv"
    if not run_csv.exists():
        raise FileNotFoundError(f"Could not resolve run.csv from {run_summary_json}")
    return run_csv


def _extract_alert_segments(one: pd.DataFrame, event_t: int) -> List[Tuple[float, float]]:
    """
    Convert alert-active timesteps into lead-space segments:
      x = event_t - t
    so failure is always at x = 0.
    """
    if one.empty:
        return []

    one = one.sort_values("t", kind="mergesort").reset_index(drop=True)
    t_vals = pd.to_numeric(one["t"], errors="coerce").astype(int).tolist()
    alert_vals = pd.to_numeric(one["alert"], errors="coerce").fillna(0).astype(int).tolist()

    segs: List[Tuple[float, float]] = []
    start_t: Optional[int] = None
    prev_t: Optional[int] = None

    for t, a in zip(t_vals, alert_vals):
        if a == 1 and start_t is None:
            start_t = int(t)
            prev_t = int(t)
            continue

        if a == 1 and start_t is not None:
            if prev_t is not None and int(t) == prev_t + 1:
                prev_t = int(t)
            else:
                # close prior segment
                lead_left = float(event_t - start_t)
                lead_right = float(event_t - (prev_t + 1))
                segs.append((lead_left, lead_right))
                start_t = int(t)
                prev_t = int(t)
            continue

        if a == 0 and start_t is not None:
            lead_left = float(event_t - start_t)
            lead_right = float(event_t - (prev_t + 1))
            segs.append((lead_left, lead_right))
            start_t = None
            prev_t = None

    if start_t is not None and prev_t is not None:
        lead_left = float(event_t - start_t)
        lead_right = float(event_t - (prev_t + 1))
        segs.append((lead_left, lead_right))

    # sort from early -> late in lead space
    segs = sorted(segs, key=lambda x: (-x[0], -x[1]))
    return segs


def _first_alert_lead_from_segments(segs: List[Tuple[float, float]]) -> Optional[float]:
    if not segs:
        return None
    # earliest alert = largest lead
    return max(s[0] for s in segs)


def _unit_plot_rows(fleet_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for _, r in fleet_df.iterrows():
        unit_id = int(r["unit_id"])
        event_t_raw = r.get("event_t")
        if pd.isna(event_t_raw):
            event_t = None
        else:
            event_t = int(event_t_raw)

        run_csv = _resolve_run_csv(r)
        run_df = _load_csv(run_csv)
        one = _dedup_steps(run_df)

        segs: List[Tuple[float, float]] = []
        if event_t is not None and "alert" in one.columns and "t" in one.columns:
            segs = _extract_alert_segments(one, event_t=event_t)

        classification = str(r.get("first_alert_classification", "missing"))
        if classification not in CLASS_ORDER:
            classification = "missing"

        first_alert_lead = r.get("first_alert_lead_steps")
        if pd.isna(first_alert_lead):
            first_alert_lead = _first_alert_lead_from_segments(segs)

        rows.append(
            {
                "unit_id": unit_id,
                "classification": classification,
                "is_warning_eval_eligible": bool(r["is_warning_eval_eligible"]),
                "warning_eval_eligibility_reason": str(r["warning_eval_eligibility_reason"]),
                "eval_timesteps_after_warmup": int(r["eval_timesteps_after_warmup"]),
                "event_t": event_t,
                "first_alert_lead_steps": first_alert_lead,
                "segments": segs,
                "n_rows": int(r["n_rows"]),
                "warmup_steps": int(r["warmup_steps"]),
            }
        )

    out = pd.DataFrame(rows)

    # Sorting for readability:
    # 1) eligible first
    # 2) then by classification
    # 3) then by first-alert lead descending (earlier alerts higher in group)
    # 4) then by unit id
    def _lead_sort_val(x: Any) -> float:
        if pd.isna(x):
            return -1e9
        return float(x)

    out["_eligible_sort"] = (~out["is_warning_eval_eligible"]).astype(int)
    out["_class_sort"] = out["classification"].map(CLASS_ORDER).fillna(999).astype(int)
    out["_lead_sort"] = out["first_alert_lead_steps"].map(_lead_sort_val)

    out = out.sort_values(
        ["_eligible_sort", "_class_sort", "_lead_sort", "unit_id"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    return out


def _classification_counts(df: pd.DataFrame) -> pd.Series:
    s = df["classification"].fillna("missing")
    counts = s.value_counts()
    ordered = [k for k in CLASS_ORDER.keys() if k in counts.index]
    return counts.reindex(ordered).fillna(0).astype(int)


def _draw_fleet_timeline(
    ax,
    unit_df: pd.DataFrame,
    *,
    warning_start: int,
    warning_end: int,
) -> None:
    n = len(unit_df)
    y_positions = list(range(n))

    # warning band
    ax.axvspan(warning_end, warning_start, color="magenta", alpha=0.10, zorder=0)

    for i, (_, row) in enumerate(unit_df.iterrows()):
        y = i
        cls = str(row["classification"])
        color = CLASS_COLORS.get(cls, CLASS_COLORS["missing"])
        eligible = bool(row["is_warning_eval_eligible"])

        # visually distinguish ineligible units
        row_alpha = 1.0 if eligible else 0.30
        row_lw = 4.0 if eligible else 2.0

        # faint row background for ineligible units
        if not eligible:
            ax.axhspan(y - 0.42, y + 0.42, color="lightgray", alpha=0.18, zorder=0)

        segs = row["segments"] or []
        for lead_left, lead_right in segs:
            # width is positive because lead_left > lead_right
            ax.broken_barh(
                [(lead_right, lead_left - lead_right)],
                (y - 0.34, 0.68),
                facecolors=color,
                edgecolors=color,
                alpha=row_alpha,
                linewidth=row_lw,
                zorder=3,
            )

        # first alert marker, if present
        fa = row["first_alert_lead_steps"]
        if pd.notna(fa):
            ax.plot(
                [float(fa)],
                [y],
                marker="o",
                markersize=4.5 if eligible else 3.5,
                color=color,
                alpha=row_alpha,
                zorder=4,
            )

        # failure at x=0
        ax.plot([0], [y], marker="|", markersize=8, color="black", alpha=0.9, zorder=5)

    ax.set_yticks(y_positions)
    y_labels = []
    for _, row in unit_df.iterrows():
        label = f"{int(row['unit_id']):03d}"
        if not bool(row["is_warning_eval_eligible"]):
            label += "*"
        y_labels.append(label)
    ax.set_yticklabels(y_labels, fontsize=7)

    ax.set_xlim(max(warning_start + 20, 130), -2)
    ax.set_ylim(-1, n)
    ax.set_xlabel("Steps before failure")
    ax.set_ylabel("Unit")
    ax.set_title("Fleet alert timeline (all units; * = ineligible)")
    ax.grid(True, axis="x", alpha=0.25)
    ax.invert_yaxis()


def _draw_lead_hist(ax, unit_df: pd.DataFrame, *, warning_start: int, warning_end: int) -> None:
    eligible = unit_df[unit_df["is_warning_eval_eligible"] == True].copy()  # noqa: E712
    leads = pd.to_numeric(eligible["first_alert_lead_steps"], errors="coerce").dropna()

    if not leads.empty:
        bins = list(range(0, max(int(leads.max()) + 5, warning_start + 5), 5))
        if len(bins) < 3:
            bins = 10
        ax.hist(leads, bins=bins, color="tab:blue", alpha=0.75, edgecolor="white")
    ax.axvspan(warning_end, warning_start, color="magenta", alpha=0.10)
    ax.axvline(warning_start, color="magenta", linestyle="--", linewidth=1.5)
    ax.axvline(warning_end, color="magenta", linestyle="--", linewidth=1.5)
    ax.set_xlim(max(warning_start + 20, 130), -2)
    ax.set_xlabel("First-alert lead steps")
    ax.set_ylabel("Eligible unit count")
    ax.set_title("Eligible-unit first-alert lead distribution")
    ax.grid(True, axis="y", alpha=0.25)


def _draw_scorecard(ax, unit_df: pd.DataFrame) -> None:
    all_counts = _classification_counts(unit_df)
    eligible_counts = _classification_counts(unit_df[unit_df["is_warning_eval_eligible"] == True])  # noqa: E712

    labels = [CLASS_LABELS[k] for k in all_counts.index]
    colors = [CLASS_COLORS[k] for k in all_counts.index]
    xs = list(range(len(labels)))
    width = 0.38

    ax.bar(
        [x - width / 2 for x in xs],
        all_counts.values,
        width=width,
        color=colors,
        alpha=0.45,
        label="All units",
    )
    ax.bar(
        [x + width / 2 for x in xs],
        [int(eligible_counts.get(k, 0)) for k in all_counts.index],
        width=width,
        color=colors,
        alpha=0.95,
        label="Eligible only",
    )

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Unit count")
    ax.set_title("Outcome scorecard")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fleet-summary-csv", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--title",
        default=None,
        help="Optional explicit title. Default: infer dataset from summary JSON.",
    )
    args = ap.parse_args()

    fleet_csv = Path(args.fleet_summary_csv)
    fleet_df = _load_csv(fleet_csv)

    required_cols = {
        "unit_id",
        "first_alert_classification",
        "first_alert_lead_steps",
        "event_t",
        "run_summary_json",
        "is_warning_eval_eligible",
        "warning_eval_eligibility_reason",
        "eval_timesteps_after_warmup",
        "n_rows",
        "warmup_steps",
    }
    missing = required_cols - set(fleet_df.columns)
    if missing:
        raise ValueError(f"fleet summary missing required columns: {sorted(missing)}")

    # infer warning window from the first unit's run_summary.json
    first_summary_path = Path(str(fleet_df.iloc[0]["run_summary_json"]))
    first_summary = _load_json(first_summary_path)
    pw = (first_summary.get("predictive_warning_eval") or {}).get("contract") or {}
    ww = pw.get("warning_window") or {}
    warning_start = int(ww.get("start_steps_before_event", 60))
    warning_end = int(ww.get("end_steps_before_event", 0))

    unit_df = _unit_plot_rows(fleet_df)

    out_path = Path(args.out) if args.out else fleet_csv.with_name("fleet_figure.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 20))
    gs = fig.add_gridspec(3, 1, height_ratios=[8.5, 2.2, 2.0], hspace=0.25)

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[2, 0])

    _draw_fleet_timeline(ax0, unit_df, warning_start=warning_start, warning_end=warning_end)
    _draw_lead_hist(ax1, unit_df, warning_start=warning_start, warning_end=warning_end)
    _draw_scorecard(ax2, unit_df)

    eligible_n = int((unit_df["is_warning_eval_eligible"] == True).sum())  # noqa: E712
    total_n = int(len(unit_df))
    matched_n = int(
        (
            (unit_df["classification"] == "in_warning_window")
            & (unit_df["is_warning_eval_eligible"] == True)  # noqa: E712
        ).sum()
    )

    resolved_title = _infer_dataset_label(
        fleet_summary_csv=fleet_csv,
        fleet_df=fleet_df,
        explicit_title=args.title,
    )

    fig.suptitle(
        f"{resolved_title}\nEligible units: {eligible_n}/{total_n} | "
        f"In-window first alerts among eligible: {matched_n}/{eligible_n}",
        fontsize=16,
        y=0.995,
    )
    fig.subplots_adjust(left=0.12, right=0.98, top=0.96, bottom=0.05)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
