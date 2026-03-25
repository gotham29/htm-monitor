#demo/eval_cmpass_fleet.py

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import pandas as pd
import yaml


@dataclass(frozen=True)
class FleetEvalPaths:
    repo_root: Path
    dataset: str
    dataset_dir: str
    scored_csv: Path
    defaults_yaml: Path
    outputs_root: Path
    generated_root: Path
    units_root: Path
    runs_root: Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    text = path.read_text()
    if not text.strip():
        raise ValueError(f"YAML at {path} is empty")

    try:
        obj = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML at {path}: {e}") from e

    if not isinstance(obj, dict):
        raise ValueError(
            f"YAML at {path} must parse to a mapping, got {type(obj).__name__}"
        )
    return obj


def _write_yaml(path: Path, obj: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(yaml.safe_dump(dict(obj), sort_keys=False))
    tmp_path.replace(path)


def _deep_merge_dicts(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    out = json.loads(json.dumps(base))
    for k, v in override.items():
        if (
            k in out
            and isinstance(out[k], dict)
            and isinstance(v, dict)
        ):
            out[k] = _deep_merge_dicts(out[k], v)
        else:
            out[k] = json.loads(json.dumps(v))
    return out


def _required_columns_present(df: pd.DataFrame, required: Sequence[str], *, ctx: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{ctx} missing required columns: {missing}")


def _load_scored(scored_csv: Path) -> pd.DataFrame:
    if not scored_csv.exists():
        raise FileNotFoundError(f"Missing scored CSV: {scored_csv}")

    df = pd.read_csv(scored_csv)
    _required_columns_present(
        df,
        ["unit_id", "cycle", "failure_cycle", "rul_at_cycle"],
        ctx=str(scored_csv),
    )

    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    if not sensor_cols:
        raise ValueError(f"No sensor_* columns found in {scored_csv}")

    df["unit_id"] = pd.to_numeric(df["unit_id"], errors="raise").astype(int)
    df["max_cycle_observed"] = pd.to_numeric(
        df["max_cycle_observed"], errors="raise"
    ).astype(int)
    df["rul"] = pd.to_numeric(df["rul"], errors="raise").astype(int)
    df["cycle"] = pd.to_numeric(df["cycle"], errors="raise").astype(int)
    df["failure_cycle"] = pd.to_numeric(df["failure_cycle"], errors="raise").astype(int)
    df["rul_at_cycle"] = pd.to_numeric(df["rul_at_cycle"], errors="raise").astype(int)

    return df.sort_values(["unit_id", "cycle"], kind="mergesort").reset_index(drop=True)


def _all_sensor_columns(df: pd.DataFrame) -> List[str]:
    return sorted([c for c in df.columns if c.startswith("sensor_")], key=lambda s: int(s.split("_")[1]))


def _infer_sensor_ranges(
    df: pd.DataFrame,
    sensors: Sequence[str],
    *,
    quantile_lo: float = 0.001,
    quantile_hi: float = 0.999,
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for s in sensors:
        if s not in df.columns:
            raise ValueError(f"Requested sensor '{s}' missing from scored dataframe")
        vals = pd.to_numeric(df[s], errors="coerce").dropna()
        if vals.empty:
            raise ValueError(f"Sensor '{s}' has no numeric values")
        lo = float(vals.quantile(quantile_lo))
        hi = float(vals.quantile(quantile_hi))
        if hi <= lo:
            lo = float(vals.min())
            hi = float(vals.max())
        if hi <= lo:
            hi = lo + 1.0
        out[s] = {"minVal": lo, "maxVal": hi}
    return out


def _build_timestamps_for_unit(unit_df: pd.DataFrame) -> List[str]:
    """
    Use a synthetic but stable hourly timestamp series so the rest of the repo can
    keep using timestamp-based labels and plots.
    """
    n = len(unit_df)
    ts = pd.date_range("2000-01-01 00:00:00", periods=n, freq="H")
    return [t.strftime("%Y-%m-%d %H:%M:%S") for t in ts]


def _write_unit_sensor_csvs(
    unit_df: pd.DataFrame,
    *,
    sensors: Sequence[str],
    unit_dir: Path,
) -> Dict[str, str]:
    unit_dir.mkdir(parents=True, exist_ok=True)

    ts_list = _build_timestamps_for_unit(unit_df)
    out_paths: Dict[str, str] = {}

    for s in sensors:
        sensor_df = pd.DataFrame(
            {
                "timestamp": ts_list,
                "value": pd.to_numeric(unit_df[s], errors="coerce"),
            }
        )
        path = unit_dir / f"{s}.csv"
        sensor_df.to_csv(path, index=False)
        out_paths[s] = str(path)

    return out_paths


def _failure_timestamp_for_unit(unit_df: pd.DataFrame) -> str:
    """
    For test-set units, the available rows stop before actual failure.
    We evaluate against the final observed cycle as the event timestamp for this first fleet pass,
    matching the current single-unit demo pattern already in the repo.
    """
    ts_list = _build_timestamps_for_unit(unit_df)
    return ts_list[-1]


def _unit_eval_steps(unit_df: pd.DataFrame, warmup_steps: int) -> int:
    n = int(len(unit_df))
    return max(0, n - int(warmup_steps))


def _unit_is_eligible_for_warning_eval(
    *,
    unit_df: pd.DataFrame,
    warmup_steps: int,
    warning_start: int,
    min_eval_steps_floor: int = 25,
) -> tuple[bool, str, int]:
    """
    Heuristic eligibility gate for fleet summary interpretation.

    Rationale:
      - if a unit has almost no post-warmup timesteps, warning-window scoring is not meaningful
      - we should still RUN it and save artifacts, but mark it in the fleet summary
    """
    eval_steps = _unit_eval_steps(unit_df, warmup_steps)
    required_steps = max(int(min_eval_steps_floor), int(warning_start))
    eligible = int(eval_steps) >= int(required_steps)
    reason = "eligible" if eligible else f"insufficient_eval_steps(<{required_steps})"
    return bool(eligible), str(reason), int(eval_steps)


def _unit_warmup_steps(unit_df: pd.DataFrame, default_warmup: int) -> int:
    """
    Ensure at least one post-warmup timestep remains for analysis.

    Rules:
      - empty unit is invalid
      - 1-row unit -> warmup 0
      - otherwise clamp warmup to at most n_rows - 1
    """
    n = int(len(unit_df))
    if n <= 0:
        raise ValueError("unit_df must contain at least one row")
    if n == 1:
        return 0
    return max(0, min(int(default_warmup), n - 1))


def _warning_contract_from_base_config(base_config: Mapping[str, Any]) -> tuple[int, int, int]:
    evaluation = base_config.get("evaluation") or {}
    predictive = evaluation.get("predictive_warning") or {}
    warning = predictive.get("warning_window") or {}

    start_steps = int(warning.get("start_steps_before_event", 60))
    end_steps = int(warning.get("end_steps_before_event", 0))

    if start_steps < 0 or end_steps < 0:
        raise ValueError("warning window steps must be >= 0")
    if end_steps > start_steps:
        raise ValueError("warning_window.end_steps_before_event must be <= start_steps_before_event")

    min_eval_steps_floor = max(25, start_steps)
    return start_steps, end_steps, min_eval_steps_floor


def _build_unit_config(
    *,
    base_config: Dict[str, Any],
    sensor_ranges: Dict[str, Dict[str, float]],
    sensors: Sequence[str],
    csv_paths_by_sensor: Dict[str, str],
    failure_timestamp: str,
    unit_id: int,
    warmup_steps: int,
) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(base_config))  # deep-ish copy via json-safe structure

    # Features
    feature_block: Dict[str, Any] = {
        "timestamp": {
            "type": "datetime",
            "format": "%Y-%m-%d %H:%M:%S",
            "timeOfDay": 21,
            "weekend": 0,
            "dayOfWeek": 0,
            "holiday": 0,
            "season": 0,
            "encode": False,
        }
    }
    for i, s in enumerate(sensors, start=101):
        feature_block[s] = {
            "type": "float",
            "size": 2048,
            "activeBits": 40,
            "numBuckets": 80,
            "minVal": float(sensor_ranges[s]["minVal"]),
            "maxVal": float(sensor_ranges[s]["maxVal"]),
            "seed": int(i),
        }
    cfg["features"] = feature_block

    # Models
    model_block: Dict[str, Any] = {}
    for s in sensors:
        model_block[f"{s}_model"] = {"sources": [s], "features": [s]}
    cfg["models"] = model_block

    # Data sources
    source_block: List[Dict[str, Any]] = []
    for s in sensors:
        source_block.append(
            {
                "name": s,
                "kind": "csv",
                "path": csv_paths_by_sensor[s],
                "timestamp_col": "timestamp",
                "timestamp_format": "%Y-%m-%d %H:%M:%S",
                "fields": {s: "value"},
                "labels": {"timestamps": [failure_timestamp]},
            }
        )

    if "data" not in cfg or not isinstance(cfg["data"], dict):
        cfg["data"] = {}
    cfg["data"]["timebase"] = {"mode": "union", "on_missing": "hold_last"}
    cfg["data"]["sources"] = source_block

    # Preserve predictive-evaluation semantics from base_config (single source of truth)
    if "evaluation" in cfg:
        cfg["evaluation"] = json.loads(json.dumps(base_config["evaluation"]))

    decision_cfg = cfg.get("decision") or {}
    decision_window = decision_cfg.get("window") or {}
    k_models = int(decision_cfg.get("k", 4))
    window_size = int(decision_window.get("size", 12))

    # Ground-truth system definition
    cfg["ground_truth"] = {
        "system": {
            "method": "kofn_window",
            "k": k_models,
            "window": {
                "size": window_size,
                "per_source_hits": 1,
            },
        }
    }

    # Keep plots off for fleet runs unless user wants them
    plot_cfg = cfg.get("plot") or {}
    if not isinstance(plot_cfg, dict):
        plot_cfg = {}
    plot_cfg["enable"] = False
    plot_cfg["show_ground_truth"] = True
    plot_cfg["show_warmup_span"] = True
    plot_cfg["step_pause_s"] = 0.0
    cfg["plot"] = plot_cfg

    run_cfg = cfg.get("run") or {}
    if not isinstance(run_cfg, dict):
        run_cfg = {}
    run_cfg["warmup_steps"] = int(warmup_steps)
    cfg["run"] = run_cfg

    if "htm_params" not in cfg:
        raise ValueError("Generated unit config is missing required top-level key 'htm_params'")
    return cfg


def _extract_unit_result(run_summary_path: Path, unit_id: int) -> Dict[str, Any]:
    if not run_summary_path.exists():
        raise FileNotFoundError(f"Missing run summary: {run_summary_path}")

    summary = json.loads(run_summary_path.read_text())
    pw = summary.get("predictive_warning_eval") or {}
    per_event = pw.get("per_event") or []
    use_case = summary.get("use_case_semantics") or {}
    first_event = per_event[0] if per_event else {}

    alerts = summary.get("alerts") or {}
    episodes = (alerts.get("episodes") or {}).get("episodes") or []
    timesteps = summary.get("timesteps") or {}

    return {
        "unit_id": int(unit_id),
        "timesteps_count": int(timesteps.get("count") or 0),
        "alert_timesteps": int(alerts.get("alert_timesteps") or 0),
        "alert_rate": float(alerts.get("alert_rate") or 0.0),
        "episode_count": int((alerts.get("episodes") or {}).get("count") or 0),
        "event_count": int(pw.get("event_count") or 0),
        "matched_events_in_warning_window": int(pw.get("matched_events_in_warning_window") or 0),
        "first_alert_classification": first_event.get("first_alert_classification"),
        "first_alert_t": first_event.get("first_alert_t"),
        "first_alert_ts": first_event.get("first_alert_ts"),
        "first_alert_lead_steps": first_event.get("first_alert_lead_steps"),
        "first_alert_lead_minutes": first_event.get("first_alert_lead_minutes"),
        "has_any_in_warning_window": bool(first_event.get("has_any_in_warning_window", False)),
        "event_t": first_event.get("event_t"),
        "event_ts": first_event.get("event_ts"),
        "run_summary_json": str(run_summary_path),
        "episode_summary_in_warning_window": int(((pw.get("episode_summary") or {}).get("in_warning_window") or 0)),
        "episode_summary_too_early": int(((pw.get("episode_summary") or {}).get("too_early") or 0)),
        "episode_summary_too_late": int(((pw.get("episode_summary") or {}).get("too_late") or 0)),
        "episode_summary_unscored": int(((pw.get("episode_summary") or {}).get("unscored") or 0)),
        "evaluation_mode": use_case.get("evaluation_mode"),
        "run_figure_png": str(run_summary_path.parent / "run_figure.png"),
    }


def _fleet_summary_dict(
    per_unit_df: pd.DataFrame,
    *,
    dataset: str,
    sensors: Sequence[str],
    warning_start: int,
    warning_end: int,
) -> Dict[str, Any]:
    if per_unit_df.empty:
        raise ValueError("per_unit_df is empty")

    cls_counts = per_unit_df["first_alert_classification"].fillna("missing").value_counts(dropna=False).to_dict()

    in_window = pd.to_numeric(per_unit_df["first_alert_lead_steps"], errors="coerce")
    in_window = in_window[per_unit_df["first_alert_classification"] == "in_warning_window"].dropna()

    lead_stats = None
    if not in_window.empty:
        lead_stats = {
            "count": int(in_window.shape[0]),
            "mean": float(in_window.mean()),
            "median": float(in_window.median()),
            "p90": float(in_window.quantile(0.90)),
            "min": float(in_window.min()),
            "max": float(in_window.max()),
        }

    n_units = int(per_unit_df["unit_id"].nunique())

    return {
        "dataset": f"NASA CMAPSS {dataset}",
        "evaluation_mode": "predictive_warning",
        "warning_window": {
            "start_steps_before_event": int(warning_start),
            "end_steps_before_event": int(warning_end),
        },
        "sensors": list(sensors),
        "eligibility": {
            "eligible_unit_count": int((per_unit_df["is_warning_eval_eligible"] == True).sum()),  # noqa: E712
            "ineligible_unit_count": int((per_unit_df["is_warning_eval_eligible"] == False).sum()),  # noqa: E712
            "eligible_rate": float((per_unit_df["is_warning_eval_eligible"] == True).sum()) / float(n_units),  # noqa: E712
            "reason_counts": {
                str(k): int(v)
                for k, v in per_unit_df["warning_eval_eligibility_reason"].fillna("missing").value_counts(dropna=False).to_dict().items()
            },
        },
        "unit_count": n_units,
        "first_alert_classification_counts": {str(k): int(v) for k, v in cls_counts.items()},
        "first_alert_classification_rates": {
            str(k): float(v) / float(n_units) for k, v in cls_counts.items()
        },
        "matched_in_window_units": int((per_unit_df["has_any_in_warning_window"] == True).sum()),  # noqa: E712
        "matched_in_window_rate": float((per_unit_df["has_any_in_warning_window"] == True).sum()) / float(n_units),  # noqa: E712
        "episode_count_stats": {
            "mean": float(per_unit_df["episode_count"].mean()),
            "median": float(per_unit_df["episode_count"].median()),
            "min": int(per_unit_df["episode_count"].min()),
            "max": int(per_unit_df["episode_count"].max()),
        },
        "alert_rate_stats": {
            "mean": float(per_unit_df["alert_rate"].mean()),
            "median": float(per_unit_df["alert_rate"].median()),
            "min": float(per_unit_df["alert_rate"].min()),
            "max": float(per_unit_df["alert_rate"].max()),
        },
        "in_window_first_alert_lead_steps_stats": lead_stats,
    }


def _fleet_markdown(summary: Mapping[str, Any], per_unit_df: pd.DataFrame) -> str:
    cls_counts = summary["first_alert_classification_counts"]
    lead_stats = summary["in_window_first_alert_lead_steps_stats"]

    lines: List[str] = []
    lines.append(f"# {summary['dataset']} Fleet Evaluation\n\n")
    lines.append(f"- Dataset: **{summary['dataset']}**\n")
    lines.append(f"- Evaluation mode: **{summary['evaluation_mode']}**\n")
    lines.append(
        f"- Warning window: **[{summary['warning_window']['start_steps_before_event']}, "
        f"{summary['warning_window']['end_steps_before_event']}] steps before failure**\n"
    )
    lines.append(f"- Sensors: `{summary['sensors']}`\n")
    lines.append(f"- Units evaluated: **{summary['unit_count']}**\n")
    lines.append(
        f"- Warning-eval eligible units: **{summary['eligibility']['eligible_unit_count']} / "
        f"{summary['unit_count']}**\n"
    )
    lines.append(f"- Matched in warning window: **{summary['matched_in_window_units']} / {summary['unit_count']}**\n\n")
    
    lines.append("## Eligibility\n\n")
    lines.append(f"- eligible units: **{summary['eligibility']['eligible_unit_count']}**\n")
    lines.append(f"- ineligible units: **{summary['eligibility']['ineligible_unit_count']}**\n")
    lines.append(f"- reason counts: `{summary['eligibility']['reason_counts']}`\n\n")

    lines.append("## First-alert classification counts\n\n")
    lines.append("| Classification | Count |\n")
    lines.append("|---|---:|\n")
    for k in sorted(cls_counts.keys()):
        lines.append(f"| {k} | {cls_counts[k]} |\n")
    lines.append("\n")

    if lead_stats is not None:
        lines.append("## In-window first-alert lead stats\n\n")
        lines.append(f"- count: **{lead_stats['count']}**\n")
        lines.append(f"- mean: **{lead_stats['mean']:.3f}** steps\n")
        lines.append(f"- median: **{lead_stats['median']:.3f}** steps\n")
        lines.append(f"- p90: **{lead_stats['p90']:.3f}** steps\n")
        lines.append(f"- min / max: **{lead_stats['min']:.3f} / {lead_stats['max']:.3f}** steps\n\n")

    lines.append("## Per-unit preview\n\n")
    preview_cols = [
        "unit_id",
        "is_warning_eval_eligible",
        "warning_eval_eligibility_reason",
        "eval_timesteps_after_warmup",
        "first_alert_classification",
        "first_alert_lead_steps",
        "has_any_in_warning_window",
        "episode_count",
        "alert_rate",
    ]
    preview = per_unit_df[preview_cols].sort_values("unit_id", kind="mergesort").head(20)

    lines.append("| unit_id | eligible | reason | eval_steps | first_alert_classification | first_alert_lead_steps | in_window | episode_count | alert_rate |\n")
    lines.append("|---:|---|---|---:|---|---:|---|---:|---:|\n")
    for _, row in preview.iterrows():
        lead = row["first_alert_lead_steps"]
        lead_txt = "—" if pd.isna(lead) else f"{float(lead):.0f}"
        unit_label = f"{int(row['unit_id'])}"
        if not bool(row["is_warning_eval_eligible"]):
            unit_label += " *"
        lines.append(
            f"| {unit_label} | {bool(row['is_warning_eval_eligible'])} | {row['warning_eval_eligibility_reason']} | "
            f"{int(row['eval_timesteps_after_warmup'])} | {row['first_alert_classification']} | {lead_txt} | "
            f"{bool(row['has_any_in_warning_window'])} | {int(row['episode_count'])} | {float(row['alert_rate']):.4f} |\n"
        )

    lines.append("\n`*` marks units that are likely too short to fairly judge this warning-window contract.\n")

    return "".join(lines)


def _run_unit_pipeline(
    *,
    defaults_yaml: Path,
    config_yaml: Path,
    run_dir: Path,
    python_exe: str,
) -> None:
    cmd = [
        python_exe,
        "-m",
        "htm_monitor.cli.run_pipeline",
        "--defaults",
        str(defaults_yaml),
        "--config",
        str(config_yaml),
        "--run-dir",
        str(run_dir),
        "--no-plot",
    ]
    subprocess.run(cmd, check=True)


def _run_unit_static_figure(
    *,
    config_yaml: Path,
    run_dir: Path,
    python_exe: str,
) -> None:
    cmd = [
        python_exe,
        "-m",
        "htm_monitor.cli.plot_run_figure",
        "--run-dir",
        str(run_dir),
        "--config",
        str(config_yaml),
        "--out",
        str(run_dir / "analysis" / "run_figure.png"),
    ]
    subprocess.run(cmd, check=True)


def _default_base_config() -> Dict[str, Any]:
    return {
        "run": {
            "warmup_steps": 40,
            "learn_after_warmup": True,
        },
        "evaluation": {
            "mode": "predictive_warning",
            "predictive_warning": {
                "event_type": "engine_failure",
                "target_source": "system",
                "warning_window": {
                    "start_steps_before_event": 60,
                    "end_steps_before_event": 0,
                },
                "too_early_window": {
                    "min_steps_before_event": 120,
                },
            },
        },
        "decision": {
            "score_key": "anomaly_probability",
            "threshold": 0.97,
            "method": "kofn_window",
            "k": 5,
            "window": {
                "size": 12,
                "per_model_hits": 2,
            },
        },
        "plot": {
            "enable": False,
            "step_pause_s": 0.0,
            "window": 220,
            "show_ground_truth": True,
            "show_warmup_span": True,
        },
    }


def _apply_cli_overrides(
    base_config: Dict[str, Any],
    *,
    threshold: float | None,
    k: int | None,
    window_size: int | None,
    per_model_hits: int | None,
    warning_start: int | None,
    warning_end: int | None,
    too_early_min: int | None,
) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(base_config))

    decision = cfg.setdefault("decision", {})
    decision_window = decision.setdefault("window", {})
    evaluation = cfg.setdefault("evaluation", {})
    predictive = evaluation.setdefault("predictive_warning", {})
    warning_window = predictive.setdefault("warning_window", {})
    too_early_window = predictive.setdefault("too_early_window", {})

    if threshold is not None:
        decision["threshold"] = float(threshold)
    if k is not None:
        if int(k) <= 0:
            raise ValueError("--k must be > 0")
        decision["k"] = int(k)
    if window_size is not None:
        if int(window_size) <= 0:
            raise ValueError("--window-size must be > 0")
        decision_window["size"] = int(window_size)
    if per_model_hits is not None:
        if int(per_model_hits) <= 0:
            raise ValueError("--per-model-hits must be > 0")
        decision_window["per_model_hits"] = int(per_model_hits)
    if warning_start is not None:
        if int(warning_start) < 0:
            raise ValueError("--warning-start must be >= 0")
        warning_window["start_steps_before_event"] = int(warning_start)
    if warning_end is not None:
        if int(warning_end) < 0:
            raise ValueError("--warning-end must be >= 0")
        warning_window["end_steps_before_event"] = int(warning_end)
    if too_early_min is not None:
        if int(too_early_min) < 0:
            raise ValueError("--too-early-min must be >= 0")
        too_early_window["min_steps_before_event"] = int(too_early_min)

    start_steps = int(warning_window.get("start_steps_before_event", 60))
    end_steps = int(warning_window.get("end_steps_before_event", 0))
    if end_steps > start_steps:
        raise ValueError(
            "warning_window.end_steps_before_event must be <= start_steps_before_event"
        )

    return cfg


def _build_paths(repo_root: Path, dataset: str, output_name: str) -> FleetEvalPaths:
    ds = str(dataset).upper()
    dataset_dir = f"cmapss_{ds.lower()}"
    outputs_root = repo_root / "outputs" / output_name
    generated_root = repo_root / "data" / dataset_dir / "generated" / output_name
    units_root = generated_root / "units"
    runs_root = outputs_root / "per_unit"
    return FleetEvalPaths(
        repo_root=repo_root,
        dataset=ds,
        dataset_dir=dataset_dir,
        scored_csv=repo_root / "data" / dataset_dir / "demo" / f"{ds.lower()}_test_scored.csv",
        defaults_yaml=repo_root / "configs" / "htm_defaults.yaml",
        outputs_root=outputs_root,
        generated_root=generated_root,
        units_root=units_root,
        runs_root=runs_root,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        required=True,
        choices=["FD001", "FD002", "FD003", "FD004"],
        help="CMAPSS subset to evaluate",
    )
    ap.add_argument(
        "--scored-csv",
        default=None,
        help="Override path to data/cmapss_<fdxxx>/demo/<fdxxx>_test_scored.csv",
    )
    ap.add_argument(
        "--defaults",
        default=None,
        help="Override path to configs/htm_defaults.yaml",
    )
    ap.add_argument(
        "--output-name",
        default=None,
        help="Name for outputs/ and generated/ artifact roots "
             "(default: cmapss_<dataset>_fleet_eval)",
    )
    ap.add_argument(
        "--units",
        nargs="*",
        type=int,
        default=None,
        help="Optional explicit unit ids to evaluate (default: all units in scored CSV)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of units after filtering/sorting",
    )
    ap.add_argument(
        "--sensors",
        nargs="*",
        default=None,
        help="Sensors to include in per-unit configs. Default: all sensor_* columns in scored CSV.",
    )
    ap.add_argument(
        "--python-exe",
        default=sys.executable,
        help="Python executable to use for subprocess runs",
    )
    ap.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip per-unit static figure generation",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override decision.threshold",
    )
    ap.add_argument(
        "--k",
        type=int,
        default=None,
        help="Override decision.k",
    )
    ap.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Override decision.window.size",
    )
    ap.add_argument(
        "--per-model-hits",
        type=int,
        default=None,
        help="Override decision.window.per_model_hits",
    )
    ap.add_argument(
        "--warning-start",
        type=int,
        default=None,
        help="Override evaluation.predictive_warning.warning_window.start_steps_before_event",
    )
    ap.add_argument(
        "--warning-end",
        type=int,
        default=None,
        help="Override evaluation.predictive_warning.warning_window.end_steps_before_event",
    )
    ap.add_argument(
        "--too-early-min",
        type=int,
        default=None,
        help="Override evaluation.predictive_warning.too_early_window.min_steps_before_event",
    )
    args = ap.parse_args()

    repo = _repo_root()
    dataset = str(args.dataset).upper()
    output_name = args.output_name or f"cmapss_{dataset.lower()}_fleet_eval"
    paths = _build_paths(repo, dataset, output_name)

    scored_csv = Path(args.scored_csv) if args.scored_csv else paths.scored_csv
    defaults_yaml = Path(args.defaults) if args.defaults else paths.defaults_yaml

    paths.outputs_root.mkdir(parents=True, exist_ok=True)
    paths.generated_root.mkdir(parents=True, exist_ok=True)
    paths.units_root.mkdir(parents=True, exist_ok=True)
    paths.runs_root.mkdir(parents=True, exist_ok=True)

    df = _load_scored(scored_csv)
    if args.sensors is None or len(args.sensors) == 0:
        sensors = _all_sensor_columns(df)
    else:
        sensors = list(args.sensors)

    if not sensors:
        raise ValueError("Resolved sensor list is empty")
    sensor_ranges = _infer_sensor_ranges(df, sensors)

    unit_ids = sorted(df["unit_id"].unique().tolist())
    if args.units:
        req = sorted(set(int(x) for x in args.units))
        missing = [u for u in req if u not in unit_ids]
        if missing:
            raise ValueError(f"Requested unit ids missing from scored CSV: {missing}")
        unit_ids = req

    if args.limit is not None:
        unit_ids = unit_ids[: int(args.limit)]

    defaults_cfg = _load_yaml(defaults_yaml)
    cli_base = _apply_cli_overrides(
        _default_base_config(),
        threshold=args.threshold,
        k=args.k,
        window_size=args.window_size,
        per_model_hits=args.per_model_hits,
        warning_start=args.warning_start,
        warning_end=args.warning_end,
        too_early_min=args.too_early_min,
    )
    base_config = _deep_merge_dicts(defaults_cfg, cli_base)

    required_top_keys = ["htm_params", "run", "decision", "plot", "evaluation"]
    missing = [k for k in required_top_keys if k not in base_config]
    if missing:
        raise ValueError(f"Base config missing required keys after defaults merge: {missing}")
    per_unit_rows: List[Dict[str, Any]] = []

    warning_start, warning_end, min_eval_steps_floor = _warning_contract_from_base_config(
        base_config
    )

    for unit_id in unit_ids:
        unit_df = df[df["unit_id"] == unit_id].sort_values("cycle", kind="mergesort").reset_index(drop=True)
        if unit_df.empty:
            raise ValueError(f"Unit {unit_id} produced empty dataframe slice")

        unit_data_dir = paths.units_root / f"unit_{unit_id:03d}"
        unit_run_dir = paths.runs_root / f"unit_{unit_id:03d}"
        unit_cfg_path = unit_data_dir / "config.yaml"

        csv_paths_by_sensor = _write_unit_sensor_csvs(
            unit_df,
            sensors=sensors,
            unit_dir=unit_data_dir,
        )
        failure_timestamp = _failure_timestamp_for_unit(unit_df)
        default_warmup = int(((base_config.get("run") or {}).get("warmup_steps", 0)) or 0)
        warmup_steps = _unit_warmup_steps(unit_df, default_warmup)

        is_eligible, eligibility_reason, eval_steps_after_warmup = _unit_is_eligible_for_warning_eval(
            unit_df=unit_df,
            warmup_steps=warmup_steps,
            warning_start=warning_start,
            min_eval_steps_floor=min_eval_steps_floor,
        )

        unit_cfg = _build_unit_config(
            base_config=base_config,
            sensor_ranges=sensor_ranges,
            sensors=sensors,
            csv_paths_by_sensor=csv_paths_by_sensor,
            failure_timestamp=failure_timestamp,
            unit_id=unit_id,
            warmup_steps=warmup_steps,
        )
        _write_yaml(unit_cfg_path, unit_cfg)

        print(f"[fleet] unit {unit_id:03d}: running pipeline (warmup_steps={warmup_steps}, rows={len(unit_df)})")
        _run_unit_pipeline(
            defaults_yaml=defaults_yaml,
            config_yaml=unit_cfg_path,
            run_dir=unit_run_dir,
            python_exe=args.python_exe,
        )
        if not args.skip_figures:
            print(f"[fleet] unit {unit_id:03d}: building static figure")
            _run_unit_static_figure(
                config_yaml=unit_cfg_path,
                run_dir=unit_run_dir,
                python_exe=args.python_exe,
            )
        unit_summary_path = unit_run_dir / "analysis" / "run_summary.json"
        row = _extract_unit_result(unit_summary_path, unit_id)
        row["n_rows"] = int(len(unit_df))
        row["warmup_steps"] = int(warmup_steps)
        row["eval_timesteps_after_warmup"] = int(eval_steps_after_warmup)
        row["is_warning_eval_eligible"] = bool(is_eligible)
        row["warning_eval_eligibility_reason"] = str(eligibility_reason)
        row["failure_timestamp"] = str(failure_timestamp)
        per_unit_rows.append(row)

    per_unit_df = pd.DataFrame(per_unit_rows).sort_values("unit_id", kind="mergesort").reset_index(drop=True)

    fleet_summary = _fleet_summary_dict(
        per_unit_df,
        dataset=dataset,
        sensors=sensors,
        warning_start=warning_start,
        warning_end=warning_end,
    )

    eligible_df = per_unit_df[per_unit_df["is_warning_eval_eligible"] == True].copy()  # noqa: E712
    eligible_summary = None
    if not eligible_df.empty:
        eligible_summary = _fleet_summary_dict(
            eligible_df,
            dataset=dataset,
            sensors=sensors,
            warning_start=warning_start,
            warning_end=warning_end,
        )

    per_unit_csv = paths.outputs_root / "fleet_summary.csv"
    summary_json = paths.outputs_root / "fleet_summary.json"
    summary_md = paths.outputs_root / "fleet_summary.md"
    eligible_summary_json = paths.outputs_root / "fleet_summary_eligible_only.json"
    eligible_summary_md = paths.outputs_root / "fleet_summary_eligible_only.md"

    per_unit_df.to_csv(per_unit_csv, index=False)
    summary_json.write_text(json.dumps(fleet_summary, indent=2))
    summary_md.write_text(_fleet_markdown(fleet_summary, per_unit_df))
    if eligible_summary is not None:
        eligible_summary_json.write_text(json.dumps(eligible_summary, indent=2))
        eligible_summary_md.write_text(_fleet_markdown(eligible_summary, eligible_df))

    print(f"[fleet] wrote -> {per_unit_csv}")
    print(f"[fleet] wrote -> {summary_json}")
    print(f"[fleet] wrote -> {summary_md}")
    if eligible_summary is not None:
        print(f"[fleet] wrote -> {eligible_summary_json}")
        print(f"[fleet] wrote -> {eligible_summary_md}")


if __name__ == "__main__":
    main()
