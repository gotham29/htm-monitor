# src/htm_monitor/cli/run_pipeline.py

from __future__ import annotations
import argparse
import csv
import json
import sys
import logging
import time
import yaml
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Set, Tuple

from htm_monitor.utils.config import build_from_config
from htm_monitor.diagnostics.run_diagnostics import RunDiagnostics, open_diag_writers

from htm_monitor.viz.live_plot import LivePlot
from htm_monitor.cli.analyze_run import load_run as _load_run_csv, summarize as _summarize_run


def run(
    stream: Iterable[Mapping],
    engine,
    decision,
    on_update: Optional[Callable] = None,
    *,
    diag: Optional[RunDiagnostics] = None,
    warmup_steps: int = 0,
    learn_after_warmup: bool = True,
):
    warmup_steps = int(warmup_steps)
    if warmup_steps < 0:
        raise ValueError("run.warmup_steps must be >= 0")

    for t, row in enumerate(stream):
        in_warmup = t < warmup_steps
        learn = True if in_warmup else bool(learn_after_warmup)

        model_outputs = engine.step(row, timestep=t, diag=diag, learn=learn)

        # IMPORTANT: do not feed warmup steps into decision buffers.
        if in_warmup:
            result = {
                "alert": 0,
                "system_hot": 0,
                "system_hot_count": 0,
                "system_hot_streak": 0,
                "instant_hot_by_model": {},
                "model_warmth_by_model": {},
                "group_instant_count": {},
                "group_warmth": {},
                "group_hot": {},
            }
        else:
            result = decision.step(model_outputs)

        if on_update:
            on_update(t, row, model_outputs, result, in_warmup=in_warmup, learn=learn)

        yield result


@dataclass(frozen=True)
class _SourceCfg:
    name: str
    path: str
    timestamp_col: str
    timestamp_format: str
    fields: Dict[str, str]  # canonical_name -> column_name
    unit: Optional[str] = None
    event_windows: Optional[List[Dict[str, str]]] = None  # {name,start,end,kind}


def _parse_ts(s: str, fmt: str) -> datetime:
    return datetime.strptime(s, fmt)


def _csv_events(src: _SourceCfg) -> Iterator[Tuple[datetime, Dict[str, Any]]]:
    with open(src.path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            ts_raw = row.get(src.timestamp_col)
            if not ts_raw:
                continue
            ts = _parse_ts(ts_raw, src.timestamp_format)
            out: Dict[str, Any] = {"timestamp": ts_raw}
            for canon, col in src.fields.items():
                v = row.get(col)
                if v is None or v == "":
                    out[canon] = None
                else:
                    out[canon] = float(v)
            yield ts, out


def _timebase_union(iters: Dict[str, Iterator[Tuple[datetime, Dict[str, Any]]]]):
    heads: Dict[str, Optional[Tuple[datetime, Dict[str, Any]]]] = {k: next(v, None) for k, v in iters.items()}
    while True:
        active = {k: h for k, h in heads.items() if h is not None}
        if not active:
            return
        t_min = min(h[0] for h in active.values())
        rows: Dict[str, Dict[str, Any]] = {}
        for name, h in list(active.items()):
            if h[0] == t_min:
                rows[name] = h[1]
                heads[name] = next(iters[name], None)
        yield rows


def _timebase_intersection(iters: Dict[str, Iterator[Tuple[datetime, Dict[str, Any]]]]):
    heads: Dict[str, Optional[Tuple[datetime, Dict[str, Any]]]] = {k: next(v, None) for k, v in iters.items()}
    while True:
        if any(h is None for h in heads.values()):
            return
        t_max = max(h[0] for h in heads.values() if h is not None)
        advanced = False
        for name, h in list(heads.items()):
            while h is not None and h[0] < t_max:
                h = next(iters[name], None)
                heads[name] = h
                advanced = True
            if h is None:
                return
        if advanced:
            continue
        # all equal
        rows = {name: h[1] for name, h in heads.items() if h is not None}
        for name in heads.keys():
            heads[name] = next(iters[name], None)
        yield rows


def _parse_event_windows(labels: Mapping[str, Any], ts_format: str) -> Optional[List[Dict[str, str]]]: 
    if not isinstance(labels, Mapping):
        return None
    evs = labels.get("event_windows")
    if evs is None:
         return None
    if not isinstance(evs, list):
        raise ValueError("labels.event_windows must be a list")

    out: List[Dict[str, str]] = []
    for i, ev in enumerate(evs):
        if not isinstance(ev, Mapping):
            raise ValueError(f"labels.event_windows[{i}] must be a mapping")
        start = ev.get("start")
        end = ev.get("end")
        name = ev.get("name", "")
        kind = str(ev.get("kind") or "primary_gt").strip() or "primary_gt"
        if kind not in {"primary_gt", "explanatory"}:
            raise ValueError(
                f"labels.event_windows[{i}].kind must be 'primary_gt' or 'explanatory'"
            )
        if not isinstance(start, str) or not start.strip():
            raise ValueError(f"labels.event_windows[{i}].start must be a non-empty string")
        if not isinstance(end, str) or not end.strip():
            raise ValueError(f"labels.event_windows[{i}].end must be a non-empty string")
        _parse_ts(start, ts_format)
        _parse_ts(end, ts_format)
        out.append(
            {
                "name": str(name).strip() if isinstance(name, str) else "",
                "start": start,
                "end": end,
                "kind": kind,
            }
        )
    return out

def _load_sources(cfg: dict) -> Dict[str, _SourceCfg]:
    out: Dict[str, _SourceCfg] = {}
    for s in cfg["data"]["sources"]:
        labels = s.get("labels") or {}
        event_windows = _parse_event_windows(labels, s["timestamp_format"])
        out[s["name"]] = _SourceCfg(
            name=s["name"],
            path=s["path"],
            timestamp_col=s["timestamp_col"],
            timestamp_format=s["timestamp_format"],
            unit=str(s.get("unit")).strip() if s.get("unit") is not None else None,
            fields=dict(s.get("fields") or {}),
            event_windows=event_windows,
        )
    return out


def _ts_in_event_windows(ts: Optional[str], event_windows: Optional[List[Dict[str, str]]]) -> bool:
    if ts is None or not event_windows:
        return False
    for ev in event_windows:
        start = ev.get("start")
        end = ev.get("end")
        if start is not None and end is not None and start <= ts <= end:
            return True
    return False


def _load_gt_by_source(
    sources: Dict[str, _SourceCfg],
    enabled: bool,
    *,
    primary_only: bool = False,
) -> Optional[Dict[str, List[Dict[str, str]]]]:
    if not enabled:
        return None
    out: Dict[str, List[Dict[str, str]]] = {}
    for name, s in sources.items():
        if not s.event_windows:
            continue
        wins = list(s.event_windows)
        if primary_only:
            wins = [w for w in wins if str(w.get("kind") or "primary_gt") == "primary_gt"]
        if wins:
            out[name] = wins
    return out


def _pad_limits(vmin: float, vmax: float) -> Tuple[float, float]:
    if vmax < vmin:
        vmin, vmax = vmax, vmin
    span = vmax - vmin
    if span <= 0.0:
        pad = max(1.0, abs(vmin) * 0.05, 0.5)
    else:
        pad = max(span * 0.08, 0.5)
    return (vmin - pad, vmax + pad)


def _source_feature_minmax(src: _SourceCfg) -> Dict[str, Tuple[float, float]]:
    """
    Scan one source CSV and compute min/max for each canonical feature listed in src.fields.
    Returns canonical feature -> (min, max).
    """
    out: Dict[str, Tuple[float, float]] = {}
    mins: Dict[str, float] = {}
    maxs: Dict[str, float] = {}

    with open(src.path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            for canon, col in src.fields.items():
                raw = row.get(col)
                if raw is None or raw == "":
                    continue
                try:
                    val = float(raw)
                except Exception:
                    continue
                if canon not in mins or val < mins[canon]:
                    mins[canon] = val
                if canon not in maxs or val > maxs[canon]:
                    maxs[canon] = val

    for canon in mins.keys():
        out[canon] = (mins[canon], maxs[canon])
    return out


def _build_live_value_y_lims(
    *,
    sources: Dict[str, _SourceCfg],
    model_sources: Dict[str, List[str]],
    model_value_fields: Dict[str, List[str]],
) -> Dict[str, Tuple[float, float]]:
    """
    Compute fixed y-limits for each model's value axis from the full source CSV(s),
    so the live plot remains visually stable and never clips late peaks.
    """
    per_source_feature_mm: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for src_name, src_cfg in sources.items():
        per_source_feature_mm[src_name] = _source_feature_minmax(src_cfg)

    out: Dict[str, Tuple[float, float]] = {}
    for model_name, feat_names in model_value_fields.items():
        srcs = model_sources.get(model_name) or []
        vals_min: List[float] = []
        vals_max: List[float] = []

        for feat in feat_names:
            for src_name in srcs:
                mm = per_source_feature_mm.get(src_name, {}).get(feat)
                if mm is None:
                    continue
                vals_min.append(mm[0])
                vals_max.append(mm[1])

        if vals_min and vals_max:
            out[model_name] = _pad_limits(min(vals_min), max(vals_max))

    return out


def _ts_any(rows_by_source: Mapping[str, Mapping[str, Any]]) -> Optional[str]:
    for r in rows_by_source.values():
        if r and r.get("timestamp"):
            return r.get("timestamp")
    return None


def _load_predictive_warning_cfg(config_path: str) -> Optional[Dict[str, Any]]:
    cfg = yaml.safe_load(Path(config_path).read_text())
    if not isinstance(cfg, dict):
        return None
    evaluation = cfg.get("evaluation") or {}
    if not isinstance(evaluation, dict):
        return None
    if str(evaluation.get("mode") or "").strip() != "predictive_warning":
        return None
    pw = evaluation.get("predictive_warning")
    return pw if isinstance(pw, dict) else None


def _predictive_live_plot_context(
    *,
    config_path: str,
    sources: Dict[str, "_SourceCfg"],
    stream_factory: Callable[[], Iterable[Mapping[str, Mapping[str, Any]]]],
) -> Dict[str, Optional[int]]:
    """
    Precompute failure_t and warning-window t bounds for live plotting.

    Assumption for current CMAPSS single-unit demo:
      - the predictive target event is a shared failure timestamp present in all sources
      - mapping timestamp -> t can be recovered from the same timebase stream used by the run
    """
    pw = _load_predictive_warning_cfg(config_path)
    if not isinstance(pw, dict):
        return {"failure_t": None, "warning_start_t": None, "warning_end_t": None}

    ww = pw.get("warning_window") or {}
    start_before = ww.get("start_steps_before_event")
    end_before = ww.get("end_steps_before_event")
    if not isinstance(start_before, int) or not isinstance(end_before, int):
        return {"failure_t": None, "warning_start_t": None, "warning_end_t": None}

    # current demo assumption: all source GT timestamps point to the same failure time
    all_gt = sorted({ts for s in sources.values() for ts in (s.gt_timestamps or set())})
    if not all_gt:
        return {"failure_t": None, "warning_start_t": None, "warning_end_t": None}
    failure_ts = all_gt[-1]

    failure_t: Optional[int] = None
    for t, rows_by_source in enumerate(stream_factory()):
        ts_any = _ts_any(rows_by_source)
        if ts_any == failure_ts:
            failure_t = int(t)
            break

    if failure_t is None:
        return {"failure_t": None, "warning_start_t": None, "warning_end_t": None}

    return {
        "failure_t": int(failure_t),
        "warning_start_t": int(failure_t - start_before),
        "warning_end_t": int(failure_t - end_before),
    }


def _configure_logging(level: str, log_file: Optional[str]) -> None:
    lvl = getattr(logging, (level or "INFO").upper(), None)
    if not isinstance(lvl, int):
        raise ValueError(f"Invalid --log-level '{level}'. Try INFO, WARNING, DEBUG.")

    handlers = []
    # Always log to console so you can see traces interactively.
    handlers.append(logging.StreamHandler())
    if log_file:
        handlers.append(logging.FileHandler(log_file, mode="w"))

    logging.basicConfig(
        level=lvl,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )


def _derive_run_name(config_path: str) -> str:
    # configs/real_tweets_trio.yaml -> real_tweets_trio
    return Path(config_path).stem


@dataclass(frozen=True)
class _RunPaths:
    run_dir: Optional[Path]          # provided or derived; only set in run-dir mode
    out_csv: Path
    manifest: Path
    analysis_dir: Optional[Path]     # only meaningful in run-dir mode


def _resolve_run_paths(
    *,
    config_path: str,
    out: Optional[str],
    run_dir: Optional[str],
    run_name: Optional[str],
) -> _RunPaths:
    # Precedence:
    #   1) --run-dir (canonical run-folder mode)
    #   2) legacy --out (file-first mode; manifest next to csv)
    if run_dir:
        rd = Path(run_dir)
        name = run_name or _derive_run_name(config_path)
        # If user passed a directory that doesn't already include the name,
        # we respect it as-is (no extra nesting); run_name is mainly for auto defaulting.
        rd.mkdir(parents=True, exist_ok=True)
        out_csv = rd / "run.csv"
        manifest = rd / "run.manifest.json"
        analysis_dir = rd / "analysis"
        return _RunPaths(run_dir=rd, out_csv=out_csv, manifest=manifest, analysis_dir=analysis_dir)

    # Legacy: out points to a CSV file path.
    out_csv = Path(out or "outputs/out.csv")
    if out_csv.exists() and out_csv.is_dir():
        raise ValueError(
            f"--out expects a CSV file path, but got a directory: {out_csv}\n"
            "Use --run-dir <dir> for canonical run-folder mode, or pass --out <dir>/run.csv."
        )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest = out_csv.with_suffix(".manifest.json")
    return _RunPaths(run_dir=None, out_csv=out_csv, manifest=manifest, analysis_dir=None)


def _write_manifest(
    out_csv: Path,
    manifest_path: Path,
    run_paths: Optional[_RunPaths],
    defaults_path: str,
    config_path: str,
    cfg: dict,
    sources: Dict[str, _SourceCfg],
    model_sources: Dict[str, List[str]],
    plot_enabled: bool,
    plot_was_skipped: bool,
    t_min: int,
    t_max: int,
    ts_start: Optional[str],
    ts_end: Optional[str],
    warmup_steps: int,
    learn_after_warmup: bool,
    decision_score_key_effective: str,
) -> Path:
    gt_by_source = {name: list(sc.event_windows or []) for name, sc in sources.items() if sc.event_windows}
    gt_all: List[Dict[str, str]] = []
    for windows in gt_by_source.values():
        gt_all.extend(windows)

    manifest = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "python": {"version": sys.version.split()[0]},
        "inputs": {"defaults": defaults_path, "config": config_path},
        "data": {
            "timebase_mode": (cfg.get("data", {}).get("timebase", {}).get("mode") or "union"),
            "on_missing": (cfg.get("data", {}).get("timebase", {}).get("on_missing") or "hold_last"),
            "sources": {
                name: {
                    "path": sc.path,
                    "timestamp_col": sc.timestamp_col,
                    "timestamp_format": sc.timestamp_format,
                    "fields": sc.fields,
                }
                for name, sc in sources.items()
            },
            "ground_truth": {"by_source": gt_by_source, "all": gt_all},
        },
        "models": {
            "names": sorted(list(model_sources.keys())),
            "model_sources": model_sources,
        },
        "decision": {
            "method": (cfg.get("decision", {}) or {}).get("method"),
            "score_key": (cfg.get("decision", {}) or {}).get("score_key"),
            "score_key_effective": str(decision_score_key_effective),
            "threshold": (cfg.get("decision", {}) or {}).get("threshold"),
            "model_warmth": (cfg.get("decision", {}) or {}).get("model_warmth"),
            "system": (cfg.get("decision", {}) or {}).get("system"),
            "groups": (cfg.get("decision", {}) or {}).get("groups"),
        },
        "run": {
            "out_csv": str(out_csv),
            "timesteps": {"min": int(t_min), "max": int(t_max), "count": int(t_max - t_min + 1) if t_max >= t_min else 0},
            "time_range": {"start": ts_start, "end": ts_end},
            "warmup_steps": int(warmup_steps),
            "learn_after_warmup": bool(learn_after_warmup),
        },
        "plot": {"enabled_in_config": bool((cfg.get("plot") or {}).get("enable")), "enabled_effective": bool(plot_enabled), "skipped_by_cli": bool(plot_was_skipped)},
    }

    # If we're in run-dir mode, add a tiny stable artifact contract (relative names).
    if run_paths is not None and run_paths.run_dir is not None:
        manifest["artifacts"] = {
            "run_csv": "run.csv",
            "manifest": "run.manifest.json",
            "analysis_dir": "analysis",
        }
        manifest["run"]["run_dir"] = str(run_paths.run_dir)

    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def _validate_disabled_feature_not_in_models(cfg: dict) -> None:
    """
    Defensive invariant:
      If a feature has encode: false, it must not appear in any model's features list.

    Why:
      The HTM model may legitimately treat encode:false features as "metadata-only" (0-bit SDR),
      but some diagnostic paths build per-feature encoding maps only for encoded features.
      Keeping disabled features in model feature lists can cause KeyError / misalignment.
    """
    feats = cfg.get("features") or {}
    if not isinstance(feats, dict):
        return

    disabled: Set[str] = set()
    for fname, fcfg in feats.items():
        if isinstance(fcfg, dict) and (fcfg.get("encode") is False):
            disabled.add(str(fname))

    if not disabled:
        return

    models = cfg.get("models") or {}
    if not isinstance(models, dict):
        return

    offenders: List[str] = []
    for mname, mcfg in models.items():
        if not isinstance(mcfg, dict):
            continue
        flist = mcfg.get("features") or []
        if not isinstance(flist, list):
            continue
        bad = sorted([f for f in flist if f in disabled])
        if bad:
            offenders.append(f"{mname}: {bad}")

    if offenders:
        raise ValueError(
            "Config invariant violated: feature(s) have encode:false but still appear in model feature lists.\n"
            f"  disabled_features={sorted(disabled)}\n"
            "  offenders:\n    - " + "\n    - ".join(offenders) + "\n"
            "Fix: remove these features from cfg.models.<model>.features (or set encode:true)."
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--defaults", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="outputs/out.csv", help="Legacy: write CSV to this file (manifest written next to it)")
    ap.add_argument("--run-dir", default=None, help="Canonical: write artifacts into this directory (run.csv, run.manifest.json)")
    ap.add_argument("--run-name", default=None, help="Optional label for the run (mostly for UX / future defaults)")
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--log-file", default=None)
    ap.add_argument("--no-plot", action="store_true", help="Skip live plotting (useful for large runs)")
    ap.add_argument(
        "--no-analyze",
        action="store_true",
        help="Skip post-run analysis (run_summary.json/md). Default: analyze in --run-dir mode.",
    )
    ap.add_argument(
        "--analyze-max-lag-steps",
        type=int,
        default=50,
        help="Max detection lag in steps for GT matching (used by analyze_run).",
    )
    ap.add_argument(
        "--analyze-non-strict",
        action="store_true",
        help="Allow missing/empty GT in analysis without raising.",
    )
    ap.add_argument(
        "--diag-encoding",
        default=None,
        help="Write per-feature encoding stability diagnostics to this CSV (default: <run-dir>/analysis/encoding_diag.csv when --run-dir is used)",
    )
    ap.add_argument(
        "--diag-tm",
        default=None,
        help="Write TM predictive-vs-active diagnostics to this CSV (default: <run-dir>/analysis/tm_diag.csv when --run-dir is used)",
    )
    args = ap.parse_args()

    _configure_logging(args.log_level, args.log_file)

    cfg, engine, decision, model_sources = build_from_config(args.defaults, args.config)

    # Hard invariant check (fail fast with a clear message)
    _validate_disabled_feature_not_in_models(cfg)

    # --- Run lifecycle policy (explicit) ---
    # Defaults preserve current behavior (warmup=0, learning stays on).
    run_cfg = cfg.get("run") or {}
    if not isinstance(run_cfg, dict):
        raise ValueError("Config.run must be a mapping if provided")
    warmup_steps = int(run_cfg.get("warmup_steps", 0) or 0)
    if warmup_steps < 0:
        raise ValueError("run.warmup_steps must be >= 0")
    learn_after_warmup = bool(run_cfg.get("learn_after_warmup", True))

    # Freeze effective decision score key (what we actually write into 'score' column)
    decision_score_key_effective = getattr(decision, "score_key", None)
    if not isinstance(decision_score_key_effective, str) or not decision_score_key_effective.strip():
        decision_score_key_effective = "p"
    decision_score_key_effective = str(decision_score_key_effective).strip()

    # Normalize "semantic" config names to the actual keys produced by engine.step()
    # (This is THE place to do it, so plotting and CSV writing are consistent.)
    _SCORE_KEY_ALIASES = {
        "anomaly_probability": "p",
        "probability": "p",
        "p": "p",
        "anomaly_score": "raw",
        "raw": "raw",
        "likelihood": "likelihood",
    }
    decision_score_key_engine = _SCORE_KEY_ALIASES.get(decision_score_key_effective, decision_score_key_effective)

    run_paths = _resolve_run_paths(
        config_path=args.config,
        out=args.out,
        run_dir=args.run_dir,
        run_name=args.run_name,
    )

    # --- diagnostics outputs (artifact-grade evidence) ---
    # Defaults only apply in run-dir mode; legacy mode requires explicit paths.
    enc_path: Optional[Path] = None
    tm_path: Optional[Path] = None
    if args.diag_encoding is not None:
        enc_path = Path(args.diag_encoding)
    elif run_paths.analysis_dir is not None:
        enc_path = run_paths.analysis_dir / "encoding_diag.csv"

    if args.diag_tm is not None:
        tm_path = Path(args.diag_tm)
    elif run_paths.analysis_dir is not None:
        tm_path = run_paths.analysis_dir / "tm_diag.csv"

    # If user did not request diagnostics and we're not in run-dir mode, keep disabled.
    enable_diag = (args.diag_encoding is not None) or (args.diag_tm is not None) or (run_paths.analysis_dir is not None)
    diag: Optional[RunDiagnostics] = None
    diag_handles: Dict[str, Any] = {}
    if enable_diag:
        # If run-dir mode, ensure analysis dir exists for the default paths
        if run_paths.analysis_dir is not None:
            run_paths.analysis_dir.mkdir(parents=True, exist_ok=True)
        enc_w, tm_w, handles = open_diag_writers(encoding_path=enc_path, tm_path=tm_path)
        diag = RunDiagnostics(encoding_writer=enc_w, tm_writer=tm_w)
        diag_handles = handles

    model_value_fields: Dict[str, List[str]] = {}
    for model_name, mcfg in (cfg.get("models") or {}).items():
        # All non-timestamp features for this model (used only for plotting)
        feats = list(mcfg.get("features") or [])
        model_value_fields[model_name] = [f for f in feats if f != "timestamp"]

    sources = _load_sources(cfg)
    mode = (cfg.get("data", {}).get("timebase", {}).get("mode") or "union").lower()
    if mode not in ("union", "intersection"):
        raise ValueError("data.timebase.mode must be 'union' or 'intersection'")

    model_units: Dict[str, str] = {}
    for model_name, srcs in model_sources.items():
        unit_val: Optional[str] = None
        for s in (srcs or []):
            sc = sources.get(s)
            if sc is not None and isinstance(sc.unit, str) and sc.unit.strip():
                unit_val = sc.unit.strip()
                break
        if unit_val:
            model_units[model_name] = unit_val

    def _make_stream():
        iters = {name: _csv_events(sc) for name, sc in sources.items()}
        return _timebase_intersection(iters) if mode == "intersection" else _timebase_union(iters)

    stream = _make_stream()

    plot_cfg = cfg.get("plot") or {}
    plot = None
    show_gt = bool(plot_cfg.get("show_ground_truth"))
    # Live plot should only show primary GT windows.
    # Explanatory windows remain part of offline analysis semantics.
    gt_by_source = _load_gt_by_source(sources, show_gt, primary_only=True)
    plot_enabled_effective = bool(plot_cfg.get("enable")) and (not args.no_plot)

    predictive_plot_ctx = _predictive_live_plot_context(
        config_path=args.config,
        sources=sources,
        stream_factory=_make_stream,
    )

    live_value_y_lims: Dict[str, Tuple[float, float]] = {}
    if plot_enabled_effective:
        live_value_y_lims = _build_live_value_y_lims(
            sources=sources,
            model_sources=model_sources,
            model_value_fields=model_value_fields,
        )

    if plot_enabled_effective:
        show_warmup_span = bool(plot_cfg.get("show_warmup_span", True))
        max_label_len = int(plot_cfg.get("max_label_len", 16) or 16)
        raw_label_colors = plot_cfg.get("model_label_colors") or {}
        if raw_label_colors is not None and not isinstance(raw_label_colors, dict):
            raise ValueError("plot.model_label_colors must be a mapping if provided")
        model_label_colors = {str(k): str(v) for k, v in dict(raw_label_colors).items()} if isinstance(raw_label_colors, dict) else {}
        rec_cfg = (plot_cfg.get("record") or {}) if isinstance(plot_cfg.get("record"), dict) else {}
        rec_enable = bool(rec_cfg.get("enable", False))
        rec_dir = rec_cfg.get("dir")
        rec_every = int(rec_cfg.get("every", 1) or 1)
        rec_dpi = int(rec_cfg.get("dpi", 140) or 140)

    decision_groups_cfg = ((cfg.get("decision") or {}).get("groups") or {})
    if decision_groups_cfg is not None and not isinstance(decision_groups_cfg, dict):
        raise ValueError("decision.groups must be a mapping if provided")

    model_group_membership: Dict[str, str] = {}
    if isinstance(decision_groups_cfg, dict):
        for gname, gspec in decision_groups_cfg.items():
            if not isinstance(gspec, dict):
                continue
            members = gspec.get("members") or []
            if not isinstance(members, list):
                continue
            for m in members:
                model_group_membership[str(m)] = str(gname)

    if plot_enabled_effective:
        plot = LivePlot(
            window=int(plot_cfg.get("window", 300)),
            refresh_every=1,
            value_y_lims=live_value_y_lims,
            plot_score_key="p",  # IMPORTANT: engine emits p; do NOT pass anomaly_probability here
            max_label_len=max_label_len,
            model_units=model_units,
            model_label_colors=model_label_colors,
            model_group_membership=model_group_membership,
            show_warmup_span=show_warmup_span,
            record_dir=str(rec_dir) if (rec_enable and rec_dir) else None,
            record_every=rec_every,
            record_dpi=rec_dpi,
        )

    step_pause = float(plot_cfg.get("step_pause_s", 0.0) or 0.0)

    outp = run_paths.out_csv

    # We'll capture basic run boundaries for the manifest
    seen_any = False
    ts_first: Optional[str] = None
    ts_last: Optional[str] = None
    t_first: Optional[int] = None
    t_last: Optional[int] = None

    with open(outp, "w", newline="") as g:
        w = csv.DictWriter(
            g,
            fieldnames=[
                "t", "timestamp", "model",
                "in_warmup", "learn",
                "raw", "p", "likelihood", "score",
                "system_hot", "system_hot_count", "system_hot_streak", "alert",
                "instant_hot_by_model",
                "model_warmth_by_model",
                "group_instant_count",
                "group_warmth",
                "group_hot",
            ],
        )
        w.writeheader()

        def on_update(t, rows_by_source, model_outputs, result, *, in_warmup: bool, learn: bool):
            nonlocal seen_any, ts_first, ts_last, t_first, t_last
            ts_any = _ts_any(rows_by_source)
            if not seen_any:
                seen_any = True
                t_first = int(t)
                ts_first = ts_any
            t_last = int(t)
            ts_last = ts_any
            # ----- Plot (generic) -----
            if plot is not None:
                # Build a plot payload:
                # values_by_model: model -> {feature -> value}
                values_by_model: Dict[str, Dict[str, Optional[float]]] = {}
                for model_name in model_outputs.keys():
                    feats = model_value_fields.get(model_name) or []
                    srcs = model_sources.get(model_name) or []
                    per_feat: Dict[str, Optional[float]] = {}

                    for f in feats:
                        v: Optional[float] = None
                        for s in srcs:
                            r = rows_by_source.get(s)
                            if isinstance(r, Mapping) and r.get(f) is not None:
                                rv = r.get(f)
                                v = float(rv) if isinstance(rv, (int, float)) else None
                                break
                        per_feat[f] = v

                    values_by_model[model_name] = per_feat

                gt_by_model: Optional[Dict[str, float]] = None
                gt_system_flag = False
                if gt_by_source and ts_any is not None:
                    gt_by_model = {}
                    for model_name, srcs in model_sources.items():
                        model_flag = any(
                            _ts_in_event_windows(ts_any, gt_by_source.get(s))
                            for s in srcs
                        )
                        gt_by_model[model_name] = 1.0 if model_flag else 0.0

                    gt_sources_hot = sum(
                        1
                        for s in sources.keys()
                        if _ts_in_event_windows(ts_any, gt_by_source.get(s))
                    )
                    gt_cfg = (cfg.get("ground_truth") or {}).get("system") or {}
                    gt_k = int(gt_cfg.get("k", 2) or 2)
                    gt_system_flag = gt_sources_hot >= gt_k

                plot_row: Dict[str, Any] = {
                    "timestamp": _ts_any(rows_by_source),
                    "values_by_model": values_by_model,
                    "gt_by_model": gt_by_model,
                    "gt_system": 1.0 if gt_system_flag else 0.0,
                    "in_warmup": bool(in_warmup),
                }
                # Predictive-warning context for live plot overlays
                if predictive_plot_ctx.get("failure_t") is not None:
                    plot_row["failure_t"] = int(predictive_plot_ctx["failure_t"])
                if (
                    predictive_plot_ctx.get("warning_start_t") is not None
                    and predictive_plot_ctx.get("warning_end_t") is not None
                ):
                    plot_row["warning_window"] = {
                        "start_t": int(predictive_plot_ctx["warning_start_t"]),
                        "end_t": int(predictive_plot_ctx["warning_end_t"]),
                    }
                plot.update(t, plot_row, model_outputs, result)
                if step_pause > 0:
                    time.sleep(step_pause)

            system_hot = result.get("system_hot") if isinstance(result, dict) else None
            system_hot_count = result.get("system_hot_count") if isinstance(result, dict) else None
            system_hot_streak = result.get("system_hot_streak") if isinstance(result, dict) else None

            alert = result.get("alert") if isinstance(result, dict) else None
            instant_hot_by_model = result.get("instant_hot_by_model") if isinstance(result, dict) else None
            model_warmth_by_model = result.get("model_warmth_by_model") if isinstance(result, dict) else None
            group_instant_count = result.get("group_instant_count") if isinstance(result, dict) else None
            group_warmth = result.get("group_warmth") if isinstance(result, dict) else None
            group_hot = result.get("group_hot") if isinstance(result, dict) else None

            instant_hot_by_model_json = json.dumps(instant_hot_by_model, sort_keys=True) if instant_hot_by_model is not None else None
            model_warmth_by_model_json = json.dumps(model_warmth_by_model, sort_keys=True) if model_warmth_by_model is not None else None
            group_instant_count_json = json.dumps(group_instant_count, sort_keys=True) if group_instant_count is not None else None
            group_warmth_json = json.dumps(group_warmth, sort_keys=True) if group_warmth is not None else None
            group_hot_json = json.dumps(group_hot, sort_keys=True) if group_hot is not None else None

            # write one line per model output
            score_key = decision_score_key_engine
            for model_name, out in model_outputs.items():
                score_val = out.get(score_key)
                w.writerow(
                    {
                        "t": t,
                        "timestamp": ts_any,
                        "model": model_name,
                        "in_warmup": int(bool(in_warmup)),
                        "learn": int(bool(learn)),
                        "raw": out.get("raw"),
                        "p": out.get("p"),
                        "likelihood": out.get("likelihood"),
                        "score": score_val,
                        "system_hot": system_hot,
                        "system_hot_count": system_hot_count,
                        "system_hot_streak": system_hot_streak,
                        "alert": alert,
                        "instant_hot_by_model": instant_hot_by_model_json,
                        "model_warmth_by_model": model_warmth_by_model_json,
                        "group_instant_count": group_instant_count_json,
                        "group_warmth": group_warmth_json,
                        "group_hot": group_hot_json,
                    }
                )

        for _ in run(
            stream,
            engine,
            decision,
            on_update=on_update,
            diag=diag,
            warmup_steps=warmup_steps,
            learn_after_warmup=learn_after_warmup,
        ):
            pass

    # close diag handles
    for h in diag_handles.values():
        h.close()

    if plot is not None and run_paths.analysis_dir is not None:
        final_plot_path = run_paths.analysis_dir / "live_plot_final.png"
        plot.save_snapshot(str(final_plot_path))
        print(f"[run] wrote -> {final_plot_path}")

    print(f"[run] wrote -> {outp}")
    # Write companion manifest JSON next to the CSV for downstream analyze_run / README snapshots.
    if seen_any and t_first is not None and t_last is not None:
        mp = _write_manifest(
            out_csv=outp,
            manifest_path=run_paths.manifest,
            run_paths=run_paths,
            defaults_path=args.defaults,
            config_path=args.config,
            cfg=cfg,
            sources=sources,
            model_sources=model_sources,
            plot_enabled=plot_enabled_effective,
            plot_was_skipped=bool(args.no_plot),
            t_min=t_first,
            t_max=t_last,
            ts_start=ts_first,
            ts_end=ts_last,
            warmup_steps=warmup_steps,
            learn_after_warmup=learn_after_warmup,
            decision_score_key_effective=decision_score_key_effective,
        )
        print(f"[run] wrote -> {mp}")
    else:
        print("[run] warning: no rows processed; manifest not written.")

    # --- Post-run analysis (artifact-first) ---
    # Default behavior: if --run-dir is used, emit analysis unless explicitly disabled.
    should_analyze = (run_paths.analysis_dir is not None) and (not bool(args.no_analyze))
    if should_analyze:
        out_dir = run_paths.analysis_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        df = _load_run_csv(str(outp))
        has_any_gt = any(bool(sc.event_windows) for sc in sources.values())
        _summarize_run(
            df,
            config_path=args.config,
            out_dir=str(out_dir),
            max_lag_steps=int(args.analyze_max_lag_steps),
            threshold_override=None,
            prefer_hot_by_model=True,
            strict=(
                (not bool(args.analyze_non_strict))
                and bool(has_any_gt)
            ),
        )
        print(f"[run] wrote -> {out_dir / 'run_summary.json'}")
        print(f"[run] wrote -> {out_dir / 'run_summary.md'}")


if __name__ == "__main__":
    main()
