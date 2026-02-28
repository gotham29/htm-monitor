# demo/run_pipeline.py

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Set, Tuple

from htm_monitor.utils.config import build_from_config

from demo.live_plot import LivePlot


def run(
    stream: Iterable[Mapping],
    engine,
    decision,
    on_update: Optional[Callable] = None,
):
    for t, row in enumerate(stream):
        model_outputs = engine.step(row, timestep=t)
        result = decision.step(model_outputs)

        if on_update:
            on_update(t, row, model_outputs, result)

        yield result


@dataclass(frozen=True)
class _SourceCfg:
    name: str
    path: str
    timestamp_col: str
    timestamp_format: str
    fields: Dict[str, str]  # canonical_name -> column_name
    gt_timestamps: Optional[Set[str]] = None  # inline ground truth timestamps


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


def _parse_gt_timestamps(labels: Mapping[str, Any], ts_format: str) -> Optional[Set[str]]:
    """
    Parse labels.timestamps as a set of timestamp strings.
    We keep strings because the plot uses the CSV's raw timestamp strings.
    Validate format by attempting datetime.strptime for each entry.
    """
    if not isinstance(labels, Mapping):
        return None
    ts_list = labels.get("timestamps")
    if ts_list is None:
        return None
    if not isinstance(ts_list, list):
        raise ValueError("labels.timestamps must be a list of timestamp strings")

    out: Set[str] = set()
    for v in ts_list:
        if not isinstance(v, str):
            raise ValueError("labels.timestamps entries must be strings")
        _parse_ts(v, ts_format)  # validation
        out.add(v)
    return out


def _load_sources(cfg: dict) -> Dict[str, _SourceCfg]:
    out: Dict[str, _SourceCfg] = {}
    for s in cfg["data"]["sources"]:
        labels = s.get("labels") or {}
        gt = _parse_gt_timestamps(labels, s["timestamp_format"])
        out[s["name"]] = _SourceCfg(
            name=s["name"],
            path=s["path"],
            timestamp_col=s["timestamp_col"],
            timestamp_format=s["timestamp_format"],
            fields=dict(s.get("fields") or {}),
            gt_timestamps=gt,
        )
    return out


def _load_gt_set(sources: Dict[str, _SourceCfg], enabled: bool) -> Optional[set]:
    if not enabled:
        return None
    gt: set = set()  # union GT across sources (keeps plot generic)
    for s in sources.values():
        if s.gt_timestamps:
            gt.update(s.gt_timestamps)
    return gt


def _load_gt_by_source(sources: Dict[str, _SourceCfg], enabled: bool) -> Optional[Dict[str, set]]:
    if not enabled:
        return None
    out: Dict[str, set] = {}
    for name, s in sources.items():
        if s.gt_timestamps:
            out[name] = set(s.gt_timestamps)
    return out


def _ts_any(rows_by_source: Mapping[str, Mapping[str, Any]]) -> Optional[str]:
    for r in rows_by_source.values():
        if r and r.get("timestamp"):
            return r.get("timestamp")
    return None


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
    defaults_path: str,
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
) -> Path:
    gt_by_source = {name: sorted(list(sc.gt_timestamps or [])) for name, sc in sources.items() if sc.gt_timestamps}
    gt_all: List[str] = sorted({ts for v in gt_by_source.values() for ts in v})

    manifest = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "python": {"version": sys.version.split()[0]},
        "inputs": {"defaults": defaults_path, "config": config_path},
        "data": {
            "timebase_mode": (cfg.get("data", {}).get("timebase", {}).get("mode") or "union"),
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
            "threshold": (cfg.get("decision", {}) or {}).get("threshold"),
            "k": (cfg.get("decision", {}) or {}).get("k"),
            "window": (cfg.get("decision", {}) or {}).get("window"),
        },
        "run": {
            "out_csv": str(out_csv),
            "timesteps": {"min": int(t_min), "max": int(t_max), "count": int(t_max - t_min + 1) if t_max >= t_min else 0},
            "time_range": {"start": ts_start, "end": ts_end},
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
    args = ap.parse_args()

    _configure_logging(args.log_level, args.log_file)

    cfg, engine, decision, model_sources = build_from_config(args.defaults, args.config)

    run_paths = _resolve_run_paths(
        defaults_path=args.defaults,
        config_path=args.config,
        out=args.out,
        run_dir=args.run_dir,
        run_name=args.run_name,
    )

    model_value_fields: Dict[str, List[str]] = {}
    for model_name, mcfg in (cfg.get("models") or {}).items():
        # All non-timestamp features for this model (used only for plotting)
        feats = list(mcfg.get("features") or [])
        model_value_fields[model_name] = [f for f in feats if f != "timestamp"]

    sources = _load_sources(cfg)
    iters = {name: _csv_events(sc) for name, sc in sources.items()}
    mode = (cfg.get("data", {}).get("timebase", {}).get("mode") or "union").lower()
    if mode not in ("union", "intersection"):
        raise ValueError("data.timebase.mode must be 'union' or 'intersection'")
    stream = _timebase_intersection(iters) if mode == "intersection" else _timebase_union(iters)

    plot_cfg = cfg.get("plot") or {}
    plot = None
    gt_by_source = _load_gt_by_source(sources, bool(plot_cfg.get("show_ground_truth")))
    plot_enabled_effective = bool(plot_cfg.get("enable")) and (not args.no_plot)
    if plot_enabled_effective:
        gt_set = _load_gt_set(sources, bool(plot_cfg.get("show_ground_truth")))
        plot = LivePlot(
            window=int(plot_cfg.get("window", 1000)),
            refresh_every=1,  # you want every step visible
            gt_timestamps=gt_set,
            likelihood_threshold=getattr(decision, "threshold", None),
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
                "raw", "p", "likelihood", "score",
                "system_score", "alert",
                "hot_by_model",
            ],
        )
        w.writeheader()

        def on_update(t, rows_by_source, model_outputs, result):
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

                # Ground truth per model (derived from the model's source labels)
                gt_by_model: Optional[Dict[str, set]] = None
                if gt_by_source:
                    gt_by_model = {}
                    for model_name, srcs in model_sources.items():
                        acc: Set[str] = set()
                        for s in srcs:
                            sgt = gt_by_source.get(s)
                            if sgt:
                                acc.update(sgt)
                        if acc:
                            gt_by_model[model_name] = acc

                plot_row: Dict[str, Any] = {
                    "timestamp": _ts_any(rows_by_source),
                    "values_by_model": values_by_model,
                    "gt_by_model": gt_by_model,
                }
                plot.update(t, plot_row, model_outputs, result)
                if step_pause > 0:
                    time.sleep(step_pause)

            sys_score = result.get("system_score") if isinstance(result, dict) else None
            alert = result.get("alert") if isinstance(result, dict) else None
            hot_by_model = result.get("hot_by_model") if isinstance(result, dict) else None
            hot_by_model_json = json.dumps(hot_by_model, sort_keys=True) if hot_by_model is not None else None

            # write one line per model output
            score_key = getattr(decision, "score_key", None)
            score_key = score_key if isinstance(score_key, str) and score_key else "p"
            for model_name, out in model_outputs.items():
                score_val = out.get(score_key)
                w.writerow(
                    {
                        "t": t,
                        "timestamp": ts_any,
                        "model": model_name,
                        "raw": out.get("raw"),
                        "p": out.get("p"),
                        "likelihood": out.get("likelihood"),
                        "system_score": sys_score,
                        "score": score_val,
                        "alert": alert,
                        "hot_by_model": hot_by_model_json,
                    }
                )

        for _ in run(stream, engine, decision, on_update=on_update):
            pass

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
        )
        print(f"[run] wrote -> {mp}")
    else:
        print("[run] warning: no rows processed; manifest not written.")


if __name__ == "__main__":
    main()
