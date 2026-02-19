# demo/run_pipeline.py

import argparse
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, Optional, Tuple

from htm_monitor.orchestration.engine import Engine
from htm_monitor.orchestration.decision import Decision
from htm_monitor.utils.config import build_from_config, load_yaml, merge_dicts

from demo.live_plot import LivePlot


def run(
    stream: Iterable[Mapping],
    engine: Engine,
    decision: Decision,
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
    labels_path: Optional[str] = None
    labels_series_key: Optional[str] = None


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


def _load_sources(cfg: dict) -> Dict[str, _SourceCfg]:
    out: Dict[str, _SourceCfg] = {}
    for s in cfg["data"]["sources"]:
        labels = s.get("labels") or {}
        out[s["name"]] = _SourceCfg(
            name=s["name"],
            path=s["path"],
            timestamp_col=s["timestamp_col"],
            timestamp_format=s["timestamp_format"],
            fields=dict(s.get("fields") or {}),
            labels_path=labels.get("combined_labels_path"),
            labels_series_key=labels.get("series_key"),
        )
    return out


def _load_gt_set(sources: Dict[str, _SourceCfg], enabled: bool) -> Optional[set]:
    if not enabled:
        return None
    # union GT across sources (keeps plot generic)
    gt: set = set()
    for s in sources.values():
        if not s.labels_path or not s.labels_series_key:
            continue
        labels = json.loads(Path(s.labels_path).read_text())
        gt.update(labels.get(s.labels_series_key, []))
    return gt


def _model_value_field(model_features: Iterable[str]) -> Optional[str]:
    """
    Pick the model's "value-like" feature for plotting.
    Simple rule: first feature name that is not 'timestamp'.
    """
    for f in model_features:
        if f != "timestamp":
            return f
    return None


def _load_gt_by_source(sources: Dict[str, _SourceCfg], enabled: bool) -> Optional[Dict[str, set]]:
    if not enabled:
        return None
    out: Dict[str, set] = {}
    for name, s in sources.items():
        if not s.labels_path or not s.labels_series_key:
            continue
        labels = json.loads(Path(s.labels_path).read_text())
        out[name] = set(labels.get(s.labels_series_key, []))
    return out


def _ts_any(rows_by_source: Mapping[str, Mapping[str, Any]]) -> Optional[str]:
    for r in rows_by_source.values():
        if r and r.get("timestamp"):
            return r.get("timestamp")
    return None


def _value_for_model(
    rows_by_source: Mapping[str, Mapping[str, Any]],
    model_source: str,
    value_field: Optional[str],
) -> Optional[float]:
    if not value_field:
        return None
    row = rows_by_source.get(model_source)
    if not row:
        return None
    v = row.get(value_field)
    return float(v) if isinstance(v, (int, float)) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--defaults", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="outputs/out.csv")
    args = ap.parse_args()

    cfg, engine, decision = build_from_config(args.defaults, args.config)

    # strict + simple: every model must declare its source (config enforces, but keep local mapping handy)
    model_sources: Dict[str, str] = {}
    model_value_fields: Dict[str, Optional[str]] = {}
    for model_name, mcfg in (cfg.get("models") or {}).items():
        src = mcfg.get("source")
        if not src:
            raise ValueError(f"Model '{model_name}' missing required key: source")
        model_sources[model_name] = str(src)
        model_value_fields[model_name] = _model_value_field(mcfg.get("features") or [])

    sources = _load_sources(cfg)
    iters = {name: _csv_events(sc) for name, sc in sources.items()}
    mode = (cfg.get("data", {}).get("timebase", {}).get("mode") or "union").lower()
    stream = _timebase_intersection(iters) if mode == "intersection" else _timebase_union(iters)

    plot_cfg = cfg.get("plot") or {}
    plot = None
    gt_by_source = _load_gt_by_source(sources, bool(plot_cfg.get("show_ground_truth")))
    if bool(plot_cfg.get("enable")):
        gt_set = _load_gt_set(sources, bool(plot_cfg.get("show_ground_truth")))
        plot = LivePlot(
            window=int(plot_cfg.get("window", 1000)),
            refresh_every=1,  # you want every step visible
            gt_timestamps=gt_set,
            likelihood_threshold=getattr(decision, "threshold", None),
        )
    step_pause = float(plot_cfg.get("step_pause_s", 0.0) or 0.0)

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    with open(outp, "w", newline="") as g:
        w = csv.DictWriter(
            g,
            fieldnames=["t", "timestamp", "model", "raw", "likelihood", "system_score", "alert"],
        )
        w.writeheader()

        def on_update(t, rows_by_source, model_outputs, result):
            # ----- Plot (generic) -----
            if plot is not None:
                # Build a plot payload that is model-aware (multi-signal), while keeping backward-compat 'value'.
                values_by_model: Dict[str, Optional[float]] = {}
                for model_name in model_outputs.keys():
                    src = model_sources.get(model_name)
                    vf = model_value_fields.get(model_name)
                    if not src:
                        values_by_model[model_name] = None
                        continue
                    values_by_model[model_name] = _value_for_model(rows_by_source, src, vf)

                # Ground truth per model (derived from the model's source labels)
                gt_by_model: Optional[Dict[str, set]] = None
                if gt_by_source:
                    gt_by_model = {}
                    for model_name, src in model_sources.items():
                        sgt = gt_by_source.get(src)
                        if sgt:
                            gt_by_model[model_name] = sgt

                plot_row: Dict[str, Any] = {
                    "timestamp": _ts_any(rows_by_source),
                    # keep old LivePlot working for now:
                    "value": next((v for v in values_by_model.values() if v is not None), None),
                    # new (used by next live_plot diff):
                    "values_by_model": values_by_model,
                    "gt_by_model": gt_by_model,
                }
                plot.update(t, plot_row, model_outputs, result)
                if step_pause > 0:
                    time.sleep(step_pause)

            sys_score = result.get("system_score") if isinstance(result, dict) else None
            alert = result.get("alert") if isinstance(result, dict) else None
            # write one line per model output
            ts_any = _ts_any(rows_by_source)
            for model_name, out in model_outputs.items():
                w.writerow(
                    {
                        "t": t,
                        "timestamp": ts_any,
                        "model": model_name,
                        "raw": out.get("raw"),
                        "likelihood": out.get("likelihood"),
                        "system_score": sys_score,
                        "alert": alert,
                    }
                )

        for _ in run(stream, engine, decision, on_update=on_update):
            pass

    print(f"[run] wrote -> {outp}")


if __name__ == "__main__":
    main()
