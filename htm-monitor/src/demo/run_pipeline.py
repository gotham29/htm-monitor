# demo/run_pipeline.py

import argparse
import csv
import time
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, Optional, Tuple, Set, List

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--defaults", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="outputs/out.csv")
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--log-file", default=None)
    args = ap.parse_args()

    _configure_logging(args.log_level, args.log_file)

    cfg, engine, decision = build_from_config(args.defaults, args.config)

    # model_sources normalized by build_from_config: model -> list[sources]
    model_sources: Dict[str, List[str]] = {}
    model_value_fields: Dict[str, List[str]] = {}
    for model_name, mcfg in (cfg.get("models") or {}).items():
        if "source" in mcfg:
            model_sources[model_name] = [str(mcfg["source"])]
        elif "sources" in mcfg:
            srcs = mcfg.get("sources")
            if not isinstance(srcs, list) or not srcs:
                raise ValueError(f"Model '{model_name}' key 'sources' must be a non-empty list[str]")
            model_sources[model_name] = [str(s) for s in srcs]
        else:
            raise ValueError(f"Model '{model_name}' must specify 'source' or 'sources'")

        # All non-timestamp features for this model (used only for plotting)
        feats = list(mcfg.get("features") or [])
        model_value_fields[model_name] = [f for f in feats if f != "timestamp"]

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
            fieldnames=[
                "t", "timestamp", "model",
                "raw", "p", "likelihood",
                "system_score", "alert",
                "hot_by_model",
            ],
        )
        w.writeheader()

        def on_update(t, rows_by_source, model_outputs, result):
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
            # write one line per model output
            ts_any = _ts_any(rows_by_source)
            for model_name, out in model_outputs.items():
                w.writerow(
                    {
                        "t": t,
                        "timestamp": ts_any,
                        "model": model_name,
                        "raw": out.get("raw"),
                        "p": out.get("p"),
                        "likelihood": out.get("likelihood"),
                        "system_score": sys_score,
                        "alert": alert,
                        "hot_by_model": hot_by_model,
                    }
                )

        for _ in run(stream, engine, decision, on_update=on_update):
            pass

    print(f"[run] wrote -> {outp}")


if __name__ == "__main__":
    main()
