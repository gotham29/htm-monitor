# src/htm_monitor/utils/config.py

import yaml
from pathlib import Path
from typing import Any, Dict, List

from htm_monitor.htm_src.feature import Feature
from htm_monitor.htm_src.htm_model import HTMmodel
from htm_monitor.orchestration.engine import Engine
from htm_monitor.orchestration.decision import Decision


def load_yaml(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def merge_dicts(a: dict, b: dict) -> dict:
    out = dict(a)
    out.update(b)
    return out


def _model_sources(model_name: str, model_cfg: dict) -> List[str]:
    """
    Normalize per-model source config.
    Exactly one of:
      - source: str
      - sources: list[str]
    """
    has_source = "source" in model_cfg
    has_sources = "sources" in model_cfg
    if has_source == has_sources:
        raise ValueError(
            f"Model '{model_name}' must specify exactly one of: 'source' or 'sources'"
        )
    if has_source:
        src = model_cfg.get("source")
        if not isinstance(src, str) or not src:
            raise ValueError(f"Model '{model_name}' key 'source' must be a non-empty string")
        return [src]

    srcs = model_cfg.get("sources")
    if not isinstance(srcs, list) or not srcs:
        raise ValueError(f"Model '{model_name}' key 'sources' must be a non-empty list[str]")
    out: List[str] = []
    for s in srcs:
        if not isinstance(s, str) or not s:
            raise ValueError(f"Model '{model_name}' key 'sources' entries must be non-empty strings")
        out.append(s)
    return out


def build_from_config(defaults_path: str, user_path: str):
    defaults = load_yaml(defaults_path)
    user = load_yaml(user_path)

    cfg = merge_dicts(defaults, user)

    # ---- Build Features ----
    features = {
        name: Feature(name, params)
        for name, params in cfg["features"].items()
    }

    # ---- Build Models ----
    models = {}
    model_sources = {}
    for model_name, model_cfg in cfg["models"].items():
        srcs = _model_sources(model_name, model_cfg)

        model_features = {
            fname: features[fname]
            for fname in model_cfg["features"]
        }

        models[model_name] = HTMmodel(
            features=model_features,
            models_params=cfg["htm_params"],
            return_pred_count=False,
        )
        model_sources[model_name] = srcs

    on_missing = (cfg.get("data", {}).get("timebase", {}).get("on_missing") or "skip").lower()
    engine = Engine(models, model_sources, on_missing=on_missing)

    dcfg = cfg["decision"]

    # Optional windowed decision config:
    # decision:
    #   method: kofn_window
    #   window:
    #     size: 12
    #     per_model_hits: 2
    wcfg: Dict[str, Any] = dict(dcfg.get("window") or {})
    window_size = int(wcfg.get("size", 1) or 1)
    per_model_hits = int(wcfg.get("per_model_hits", 1) or 1)

    decision = Decision(
        threshold=dcfg["threshold"],
        method=dcfg.get("method", "max"),
        k=dcfg.get("k"),
        window_size=window_size,
        per_model_hits=per_model_hits,
    )

    return cfg, engine, decision
