# src/htm_monitor/utils/config.py

import yaml
from pathlib import Path

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


def build_from_config(defaults_path: str, user_path: str):
    defaults = load_yaml(defaults_path)
    user = load_yaml(user_path)

    cfg = merge_dicts(defaults, user)

    # strict + simple: every model must declare its source
    for model_name, model_cfg in cfg["models"].items():
        if "source" not in model_cfg:
            raise ValueError(f"Model '{model_name}' missing required key: source")

    # ---- Build Features ----
    features = {
        name: Feature(name, params)
        for name, params in cfg["features"].items()
    }

    # ---- Build Models ----
    models = {}
    model_sources = {}
    for model_name, model_cfg in cfg["models"].items():
        model_features = {
            fname: features[fname]
            for fname in model_cfg["features"]
        }

        models[model_name] = HTMmodel(
            features=model_features,
            models_params=cfg["htm_params"],
            return_pred_count=False,
        )
        model_sources[model_name] = model_cfg["source"]

    engine = Engine(models, model_sources)

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
