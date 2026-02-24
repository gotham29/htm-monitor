# src/htm_monitor/utils/config.py

import yaml
from pathlib import Path
from typing import Any, Dict, List, Set

from htm_monitor.htm_src.feature import Feature
from htm_monitor.htm_src.htm_model import HTMmodel
from htm_monitor.orchestration.engine import Engine
from htm_monitor.orchestration.decision import Decision


def _load_yaml_mapping(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise ValueError(f"YAML config does not exist (or is not a file): {path}")

    raw = yaml.safe_load(p.read_text())
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"YAML config must be a mapping at top-level: {path}")

    # YAML foot-gun: reject non-string keys early (keeps downstream logic sane).
    bad_keys = [k for k in raw.keys() if not isinstance(k, str)]
    if bad_keys:
        raise ValueError(f"YAML config must have string keys at top-level; got: {bad_keys} in {path}")

    return raw


def load_yaml(path: str) -> dict:
    return _load_yaml_mapping(path)


def deep_merge(a: Any, b: Any) -> Any:
    """
    Deep merge mappings: b overrides a.
    - dict + dict => recursive merge
    - otherwise => b
    """
    if isinstance(a, dict) and isinstance(b, dict):
        out: Dict[str, Any] = dict(a)
        for k, v in b.items():
            out[k] = deep_merge(out.get(k), v) if k in out else v
        return out
    return b


def _require_mapping(cfg: Dict[str, Any], key: str) -> Dict[str, Any]:
    v = cfg.get(key)
    if not isinstance(v, dict):
        raise ValueError(f"Config.{key} must be a mapping")
    return v


def _optional_mapping(cfg: Dict[str, Any], key: str) -> Dict[str, Any]:
    v = cfg.get(key)
    if v is None:
        return {}
    if not isinstance(v, dict):
        raise ValueError(f"Config.{key} must be a mapping if provided")
    return v


def _assert_only_known_top_keys(cfg: Dict[str, Any], *, allow: Set[str]) -> None:
    extra = set(cfg.keys()) - set(allow)
    if extra:
        raise ValueError(f"Config contains unknown top-level key(s): {sorted(extra)}")


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


def merge_dicts(a: dict, b: dict) -> dict:
    """
    Back-compat: older demo tools import merge_dicts().
    We now do a deep merge so nested config blocks (e.g., htm_params.tm) merge correctly.
    """
    if not isinstance(a, dict) or not isinstance(b, dict):
        raise ValueError("merge_dicts expects dict + dict")
    return deep_merge(a, b)


def build_from_config(defaults_path: str, user_path: str):
    defaults = load_yaml(defaults_path)
    user = load_yaml(user_path)

    # Tighten: if someone passes a "defaults" that isn’t actually defaults, it’s a silent mess.
    # We allow user to be partial, but defaults should carry the base structure.
    cfg = deep_merge(defaults, user)

    # Optional strictness: reject unknown top-level keys (helps catch typos early).
    _assert_only_known_top_keys(
        cfg,
        allow={
            "htm_params",
            "features",
            "models",
            "data",
            "decision",
            "plot",
        },
    )

    # ---- Basic required sections (fail fast with clear errors) ----
    required = ["htm_params", "features", "models", "data", "decision"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Config missing required top-level key(s): {missing}")

    if not isinstance(cfg["features"], dict) or not cfg["features"]:
        raise ValueError("Config key 'features' must be a non-empty mapping")
    if not isinstance(cfg["models"], dict) or not cfg["models"]:
        raise ValueError("Config key 'models' must be a non-empty mapping")
    if not isinstance(cfg["decision"], dict):
        raise ValueError("Config key 'decision' must be a mapping")
    if not isinstance(cfg["htm_params"], dict):
        raise ValueError("Config key 'htm_params' must be a mapping")
    if not isinstance(cfg.get("data"), dict):
        raise ValueError("Config key 'data' must be a mapping")

    # ---- Build Features ----
    fcfg = _require_mapping(cfg, "features")
    if "timestamp" not in fcfg:
        raise ValueError("Config.features must include a 'timestamp' feature")

    features: Dict[str, Feature] = {}
    for name, params in fcfg.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Feature names must be non-empty strings")
        if not isinstance(params, dict):
            raise ValueError(f"Feature '{name}' config must be a mapping")
        features[name] = Feature(name, params)

    # ---- Build Models ----
    mcfg_all = _require_mapping(cfg, "models")
    models: Dict[str, HTMmodel] = {}
    model_sources: Dict[str, List[str]] = {}

    htm_params = _require_mapping(cfg, "htm_params")

    for model_name, model_cfg in mcfg_all.items():
        if not isinstance(model_cfg, dict):
            raise ValueError(f"Model '{model_name}' config must be a mapping")
        srcs = _model_sources(model_name, model_cfg)

        feats = model_cfg.get("features")
        if not isinstance(feats, list) or not feats:
            raise ValueError(f"Model '{model_name}' key 'features' must be a non-empty list[str]")

        model_features: Dict[str, Feature] = {}
        for fname in feats:
            if not isinstance(fname, str) or not fname:
                raise ValueError(f"Model '{model_name}' features entries must be non-empty strings")
            if fname not in features:
                raise ValueError(f"Model '{model_name}' references unknown feature '{fname}'")
            model_features[fname] = features[fname]

        models[model_name] = HTMmodel(
            features=model_features,
            models_params=htm_params,
            return_pred_count=False,
            name=model_name,
        )
        model_sources[model_name] = srcs

    data = _optional_mapping(cfg, "data")
    timebase = _optional_mapping(data, "timebase")
    on_missing = (timebase.get("on_missing") or "skip")
    on_missing = str(on_missing).lower()
    engine = Engine(models, model_sources, on_missing=on_missing)

    dcfg = _require_mapping(cfg, "decision")

    # Optional windowed decision config:
    # decision:
    #   method: kofn_window
    #   window:
    #     size: 12
    #     per_model_hits: 2
    wraw = dcfg.get("window") or {}
    if not isinstance(wraw, dict):
        raise ValueError("decision.window must be a mapping if provided")
    wcfg: Dict[str, Any] = dict(wraw)
    window_size = int(wcfg.get("size", 1) or 1)
    per_model_hits = int(wcfg.get("per_model_hits", 1) or 1)

    decision = Decision(
        threshold=dcfg.get("threshold"),
        method=dcfg.get("method", "max"),
        k=dcfg.get("k"),
        window_size=window_size,
        per_model_hits=per_model_hits,
        score_key=str(dcfg.get("score_key", "likelihood")),
    )

    return cfg, engine, decision, model_sources
