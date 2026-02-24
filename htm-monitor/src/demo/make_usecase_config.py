# src/demo/make_usecase_config.py

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Mapping

import pandas as pd
import yaml


@dataclass(frozen=True)
class SourceSpec:
    """
    One CSV source with 1+ selected numeric feature columns.

    Example (NAB):
      name="occupancy_t4013"
      path="/.../occupancy_t4013.csv"
      timestamp_col="timestamp"
      timestamp_format="%Y-%m-%d %H:%M:%S"
      fields={"occupancy_t4013": "value"}
    """

    name: str
    path: str
    timestamp_col: str
    timestamp_format: str
    # canonical_feature_name -> csv_column_name
    fields: Dict[str, str]

    def __post_init__(self) -> None:
        # Fail fast: prevent configs that "serialize fine" but explode later.
        if not isinstance(self.name, str) or not self.name.strip():
            raise TypeError("SourceSpec.name must be a non-empty string")
        if not isinstance(self.path, str) or not self.path.strip():
            raise TypeError("SourceSpec.path must be a non-empty string")
        p = Path(self.path)
        if not p.exists() or not p.is_file():
            raise ValueError(f"CSV does not exist (or is not a file): {self.path}")

        if not isinstance(self.timestamp_col, str) or not self.timestamp_col.strip():
            raise TypeError("SourceSpec.timestamp_col must be a non-empty string")
        if not isinstance(self.timestamp_format, str) or not self.timestamp_format.strip():
            raise TypeError("SourceSpec.timestamp_format must be a non-empty string")

        if not isinstance(self.fields, dict) or len(self.fields) == 0:
            raise TypeError("SourceSpec.fields must be a non-empty dict {feature_name: csv_column}")

        # Validate fields content: no blanks, all strings.
        seen_feat: set[str] = set()
        seen_cols: set[str] = set()
        for feat, col in self.fields.items():
            if not isinstance(feat, str) or not feat.strip():
                raise TypeError("SourceSpec.fields keys (feature names) must be non-empty strings")
            if not isinstance(col, str) or not col.strip():
                raise TypeError("SourceSpec.fields values (csv columns) must be non-empty strings")
            if feat in seen_feat:
                raise ValueError(f"Duplicate feature name within one source: '{feat}'")
            # Quiet foot-gun: mapping two features to the same column is almost always accidental.
            if col in seen_cols:
                raise ValueError(
                    f"Duplicate csv column '{col}' within SourceSpec.fields for source '{self.name}'. "
                    "If you really intend this, duplicate the column upstream with a new name."
                )
            seen_feat.add(feat)
            seen_cols.add(col)


def _write_text_atomic(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _require_safe_usecase_name(usecase: str) -> str:
    """
    Prevent path traversal / accidental nested paths when we do configs/<usecase>.yaml.
    Keep it simple: forbid separators and '.' segments.
    """
    if not isinstance(usecase, str) or not usecase.strip():
        raise ValueError("Use-case name is required")
    u = usecase.strip()
    if "/" in u or "\\" in u:
        raise ValueError("Use-case name must not contain path separators ('/' or '\\').")
    if u in (".", "..") or ".." in u:
        raise ValueError("Use-case name must not contain '..' path segments.")
    return u


def sources_to_dicts(sources: Sequence[SourceSpec]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for s in sources:
        out.append(
            {
                "name": s.name,
                "path": s.path,
                "timestamp_col": s.timestamp_col,
                "timestamp_format": s.timestamp_format,
                "fields": dict(s.fields),
            }
        )
    return out


def sources_from_dicts(raw: Sequence[Mapping[str, Any]]) -> List[SourceSpec]:
    out: List[SourceSpec] = []
    for i, d in enumerate(raw):
        if not isinstance(d, Mapping):
            raise TypeError(f"sources[{i}] must be a mapping")
        for k in ("name", "path", "timestamp_col", "timestamp_format", "fields"):
            if k not in d:
                raise ValueError(f"sources[{i}] missing required key '{k}'")
        fields = d.get("fields")
        if not isinstance(fields, Mapping) or len(fields) == 0:
            raise ValueError(f"sources[{i}].fields must be a non-empty mapping")
        # YAML foot-gun: non-string keys can appear; reject early.
        bad_field_keys = [k for k in fields.keys() if not isinstance(k, str)]
        if bad_field_keys:
            raise ValueError(f"sources[{i}].fields must have string keys; got: {bad_field_keys}")
        out.append(
            SourceSpec(
                name=str(d["name"]).strip(),
                path=str(d["path"]).strip(),
                timestamp_col=str(d["timestamp_col"]).strip(),
                timestamp_format=str(d["timestamp_format"]).strip(),
                fields=dict(fields),
            )
        )
    return out


def _read_series(csv_path: str, col: str) -> pd.Series:
    df = pd.read_csv(csv_path, usecols=[col], low_memory=False)
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(s) == 0:
        raise ValueError(f"No numeric values found in column '{col}' of {csv_path}")
    return s


def _read_header(csv_path: str) -> List[str]:
    # cheapest robust header read
    head = pd.read_csv(csv_path, nrows=1, low_memory=False)
    return list(head.columns)


def _require_positive_int(name: str, v: Any) -> int:
    try:
        iv = int(v)
    except Exception:
        raise ValueError(f"{name} must be an int")
    if iv <= 0:
        raise ValueError(f"{name} must be > 0")
    return iv


def _require_float_in_open01(name: str, v: Any) -> float:
    try:
        fv = float(v)
    except Exception:
        raise ValueError(f"{name} must be a float")
    if not (0.0 < fv < 1.0):
        raise ValueError(f"{name} must satisfy 0 < {name} < 1")
    return fv


def calibrate_min_max(
    s: pd.Series,
    low_q: float = 0.01,
    high_q: float = 0.99,
    margin: float = 0.03,
) -> Tuple[float, float]:
    if not (0.0 < low_q < high_q < 1.0):
        raise ValueError("low_q/high_q must satisfy 0 < low_q < high_q < 1")
    if margin < 0.0:
        raise ValueError("margin must be >= 0")

    lo = float(s.quantile(low_q))
    hi = float(s.quantile(high_q))

    # Expand by a margin of the quantile span (not absolute units).
    span = hi - lo
    if span <= 0:
        # Degenerate: constant or near-constant series.
        # Keep a tiny span so RDSE has a non-zero range.
        eps = max(abs(lo) * 1e-6, 1e-6)
        return lo - eps, hi + eps

    pad = margin * span
    return lo - pad, hi + pad


def build_usecase_config(
    usecase: str,
    sources: Sequence[SourceSpec],
    *,
    # RDSE defaults (Phase 1)
    rdse_size: int = 2048,
    rdse_active_bits: int = 40,
    rdse_num_buckets: int = 130,
    seed_base: int = 42,
    # calibration defaults
    low_q: float = 0.01,
    high_q: float = 0.99,
    margin: float = 0.03,
    # model layout
    #   separate: one model per feature
    #   single: one model containing all features
    #   chunk: fixed-size chunks of features per model
    model_layout: str = "separate",
    chunk_size: int = 2,
    # monitoring defaults
    timebase_mode: str = "union",
    on_missing: str = "hold_last",
    # decision defaults
    decision_score_key: str = "anomaly_probability",
    decision_threshold: float = 0.997,
    decision_method: str = "kofn_window",
    decision_k: int = 2,
    decision_window_size: int = 24,
    decision_per_model_hits: int = 2,
    # plot defaults
    plot_enable: bool = True,
    plot_window: int = 1000,
    plot_show_ground_truth: bool = True,
    plot_step_pause_s: float = 0.01,
) -> Dict[str, Any]:
    if not usecase or not isinstance(usecase, str):
        raise ValueError("usecase must be a non-empty string")
    if not sources:
        raise ValueError("At least one source is required")
    if model_layout not in ("separate", "single", "chunk"):
        raise ValueError("model_layout must be one of: separate, single, chunk")
    if model_layout == "chunk" and int(chunk_size) < 1:
        raise ValueError("chunk_size must be >= 1")

    # Guardrails: stop silent nonsense early.
    rdse_size = _require_positive_int("rdse_size", rdse_size)
    rdse_active_bits = _require_positive_int("rdse_active_bits", rdse_active_bits)
    rdse_num_buckets = _require_positive_int("rdse_num_buckets", rdse_num_buckets)
    if rdse_active_bits > rdse_size:
        raise ValueError("rdse_active_bits must be <= rdse_size")

    low_q = _require_float_in_open01("low_q", low_q)
    high_q = _require_float_in_open01("high_q", high_q)
    if not (low_q < high_q):
        raise ValueError("low_q/high_q must satisfy low_q < high_q")
    if float(margin) < 0.0:
        raise ValueError("margin must be >= 0")

    decision_k = _require_positive_int("decision_k", decision_k)
    decision_window_size = _require_positive_int("decision_window_size", decision_window_size)
    decision_per_model_hits = _require_positive_int("decision_per_model_hits", decision_per_model_hits)
    decision_threshold = float(decision_threshold)
    if not (0.0 <= decision_threshold <= 1.0):
        raise ValueError("decision_threshold must be in [0, 1]")

    # ---- Features ----
    features: Dict[str, Any] = {
        "timestamp": {
            "type": "datetime",
            # NAB convention; user can edit after generation if needed
            "format": "%Y-%m-%d %H:%M:%S",
            # DateEncoder: must have at least one > 0
            "timeOfDay": 21,
            "weekend": 0,
            "dayOfWeek": 0,
            "holiday": 0,
            "season": 0,
        }
    }

    # Each source defines 1+ features.
    all_feature_names: List[str] = []
    seed_i = 0
    for src in sources:
        # Validate CSV schema up-front so we don't emit broken configs.
        cols = _read_header(src.path)
        if src.timestamp_col not in cols:
            raise ValueError(
                f"timestamp_col '{src.timestamp_col}' not found in CSV header for source '{src.name}': {src.path}"
            )
        # Ensure all referenced columns exist (and give a sharp error).
        missing = [c for c in src.fields.values() if c not in cols]
        if missing:
            raise ValueError(
                f"Selected column(s) {missing} not found in CSV header for source '{src.name}': {src.path}"
            )
        for feat_name, col_name in src.fields.items():
            if feat_name in features:
                raise ValueError(f"Duplicate feature name '{feat_name}' across sources. Feature names must be unique.")
            s = _read_series(src.path, col_name)
            mn, mx = calibrate_min_max(s, low_q=low_q, high_q=high_q, margin=margin)

            features[feat_name] = {
                "type": "float",
                "size": int(rdse_size),
                "activeBits": int(rdse_active_bits),
                "numBuckets": int(rdse_num_buckets),
                "minVal": float(mn),
                "maxVal": float(mx),
                "seed": int(seed_base + seed_i),
            }
            all_feature_names.append(feat_name)
            seed_i += 1

    # ---- Models ----
    models: Dict[str, Any] = {}
    if model_layout == "separate":
        # one model per feature; source list = the source that defines it
        for src in sources:
            for feat_name in src.fields:
                model_name = f"{feat_name}_model"
                models[model_name] = {
                    "sources": [src.name],
                    "features": ["timestamp", feat_name],
                }
    elif model_layout == "single":
        # one model for everything; include all sources; include all features
        models[f"{usecase}_model"] = {
            "sources": [s.name for s in sources],
            "features": ["timestamp"] + list(all_feature_names),
        }
    elif model_layout == "chunk":
        # stable chunking order: all_feature_names in the order collected
        feats = list(all_feature_names)
        k = int(chunk_size)
        for i in range(0, len(feats), k):
            chunk = feats[i : i + k]
            model_name = f"{usecase}_chunk_{(i // k) + 1}_model"
            # include only sources that define at least one feature in this chunk
            chunk_sources: List[str] = []
            for src in sources:
                if any(f in src.fields for f in chunk):
                    chunk_sources.append(src.name)
            models[model_name] = {
                "sources": chunk_sources,
                "features": ["timestamp"] + chunk,
            }

    # ---- Data sources ----
    data_sources: List[Dict[str, Any]] = []
    for src in sources:
        data_sources.append(
            {
                "name": src.name,
                "kind": "csv",
                "path": src.path,
                "timestamp_col": src.timestamp_col,
                "timestamp_format": src.timestamp_format,
                "fields": dict(src.fields),
                # labels can be added later by hand or by a future “label import” step
            }
        )

    cfg: Dict[str, Any] = {
        "features": features,
        "models": models,
        "data": {
            "timebase": {
                "mode": str(timebase_mode),
                "on_missing": str(on_missing),
            },
            "sources": data_sources,
        },
        "decision": {
            "score_key": str(decision_score_key),
            "threshold": float(decision_threshold),
            "method": str(decision_method),
            "k": int(decision_k),
            "window": {
                "size": int(decision_window_size),
                "per_model_hits": int(decision_per_model_hits),
            },
        },
        "plot": {
            "enable": bool(plot_enable),
            "step_pause_s": float(plot_step_pause_s),
            "window": int(plot_window),
            "show_ground_truth": bool(plot_show_ground_truth),
        },
    }

    return cfg


def _prompt(msg: str, default: Optional[str] = None) -> str:
    if default is not None:
        raw = input(f"{msg} [{default}]: ").strip()
        return raw if raw else default
    return input(f"{msg}: ").strip()


def _prompt_int(msg: str, default: int) -> int:
    raw = _prompt(msg, str(default))
    try:
        v = int(raw)
    except Exception:
        raise ValueError(f"{msg} must be an int")
    return v


def _prompt_float(msg: str, default: float) -> float:
    raw = _prompt(msg, str(default))
    try:
        v = float(raw)
    except Exception:
        raise ValueError(f"{msg} must be a float")
    return v


def _prompt_choice(msg: str, choices: Sequence[str], default: str) -> str:
    raw = _prompt(msg, default)
    v = raw.strip().lower()
    if v not in [c.lower() for c in choices]:
        raise ValueError(f"{msg} must be one of: {', '.join(choices)}")
    return v


def _parse_csv_list(raw: str) -> List[str]:
    items = [x.strip() for x in (raw or "").split(",")]
    return [x for x in items if x]


def collect_sources_interactive() -> List[SourceSpec]:
    n = _prompt_int("How many CSV sources?", 2)
    out: List[SourceSpec] = []
    for i in range(n):
        print(f"\n--- Source {i+1}/{n} ---")
        path = _prompt("CSV path")
        p = Path(path)
        if not p.exists():
            raise ValueError(f"CSV does not exist: {path}")

        # Light header peek to help user pick names/columns
        head = pd.read_csv(path, nrows=1)
        cols = list(head.columns)
        print(f"Detected columns: {cols}")

        name = _prompt("Source name (identifier)", p.stem)
        ts_col = _prompt("Timestamp column", "timestamp")
        ts_fmt = _prompt("Timestamp format", "%Y-%m-%d %H:%M:%S")
        # Phase 2A: allow multiple numeric columns (comma-separated).
        # We keep it simple: user types "value" or "x1,x2,x3".
        raw_cols = _prompt("Value column(s) (comma-separated)", "value")
        sel_cols = _parse_csv_list(raw_cols)
        if not sel_cols:
            raise ValueError("You must provide at least one value column")
        for c in sel_cols:
            if c not in cols:
                raise ValueError(f"Column '{c}' not found in {path}")

        fields: Dict[str, str] = {}
        for c in sel_cols:
            # Default feature name: <stem> if single col, else <stem>_<col>
            default_feat = p.stem if len(sel_cols) == 1 else f"{p.stem}_{c}"
            feat = _prompt(f"Canonical feature name for column '{c}'", default_feat)
            if not feat:
                raise ValueError("Feature name cannot be empty")
            if feat in fields:
                raise ValueError(f"Duplicate feature name '{feat}' within the same source")
            fields[feat] = c

        if not fields:
            raise ValueError("You must select at least one numeric column")

        out.append(
            SourceSpec(
                name=name,
                path=str(p),
                timestamp_col=ts_col,
                timestamp_format=ts_fmt,
                fields=fields,
            )
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive wizard to generate configs/<usecase>.yaml from CSV data.")
    ap.add_argument("--out-dir", default="configs", help="Where to write <usecase>.yaml")
    ap.add_argument("--usecase", default=None, help="If set, skip prompt and use this use-case name")
    ap.add_argument("--low", type=float, default=0.01)
    ap.add_argument("--high", type=float, default=0.99)
    ap.add_argument("--margin", type=float, default=0.03)
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--activeBits", type=int, default=40)
    ap.add_argument("--numBuckets", type=int, default=130)
    ap.add_argument("--seedBase", type=int, default=42)
    args = ap.parse_args()

    raw_usecase = args.usecase if args.usecase is not None else _prompt("Use-case name (output will be configs/<usecase>.yaml)")
    usecase = _require_safe_usecase_name(raw_usecase)

    sources = collect_sources_interactive()

    print("\n--- Model layout ---")
    layout = _prompt_choice("Model layout (separate|single|chunk)", ["separate", "single", "chunk"], "separate")
    chunk_size = 2
    if layout == "chunk":
        chunk_size = _prompt_int("Chunk size (features per model)", 2)

    print("\n--- Decision config ---")
    decision_method = _prompt("decision.method", "kofn_window")
    decision_k = _prompt_int("decision.k", 2)
    decision_window_size = _prompt_int("decision.window.size", 24)
    decision_per_model_hits = _prompt_int("decision.window.per_model_hits", 2)
    decision_threshold = _prompt_float("decision.threshold", 0.997)
    decision_score_key = _prompt("decision.score_key", "anomaly_probability")

    cfg = build_usecase_config(
        usecase,
        sources,
        rdse_size=args.size,
        rdse_active_bits=args.activeBits,
        rdse_num_buckets=args.numBuckets,
        seed_base=args.seedBase,
        low_q=args.low,
        high_q=args.high,
        margin=args.margin,
        model_layout=layout,
        chunk_size=chunk_size,
        decision_method=decision_method,
        decision_k=decision_k,
        decision_window_size=decision_window_size,
        decision_per_model_hits=decision_per_model_hits,
        decision_threshold=decision_threshold,
        decision_score_key=decision_score_key,
    )

    out_dir = Path(args.out_dir)
    out_path = out_dir / f"{usecase}.yaml"
    _write_text_atomic(out_path, yaml.safe_dump(cfg, sort_keys=False))

    print(f"[make_usecase_config] wrote -> {out_path}")


if __name__ == "__main__":
    main()