# src/htm-monitor/cli/make_usecase_config.py

from __future__ import annotations

import argparse
import json
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Mapping, Set

import pandas as pd
import yaml


@dataclass(frozen=True)
class SourceSpec:
    """
    Canonical GT contract:
      event_windows = [
        {"name": "...", "kind": "primary_gt|explanatory", "start": "...", "end": "..."},
        ...
      ]

    Legacy gt_timestamps are accepted on input and normalized upstream.
    """
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
    # Optional display / domain metadata
    unit: Optional[str] = None
    # Optional canonical ground-truth windows.
    event_windows: Optional[List[Dict[str, str]]] = None

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
        if self.unit is not None:
            if not isinstance(self.unit, str) or not self.unit.strip():
                raise TypeError("SourceSpec.unit must be a non-empty string or None")
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

        if self.event_windows is not None:
            if not isinstance(self.event_windows, list):
                raise TypeError("SourceSpec.event_windows must be a list[dict] or None")
            for i, ev in enumerate(self.event_windows):
                if not isinstance(ev, Mapping):
                    raise TypeError(f"SourceSpec.event_windows[{i}] must be a mapping")
                kind = ev.get("kind", "primary_gt")
                start = ev.get("start")
                end = ev.get("end")
                if not isinstance(kind, str) or not kind.strip():
                    raise TypeError(f"SourceSpec.event_windows[{i}].kind must be a non-empty string")
                if kind.strip() not in {"primary_gt", "explanatory"}:
                    raise ValueError(
                        f"SourceSpec.event_windows[{i}].kind must be one of: primary_gt, explanatory"
                    )
                if not isinstance(start, str) or not start.strip():
                    raise TypeError(f"SourceSpec.event_windows[{i}].start must be a non-empty string")
                if not isinstance(end, str) or not end.strip():
                    raise TypeError(f"SourceSpec.event_windows[{i}].end must be a non-empty string")
                datetime.strptime(start, self.timestamp_format)
                datetime.strptime(end, self.timestamp_format)
                if datetime.strptime(end, self.timestamp_format) < datetime.strptime(start, self.timestamp_format):
                    raise ValueError(
                        f"SourceSpec.event_windows[{i}] end must be >= start for source '{self.name}'"
                    )


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
                **({"unit": s.unit} if s.unit else {}),
                "labels": (
                    {
                        "event_windows": [
                            {
                                "name": ev.get("name", ""),
                                "kind": ev.get("kind", "primary_gt"),
                                "start": ev["start"],
                                "end": ev["end"],
                            }
                            for ev in s.event_windows
                        ]
                    }
                    if s.event_windows
                    else None
                ),
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

        # Ground-truth timestamps can arrive in either format:
        #   legacy:   gt_timestamps: ["...", ...]
        #   canonical: labels: { timestamps: ["...", ...] }
        # We normalize both into SourceSpec.gt_timestamps.
        has_legacy_gt = "gt_timestamps" in d and d.get("gt_timestamps") is not None
        has_labels = "labels" in d and d.get("labels") is not None
        if has_legacy_gt and has_labels:
            raise ValueError(f"sources[{i}] cannot contain both 'gt_timestamps' and 'labels' (ambiguous GT format)")

        gt_list: Optional[List[str]] = None
        if has_legacy_gt:
            gt_raw = d.get("gt_timestamps")
            if not isinstance(gt_raw, list) or not all(isinstance(x, str) and x.strip() for x in gt_raw):
                raise ValueError(f"sources[{i}].gt_timestamps must be a list of non-empty timestamp strings")
            gt_list = list(gt_raw) or None
        elif has_labels:
            labels = d.get("labels")
            if not isinstance(labels, Mapping):
                raise ValueError(f"sources[{i}].labels must be a mapping if provided")
            ts = labels.get("timestamps")
            if ts is not None:
                if not isinstance(ts, list) or not all(isinstance(x, str) and x.strip() for x in ts):
                    raise ValueError(f"sources[{i}].labels.timestamps must be a list of non-empty timestamp strings")
                gt_list = list(ts) or None
        # YAML foot-gun: non-string keys can appear; reject early.
        bad_field_keys = [k for k in fields.keys() if not isinstance(k, str)]
        if bad_field_keys:
            raise ValueError(f"sources[{i}].fields must have string keys; got: {bad_field_keys}")
        event_windows: Optional[List[Dict[str, str]]] = None

        # canonical path
        labels = d.get("labels")
        if labels is not None:
            if not isinstance(labels, Mapping):
                raise ValueError(f"sources[{i}].labels must be a mapping if provided")
            evs = labels.get("event_windows")
            if evs is not None:
                if not isinstance(evs, list):
                    raise ValueError(f"sources[{i}].labels.event_windows must be a list")
                event_windows = []
                for j, ev in enumerate(evs):
                    if not isinstance(ev, Mapping):
                        raise ValueError(f"sources[{i}].labels.event_windows[{j}] must be a mapping")
                    kind = ev.get("kind", "primary_gt")
                    start = ev.get("start")
                    end = ev.get("end")
                    name = ev.get("name")
                    if not isinstance(kind, str) or not kind.strip():
                        raise ValueError(
                            f"sources[{i}].labels.event_windows[{j}].kind must be a non-empty string"
                        )
                    if kind.strip() not in {"primary_gt", "explanatory"}:
                        raise ValueError(
                            f"sources[{i}].labels.event_windows[{j}].kind must be one of: primary_gt, explanatory"
                        )
                    if not isinstance(start, str) or not start.strip():
                        raise ValueError(f"sources[{i}].labels.event_windows[{j}].start must be a non-empty string")
                    if not isinstance(end, str) or not end.strip():
                        raise ValueError(f"sources[{i}].labels.event_windows[{j}].end must be a non-empty string")
                    event_windows.append(
                        {
                            "name": str(name).strip() if isinstance(name, str) and name.strip() else "",
                            "kind": kind.strip(),
                            "start": start.strip(),
                            "end": end.strip(),
                        }
                    )
            elif gt_list:
                # legacy labels.timestamps -> singleton windows
                event_windows = [{"name": "", "kind": "primary_gt", "start": ts, "end": ts} for ts in gt_list]
        elif gt_list:
            # legacy top-level gt_timestamps
            event_windows = [{"name": "", "kind": "primary_gt", "start": ts, "end": ts} for ts in gt_list]

        out.append(SourceSpec(
            name=str(d["name"]).strip(),
            path=str(d["path"]).strip(),
            timestamp_col=str(d["timestamp_col"]).strip(),
            timestamp_format=str(d["timestamp_format"]).strip(),
            fields=dict(fields),
            unit=(
                str(d.get("unit")).strip()
                if d.get("unit") is not None and str(d.get("unit")).strip()
                else None
            ),
            event_windows=event_windows,
        ))
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


def _require_float_in_closed01(name: str, v: Any) -> float:
    try:
        fv = float(v)
    except Exception:
        raise ValueError(f"{name} must be a float")
    if not (0.0 <= fv <= 1.0):
        raise ValueError(f"{name} must satisfy 0 <= {name} <= 1")
    return fv


def calibrate_min_max(
    s: pd.Series,
    low_q: float = 0.01,
    high_q: float = 0.99,
    margin: float = 0.03,
    *,
    floor: Optional[float] = None,
    ceil: Optional[float] = None,
) -> Tuple[float, float]:
    if not (0.0 <= low_q < high_q <= 1.0):
        raise ValueError("low_q/high_q must satisfy 0 <= low_q < high_q <= 1")
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
        mn = lo - eps
        mx = hi + eps
        if floor is not None:
            mn = max(mn, float(floor))
        if ceil is not None:
            mx = min(mx, float(ceil))
        if mx <= mn:
            eps2 = max(abs(mn) * 1e-6, 1e-6)
            mn, mx = mn - eps2, mx + eps2
        return mn, mx

    pad = margin * span
    mn = lo - pad
    mx = hi + pad
    if floor is not None:
        mn = max(mn, float(floor))
    if ceil is not None:
        mx = min(mx, float(ceil))
    if mx <= mn:
        eps2 = max(abs(mn) * 1e-6, 1e-6)
        mn, mx = mn - eps2, mx + eps2
    return mn, mx


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
    # optional calibration clamps (e.g., counts -> floor=0)
    calibration_min_floor: Optional[float] = None,
    calibration_max_ceil: Optional[float] = None,
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
    decision_score_key: str = "p",
    decision_threshold: float = 0.99,
    decision_method: str = "grouped_consensus",
    decision_model_warmth_window_size: int = 8,
    decision_system_group_k: int = 2,
    decision_system_min_system_len: int = 3,
    decision_groups: Optional[Mapping[str, Mapping[str, Any]]] = None,
    # run lifecycle defaults
    run_warmup_steps: int = 0,
    run_learn_after_warmup: bool = True,
    # system-GT derivation defaults (separate from decision semantics)
    gt_system_window_size: Optional[int] = None,
    gt_system_per_source_hits: int = 1,
    # plot defaults
    plot_enable: bool = True,
    plot_window: int = 1000,
    plot_show_ground_truth: bool = True,
    plot_step_pause_s: float = 0.0001,
    plot_show_warmup_span: bool = True,
    plot_max_label_len: int = 16,
    plot_label_color_mode: str = "by_group",
    plot_model_label_colors: Optional[Mapping[str, str]] = None,
    # optional frozen overrides (picked once via wizard)
    feature_overrides: Optional[Mapping[str, Mapping[str, float]]] = None,
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

    low_q = _require_float_in_closed01("low_q", low_q)
    high_q = _require_float_in_closed01("high_q", high_q)
    if not (low_q < high_q):
        raise ValueError("low_q/high_q must satisfy low_q < high_q")
    if float(margin) < 0.0:
        raise ValueError("margin must be >= 0")

    decision_threshold = float(decision_threshold)
    if not (0.0 <= decision_threshold <= 1.0):
        raise ValueError("decision_threshold must be in [0, 1]")
    decision_model_warmth_window_size = _require_positive_int(
        "decision_model_warmth_window_size", decision_model_warmth_window_size
    )
    decision_system_group_k = _require_positive_int("decision_system_group_k", decision_system_group_k)
    decision_system_min_system_len = _require_positive_int("decision_system_min_system_len", decision_system_min_system_len)
    run_warmup_steps = int(run_warmup_steps)
    if run_warmup_steps < 0:
        raise ValueError("run_warmup_steps must be >= 0")
    run_learn_after_warmup = bool(run_learn_after_warmup)
    plot_max_label_len = _require_positive_int("plot_max_label_len", plot_max_label_len)
    plot_label_color_mode = str(plot_label_color_mode).strip().lower()
    if plot_label_color_mode not in {"by_group", "explicit", "none"}:
        raise ValueError("plot_label_color_mode must be one of: by_group, explicit, none")

    gt_system_per_source_hits = int(gt_system_per_source_hits)
    if gt_system_per_source_hits <= 0:
        raise ValueError("gt_system_per_source_hits must be > 0")

    # ---- Features ----
    features: Dict[str, Any] = {
        "timestamp": {
            "type": "datetime",
            "format": "%Y-%m-%d %H:%M:%S",
            "timeOfDay": 21,
            "weekend": 0,
            "dayOfWeek": 0,
            "holiday": 0,
            "season": 0,
            "encode": False
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
            mn, mx = calibrate_min_max(
                s,
                low_q=low_q,
                high_q=high_q,
                margin=margin,
                floor=calibration_min_floor,
                ceil=calibration_max_ceil,
            )

            if feature_overrides is not None and feat_name in feature_overrides:
                ov = feature_overrides.get(feat_name) or {}
                if "minVal" in ov:
                    mn = float(ov["minVal"])
                if "maxVal" in ov:
                    mx = float(ov["maxVal"])
                if float(mx) <= float(mn):
                    raise ValueError(f"feature_overrides for '{feat_name}' invalid: maxVal must be > minVal")

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
                    "features": [feat_name],
                }
    elif model_layout == "single":
        # one model for everything; include all sources; include all features
        models[f"{usecase}_model"] = {
            "sources": [s.name for s in sources],
            "features": list(all_feature_names),
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
                "features": chunk,
            }

    # ---- Data sources ----
    data_sources: List[Dict[str, Any]] = []
    for src in sources:
        data_sources.append(
            {
                "name": src.name,
                "kind": "csv",
                "path": src.path,
                **({"unit": src.unit} if src.unit else {}),
                "timestamp_col": src.timestamp_col,
                "timestamp_format": src.timestamp_format,
                "fields": dict(src.fields),
                **(
                    {
                        "labels": {
                            "event_windows": [
                                (
                                    {
                                        "name": ev["name"],
                                        "kind": ev.get("kind", "primary_gt"),
                                        "start": ev["start"],
                                        "end": ev["end"],
                                    }
                                    if ev.get("name")
                                    else {
                                        "kind": ev.get("kind", "primary_gt"),
                                        "start": ev["start"],
                                        "end": ev["end"],
                                    }
                                )
                                for ev in src.event_windows
                            ]
                        }
                    } if src.event_windows else {}
                ),
            }
        )

    decision_groups_cfg: Dict[str, Dict[str, Any]] = {}
    valid_models = set(models.keys())
    for gname, gspec in (decision_groups or {}).items():
        g = str(gname).strip()
        if not g:
            raise ValueError("decision_groups keys must be non-empty strings")
        if not isinstance(gspec, Mapping):
            raise ValueError(f"decision_groups['{g}'] must be a mapping")

        members = gspec.get("members") or []
        if not isinstance(members, Sequence) or len(members) == 0:
            raise ValueError(f"decision_groups['{g}'].members must be a non-empty sequence")
        member_list = [str(m).strip() for m in members if str(m).strip()]
        if not member_list:
            raise ValueError(f"decision_groups['{g}'].members must contain at least one model")
        unknown = [m for m in member_list if m not in valid_models]
        if unknown:
            raise ValueError(
                f"decision_groups['{g}'] contains unknown model(s): {unknown}. "
                f"Valid models: {sorted(valid_models)}"
            )

        min_instant_members = _require_positive_int(
            f"decision_groups['{g}'].min_instant_members",
            gspec.get("min_instant_members", 1),
        )
        if min_instant_members > len(member_list):
            raise ValueError(
                f"decision_groups['{g}'].min_instant_members={min_instant_members} "
                f"cannot exceed group size={len(member_list)}"
            )

        min_group_warmth = float(gspec.get("min_group_warmth", 0.5))
        if not (0.0 <= min_group_warmth <= 1.0):
            raise ValueError(f"decision_groups['{g}'].min_group_warmth must be in [0,1]")

        decision_groups_cfg[g] = {
            "members": member_list,
            "min_instant_members": int(min_instant_members),
            "min_group_warmth": float(min_group_warmth),
        }

    if not decision_groups_cfg:
        raise ValueError("decision_groups must define at least one group")
    if decision_system_group_k > len(decision_groups_cfg):
        raise ValueError(
            f"decision_system_group_k={decision_system_group_k} cannot exceed number of groups={len(decision_groups_cfg)}"
        )

    plot_label_colors_cfg: Optional[Dict[str, str]] = None
    if plot_label_color_mode == "explicit":
        raw = dict(plot_model_label_colors or {})
        if not raw:
            raise ValueError("plot_model_label_colors must be provided when plot_label_color_mode='explicit'")
        plot_label_colors_cfg = {str(k): str(v) for k, v in raw.items()}

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
        "run": {
            # Canonical run lifecycle keys consumed by run_pipeline.
            "warmup_steps": int(run_warmup_steps),
            "learn_after_warmup": bool(run_learn_after_warmup),
        },
        "decision": {
            "score_key": str(decision_score_key),
            "threshold": float(decision_threshold),
            "method": str(decision_method),
            "model_warmth": {
                "window_size": int(decision_model_warmth_window_size),
            },
            "system": {
                "group_k": int(decision_system_group_k),
                "min_system_len": int(decision_system_min_system_len),
            },
            "groups": decision_groups_cfg,
        },

        "plot": {
            "enable": bool(plot_enable),
            "step_pause_s": float(plot_step_pause_s),
            "window": int(plot_window),
            "show_ground_truth": bool(plot_show_ground_truth),
            "show_warmup_span": bool(plot_show_warmup_span),
            "max_label_len": int(plot_max_label_len),
            "label_color_mode": str(plot_label_color_mode),
            **({"model_label_colors": plot_label_colors_cfg} if plot_label_colors_cfg is not None else {}),
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


def _parse_ts_list(raw: str) -> List[str]:
    # comma-separated list of timestamps
    items = [x.strip() for x in (raw or "").split(",")]
    return [x for x in items if x]


def _parse_event_window_line(raw: str, ts_format: str) -> Dict[str, str]:
    parts = [x.strip() for x in str(raw).split("|")]
    if len(parts) not in (3, 4):
        raise ValueError(
            "Event window must be: name|start|end or name|kind|start|end"
        )
    if len(parts) == 3:
        name, start, end = parts
        kind = "primary_gt"
    else:
        name, kind, start, end = parts
    if not name:
        raise ValueError("Event window name cannot be empty")
    if kind not in {"primary_gt", "explanatory"}:
        raise ValueError("Event window kind must be one of: primary_gt, explanatory")
    start_dt = datetime.strptime(start, ts_format)
    end_dt = datetime.strptime(end, ts_format)
    if end_dt < start_dt:
        raise ValueError("Event window end must be >= start")
    return {"name": name, "kind": kind, "start": start, "end": end}


def _validate_event_windows(
    windows: List[Dict[str, str]],
    ts_format: str,
) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen = set()
    for i, ev in enumerate(windows):
        name = str(ev.get("name") or "").strip()
        kind = str(ev.get("kind") or "primary_gt").strip()
        start = str(ev.get("start") or "").strip()
        end = str(ev.get("end") or "").strip()

        if not start or not end:
            raise ValueError(f"event_windows[{i}] must contain start/end")
        if kind not in {"primary_gt", "explanatory"}:
            raise ValueError(f"event_windows[{i}].kind must be one of: primary_gt, explanatory")

        start_dt = datetime.strptime(start, ts_format)
        end_dt = datetime.strptime(end, ts_format)
        if end_dt < start_dt:
            raise ValueError(f"event_windows[{i}] end must be >= start")

        key = (name, kind, start, end)
        if key in seen:
            continue
        seen.add(key)

        out.append({"name": name, "kind": kind, "start": start, "end": end})
    return out


def _load_event_windows_from_file(
    spec: str,
    *,
    source_name: str,
    default_key: str,
    ts_format: str,
) -> Optional[List[Dict[str, str]]]:
    raw = (spec or "").strip()
    if not raw.startswith("@file:"):
        return None

    path = raw[len("@file:"):].strip()
    p = Path(path)
    if not p.exists():
        raise ValueError(f"Event-window file does not exist: {p}")

    blob = json.loads(p.read_text())

    candidate = None
    if isinstance(blob, list):
        candidate = blob
    elif isinstance(blob, dict):
        for k in (source_name, default_key, "event_windows"):
            if k in blob:
                candidate = blob[k]
                break

    if not isinstance(candidate, list):
        raise ValueError("Event-window file must resolve to list")

    return _validate_event_windows(candidate, ts_format)


def _load_gt_from_file(spec: str, *, source_name: str, default_key: str) -> Optional[List[str]]:
    """
    Support wizard input:
      Ground truth timestamps: @file:/path/to/gt_timestamps.json

    Accepted JSON shapes:
      - list[str]
      - {"timestamps": [...]}  (generic)
      - {"sA": [...], "sB": [...], ...} (keyed by source name)
      - {"meta": {...}, "<key>": [...]} (we ignore meta)
    """
    raw = (spec or "").strip()
    if not raw.startswith("@file:"):
        return None
    path = raw[len("@file:") :].strip()
    if not path:
        raise ValueError("GT file spec is empty; use @file:<path>")
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise ValueError(f"GT file does not exist: {p}")
    blob = json.loads(p.read_text())
    if isinstance(blob, list):
        return [str(x) for x in blob]
    if isinstance(blob, dict):
        # prefer explicit per-source key, then default_key (stem), then 'timestamps'
        for k in (source_name, default_key, "timestamps"):
            v = blob.get(k)
            if v is None:
                continue
            if not isinstance(v, list):
                raise ValueError(f"GT file key '{k}' must be a list of timestamp strings")
            return [str(x) for x in v]
    raise ValueError("GT file must be a JSON list or object containing a timestamps list")


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
        unit = _prompt("Unit (blank=None)", "").strip() or None
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

        print("\n(Optional) Ground-truth event windows for this source.")
        print("Enter one per line as:")
        print("  name|YYYY-mm-dd HH:MM:SS|YYYY-mm-dd HH:MM:SS")
        print("  name|primary_gt|YYYY-mm-dd HH:MM:SS|YYYY-mm-dd HH:MM:SS")
        print("  name|explanatory|YYYY-mm-dd HH:MM:SS|YYYY-mm-dd HH:MM:SS")
        print("Press Enter on empty line when done.")
        print("Or enter once:")
        print("  @file:/path/to/event_windows.json")

        event_windows: Optional[List[Dict[str, str]]] = None
        first = _prompt("Event window", "").strip()

        if first:
            loaded = _load_event_windows_from_file(
                first,
                source_name=name,
                default_key=p.stem,
                ts_format=ts_fmt,
            )
            if loaded is not None:
                event_windows = loaded
            else:
                buf = [_parse_event_window_line(first, ts_fmt)]
                while True:
                    nxt = _prompt("Event window", "").strip()
                    if not nxt:
                        break
                    buf.append(_parse_event_window_line(nxt, ts_fmt))
                event_windows = _validate_event_windows(buf, ts_fmt)

        out.append(
            SourceSpec(
                name=name,
                path=str(p),
                timestamp_col=ts_col,
                timestamp_format=ts_fmt,
                fields=fields,
                unit=unit,
                event_windows=event_windows,
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
    decision_method = _prompt("decision.method", "grouped_consensus")
    decision_threshold = _prompt_float("decision.threshold", 0.99)
    decision_score_key = _prompt("decision.score_key", "p")
    decision_model_warmth_window_size = _prompt_int("decision.model_warmth.window_size", 8)
    decision_system_group_k = _prompt_int("decision.system.group_k", 2)
    decision_system_min_system_len = _prompt_int("decision.system.min_system_len", 3)

    print("\n--- Decision groups ---")
    # Reconstruct the model names exactly as build_usecase_config will.
    feature_names: List[str] = []
    for src in sources:
        feature_names.extend(list(src.fields.keys()))

    model_names: List[str] = []
    if layout == "separate":
        for feat_name in feature_names:
            model_names.append(f"{feat_name}_model")
    elif layout == "single":
        model_names.append(f"{usecase}_model")
    elif layout == "chunk":
        k = int(chunk_size)
        for i in range(0, len(feature_names), k):
            model_names.append(f"{usecase}_chunk_{(i // k) + 1}_model")

    print(f"Available models: {', '.join(model_names)}")
    decision_groups: Dict[str, Dict[str, Any]] = {}
    num_groups = _prompt_int("decision.groups.num_groups", len(model_names))
    for i in range(num_groups):
        gname = _prompt(f"group {i+1} name", f"group_{i+1}").strip()
        if not gname:
            raise ValueError("Group name cannot be empty")
        members_raw = _prompt(
            f"models in {gname} (comma-separated)",
            ",".join([model_names[min(i, len(model_names) - 1)]]),
        )
        members = _parse_csv_list(members_raw)
        if not members:
            raise ValueError(f"Group '{gname}' must contain at least one model")
        unknown = [m for m in members if m not in model_names]
        if unknown:
            raise ValueError(f"Unknown model(s) in group '{gname}': {unknown}")

        default_min_instant = 1 if len(members) == 1 else min(2, len(members))
        min_instant_members = _prompt_int(f"{gname}.min_instant_members", default_min_instant)
        min_group_warmth = _prompt_float(f"{gname}.min_group_warmth", 0.50 if len(members) == 1 else 0.45)
        decision_groups[gname] = {
            "members": members,
            "min_instant_members": int(min_instant_members),
            "min_group_warmth": float(min_group_warmth),
        }

    print("\n--- Run lifecycle (warmup + learn policy) ---")
    warmup_steps = _prompt_int("run.warmup_steps (timesteps; 0 disables warmup)", 0)
    learn_after = _prompt_choice(
        "run.learn_after_warmup (true|false)",
        ["true", "false"],
        "true",
    ) == "true"

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
        decision_threshold=decision_threshold,
        decision_score_key=decision_score_key,
        decision_model_warmth_window_size=decision_model_warmth_window_size,
        decision_system_group_k=decision_system_group_k,
        decision_system_min_system_len=decision_system_min_system_len,
        decision_groups=decision_groups,
        run_warmup_steps=warmup_steps,
        run_learn_after_warmup=learn_after,
    )

    out_dir = Path(args.out_dir)
    out_path = out_dir / f"{usecase}.yaml"
    _write_text_atomic(out_path, yaml.safe_dump(cfg, sort_keys=False))

    print(f"[make_usecase_config] wrote -> {out_path}")


if __name__ == "__main__":
    main()
