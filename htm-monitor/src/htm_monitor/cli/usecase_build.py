# src/htm-monitor/cli/usecase_build.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Mapping

import yaml

from .make_usecase_config import build_usecase_config, sources_from_dicts


def _write_text_atomic(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _load_yaml_mapping(path: Path) -> Dict[str, Any]:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("Spec must be a mapping at top-level")
    return raw


def _normalize_sources_inplace(sources_raw: List[Dict[str, Any]]) -> None:
    """
    Spec format convenience:
      - allow source.gt_timestamps: [<ts_str>, ...]
    Canonical config format (consumed by run_pipeline):
      - source.labels.timestamps: [<ts_str>, ...]

    We normalize the spec in-place so downstream config generation is consistent.
    """
    for i, s in enumerate(sources_raw):
        if not isinstance(s, dict):
            continue

        has_gt = "gt_timestamps" in s
        has_labels = "labels" in s and s.get("labels") is not None
        if has_gt and has_labels:
            raise ValueError(
                f"Spec.sources[{i}] cannot contain both 'gt_timestamps' and 'labels' "
                "(ambiguous ground truth format)"
            )

        if has_gt:
            gt = s.get("gt_timestamps")
            # Move to canonical spot
            s.pop("gt_timestamps", None)
            s["labels"] = {"timestamps": gt}

        # Spec convenience: allow "labels: {}" (or null) without timestamps.
        # If present but empty, drop it so downstream config generation is clean.
        if "labels" in s and (s.get("labels") is None or s.get("labels") == {}):
            s.pop("labels", None)


def _validate_spec(raw: Dict[str, Any]) -> None:
    allowed = {"usecase", "sources", "params", "calibration", "overrides"}
    extra = set(raw.keys()) - allowed
    if extra:
        raise ValueError(f"Spec contains unknown top-level key(s): {sorted(extra)}")

    usecase = raw.get("usecase")
    sources_raw = raw.get("sources")
    params = raw.get("params")
    overrides = raw.get("overrides")

    if not isinstance(usecase, str) or not usecase.strip():
        raise ValueError("Spec.usecase must be a non-empty string")
    if not isinstance(sources_raw, list) or not sources_raw:
        raise ValueError("Spec.sources must be a non-empty list")
    if not isinstance(params, dict):
        raise ValueError("Spec.params must be a mapping")
    if overrides is not None and not isinstance(overrides, dict):
        raise ValueError("Spec.overrides must be a mapping if provided")

    # Validate sources shape so sources_from_dicts doesn't get fed garbage.
    for i, s in enumerate(sources_raw):
        # NOTE: normalization happens in main(); here we validate both pre- and post-normalization shapes.
        if not isinstance(s, dict):
            raise ValueError(f"Spec.sources[{i}] must be a mapping")
        for k in ("name", "path", "timestamp_col", "timestamp_format", "fields"):
            if k not in s:
                raise ValueError(f"Spec.sources[{i}] missing required key '{k}'")
        fields = s.get("fields")
        if not isinstance(fields, dict) or len(fields) == 0:
            raise ValueError(f"Spec.sources[{i}].fields must be a non-empty mapping")

        # Optional ground truth (spec convenience): gt_timestamps
        if "gt_timestamps" in s:
            gt = s.get("gt_timestamps")
            if not isinstance(gt, list) or not all(isinstance(x, str) for x in gt):
                raise ValueError(f"Spec.sources[{i}].gt_timestamps must be a list of timestamp strings")

        # Optional canonical ground truth: labels.timestamps
        if "labels" in s and s.get("labels") is not None:
            labels = s.get("labels")
            if not isinstance(labels, Mapping):
                raise ValueError(f"Spec.sources[{i}].labels must be a mapping if provided")
            if "timestamps" in labels:
                ts = labels.get("timestamps")
                if not isinstance(ts, list) or not all(isinstance(x, str) for x in ts):
                    raise ValueError(f"Spec.sources[{i}].labels.timestamps must be a list of timestamp strings")

    # YAML foot-gun: non-string keys can appear; reject them early.
    bad_param_keys = [k for k in params.keys() if not isinstance(k, str)]
    if bad_param_keys:
        raise ValueError(f"Spec.params must have string keys; got: {bad_param_keys}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build usecase config from a build spec yaml.")
    ap.add_argument("--spec", required=True, help="Path to build spec yaml produced by usecase_wizard")
    ap.add_argument("--out", default=None, help="Output config path. Default: configs/<usecase>.yaml")
    ap.add_argument("--out-dir", default="configs", help="Used if --out is not provided")
    args = ap.parse_args()

    spec_path = Path(args.spec)
    if not spec_path.exists() or not spec_path.is_file():
        raise ValueError(f"Spec does not exist: {spec_path}")

    raw = _load_yaml_mapping(spec_path)
    _validate_spec(raw)

    usecase = raw.get("usecase")
    sources_raw = raw.get("sources")
    params = raw.get("params")
    overrides = raw.get("overrides")  # optional

    # Normalize spec conveniences into canonical config shape.
    _normalize_sources_inplace(sources_raw)
    _validate_spec(raw)  # re-validate after normalization to enforce invariant

    sources = sources_from_dicts(sources_raw)

    feature_overrides = None
    if isinstance(overrides, dict):
        feats = overrides.get("features")
        if feats is not None and not isinstance(feats, dict):
            raise ValueError("Spec.overrides.features must be a mapping if provided")
        feature_overrides = feats

    cfg = build_usecase_config(usecase, sources, feature_overrides=feature_overrides, **params)

    out_path = Path(args.out) if args.out else (Path(args.out_dir) / f"{usecase}.yaml")
    _write_text_atomic(out_path, yaml.safe_dump(cfg, sort_keys=False))
    print(f"[usecase_build] wrote -> {out_path}")


if __name__ == "__main__":
    main()