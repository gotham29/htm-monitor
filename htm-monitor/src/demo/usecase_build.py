# src/demo/usecase_build.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import yaml

from demo.make_usecase_config import build_usecase_config, sources_from_dicts


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


def _validate_spec(raw: Dict[str, Any]) -> None:
    allowed = {"usecase", "sources", "params"}
    extra = set(raw.keys()) - allowed
    if extra:
        raise ValueError(f"Spec contains unknown top-level key(s): {sorted(extra)}")

    usecase = raw.get("usecase")
    sources_raw = raw.get("sources")
    params = raw.get("params")

    if not isinstance(usecase, str) or not usecase.strip():
        raise ValueError("Spec.usecase must be a non-empty string")
    if not isinstance(sources_raw, list) or not sources_raw:
        raise ValueError("Spec.sources must be a non-empty list")
    if not isinstance(params, dict):
        raise ValueError("Spec.params must be a mapping")

    # Validate sources shape so sources_from_dicts doesn't get fed garbage.
    for i, s in enumerate(sources_raw):
        if not isinstance(s, dict):
            raise ValueError(f"Spec.sources[{i}] must be a mapping")
        for k in ("name", "path", "timestamp_col", "timestamp_format", "fields"):
            if k not in s:
                raise ValueError(f"Spec.sources[{i}] missing required key '{k}'")
        fields = s.get("fields")
        if not isinstance(fields, dict) or len(fields) == 0:
            raise ValueError(f"Spec.sources[{i}].fields must be a non-empty mapping")

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

    sources = sources_from_dicts(sources_raw)
    cfg = build_usecase_config(usecase, sources, **params)

    out_path = Path(args.out) if args.out else (Path(args.out_dir) / f"{usecase}.yaml")
    _write_text_atomic(out_path, yaml.safe_dump(cfg, sort_keys=False))
    print(f"[usecase_build] wrote -> {out_path}")


if __name__ == "__main__":
    main()