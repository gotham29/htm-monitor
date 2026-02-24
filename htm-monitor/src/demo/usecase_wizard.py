# src/demo/usecase_wizard.py

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from demo.make_usecase_config import build_usecase_config, collect_sources_interactive, sources_from_dicts, sources_to_dicts


@dataclass(frozen=True)
class UsecaseBuildSpec:
    """
    Reproducible spec for generating a usecase config.
    This is what you’ll want for “power user” workflows (versioned, shareable).
    """
    usecase: str
    sources: List[Dict[str, Any]]  # serialized SourceSpec dicts (no code objects)
    params: Dict[str, Any]         # args to build_usecase_config


def _prompt(msg: str, default: Optional[str] = None) -> str:
    if default is not None:
        raw = input(f"{msg} [{default}]: ").strip()
        return raw if raw else default
    return input(f"{msg}: ").strip()


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


def _validate_params_mapping(params: Dict[str, Any]) -> None:
    # YAML foot-gun: non-string keys can appear; reject them early.
    bad = [k for k in params.keys() if not isinstance(k, str)]
    if bad:
        raise ValueError(f"Params must have string keys; got: {bad}")


def _prompt_int(msg: str, default: int) -> int:
    raw = _prompt(msg, str(default))
    try:
        return int(raw)
    except Exception:
        raise ValueError(f"{msg} must be an int")


def _prompt_float(msg: str, default: float) -> float:
    raw = _prompt(msg, str(default))
    try:
        return float(raw)
    except Exception:
        raise ValueError(f"{msg} must be a float")


def _prompt_choice(msg: str, choices: List[str], default: str) -> str:
    raw = _prompt(msg, default).strip().lower()
    allowed = [c.lower() for c in choices]
    if raw not in allowed:
        raise ValueError(f"{msg} must be one of: {', '.join(choices)}")
    return raw


def run_interactive() -> UsecaseBuildSpec:
    usecase = _require_safe_usecase_name(_prompt("Use-case name"))

    print("\n--- Sources ---")
    sources = collect_sources_interactive()

    print("\n--- Model layout ---")
    model_layout = _prompt_choice("Model layout (separate|single|chunk)", ["separate", "single", "chunk"], "separate")
    chunk_size = 2
    if model_layout == "chunk":
        chunk_size = _prompt_int("Chunk size (features per model)", 2)

    print("\n--- Calibration ---")
    low_q = _prompt_float("low_q", 0.01)
    high_q = _prompt_float("high_q", 0.99)
    margin = _prompt_float("margin", 0.03)

    print("\n--- RDSE ---")
    rdse_size = _prompt_int("rdse_size", 2048)
    rdse_active_bits = _prompt_int("rdse_active_bits", 40)
    rdse_num_buckets = _prompt_int("rdse_num_buckets", 130)
    seed_base = _prompt_int("seed_base", 42)

    print("\n--- Decision ---")
    decision_method = _prompt("decision.method", "kofn_window")
    decision_k = _prompt_int("decision.k", 2)
    decision_window_size = _prompt_int("decision.window.size", 24)
    decision_per_model_hits = _prompt_int("decision.window.per_model_hits", 2)
    decision_threshold = _prompt_float("decision.threshold", 0.997)
    decision_score_key = _prompt("decision.score_key", "anomaly_probability")

    params: Dict[str, Any] = dict(
        rdse_size=rdse_size,
        rdse_active_bits=rdse_active_bits,
        rdse_num_buckets=rdse_num_buckets,
        seed_base=seed_base,
        low_q=low_q,
        high_q=high_q,
        margin=margin,
        model_layout=model_layout,
        chunk_size=chunk_size,
        decision_method=decision_method,
        decision_k=decision_k,
        decision_window_size=decision_window_size,
        decision_per_model_hits=decision_per_model_hits,
        decision_threshold=decision_threshold,
        decision_score_key=decision_score_key,
    )
    _validate_params_mapping(params)

    return UsecaseBuildSpec(usecase=usecase, sources=sources_to_dicts(sources), params=params)


def main() -> None:
    ap = argparse.ArgumentParser(description="Power-user wizard: generate a build spec and/or final usecase config.")
    ap.add_argument("--out-dir", default="configs", help="Where to write <usecase>.yaml")
    ap.add_argument("--spec-out", default=None, help="If set, write build spec yaml here (e.g. specs/my_usecase.build.yaml)")
    ap.add_argument("--no-config", action="store_true", help="Only write build spec, skip generating final config")
    args = ap.parse_args()

    spec = run_interactive()

    if args.spec_out:
        spec_path = Path(args.spec_out)
        blob = {"usecase": spec.usecase, "sources": spec.sources, "params": spec.params}
        _write_text_atomic(spec_path, yaml.safe_dump(blob, sort_keys=False))
        print(f"[usecase_wizard] wrote build spec -> {spec_path}")

    if not args.no_config:
        sources = sources_from_dicts(spec.sources)
        cfg = build_usecase_config(spec.usecase, sources, **spec.params)
        out_dir = Path(args.out_dir)
        out_path = out_dir / f"{spec.usecase}.yaml"
        _write_text_atomic(out_path, yaml.safe_dump(cfg, sort_keys=False))
        print(f"[usecase_wizard] wrote usecase config -> {out_path}")


if __name__ == "__main__":
    main()