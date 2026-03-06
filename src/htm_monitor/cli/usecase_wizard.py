# src/htm-monitor/cli/usecase_wizard.py

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Mapping

import yaml
import pandas as pd

from .make_usecase_config import build_usecase_config, collect_sources_interactive, sources_from_dicts, sources_to_dicts
from htm_monitor.diagnostics.encoding_sanity import summarize_range, write_linear_hist_png


@dataclass(frozen=True)
class UsecaseBuildSpec:
    """
    Reproducible spec for generating a usecase config.
    This is what you’ll want for “power user” workflows (versioned, shareable).
    """
    usecase: str
    sources: List[Dict[str, Any]]  # serialized SourceSpec dicts (no code objects)
    params: Dict[str, Any]         # args to build_usecase_config
    calibration: Dict[str, Any]    # wizard-time calibration policy (for plots + defaults)
    overrides: Dict[str, Any]      # frozen per-feature overrides


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


def _open_file(path: Path) -> None:
    """
    Best-effort open; no try/except. If it fails, we print the path.
    """
    p = str(path)
    if sys.platform == "darwin":
        subprocess.run(["open", p], check=False)
    elif sys.platform.startswith("linux"):
        subprocess.run(["xdg-open", p], check=False)
    else:
        # Windows or unknown
        pass
    print(f"[wizard] plot: {p}")


def _prompt_optional_float(msg: str) -> Optional[float]:
    raw = _prompt(msg, "").strip()
    if raw == "":
        return None
    return float(raw)


def _prompt_optional_float_default(msg: str, default: Optional[float]) -> Optional[float]:
    d = "" if default is None else str(default)
    raw = _prompt(msg, d).strip()
    if raw == "":
        return default
    return float(raw)


def run_interactive() -> UsecaseBuildSpec:
    usecase = _require_safe_usecase_name(_prompt("Use-case name"))

    print("\n--- Calibration policy (used for preview plots + default config) ---")
    low_q = _prompt_float("low_q", 0.01)
    high_q = _prompt_float("high_q", 0.99)
    margin = _prompt_float("margin", 0.03)
    hist_bins = _prompt_int("hist_bins (linear)", 80)
    # For count-like signals (NAB tweets), we never want negative encoder mins.
    # User can set blank to disable (None).
    min_floor = _prompt_optional_float_default("min_floor (blank=None; for counts use 0)", 0.0)

    print("\n--- Sources ---")
    sources = collect_sources_interactive()

    print("\n--- Model layout ---")
    model_layout = _prompt_choice("Model layout (separate|single|chunk)", ["separate", "single", "chunk"], "separate")
    chunk_size = 2
    if model_layout == "chunk":
        chunk_size = _prompt_int("Chunk size (features per model)", 2)

    # Preview calibration now (show plots while wizard runs), allow one-shot overrides.
    print("\n--- Encoding sanity (preview + freeze min/max once) ---")
    overrides_features: Dict[str, Dict[str, float]] = {}
    out_plot_dir = Path("outputs") / usecase / "calibration_plots"
    out_plot_dir.mkdir(parents=True, exist_ok=True)

    # We need real SourceSpec objects to read CSVs reliably.
    src_specs = sources_from_dicts(sources_to_dicts(sources))
    for src in src_specs:
        for feat_name, col_name in src.fields.items():
            df = pd.read_csv(src.path, usecols=[col_name], low_memory=False)
            s = df[col_name]

            summ = summarize_range(s, feature=feat_name, low_q=low_q, high_q=high_q, margin=margin, floor=min_floor)
            png = write_linear_hist_png(
                s,
                summary=summ,
                out_png=out_plot_dir / f"{feat_name}.png",
                bins=int(hist_bins),
                title_suffix=f"{src.name}:{col_name}",
            )
            _open_file(png)

            print(
                f"[wizard] {feat_name}: suggested minVal={summ.min_val:.6g} maxVal={summ.max_val:.6g} "
                f"(q_low={summ.q_low:.6g} q_high={summ.q_high:.6g}, "
                f"clip_low={summ.clip_low_rate:.4f} clip_high={summ.clip_high_rate:.4f})"
            )
            print("Press Enter to accept suggested min/max, or type explicit overrides.")
            ov_min = _prompt_optional_float(f"  override {feat_name}.minVal (blank=accept)")
            ov_max = _prompt_optional_float(f"  override {feat_name}.maxVal (blank=accept)")
            if ov_min is not None or ov_max is not None:
                mn = float(ov_min) if ov_min is not None else float(summ.min_val)
                mx = float(ov_max) if ov_max is not None else float(summ.max_val)
                if mx <= mn:
                    raise ValueError(f"Invalid override for '{feat_name}': maxVal must be > minVal")
                overrides_features[feat_name] = {"minVal": mn, "maxVal": mx}
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

    print("\n--- Run lifecycle (warmup + learn policy) ---")
    run_warmup_steps = _prompt_int("run.warmup_steps (timesteps; 0 disables warmup)", 0)
    run_learn_after_warmup = (
        _prompt_choice(
            "run.learn_after_warmup (true|false)",
            ["true", "false"],
            "true",
        )
        == "true"
    )

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
        run_warmup_steps=run_warmup_steps,
        run_learn_after_warmup=run_learn_after_warmup,
    )
    _validate_params_mapping(params)

    calibration: Dict[str, Any] = dict(
        method="quantile",
        low_q=low_q,
        high_q=high_q,
        margin=margin,
        hist_bins=int(hist_bins),
        min_floor=min_floor,
    )
    overrides: Dict[str, Any] = dict(features=overrides_features)

    return UsecaseBuildSpec(
        usecase=usecase,
        sources=sources_to_dicts(sources),
        params=params,
        calibration=calibration,
        overrides=overrides,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Power-user wizard: generate a build spec and/or final usecase config.")
    ap.add_argument("--out-dir", default="configs", help="Where to write <usecase>.yaml")
    ap.add_argument("--spec-out", default=None, help="If set, write build spec yaml here (e.g. specs/my_usecase.build.yaml)")
    ap.add_argument("--no-config", action="store_true", help="Only write build spec, skip generating final config")
    args = ap.parse_args()

    spec = run_interactive()

    if args.spec_out:
        spec_path = Path(args.spec_out)
        blob = {
            "usecase": spec.usecase,
            "sources": spec.sources,
            "params": spec.params,
            "calibration": spec.calibration,
            "overrides": spec.overrides,
        }
        _write_text_atomic(spec_path, yaml.safe_dump(blob, sort_keys=False))
        print(f"[usecase_wizard] wrote build spec -> {spec_path}")

    if not args.no_config:
        sources = sources_from_dicts(spec.sources)
        feature_overrides = (spec.overrides.get("features") if isinstance(spec.overrides, dict) else None) or None
        min_floor = None
        if isinstance(spec.calibration, dict):
            min_floor = spec.calibration.get("min_floor")
        cfg = build_usecase_config(
            spec.usecase,
            sources,
            feature_overrides=feature_overrides,
            calibration_min_floor=min_floor,
            **spec.params,
        )
        out_dir = Path(args.out_dir)
        out_path = out_dir / f"{spec.usecase}.yaml"
        _write_text_atomic(out_path, yaml.safe_dump(cfg, sort_keys=False))
        print(f"[usecase_wizard] wrote usecase config -> {out_path}")


if __name__ == "__main__":
    main()