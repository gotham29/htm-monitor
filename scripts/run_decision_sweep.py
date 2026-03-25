#scripts/run_decision_sweep.py

from __future__ import annotations

import itertools
import json
import subprocess
import sys
from pathlib import Path

import yaml


# -------------------------
# Sweep space (THIS is the key)
# -------------------------

K_VALUES = [2,3] #2,3 
WINDOW_SIZES = [12,] #12,18,24
PER_MODEL_HITS = [3,4,5] #2,3,4
THRESHOLDS = [0.97, 0.98, 0.99, 0.995,] #0.97,0.98,0.99


# -------------------------
# Helpers
# -------------------------

def write_config(base_cfg: dict, out_path: Path, *, k, w, hits, thr):
    cfg = dict(base_cfg)

    cfg["decision"] = {
        "score_key": "anomaly_probability",
        "threshold": thr,
        "method": "kofn_window",
        "k": k,
        "window": {
            "size": w,
            "per_model_hits": hits,
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.dump(cfg, sort_keys=False))


def run_cmd(cmd, cwd):
    print("\n>>>", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


# -------------------------
# Main
# -------------------------

def main():
    repo = Path(__file__).resolve().parents[1]

    base_config_path = repo / "configs/generated/casio_grid.yaml"
    base_cfg = yaml.safe_load(base_config_path.read_text())

    results = []

    for k, w, hits, thr in itertools.product(
        K_VALUES, WINDOW_SIZES, PER_MODEL_HITS, THRESHOLDS
    ):
        name = f"k{k}_w{w}_h{hits}_thr{str(thr).replace('.', 'p')}"

        cfg_path = repo / "configs/generated/sweeps" / f"{name}.yaml"
        run_dir = repo / "outputs/sweeps" / name

        summary_path = run_dir / "analysis/run_summary.json"

        if summary_path.exists():
            print(f"SKIP {name}")
            continue

        write_config(base_cfg, cfg_path, k=k, w=w, hits=hits, thr=thr)

        # -------------------------
        # Run pipeline
        # -------------------------
        run_cmd(
            [
                sys.executable,
                "-m",
                "htm_monitor.cli.run_pipeline",
                "--defaults",
                "configs/htm_defaults.yaml",
                "--config",
                str(cfg_path.relative_to(repo)),
                "--run-dir",
                str(run_dir.relative_to(repo)),
                "--no-plot",
            ],
            repo,
        )

        # -------------------------
        # Analyze
        # -------------------------
        run_cmd(
            [
                sys.executable,
                "-m",
                "htm_monitor.cli.analyze_run",
                "--run-dir",
                str(run_dir.relative_to(repo)),
                "--config",
                str(cfg_path.relative_to(repo)),
            ],
            repo,
        )

        # -------------------------
        # Collect results
        # -------------------------
        j = json.loads(summary_path.read_text())
        sys_eval = j["ground_truth"]["system"]["eval"]

        results.append(
            {
                "name": name,
                "k": k,
                "window": w,
                "hits": hits,
                "thr": thr,
                "precision": sys_eval.get("precision"),
                "recall": sys_eval.get("recall"),
                "f1": sys_eval.get("f1"),
                "episodes": j["alerts"]["episodes"]["count"],
            }
        )

    # -------------------------
    # Rank results
    # -------------------------
    def score(r):
        return (
            r["recall"] or 0,
            r["precision"] or 0,
            -(r["episodes"] or 9999),
        )

    results = sorted(results, key=score, reverse=True)

    print("\n=== TOP CONFIGS ===")
    for r in results[:10]:
        print(
            f"{r['name']} | "
            f"P={r['precision']} R={r['recall']} F1={r['f1']} "
            f"episodes={r['episodes']}"
        )


if __name__ == "__main__":
    main()
