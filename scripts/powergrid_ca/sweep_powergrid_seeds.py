#scripts/sweep_powerfrig_seeds.py

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import statistics
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    return data


def dump_yaml(path: Path, data: dict) -> None:
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def feature_names_with_rdse_seed(cfg: dict) -> List[str]:
    out: List[str] = []
    feats = cfg.get("features") or {}
    if not isinstance(feats, dict):
        return out

    for feat_name, feat_cfg in feats.items():
        if not isinstance(feat_cfg, dict):
            continue
        # Common current shape: features.<name>.seed
        if "seed" in feat_cfg:
            out.append(str(feat_name))
            continue

        # Slightly more defensive: nested rdse block
        rdse = feat_cfg.get("rdse")
        if isinstance(rdse, dict) and "seed" in rdse:
            out.append(str(feat_name))
    return out


def set_feature_seed(cfg: dict, feat_name: str, seed: int) -> None:
    feat_cfg = cfg["features"][feat_name]
    if "seed" in feat_cfg:
        feat_cfg["seed"] = int(seed)
        return
    rdse = feat_cfg.get("rdse")
    if isinstance(rdse, dict) and "seed" in rdse:
        rdse["seed"] = int(seed)
        return
    raise KeyError(f"Could not find seed field for feature '{feat_name}'")


def apply_seed_block(cfg: dict, base_seed: int, ordered_features: List[str]) -> Dict[str, int]:
    assigned: Dict[str, int] = {}
    for i, feat_name in enumerate(ordered_features):
        seed = base_seed + i
        set_feature_seed(cfg, feat_name, seed)
        assigned[feat_name] = seed
    return assigned


def run_cmd(cmd: List[str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd))
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def _normalize_eps(xs: Any) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    if not isinstance(xs, list):
        return out
    for row in xs:
        if (
            isinstance(row, (list, tuple))
            and len(row) == 2
            and isinstance(row[0], (int, float))
            and isinstance(row[1], (int, float))
        ):
            s = int(row[0])
            e = int(row[1])
            if e < s:
                s, e = e, s
            out.append((s, e))
    return out


def load_run_summary(summary_path: Path) -> Dict[str, Any]:
    data = json.loads(summary_path.read_text())

    sys_eval = (
        ((data.get("ground_truth") or {}).get("system") or {}).get("eval")
        or {}
    )
    precision = sys_eval.get("precision")
    recall = sys_eval.get("recall")
    episodes = (((data.get("alerts") or {}).get("episodes") or {}).get("episodes") or [])

    fp_episodes = sys_eval.get("false_positive_episodes") or []

    time_range = data.get("time_range") or {}

    return {
        "precision": float(precision) if precision is not None else float("nan"),
        "recall": float(recall) if recall is not None else float("nan"),
        "n_alert_episodes": int(len(episodes)),
        "episodes": _normalize_eps(episodes),
        "false_positive_episodes": _normalize_eps(fp_episodes),
        "time_start": time_range.get("start"),
        "time_end": time_range.get("end"),
    }


def _overlap_or_near(a: Tuple[int, int], b: Tuple[int, int], gap_tolerance: int) -> bool:
    a0, a1 = int(a[0]), int(a[1])
    b0, b1 = int(b[0]), int(b[1])
    if a1 < b0:
        return (b0 - a1 - 1) <= int(gap_tolerance)
    if b1 < a0:
        return (a0 - b1 - 1) <= int(gap_tolerance)
    return True


def _cluster_non_gt_candidates(
    rows: List[Dict[str, Any]],
    *,
    gap_tolerance: int,
) -> List[Dict[str, Any]]:
    """
    Cluster repeated non-GT episodes across runs.
    We treat overlapping or near-touching [t_start, t_end] windows as the same candidate.
    """
    clusters: List[Dict[str, Any]] = []

    for row in rows:
        meta = {
            "run_name": row["run_name"],
            "base_seed": row["base_seed"],
            "per_model_hits": row["per_model_hits"],
            "min_alert_len": row["min_alert_len"],
            "threshold": row["threshold"],
            "precision": row["precision"],
            "recall": row["recall"],
            "run_dir": row["run_dir"],
        }

        for ep in row.get("false_positive_episodes", []):
            matched = None
            for cl in clusters:
                if _overlap_or_near((cl["t_start"], cl["t_end"]), ep, gap_tolerance):
                    matched = cl
                    break

            if matched is None:
                matched = {
                    "t_start": int(ep[0]),
                    "t_end": int(ep[1]),
                    "members": [],
                }
                clusters.append(matched)
            else:
                matched["t_start"] = min(int(matched["t_start"]), int(ep[0]))
                matched["t_end"] = max(int(matched["t_end"]), int(ep[1]))

            matched["members"].append(
                {
                    **meta,
                    "episode_t_start": int(ep[0]),
                    "episode_t_end": int(ep[1]),
                }
            )

    out: List[Dict[str, Any]] = []
    for i, cl in enumerate(sorted(clusters, key=lambda x: (x["t_start"], x["t_end"])), start=1):
        members = cl["members"]
        seeds = sorted(set(int(m["base_seed"]) for m in members))
        thresholds = sorted(set(float(m["threshold"]) for m in members))
        per_hits = sorted(set(int(m["per_model_hits"]) for m in members))
        min_lens = sorted(set(int(m["min_alert_len"]) for m in members))
        starts = [int(m["episode_t_start"]) for m in members]
        ends = [int(m["episode_t_end"]) for m in members]

        out.append(
            {
                "candidate_id": i,
                "t_start_min": int(min(starts)),
                "t_end_max": int(max(ends)),
                "t_start_median": int(statistics.median(starts)),
                "t_end_median": int(statistics.median(ends)),
                "length_median": int(statistics.median([(e - s + 1) for s, e in zip(starts, ends)])),
                "support_count": int(len(members)),
                "unique_seed_count": int(len(seeds)),
                "seed_list": seeds,
                "threshold_list": thresholds,
                "per_model_hits_list": per_hits,
                "min_alert_len_list": min_lens,
                "members": members,
            }
        )

    n_alert_episodes = len(episodes)

    out.sort(key=lambda x: (-x["unique_seed_count"], -x["support_count"], x["t_start_median"]))
    return out


def set_decision_params(
    cfg: dict,
    *,
    per_model_hits: int,
    min_alert_len: int,
) -> None:
    decision = cfg.get("decision")
    if not isinstance(decision, dict):
        raise ValueError("Config missing decision block")

    window = decision.get("window")
    if not isinstance(window, dict):
        raise ValueError("Config missing decision.window block")
    window["per_model_hits"] = int(per_model_hits)

    grouping = decision.get("grouping")
    if not isinstance(grouping, dict):
        raise ValueError("Config missing decision.grouping block")
    grouping["min_alert_len"] = int(min_alert_len)


def set_threshold(cfg: dict, threshold: float) -> None:
    decision = cfg.get("decision")
    if not isinstance(decision, dict):
        raise ValueError("Config missing decision block")
    decision["threshold"] = float(threshold)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", required=True)
    ap.add_argument("--defaults", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--base-seeds",
        nargs="+",
        type=int,
        required=True,
        help="Base seeds to try, e.g. 42 52 62 72 82",
    )
    ap.add_argument(
        "--feature-order",
        nargs="*",
        default=None,
        help="Optional explicit feature order for assigning seeds. Defaults to config order.",
    )
    ap.add_argument(
        "--per-model-hits",
        nargs="+",
        type=int,
        default=[5],
        help="Values to try for decision.window.per_model_hits",
    )
    ap.add_argument(
        "--min-alert-len",
        nargs="+",
        type=int,
        default=[6],
        help="Values to try for decision.grouping.min_alert_len",
    )
    ap.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.99],
        help="Values to try for decision.threshold",
    )
    ap.add_argument(
        "--candidate-gap-tolerance",
        type=int,
        default=6,
        help=(
            "Merge non-GT episodes into the same candidate window when they overlap or are within this many timesteps."
        ),
    )
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve()
    defaults_path = Path(args.defaults).resolve()
    config_path = Path(args.config).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_yaml(config_path)

    discovered = feature_names_with_rdse_seed(base_cfg)
    if not discovered:
        raise ValueError("No features with RDSE seed fields found in config")

    ordered_features = list(args.feature_order) if args.feature_order else discovered
    missing = [f for f in ordered_features if f not in discovered]
    if missing:
        raise ValueError(f"Requested feature_order not found in config seed-bearing features: {missing}")

    rows: List[dict] = []

    for threshold in args.thresholds:
        for per_model_hits in args.per_model_hits:
            for min_alert_len in args.min_alert_len:
                for base_seed in args.base_seeds:
                    
                    run_name = (
                        f"seed_{base_seed}"
                        f"__hits_{per_model_hits}"
                        f"__alertlen_{min_alert_len}"
                        f"__th_{threshold}"
                    )
                    cfg = load_yaml(config_path)
                    assigned = apply_seed_block(cfg, base_seed, ordered_features)
                    set_decision_params(
                        cfg,
                        per_model_hits=per_model_hits,
                        min_alert_len=min_alert_len,
                    )
                    set_threshold(cfg, threshold)

                    cfg_out = out_dir / f"{run_name}.yaml"
                    run_out = out_dir / run_name
                    if run_out.exists():
                        shutil.rmtree(run_out)
                    dump_yaml(cfg_out, cfg)

                    run_cmd(
                        [
                            sys.executable,
                            "-m",
                            "htm_monitor.cli.run_pipeline",
                            "--defaults",
                            str(defaults_path),
                            "--config",
                            str(cfg_out),
                            "--run-dir",
                            str(run_out),
                            "--no-plot",
                        ],
                        cwd=repo_root,
                    )

                    run_cmd(
                        [
                            sys.executable,
                            "-m",
                            "htm_monitor.cli.analyze_run",
                            "--run-dir",
                            str(run_out),
                            "--config",
                            str(cfg_out),
                        ],
                        cwd=repo_root,
                    )

                    summary_path = run_out / "analysis" / "run_summary.json"
                    summary = load_run_summary(summary_path)

                    row = {
                        "run_name": run_name,
                        "base_seed": base_seed,
                        "per_model_hits": per_model_hits,
                        "min_alert_len": min_alert_len,
                        "threshold": threshold,
                        "precision": summary["precision"],
                        "recall": summary["recall"],
                        "n_alert_episodes": summary["n_alert_episodes"],
                        "false_positive_episodes_json": json.dumps(summary["false_positive_episodes"]),
                        "alert_episodes_json": json.dumps(summary["episodes"]),
                        "config_path": str(cfg_out),
                        "run_dir": str(run_out),
                        "assigned_seeds_json": json.dumps(assigned, sort_keys=True),
                    }
                    rows.append(row)
                    print(row)

    csv_path = out_dir / "seed_sweep_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_name",
                "base_seed",
                "per_model_hits",
                "min_alert_len",
                "threshold",
                "precision",
                "recall",
                "n_alert_episodes",
                "false_positive_episodes_json",
                "alert_episodes_json",
                "config_path",
                "run_dir",
                "assigned_seeds_json",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    candidates = _cluster_non_gt_candidates(
        rows,
        gap_tolerance=int(args.candidate_gap_tolerance),
    )

    cand_json = out_dir / "candidate_non_gt_windows.json"
    cand_csv = out_dir / "candidate_non_gt_windows.csv"
    cand_json.write_text(json.dumps(candidates, indent=2))

    with cand_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "candidate_id",
                "t_start_min",
                "t_end_max",
                "t_start_median",
                "t_end_median",
                "length_median",
                "support_count",
                "unique_seed_count",
                "seed_list",
                "threshold_list",
                "per_model_hits_list",
                "min_alert_len_list",
            ],
        )
        writer.writeheader()
        for c in candidates:
            writer.writerow(
                {
                    "candidate_id": c["candidate_id"],
                    "t_start_min": c["t_start_min"],
                    "t_end_max": c["t_end_max"],
                    "t_start_median": c["t_start_median"],
                    "t_end_median": c["t_end_median"],
                    "length_median": c["length_median"],
                    "support_count": c["support_count"],
                    "unique_seed_count": c["unique_seed_count"],
                    "seed_list": json.dumps(c["seed_list"]),
                    "threshold_list": json.dumps(c["threshold_list"]),
                    "per_model_hits_list": json.dumps(c["per_model_hits_list"]),
                    "min_alert_len_list": json.dumps(c["min_alert_len_list"]),
                }
            )

    print(f"\nWrote summary: {csv_path}")
    print(f"Wrote candidate windows: {cand_csv}")
    print(f"Wrote candidate windows: {cand_json}")


if __name__ == "__main__":
    main()
