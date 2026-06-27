"""
Multi-seed curriculum comparison experiment (baseline).

For one learning config this script:
  1. Runs the curated curriculum (E1-E6) once.
  2. Runs the E6-only baseline once.
  3. For each of N random configs (one per seed):
       a. unordered  — environments trained in raw generation order
       b. ordered    — environments sorted by complexity (easy → hard)
  4. Generates per-run cumulative-reward plots.
  5. Generates ONE summary figure:  4 lines (curriculum, random, random-ordered,
     baseline) where random / random-ordered show mean ± 1 std band over seeds.
  6. Saves evaluation results to disk (per-seed CSV + summary CSV).

Runs are resumable: if training_curve.npz + eval_results.json already exist in
a run directory the training step is skipped.

────────────────────────────────────────────────────────────────────────────────
Workflow
────────────────────────────────────────────────────────────────────────────────
# Step 1 – generate all 30 random configs (once, shared across learning configs)
python generate_random_config.py --all-seeds \\
    --target-config configs/8x8/environments-8x8-6.json \\
    --output-dir configs/8x8

# Step 2 – run comparison for each learning config
python run_comparison.py \\
    --curriculum-config configs/8x8/environments-8x8-6.json \\
    --random-configs    configs/8x8/environments-8x8-random-*.json \\
    --learn-config      configs/8x8/learning-8x8-6.json \\
    --output            results/comparison_8x8_normal

python run_comparison.py \\
    --curriculum-config configs/8x8/environments-8x8-6.json \\
    --random-configs    configs/8x8/environments-8x8-random-*.json \\
    --learn-config      configs/8x8/learning-8x8-6-shorter.json \\
    --output            results/comparison_8x8_shorter

python run_comparison.py \\
    --curriculum-config configs/8x8/environments-8x8-6.json \\
    --random-configs    configs/8x8/environments-8x8-random-*.json \\
    --learn-config      configs/8x8/learning-8x8-6-longer.json \\
    --output            results/comparison_8x8_longer

# Re-generate plots/CSVs without re-running training:
python run_comparison.py ... --plot-only
"""

import argparse
import copy
import csv
import glob as glob_module
import json
import os
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from config_loader import load_configs
from curriculum_builder import build_curriculum, get_target_spec
from trainer import run_curriculum, run_baseline, TrainResult, evaluate_agent, make_env

# ── visual style ──────────────────────────────────────────────────────────────
COLORS = {
    "curriculum":       "tab:blue",
    "random_unordered": "tab:orange",
    "random_ordered":   "tab:green",
    "baseline":         "black",
}
LABELS = {
    "curriculum":       "Curriculum (E1-E6)",
    "random_unordered": "Random",
    "random_ordered":   "Random-ordered",
    "baseline":         "Baseline (E6 only)",
}
N_GRID = 500   # interpolation grid points for summary plot


# ── small utilities ───────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def is_done(run_dir: str) -> bool:
    """True only when both output artefacts are present."""
    return (
        os.path.exists(os.path.join(run_dir, "training_curve.npz")) and
        os.path.exists(os.path.join(run_dir, "eval_results.json"))
    )


def expand_globs(patterns: list[str]) -> list[str]:
    """Expand any glob patterns; return sorted unique paths."""
    paths = []
    for p in patterns:
        expanded = sorted(glob_module.glob(p))
        paths.extend(expanded if expanded else [p])
    return sorted(set(paths))


# ── save / load helpers ───────────────────────────────────────────────────────

def save_run(
    results: list[TrainResult],
    agent,
    target_spec,
    run_dir: str,
    eval_episodes: int,
    eval_seed: int = 123,
):
    """Save training curve, individual plot, and eval results for one run."""
    os.makedirs(run_dir, exist_ok=True)

    # Concatenate timesteps and cumulative reward across all curriculum stages.
    ts = np.array([t   for r in results for t   in r.timesteps])
    cr = np.cumsum([rew for r in results for rew in r.episode_rewards])
    np.savez_compressed(
        os.path.join(run_dir, "training_curve.npz"),
        timesteps=ts, cumulative_rewards=cr,
    )

    # Individual per-run plot.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ts, cr, color="tab:blue", linewidth=1.5)
    ax.set_xlabel("Timesteps", fontsize=11)
    ax.set_ylabel("Cumulative Reward", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "cumulative_reward.png"), dpi=200)
    plt.close(fig)

    # Final evaluation on E6.
    eval_env = make_env(target_spec, seed=eval_seed)
    mr, sr = evaluate_agent(agent, eval_env, eval_episodes)
    with open(os.path.join(run_dir, "eval_results.json"), "w") as f:
        json.dump({"mean_reward": round(mr, 6), "success_rate": round(sr, 6)}, f, indent=2)

    print(f"    saved → {run_dir}  |  mean_reward={mr:.3f}  success_rate={sr:.3f}")


def load_curve(run_dir: str, grid: np.ndarray) -> np.ndarray:
    """Load a saved training curve and interpolate onto *grid*."""
    data = np.load(os.path.join(run_dir, "training_curve.npz"))
    ts   = np.concatenate([[0], data["timesteps"]])
    cr   = np.concatenate([[0], data["cumulative_rewards"]])
    return np.interp(grid, ts, cr)


def load_eval(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "eval_results.json")) as f:
        return json.load(f)


# ── training wrappers ─────────────────────────────────────────────────────────

def run_one(
    env_cfg_path: str,
    learn_cfg: dict,
    run_dir: str,
    ordered: bool,
    seed: int,
    eval_episodes: int,
) -> None:
    """Train a curriculum from *env_cfg_path* and save results to *run_dir*."""
    if is_done(run_dir):
        print(f"    [skip] {run_dir}")
        return

    lcfg = copy.deepcopy(learn_cfg)
    lcfg["curriculum"]["seed"] = seed
    set_seed(seed)

    with open(env_cfg_path) as f:
        env_cfg = json.load(f)

    curriculum  = build_curriculum(env_cfg, lcfg, ordered=ordered)
    target_spec = get_target_spec(curriculum)

    agent, results = run_curriculum(
        curriculum=curriculum,
        eval_spec=target_spec,
        learn_cfg=lcfg,
        output_dir=run_dir,
    )
    save_run(results, agent, target_spec, run_dir, eval_episodes)


def run_baseline_once(
    curriculum_cfg_path: str,
    learn_cfg: dict,
    run_dir: str,
    seed: int,
    eval_episodes: int,
) -> None:
    """Train E6-only baseline and save results to *run_dir*."""
    if is_done(run_dir):
        print(f"    [skip] {run_dir}")
        return

    lcfg = copy.deepcopy(learn_cfg)
    lcfg["curriculum"]["seed"] = seed
    set_seed(seed)

    with open(curriculum_cfg_path) as f:
        env_cfg = json.load(f)

    curriculum  = build_curriculum(env_cfg, lcfg, ordered=True)
    target_spec = get_target_spec(curriculum)
    total_steps = sum(item["steps"] for item in curriculum)

    agent, result = run_baseline(
        target_spec=target_spec,
        total_steps=total_steps,
        learn_cfg=lcfg,
        output_dir=run_dir,
    )
    save_run([result], agent, target_spec, run_dir, eval_episodes)


# ── summary plot ──────────────────────────────────────────────────────────────

def make_summary_plot(output_dir: str, total_steps: int) -> None:
    """
    One figure with 4 lines:
      • Curriculum (E1-E6)  — single line
      • Random              — mean ± 1 std band over all seeds
      • Random-ordered      — mean ± 1 std band over all seeds
      • Baseline (E6 only)  — single dashed line
    """
    grid = np.linspace(0, total_steps, N_GRID)
    fig, ax = plt.subplots(figsize=(9, 5))

    # -- single lines (curriculum + baseline) --
    for key, subdir in [("curriculum", "curriculum"), ("baseline", "baseline")]:
        d = os.path.join(output_dir, subdir)
        if is_done(d):
            cr = load_curve(d, grid)
            ax.plot(
                grid, cr,
                label=LABELS[key], color=COLORS[key], linewidth=2.0,
                linestyle="dashed" if key == "baseline" else "solid",
            )

    # -- mean ± std bands (random variants) --
    seed_dirs = sorted(glob_module.glob(os.path.join(output_dir, "seed_*")))
    for variant, key in [("unordered", "random_unordered"), ("ordered", "random_ordered")]:
        curves = [
            load_curve(os.path.join(sd, variant), grid)
            for sd in seed_dirs
            if is_done(os.path.join(sd, variant))
        ]
        if not curves:
            continue
        arr  = np.stack(curves)           # shape (n_seeds, N_GRID)
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        n    = len(curves)
        ax.plot(grid, mean,
                label=f"{LABELS[key]} (n={n})",
                color=COLORS[key], linewidth=2.0)
        ax.fill_between(grid, mean - std, mean + std,
                        alpha=0.2, color=COLORS[key])

    ax.set_xlabel("Timesteps", fontsize=12)
    ax.set_ylabel("Cumulative Reward", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, "summary_cumulative_reward.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  Saved summary plot → {path}")


# ── evaluation CSVs ───────────────────────────────────────────────────────────

def save_eval_csvs(output_dir: str) -> None:
    """
    Write two CSV files:
      evaluation_per_seed.csv  — one row per run
      evaluation_summary.csv   — mean ± std aggregated per condition
    """
    rows = []

    for condition, subdir in [("curriculum", "curriculum"), ("baseline", "baseline")]:
        d = os.path.join(output_dir, subdir)
        if is_done(d):
            ev = load_eval(d)
            rows.append({"condition": condition, "seed_idx": "N/A", **ev})

    seed_dirs = sorted(glob_module.glob(os.path.join(output_dir, "seed_*")))
    for i, sd in enumerate(seed_dirs):
        for variant, condition in [
            ("unordered", "random_unordered"),
            ("ordered",   "random_ordered"),
        ]:
            d = os.path.join(sd, variant)
            if is_done(d):
                ev = load_eval(d)
                rows.append({"condition": condition, "seed_idx": i + 1, **ev})

    if not rows:
        print("  No eval results found — skipping CSV.")
        return

    # per-seed CSV
    per_seed_path = os.path.join(output_dir, "evaluation_per_seed.csv")
    fieldnames = ["condition", "seed_idx", "mean_reward", "success_rate"]
    with open(per_seed_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved per-seed CSV  → {per_seed_path}")

    # summary statistics CSV
    summary_path = os.path.join(output_dir, "evaluation_summary.csv")
    summary_fields = [
        "condition", "n_runs",
        "mean_reward_mean", "mean_reward_std",
        "success_rate_mean", "success_rate_std",
    ]
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for condition in ["curriculum", "baseline", "random_unordered", "random_ordered"]:
            subset = [r for r in rows if r["condition"] == condition]
            if not subset:
                continue
            rews = [r["mean_reward"]  for r in subset]
            srs  = [r["success_rate"] for r in subset]
            w.writerow({
                "condition":         condition,
                "n_runs":            len(subset),
                "mean_reward_mean":  round(float(np.mean(rews)), 4),
                "mean_reward_std":   round(float(np.std(rews)),  4) if len(rews) > 1 else "N/A",
                "success_rate_mean": round(float(np.mean(srs)),  4),
                "success_rate_std":  round(float(np.std(srs)),   4) if len(srs)  > 1 else "N/A",
            })
    print(f"  Saved summary CSV   → {summary_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed curriculum comparison experiment."
    )
    parser.add_argument(
        "--curriculum-config",
        default="../01-configurations/environment-config.json",
        help="Path to the curated curriculum environment config.",
    )
    parser.add_argument(
        "--random-configs", nargs="+",
        default=["../01-configurations/random-configs/environments-8x8-random-*.json"],
        help="Random env config files — accepts shell globs.",
    )
    parser.add_argument(
        "--learn-config",
        default="../01-configurations/learning-config.json",
        help="Learning config JSON (controls steps/env, checkpointing, etc.). "
             "Use learning-config-shorter.json or learning-config-longer.json for other budgets.",
    )
    parser.add_argument(
        "--output", default="../02-train-data/comparison_normal",
        help="Root output directory.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Global training seed for curriculum and baseline runs (default: 42).",
    )
    parser.add_argument(
        "--plot-only", action="store_true",
        help="Skip all training; regenerate plots and CSVs from saved data.",
    )
    args = parser.parse_args()

    random_configs = expand_globs(args.random_configs)
    if not random_configs:
        parser.error("No files matched --random-configs patterns.")

    os.makedirs(args.output, exist_ok=True)
    _, learn_cfg = load_configs(args.curriculum_config, args.learn_config)
    eval_episodes = learn_cfg.get("evaluation", {}).get("eval_episodes", 3000)
    total_steps   = (
        learn_cfg["curriculum"]["size"] *
        learn_cfg["curriculum"]["steps_per_environment"]
    )

    if not args.plot_only:

        # ── 1. Curated curriculum (once) ──────────────────────────────────────
        print("=" * 70)
        print("Curriculum (E1-E6)")
        print("=" * 70)
        run_one(
            args.curriculum_config, learn_cfg,
            run_dir=os.path.join(args.output, "curriculum"),
            ordered=True, seed=args.seed, eval_episodes=eval_episodes,
        )

        # ── 2. E6-only baseline (once) ────────────────────────────────────────
        print("=" * 70)
        print("Baseline (E6 only)")
        print("=" * 70)
        run_baseline_once(
            args.curriculum_config, learn_cfg,
            run_dir=os.path.join(args.output, "baseline"),
            seed=args.seed, eval_episodes=eval_episodes,
        )

        # ── 3. Random configs — unordered + ordered per seed ──────────────────
        for i, rc in enumerate(random_configs):
            seed_label = f"seed_{i + 1:02d}"
            print("=" * 70)
            print(f"Random config {i + 1}/{len(random_configs)}: {Path(rc).name}")
            print("=" * 70)
            for ordered in [False, True]:
                variant = "ordered" if ordered else "unordered"
                print(f"  variant: {variant}")
                run_one(
                    rc, learn_cfg,
                    run_dir=os.path.join(args.output, seed_label, variant),
                    ordered=ordered, seed=args.seed, eval_episodes=eval_episodes,
                )

    # ── 4. Summary plot ───────────────────────────────────────────────────────
    print("=" * 70)
    print("Generating summary plot and evaluation CSVs")
    print("=" * 70)
    make_summary_plot(args.output, total_steps)
    save_eval_csvs(args.output)

    print(f"\nAll done. Results in: {args.output}")


if __name__ == "__main__":
    main()
