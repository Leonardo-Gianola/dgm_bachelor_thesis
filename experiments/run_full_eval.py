"""
Full evaluation runner for DGM scheduler comparison (thesis).

Compares all four schedulers — Baseline, Hyperband, ASHA, GA — on
swe_verified_mini using minimax/minimax-m2.5 via OpenRouter.

Fair comparison: all methods use --generation_task_budget_total 100.

Usage:
  python experiments/run_full_eval.py [--dry_run] [--force_rerun] [--analyze]

  --dry_run      Print commands without executing.
  --force_rerun  Ignore saved state, rerun everything.
  --analyze      Run compare_scheduler_runs.py on completed runs and exit.

State is persisted in experiments/state/full_eval_state.json so
interrupted runs resume automatically.

Estimated cost (minimax-m2.5 @ $0.118/1M in, $0.99/1M out):
  ~100 tasks/gen * 10 gens * 4 methods * 2 seeds = ~8000 task-evals
  Rough total: $40-80 depending on context length per task.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DGM = REPO_ROOT / "output_dgm"
STATE_DIR = Path(__file__).parent / "state"
RESULTS_DIR = Path(__file__).parent / "results"
PYTHON = sys.executable

# ---------------------------------------------------------------------------
# Experiment configurations — 4 methods x 2 seeds x 10 generations
# All use generation_task_budget_total=100 for fair comparison.
# ---------------------------------------------------------------------------

FULL_EVAL_CONFIGS = (
    # ---- Baseline ----
    [
        {
            "label": f"full_baseline_s{s}",
            "est_tasks_per_gen": 100,
            "num_gens": 10,
            "args": [
                "--scheduler", "baseline",
                "--max_generation", "10",
                "--selfimprove_size", "2",
                "--generation_task_budget_total", "100",
                "--seed", str(s),
                "--benchmark", "swe_verified_mini",
            ],
        }
        for s in [0, 1]
    ]
    # ---- Hyperband (eta=5, budgets=[2,10,50], n_initial=10 -> 76 tasks/gen) ----
    + [
        {
            "label": f"full_hyperband_s{s}",
            "est_tasks_per_gen": 76,
            "num_gens": 10,
            "args": [
                "--scheduler", "hyperband",
                "--max_generation", "10",
                "--selfimprove_size", "2",
                "--hyperband_eta", "5",
                "--hyperband_budgets", "2,10,50",
                "--hyperband_initial_children", "10",
                "--generation_task_budget_total", "100",
                "--seed", str(s),
                "--benchmark", "swe_verified_mini",
            ],
        }
        for s in [0, 1]
    ]
    # ---- ASHA (same budget params as Hyperband for direct comparison) ----
    + [
        {
            "label": f"full_asha_s{s}",
            "est_tasks_per_gen": 76,
            "num_gens": 10,
            "args": [
                "--scheduler", "asha",
                "--max_generation", "10",
                "--selfimprove_size", "2",
                "--hyperband_eta", "5",
                "--hyperband_budgets", "2,10,50",
                "--hyperband_initial_children", "10",
                "--generation_task_budget_total", "100",
                "--seed", str(s),
                "--benchmark", "swe_verified_mini",
            ],
        }
        for s in [0, 1]
    ]
    # ---- GA (blind mutation + tournament selection) ----
    + [
        {
            "label": f"full_ga_s{s}",
            "est_tasks_per_gen": 100,
            "num_gens": 10,
            "args": [
                "--scheduler", "ga",
                "--max_generation", "10",
                "--selfimprove_size", "2",
                "--choose_selfimproves_method", "tournament",
                "--ga_mutation_temperature", "1.0",
                "--generation_task_budget_total", "100",
                "--seed", str(s),
                "--benchmark", "swe_verified_mini",
            ],
        }
        for s in [0, 1]
    ]
)


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def load_state() -> dict:
    path = STATE_DIR / "full_eval_state.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = STATE_DIR / "full_eval_state.json"
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Output dir detection
# ---------------------------------------------------------------------------

def _snapshot_output_dirs() -> set:
    if not OUTPUT_DGM.exists():
        return set()
    return {p for p in OUTPUT_DGM.iterdir() if p.is_dir()}


def detect_new_run_dir(before: set) -> Path | None:
    after = {p for p in OUTPUT_DGM.iterdir() if p.is_dir()} if OUTPUT_DGM.exists() else set()
    new = after - before
    if not new:
        return None
    if len(new) == 1:
        return next(iter(new))
    return max(new, key=lambda d: d.stat().st_mtime)


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def estimate_cost(config: dict) -> int:
    return config["est_tasks_per_gen"] * config["num_gens"]


def run_config(config: dict, state: dict, dry_run: bool = False) -> str:
    label = config["label"]

    if label in state and state[label].get("status") == "completed":
        print(f"  [SKIP] {label} (already completed -> {state[label]['output_dir']})")
        return state[label]["output_dir"]

    est = estimate_cost(config)
    cmd_str = " ".join(config["args"])
    print(f"\n  [RUN ] {label}")
    print(f"         est. {est} tasks | args: {cmd_str}")

    if dry_run:
        print("         [DRY RUN — not executing]")
        return "(dry_run)"

    before = _snapshot_output_dirs()
    cmd = [PYTHON, str(REPO_ROOT / "DGM_outer.py")] + config["args"]

    state[label] = {"status": "running", "output_dir": None, "start_time": time.time()}
    save_state(state)

    t0 = time.time()
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    elapsed = time.time() - t0

    run_dir = detect_new_run_dir(before)
    run_dir_str = str(run_dir) if run_dir else "(unknown)"

    status = "completed" if result.returncode == 0 else f"failed_rc{result.returncode}"
    state[label] = {
        "status": status,
        "output_dir": run_dir_str,
        "start_time": t0,
        "end_time": time.time(),
        "wall_clock_seconds": elapsed,
    }
    save_state(state)
    print(f"  [{status.upper()}] {label} -> {run_dir_str} ({elapsed:.0f}s)")
    return run_dir_str


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def run_analysis(state: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_dirs = [
        v["output_dir"]
        for v in state.values()
        if v.get("status") == "completed"
        and v.get("output_dir") not in (None, "(unknown)", "(dry_run)")
    ]
    if not run_dirs:
        print("No completed runs to analyze.")
        return

    output_csv = str(RESULTS_DIR / "full_eval_comparison.csv")
    cmd = [
        PYTHON, str(REPO_ROOT / "compare_scheduler_runs.py"),
        "--run_dirs", *run_dirs,
        "--output_csv", output_csv,
    ]
    print(f"\nRunning analysis -> {output_csv}")
    subprocess.run(cmd, cwd=REPO_ROOT)
    print("Done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Full DGM scheduler evaluation runner.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing.")
    parser.add_argument("--force_rerun", action="store_true", help="Ignore saved state, rerun all.")
    parser.add_argument("--analyze", action="store_true", help="Run analysis on completed runs and exit.")
    args = parser.parse_args()

    state = {} if args.force_rerun else load_state()

    if args.analyze:
        run_analysis(state)
        return

    configs = FULL_EVAL_CONFIGS
    total_est = sum(estimate_cost(c) for c in configs)
    done = sum(1 for c in configs if state.get(c["label"], {}).get("status") == "completed")

    print(f"\n{'='*60}")
    print(f"FULL EVAL: {len(configs)} runs | ~{total_est} estimated tasks total")
    print(f"Methods: Baseline, Hyperband, ASHA, GA | Seeds: 0, 1 | Gens: 10")
    print(f"Already completed: {done}/{len(configs)}")
    print(f"{'='*60}")

    for config in configs:
        run_config(config, state, dry_run=args.dry_run)

    completed = [
        state[c["label"]]["output_dir"]
        for c in configs
        if state.get(c["label"], {}).get("status") == "completed"
    ]
    print(f"\nAll runs done. {len(completed)}/{len(configs)} completed.")

    if not args.dry_run and completed:
        run_analysis(state)


if __name__ == "__main__":
    main()
