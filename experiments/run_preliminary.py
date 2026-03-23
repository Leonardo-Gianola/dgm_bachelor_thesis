"""
Preliminary experiment runner for DGM scheduler comparison (thesis).

Usage:
  python experiments/run_preliminary.py --phase 1 [--dry_run] [--force_rerun]
  python experiments/run_preliminary.py --phase 1 --analyze
  python experiments/run_preliminary.py --phase 2 [--dry_run]
  python experiments/run_preliminary.py --phase 2 --analyze
  python experiments/run_preliminary.py --phase 3 --best_eta 5 --best_budgets 2,10,50 --best_n 10
  python experiments/run_preliminary.py --phase 3 --analyze

State is persisted in experiments/state/phase{N}_state.json so interrupted runs resume
automatically. Completed runs are skipped unless --force_rerun is passed.

Fair comparison design:
  Both schedulers use --generation_task_budget_total 100, giving ~100 tasks/generation.
  Baseline (2 children): up to 2×50 = 100 tasks/gen (worst case).
  Hyperband (10 initial, eta=5, budgets=[2,10,50]): 10×2 + 2×8 + 1×40 = 76 tasks/gen.
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
# Experiment configurations
# ---------------------------------------------------------------------------

PHASE1_CONFIGS = (
    [
        {
            "label": f"p1_baseline_s{s}",
            "est_tasks_per_gen": 100,
            "num_gens": 8,
            "args": [
                "--scheduler", "baseline",
                "--max_generation", "8",
                "--selfimprove_size", "2",
                "--generation_task_budget_total", "100",
                "--seed", str(s),
                "--benchmark", "swe_verified_mini",
            ],
        }
        for s in [0, 1, 2]
    ]
    + [
        {
            "label": f"p1_hb_default_s{s}",
            "est_tasks_per_gen": 76,
            "num_gens": 8,
            "args": [
                "--scheduler", "hyperband",
                "--max_generation", "8",
                "--selfimprove_size", "2",
                "--hyperband_eta", "5",
                "--hyperband_budgets", "2,10,50",
                "--hyperband_initial_children", "10",
                "--generation_task_budget_total", "100",
                "--seed", str(s),
                "--benchmark", "swe_verified_mini",
            ],
        }
        for s in [0, 1, 2]
    ]
)

# Phase 2: HP sweep — eta × budgets (9 combos) + initial_children ablation (3 combos)
# + 2 baseline reference runs. All 1 seed, 5 generations.
# n_initial chosen so _budget_cost(n, budgets, eta) <= 100.
# Verified costs (tasks/gen):
#   eta=3, b=[2,10,50], n=9  -> 82   eta=5, b=[2,10,50], n=10 -> 76
#   eta=10,b=[2,10,50], n=15 -> 86   eta=3, b=[5,20,50], n=6  -> 90
#   eta=5, b=[5,20,50], n=8  -> 100  eta=10,b=[5,20,50], n=10 -> 95
#   eta=3, b=[1,5,50],  n=9  -> 66   eta=5, b=[1,5,50],  n=25 -> 90
#   eta=10,b=[1,5,50],  n=39 -> 100  n=4 -> 56, n=8 -> 72, n=16 -> 104
PHASE2_CONFIGS = [
    # --- eta x budgets sweep (eta=3) ---
    {
        "label": "p2_hb_eta3_b210_s0",
        "est_tasks_per_gen": 82,
        "num_gens": 5,
        "args": [
            "--scheduler", "hyperband", "--max_generation", "5",
            "--selfimprove_size", "2", "--hyperband_eta", "3",
            "--hyperband_budgets", "2,10,50", "--hyperband_initial_children", "9",
            "--generation_task_budget_total", "100", "--seed", "0",
            "--benchmark", "swe_verified_mini",
        ],
    },
    {
        "label": "p2_hb_eta3_b520_s0",
        "est_tasks_per_gen": 90,
        "num_gens": 5,
        "args": [
            "--scheduler", "hyperband", "--max_generation", "5",
            "--selfimprove_size", "2", "--hyperband_eta", "3",
            "--hyperband_budgets", "5,20,50", "--hyperband_initial_children", "6",
            "--generation_task_budget_total", "100", "--seed", "0",
            "--benchmark", "swe_verified_mini",
        ],
    },
    {
        "label": "p2_hb_eta3_b150_s0",
        "est_tasks_per_gen": 66,
        "num_gens": 5,
        "args": [
            "--scheduler", "hyperband", "--max_generation", "5",
            "--selfimprove_size", "2", "--hyperband_eta", "3",
            "--hyperband_budgets", "1,5,50", "--hyperband_initial_children", "9",
            "--generation_task_budget_total", "100", "--seed", "0",
            "--benchmark", "swe_verified_mini",
        ],
    },
    # --- eta x budgets sweep (eta=5) ---
    {
        "label": "p2_hb_eta5_b210_s0",
        "est_tasks_per_gen": 76,
        "num_gens": 5,
        "args": [
            "--scheduler", "hyperband", "--max_generation", "5",
            "--selfimprove_size", "2", "--hyperband_eta", "5",
            "--hyperband_budgets", "2,10,50", "--hyperband_initial_children", "10",
            "--generation_task_budget_total", "100", "--seed", "0",
            "--benchmark", "swe_verified_mini",
        ],
    },
    {
        "label": "p2_hb_eta5_b520_s0",
        "est_tasks_per_gen": 100,
        "num_gens": 5,
        "args": [
            "--scheduler", "hyperband", "--max_generation", "5",
            "--selfimprove_size", "2", "--hyperband_eta", "5",
            "--hyperband_budgets", "5,20,50", "--hyperband_initial_children", "8",
            "--generation_task_budget_total", "100", "--seed", "0",
            "--benchmark", "swe_verified_mini",
        ],
    },
    {
        "label": "p2_hb_eta5_b150_s0",
        "est_tasks_per_gen": 90,
        "num_gens": 5,
        "args": [
            "--scheduler", "hyperband", "--max_generation", "5",
            "--selfimprove_size", "2", "--hyperband_eta", "5",
            "--hyperband_budgets", "1,5,50", "--hyperband_initial_children", "25",
            "--generation_task_budget_total", "100", "--seed", "0",
            "--benchmark", "swe_verified_mini",
        ],
    },
    # --- eta x budgets sweep (eta=10) ---
    {
        "label": "p2_hb_eta10_b210_s0",
        "est_tasks_per_gen": 86,
        "num_gens": 5,
        "args": [
            "--scheduler", "hyperband", "--max_generation", "5",
            "--selfimprove_size", "2", "--hyperband_eta", "10",
            "--hyperband_budgets", "2,10,50", "--hyperband_initial_children", "15",
            "--generation_task_budget_total", "100", "--seed", "0",
            "--benchmark", "swe_verified_mini",
        ],
    },
    {
        "label": "p2_hb_eta10_b520_s0",
        "est_tasks_per_gen": 95,
        "num_gens": 5,
        "args": [
            "--scheduler", "hyperband", "--max_generation", "5",
            "--selfimprove_size", "2", "--hyperband_eta", "10",
            "--hyperband_budgets", "5,20,50", "--hyperband_initial_children", "10",
            "--generation_task_budget_total", "100", "--seed", "0",
            "--benchmark", "swe_verified_mini",
        ],
    },
    {
        "label": "p2_hb_eta10_b150_s0",
        "est_tasks_per_gen": 100,
        "num_gens": 5,
        "args": [
            "--scheduler", "hyperband", "--max_generation", "5",
            "--selfimprove_size", "2", "--hyperband_eta", "10",
            "--hyperband_budgets", "1,5,50", "--hyperband_initial_children", "39",
            "--generation_task_budget_total", "100", "--seed", "0",
            "--benchmark", "swe_verified_mini",
        ],
    },
    # --- initial_children ablation (eta=5, budgets=[2,10,50]) ---
    {
        "label": "p2_hb_n4_eta5_b210_s0",
        "est_tasks_per_gen": 56,
        "num_gens": 5,
        "args": [
            "--scheduler", "hyperband", "--max_generation", "5",
            "--selfimprove_size", "2", "--hyperband_eta", "5",
            "--hyperband_budgets", "2,10,50", "--hyperband_initial_children", "4",
            "--generation_task_budget_total", "100", "--seed", "0",
            "--benchmark", "swe_verified_mini",
        ],
    },
    {
        "label": "p2_hb_n8_eta5_b210_s0",
        "est_tasks_per_gen": 72,
        "num_gens": 5,
        "args": [
            "--scheduler", "hyperband", "--max_generation", "5",
            "--selfimprove_size", "2", "--hyperband_eta", "5",
            "--hyperband_budgets", "2,10,50", "--hyperband_initial_children", "8",
            "--generation_task_budget_total", "100", "--seed", "0",
            "--benchmark", "swe_verified_mini",
        ],
    },
    {
        "label": "p2_hb_n16_eta5_b210_s0",
        "est_tasks_per_gen": 104,  # slightly over 100 — acceptable
        "num_gens": 5,
        "args": [
            "--scheduler", "hyperband", "--max_generation", "5",
            "--selfimprove_size", "2", "--hyperband_eta", "5",
            "--hyperband_budgets", "2,10,50", "--hyperband_initial_children", "16",
            "--generation_task_budget_total", "100", "--seed", "0",
            "--benchmark", "swe_verified_mini",
        ],
    },
    # --- Baseline reference runs ---
    {
        "label": "p2_baseline_sz2_s0",
        "est_tasks_per_gen": 100,
        "num_gens": 5,
        "args": [
            "--scheduler", "baseline", "--max_generation", "5",
            "--selfimprove_size", "2",
            "--generation_task_budget_total", "100", "--seed", "0",
            "--benchmark", "swe_verified_mini",
        ],
    },
    {
        "label": "p2_baseline_sz4_s0",
        "est_tasks_per_gen": 200,
        "num_gens": 5,
        "args": [
            "--scheduler", "baseline", "--max_generation", "5",
            "--selfimprove_size", "4",
            "--generation_task_budget_total", "200", "--seed", "0",
            "--benchmark", "swe_verified_mini",
        ],
    },
]

# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


def load_state(phase: int) -> dict:
    path = STATE_DIR / f"phase{phase}_state.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_state(phase: int, state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = STATE_DIR / f"phase{phase}_state.json"
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
    # Multiple new dirs (unexpected) — return newest by mtime
    return max(new, key=lambda d: d.stat().st_mtime)


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


def estimate_cost(config: dict) -> int:
    return config["est_tasks_per_gen"] * config["num_gens"]


def run_config(config: dict, phase: int, state: dict, dry_run: bool = False) -> str:
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
    save_state(phase, state)

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
    save_state(phase, state)
    print(f"  [{status.upper()}] {label} -> {run_dir_str} ({elapsed:.0f}s)")
    return run_dir_str


def run_phase(phase: int, configs: list, dry_run: bool = False, force_rerun: bool = False) -> dict:
    state = load_state(phase)
    if force_rerun:
        state = {}

    total_est = sum(estimate_cost(c) for c in configs)
    done = sum(1 for c in configs if c["label"] in state and state[c["label"]].get("status") == "completed")
    print(f"\n{'='*60}")
    print(f"PHASE {phase}: {len(configs)} runs | ~{total_est} estimated tasks total")
    print(f"Already completed: {done}/{len(configs)}")
    print(f"{'='*60}")

    for config in configs:
        run_config(config, phase, state, dry_run=dry_run)

    completed = [
        state[c["label"]]["output_dir"]
        for c in configs
        if state.get(c["label"], {}).get("status") == "completed"
    ]
    print(f"\nPhase {phase} done. {len(completed)}/{len(configs)} runs completed.")
    return state


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def run_analysis(phase: int, output_csv: str) -> None:
    state = load_state(phase)
    run_dirs = [
        v["output_dir"]
        for v in state.values()
        if v.get("status") == "completed"
        and v.get("output_dir") not in (None, "(unknown)", "(dry_run)")
    ]
    if not run_dirs:
        print(f"No completed runs found for phase {phase}. Nothing to analyze.")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_json = output_csv.replace(".csv", ".json")
    cmd = [
        PYTHON, str(REPO_ROOT / "compare_scheduler_runs.py"),
        "--run_dirs", *run_dirs,
        "--output_csv", output_csv,
        "--output_json", output_json,
    ]
    print(f"\nAnalyzing {len(run_dirs)} runs...")
    print(f"  {' '.join(cmd)}")
    subprocess.run(cmd, cwd=REPO_ROOT)
    print(f"\nResults written to:\n  {output_csv}\n  {output_json}")


# ---------------------------------------------------------------------------
# Phase 3 config builder (called after Phase 2 analysis)
# ---------------------------------------------------------------------------


def build_phase3_configs(
    best_eta: int,
    best_budgets: str,
    best_n: int,
    budget_total: int = 100,
    seeds: list | None = None,
) -> list:
    seeds = seeds or [0, 1, 2]
    configs = []
    for s in seeds:
        configs.append({
            "label": f"p3_hb_best_s{s}",
            "est_tasks_per_gen": budget_total,
            "num_gens": 8,
            "args": [
                "--scheduler", "hyperband",
                "--max_generation", "8",
                "--selfimprove_size", "2",
                "--hyperband_eta", str(best_eta),
                "--hyperband_budgets", best_budgets,
                "--hyperband_initial_children", str(best_n),
                "--generation_task_budget_total", str(budget_total),
                "--seed", str(s),
                "--benchmark", "swe_verified_mini",
            ],
        })
    for s in seeds:
        configs.append({
            "label": f"p3_baseline_s{s}",
            "est_tasks_per_gen": 100,
            "num_gens": 8,
            "args": [
                "--scheduler", "baseline",
                "--max_generation", "8",
                "--selfimprove_size", "2",
                "--generation_task_budget_total", "100",
                "--seed", str(s),
                "--benchmark", "swe_verified_mini",
            ],
        })
    return configs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run preliminary DGM experiments comparing baseline vs Hyperband.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--analyze", action="store_true",
                        help="Run compare_scheduler_runs.py on completed runs after (or instead of) running.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing them.")
    parser.add_argument("--force_rerun", action="store_true",
                        help="Ignore existing state and rerun all configs.")
    parser.add_argument("--run_only", action="store_true",
                        help="Run experiments without analyzing (default when --analyze is absent).")

    # Phase 3 requires the best config from Phase 2
    parser.add_argument("--best_eta", type=int, default=5,
                        help="Best eta from Phase 2 sweep (for Phase 3).")
    parser.add_argument("--best_budgets", type=str, default="2,10,50",
                        help="Best budgets string from Phase 2 sweep (for Phase 3).")
    parser.add_argument("--best_n", type=int, default=10,
                        help="Best initial_children from Phase 2 sweep (for Phase 3).")
    parser.add_argument("--best_budget_total", type=int, default=100,
                        help="generation_task_budget_total for Phase 3 Hyperband run.")

    args = parser.parse_args()

    phase_csv = {
        1: str(RESULTS_DIR / "phase1_comparison.csv"),
        2: str(RESULTS_DIR / "phase2_sweep.csv"),
        3: str(RESULTS_DIR / "phase3_validation.csv"),
    }

    if args.phase == 1:
        run_phase(1, PHASE1_CONFIGS, dry_run=args.dry_run, force_rerun=args.force_rerun)
        if args.analyze:
            run_analysis(1, phase_csv[1])

    elif args.phase == 2:
        run_phase(2, PHASE2_CONFIGS, dry_run=args.dry_run, force_rerun=args.force_rerun)
        if args.analyze:
            run_analysis(2, phase_csv[2])

    elif args.phase == 3:
        configs = build_phase3_configs(
            best_eta=args.best_eta,
            best_budgets=args.best_budgets,
            best_n=args.best_n,
            budget_total=args.best_budget_total,
        )
        run_phase(3, configs, dry_run=args.dry_run, force_rerun=args.force_rerun)
        if args.analyze:
            run_analysis(3, phase_csv[3])

    if args.analyze and not args.dry_run:
        print(f"\nTo analyze later without rerunning:")
        print(f"  python experiments/run_preliminary.py --phase {args.phase} --analyze --dry_run")


if __name__ == "__main__":
    main()
