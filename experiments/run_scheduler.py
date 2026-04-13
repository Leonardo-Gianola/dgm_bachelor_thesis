"""
Single-scheduler evaluation runner with live per-generation metrics.

Runs one scheduler across 1 seed (5 generations), printing a metrics table
after every generation. Budget: 50 tasks/gen → ~$80 for all 4 schedulers.

Per-generation task usage:
  Baseline/GA:      2 children × (small=3 + medium=12) = 30 tasks  (no full stage)
  Hyperband/ASHA:   1 child  × (3 + 12 + 35)          = 50 tasks  (full eval)

The budget asymmetry is intentional: each scheduler uses its optimal strategy
within the 50-task budget. This IS the research question being compared.

Run order for risk management (most important first):
  1. baseline   — control group
  2. hyperband  — most theoretically interesting
  3. ga         — different selection mechanism
  4. asha       — similar to hyperband, compare last

Usage:
  python experiments/run_scheduler.py --scheduler baseline  [--dry_run]
  python experiments/run_scheduler.py --scheduler hyperband [--dry_run]
  python experiments/run_scheduler.py --scheduler asha      [--dry_run]
  python experiments/run_scheduler.py --scheduler ga        [--dry_run]

  --seeds N      Override seeds (default: 0)
  --gens N       Override generations per seed (default: 5)
  --dry_run      Print commands without executing
  --force_rerun  Ignore saved state, rerun

State: experiments/state/run_{scheduler}_state.json
       Interrupted runs resume automatically.
"""

import argparse
import json
import os
import subprocess
import sys
import time
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DGM = REPO_ROOT / "output_dgm"
STATE_DIR = Path(__file__).parent / "state"
RESULTS_DIR = Path(__file__).parent / "results"
PYTHON = sys.executable

# Budget cap per generation. Baseline/GA: 2 children get small+medium only (30 tasks).
# Hyperband/ASHA: auto-selects n=1 child through all 3 stages (50 tasks exactly).
# Total: 4 schedulers × 1 seed × 5 gens × avg(40 tasks) ≈ 800 tasks ≈ $80.
GENERATION_TASK_BUDGET = 50

SCHEDULER_ARGS = {
    "baseline": [
        "--scheduler", "baseline",
        "--selfimprove_size", "2",
        "--generation_task_budget_total", str(GENERATION_TASK_BUDGET),
    ],
    "hyperband": [
        "--scheduler", "hyperband",
        "--selfimprove_size", "2",
        "--hyperband_eta", "5",
        "--hyperband_budgets", "3,15,50",
        "--generation_task_budget_total", str(GENERATION_TASK_BUDGET),
    ],
    "asha": [
        "--scheduler", "asha",
        "--selfimprove_size", "2",
        "--hyperband_eta", "5",
        "--hyperband_budgets", "3,15,50",
        "--generation_task_budget_total", str(GENERATION_TASK_BUDGET),
    ],
    "ga": [
        "--scheduler", "ga",
        "--selfimprove_size", "2",
        "--choose_selfimproves_method", "tournament",
        "--ga_mutation_temperature", "1.0",
        "--generation_task_budget_total", str(GENERATION_TASK_BUDGET),
    ],
}

EST_TASKS_PER_GEN = {
    "baseline": 30,
    "hyperband": 50,
    "asha": 50,
    "ga": 30,
}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def state_path(scheduler: str) -> Path:
    return STATE_DIR / f"run_{scheduler}_state.json"


def load_state(scheduler: str) -> dict:
    p = state_path(scheduler)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_state(scheduler: str, state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state_path(scheduler).write_text(json.dumps(state, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Per-generation metrics monitor
# ---------------------------------------------------------------------------

def _fmt_score(v) -> str:
    if v is None:
        return "  —   "
    return f"{v:.3f}"


def _fmt_time(seconds) -> str:
    if seconds is None:
        return "  —  "
    m, s = divmod(int(seconds), 60)
    return f"{m:3d}m{s:02d}s"


def _print_header():
    print(
        f"\n{'Gen':>4}  {'Children':>10}  {'Compiled':>9}  "
        f"{'Best':>6}  {'Avg':>6}  {'Tasks':>6}  {'Time':>8}  {'Archive':>8}"
    )
    print("-" * 72)


def _print_gen_row(gen_data: dict):
    gen        = gen_data.get("generation", "?")
    generated  = gen_data.get("children_generated_count", 0)
    compiled   = gen_data.get("children_compiled_count", 0)
    best       = gen_data.get("best_child_score")
    avg        = gen_data.get("avg_child_score")
    tasks      = gen_data.get("evaluation_budget_tasks_consumed", 0)
    wall       = gen_data.get("generation_wall_clock_seconds")
    archive    = gen_data.get("archive_size", "?")
    print(
        f"{gen:>4}  {generated:>4}/{compiled:<5}  {compiled:>4}/{generated:<4}  "
        f"{_fmt_score(best)}  {_fmt_score(avg)}  {tasks:>6}  {_fmt_time(wall)}  {archive:>8}"
    )


def _monitor_jsonl(jsonl_path: Path, stop_event: threading.Event, scheduler: str, seed: int):
    """Background thread: tail dgm_metadata.jsonl and print new generation rows."""
    seen_gens = set()
    header_printed = False

    while not stop_event.is_set():
        if jsonl_path.exists():
            try:
                lines = jsonl_path.read_text(encoding="utf-8").strip().split("\n")
                for line in lines:
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    gen = data.get("generation")
                    if gen is not None and gen not in seen_gens:
                        if not header_printed:
                            print(f"\n[{scheduler.upper()} seed={seed}] Generation metrics:")
                            _print_header()
                            header_printed = True
                        _print_gen_row(data)
                        seen_gens.add(gen)
            except Exception:
                pass
        stop_event.wait(timeout=15)  # poll every 15s


# ---------------------------------------------------------------------------
# Output dir detection
# ---------------------------------------------------------------------------

def _snapshot_dirs() -> set:
    return {p for p in OUTPUT_DGM.iterdir() if p.is_dir()} if OUTPUT_DGM.exists() else set()


def _wait_for_new_dir(before: set, timeout: float = 120.0) -> Path | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if OUTPUT_DGM.exists():
            new = {p for p in OUTPUT_DGM.iterdir() if p.is_dir()} - before
            if new:
                return max(new, key=lambda d: d.stat().st_mtime)
        time.sleep(2)
    return None


# ---------------------------------------------------------------------------
# Run one seed
# ---------------------------------------------------------------------------

def run_seed(scheduler: str, seed: int, num_gens: int, state: dict, dry_run: bool) -> str:
    label = f"{scheduler}_s{seed}"

    if state.get(label, {}).get("status") == "completed":
        print(f"  [SKIP] {label} already completed -> {state[label]['output_dir']}")
        return state[label]["output_dir"]

    extra_args = SCHEDULER_ARGS[scheduler]
    cmd = [
        PYTHON, str(REPO_ROOT / "DGM_outer.py"),
        "--benchmark", "swe_verified_mini",
        "--max_generation", str(num_gens),
        "--seed", str(seed),
        *extra_args,
    ]
    est = EST_TASKS_PER_GEN[scheduler] * num_gens
    print(f"\n  [RUN ] {label}  |  est. {est} tasks  |  {' '.join(cmd[2:])}")

    if dry_run:
        print("         [DRY RUN — not executing]")
        return "(dry_run)"

    before = _snapshot_dirs()
    state[label] = {"status": "running", "output_dir": None, "start_time": time.time()}

    proc = subprocess.Popen(cmd, cwd=REPO_ROOT)

    # Detect output dir (created within first few seconds)
    run_dir = _wait_for_new_dir(before, timeout=120)
    jsonl_path = (run_dir / "dgm_metadata.jsonl") if run_dir else None

    # Start background monitor
    stop_event = threading.Event()
    if jsonl_path:
        monitor = threading.Thread(
            target=_monitor_jsonl,
            args=(jsonl_path, stop_event, scheduler, seed),
            daemon=True,
        )
        monitor.start()
    else:
        print(f"  [WARN] Could not detect output dir — no live metrics for {label}")
        monitor = None

    t0 = time.time()
    proc.wait()
    elapsed = time.time() - t0

    stop_event.set()
    if monitor:
        monitor.join(timeout=5)

    run_dir_str = str(run_dir) if run_dir else "(unknown)"
    status = "completed" if proc.returncode == 0 else f"failed_rc{proc.returncode}"

    state[label] = {
        "status": status,
        "output_dir": run_dir_str,
        "start_time": t0,
        "end_time": time.time(),
        "wall_clock_seconds": elapsed,
        "scheduler": scheduler,
        "seed": seed,
    }
    print(f"\n  [{status.upper()}] {label} -> {run_dir_str} ({elapsed/3600:.2f}h)")
    return run_dir_str


# ---------------------------------------------------------------------------
# Summary table after all seeds
# ---------------------------------------------------------------------------

def _print_summary(scheduler: str, state: dict, seeds: list):
    print(f"\n{'='*60}")
    print(f"SUMMARY — {scheduler.upper()}")
    print(f"{'='*60}")
    for seed in seeds:
        label = f"{scheduler}_s{seed}"
        info = state.get(label, {})
        status = info.get("status", "not_run")
        run_dir = info.get("output_dir", "—")
        wall = info.get("wall_clock_seconds")
        print(f"  seed={seed}  {status:<20}  {_fmt_time(wall)}  {run_dir}")

        # Print final gen metrics if available
        if run_dir and run_dir not in ("(unknown)", "(dry_run)"):
            jsonl = Path(run_dir) / "dgm_metadata.jsonl"
            if jsonl.exists():
                lines = [l for l in jsonl.read_text().strip().split("\n") if l.strip()]
                if lines:
                    last = json.loads(lines[-1])
                    print(
                        f"         Final gen {last.get('generation')}  "
                        f"best={_fmt_score(last.get('best_child_score'))}  "
                        f"archive={last.get('archive_size')}  "
                        f"total_tasks≈{sum(json.loads(l).get('evaluation_budget_tasks_consumed',0) for l in lines if l.strip())}"
                    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run one DGM scheduler with live metrics.")
    parser.add_argument(
        "--scheduler", required=True,
        choices=["baseline", "hyperband", "asha", "ga"],
        help="Scheduler to run.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0], metavar="N",
                        help="Seeds to run (default: 0).")
    parser.add_argument("--gens", type=int, default=5, help="Generations per seed (default: 5).")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force_rerun", action="store_true")
    args = parser.parse_args()

    scheduler = args.scheduler
    state = {} if args.force_rerun else load_state(scheduler)

    est_total = EST_TASKS_PER_GEN[scheduler] * args.gens * len(args.seeds)
    print(f"\n{'='*60}")
    print(f"SCHEDULER: {scheduler.upper()}  |  seeds={args.seeds}  |  gens={args.gens}")
    print(f"Est. tasks: ~{est_total}  |  budget: {GENERATION_TASK_BUDGET}/gen")
    print(f"{'='*60}")

    for seed in args.seeds:
        run_seed(scheduler, seed, args.gens, state, args.dry_run)
        save_state(scheduler, state)

    _print_summary(scheduler, state, args.seeds)


if __name__ == "__main__":
    main()
