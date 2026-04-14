# CLAUDE.md — DGM Bachelor Thesis

## Project

Darwin Gödel Machine (DGM) fork — bachelor thesis comparing evaluation schedulers
(baseline, hyperband, asha, ga) on `swe_verified_mini`.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# API keys required
export OPENAI_API_KEY='...'
export ANTHROPIC_API_KEY='...'
# Docker must be running
docker run hello-world
```

SWE-bench submodule: `swe_bench/SWE-bench/` (pinned to `dc4c087c2b9e4cefebf2e3d201d27e36`).

Bootstrap initial archive before any run:
```bash
python test_swebench.py --benchmark swe_verified_mini --full_mini \
  --agent_dir initial_swe_verified_mini --write_agent_metadata
```

## Key Files

| File | Purpose |
|------|---------|
| `DGM_outer.py` | Main entry point — evolution loop |
| `schedulers.py` | BaselineScheduler, HyperbandScheduler, ASHAScheduler, GAScheduler |
| `self_improve_step.py` | Child generation & evaluation logic |
| `coding_agent.py` | Initial coding agent implementation |
| `benchmarks/config.py` | Benchmark registry and subset loading |
| `experiments/run_preliminary.py` | Thesis experiment runner (phases 1–3) |
| `compare_scheduler_runs.py` | Aggregate run stats into CSV/JSON |

## Running

```bash
# Default run (baseline scheduler, swe_verified_mini)
python DGM_outer.py

# Specific scheduler
python DGM_outer.py --scheduler hyperband --hyperband_eta 5 \
  --hyperband_budgets 2,10,50 --hyperband_initial_children 10

# Continue interrupted run
python DGM_outer.py --continue_from output_dgm/<run_id>

# Single task smoke test
python test_swebench.py --benchmark swe_verified_mini \
  --single_task django__django-11790 --agent_dir initial_swe_verified_mini
```

Output lands in `output_dgm/<timestamp>/`.

## Experiments (Thesis)

```bash
# Phase 1: baseline vs hyperband (3 seeds, 8 gens each)
python experiments/run_preliminary.py --phase 1
python experiments/run_preliminary.py --phase 1 --analyze

# Phase 2: HP sweep (eta × budgets, 9 combos + ablations, 5 gens)
python experiments/run_preliminary.py --phase 2
python experiments/run_preliminary.py --phase 2 --analyze

# Phase 3: validation with best HP from phase 2 (3 seeds, 8 gens)
python experiments/run_preliminary.py --phase 3 \
  --best_eta 5 --best_budgets 2,10,50 --best_n 10
```

State persists in `experiments/state/phase{N}_state.json` — interrupted runs resume automatically.
Results go to `experiments/results/`.

## Budget Constraint

Keep cost ≤ $90 per full experiment. Reference task costs per generation:
- Baseline (2 children, 50 tasks): ~100 tasks/gen
- Hyperband (10 init, eta=5, budgets=[2,10,50]): ~76 tasks/gen

Use `--generation_task_budget_total 100` to cap task spend per generation.

## Known Quirks

- **Corrupt JSONL lines**: `load_dgm_metadata` skips malformed lines silently — normal.
- **MiniMax model**: wraps tool calls in JSON tags; `self_improve_step.py` handles this.
- **Empty predictions**: `swe_bench/report.py` skips instances with no patch — normal.
- **Context length errors**: repeated `"Error code: 400 - {'message': 'Input is too long'}"` in logs means the agent hit context limit; DGM detects this and may target `solve_contextlength` improvements.

## Tests

```bash
pytest tests/
```

Tests cover bash tool, edit tool, and benchmark registry. They do NOT run Docker.

## Architecture Notes

- Each DGM generation: choose parents → generate children (parallel) → evaluate children (staged) → update archive.
- `BaselineScheduler`: 3-stage eval (small → medium → full), promote on score threshold.
- `HyperbandScheduler`: synchronous SHA — all candidates finish rung before promotion.
- `ASHAScheduler`: async SHA — promote as soon as candidate ranks in top-K of rung completions.
- `GAScheduler`: blind high-temperature mutation (no error-log context), same 3-stage eval as baseline.
- Archive stores all compiled children by default (`--update_archive keep_all`).
- `dgm_metadata.jsonl` in each run dir is append-only; each line = one generation.
