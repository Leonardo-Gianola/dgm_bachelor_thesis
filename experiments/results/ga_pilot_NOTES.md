# GA scheduler — 4-gen pilot run

**Run ID:** `20260513093044_666042`
**Scheduler:** ga (blind high-temperature mutation, no error-log context)
**Children per gen:** 2 (`--selfimprove_size 2`)
**Generations:** 4 (`--max_generation 4`)
**Tournament k:** 3 (`--ga_tournament_k 3`) — note: only used if `--choose_selfimproves_method tournament`; default here was `score_child_prop`
**Mutation temperature:** 1.0 (`--ga_mutation_temperature 1.0`)
**Per-gen task budget:** 100 (`--generation_task_budget_total 100`)
**Benchmark:** `swe_verified_mini` (50 SWE-bench Verified Mini tasks)
**Model:** `openrouter/minimax/minimax-m2.5`
**Start:** 2026-05-13 09:30 CEST
**End:** 2026-05-13 19:08 CEST
**Wall time:** 9h 38m (9.63h)
**Total cost:** **$26.00** (actual, OpenRouter dashboard)

## Initial baseline

| Metric | Value |
|--------|-------|
| Resolved | 16 / 47 |
| Empty patches | 11 |
| Accuracy | **34.04%** |

## Per-generation final-rung scores

| Gen | Wall | Children gen / comp / fully-eval | Promoted to archive | Parent | Entry | Final-rung r/s | Accuracy |
|-----|------|-----------------------------------|---------------------|--------|-------|----------------|----------|
| 0   | 0h29m | 2 / 1 / 0 | `20260513_093044_671020` (killed) | initial | sphinx-doc__sphinx-9281 | 1/1 | **100%** (single-task fluke) |
| 1   | **0s** | **0 / 0 / 0** | — (gen SKIPPED) | — | — | — | — |
| 2   | 4h21m | 2 / 1 / 0 | `20260513_095931_355150` | `..._671020` | solve_stochasticity | 20/29 | **68.97%** |
| 3   | 4h48m | 1 / 1 / 1 | `20260513_142007_309717` | `..._355150` | sphinx-doc__sphinx-8269 | **24/34** | **70.59%** 🏆 |

## Rung mechanics per generation

3 rungs per gen (budgets `[3, 15, 50]` per `_budget_cost` defaults for baseline-shape scheduler when GA uses baseline 3-stage):

| Gen | Rung 0 in/promoted/killed | Rung 1 in/promoted/killed | Rung 2 in/promoted/killed | Budget consumed |
|-----|---------------------------|---------------------------|---------------------------|-----------------|
| 0   | 2 / 0 / 2                 | 0 / 0 / 0                 | 0 / 0 / 0                 | 6/100           |
| 1   | 0 / 0 / 0                 | 0 / 0 / 0                 | 0 / 0 / 0                 | 0/100 (skipped) |
| 2   | 2 / 1 / 1                 | 1 / 1 / 0                 | 1 / 1 / 0                 | 53/100          |
| 3   | 1 / 1 / 0                 | 1 / 1 / 0                 | 1 / 1 / 0                 | 50/100          |

## Self-improve targets per gen

| Gen | Entries |
|-----|---------|
| 0   | `[('initial', 'sphinx-doc__sphinx-9281'), ('initial', 'sphinx-doc__sphinx-8269')]` |
| 1   | `[]` — **SKIPPED** (see anomaly below) |
| 2   | `[('20260513_093044_671020', 'solve_stochasticity'), ('20260513_093044_671020', 'solve_stochasticity')]` — same parent twice |
| 3   | `[('20260513_095931_355150', 'sphinx-doc__sphinx-8269')]` — only 1 entry (partial skip) |

## Generation wall times

| Gen | Started | Ended | Duration |
|-----|---------|-------|----------|
| 0   | 09:30   | 09:59 | 29m       |
| 1   | 09:59   | 09:59 | 0s        |
| 2   | 09:59   | 14:20 | 4h21m     |
| 3   | 14:20   | 19:08 | 4h48m     |

## Key observations

- **Initial → Best: +36.55pp absolute, +107% relative** (34.04% → 70.59%)
- **GA matched ~85% of HB's lift at ~⅓ HB's cost** ($26 vs $44.98). Best cost-efficiency among the three schedulers.
- **Compile rate 60%** (3/5 actual children compiled) — blind mutation without error-log context produces more broken patches. The `4dbdf02` argparse guard catches them at compile-time before wasting eval budget — good defense but signals fragility of blind mutation.
- **Effective gens: 2/4** — gen 1 fully skipped, gen 0 produced no archive-worthy child (both killed at rung 0). Only gens 2 and 3 contributed real progress.
- **`309717` (gen 3 winner) had 16 incomplete tasks at rung 2** — agent timeouts (30-min cap) for tasks like django__django-11815, sphinx-doc__sphinx-10673, etc. Excluded from denominator per `metric_flaw_empty_patches.md` convention.
- **Empty-patch escape persisted** — `671020` rate=1.0 inflated selection probabilities; dominated gens 1+2 parent sampling.

## Anomaly: gen 1 silent skip

**Root cause:** `DGM_outer.py:165` — `if not unresolved_ids: continue`. Silently skips a parent when its `unresolved_ids` list is empty AND none of the three 25% special-entry rolls (`solve_empty_patches`, `solve_stochasticity`, `solve_contextlength`) fire.

**Trigger sequence at gen 1:**
1. Archive = `[initial, 20260513_093044_671020]`.
2. `671020` killed at rung 0 with only 3 stage_small tasks evaluated; `total_unresolved_ids = []`.
3. Selection method `score_child_prop` with sigmoid weighting → `P(671020) ≈ 85.5%`, `P(initial) ≈ 14.5%`.
4. With `--selfimprove_size 2` and `seed=0`, both picks landed on `671020`.
5. For each pick, all three 25% rolls missed (0.75³ = 42% per parent; 18% chance for both).
6. Empty `unresolved_ids` + all dice miss → `continue` → no entry appended.
7. Result: `selfimprove_entries = []`, gen 1 instantly closed with zero children.

**Reproducibility:** deterministic given `seed=0`. Would not necessarily skip with different seed.

**Gen 3 partial:** same path hit on 1 of 2 parent slots → only 1 entry generated → `--selfimprove_size 2` effectively became 1.

**Fix candidates (post-run):**
1. Cheap: fall back to `entry_ids = resolved_ids + empty_ids` when `unresolved_ids` is empty.
2. Better: exclude killed-rung-0 children from parent eligibility upfront (line 67-85 candidate filter).
3. Architectural: weight selection probabilities by `evaluated_task_count` so `n=1` flukes don't dominate.

## Notable failures

- `093044_669420` (gen 0): compile fail. Likely argparse contract violation.
- `093044_671020` (gen 0): compiled but killed at rung 0 after 1 submitted patch. Subsequently polluted archive selection.
- `095931_353642` (gen 2): compile fail. Sister child to gen-2 winner.

## Cost breakdown (rough; actual = $26 OpenRouter)

| Component | Tasks/Calls | Est. USD |
|-----------|-------------|----------|
| Patch-gen LLM (5 mutation calls) | 5 | ~$1 |
| Eval (109 tasks × ~$0.22/task effective) | 109 | ~$25 |
| **Total actual** | | **$26.00** |

Effective per-task eval cost ~$0.22 (higher than earlier $0.11 calibration — possibly long-context tasks more expensive than baseline pilot, or many timeout-retries inflated cost). Worth investigating for ASHA budgeting.

## Output dir layout (gitignored)

```
output_dgm/20260513093044_666042/
├── dgm_outer.log
├── dgm_metadata.jsonl                          # 4 lines, 1 per gen
├── initial/
├── 20260513_093044_669420/                     # gen 0 child A (compile fail)
├── 20260513_093044_671020/                     # gen 0 child B (killed @ rung 0, archive pollutant)
├── 20260513_095931_353642/                     # gen 2 child A (compile fail)
├── 20260513_095931_355150/                     # gen 2 winner (68.97%)
└── 20260513_142007_309717/                     # gen 3 winner (70.59% — best)
```

Each promoted child dir contains:
- `model_patch.diff` — the GA blind-mutation patch
- `self_evo.md` — agent's reasoning trace
- `metadata.json` — child metadata + token usage
- `predictions/` — eval predictions per task
- `*_stage_{small,medium,full}_0.json` — score per rung (budgets 3/15/50)

## Three-scheduler comparison (final)

| Metric | Baseline | HB n=15 | **GA** |
|--------|----------|---------|--------|
| Wall | 9.98h | 30.0h | **9.63h** |
| Cost | $32.47 | $44.98 | **$26.00** |
| Children spawned | 6 | 54 | **5** |
| Compile rate | 100% | 70% | **60%** |
| Reached final rung | 5/6 | 4/54 = 7.4% | **2/5 = 40%** |
| Archive end | 7 | 5 | **4** |
| **Best end-of-run** | 28/47 = **59.57%** | 22/29 = **75.86%** | **24/34 = 70.59%** |
| Initial → Best Δ | +25.53pp | +41.82pp | **+36.55pp** |
| Productive gens | 4/4 | 4/4 | **2/4** |
| $ per pp gained | $1.27 | $1.08 | **$0.71** |

**GA is the cost-efficiency winner.** Best $/pp of improvement, despite losing half the gens to skip/compile issues.
