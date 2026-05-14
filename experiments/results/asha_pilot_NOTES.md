# ASHA scheduler — 4-gen n=15 run

**Run ID:** `20260513201108_583607`
**Scheduler:** asha (Asynchronous Successive Halving Algorithm)
**Initial children per gen:** 15 (`--hyperband_initial_children 15`)
**Eta:** 5 (`--hyperband_eta 5`)
**Budgets:** `[2, 10, 50]` (`--hyperband_budgets 2,10,50`)
**Generations:** 4 (`--max_generation 4`)
**Selfimprove size:** 2 (`--selfimprove_size 2`) — parent selection slots per gen
**Selfimprove workers:** 2 (`--selfimprove_workers 2`)
**Per-gen task budget:** 100 (`--generation_task_budget_total 100`)
**Benchmark:** `swe_verified_mini` (50 SWE-bench Verified Mini tasks)
**Model:** `openrouter/minimax/minimax-m2.5`
**Start:** 2026-05-13 20:11 CEST
**End:** 2026-05-14 16:31 CEST
**Wall time:** 20h 20m (20.34h)
**Total cost:** **$39.00** (actual, OpenRouter dashboard)

## Initial baseline

| Metric | Value |
|--------|-------|
| Resolved | 16 / 47 |
| Empty patches | 11 |
| Accuracy | **34.04%** |

## Per-generation final-rung scores

| Gen | Wall | Compiled | Promoted to archive | Parent | Entry | Final-rung r/s | Accuracy |
|-----|------|----------|---------------------|--------|-------|----------------|----------|
| 0   | 6.70h | 12/15    | `20260513_201108_588422` | initial | sphinx-doc__sphinx-10435 | 27/41 | **65.85%** 🏆 |
| 1   | 4.48h | 11/15    | `20260514_025310_957460` | initial | sphinx-doc__sphinx-10323 | 23/46 | **50.00%** |
| 2   | 4.45h | 12/15    | `20260514_072145_131086` | `..._957460` | sphinx-doc__sphinx-7748   | 26/46 | **56.52%** |
| 3   | 4.71h | 10/15    | `20260514_115424_040075` | `..._957460` | django__django-12406      | 22/45 | **48.89%** |

## Rung mechanics per generation

Each gen ran 3 rungs (eta=5, budgets `[2, 10, 50]`):

| Gen | Rung 0 in/promoted/killed | Rung 1 in/promoted/killed | Rung 2 in/promoted/killed | Budget consumed |
|-----|---------------------------|---------------------------|---------------------------|-----------------|
| 0   | 12 / 3 / 9                | 3 / 1 / 2                 | 1 / 1 / 0                 | 88/100          |
| 1   | 11 / 3 / 8                | 3 / 1 / 2                 | 1 / 1 / 0                 | 86/100          |
| 2   | 12 / 3 / 9                | 3 / 1 / 2                 | 1 / 1 / 0                 | 88/100          |
| 3   | 10 / 3 / 7                | 3 / 1 / 2                 | 1 / 1 / 0                 | 84/100          |

Note: rung-0 `num_candidates_in` < 15 because compile-failed children were never submitted to rung-0 eval.

## Generation wall times

| Gen | Started      | Ended        | Duration |
|-----|--------------|--------------|----------|
| 0   | 05-13 20:11  | 05-14 02:53  | 6h42m    |
| 1   | 05-14 02:53  | 05-14 07:21  | 4h28m    |
| 2   | 05-14 07:21  | 05-14 11:48  | 4h27m    |
| 3   | 05-14 11:48  | 05-14 16:31  | 4h43m    |

## Async benefit (vs HB synchronous)

Gen 0 ran HB-like (6.70h) because all 15 children processed rung 0 before any rung-1 future could pull from the worker pool. From gen 1 onward, workers stayed mixed-rung-busy and average wall dropped to **4.55h/gen**.

| Scheduler | Wall (4 gens) | vs HB |
|-----------|--------------|-------|
| HB n=15 | 30.0h | baseline |
| **ASHA n=15** | **20.34h** | **-32%** |

Async parallelism win **measurable but partial** — limited by `selfimprove_workers=2` and FIFO ThreadPoolExecutor. With `workers ≥ n_children` and a priority queue, the speedup would be larger.

## Key observations

- **Peak: 65.85% (gen 0)** vs HB's 75.86% — ASHA underperforms HB on max-quality
- **Initial → Best: +31.81pp absolute, +93% relative** (34.04% → 65.85%)
- **Archive non-monotonic:** gen 0 > gen 2 (56.52%) > gen 1 (50%) > gen 3 (48.89%). Score regression from gen 1 onward; never recovered.
- **`score_child_prop` selection picked `131086` (gen-2 winner, 56.52%) most often in gen 3** despite `588422` having higher rate (65.85%) — selection penalizes parents with existing children.
- **Compile rate average 75%** (range 67-80%): worse than baseline (100%), better than GA (60%), similar to HB (70-80%).
- **Rung-0 saturation by gen 3:** all 10 evaluated children scored 1.0 at rung 0. Selection by speed not score.

## Selection-policy sub-optimality (gen 2 case study)

Three children reached rung 1 in gen 2:

| Child | Cum rate after rung 1 | Promoted? | Why |
|-------|------------------------|-----------|------|
| `131086` | 0.70 (7/10) | **YES** | Finished rung 1 first |
| `132561` | **0.818 (9/11)** | NO (killed) | Finished rung 1 later, quota=1 full |
| `252024` | 0.70 (7/10) | NO | Same as 132561 |

ASHA promoted `131086` to rung 2 → final rung-2 rate 56.52%.

HB synchronous would have ranked all three at rung-1 close and promoted `132561` (highest score). True counterfactual unknown without re-evaluation, but illustrates the **algorithmic difference** thesis-relevantly.

## Cost breakdown

| Component | Tasks/Calls | Est. USD |
|-----------|-------------|----------|
| Patch-gen LLM (60 mutation calls) | 60 | ~$3 |
| Eval (346 tasks × ~$0.10/task) | 346 | ~$36 |
| **Total actual** | | **$39.00** |

Per-task effective ~$0.11 — close to HB's calibration ($0.12/task) and below GA's ($0.24). Likely because ASHA's rung-0 task variance was lower (no long-tail django tasks dominating).

## Pre-launch fix required

`schedulers.py:15` originally had `GENERATION_TIMEOUT_SECONDS = 1.5h` reused both for patch-gen and ASHA's full eval loop. ASHA would have timed out mid-rung-0 at 1.5h.

Fix (commit `5945c0d`):

```python
GENERATION_TIMEOUT_SECONDS = 1.5 * 60 * 60  # patch-generation phase only
ASHA_GENERATION_EVAL_TIMEOUT_SECONDS = 36 * 60 * 60  # ASHA async eval loop
```

`ASHAScheduler.run_generation` deadline switched to the new constant. First ASHA launch (commit `c26996a`-era code path) was killed pre-eval; relaunched with the patch.

## Output dir layout (gitignored)

```
output_dgm/20260513201108_583607/
├── dgm_outer.log
├── dgm_metadata.jsonl                          # 4 lines, 1 per gen
├── initial/
├── 20260513_201108_588422/                     # gen 0 winner (65.85%) — BEST
├── 20260514_025310_957460/                     # gen 1 winner (50.00%)
├── 20260514_072145_131086/                     # gen 2 winner (56.52%)
├── 20260514_115424_040075/                     # gen 3 winner (48.89%)
└── <55 killed children dirs>
```

Each promoted child dir contains:
- `model_patch.diff` — the self-improvement patch
- `self_evo.md` — agent's reasoning trace
- `metadata.json` — child metadata + token usage
- `predictions/` — eval predictions per task
- `*_asha_rung_{0,1,2}_0.json` — score per rung (corresponds to budgets 2/10/50)

## Four-scheduler comparison (final)

| Metric | Baseline | HB n=15 | GA | **ASHA** |
|--------|----------|---------|------|----------|
| Wall | 9.98h | 30.0h | 9.63h | **20.34h** |
| Cost | $32.47 | $44.98 | $26.00 | **$39.00** |
| Children spawned | 6 | 54 | 5 | **60** |
| Compile rate | 100% | 70% | 60% | **75%** |
| Reached final rung | 5/6 | 4/54 | 2/5 | **4/60** |
| Archive end | 7 | 5 | 4 | **5** |
| **Best end-of-run** | 28/47 = 59.57% | 22/29 = **75.86%** | 24/34 = 70.59% | **27/41 = 65.85%** |
| Initial → Best Δ | +25.53pp | **+41.82pp** | +36.55pp | +31.81pp |
| $/pp gained | $1.27 | $1.08 | **$0.71** | $1.23 |
| Productive gens | 4/4 | 4/4 | 2/4 | **4/4** |
| Async speedup vs HB | — | — | — | **-32%** wall |

- **HB wins absolute quality** (75.86%)
- **GA wins cost-efficiency** ($0.71/pp)
- **ASHA wins wall-time** vs HB (32% faster) at small quality cost
- **Baseline wins compile reliability** (100%) but lowest peak
