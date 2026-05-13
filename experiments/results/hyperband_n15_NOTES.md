# Hyperband scheduler — 4-gen n=15 run

**Run ID:** `20260511231317_229925`
**Scheduler:** hyperband
**Initial children per gen:** 15 (`--hyperband_initial_children 15`)
**Eta:** 5 (`--hyperband_eta 5`)
**Budgets:** `[2, 10, 50]` (`--hyperband_budgets 2,10,50`)
**Generations:** 4 (`--max_generation 4`)
**Selfimprove size:** 2 (`--selfimprove_size 2`) — parent selection slots per gen
**Per-gen task budget:** 100 (`--generation_task_budget_total 100`)
**Benchmark:** `swe_verified_mini` (50 SWE-bench Verified Mini tasks)
**Model:** `openrouter/minimax/minimax-m2.5`
**Start:** 2026-05-11 23:13 CEST
**End:** 2026-05-13 05:14 CEST
**Wall time:** ~30h (includes 2 idle nights with sleep-inhibitor on)
**Total cost:** **$44.98** (~$6.59 patch-gen tokens, ~$38 eval)

## Initial baseline

| Metric | Value |
|--------|-------|
| Resolved | 16 / 47 |
| Empty patches | 11 |
| Accuracy | **34.04%** |

## Per-generation full-eval scores

| Gen | Wall | Compiled | Promoted to archive | Parent | Entry | Resolved/Submitted (final rung) | Accuracy |
|-----|------|----------|---------------------|--------|-------|--------------------------------|----------|
| 0   | 6.2h | 11/15    | `20260511_235216_700212` | initial | `sphinx-doc__sphinx-8035` | 32/54 | **59.26%** |
| 1   | 8.1h | 12/15    | `20260512_060427_158796` | initial | `solve_empty_patches`     | 14/20 | **70.00%** |
| 2   | 7.9h | 7/14     | `20260512_141320_329009` | `..._235216_700212` | `solve_stochasticity` | 29/43 | **67.44%** |
| 3   | 7.7h | 8/10     | `20260512_214141_888600` | `..._060427_158796` | `django__django-12308` | 22/29 | **75.86%** 🏆 |

## Rung mechanics per generation

Each gen ran 3 rungs (eta=5, budgets `[2, 10, 50]`):

| Gen | Rung 0 in/promoted/killed | Rung 1 in/promoted/killed | Rung 2 in/promoted/killed | Budget consumed |
|-----|---------------------------|---------------------------|---------------------------|-----------------|
| 0   | 15 / 3 / 12               | 3 / 1 / 2                 | 1 / 1 / 0                 | 94/100          |
| 1   | 15 / 3 / 12               | 3 / 1 / 2                 | 1 / 1 / 0                 | 94/100          |
| 2   | 14 / 3 / 11               | 3 / 1 / 2                 | 1 / 1 / 0                 | 92/100          |
| 3   | 10 / 2 / 8                | 2 / 1 / 1                 | 1 / 1 / 0                 | 76/100          |

## Self-improve targets per gen

Memory shows entries are parent-and-task pairs at the point of patch generation:

| Gen | Best-promoted entry |
|-----|---------------------|
| 0   | `('initial', 'sphinx-doc__sphinx-8035')` |
| 1   | `('initial', 'solve_empty_patches')` |
| 2   | `('20260511_235216_700212', 'solve_stochasticity')` |
| 3   | `('20260512_060427_158796', 'django__django-12308')` |

## Generation wall times

| Gen | Started      | Ended        | Duration |
|-----|--------------|--------------|----------|
| 0   | 05-11 23:13  | 05-12 05:27  | 6h14m    |
| 1   | 05-12 05:27  | 05-12 13:34  | 8h07m    |
| 2   | 05-12 13:34  | 05-12 21:30  | 7h56m    |
| 3   | 05-12 21:30  | 05-13 05:14  | 7h44m    |

## Key observations

- **Initial → Best: +41.82pp absolute, +123% relative** (34.04% → 75.86%)
- Late-gen winners trend **higher rate but lower absolute solved** — `235216` (gen-0) solved most (32 vs 14/29/22 later)
- **Empty-patch escape inflates later-gen rates** (e.g. gen-3 winner 22/29 = 0.76; 21 tasks excluded as empty patches). Documented in `metric_flaw_empty_patches.md`. Not a thesis bug — inherited from upstream DGM/Sakana.
- Compile yields dropped over gens: 11/15 → 12/15 → 7/14 → 8/10 — children inheriting more changes more likely to break argparse contract or other compile-time checks
- Each gen consumed near the 100-task budget envelope (94, 94, 92, 76)
- Synchronous SHA: all candidates finish a rung before the next rung starts — no async promotion (use ASHAScheduler for that)

## Notable failures

- `coding_agent.py` argparse contract violations caused most compile-rejects. Fixed at commit `4dbdf02` "fix(self_improve): argparse guard + prompt constraint on required CLI flags" pre-run; partial protection only — some children still removed flags.

## Cost breakdown

| Component | USD |
|-----------|-----|
| Patch-gen LLM tokens | ~$6.59 |
| Eval (SWE-bench harness, agent calls) | ~$38 |
| **Total** | **$44.98** |

Within $49 budget envelope. Estimate was $40-50.

## Output dir layout (gitignored)

```
output_dgm/20260511231317_229925/
├── dgm_outer.log
├── dgm_metadata.jsonl                          # 4 lines, 1 per gen
├── initial/
├── 20260511_235216_700212/                     # gen 0 winner (59.26%)
├── 20260512_060427_158796/                     # gen 1 winner (70.00%)
├── 20260512_141320_329009/                     # gen 2 winner (67.44%)
├── 20260512_214141_888600/                     # gen 3 winner (75.86% — best)
└── <many killed children dirs>
```

Each promoted child dir contains:
- `model_patch.diff` — the self-improvement patch
- `self_evo.md` — agent's reasoning trace
- `metadata.json` — child metadata + token usage
- `predictions/` — eval predictions per task
- `*_stage_{small,medium,full}_0.json` — score per rung (corresponds to budgets 2/10/50)
