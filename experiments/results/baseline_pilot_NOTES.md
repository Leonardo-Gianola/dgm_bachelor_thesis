# Baseline scheduler — 4-gen pilot run

**Run ID:** `20260505004943_827754`
**Scheduler:** baseline
**Children per gen:** 2 (`--selfimprove_size 2`)
**Generations:** 4 (`--max_generation 4`)
**Benchmark:** `swe_verified_mini` (50 SWE-bench Verified Mini tasks)
**Model:** `openrouter/minimax/minimax-m2.5`
**Start:** 2026-05-05 00:49 CEST
**End:** 2026-05-05 10:48 CEST
**Wall time:** ~10h
**Total cost:** **$32.47**

## Initial baseline

| Metric | Value |
|--------|-------|
| Resolved | 16 / 47 |
| Empty patches | 11 |
| Accuracy | **34.04%** |

## Per-generation full-eval scores

| Gen | Child run_id | Parent | Resolved/Submitted | Empty | Accuracy |
|-----|--------------|--------|--------------------|-------|----------|
| 0a | `20260505_004943_832659` | initial | 25/50 | 2 | **50.00%** |
| 0b | `20260505_004943_832695` | initial | n/a (stage_small only, did not promote) | — | 33% (stage_small) |
| 1 | `20260505_040512_396618` | `..._832659` | 26/48 | 0 | **54.17%** |
| 2a | `20260505_055913_315290` | `..._040512_396618` | 26/48 | 0 | **54.17%** |
| 2b | `20260505_055913_316421` | `..._040512_396618` | 21/46 | 2 | 45.65% |
| 3 | `20260505_081634_345239` | `..._040512_396618` | **28/47** | 0 | **59.57%** 🏆 |

## Self-improve targets per gen

| Gen | Targets |
|-----|---------|
| 0 | `('initial', 'sphinx-doc__sphinx-9281')`, `('initial', 'sphinx-doc__sphinx-8269')` |
| 1 | `('..._832659', 'sphinx-doc__sphinx-7748')` |
| 2 | `('..._040512_396618', 'solve_stochasticity')` ×2 |
| 3 | `('..._040512_396618', 'sphinx-doc__sphinx-8475')` |

## Generation wall times

| Gen | Started | Ended | Duration |
|-----|---------|-------|----------|
| 0 | 00:49 | 04:05 | 3h16m |
| 1 | 04:05 | 05:59 | 1h54m |
| 2 | 05:59 | 08:16 | 2h17m |
| 3 | 08:16 | 10:48 | 2h32m |

## Key observations

- **Initial → Best: +25.53 pp absolute, +75% relative** (34.04% → 59.57%)
- Empty-patch count fell sharply after gen 0 (11 → 2 → 0 → 0 → 0)
- Gen 0 Child B was weak (1/3 stage-0, never promoted) — half of gen 0 compute on losing branch
- Gen 2 plateaued at 54.17% (matched gen 1, no improvement)
- Gen 3 final child broke through plateau to 59.57%
- Child A's gen 0 contribution: a `tools/search.py` code-search wrapper around grep/find

## Notable failures

- Gen 0 Child B's stage-0 evals on `django__django-11790` and `django__django-12050` hit `timeout 1800` (30-min agent cap), produced empty patches.

## Output dir layout

```
output_dgm/20260505004943_827754/
├── dgm_outer.log
├── dgm_metadata.jsonl                          # 274 lines, 1 per gen
├── initial/                                    # bootstrap baseline
├── 20260505_004943_832659/                     # gen 0 Child A (50%)
├── 20260505_004943_832695/                     # gen 0 Child B (weak)
├── 20260505_040512_396618/                     # gen 1 (54.2%)
├── 20260505_055913_315290/                     # gen 2a (54.2%)
├── 20260505_055913_316421/                     # gen 2b (45.7%)
└── 20260505_081634_345239/                     # gen 3 (59.6% — best)
```

Each child dir contains:
- `model_patch.diff` — the self-improvement patch
- `self_evo.md` — agent's reasoning trace
- `metadata.json` — child metadata + token usage
- `predictions/` — eval predictions per task
- `*_stage_{small,medium,full}_0.json` — score per rung
