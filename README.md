# Multi-fidelity evaluation schedulers for the Darwin Gödel Machine

BSc thesis, University of Southern Denmark, spring 2026. Supervisor: Serkan Ayvaz. Graded 12 on the Danish scale.

This is a fork of the [Darwin Gödel Machine](https://github.com/jennyzzt/dgm) ([paper](https://arxiv.org/abs/2505.22954), Zhang et al., Sakana AI). The upstream system evolves a population of coding agents that rewrite their own source code, scores each child on SWE-bench, and keeps the best ones in an archive that seeds the next generation.

**The question this thesis asks.** Most of the compute in that loop is spent fully evaluating children that end up worse than their parent. Can a multi-fidelity evaluation scheduler cut that waste without giving up final accuracy?

**What I did.** I added four evaluation schedulers behind a single flag, held everything else fixed (same model, same initial archive, same per-generation task budget, same seed), and ran each one for four generations on a 50-task subset of SWE-bench Verified. That makes the scheduler the only independent variable.

---

## The four schedulers

All four live in [`schedulers.py`](schedulers.py) and are selected with `--scheduler`.

| Scheduler | Idea |
|---|---|
| `baseline` | The upstream policy. Every child is evaluated through a fixed three-stage pipeline, then promoted on score. This is the reference cost. |
| `hyperband` | Synchronous successive halving. Many children start on a small task budget, survivors are re-evaluated on larger rungs, so cheap failures die cheap. |
| `asha` | The asynchronous version. A free worker promotes whichever candidate is currently promotable instead of waiting for a whole rung to finish, which keeps machines busy at the cost of ranking on less information. |
| `ga` | A deliberate control. A blind high-temperature mutation loop with no error-log context, to see how much of the improvement comes from informed self-modification at all rather than from search pressure. |

Shared controls, so the comparison is fair: `--generation_task_budget_total` caps the tasks any scheduler may spend per generation, `--selfimprove_size` fixes the parent slots, and every run starts from the same bootstrapped archive.

## Results

Four single-seed pilot runs, four generations each, `swe_verified_mini` (50 SWE-bench Verified tasks), model `minimax/minimax-m2.5` via OpenRouter. All four start from the same initial agent at **34.04%** (16 of 47 submitted).

| Scheduler | Best agent | Lift over the initial agent | Cost | Wall time |
|---|---:|---:|---:|---:|
| `baseline` | 59.57% (28/47) | +25.5 pp | $32.47 | ~10 h |
| `hyperband` | **75.86%** (22/29) | +41.8 pp | $44.98 | ~30 h |
| `ga` | 70.59% (24/34) | +36.6 pp | **$26.00** | 9.6 h |
| `asha` | 65.85% (27/41) | +31.8 pp | $39.00 | 20.3 h |

Two readings, and both are in the thesis:

- **Hyperband buys the highest peak, and it is not cheap.** It reached the best agent of the four but spent the most to get there.
- **The blind genetic algorithm is the cost-efficiency winner.** It captured about 85% of Hyperband's lift for roughly a third of the money, which is an uncomfortable result for the assumption that informed self-modification is what drives the loop.

**Read the denominators before quoting the accuracy numbers.** The DGM metric is `resolved / submitted`, and a task the agent never emits a prediction for drops out of the denominator entirely, while an empty patch stays in it as a failure. Hyperband's 75.86% sits on 29 submitted tasks rather than 47, so part of that headline is a denominator artefact. The results chapter works through this rather than hiding it.

**What this evidence does not support.** n=1 per scheduler, one seed, four generations, no significance tests. The API budget ran out before the planned multi-seed phases, so the experiments were frozen and the thesis is written as an exploratory cost-versus-accuracy reading of four candidates, not as a ranking. Total spend across all four pilots was about $142.

Per-run detail, including per-generation tables and narrative notes, is in [`experiments/results/`](experiments/results/). Figures are regenerated with `python experiments/results/make_plots.py`.

## What is mine and what is upstream

Mine:

```
schedulers.py                     the four schedulers
DGM_outer.py                      --scheduler wiring, shared budget controls
self_improve_step.py              child generation and evaluation split out for reuse by schedulers
experiments/run_scheduler.py      per-scheduler runner with live generation metrics
experiments/run_full_eval.py      full evaluation of a chosen agent
compare_scheduler_runs.py         aggregate runs into CSV and JSON
experiments/results/              pilot results, notes and plots
benchmarks/                       mini-benchmark caching for offline reproducible runs
thesis/                           the thesis itself, LaTeX, SDU template
website/                          small run viewer
```

Upstream, kept mostly as it was: the evolution loop structure, the coding agent, the SWE-bench and Polyglot harnesses, the Docker sandboxing and the initial archives.

Along the way the fork also picked up the fixes that long unattended runs force on you: a per-image Docker build lock, container startup retry with exponential backoff, empty-patch retry, corrupt-JSONL tolerance in the summary printer, a separate timeout for the ASHA evaluation loop, and a much shorter per-task agent timeout.

## Running it

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

export OPENAI_API_KEY='...'
export ANTHROPIC_API_KEY='...'
docker run hello-world          # Docker must be working, every child runs sandboxed

cd swe_bench && git clone https://github.com/princeton-nlp/SWE-bench.git
cd SWE-bench && git checkout dc4c087c2b9e4cefebf2e3d201d27e36 && pip install -e . && cd ../../

python -m benchmarks.cache_swe_verified_mini    # cache the mini benchmark
```

Bootstrap the initial archive once:

```bash
python test_swebench.py --benchmark swe_verified_mini --full_mini \
  --agent_dir initial_swe_verified_mini --write_agent_metadata
```

Then run a scheduler:

```bash
# reference
python DGM_outer.py --scheduler baseline --max_generation 4

# multi-fidelity
python DGM_outer.py --scheduler hyperband \
  --hyperband_eta 5 --hyperband_budgets 2,10,50 --hyperband_initial_children 15 \
  --max_generation 4 --generation_task_budget_total 100

# resume an interrupted run
python DGM_outer.py --continue_from output_dgm/<run_id>
```

Output lands in `output_dgm/<timestamp>/`. Compare finished runs with `python compare_scheduler_runs.py`.

A word of warning if you want to reproduce this: a single four-generation run took between 10 and 30 hours of wall time and tens of dollars of API spend. Start with `--single_task` on `test_swebench.py`.

## The thesis

LaTeX sources are in [`thesis/`](thesis/). Build with `make -C thesis`. Chapters follow the experiment: background on self-improving systems and multi-fidelity search, the scheduler designs, the implementation, the results with the metric caveats, and a conclusion that treats the scheduler choice as a real knob in self-improving systems rather than a solved one. `thesis/ai-declaration.tex` documents how AI tooling was used, as required.

## Credit and license

The DGM framework, the coding agent and the benchmark harnesses are the work of the upstream authors and are used under Apache 2.0. My contribution is the scheduler layer, the experimental protocol, the results and the thesis.

```bibtex
@article{zhang2025darwin,
  title  = {Darwin G\"odel Machine: Open-Ended Evolution of Self-Improving Agents},
  author = {Zhang, Jenny and Hu, Shengran and Lu, Cong and Lange, Robert and Clune, Jeff},
  journal= {arXiv preprint arXiv:2505.22954},
  year   = {2025}
}
```
