#!/usr/bin/env python3
"""Generate Results-chapter figures for the thesis.

Run:  python experiments/results/make_plots.py
Out:  experiments/results/fig_*.pdf  (5 figures)

IMPORTANT — data provenance
---------------------------
Accuracy numbers are *hardcoded from the verified *_NOTES.md tables*, NOT
read from the JSON. The JSON `best_child_score` field is the gamed metric
(tiny rung-0 denominators -> spurious 1.0 scores). Only the NOTES
full-/final-rung resolved/submitted figures are trustworthy. See
chap-results.tex "Metric Validity" section and the
metric_flaw_empty_patches memory.

Cost / wall / eval-task counts ARE pulled from JSON where present
(run-total only; per-generation cost is not tracked anywhere).

n=1 single seed per scheduler. Descriptive only, no error bars.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

SCHED = ["baseline", "ga", "hyperband", "asha"]
LABEL = {"baseline": "Baseline", "ga": "GA",
         "hyperband": "Hyperband", "asha": "ASHA"}
COLOR = {"baseline": "#4c72b0", "ga": "#55a868",
         "hyperband": "#c44e52", "asha": "#8172b3"}
JSON_FILE = {
    "baseline": "baseline_pilot.json",
    "ga": "ga_pilot.json",
    "hyperband": "hyperband_n15.json",
    "asha": "asha_pilot.json",
}

INITIAL_ACC = 34.04  # 16/47, identical for all 4 runs

# Honest per-generation accuracy (best promoted child, final-rung
# resolved/submitted) transcribed from *_NOTES.md. None = generation
# produced no archive-worthy child (GA gen 1 was silently skipped).
# GA gen 0 = 100% is a single-task (1/1) rung-0 fluke -> flagged, not a curve point.
PER_GEN_ACC = {
    "baseline":  [50.00, 54.17, 54.17, 59.57],
    "ga":        [None,  None,  68.97, 70.59],   # g0 fluke omitted, g1 skipped
    "hyperband": [59.26, 70.00, 67.44, 75.86],
    "asha":      [65.85, 50.00, 56.52, 48.89],
}
GA_GEN0_FLUKE = 100.0  # 1/1 rung-0, annotated separately

# Best-of-run peak (gen it occurred) from NOTES four-scheduler table.
PEAK = {"baseline": 59.57, "ga": 70.59, "hyperband": 75.86, "asha": 65.85}
PEAK_GEN = {"baseline": 3, "ga": 3, "hyperband": 3, "asha": 0}

# Honest run totals from NOTES (cross-checked vs JSON for cost/wall).
COST = {"baseline": 32.47, "ga": 26.00, "hyperband": 44.98, "asha": 39.00}
WALL_H = {"baseline": 9.98, "ga": 9.63, "hyperband": 30.0, "asha": 20.34}
EVAL_TASKS = {"baseline": 253, "ga": 109, "hyperband": 356, "asha": 346}
INTENDED_TASKS = {"baseline": 253, "ga": 103, "hyperband": 324, "asha": 346}
ACTUAL_TASKS   = {"baseline": 324, "ga":  91, "hyperband": 250, "asha": 368}
COMPILE_PCT = {"baseline": 100, "ga": 60, "hyperband": 70, "asha": 75}


def dollars_per_pp(s):
    return COST[s] / (PEAK[s] - INITIAL_ACC)


def load_json(s):
    with open(os.path.join(HERE, JSON_FILE[s])) as f:
        return json.load(f)


def save(fig, name):
    path = os.path.join(HERE, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------------------
# Fig 1 — cost vs best-of-run peak  (central [N.3] figure)
# ---------------------------------------------------------------------------
def fig_cost_vs_peak():
    # Per-scheduler annotation offset (points) and alignment so labels
    # never overlap each other or the initial-archive guideline.
    ANNOT = {
        "baseline":  {"xytext": (10,  6), "ha": "left",  "va": "bottom"},
        "ga":        {"xytext": (10, -4), "ha": "left",  "va": "top"},
        "hyperband": {"xytext": (-10, 0), "ha": "right", "va": "center"},
        "asha":      {"xytext": (10, -6), "ha": "left",  "va": "top"},
    }
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for s in SCHED:
        ax.scatter(COST[s], PEAK[s], s=140, color=COLOR[s],
                   edgecolor="black", linewidth=0.6, zorder=3)
        a = ANNOT[s]
        ax.annotate(f"{LABEL[s]}\n\\${COST[s]:.0f}, {PEAK[s]:.1f}%, "
                    f"\\${dollars_per_pp(s):.2f}/pp",
                    (COST[s], PEAK[s]), textcoords="offset points",
                    xytext=a["xytext"], ha=a["ha"], va=a["va"],
                    fontsize=8.5)
    ax.axhline(INITIAL_ACC, ls="--", c="gray", lw=0.9)
    ax.text(ax.get_xlim()[1], INITIAL_ACC + 0.6,
            f"initial archive {INITIAL_ACC:.2f}%", ha="right",
            fontsize=8, color="gray")
    ax.set_xlabel("Total run cost (USD)")
    ax.set_ylabel("Best-of-run accuracy (resolved/submitted, %)")
    ax.set_title("Cost vs. peak accuracy (n=1; lower-right = better value)")
    ax.grid(True, ls=":", alpha=0.5)
    # Margin so left/top annotations are not clipped by axes.
    ax.margins(x=0.12, y=0.10)
    save(fig, "fig_cost_vs_peak.pdf")


# ---------------------------------------------------------------------------
# Fig 2 — cheapest scheduler clearing an accuracy threshold
# ---------------------------------------------------------------------------
def fig_cost_threshold():
    thresholds = list(range(55, 80, 1))
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for s in SCHED:
        xs, ys = [], []
        for t in thresholds:
            if PEAK[s] >= t:
                xs.append(t)
                ys.append(COST[s])
        if xs:
            ax.step(xs, ys, where="post", color=COLOR[s],
                    lw=2, label=f"{LABEL[s]} (peak {PEAK[s]:.1f}%)")
    ax.set_xlabel("Required best-of-run accuracy threshold (%)")
    ax.set_ylabel("Total run cost to achieve it (USD)")
    ax.set_title("Cheapest scheduler clearing each accuracy threshold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, ls=":", alpha=0.5)
    save(fig, "fig_cost_threshold.pdf")


# ---------------------------------------------------------------------------
# Fig 3 — honest accuracy per generation
# ---------------------------------------------------------------------------
def fig_accuracy_per_gen():
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    gens = [0, 1, 2, 3]
    for s in SCHED:
        xs = [g for g, v in zip(gens, PER_GEN_ACC[s]) if v is not None]
        ys = [v for v in PER_GEN_ACC[s] if v is not None]
        ax.plot(xs, ys, "-o", color=COLOR[s], label=LABEL[s], lw=1.8)
    # GA gen-0 single-task fluke, flagged not connected
    ax.scatter([0], [GA_GEN0_FLUKE], marker="x", color=COLOR["ga"],
               s=70, zorder=4)
    ax.annotate("GA g0 = 1/1 rung-0\nfluke (not capability)",
                (0, GA_GEN0_FLUKE), textcoords="offset points",
                xytext=(6, -4), fontsize=7.5, color=COLOR["ga"])
    ax.annotate("GA g1 skipped\n(seed=0 selection bug)",
                (1, 35), textcoords="offset points", xytext=(4, 0),
                fontsize=7.5, color=COLOR["ga"])
    ax.axhline(INITIAL_ACC, ls="--", c="gray", lw=0.9)
    ax.text(3, INITIAL_ACC + 0.7, f"initial {INITIAL_ACC:.2f}%",
            ha="right", fontsize=8, color="gray")
    ax.set_xticks(gens)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best promoted child accuracy (%)")
    ax.set_title("Per-generation trajectory (final-rung, empty patches "
                 "excluded)")
    ax.legend(fontsize=8)
    ax.grid(True, ls=":", alpha=0.5)
    save(fig, "fig_accuracy_per_gen.pdf")


# ---------------------------------------------------------------------------
# Fig 4 — evaluation tasks consumed
# ---------------------------------------------------------------------------
def fig_eval_tasks():
    import numpy as np
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    xs = np.arange(len(SCHED))
    w = 0.38
    intended = [INTENDED_TASKS[s] for s in SCHED]
    actual   = [ACTUAL_TASKS[s]   for s in SCHED]
    ax.bar(xs - w/2, intended, w, label="Intended (planned)",
           color="#9ecae1", edgecolor="black", linewidth=0.5)
    ax.bar(xs + w/2, actual,   w, label="Actual exposure (R+U+E)",
           color="#3182bd", edgecolor="black", linewidth=0.5)
    ymax = max(max(intended), max(actual))
    for i, v in enumerate(intended):
        ax.text(i - w/2, v + ymax * 0.012, str(v), ha="center", fontsize=8)
    for i, v in enumerate(actual):
        ax.text(i + w/2, v + ymax * 0.012, str(v), ha="center", fontsize=8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([LABEL[s] for s in SCHED])
    ax.set_ylabel("Evaluation tasks (4 generations)")
    ax.set_title("Workload: intended vs actual exposure")
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    save(fig, "fig_eval_tasks.pdf")


# ---------------------------------------------------------------------------
# Fig 5 — compile rate
# ---------------------------------------------------------------------------
def fig_compile_rate():
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    xs = range(len(SCHED))
    ax.bar(xs, [COMPILE_PCT[s] for s in SCHED],
           color=[COLOR[s] for s in SCHED], edgecolor="black", linewidth=0.5)
    for i, s in enumerate(SCHED):
        ax.text(i, COMPILE_PCT[s] + 1.5, f"{COMPILE_PCT[s]}%",
                ha="center", fontsize=9)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([LABEL[s] for s in SCHED])
    ax.set_ylim(0, 110)
    ax.set_ylabel("Child compile rate (%)")
    ax.set_title("Compile reliability of generated children")
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    save(fig, "fig_compile_rate.pdf")


if __name__ == "__main__":
    # touch JSON so a schema drift surfaces loudly rather than silently
    for s in SCHED:
        load_json(s)
    fig_cost_vs_peak()
    fig_cost_threshold()
    fig_accuracy_per_gen()
    fig_eval_tasks()
    fig_compile_rate()
    print("\n$/pp:", {s: round(dollars_per_pp(s), 3) for s in SCHED})
