import argparse
import csv
import json
import os

from utils.evo_utils import load_dgm_metadata


def summarize_run(run_dir):
    metadata_path = os.path.join(run_dir, "dgm_metadata.jsonl")
    generations = load_dgm_metadata(metadata_path)
    if not generations:
        raise ValueError(f"No generation metadata found in {run_dir}")

    final_generation = generations[-1]
    rung_promotions = []
    for generation in generations:
        for rung in generation.get("rungs", []):
            rung_promotions.append({
                "generation": generation["generation"],
                "rung": rung.get("rung"),
                "budget_size": rung.get("budget_size"),
                "num_candidates_in": rung.get("num_candidates_in"),
                "num_promoted": rung.get("num_promoted"),
                "num_killed": rung.get("num_killed"),
            })

    return {
        "run_dir": run_dir,
        "scheduler_name": final_generation.get("scheduler_name"),
        "benchmark_name": final_generation.get("benchmark_name"),
        "num_generations": len(generations),
        "best_child_score": max(g.get("best_child_score", 0) for g in generations),
        "avg_child_score_last_generation": final_generation.get("avg_child_score", 0),
        "wall_clock_seconds_total": sum(g.get("generation_wall_clock_seconds", 0) for g in generations),
        "children_generated_total": sum(g.get("children_generated_count", 0) for g in generations),
        "children_compiled_total": sum(g.get("children_compiled_count", 0) for g in generations),
        "children_fully_evaluated_total": sum(g.get("children_fully_evaluated_count", 0) for g in generations),
        "evaluation_budget_tasks_consumed_total": sum(g.get("evaluation_budget_tasks_consumed", 0) for g in generations),
        "final_archive_size": final_generation.get("archive_size", len(final_generation.get("archive", []))),
        "rung_promotions": rung_promotions,
        "generations": generations,
    }


def write_csv(summaries, output_path):
    fieldnames = [
        "run_dir",
        "scheduler_name",
        "benchmark_name",
        "num_generations",
        "best_child_score",
        "avg_child_score_last_generation",
        "wall_clock_seconds_total",
        "children_generated_total",
        "children_compiled_total",
        "children_fully_evaluated_total",
        "evaluation_budget_tasks_consumed_total",
        "final_archive_size",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: summary[key] for key in fieldnames})


def main():
    parser = argparse.ArgumentParser(description="Compare baseline and Hyperband DGM runs.")
    parser.add_argument("--run_dirs", nargs="+", required=True, help="One or more output_dgm run directories.")
    parser.add_argument("--output_json", default=None, help="Optional path for the full JSON comparison summary.")
    parser.add_argument("--output_csv", default=None, help="Optional path for a flat CSV summary.")
    args = parser.parse_args()

    summaries = [summarize_run(run_dir) for run_dir in args.run_dirs]
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)
    if args.output_csv:
        write_csv(summaries, args.output_csv)
    if not args.output_json and not args.output_csv:
        print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
