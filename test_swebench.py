import argparse
import datetime
import json
import os

from benchmarks.config import BENCHMARKS, get_benchmark, get_dataset_source, load_benchmark_subset
from benchmarks.swe_verified_harness import harness as verified_harness
from benchmarks.swe_verified_report import make_report as make_verified_report
from polyglot.harness import harness as polyglot_harness
from utils.common_utils import load_json_file
from utils.evo_utils import get_all_performance, get_model_patch_paths_from_agent_dir


def resolve_task_list(args):
    benchmark = get_benchmark(args.benchmark)
    if args.single_task:
        return [args.single_task], "single_task"
    if args.subset:
        return load_json_file(args.subset), os.path.splitext(os.path.basename(args.subset))[0]
    if args.full_mini:
        if benchmark.full_subset_name is None:
            raise ValueError(f"Benchmark {args.benchmark} does not define a full mini subset.")
        return load_benchmark_subset(args.benchmark, benchmark.full_subset_name), benchmark.full_subset_name

    default_subset = benchmark.default_manual_subset or benchmark.stage_subset_names[0]
    return load_benchmark_subset(args.benchmark, default_subset), default_subset


def resolve_patch_paths(args):
    if args.model_patch_paths:
        return args.model_patch_paths.split(",")
    if args.agent_dir:
        return get_model_patch_paths_from_agent_dir(args.agent_dir)
    return None


def maybe_write_agent_metadata(args, subset_name, task_count, output_dir, model_name_or_path, evaluation_dirs):
    if not args.write_agent_metadata or not args.agent_dir:
        return

    metadata_path = os.path.join(args.agent_dir, "metadata.json")
    metadata = load_json_file(metadata_path) if os.path.exists(metadata_path) else {}
    _, overall_performance = get_all_performance(model_name_or_path, results_dir=output_dir)
    metadata.update(
        {
            "benchmark_name": args.benchmark,
            "dataset_source": get_dataset_source(args.benchmark),
            "overall_performance": overall_performance,
            "benchmark_performance": overall_performance,
            "evaluation_dirs": [str(dn) for dn in evaluation_dirs],
            "benchmark_prediction_dirs": [str(dn) for dn in evaluation_dirs],
            "evaluated_subset_names": [subset_name],
            "evaluated_task_count": overall_performance.get("total_submitted_instances", 0) if overall_performance else 0,
            "budget_name": subset_name,
            "budget_size": task_count,
            "search_strategy": "manual",
            "bootstrap_placeholder": False,
        }
    )
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Run benchmark evaluations on the agent.")
    parser.add_argument("--benchmark", type=str, default="swe_verified_mini", choices=sorted(BENCHMARKS), help="Benchmark configuration to use.")
    parser.add_argument("--agent_dir", type=str, default=None, help="Archive directory for the agent to evaluate.")
    parser.add_argument("--model_patch_paths", type=str, default=None, help="Comma-separated paths to model patches.")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="Override the model/run name used for outputs.")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to store benchmark outputs.")
    parser.add_argument("--max_workers", type=int, default=5, help="Number of workers to use.")
    parser.add_argument("--num_evals", type=int, default=1, help="Repeated number of benchmark evaluations.")
    parser.add_argument("--num_evals_parallel", type=int, default=1, help="Number of parallel repeated evaluations.")
    parser.add_argument("--num_eval_procs", type=int, default=5, help="Parallel processes per report generation.")
    parser.add_argument("--write_agent_metadata", default=False, action="store_true", help="Write benchmark results back into the selected agent archive metadata.")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--single_task", type=str, default=None, help="Evaluate a single benchmark task by instance ID.")
    mode_group.add_argument("--subset", type=str, default=None, help="Path to a JSON file containing benchmark instance IDs.")
    mode_group.add_argument("--full_mini", default=False, action="store_true", help="Evaluate the full SWE-bench Verified Mini 50-task benchmark.")

    args = parser.parse_args()
    benchmark = get_benchmark(args.benchmark)

    test_task_list, subset_name = resolve_task_list(args)
    model_patch_paths = resolve_patch_paths(args)

    if args.model_name_or_path is not None:
        model_name_or_path = args.model_name_or_path
    elif args.agent_dir:
        model_name_or_path = os.path.basename(os.path.abspath(args.agent_dir))
    else:
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        model_name_or_path = f"baseline_{run_id}"

    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_dir = os.path.join("manual_benchmark_runs", f"{benchmark.name}_{subset_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    prediction_dir = os.path.join(output_dir, benchmark.prediction_dir_name)
    if benchmark.kind == "polyglot":
        evaluation_dirs = polyglot_harness(
            test_task_list=test_task_list,
            num_samples=-1,
            max_workers=min(args.max_workers, len(test_task_list)),
            model_name_or_path=model_name_or_path,
            model_patch_paths=model_patch_paths,
            num_evals=args.num_evals,
            num_evals_parallel=min(args.num_evals_parallel, args.num_evals),
            pred_dname=prediction_dir,
            output_dir=output_dir,
        )
    else:
        evaluation_dirs = verified_harness(
            test_task_list=test_task_list,
            num_samples=-1,
            max_workers=min(args.max_workers, len(test_task_list)),
            model_name_or_path=model_name_or_path,
            model_patch_paths=model_patch_paths,
            num_evals=args.num_evals,
            num_evals_parallel=args.num_evals_parallel,
            pred_dname=prediction_dir,
            benchmark_name=args.benchmark,
        )
        make_verified_report(
            evaluation_dirs,
            run_ids=[f"{i:03}" for i in range(len(evaluation_dirs))],
            benchmark_name=args.benchmark,
            dataset_name=get_dataset_source(args.benchmark),
            output_dir=output_dir,
            dnames_workers=args.num_evals_parallel,
            num_eval_procs=args.num_eval_procs,
        )

    maybe_write_agent_metadata(args, subset_name, len(test_task_list), output_dir, model_name_or_path, evaluation_dirs)


if __name__ == "__main__":
    main()
