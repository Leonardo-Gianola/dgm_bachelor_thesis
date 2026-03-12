import argparse
import datetime
import json
import math
import os
import random
import shutil
import time

from benchmarks.config import BENCHMARKS, get_benchmark, get_cumulative_stage_task_counts, load_benchmark_subset
from prompts.self_improvement_prompt import find_selfimprove_eval_logs
from schedulers import ASHAScheduler, BaselineScheduler, HyperbandScheduler
from utils.common_utils import load_json_file
from utils.docker_utils import setup_logger
from utils.evo_utils import load_dgm_metadata, is_compiled_self_improve

def initialize_run(output_dir, benchmark_name, prevrun_dir=None):
    # Initialize archive
    start_gen_num = 0
    if not prevrun_dir:
        archive = ['initial']
    else:
        # Load previous run's archive
        metadata_path = os.path.join(prevrun_dir, "dgm_metadata.jsonl")
        metadata = load_dgm_metadata(metadata_path, last_only=True)
        archive = metadata['archive']
        start_gen_num = metadata['generation'] + 1

    # Copy cached initial version into experiment dir
    benchmark = get_benchmark(benchmark_name)
    initial_archive_path = benchmark.initial_archive_path
    if initial_archive_path is None or not initial_archive_path.exists():
        raise RuntimeError(
            f"Initial archive for benchmark '{benchmark_name}' is missing at "
            f"{initial_archive_path}. Bootstrap it with test_swebench.py first."
        )
    initial_output_path = os.path.join(output_dir, "initial")
    if not prevrun_dir and not os.path.exists(initial_output_path):
        shutil.copytree(initial_archive_path, initial_output_path, dirs_exist_ok=True)
    
    return archive, start_gen_num

def any_exceeding_context_length(output_dir, commit_id, instance_ids):
    """
    Check if any of the issues have exceeded the context length.
    """
    for instance_id in instance_ids:
        md_logs, _, _, _ = find_selfimprove_eval_logs(instance_id, output_dir, commit_id, filter=False)
        if not md_logs:
            continue
        md_log = md_logs[0]
        error_str = "Error in get_response_withtools: Error code: 400 - {'message': 'Input is too long for requested model.'}"
        # Repeated error_str means no attempt to fix it
        if f'{error_str}\n{error_str}' in md_log:
            return True
    return False

def choose_selfimproves(output_dir, archive, selfimprove_size, method='random', run_baseline=None, benchmark_name='swe_verified_mini', rng=None):
    """
    Choose self-improve attempts for the current generation.
    """
    selfimprove_entries = []
    rng = rng or random

    # Get parent candidates
    candidates = {}
    for commit in archive:
        try:
            metadata_path = os.path.join(output_dir, commit, "metadata.json")
            metadata = load_json_file(metadata_path)
            candidates[commit] = {
                'accuracy_score': metadata['overall_performance']['accuracy_score'],
                'total_unresolved_ids': metadata['overall_performance']['total_unresolved_ids'],
                'total_emptypatch_ids': metadata['overall_performance']['total_emptypatch_ids'],
                'total_resolved_ids': metadata['overall_performance']['total_resolved_ids'],
                'children_count': 0,
            }
            # update children count, parent should already be in the archive
            if commit != 'initial':
                parent_commit = metadata['parent_commit']
                candidates[parent_commit]['children_count'] += 1
        except Exception as e:
            # probably because swe-eval failed, generated code did not compile, etc.
            print(f"{commit} not eligible for being a parent: {e}")
            continue

    # Choose parents based on method and baseline
    if run_baseline == 'no_darwin':
        # Always take the last commit
        commits = list(candidates.keys())
        parent_commits = commits[-1:]
    elif method == 'score_prop':
        # Choose parents based on score
        commits = list(candidates.keys())
        scores = [candidates[commit]['accuracy_score'] for commit in commits]
        scores = [1 / (1 + math.exp(-10*(score-0.5))) for score in scores]
        probabilities = [score / sum(scores) for score in scores]
        print(commits)
        parent_commits = rng.choices(commits, probabilities, k=selfimprove_size)
    elif method == 'score_child_prop':
        # Choose parents based on score and the number of children
        commits = list(candidates.keys())
        scores = [candidates[commit]['accuracy_score'] for commit in commits]
        scores = [1 / (1 + math.exp(-10*(score-0.5))) for score in scores]
        children_counts = [candidates[commit]['children_count'] for commit in commits]
        children_counts = [1 / (1 + count) for count in children_counts]
        probabilities = [score * count for score, count in zip(scores, children_counts)]
        probabilities = [prob / sum(probabilities) for prob in probabilities]
        parent_commits = rng.choices(commits, probabilities, k=selfimprove_size)
    elif method == 'best':
        # Choose parents with the best score
        sorted_commits = sorted(candidates, key=lambda x: candidates[x]['accuracy_score'], reverse=True)
        parent_commits = sorted_commits[:min(selfimprove_size, len(sorted_commits))]
        if len(parent_commits) < selfimprove_size:
            parent_commits.extend(rng.choices(parent_commits, k=selfimprove_size - len(parent_commits)))
    else:
        # Choose parents randomly
        parent_commits = rng.choices(list(candidates.keys()), k=selfimprove_size)

    benchmark = get_benchmark(benchmark_name)
    is_polyglot = benchmark.kind == "polyglot"

    # Choose entries for each parent
    for parent_commit in parent_commits:
        empty_ids = candidates[parent_commit]['total_emptypatch_ids']
        resolved_ids = candidates[parent_commit]['total_resolved_ids']
        unresolved_ids = candidates[parent_commit]['total_unresolved_ids']
        
        if is_polyglot:
            entry_ids = empty_ids + unresolved_ids
            if not entry_ids:
                entry_ids = resolved_ids + empty_ids + unresolved_ids
        else:
            num_total_ids = len(empty_ids) + len(resolved_ids) + len(unresolved_ids)

            # Solve empty patches
            if len(empty_ids) >= 0.1 * num_total_ids and rng.random() < 0.25:
                entry = 'solve_empty_patches'
                selfimprove_entries.append((parent_commit, entry))
                continue

            # Solve stochasticity
            if rng.random() < 0.25:
                entry = 'solve_stochasticity'
                selfimprove_entries.append((parent_commit, entry))
                continue

            # Solve context length
            if any_exceeding_context_length(output_dir, parent_commit, empty_ids + unresolved_ids) and \
                rng.random() < 0.25:
                entry = 'solve_contextlength'
                selfimprove_entries.append((parent_commit, entry))
                continue

            # Choose a random unresolved entry
            if not unresolved_ids:
                continue
            entry_ids = unresolved_ids
        entry = rng.choice(entry_ids)
        selfimprove_entries.append((parent_commit, entry))

    return selfimprove_entries

def filter_compiled(run_ids, output_dir, expected_task_counts=None, logger=None):
    """
    Filter out runs that did not compile or have all empty patches.
    """
    run_ids_compiled = []

    logger.info(f"expected_task_counts: {expected_task_counts}")
    for run_id in run_ids:
        metadata_path = os.path.join(output_dir, run_id, "metadata.json")
        metadata = load_json_file(metadata_path)
        logger.info(f"{run_id} metadata: {metadata}")
        if is_compiled_self_improve(metadata, expected_task_counts=expected_task_counts, logger=logger):
            run_ids_compiled.append(run_id)
    return run_ids_compiled

def get_original_score(output_dir):
    """
    Get the original score from the initial version.
    """
    metadata = load_json_file(os.path.join(output_dir, "initial", "metadata.json"))
    return metadata["overall_performance"]["accuracy_score"]

def update_archive(output_dir, archive, new_ids, method='keep_all', noise_leeway=0.1):
    """
    Update the archive with the new self-improve runs.
    """
    if method == 'keep_better':
        # keep only better ones
        original_score = get_original_score(output_dir) - noise_leeway
        for run_id in new_ids:
            metadata = load_json_file(os.path.join(output_dir, run_id, "metadata.json"))
            score = metadata["overall_performance"]["accuracy_score"]
            if score >= original_score:
                archive.append(run_id)
    else:
        # keep everything
        archive += new_ids

    return archive

def get_full_eval_threshold(output_dir, archive, benchmark_name):
    """
    Get the threshold for full evaluation.
    """
    archive_scores = []
    num_full_eval = get_cumulative_stage_task_counts(benchmark_name)[-1]

    # Get original score
    original_score = get_original_score(output_dir)
    archive_scores.append(original_score)

    # Get scores from the archive
    for run_id in archive:
        metadata = load_json_file(os.path.join(output_dir, run_id, "metadata.json"))
        total_submitted_instances = metadata["overall_performance"]["total_submitted_instances"]
        # Skip if node did not have full evaluation
        if total_submitted_instances < num_full_eval * 0.9:
            continue
        score = metadata["overall_performance"]["accuracy_score"]
        archive_scores.append(score)

    # Get threshold, second highest score
    threshold = sorted(archive_scores, reverse=True)[1] if len(archive_scores) > 1 else archive_scores[0]
    # Ensure threshold is at least 0.4
    threshold = max(threshold, 0.4)

    return threshold


def parse_budget_list(raw_value):
    values = [int(part.strip()) for part in raw_value.split(',') if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of positive integers.")
    if sorted(values) != values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("Hyperband budgets must be positive integers in ascending order.")
    return values

def main():
    parser = argparse.ArgumentParser(description="Darwin Godel Machine!")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="swe_verified_mini",
        choices=sorted(BENCHMARKS),
        help="Benchmark configuration to use.",
    )
    parser.add_argument("--max_generation", type=int, default=80, help="Maximum number of evolution iterations.")
    parser.add_argument("--selfimprove_size", type=int, default=2, help="Number of self-improvements attempts per DGM generation.")
    parser.add_argument("--selfimprove_workers", type=int, default=2, help="Number of parallel workers for self-improvement attempts.")
    parser.add_argument(
        "--choose_selfimproves_method", type=str, default='score_child_prop',
        choices=['random', 'score_prop', 'score_child_prop', 'best'],
        help="Method to choose self-improve attempts.",
    )
    parser.add_argument("--continue_from", type=str, default=None, help="Directory to continue the run from.")
    parser.add_argument("--update_archive", type=str, default='keep_all', choices=['keep_better', 'keep_all'], help="Method to update the archive.")
    # self-improve arguments
    parser.add_argument("--num_benchmark_evals", type=int, default=1, help="Number of repeated benchmark evaluations to run for each self-improve attempt.")
    parser.add_argument('--post_improve_diagnose', default=False, action='store_true', help='Diagnose the self-improvement after evaluation')
    parser.add_argument("--shallow_eval", default=False, action='store_true', help="Run only the first-stage benchmark evaluation for each self-improve attempt.")
    parser.add_argument("--eval_noise", type=float, default=0.1, help="Noise leeway for evaluation.")
    parser.add_argument("--skip_final_eval", default=False, action='store_true', help="Do not run the final evaluation stage even if a node qualifies.")
    parser.add_argument("--scheduler", type=str, default="baseline", choices=["baseline", "hyperband", "asha"], help="Scheduler to use for child evaluation.")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed for reproducible parent/task selection.")
    parser.add_argument("--generation_task_budget_total", type=int, default=None, help="Optional total task-evaluation budget per generation.")
    parser.add_argument("--hyperband_eta", type=int, default=5, help="Hyperband reduction factor.")
    parser.add_argument("--hyperband_budgets", type=parse_budget_list, default=[2, 10, 50], help="Comma-separated cumulative task budgets for Hyperband, e.g. 2,10,50.")
    parser.add_argument("--hyperband_initial_children", type=int, default=None, help="Override the number of initial Hyperband children.")
    # baselines
    parser.add_argument("--run_baseline", type=str, default=None, choices=['no_selfimprove', 'no_darwin'], help="Baseline to run.")
    args = parser.parse_args()

    # Variables for this DGM run
    if not args.continue_from:
        run_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S_%f")
    else:
        run_id = os.path.basename(args.continue_from)
        
    output_dir = os.path.join("./output_dgm", run_id)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize
    benchmark = get_benchmark(args.benchmark)
    archive, start_gen_num = initialize_run(output_dir, args.benchmark, prevrun_dir=args.continue_from)

    stage_subset_names = benchmark.stage_subset_names
    stage_small_issues = load_benchmark_subset(args.benchmark, stage_subset_names[0])
    stage_medium_issues = load_benchmark_subset(args.benchmark, stage_subset_names[1]) if len(stage_subset_names) > 1 else None
    final_stage_issues = load_benchmark_subset(args.benchmark, stage_subset_names[2]) if len(stage_subset_names) > 2 else None
    expected_task_counts = get_cumulative_stage_task_counts(args.benchmark)

    if args.scheduler in ("hyperband", "asha") and args.benchmark != "swe_verified_mini":
        raise ValueError("Hyperband and ASHA schedulers are currently implemented for swe_verified_mini only.")

    # Set up logger
    logger = setup_logger(os.path.join(output_dir, "dgm_outer.log"))
    logger.info(f"Starting DGM run {run_id} with arguments: {vars(args)}")
    logger.info(f"Archive: {archive}")
    if args.scheduler == "baseline":
        scheduler = BaselineScheduler(
            args,
            benchmark,
            logger,
            stage_small_issues,
            stage_medium_issues,
            final_stage_issues,
            expected_task_counts=[expected_task_counts[0]] if args.shallow_eval else expected_task_counts,
        )
    elif args.scheduler == "hyperband":
        scheduler = HyperbandScheduler(args, benchmark, logger)
    else:  # asha
        scheduler = ASHAScheduler(args, benchmark, logger)

    # Run the DGM
    for gen_num in range(start_gen_num, args.max_generation):
        generation_seed = args.seed + gen_num
        generation_rng = random.Random(generation_seed)
        generation_start = time.time()
        generation_budget_total = args.generation_task_budget_total or (args.selfimprove_size * expected_task_counts[-1])

        # Choose self-improve attempts
        selfimprove_entries = choose_selfimproves(
            output_dir, archive, scheduler.get_generation_child_count(generation_seed),
            method=args.choose_selfimproves_method,
            run_baseline=args.run_baseline,
            benchmark_name=args.benchmark,
            rng=generation_rng,
        )
        logger.info(f"Self-improve entries for generation {gen_num}: {selfimprove_entries}")
        scheduler_result = scheduler.run_generation(
            output_dir,
            selfimprove_entries,
            generation_seed,
            None if args.skip_final_eval else get_full_eval_threshold(output_dir, archive, args.benchmark),
        )
        selfimprove_ids = scheduler_result["children"]
        selfimprove_ids_compiled = scheduler_result["children_compiled"]

        # Update archive
        logger.info(f"Updating archive for generation {gen_num}")
        archive = update_archive(
            output_dir,
            archive,
            scheduler_result["archive_candidates"],
            method=args.update_archive,
            noise_leeway=args.eval_noise,
        )
        generation_end = time.time()

        # Save DGM state
        with open(os.path.join(output_dir, "dgm_metadata.jsonl"), "a") as f:
            f.write(json.dumps({
                "benchmark_name": args.benchmark,
                "scheduler_name": args.scheduler,
                "seed": generation_seed,
                "generation": gen_num,
                "generation_start_time": generation_start,
                "generation_end_time": generation_end,
                "generation_wall_clock_seconds": generation_end - generation_start,
                "selfimprove_entries": selfimprove_entries,
                "children": selfimprove_ids,
                "children_compiled": selfimprove_ids_compiled,
                "archive_candidates": scheduler_result["archive_candidates"],
                "children_generated_count": len(selfimprove_ids),
                "children_compiled_count": scheduler_result["children_compiled_count"],
                "children_fully_evaluated_count": scheduler_result["children_fully_evaluated_count"],
                "best_child_score": scheduler_result["best_child_score"],
                "avg_child_score": scheduler_result["avg_child_score"],
                "archive_size": len(archive),
                "evaluation_budget_tasks_total": generation_budget_total,
                "evaluation_budget_tasks_consumed": scheduler_result["evaluation_budget_tasks_consumed"],
                "rungs": scheduler_result["rungs"],
                "archive": archive,
            }, indent=2) + "\n")

if __name__ == "__main__":
    main()

