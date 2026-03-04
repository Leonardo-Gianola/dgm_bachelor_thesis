import argparse
import datetime
import json
import os
import shutil
import docker

from benchmarks.config import BENCHMARKS, get_benchmark, get_dataset_source, load_benchmark_dataset
from benchmarks.swe_verified_harness import harness as verified_harness
from benchmarks.swe_verified_report import make_report as make_verified_report
from llm import create_client, get_response_from_llm, extract_json_between_markers
from llm_withtools import OPENAI_MODEL
from prompts.self_improvement_prompt import get_diagnose_prompt_polyglot, get_diagnose_prompt_swe, get_problem_description_prompt
from prompts.diagnose_improvement_prompt import get_diagnose_improvement_prompt
from prompts.testrepo_prompt import get_test_description
from polyglot.harness import harness as polyglot_harness
from utils.evo_utils import get_model_patch_paths, get_all_performance, is_compiled_self_improve
from utils.docker_utils import (
    build_dgm_container,
    cleanup_container,
    copy_from_container,
    copy_to_container,
    log_container_output,
    remove_existing_container,
    setup_logger,
    safe_log,
)

dataset = None
diagnose_model = OPENAI_MODEL

def diagnose_problem(entry, commit, root_dir, out_dir, patch_files=None, max_attempts=3, benchmark_name="swe_verified_mini"):
    patch_files = patch_files or []
    benchmark = get_benchmark(benchmark_name)
    client = create_client(diagnose_model)
    if benchmark.kind == "polyglot":
        diagnose_sys_message, diagnose_prompt = get_diagnose_prompt_polyglot(
            entry, commit, root_dir, out_dir, dataset,
            patch_files=patch_files,
        )
    else:
        diagnose_sys_message, diagnose_prompt = get_diagnose_prompt_swe(
            entry, commit, root_dir, out_dir, dataset,
            patch_files=patch_files,
        )
    try:
        response, msg_history = get_response_from_llm(
            msg=diagnose_prompt,
            client=client[0],
            model=client[1],
            system_message=diagnose_sys_message,
            print_debug=False,
            msg_history=None,
        )
        safe_log(f"Message history: {msg_history}")
        response_json = extract_json_between_markers(response)
        assert response_json, "empty response json"
        problem_statement = get_problem_description_prompt(response_json, benchmark.kind == "polyglot")
    except Exception as e:
        # Exception most probably due to not having json in the response
        safe_log(f"Error while diagnosing the problem: {e}")
        if max_attempts > 0:
            return diagnose_problem(
                entry, commit, root_dir, out_dir,
                patch_files=patch_files,
                max_attempts=max_attempts-1,
                benchmark_name=benchmark_name,
            )
        else:
            return None
    return problem_statement

def diagnose_improvement(
        entry, parent_commit, root_dir, model_patch_file, out_dir, run_id,
        patch_files=[], max_attempts=3,
    ):
    """
    Diagnose the improvement of the model patch.

    Args:
        entry (str): The task entry to improve.
        parent_commit (str): The commit hash of the parent commit.
        root_dir (str): The root directory of the repository.
        model_patch_file (str): The path to the model patch file.
        out_dir (str): The output directory.
        run_id (str): The run id of the self-improvement attempt.
        patch_files (list): The list of patch files before self-improvement.
        max_attempts (int): The maximum number of attempts to diagnose the improvement.
    
    Returns:
        dict: The improvement diagnosis.
    """
    client = create_client(diagnose_model)
    diagnose_sys_message, diagnose_prompt = get_diagnose_improvement_prompt(
        entry, parent_commit, root_dir, model_patch_file, out_dir, run_id, dataset,
        patch_files=patch_files,
    )
    safe_log(f"Diagnosing the improvement: {diagnose_prompt}")
    try:
        response, msg_history = get_response_from_llm(
            msg=diagnose_prompt,
            client=client[0],
            model=client[1],
            system_message=diagnose_sys_message,
            print_debug=False,
            msg_history=None,
        )
        safe_log(f"Message history: {msg_history}")
        response_json = extract_json_between_markers(response)
        assert response_json, "empty response json"
        improvement_diagnosis = response_json
    except Exception as e:
        # Exception most probably due to not having json in the response
        safe_log(f"Error while diagnosing the improvement: {e}")
        if max_attempts > 0:
            return diagnose_improvement(
                entry, parent_commit, root_dir, model_patch_file, out_dir, run_id,
                patch_files=patch_files, max_attempts=max_attempts-1,
            )
        else:
            return None
    return improvement_diagnosis

def save_metadata(metadata, output_dir):
    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)

def _apply_benchmark_metadata(
        metadata,
        benchmark_name,
        output_dir,
        model_name_or_path,
        evaluation_dirs,
        evaluated_subset_names,
        budget_name,
        budget_size,
    ):
    performances, overall_performance = get_all_performance(model_name_or_path, results_dir=output_dir)
    metadata['evaluation_dirs'] = [str(dn) for dn in evaluation_dirs]
    metadata['benchmark_prediction_dirs'] = metadata['evaluation_dirs']
    metadata['benchmark_name'] = benchmark_name
    metadata['dataset_source'] = get_dataset_source(benchmark_name)
    metadata['overall_performance'] = overall_performance
    metadata['benchmark_performance'] = overall_performance
    metadata['evaluated_subset_names'] = evaluated_subset_names
    metadata['evaluated_task_count'] = overall_performance.get('total_submitted_instances', 0) if overall_performance else 0
    metadata['budget_name'] = budget_name
    metadata['budget_size'] = budget_size
    return performances, overall_performance


def run_harness_verified(
        benchmark_name, entry, model_name_or_path, patch_files, num_evals, output_dir, metadata, run_id,
        test_more_threshold, test_task_list, test_task_list_more,
        full_eval_threshold=None, final_eval_task_list=None,
        stage_subset_name="stage_small", more_subset_name="stage_medium", final_subset_name="stage_full",
    ):
    all_evaluation_dirs = []
    evaluated_subset_names = []
    safe_log('Start harness')
    test_task_list = [entry] if test_task_list is None else test_task_list
    dnames = verified_harness(
        test_task_list=test_task_list,
        num_samples=-1,
        max_workers=min(5, len(test_task_list)),
        model_name_or_path=model_name_or_path,
        model_patch_paths=patch_files,
        num_evals=num_evals,
        num_evals_parallel=5,
        pred_dname=os.path.join(output_dir, "predictions"),
        benchmark_name=benchmark_name,
    )
    all_evaluation_dirs.extend(dnames)
    evaluated_subset_names.append(stage_subset_name)
    safe_log('Start make_report')
    make_verified_report(
        dnames,
        run_ids=[f"{run_id}_{i}" for i in range(len(dnames))],
        benchmark_name=benchmark_name,
        dataset_name=get_dataset_source(benchmark_name),
        output_dir=output_dir,
        dnames_workers=5,
    )
    safe_log('Start get_performance')
    performances, overall_performance = _apply_benchmark_metadata(
        metadata,
        benchmark_name,
        output_dir,
        model_name_or_path,
        all_evaluation_dirs,
        evaluated_subset_names,
        stage_subset_name,
        len(test_task_list),
    )
    safe_log("End of evaluation")

    # Check if additional evaluation should be run
    if (overall_performance and \
        test_more_threshold is not None and test_task_list_more is not None and \
            overall_performance.get('total_resolved_instances', 0) >= len(test_task_list) * test_more_threshold):
        safe_log("Start additional evaluation cycle")
        dnames = verified_harness(
            test_task_list=test_task_list_more,
            num_samples=-1,
            max_workers=min(5, len(test_task_list_more)),
            model_name_or_path=model_name_or_path,
            model_patch_paths=patch_files,
            num_evals=num_evals,
            num_evals_parallel=5,
            pred_dname=os.path.join(output_dir, "predictions"),
            benchmark_name=benchmark_name,
        )
        all_evaluation_dirs.extend(dnames)
        evaluated_subset_names.append(more_subset_name)
        safe_log('Start make_report more')
        make_verified_report(
            dnames,
            run_ids=[f"{run_id}_{i}" for i in range(len(dnames))],
            benchmark_name=benchmark_name,
            dataset_name=get_dataset_source(benchmark_name),
            output_dir=output_dir,
            dnames_workers=5,
        )
        safe_log('Start get_performance')
        cumulative_budget = len(test_task_list) + len(test_task_list_more)
        performances, overall_performance = _apply_benchmark_metadata(
            metadata,
            benchmark_name,
            output_dir,
            model_name_or_path,
            all_evaluation_dirs,
            evaluated_subset_names,
            more_subset_name,
            cumulative_budget,
        )
        safe_log("End of evaluation more")

    # Run the final evaluation once a run looks competitive.
    if (
        overall_performance and
        full_eval_threshold is not None and
        final_eval_task_list is not None and
        overall_performance.get('accuracy_score', 0) >= full_eval_threshold
    ):
        safe_log("Start full evaluation cycle")
        dnames = verified_harness(
            test_task_list=final_eval_task_list,
            num_samples=-1,
            max_workers=min(5, len(final_eval_task_list)),
            model_name_or_path=model_name_or_path,
            model_patch_paths=patch_files,
            num_evals=num_evals,
            num_evals_parallel=5,
            pred_dname=os.path.join(output_dir, "predictions"),
            benchmark_name=benchmark_name,
        )
        all_evaluation_dirs.extend(dnames)
        evaluated_subset_names.append(final_subset_name)
        safe_log('Start make_report full')
        make_verified_report(
            dnames,
            run_ids=[f"{run_id}_{i}" for i in range(len(dnames))],
            benchmark_name=benchmark_name,
            dataset_name=get_dataset_source(benchmark_name),
            output_dir=output_dir,
            dnames_workers=5,
        )
        safe_log('Start get_performance full')
        cumulative_budget = len(test_task_list)
        if test_task_list_more is not None:
            cumulative_budget += len(test_task_list_more)
        cumulative_budget += len(final_eval_task_list)
        performances, overall_performance = _apply_benchmark_metadata(
            metadata,
            benchmark_name,
            output_dir,
            model_name_or_path,
            all_evaluation_dirs,
            evaluated_subset_names,
            final_subset_name,
            cumulative_budget,
        )
        safe_log("End of full evaluation")

def run_harness_polyglot(
        benchmark_name, entry, model_name_or_path, patch_files, num_evals, output_dir, metadata, run_id,
        test_more_threshold, test_task_list, test_task_list_more,
        stage_subset_name="stage_small", more_subset_name="stage_medium",
    ):
    all_evaluation_dirs = []
    evaluated_subset_names = []
    safe_log('Start harness')
    test_task_list = [entry] if test_task_list is None else test_task_list
    safe_log(f'workers {min(10, len(test_task_list))}')
    dnames = polyglot_harness(
        test_task_list=test_task_list,
        num_samples=-1,
        max_workers=min(10, len(test_task_list)),
        model_name_or_path=model_name_or_path,
        model_patch_paths=patch_files,
        num_evals=num_evals,
        num_evals_parallel=min(5, num_evals),
        pred_dname=os.path.join(output_dir, "predictions"),
        output_dir=output_dir
    )
    all_evaluation_dirs.extend(dnames)
    evaluated_subset_names.append(stage_subset_name)
    safe_log('Start get_performance')
    performances, overall_performance = _apply_benchmark_metadata(
        metadata,
        benchmark_name,
        output_dir,
        model_name_or_path,
        all_evaluation_dirs,
        evaluated_subset_names,
        stage_subset_name,
        len(test_task_list),
    )
    safe_log("End of evaluation")

    # Check if additional evaluation should be run
    if (overall_performance and \
        test_more_threshold is not None and test_task_list_more is not None and \
            overall_performance.get('total_resolved_instances', 0) >= len(test_task_list) * test_more_threshold):
        safe_log("Start additional evaluation cycle")
        dnames = polyglot_harness(
            test_task_list=test_task_list_more,
            num_samples=-1,
            max_workers=50,
            model_name_or_path=model_name_or_path,
            model_patch_paths=patch_files,
            num_evals=num_evals,
            num_evals_parallel=min(5, num_evals),
            pred_dname=os.path.join(output_dir, "predictions"),
            output_dir=output_dir
        )
        all_evaluation_dirs.extend(dnames)
        evaluated_subset_names.append(more_subset_name)
        safe_log('Start get_performance')
        performances, overall_performance = _apply_benchmark_metadata(
            metadata,
            benchmark_name,
            output_dir,
            model_name_or_path,
            all_evaluation_dirs,
            evaluated_subset_names,
            more_subset_name,
            len(test_task_list) + len(test_task_list_more),
        )
        metadata['overall_performance_deep'] = overall_performance
        safe_log("End of evaluation more")

def self_improve(
    parent_commit='initial',  # 'initial' if starting from original dgm, else the run_id
    output_dir='output_selfimprove/',
    force_rebuild=False,
    num_evals=1,
    post_improve_diagnose=True,
    entry=None,
    test_task_list=None,  # None means the entry above only
    # Additional evaluation parameters
    test_more_threshold=None,
    test_task_list_more=None,
    full_eval_threshold=None,
    final_eval_task_list=None,
    # Run baseline
    run_baseline=None,
    benchmark_name='swe_verified_mini',
    search_strategy='dgm',
    rung=None,
):  

    global dataset
    benchmark = get_benchmark(benchmark_name)
    dataset = load_benchmark_dataset(benchmark_name)

    # Variables for this self-improvement attempt
    metadata = {}
    root_dir = os.path.abspath('./')  # root_dir should be /dgm
    run_id = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out_dir_base = output_dir  # out_dir_base should be /dgm/output_selfimprove/ or /dgm/output_dgm/{dgm_run_id}/
    output_dir = os.path.join(root_dir, f"{output_dir}/{run_id}/")
    os.makedirs(output_dir, exist_ok=True)
    metadata['run_id'] = run_id
    metadata['parent_commit'] = parent_commit
    metadata['benchmark_name'] = benchmark_name
    metadata['dataset_source'] = get_dataset_source(benchmark_name)
    metadata['search_strategy'] = search_strategy
    metadata['rung'] = rung

    # Set up logger
    logger = setup_logger(os.path.join(output_dir, "self_improve.log"))

    # Create and start the Docker container
    image_name = "dgm"
    container_name = f"dgm-container-{run_id}"
    client = docker.from_env()
    container = None
    container_cleaned = False
    patch_files = get_model_patch_paths(root_dir, os.path.join(output_dir, '../'), parent_commit)
    try:
        # Remove any existing container with the same name
        remove_existing_container(client, container_name)
        # Now create and start the container
        container = build_dgm_container(
            client, root_dir, image_name, container_name,
            force_rebuild=force_rebuild,
        )
        if container is None:
            raise RuntimeError("Failed to start container")

        if benchmark.kind == "polyglot":
            # remove the swe version of coding_agent.py
            exec_result = container.exec_run("rm /dgm/coding_agent.py", workdir='/')
            log_container_output(exec_result)
            # rename coding_agent_polyglot.py to coding_agent.py
            exec_result = container.exec_run("mv /dgm/coding_agent_polyglot.py /dgm/coding_agent.py", workdir='/')
            log_container_output(exec_result)
            # remove swe-specific files utils/eval_utils.py and utils/swe_log_parsers.py
            exec_result = container.exec_run("rm /dgm/utils/eval_utils.py", workdir='/')
            log_container_output(exec_result)
            exec_result = container.exec_run("rm /dgm/utils/swe_log_parsers.py", workdir='/')
            log_container_output(exec_result)
        else:
            # remove the polyglot version of coding_agent.py
            exec_result = container.exec_run("rm /dgm/coding_agent_polyglot.py", workdir='/')

        if run_baseline not in ['no_selfimprove']:
            for patch_file in patch_files:
                copy_to_container(container, patch_file, '/dgm/parent_patch.txt')
                exec_result = container.exec_run("/bin/sh -c 'patch -p1 < /dgm/parent_patch.txt'", workdir='/dgm')
                log_container_output(exec_result)
                exec_result = container.exec_run("rm /dgm/parent_patch.txt", workdir='/dgm')
                log_container_output(exec_result)

        # Commit this version of dgm, so that irrelevant changes are not included in the patch
        exec_result = container.exec_run("git add --all", workdir='/dgm/')
        log_container_output(exec_result)
        exec_result = container.exec_run("git -c user.name='user' -c user.email='you@example.com' commit -m 'a nonsense commit message'", workdir='/dgm/')
        log_container_output(exec_result)
        exec_result = container.exec_run("git rev-parse HEAD", workdir='/dgm/')
        log_container_output(exec_result)
        commit_hash = exec_result.output.decode('utf-8').strip()

        # Install requirements again in case of any changes
        exec_result = container.exec_run("python -m pip install -r /dgm/requirements.txt", workdir='/')
        log_container_output(exec_result)

        # Get tasks to improve
        if entry:
            safe_log(f"Task to improve: {entry}")
            problem_statement = diagnose_problem(
                entry,
                parent_commit,
                root_dir,
                out_dir_base,
                patch_files=patch_files,
                benchmark_name=benchmark_name,
            )
            safe_log(f"problem_statement: {problem_statement}")
        else:
            safe_log("No entry provided. Exiting.")
            save_metadata(metadata, output_dir)
            return metadata

        metadata['entry'] = entry
        metadata['problem_statement'] = problem_statement
        # If problem statement is not found, exit
        if not problem_statement:
            safe_log("Failed to diagnose the problem statement. Exiting.")
            save_metadata(metadata, output_dir)
            return metadata

        # Run self-improvement
        safe_log("Running self-improvement")
        chat_history_file_container = "/dgm/self_evo.md"
        test_description = get_test_description(swerepo=False)
        env_vars = {
            "ANTHROPIC_API_KEY": os.getenv('ANTHROPIC_API_KEY'),
            "AWS_REGION": os.getenv('AWS_REGION'),
            "AWS_REGION_NAME": os.getenv('AWS_REGION_NAME'),
            "AWS_ACCESS_KEY_ID": os.getenv('AWS_ACCESS_KEY_ID'),
            "AWS_SECRET_ACCESS_KEY": os.getenv('AWS_SECRET_ACCESS_KEY'),
            "OPENAI_API_KEY": os.getenv('OPENAI_API_KEY'),
            "OPENROUTER_API_KEY": os.getenv('OPENROUTER_API_KEY'),
        }
        cmd = [
            "timeout", "1800",  # 30min timeout
            "python", "/dgm/coding_agent.py",
            "--problem_statement", problem_statement,
            "--git_dir", "/dgm/",
            "--chat_history_file", chat_history_file_container,
            "--base_commit", commit_hash,
            "--outdir", "/dgm/",
            "--test_description", test_description,
            "--self_improve",
        ]
        exec_result = container.exec_run(cmd, environment=env_vars, workdir='/')
        log_container_output(exec_result)

        # Copy output files back to host
        chat_history_file = os.path.join(output_dir, "self_evo.md")
        copy_from_container(container, chat_history_file_container, chat_history_file)
        model_patch_file = os.path.join(output_dir, "model_patch.diff")
        copy_from_container(container, "/dgm/model_patch.diff", model_patch_file)

        # Try reading the patch file to validate it
        try:
            # Check if patch file exists and is not empty
            if not os.path.exists(model_patch_file):
                raise Exception("Model patch file is empty or does not exist")
            with open(model_patch_file, 'r') as f:
                patch_content = f.read()
                if not patch_content.strip():
                    raise Exception("Model patch file is empty")
        except Exception as e:
            safe_log(f"Failed to read model patch file: {str(e)}")
            save_metadata(metadata, output_dir)
            return metadata

        patch_files.append(model_patch_file)
    finally:
        if container is not None and not container_cleaned:
            cleanup_container(container)
            container_cleaned = True

    # Evaluate the performance of the self-improvement
    model_patch_exists = os.path.exists(model_patch_file)
    metadata['model_patch_exists'] = model_patch_exists
    model_patch_notempty = os.path.getsize(model_patch_file) > 0
    metadata['model_patch_notempty'] = model_patch_notempty
    model_name_or_path = run_id
    if model_patch_exists and model_patch_notempty:
        try:
            if benchmark.kind != "polyglot":
                run_harness_verified(
                    benchmark_name,
                    entry, model_name_or_path, patch_files, num_evals, output_dir, metadata, run_id,
                    test_more_threshold, test_task_list, test_task_list_more,
                    full_eval_threshold=full_eval_threshold,
                    final_eval_task_list=final_eval_task_list,
                )
            else:
                run_harness_polyglot(
                    benchmark_name,
                    entry, model_name_or_path, patch_files, num_evals, output_dir, metadata, run_id,
                    test_more_threshold, test_task_list, test_task_list_more,
                )
        except Exception as e:
            safe_log(f"Error while evaluating the self-improvement: {e}")

    # Post-self-improvement diagnosis
    if post_improve_diagnose:
        safe_log("Diagnosing the self-improvement")
        metadata['is_compiled'] = is_compiled_self_improve(metadata)
        if metadata['is_compiled']:
            safe_log("The self-improvement succeed to be complied")
            improvement_diagnosis = diagnose_improvement(
                entry, parent_commit, root_dir,
                model_patch_file, out_dir_base, run_id,
                patch_files=patch_files,
            )
            metadata['improvement_diagnosis'] = improvement_diagnosis
            safe_log(f"Improvement diagnosis: {improvement_diagnosis}")
        else:
            safe_log("The self-improvement fail to be complied")
            metadata['improvement_diagnosis'] = "Fail to complied. Ignore this."

    # Save metadata of this self-improvement attempt
    save_metadata(metadata, output_dir)
    return metadata

def main():
    parser = argparse.ArgumentParser(description="Self-improvement step for the repository.")
    parser.add_argument('--parent_commit', default="initial", type=str, help='Current commit to find the eval results, "initial" if starting from original dgm, else the run_id')
    parser.add_argument('--output_dir', default="./output_selfimprove", type=str, help='Directory to store the output')
    parser.add_argument('--force_rebuild', default=False, action='store_true', help='Force rebuild of the Docker image')
    parser.add_argument('--benchmark', default="swe_verified_mini", choices=sorted(BENCHMARKS), help='Benchmark configuration to use')
    parser.add_argument('--num_evals', default=1, type=int, help='Repeated number of benchmark evaluations after self-improvement')
    parser.add_argument('--no_post_improve_diagnose', default=False, action='store_true', help='Skip diagnosing the self-improvement after evaluation')
    parser.add_argument('--entry', default="django__django-10999", type=str, help='Task entry to improve')
    parser.add_argument('--test_task_list', default=None, type=str, help='List of tasks to evaluate the self-improvement')
    args = parser.parse_args()

    # Copy cached initial version into experiment dir
    benchmark = get_benchmark(args.benchmark)
    if benchmark.initial_archive_path and benchmark.initial_archive_path.exists():
        shutil.copytree(benchmark.initial_archive_path, os.path.join(args.output_dir, "initial"), dirs_exist_ok=True)

    metadata = self_improve(
        parent_commit=args.parent_commit,
        output_dir=args.output_dir,
        force_rebuild=args.force_rebuild,
        num_evals=args.num_evals,
        post_improve_diagnose=not args.no_post_improve_diagnose,
        entry=args.entry,
        test_task_list=args.test_task_list,
        benchmark_name=args.benchmark,
    )

if __name__ == "__main__":
    main()
