import argparse
import datetime
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

import docker

from benchmarks.config import load_benchmark_dataset, load_benchmark_subset
from prompts.testrepo_prompt import get_test_description

VENDORED_SWEBENCH_ROOT = Path(__file__).resolve().parents[1] / "swe_bench" / "SWE-bench"
if str(VENDORED_SWEBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDORED_SWEBENCH_ROOT))

from swebench.harness.docker_build import build_instance_images, build_container, cleanup_container
from swebench.harness.test_spec import make_test_spec

from swe_bench.utils import (
    copy_from_container,
    copy_to_container,
    log_container_output,
    remove_existing_container,
    safe_log,
    setup_logger,
)


REPO_ROOT_MARKERS = (
    "coding_agent.py",
    "requirements.txt",
    "llm.py",
    "llm_withtools.py",
    "prompts",
    "swe_bench",
)


def get_repo_root():
    repo_root = Path(__file__).resolve().parents[1]
    missing_markers = [marker for marker in REPO_ROOT_MARKERS if not (repo_root / marker).exists()]
    if missing_markers:
        raise FileNotFoundError(f"Could not resolve repo root from {repo_root}. Missing: {missing_markers}")
    return repo_root


def process_entry(entry, out_dname, model_name_or_path, model_patch_paths):
    instance_id = entry["instance_id"]
    problem_statement = entry["problem_statement"]
    base_commit = entry["base_commit"]
    chat_history_file = out_dname / f"{instance_id}.md"
    out_fname = out_dname / f"{instance_id}.json"
    eval_file = out_dname / f"{instance_id}_eval.sh"

    if out_fname.exists():
        print(f"Skipping existing entry {instance_id}")
        return {"success": True, "instance_id": instance_id}

    client = None
    container = None
    logger = None
    try:
        repo_root = get_repo_root()
        client = docker.from_env()
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        logger = setup_logger(str(out_dname / f"{instance_id}_docker.log"))
        nocache = True
        test_spec = make_test_spec(entry)
        container_name = test_spec.get_instance_container_name(run_id)
        remove_existing_container(client, container_name)
        container = build_container(test_spec, client, run_id, logger, nocache, force_rebuild=False)
        container.start()
        logger.info(f"Resolved repo root to: {repo_root}")

        copy_to_container(container, repo_root / "coding_agent.py", "/dgm/coding_agent.py")
        copy_to_container(container, repo_root / "requirements.txt", "/dgm/requirements.txt")
        copy_to_container(container, repo_root / "pytest.ini", "/dgm/pytest.ini")
        copy_to_container(container, repo_root / "tools", "/dgm/tools/")
        copy_to_container(container, repo_root / "utils", "/dgm/utils/")
        copy_to_container(container, repo_root / "tests", "/dgm/tests/")
        copy_to_container(container, repo_root / "prompts", "/dgm/prompts/")
        copy_to_container(container, repo_root / "llm.py", "/dgm/llm.py")
        copy_to_container(container, repo_root / "llm_withtools.py", "/dgm/llm_withtools.py")
        chat_history_file_container = f"/dgm/{chat_history_file.name}"

        logger.info("Setting up environment")
        eval_script = test_spec.eval_script
        eval_file.write_text(eval_script, encoding="utf-8")
        copy_to_container(container, eval_file, "/eval.sh")
        exec_result = container.exec_run("/bin/bash /eval.sh", workdir="/")
        log_container_output(exec_result)
        exec_result = container.exec_run("rm /eval.sh", workdir="/")
        log_container_output(exec_result)

        test_description = get_test_description(eval_script=eval_script, swerepo=True)

        if model_patch_paths:
            safe_log("Applying model patches")
            for model_patch_path in model_patch_paths:
                copy_to_container(container, model_patch_path, "/dgm/parent_patch.txt")
                exec_result = container.exec_run("/bin/sh -c 'patch -p1 < /dgm/parent_patch.txt'", workdir="/dgm")
                log_container_output(exec_result)
                exec_result = container.exec_run("rm /dgm/parent_patch.txt", workdir="/dgm")
                log_container_output(exec_result)

        safe_log("Installing more requirements")
        exec_result = container.exec_run("python -m pip install -r /dgm/requirements.txt", workdir="/")
        log_container_output(exec_result)

        env_vars = {
            "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
            "AWS_REGION": os.getenv("AWS_REGION"),
            "AWS_REGION_NAME": os.getenv("AWS_REGION_NAME"),
            "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
            "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
        }
        safe_log("Running the agent")
        cmd = [
            "timeout",
            "32400",
            "python",
            "/dgm/coding_agent.py",
            "--problem_statement",
            problem_statement,
            "--git_dir",
            "/testbed/",
            "--chat_history_file",
            chat_history_file_container,
            "--base_commit",
            base_commit,
            "--outdir",
            "/dgm/",
            "--test_description",
            test_description,
            "--instance_id",
            instance_id,
        ]
        exec_result = container.exec_run(cmd, environment=env_vars, workdir="/")
        log_container_output(exec_result)

        logger.info("Copying output files back to host")
        copy_from_container(container, chat_history_file_container, chat_history_file)
        exec_result = container.exec_run(f"find /dgm/ -name '{instance_id}_*.md'", workdir="/")
        chat_history_files_container = exec_result.output.decode().split()
        for chat_history_path_container in chat_history_files_container:
            extra_chat_history = out_dname / Path(chat_history_path_container).name
            copy_from_container(container, chat_history_path_container, extra_chat_history)

        logger.info("Getting model_patch")
        exec_result = container.exec_run("cat /dgm/model_patch.diff")
        log_container_output(exec_result)
        model_patch = exec_result.output.decode()

        proposed_model_patches = []
        exec_result = container.exec_run("find /dgm/ -name 'model_patch_*.diff'", workdir="/")
        model_patch_files_container = exec_result.output.decode().split()
        for model_patch_file_container in model_patch_files_container:
            exec_result = container.exec_run(f"cat {model_patch_file_container}")
            log_container_output(exec_result)
            proposed_model_patches.append(exec_result.output.decode())

        result = {
            "instance_id": instance_id,
            "model_name_or_path": model_name_or_path,
            "model_patch": model_patch,
            "proposed_model_patches": proposed_model_patches,
        }
        out_fname.write_text(json.dumps(result, indent=4), encoding="utf-8")
        return {"success": True, "instance_id": instance_id}
    except Exception as e:
        print(f"Error processing entry {instance_id}: {e}")
        return {"success": False, "instance_id": instance_id, "error": str(e)}
    finally:
        try:
            cleanup_container(client, container, logger)
        except Exception as cleanup_error:
            print(f"Error cleaning up Docker container for {instance_id}: {cleanup_error}")


def harness(
    test_task_list=None,
    num_samples=-1,
    max_workers=4,
    docker_build_workers=1,
    model_name_or_path=None,
    model_patch_paths=None,
    num_evals=1,
    num_evals_parallel=1,
    pred_dname="./benchmark_predictions",
    benchmark_name="swe_verified_mini",
):
    dataset = load_benchmark_dataset(benchmark_name)

    if model_name_or_path is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name_or_path = f"{timestamp}--claude-3-5-sonnet-20241022"
    pred_dname = Path(pred_dname)
    pred_dname.mkdir(exist_ok=True, parents=True)

    entries = list(dataset)
    if test_task_list:
        entries = [entry for entry in entries if entry["instance_id"] in test_task_list]
    if num_samples > 0:
        entries = entries[:num_samples]

    client = docker.from_env()
    build_instance_images(
        client,
        dataset=entries,
        force_rebuild=False,
        max_workers=docker_build_workers,
    )

    def process_evaluation(eval_idx):
        model_name_or_path_inst = f"{model_name_or_path}_{eval_idx}"
        out_dname = pred_dname / model_name_or_path_inst
        out_dname.mkdir(exist_ok=True)

        print(f"Starting evaluation {eval_idx} for model {model_name_or_path}")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_entry = {
                executor.submit(process_entry, entry, out_dname, model_name_or_path_inst, model_patch_paths): entry
                for entry in entries
            }
            for future in as_completed(future_to_entry):
                result = future.result()
                if result["success"]:
                    print(f"Successfully processed entry {result['instance_id']} for eval {eval_idx}")
                else:
                    print(
                        f"Failed to process entry {result['instance_id']} for eval {eval_idx}: "
                        f"{result.get('error', 'Unknown error')}"
                    )
        return out_dname

    with ThreadPoolExecutor(max_workers=num_evals_parallel) as eval_executor:
        out_dnames = list(eval_executor.map(process_evaluation, range(num_evals)))

    print(f"All evaluations completed for model {model_name_or_path}")
    return out_dnames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="swe_verified_mini", help="Benchmark configuration to use.")
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples to process.")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of concurrent threads.")
    parser.add_argument(
        "--docker_build_workers",
        type=int,
        default=1,
        help="Maximum number of concurrent Docker image builds.",
    )
    parser.add_argument("--model_name_or_path", type=str, default=None, help="Model name or path.")
    parser.add_argument("--model_patch_paths", type=str, default=None, help="Comma-separated model patch paths.")
    parser.add_argument("--num_evals", type=int, default=1, help="Repeated evaluations to run.")
    parser.add_argument("--num_evals_parallel", type=int, default=1, help="Parallel repeated evaluations.")
    parser.add_argument("--pred_dname", type=str, default="./benchmark_predictions", help="Output directory for predictions.")
    parser.add_argument("--subset", type=str, default=None, help="Named benchmark subset to process.")
    args = parser.parse_args()

    test_task_list = load_benchmark_subset(args.benchmark, args.subset) if args.subset else None
    model_patch_paths = args.model_patch_paths.split(",") if args.model_patch_paths else None

    harness(
        test_task_list=test_task_list,
        num_samples=args.num_samples,
        max_workers=args.max_workers,
        docker_build_workers=args.docker_build_workers,
        model_name_or_path=args.model_name_or_path,
        model_patch_paths=model_patch_paths,
        num_evals=args.num_evals,
        num_evals_parallel=args.num_evals_parallel,
        pred_dname=args.pred_dname,
        benchmark_name=args.benchmark,
    )


if __name__ == "__main__":
    main()
