import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess

from benchmarks.config import get_dataset_source


def load_predictions(paths):
    prediction_paths = []
    for path in paths:
        path = Path(path)
        if path.is_file():
            prediction_paths.append(path)
        elif path.is_dir():
            prediction_paths += list(path.glob("*.json"))
        else:
            assert False, path

    predictions = {}
    for fname in prediction_paths:
        pred = json.loads(fname.read_text())
        if "instance_id" not in pred:
            print("Skipping json without instance_id", fname)
            continue

        inst = pred["instance_id"]
        pred["json_fname"] = str(fname)
        predictions[inst] = pred

    return predictions


def remove_patches_to_tests(model_patch):
    lines = model_patch.splitlines(keepends=True)
    filtered_lines = []
    is_tests = False

    for line in lines:
        if line.startswith("diff --git a/"):
            pieces = line.split()
            to = pieces[-1]
            if to.startswith("b/") and (
                "/test/" in to
                or "/tests/" in to
                or "/testing/" in to
                or "/test_" in to
                or "/tox.ini" in to
            ):
                is_tests = True
            else:
                is_tests = False

        if not is_tests:
            filtered_lines.append(line)

    return "".join(filtered_lines)


def preds_to_jsonl(dname, predictions):
    dname = Path(dname)

    predictions_jsonl = str(dname / "all_preds.jsonl")
    model_name_or_path = list(predictions.values())[0]["model_name_or_path"]
    with open(predictions_jsonl, "w", encoding="utf-8") as fh:
        for pred in predictions.values():
            assert model_name_or_path == pred["model_name_or_path"]
            minimal_pred = {
                "model_name_or_path": model_name_or_path,
                "model_patch": remove_patches_to_tests(pred["model_patch"]),
                "instance_id": pred["instance_id"],
            }
            fh.write(json.dumps(minimal_pred) + "\n")
    return predictions_jsonl


def run_evals(predictions_jsonl, run_id, dataset_name, root_dir, output_dir, num_eval_procs=5):
    run_evals_cmd = [
        "python",
        os.path.join(root_dir, "./swe_bench/SWE-bench/swebench/harness/run_evaluation.py"),
        "--dataset_name",
        dataset_name,
        "--predictions_path",
        predictions_jsonl,
        "--max_workers",
        str(num_eval_procs),
        "--run_id",
        run_id,
    ]
    subprocess.run(run_evals_cmd, check=True, cwd=output_dir)


def make_report(
    dnames,
    run_ids=None,
    benchmark_name="swe_verified_mini",
    dataset_name=None,
    output_dir=".",
    dnames_workers=None,
    num_eval_procs=5,
):
    root_dir = os.path.abspath(os.getcwd())
    output_dir = os.path.join(root_dir, output_dir)
    dataset_name = dataset_name or get_dataset_source(benchmark_name)
    if dataset_name is None:
        raise ValueError(f"Unable to resolve dataset source for benchmark {benchmark_name}.")

    def process_single_dname(dname, run_id):
        dname = Path(os.path.join(root_dir, dname))
        predictions = load_predictions([dname])
        predictions_jsonl = preds_to_jsonl(dname, predictions)
        run_evals(predictions_jsonl, run_id, dataset_name, root_dir, output_dir, num_eval_procs=num_eval_procs)
        print(f"Report generated for {dname}")

    if run_ids is None or len(run_ids) != len(dnames):
        run_ids = [f"{i:03}" for i in range(len(dnames))]
    if dnames_workers is None:
        dnames_workers = len(dnames)
    with ThreadPoolExecutor(max_workers=dnames_workers) as executor:
        list(executor.map(process_single_dname, dnames, run_ids))

    print("All reports generated.")


def main():
    parser = argparse.ArgumentParser(description="Run evaluations on prediction directories.")
    parser.add_argument("--dnames", type=str, nargs="+", help="Prediction directories to evaluate.")
    parser.add_argument("--run_ids", type=str, nargs="+", default=None, help="Run IDs for each directory.")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="swe_verified_mini",
        choices=sorted(["swe_verified_mini", "swe_verified_legacy"]),
        help="Benchmark configuration to use.",
    )
    parser.add_argument("--dataset_name", type=str, default=None, help="Override dataset name or local dataset path.")
    parser.add_argument("--dnames_workers", type=int, default=None, help="Parallel workers across prediction dirs.")
    parser.add_argument("--num_eval_procs", type=int, default=5, help="Parallel workers inside SWE-bench evaluation.")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory for reports.")
    args = parser.parse_args()

    make_report(
        args.dnames,
        run_ids=args.run_ids,
        benchmark_name=args.benchmark,
        dataset_name=args.dataset_name,
        dnames_workers=args.dnames_workers,
        num_eval_procs=args.num_eval_procs,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
