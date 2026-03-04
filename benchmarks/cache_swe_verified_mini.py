import json

from benchmarks.config import get_benchmark, load_reference_instance_ids


def main():
    benchmark = get_benchmark("swe_verified_mini")
    if benchmark.dataset_name is None or benchmark.dataset_cache_path is None:
        raise ValueError("Mini benchmark must define both a remote dataset name and a cache path.")

    from datasets import load_dataset

    dataset = list(load_dataset(benchmark.dataset_name, split="test"))
    instance_ids = [entry["instance_id"] for entry in dataset]
    expected_instance_ids = load_reference_instance_ids("swe_verified_mini")
    if instance_ids != expected_instance_ids:
        raise ValueError(
            "Downloaded dataset instance IDs do not match the pinned mini benchmark IDs.\n"
            f"Expected {len(expected_instance_ids)} IDs, got {len(instance_ids)}."
        )

    cache_path = benchmark.dataset_cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    print(f"Cached {len(dataset)} SWE-bench Verified Mini tasks to {cache_path}")


if __name__ == "__main__":
    main()
