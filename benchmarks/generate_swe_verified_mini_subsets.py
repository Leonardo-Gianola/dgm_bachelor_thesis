import json

from benchmarks.config import get_benchmark, load_reference_instance_ids


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main():
    benchmark = get_benchmark("swe_verified_mini")
    if benchmark.subset_dir is None:
        raise ValueError("Mini benchmark subset directory is not configured.")

    instance_ids = load_reference_instance_ids("swe_verified_mini")
    subset_payloads = {
        "stage_small.json": instance_ids[:3],
        "stage_medium.json": instance_ids[3:15],
        "stage_full_remaining.json": instance_ids[15:],
        "full_mini_50.json": instance_ids,
        "rung0_1.json": instance_ids[:1],
        "rung1_5.json": instance_ids[:5],
        "rung2_15.json": instance_ids[:15],
    }

    for file_name, payload in subset_payloads.items():
        write_json(benchmark.subset_dir / file_name, payload)

    manifest = {
        "benchmark_name": benchmark.name,
        "dataset_name": benchmark.dataset_name,
        "dataset_cache_path": str(benchmark.dataset_cache_path),
        "source_instance_ids_path": str(benchmark.dataset_instance_ids_path),
        "subset_sizes": {file_name: len(payload) for file_name, payload in subset_payloads.items()},
        "cumulative_stage_sizes": [3, 15, 50],
    }
    write_json(benchmark.subset_dir / "manifest.json", manifest)
    print(f"Generated subset files in {benchmark.subset_dir}")


if __name__ == "__main__":
    main()
