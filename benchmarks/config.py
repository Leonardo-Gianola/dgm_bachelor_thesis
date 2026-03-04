from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

from utils.common_utils import load_json_file


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    kind: str
    description: str
    dataset_name: str | None = None
    dataset_cache_relpath: str | None = None
    dataset_instance_ids_relpath: str | None = None
    subset_dir_relpath: str | None = None
    initial_archive_relpath: str | None = None
    stage_subset_names: tuple[str, ...] = ()
    default_manual_subset: str | None = None
    full_subset_name: str | None = None
    prediction_dir_name: str = "predictions"
    subset_files: dict[str, str] = field(default_factory=dict)
    legacy: bool = False

    @property
    def dataset_cache_path(self) -> Path | None:
        if not self.dataset_cache_relpath:
            return None
        return REPO_ROOT / self.dataset_cache_relpath

    @property
    def dataset_instance_ids_path(self) -> Path | None:
        if not self.dataset_instance_ids_relpath:
            return None
        return REPO_ROOT / self.dataset_instance_ids_relpath

    @property
    def subset_dir(self) -> Path | None:
        if not self.subset_dir_relpath:
            return None
        return REPO_ROOT / self.subset_dir_relpath

    @property
    def initial_archive_path(self) -> Path | None:
        if not self.initial_archive_relpath:
            return None
        return REPO_ROOT / self.initial_archive_relpath

    def resolve_subset_path(self, subset_name: str) -> Path:
        if self.subset_dir is None:
            raise ValueError(f"Benchmark {self.name} does not define subset files.")
        file_name = self.subset_files.get(subset_name)
        if file_name is None:
            raise KeyError(f"Unknown subset '{subset_name}' for benchmark {self.name}.")
        return self.subset_dir / file_name


BENCHMARKS: dict[str, BenchmarkConfig] = {
    "swe_verified_mini": BenchmarkConfig(
        name="swe_verified_mini",
        kind="swe_verified",
        description="Default benchmark. SWE-bench Verified Mini with 50 curated tasks.",
        dataset_name="MariusHobbhahn/swe-bench-verified-mini",
        dataset_cache_relpath="benchmarks/data/swe_verified_mini/test.json",
        dataset_instance_ids_relpath="benchmarks/data/swe_verified_mini/instance_ids.json",
        subset_dir_relpath="benchmarks/subsets/swe_verified_mini",
        initial_archive_relpath="initial_swe_verified_mini",
        stage_subset_names=("stage_small", "stage_medium", "stage_full"),
        default_manual_subset="stage_small",
        full_subset_name="full_mini_50",
        subset_files={
            "stage_small": "stage_small.json",
            "stage_medium": "stage_medium.json",
            "stage_full": "stage_full_remaining.json",
            "full_mini_50": "full_mini_50.json",
            "rung0_1": "rung0_1.json",
            "rung1_5": "rung1_5.json",
            "rung2_15": "rung2_15.json",
        },
    ),
    "swe_verified_legacy": BenchmarkConfig(
        name="swe_verified_legacy",
        kind="swe_verified",
        description="Legacy SWE-bench Verified workflow with the repo's historical subsets.",
        dataset_name="princeton-nlp/SWE-bench_Verified",
        subset_dir_relpath="swe_bench/subsets",
        initial_archive_relpath="initial",
        stage_subset_names=("stage_small", "stage_medium", "stage_full"),
        default_manual_subset="stage_small",
        subset_files={
            "stage_small": "small.json",
            "stage_medium": "medium.json",
            "stage_full": "big.json",
            "small": "small.json",
            "medium": "medium.json",
            "big": "big.json",
        },
        legacy=True,
    ),
    "polyglot_legacy": BenchmarkConfig(
        name="polyglot_legacy",
        kind="polyglot",
        description="Legacy Polyglot benchmark workflow.",
        dataset_cache_relpath="polyglot/polyglot_benchmark_metadata.json",
        subset_dir_relpath="polyglot/subsets",
        initial_archive_relpath="initial_polyglot",
        stage_subset_names=("stage_small", "stage_medium"),
        default_manual_subset="stage_small",
        subset_files={
            "stage_small": "small.json",
            "stage_medium": "medium.json",
            "small": "small.json",
            "medium": "medium.json",
        },
        legacy=True,
    ),
}


def get_benchmark(benchmark_name: str) -> BenchmarkConfig:
    try:
        return BENCHMARKS[benchmark_name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown benchmark '{benchmark_name}'. Choices: {', '.join(sorted(BENCHMARKS))}"
        ) from exc


def get_dataset_source(benchmark_name: str) -> str | None:
    benchmark = get_benchmark(benchmark_name)
    if benchmark.dataset_cache_path and benchmark.dataset_cache_path.exists():
        return str(benchmark.dataset_cache_path.resolve())
    return benchmark.dataset_name


def _load_json_dataset(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list dataset in {path}, got {type(data).__name__}.")
    return data


def load_benchmark_dataset(benchmark_name: str) -> list[dict[str, Any]]:
    benchmark = get_benchmark(benchmark_name)
    dataset_source = get_dataset_source(benchmark_name)
    if dataset_source is None:
        raise ValueError(f"Benchmark {benchmark_name} does not define a dataset source.")

    source_path = Path(dataset_source)
    if source_path.exists():
        return _load_json_dataset(source_path)

    dataset = load_dataset(dataset_source, split="test")
    return list(dataset)


def load_benchmark_subset(benchmark_name: str, subset_name: str) -> list[str]:
    benchmark = get_benchmark(benchmark_name)
    subset_path = benchmark.resolve_subset_path(subset_name)
    return load_json_file(str(subset_path))


def load_reference_instance_ids(benchmark_name: str) -> list[str]:
    benchmark = get_benchmark(benchmark_name)
    if benchmark.dataset_instance_ids_path and benchmark.dataset_instance_ids_path.exists():
        return load_json_file(str(benchmark.dataset_instance_ids_path))
    if benchmark.full_subset_name:
        return load_benchmark_subset(benchmark_name, benchmark.full_subset_name)
    dataset = load_benchmark_dataset(benchmark_name)
    return [entry["instance_id"] for entry in dataset]


def get_cumulative_stage_task_counts(benchmark_name: str) -> list[int]:
    benchmark = get_benchmark(benchmark_name)
    counts: list[int] = []
    total = 0
    for subset_name in benchmark.stage_subset_names:
        total += len(load_benchmark_subset(benchmark_name, subset_name))
        counts.append(total)
    return counts
