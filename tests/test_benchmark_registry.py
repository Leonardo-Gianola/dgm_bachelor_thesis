from benchmarks.config import BENCHMARKS, get_cumulative_stage_task_counts, load_benchmark_subset


def test_mini_subsets_match_expected_sizes():
    assert len(load_benchmark_subset("swe_verified_mini", "stage_small")) == 3
    assert len(load_benchmark_subset("swe_verified_mini", "stage_medium")) == 12
    assert len(load_benchmark_subset("swe_verified_mini", "stage_full")) == 35
    assert len(load_benchmark_subset("swe_verified_mini", "full_mini_50")) == 50
    assert len(load_benchmark_subset("swe_verified_mini", "rung0_1")) == 1
    assert len(load_benchmark_subset("swe_verified_mini", "rung1_5")) == 5
    assert len(load_benchmark_subset("swe_verified_mini", "rung2_15")) == 15


def test_mini_cumulative_stage_sizes_are_pinned():
    assert get_cumulative_stage_task_counts("swe_verified_mini") == [3, 15, 50]


def test_benchmark_registry_exposes_legacy_paths():
    assert "swe_verified_mini" in BENCHMARKS
    assert "swe_verified_legacy" in BENCHMARKS
    assert "polyglot_legacy" in BENCHMARKS
