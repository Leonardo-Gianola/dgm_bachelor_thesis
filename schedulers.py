import math
import os
import random
import time
import traceback
import json
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from benchmarks.config import load_reference_instance_ids
from self_improve_step import evaluate_existing_child, finalize_child_metadata, generate_child_patch
from utils.common_utils import load_json_file
from utils.evo_utils import is_compiled_self_improve


GENERATION_TIMEOUT_SECONDS = 1.5 * 60 * 60


def _load_child_metadata(output_dir, run_id):
    return load_json_file(os.path.join(output_dir, run_id, "metadata.json"))


def _save_child_metadata(output_dir, run_id, metadata):
    with open(os.path.join(output_dir, run_id, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)


def _child_score(metadata):
    return (metadata.get("overall_performance") or {}).get("accuracy_score", 0)


def _child_tokens(metadata):
    return metadata.get("token_usage_total", 0)


def _rank_children(run_ids, output_dir):
    def sort_key(run_id):
        metadata = _load_child_metadata(output_dir, run_id)
        return (-_child_score(metadata), _child_tokens(metadata), run_id)

    return sorted(run_ids, key=sort_key)


def _summarize_children(run_ids, output_dir, full_budget_size):
    child_scores = []
    compiled_count = 0
    fully_evaluated_count = 0
    for run_id in run_ids:
        metadata = _load_child_metadata(output_dir, run_id)
        score = _child_score(metadata)
        child_scores.append(score)
        if is_compiled_self_improve(metadata):
            compiled_count += 1
        if metadata.get("evaluated_task_count", 0) >= full_budget_size:
            fully_evaluated_count += 1
    return {
        "best_child_score": max(child_scores) if child_scores else 0,
        "avg_child_score": sum(child_scores) / len(child_scores) if child_scores else 0,
        "children_compiled_count": compiled_count,
        "children_fully_evaluated_count": fully_evaluated_count,
    }


def _promote_top_k(run_ids, output_dir, k, require_positive=False):
    if k <= 0:
        return []
    ranked = _rank_children(run_ids, output_dir)
    promoted = []
    for run_id in ranked:
        metadata = _load_child_metadata(output_dir, run_id)
        if require_positive and _child_score(metadata) <= 0:
            continue
        promoted.append(run_id)
        if len(promoted) >= k:
            break
    return promoted


def _mark_promotion_decisions(output_dir, input_ids, promoted_ids, rung_index):
    promoted_set = set(promoted_ids)
    for run_id in input_ids:
        metadata = _load_child_metadata(output_dir, run_id)
        metadata["promotion_decision"] = "promoted" if run_id in promoted_set else "killed"
        if run_id in promoted_set:
            metadata["promoted_from_rung"] = rung_index
        _save_child_metadata(output_dir, run_id, metadata)


class BaseScheduler:
    def __init__(self, args, benchmark, logger):
        self.args = args
        self.benchmark = benchmark
        self.logger = logger

    def get_generation_child_count(self, generation_seed):
        raise NotImplementedError

    def run_generation(self, output_dir, selfimprove_entries, generation_seed, full_eval_threshold):
        raise NotImplementedError

    def _generate_children(self, output_dir, selfimprove_entries, search_strategy):
        child_ids = []
        executor = ThreadPoolExecutor(max_workers=self.args.selfimprove_workers)
        future_to_job = {
            executor.submit(
                generate_child_patch,
                parent_commit='initial' if self.args.run_baseline == 'no_selfimprove' else parent_commit,
                output_dir=output_dir,
                force_rebuild=False,
                entry=entry,
                benchmark_name=self.args.benchmark,
                search_strategy=search_strategy,
                rung=0,
            ): (parent_commit, entry)
            for parent_commit, entry in selfimprove_entries
        }
        pending = set(future_to_job)
        deadline = time.monotonic() + GENERATION_TIMEOUT_SECONDS
        try:
            while pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                done, pending = wait(pending, timeout=remaining, return_when=FIRST_COMPLETED)
                if not done:
                    break
                for future in done:
                    try:
                        metadata = future.result()
                        child_ids.append(metadata["run_id"])
                    except Exception as exc:
                        parent_commit, entry = future_to_job[future]
                        self.logger.error(
                            f"Child generation failed for parent={parent_commit}, entry={entry}: {exc}"
                        )
                        self.logger.error(traceback.format_exc())
            if pending:
                for future in pending:
                    parent_commit, entry = future_to_job[future]
                    self.logger.error(
                        f"Child generation timed out for parent={parent_commit}, entry={entry}"
                    )
                    future.cancel()
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        return child_ids

    def _evaluate_children(self, output_dir, run_ids, task_list, subset_name, budget_name, budget_size, rung):
        if not run_ids or not task_list:
            return []

        evaluated_ids = []
        executor = ThreadPoolExecutor(max_workers=self.args.selfimprove_workers)
        future_to_run_id = {
            executor.submit(
                evaluate_existing_child,
                run_id,
                output_dir,
                self.args.num_benchmark_evals,
                self.args.benchmark,
                task_list,
                subset_name,
                budget_name,
                budget_size,
                self.args.scheduler,
                rung,
            ): run_id
            for run_id in run_ids
        }
        try:
            for future, run_id in future_to_run_id.items():
                try:
                    future.result()
                    evaluated_ids.append(run_id)
                except Exception as exc:
                    self.logger.error(f"Evaluation failed for child={run_id}: {exc}")
                    self.logger.error(traceback.format_exc())
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
        return evaluated_ids


class BaselineScheduler(BaseScheduler):
    def __init__(self, args, benchmark, logger, stage_small_issues, stage_medium_issues, final_stage_issues, expected_task_counts):
        super().__init__(args, benchmark, logger)
        self.stage_small_issues = stage_small_issues
        self.stage_medium_issues = stage_medium_issues
        self.final_stage_issues = final_stage_issues
        self.expected_task_counts = expected_task_counts
        self.test_more_threshold = 0.4

    def get_generation_child_count(self, generation_seed):
        child_count = self.args.selfimprove_size
        if self.args.generation_task_budget_total:
            max_affordable = max(1, self.args.generation_task_budget_total // max(1, len(self.stage_small_issues)))
            child_count = min(child_count, max_affordable)
        return child_count

    def run_generation(self, output_dir, selfimprove_entries, generation_seed, full_eval_threshold):
        children = self._generate_children(output_dir, selfimprove_entries, search_strategy="baseline")
        consumed_budget = 0
        rungs = []
        promoted_stage1 = []
        promoted_stage2 = []

        self._evaluate_children(
            output_dir,
            children,
            self.stage_small_issues,
            subset_name="stage_small",
            budget_name="stage_small",
            budget_size=len(self.stage_small_issues),
            rung=0,
        )
        consumed_budget += len(children) * len(self.stage_small_issues)
        for run_id in children:
            metadata = _load_child_metadata(output_dir, run_id)
            if (
                (metadata.get("overall_performance") or {}).get("total_resolved_instances", 0) >=
                len(self.stage_small_issues) * self.test_more_threshold
            ):
                promoted_stage1.append(run_id)

        if self.args.generation_task_budget_total and self.stage_medium_issues:
            remaining = max(0, self.args.generation_task_budget_total - consumed_budget)
            max_promotions = remaining // len(self.stage_medium_issues)
            promoted_stage1 = _promote_top_k(promoted_stage1, output_dir, max_promotions)
        _mark_promotion_decisions(output_dir, children, promoted_stage1, 0)
        rungs.append({
            "rung": 0,
            "budget_size": len(self.stage_small_issues),
            "tasks_added": len(self.stage_small_issues),
            "num_candidates_in": len(children),
            "num_promoted": len(promoted_stage1),
            "num_killed": len(children) - len(promoted_stage1),
        })

        if promoted_stage1 and self.stage_medium_issues:
            self._evaluate_children(
                output_dir,
                promoted_stage1,
                self.stage_medium_issues,
                subset_name="stage_medium",
                budget_name="stage_medium",
                budget_size=len(self.stage_small_issues) + len(self.stage_medium_issues),
                rung=1,
            )
            consumed_budget += len(promoted_stage1) * len(self.stage_medium_issues)

        source_for_final = promoted_stage1 if self.stage_medium_issues else children
        for run_id in source_for_final:
            metadata = _load_child_metadata(output_dir, run_id)
            if _child_score(metadata) >= full_eval_threshold:
                promoted_stage2.append(run_id)
        if self.args.generation_task_budget_total and self.final_stage_issues:
            remaining = max(0, self.args.generation_task_budget_total - consumed_budget)
            max_promotions = remaining // len(self.final_stage_issues)
            promoted_stage2 = _promote_top_k(promoted_stage2, output_dir, max_promotions)
        if self.final_stage_issues:
            _mark_promotion_decisions(output_dir, source_for_final, promoted_stage2, 1)

        if self.final_stage_issues:
            rungs.append({
                "rung": 1,
                "budget_size": len(self.stage_small_issues) + len(self.stage_medium_issues or []),
                "tasks_added": len(self.stage_medium_issues or []),
                "num_candidates_in": len(source_for_final),
                "num_promoted": len(promoted_stage2),
                "num_killed": len(source_for_final) - len(promoted_stage2),
            })
        if promoted_stage2 and self.final_stage_issues:
            cumulative_budget = len(self.stage_small_issues)
            if self.stage_medium_issues:
                cumulative_budget += len(self.stage_medium_issues)
            cumulative_budget += len(self.final_stage_issues)
            self._evaluate_children(
                output_dir,
                promoted_stage2,
                self.final_stage_issues,
                subset_name="stage_full",
                budget_name="stage_full",
                budget_size=cumulative_budget,
                rung=2,
            )
            consumed_budget += len(promoted_stage2) * len(self.final_stage_issues)
            rungs.append({
                "rung": 2,
                "budget_size": cumulative_budget,
                "tasks_added": len(self.final_stage_issues),
                "num_candidates_in": len(promoted_stage2),
                "num_promoted": len(promoted_stage2),
                "num_killed": 0,
            })

        for run_id in children:
            finalize_child_metadata(run_id, output_dir, post_improve_diagnose=self.args.post_improve_diagnose)

        compiled_children = [
            run_id for run_id in children
            if is_compiled_self_improve(
                _load_child_metadata(output_dir, run_id),
                expected_task_counts=self.expected_task_counts,
            )
        ]
        summary = _summarize_children(children, output_dir, self.expected_task_counts[-1])
        return {
            "children": children,
            "children_compiled": compiled_children,
            "archive_candidates": compiled_children,
            "evaluation_budget_tasks_consumed": consumed_budget,
            "rungs": rungs,
            **summary,
        }


class HyperbandScheduler(BaseScheduler):
    def __init__(self, args, benchmark, logger):
        super().__init__(args, benchmark, logger)
        self.eta = args.hyperband_eta
        self.budgets = args.hyperband_budgets
        self.reference_instance_ids = load_reference_instance_ids(args.benchmark)

    def _generation_budget_target(self):
        if self.args.generation_task_budget_total:
            return self.args.generation_task_budget_total
        return self.args.selfimprove_size * self.budgets[-1]

    def _budget_cost(self, initial_children):
        total = 0
        previous_budget = 0
        candidates = initial_children
        for budget in self.budgets:
            total += candidates * (budget - previous_budget)
            previous_budget = budget
            candidates = math.ceil(candidates / self.eta)
        return total

    def get_generation_child_count(self, generation_seed):
        if self.args.hyperband_initial_children:
            return self.args.hyperband_initial_children

        target_budget = self._generation_budget_target()
        best_n = 1
        for n in range(1, 1000):
            if self._budget_cost(n) <= target_budget:
                best_n = n
            else:
                break
        return best_n

    def _task_increments(self, generation_seed):
        task_order = list(self.reference_instance_ids)
        rng = random.Random(generation_seed)
        rng.shuffle(task_order)
        increments = []
        previous_budget = 0
        for budget in self.budgets:
            increments.append(task_order[previous_budget:budget])
            previous_budget = budget
        return increments

    def run_generation(self, output_dir, selfimprove_entries, generation_seed, full_eval_threshold):
        if self.benchmark.kind == "polyglot":
            raise ValueError("Hyperband scheduler is only implemented for SWE-style task budgets.")

        children = self._generate_children(output_dir, selfimprove_entries, search_strategy="hyperband")
        task_increments = self._task_increments(generation_seed)
        consumed_budget = 0
        rungs = []
        current_candidates = children

        for rung_index, (budget_size, task_increment) in enumerate(zip(self.budgets, task_increments)):
            if not current_candidates:
                rungs.append({
                    "rung": rung_index,
                    "budget_size": budget_size,
                    "tasks_added": len(task_increment),
                    "num_candidates_in": 0,
                    "num_promoted": 0,
                    "num_killed": 0,
                })
                continue

            self._evaluate_children(
                output_dir,
                current_candidates,
                task_increment,
                subset_name=f"hyperband_rung_{rung_index}",
                budget_name=f"hyperband_rung_{rung_index}",
                budget_size=budget_size,
                rung=rung_index,
            )
            consumed_budget += len(current_candidates) * len(task_increment)

            if rung_index == len(self.budgets) - 1:
                promoted = current_candidates
            else:
                promote_count = math.ceil(len(current_candidates) / self.eta)
                promoted = _promote_top_k(
                    current_candidates,
                    output_dir,
                    promote_count,
                    require_positive=True,
                )
            _mark_promotion_decisions(output_dir, current_candidates, promoted, rung_index)
            rungs.append({
                "rung": rung_index,
                "budget_size": budget_size,
                "tasks_added": len(task_increment),
                "num_candidates_in": len(current_candidates),
                "num_promoted": len(promoted),
                "num_killed": len(current_candidates) - len(promoted),
            })
            current_candidates = promoted

        winner = None
        finalists = _rank_children(current_candidates, output_dir) if current_candidates else []
        if finalists:
            winner = finalists[0]

        for run_id in children:
            finalize_child_metadata(run_id, output_dir, post_improve_diagnose=False)
        if winner:
            finalize_child_metadata(winner, output_dir, post_improve_diagnose=self.args.post_improve_diagnose)

        compiled_children = [
            run_id for run_id in children
            if is_compiled_self_improve(_load_child_metadata(output_dir, run_id))
        ]
        summary = _summarize_children(children, output_dir, self.budgets[-1])
        archive_candidates = [winner] if winner else []
        return {
            "children": children,
            "children_compiled": compiled_children,
            "archive_candidates": archive_candidates,
            "evaluation_budget_tasks_consumed": consumed_budget,
            "rungs": rungs,
            **summary,
        }
