import json
import os

from utils.common_utils import load_json_file, read_file


def load_dgm_metadata(dgm_metadata_path, last_only=False):
    # Load all archives from given metadata file
    if not os.path.exists(dgm_metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {dgm_metadata_path}")
    # Read all JSON entries from the metadata file
    content = read_file(dgm_metadata_path)
    json_entries = content.split('\n{')
    # Parse all JSON entries
    dgm_metadata = []
    for json_entry in json_entries:
        # Add back the { if it was removed by split
        if not json_entry.startswith('{'):
            json_entry = '{' + json_entry
        # Parse the JSON entry
        metadata = json.loads(json_entry)
        dgm_metadata.append(metadata)

    if last_only:
        return dgm_metadata[-1]
    return dgm_metadata

def get_model_patch_paths(root_dir, dgm_dir, parent_commit):
    prev_commit = parent_commit
    patch_files = []
    while prev_commit != 'initial':
        parent_dir = os.path.join(root_dir, dgm_dir, prev_commit)
        parent_patch_file = os.path.join(parent_dir, "model_patch.diff")
        if os.path.exists(parent_patch_file):
            patch_files.append(parent_patch_file)
        else:
            print(f"Parent patch file not found: {parent_patch_file}")
        # find next parent commit in the metadata
        parent_metadata = load_json_file(os.path.join(parent_dir, "metadata.json"))
        prev_commit = parent_metadata.get('parent_commit', 'initial')
    return patch_files[::-1]  # reverse the list to get the correct order


def get_model_patch_paths_from_agent_dir(agent_dir):
    """
    Resolve the ordered patch chain for a stored agent archive directory.

    The expected layout is the DGM archive layout where each child directory
    stores a `metadata.json` with a `parent_commit` that points to a sibling
    directory under the same run root.
    """
    agent_dir = os.path.abspath(agent_dir)
    metadata_path = os.path.join(agent_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        return []

    patch_files = []
    current_dir = agent_dir
    while True:
        model_patch_file = os.path.join(current_dir, "model_patch.diff")
        if os.path.exists(model_patch_file):
            patch_files.append(model_patch_file)

        metadata_path = os.path.join(current_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            break

        metadata = load_json_file(metadata_path)
        parent_commit = metadata.get("parent_commit", "initial")
        if parent_commit == "initial":
            break
        current_dir = os.path.join(os.path.dirname(current_dir), parent_commit)

    return patch_files[::-1]

def get_all_performance(run_keyword, results_dir='./swe_bench'):
    """
    Retrieve performance results for all runs based on the provided keyword.

    Args:
        run_keyword (str): A keyword used to identify the target runs' evaluation results.

    Returns:
        list: A list of dictionaries, each containing performance results for a matching run.
    """
    # Find all JSON files in eval_results_dir matching the keyword
    matching_files = [
        f for f in os.listdir(results_dir)
        if f.endswith('.json') and run_keyword in f
    ]
    
    # Return an empty list if no matches are found
    if not matching_files:
        print(f"No evaluation files found matching the keyword '{run_keyword}'.")
        return None, None
    
    # Process each matching file
    performance_results = []
    total_resolved_instances = 0
    total_submitted_instances = 0
    total_unresolved_ids = []
    total_resolved_ids = []
    total_emptypatch_ids = []
    for file_name in matching_files:
        eval_agent_path = os.path.join(results_dir, file_name)
        eval_results = load_json_file(eval_agent_path)
        resolved_instances = eval_results.get('resolved_instances', 0)
        submitted_instances = eval_results.get('submitted_instances', 0)
        total_resolved_instances += resolved_instances
        total_submitted_instances += submitted_instances
        accuracy_score = resolved_instances / submitted_instances if submitted_instances > 0 else 0
        performance_results.append({'file': file_name, 'accuracy_score': accuracy_score, **eval_results})
        total_unresolved_ids.extend(eval_results.get('unresolved_ids', []))
        total_emptypatch_ids.extend(eval_results.get('empty_patch_ids', []))
        total_resolved_ids.extend(eval_results.get('resolved_ids', []))

    # Calculate the overall accuracy score
    overall_performance = {}
    overall_performance['accuracy_score'] = total_resolved_instances / total_submitted_instances if total_submitted_instances > 0 else 0
    overall_performance['total_resolved_instances'] = total_resolved_instances
    overall_performance['total_submitted_instances'] = total_submitted_instances
    overall_performance['files'] = matching_files
    overall_performance['total_unresolved_ids'] = total_unresolved_ids
    overall_performance['total_emptypatch_ids'] = total_emptypatch_ids
    overall_performance['total_resolved_ids'] = total_resolved_ids
    
    return performance_results, overall_performance

def is_compiled_self_improve(metadata, expected_task_counts=None, logger=None):
    """
    Checks if the run was properly compiled and 'self-improved' by verifying:
      1. The 'overall_performance' dict has the required keys:
         ('accuracy_score', 'total_unresolved_ids', 'total_resolved_ids', 'total_emptypatch_ids').
      2. There is at least one non-empty patch (resolved + unresolved > 0).
      3. If expected_task_counts is provided, the planned cumulative budget
         (from budget_history) matches one of the expected counts.

    Returns True if all conditions are met, else False.
    """
    overall_perf = metadata.get('overall_performance', {})
    required_keys = ['accuracy_score', 'total_unresolved_ids', 'total_resolved_ids', 'total_emptypatch_ids']

    def log_info(message):
        if logger is not None:
            logger.info(message)

    # 1. Must have the required keys
    if not overall_perf or not all(k in overall_perf for k in required_keys):
        log_info("no required keys")
        return False

    # 2. Must have at least one non-empty patch
    num_resolved = len(overall_perf['total_resolved_ids'])
    num_unresolved = len(overall_perf['total_unresolved_ids'])
    if (num_resolved + num_unresolved) == 0:
        log_info("no non-empty patch")
        return False

    # 3. If specified, child must have completed a planned stage budget.
    #    Use the *planned* cumulative budget from budget_history rather than
    #    overall_performance.total_submitted_instances: SWE-bench eval can
    #    submit extra tasks (retries) that inflate the actual count beyond
    #    the planned cumulative size, causing strict equality to reject
    #    otherwise-valid children.
    if expected_task_counts:
        normalized_counts = sorted(set(expected_task_counts))
        budget_history = metadata.get('budget_history') or []
        if not budget_history:
            log_info("no budget_history present; cannot verify planned stage")
            return False
        last_planned = budget_history[-1].get('cumulative_budget_size')
        if last_planned not in normalized_counts:
            log_info(f"planned budget {last_planned} not in expected counts: {normalized_counts}")
            return False

    return True
