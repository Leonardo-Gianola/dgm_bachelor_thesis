"""
Blind GA mutation prompt.

Unlike the standard DGM diagnosis prompt (which reads task-specific error logs
and reasons about a targeted fix), this prompt asks the LLM to propose a
*random, exploratory* mutation to the coding agent without any failure context.
High-temperature sampling is used at call-time to maximise diversity.
"""

from prompts.self_improvement_prompt import (
    coding_agent_summary,
    coding_agent_summary_polyglot,
    problem_description_prompt,
    get_current_code,
)

GA_MUTATION_SYSTEM_MESSAGE = """Here is the implementation of the coding agent.

# Coding Agent Implementation
----- Coding Agent Implementation Start -----
{code}
----- Coding Agent Implementation End -----

Your task is to propose ONE creative, exploratory mutation to the coding agent's implementation.
Unlike standard improvements driven by failure analysis, this mutation should explore an unexpected \
or unconventional direction — a "blind" perturbation that could serendipitously improve the agent's \
coding ability.
Be creative and bold: consider unusual tool designs, novel prompting strategies, alternative workflow \
architectures, or unexpected heuristics. The mutation must still be a coherent code change; it should \
not break the agent's ability to run."""

GA_MUTATION_PROMPT = """Propose ONE creative, exploratory mutation to the coding agent.

Do NOT analyze specific task failures or error logs. Instead, reason broadly about the agent's design \
and suggest a random, high-variance change that could potentially improve its performance on coding tasks.

Respond precisely in the following format including the JSON start and end markers:

```json
<JSON>
```

In <JSON>, provide a JSON response with the following fields:
- "mutation_rationale": A brief description of the proposed mutation and why it is interesting to explore, \
even without direct evidence it will help.
- "implementation_suggestion": Referring to the coding agent's summary and implementation, describe \
concretely what should be added or changed (file, function, or new tool) to implement the mutation.
- "problem_description": Phrase the mutation as a GitHub issue description. It should clearly describe \
the change so that a software engineer viewing the issue and the repository can implement it.

Your response will be automatically parsed, so ensure that the string response is precisely in the \
correct format. Do NOT include the `<JSON>` tag in your output."""


def get_ga_mutation_prompt(root_dir, patch_files=None, is_polyglot=False):
    """Return (system_message, user_prompt) for a blind GA mutation.

    Args:
        root_dir: Repository root used to read current coding-agent source files.
        patch_files: Optional list of patch files to include in the code context.
        is_polyglot: Whether this run uses the polyglot coding agent.

    Returns:
        Tuple[str, str]: (system_message, user_prompt)
    """
    code_files = ['coding_agent.py', 'tools/', 'utils/']
    exclude_files = [
        'utils/evo_utils.py',
        'utils/docker_utils.py',
        'utils/swe_log_parsers.py',
        'prompts/self_improvement_prompt.py',
    ]
    code_text = get_current_code(
        root_dir,
        code_files,
        patch_files=patch_files or [],
        exclude_files=exclude_files,
        is_polyglot=is_polyglot,
    )
    summary = coding_agent_summary_polyglot if is_polyglot else coding_agent_summary
    system_message = summary + GA_MUTATION_SYSTEM_MESSAGE.format(code=code_text)
    return system_message, GA_MUTATION_PROMPT


def make_ga_problem_statement(response_json, is_polyglot=False):
    """Convert the JSON response into a problem_statement for coding_agent.py.

    The JSON must contain 'implementation_suggestion' and 'problem_description' keys.
    """
    summary = coding_agent_summary_polyglot if is_polyglot else coding_agent_summary
    return summary + problem_description_prompt.format(
        implementation_suggestion=response_json["implementation_suggestion"],
        problem_description=response_json["problem_description"],
    )
