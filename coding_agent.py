
import argparse
import subprocess
import logging
from logging.handlers import RotatingFileHandler
import os
import re
import threading

from llm_withtools import OPENAI_MODEL, chat_with_agent
from utils.eval_utils import get_report_score, msg_history_to_report, score_tie_breaker
from utils.git_utils import diff_versus_commit, reset_to_commit, apply_patch

# Thread-local storage for logger instances
thread_local = threading.local()

def get_thread_logger():
    """
    Get the logger instance specific to the current thread.
    Returns None if no logger has been set for this thread.
    """
    return getattr(thread_local, 'logger', None)

def set_thread_logger(logger):
    """
    Set the logger instance for the current thread.
    """
    thread_local.logger = logger

def setup_logger(log_file='./chat_history.md', level=logging.INFO):
    """
    Set up a logger with both file and console handlers.
    """
    # Create logger with a unique name based on thread ID
    logger = logging.getLogger(f'AgenticSystem-{threading.get_ident()}')
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Create formatters
    file_formatter = logging.Formatter('%(message)s')
    
    # Create and set up file handler
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    
    # Store logger in thread-local storage
    set_thread_logger(logger)
    
    return logger

def safe_log(message, level=logging.INFO):
    """
    Thread-safe logging function that ensures messages go to the correct logger.
    """
    logger = get_thread_logger()
    if logger:
        logger.log(level, message)
    else:
        print(f"Warning: No logger found for thread {threading.get_ident()}")

def is_patch_empty(patch_str):
    """Check if a patch string is empty or whitespace-only."""
    return not patch_str or not patch_str.strip()


def is_test_only_patch(patch_str):
    """Check if a patch only modifies test files (under test/ or tests/ dirs)."""
    if not patch_str or not patch_str.strip():
        return False
    file_pattern = re.compile(r'diff --git a/(.*?) b/(.*?)(?:\n|$)')
    files_in_patch = {m.group(1) for m in file_pattern.finditer(patch_str)}
    if not files_in_patch:
        return False
    return all(f.startswith(('test/', 'tests/')) for f in files_in_patch)


def validate_patch(patch_str):
    """Validate patch: not empty and not test-only. Returns (is_valid, reason)."""
    if is_patch_empty(patch_str):
        return False, "patch is empty"
    if is_test_only_patch(patch_str):
        return False, "patch only modifies test files"
    return True, "valid"


class AgenticSystem:
    def __init__(
            self,
            problem_statement,
            git_tempdir,
            base_commit,
            chat_history_file='./chat_history.md',
            test_description=None,
            self_improve=False,
            instance_id=None,
            max_retries=3,
        ):
        self.problem_statement = problem_statement
        self.git_tempdir = git_tempdir
        self.base_commit = base_commit
        self.chat_history_file = chat_history_file
        self.test_description = test_description
        self.self_improve = self_improve
        self.instance_id = instance_id if not self_improve else 'dgm'
        self.code_model = OPENAI_MODEL
        self.max_retries = max_retries
        self.last_token_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

        # Initialize logger and store it in thread-local storage
        self.logger = setup_logger(chat_history_file)
        
        # Clear the log file
        with open(chat_history_file, 'w') as f:
            f.write('')

    def get_current_edits(self):
        diff = str(diff_versus_commit(self.git_tempdir, self.base_commit))
        return diff

    def get_regression_tests(self):
        """
        Get the regression tests from the repository.
        """
        instruction = f"""I have uploaded a Python code repository in the directory {self.git_tempdir}.

<problem_description>
{self.problem_statement}
</problem_description>

<test_description>
{self.test_description}
</test_description>

Your task is to identify regression tests in the {self.git_tempdir} directory that should pass both before and after addressing the <problem_description>. I have already taken care of the required dependencies.
At the end, please provide a summary that includes where the regression tests are located, what they are testing, and how they can be executed.
"""

        new_msg_history = chat_with_agent(instruction, model=self.code_model, msg_history=[], logging=safe_log)
        regression_tests_summary = new_msg_history[-1]
        try:
            regression_tests_summary = regression_tests_summary['content'][-1]['text']
        except:
            pass
        return regression_tests_summary

    def run_regression_tests(self, regression_tests_summary):
        """
        Run the regression tests and get the test report.
        """
        code_diff = self.get_current_edits()
        instruction = f"""I have uploaded a Python code repository in the directory {self.git_tempdir}. There is an attempt to address the problem statement. Please review the changes and run the regression tests.

<problem_description>
{self.problem_statement}
</problem_description>

<attempted_solution>
{code_diff}
</attempted_solution>

<test_description>
{self.test_description}
</test_description>

<regression_tests_summary>
{regression_tests_summary}
</regression_tests_summary>

Your task is to run the regression tests in the {self.git_tempdir} directory to ensure that the changes made to the code address the <problem_description>.
"""
        new_msg_history = chat_with_agent(instruction, model=self.code_model, msg_history=[], logging=safe_log)
        test_report = msg_history_to_report(self.instance_id, new_msg_history, model=self.code_model)
        return test_report

    def _build_instruction(self, retry_reason=None):
        """Build the instruction prompt, with stronger wording on retries."""
        base = f"""I have uploaded a Python code repository in the directory {self.git_tempdir}. Help solve the following problem.

<problem_description>
{self.problem_statement}
</problem_description>

<test_description>
{self.test_description}
</test_description>

Your task is to make changes to the files in the {self.git_tempdir} directory to address the <problem_description>. I have already taken care of the required dependencies.

CRITICAL INSTRUCTIONS — YOU MUST FOLLOW THESE:
1. You MUST use the editor tool with command "edit" or "create" to modify source files, or use the bash tool to write files. Do NOT just describe changes — actually apply them.
2. After making changes, verify them by using the bash tool to run: git diff
3. Your changes MUST appear in the git diff. If no files are modified, the task is considered FAILED.
4. Focus on modifying the PRIMARY SOURCE CODE, not just test files.
"""
        if retry_reason:
            base += f"""
WARNING: Your previous attempt FAILED because: {retry_reason}.
You MUST actually edit files this time. Use the editor tool with command="edit" and provide the full file_text, or use bash to write files directly.
"""
        return base

    def forward(self):
        """
        The forward function with retry logic for empty/invalid patches.
        """
        instruction = self._build_instruction()

        new_msg_history, token_usage = chat_with_agent(
            instruction,
            model=self.code_model,
            msg_history=[],
            logging=safe_log,
            return_usage=True,
        )
        self.last_token_usage = token_usage

        # Validate the generated patch
        patch = diff_versus_commit(self.git_tempdir, self.base_commit)
        is_valid, reason = validate_patch(patch)

        attempt = 1
        while not is_valid and attempt < self.max_retries:
            safe_log(f"Patch validation failed: {reason}. Retry {attempt + 1}/{self.max_retries}")

            # Reset to base commit before retry
            reset_to_commit(self.git_tempdir, self.base_commit)

            # Retry with stronger instruction
            retry_instruction = self._build_instruction(retry_reason=reason)
            new_msg_history, token_usage = chat_with_agent(
                retry_instruction,
                model=self.code_model,
                msg_history=[],
                logging=safe_log,
                return_usage=True,
            )
            self.last_token_usage = {
                "input_tokens": self.last_token_usage["input_tokens"] + token_usage["input_tokens"],
                "output_tokens": self.last_token_usage["output_tokens"] + token_usage["output_tokens"],
                "total_tokens": self.last_token_usage["total_tokens"] + token_usage["total_tokens"],
            }

            patch = diff_versus_commit(self.git_tempdir, self.base_commit)
            is_valid, reason = validate_patch(patch)
            attempt += 1

        if not is_valid:
            safe_log(f"Patch validation failed after {self.max_retries} attempts: {reason}")

        return new_msg_history

def main():
    parser = argparse.ArgumentParser(description='Process repository with an agentic system.')
    parser.add_argument('--problem_statement', required=True, help='The problem statement to process')
    parser.add_argument('--git_dir', required=True, help='Path to git repository directory')
    parser.add_argument('--base_commit', required=True, help='Base commit hash to compare against')
    parser.add_argument('--chat_history_file', required=True, help='Path to chat history file')
    parser.add_argument('--outdir', required=False, default="/dgm/", help='Output directory')
    parser.add_argument('--test_description', default=None, required=False, help='Description of how to test the repository')
    parser.add_argument('--self_improve', default=False, action='store_true', help='Whether to self-improve the repository or solving swe')
    parser.add_argument('--instance_id', default=None, help='Instance ID for SWE issue')
    args = parser.parse_args()

    # Process the repository
    agentic_system = AgenticSystem(
        problem_statement=args.problem_statement,
        git_tempdir=args.git_dir,
        base_commit=args.base_commit,
        chat_history_file=args.chat_history_file,
        test_description=args.test_description,
        self_improve=args.self_improve,
        instance_id=args.instance_id,
    )

    # Run the agentic system to try to solve the problem
    agentic_system.forward()

    # Get code diff and save to model_patch.diff
    model_patch = diff_versus_commit(args.git_dir, args.base_commit)
    model_patch_outfile = os.path.join(args.outdir, 'model_patch.diff') if args.outdir else 'model_patch.diff'
    with open(model_patch_outfile, 'w') as f:
        f.write(model_patch)

    token_usage_outfile = os.path.join(args.outdir, 'token_usage.json') if args.outdir else 'token_usage.json'
    with open(token_usage_outfile, 'w') as f:
        import json
        json.dump(agentic_system.last_token_usage, f, indent=2)

if __name__ == "__main__":
    main()
