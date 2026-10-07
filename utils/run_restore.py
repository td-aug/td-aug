"""Shared filesystem helpers for restoring saved experiment runs."""

import glob
import json
import os


def resolve_run_directory(run_path: str) -> str:
    """Resolve run directory, expanding glob patterns and stripping checkpoint files."""
    if run_path is None:
        raise ValueError("Expected a run path, got None.")
    candidates = glob.glob(run_path)
    if not candidates and os.path.isdir(run_path):
        candidates = [run_path]
    if not candidates:
        raise FileNotFoundError(f"Could not find any run at path: {run_path}")
    if len(candidates) > 1:
        raise ValueError(f"Ambiguous run path {run_path}, matched: {candidates}")
    resolved = candidates[0]
    if os.path.isfile(resolved):
        resolved = os.path.dirname(resolved)
    return resolved


def load_run_flags(run_dir: str) -> dict:
    """Load the saved flags.json from a flow run."""
    flags_path = os.path.join(run_dir, "flags.json")
    if not os.path.exists(flags_path):
        raise FileNotFoundError(f"No flags.json found in {run_dir}")
    with open(flags_path, "r") as f:
        flag_dict = json.load(f)
    if "agent" not in flag_dict:
        raise ValueError(f"flags.json in {run_dir} does not contain an agent config.")
    return flag_dict
