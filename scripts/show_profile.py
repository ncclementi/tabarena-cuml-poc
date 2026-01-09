#!/usr/bin/env python3
"""Script to display and analyze cProfile .prof file results."""

import argparse
import json
import pstats
import re
import sys
from pathlib import Path

from benchmark_db import get_database_path, load_benchmark_runs, load_benchmark_timings


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parent.parent


def lookup_profile_from_run_id(run_id: str) -> tuple[Path | None, str | None]:
    """Look up the profile path for a given run ID.

    Returns (profile_path, error_message). If successful, error_message is None.
    """
    db_path = get_database_path()
    df_runs = load_benchmark_runs(db_path=db_path)
    df_timings = load_benchmark_timings(db_path=db_path)

    if df_runs.empty:
        return None, "No benchmark runs found in database."

    # Match run_id by prefix
    mask = df_runs["run_id"].str.startswith(run_id)
    matched_runs = df_runs[mask]

    if matched_runs.empty:
        return None, f"No run found with run_id starting with '{run_id}'"

    if len(matched_runs) > 1:
        run_ids = matched_runs["run_id"].tolist()
        return None, f"Multiple runs match '{run_id}'. Please be more specific:\n" + "\n".join(
            f"  - {rid}" for rid in run_ids
        )

    row = matched_runs.iloc[0]
    full_run_id = row["run_id"]

    # Get experiment_id from timings table
    if df_timings.empty or "stage_metadata.experiment_id" not in df_timings.columns:
        return None, f"No timing data found for run '{full_run_id}'"

    timing_row = df_timings[df_timings["run_id"] == full_run_id]
    if timing_row.empty:
        return None, f"No timing data found for run '{full_run_id}'"

    experiment_id = timing_row["stage_metadata.experiment_id"].dropna().iloc[0] if not timing_row[
        "stage_metadata.experiment_id"
    ].dropna().empty else None

    if not experiment_id:
        return None, f"No experiment_id found for run '{full_run_id}'"

    # Get datasets
    datasets_json = row.get("datasets")
    if not datasets_json:
        return None, f"No datasets found for run '{full_run_id}'"

    try:
        datasets = json.loads(datasets_json)
    except (json.JSONDecodeError, TypeError):
        return None, f"Failed to parse datasets for run '{full_run_id}'"

    if not datasets:
        return None, f"Empty datasets list for run '{full_run_id}'"

    # Build profile path (use first dataset if multiple)
    dataset = datasets[0]
    profile_path = get_project_root() / "cprofiles" / experiment_id / f"{dataset}.prof"

    if not profile_path.exists():
        return None, f"Profile not found: {profile_path}"

    return profile_path, None


def show_profile(
    prof_path: Path,
    sort_key: str,
    num_lines: int,
    filter_pattern: str | None = None,
    show_callers: str | None = None,
    show_callees: str | None = None,
):
    """Display profile statistics with optional filtering."""
    stats = pstats.Stats(str(prof_path))
    stats.strip_dirs()
    stats.sort_stats(sort_key)

    if show_callers:
        print(f"\n{'='*80}")
        print(f"CALLERS of functions matching '{show_callers}':")
        print(f"{'='*80}\n")
        stats.print_callers(show_callers, num_lines)
    elif show_callees:
        print(f"\n{'='*80}")
        print(f"CALLEES of functions matching '{show_callees}':")
        print(f"{'='*80}\n")
        stats.print_callees(show_callees, num_lines)
    elif filter_pattern:
        stats.print_stats(filter_pattern, num_lines)
    else:
        stats.print_stats(num_lines)


def list_functions(prof_path: Path, pattern: str | None = None):
    """List all functions in the profile, optionally filtered."""
    stats = pstats.Stats(str(prof_path))
    stats.strip_dirs()

    # Get all function keys
    func_keys = list(stats.stats.keys())

    print(f"\nFunctions in profile ({len(func_keys)} total):\n")

    for filename, line, func_name in sorted(func_keys, key=lambda x: x[2]):
        full_name = f"{filename}:{line}({func_name})"
        if pattern is None or re.search(pattern, full_name, re.IGNORECASE):
            data = stats.stats[(filename, line, func_name)]
            cumtime = data[3]  # cumulative time
            tottime = data[2]  # total time
            calls = data[0]  # number of calls
            print(f"  {cumtime:8.3f}s cum, {tottime:8.3f}s tot, {calls:6d} calls: {full_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Display and analyze cProfile .prof file results",
        epilog="""
Examples:
  %(prog)s 5a8ee16c                          # Show top 30 by cumulative time
  %(prog)s 5a8ee16c -f get_info              # Filter to functions matching 'get_info'
  %(prog)s 5a8ee16c --callers get_info       # Show what calls get_info
  %(prog)s 5a8ee16c --callees get_info       # Show what get_info calls
  %(prog)s 5a8ee16c --list                   # List all functions
  %(prog)s 5a8ee16c --list -f export         # List functions matching 'export'
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("prof_file_or_run_id", help="Path to .prof file or run ID")
    parser.add_argument(
        "-n", "--lines", type=int, default=30, help="Number of lines to show (default: 30)"
    )
    parser.add_argument(
        "-s",
        "--sort",
        default="cumulative",
        choices=["cumulative", "time", "calls", "name"],
        help="Sort order (default: cumulative)",
    )
    parser.add_argument(
        "-f", "--filter", dest="filter_pattern", help="Filter functions by regex pattern"
    )
    parser.add_argument(
        "--callers", metavar="PATTERN", help="Show callers of functions matching PATTERN"
    )
    parser.add_argument(
        "--callees", metavar="PATTERN", help="Show callees of functions matching PATTERN"
    )
    parser.add_argument(
        "--list", action="store_true", help="List all functions (combine with -f to filter)"
    )
    args = parser.parse_args()

    input_arg = args.prof_file_or_run_id
    prof_path = Path(input_arg)

    # First, try as a file path
    if not (prof_path.exists() and prof_path.is_file()):
        # Not a file - try as run_id
        profile_path, error = lookup_profile_from_run_id(input_arg)

        if error:
            # Provide helpful error message based on input
            if "/" in input_arg or input_arg.endswith(".prof"):
                print(f"Error: File not found: {input_arg}", file=sys.stderr)
            else:
                print(f"Error: {error}", file=sys.stderr)
            sys.exit(1)
        prof_path = profile_path

    try:
        if args.list:
            list_functions(prof_path, args.filter_pattern)
        else:
            show_profile(
                prof_path,
                args.sort,
                args.lines,
                filter_pattern=args.filter_pattern,
                show_callers=args.callers,
                show_callees=args.callees,
            )
    except Exception as e:
        print(f"Error loading profile: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
