#!/usr/bin/env python
"""CLI tool to inspect benchmark results from the SQLite database."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add scripts directory to path for local module imports
sys.path.insert(0, str(Path(__file__).parent))

import click
import pandas as pd

from benchmark_db import (
    get_database_path,
    load_benchmark_runs,
    load_benchmark_timings,
)


def _parse_profiling_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Parse profiling flags from stage metadata columns.

    Extracts profiling flags from direct columns:
    - stage_metadata.cprofile -> profiling_cprofile
    - stage_metadata.cuml_accel_profile -> profiling_cuml_accel_profile
    """
    df = df.copy()

    # Initialize default values
    df["profiling_cprofile"] = False
    df["profiling_cuml_accel_profile"] = False

    # Extract from direct columns (--metadata key=value format)
    if "stage_metadata.cprofile" in df.columns:
        df["profiling_cprofile"] = df["stage_metadata.cprofile"].fillna(False).astype(bool)
    if "stage_metadata.cuml_accel_profile" in df.columns:
        df["profiling_cuml_accel_profile"] = df["stage_metadata.cuml_accel_profile"].fillna(False).astype(bool)

    return df


def _infer_num_gpus_from_cuda_device_count(
    df: pd.DataFrame, df_runs: pd.DataFrame
) -> pd.DataFrame:
    """Fill NaN values in num_gpus column with cuda.device_count from run metadata.

    When num_gpus is not explicitly set in the experiment config, infer it from
    the cuda.device_count captured in the run's system metadata.

    Args:
        df: DataFrame with num_gpus column and run_id for joining
        df_runs: DataFrame from load_benchmark_runs with cuda.device_count

    Returns:
        DataFrame with NaN num_gpus values filled from cuda.device_count
    """
    if "num_gpus" not in df.columns:
        return df

    df = df.copy()

    # Get cuda.cuda_device_count from runs data
    if "cuda.cuda_device_count" in df_runs.columns:
        cuda_device_counts = df_runs[["run_id", "cuda.cuda_device_count"]].copy()
        cuda_device_counts = cuda_device_counts.rename(
            columns={"cuda.cuda_device_count": "_cuda_device_count"}
        )
        cuda_device_counts["_cuda_device_count"] = pd.to_numeric(
            cuda_device_counts["_cuda_device_count"], errors="coerce"
        )

        # Merge to get cuda device count per row
        df = df.merge(cuda_device_counts, on="run_id", how="left")

        # Fill NaN num_gpus with cuda device count
        mask = pd.isna(df["num_gpus"])
        df.loc[mask, "num_gpus"] = df.loc[mask, "_cuda_device_count"]

        # Drop temporary column
        df = df.drop(columns=["_cuda_device_count"])

    return df


@click.group()
@click.option(
    "--db",
    "db_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to SQLite database. Defaults to benchmark_results.db in project root.",
)
@click.pass_context
def cli(ctx, db_path: Path | None):
    """Inspect benchmark results from SQLite database."""
    ctx.ensure_object(dict)
    ctx.obj["db_path"] = db_path or get_database_path()

    if not ctx.obj["db_path"].exists():
        click.echo(f"Database not found: {ctx.obj['db_path']}", err=True)
        ctx.exit(1)


def _detect_cprofile_from_disk(experiment_name: str, datasets_json: str | None, db_path: Path) -> bool:
    """Check if cProfile data exists on disk for the experiment.

    cProfile files are stored in cprofiles/{experiment_id}/{dataset}.prof
    where experiment_name is typically {experiment_id}_{dataset}.
    """
    if not experiment_name or not datasets_json:
        return False

    # Parse datasets from JSON
    try:
        datasets = json.loads(datasets_json)
        if not datasets:
            return False
        dataset = datasets[0]  # Typically one dataset per run
    except (json.JSONDecodeError, IndexError, TypeError):
        return False

    # Extract experiment_id from experiment_name (format: {experiment_id}_{dataset})
    # The experiment_id is the part before the dataset name
    if f"_{dataset}" in experiment_name:
        experiment_id = experiment_name.rsplit(f"_{dataset}", 1)[0]
    else:
        # Fallback: use the whole experiment name
        experiment_id = experiment_name

    # Check if cprofile file exists
    project_root = db_path.parent
    cprofile_path = project_root / "cprofiles" / experiment_id / f"{dataset}.prof"
    return cprofile_path.exists()


@cli.command()
@click.option("--experiment", "-e", default=None, help="Filter by experiment name")
@click.option("--limit", "-n", default=20, type=int, help="Max rows to show")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option(
    "--no-infer-gpu-count",
    is_flag=True,
    default=False,
    help="Don't infer num_gpus from cuda_device_count when num_gpus is not set.",
)
@click.pass_context
def runs(ctx, experiment: str | None, limit: int, as_json: bool, no_infer_gpu_count: bool):
    """List benchmark runs with summary info."""
    df = load_benchmark_runs(experiment_name=experiment, db_path=ctx.obj["db_path"])

    if df.empty:
        click.echo("No benchmark runs found.")
        return

    # Load timing data to get num_gpus and profiling flags
    df_timings = load_benchmark_timings(experiment_name=experiment, db_path=ctx.obj["db_path"])

    # Extract num_gpus and profiling flags from model_fit stage
    if not df_timings.empty:
        df_stage = df_timings[df_timings["stage"] == "model_fit"].copy()
        df_stage = _parse_profiling_flags(df_stage)

        # Rename num_gpus column if present
        if "stage_metadata.num_gpus" in df_stage.columns:
            df_stage = df_stage.rename(columns={"stage_metadata.num_gpus": "num_gpus"})
            # Convert to numeric (may be stored as string due to SQLite/pandas type handling)
            df_stage["num_gpus"] = pd.to_numeric(df_stage["num_gpus"], errors="coerce")

        # Select columns to merge
        merge_cols = ["run_id"]
        if "num_gpus" in df_stage.columns:
            merge_cols.append("num_gpus")
        if "profiling_cprofile" in df_stage.columns:
            merge_cols.append("profiling_cprofile")
        if "profiling_cuml_accel_profile" in df_stage.columns:
            merge_cols.append("profiling_cuml_accel_profile")

        df_stage_subset = df_stage[merge_cols].drop_duplicates(subset=["run_id"])
        df = df.merge(df_stage_subset, on="run_id", how="left")

        # Infer num_gpus from cuda.device_count if not set (enabled by default)
        if not no_infer_gpu_count and "num_gpus" in df.columns:
            df = _infer_num_gpus_from_cuda_device_count(df, df)

    # Detect cProfile from disk if not already set (external cProfile invocation)
    db_path = ctx.obj["db_path"]
    if "profiling_cprofile" not in df.columns:
        df["profiling_cprofile"] = False
    df["profiling_cprofile"] = df.apply(
        lambda row: row.get("profiling_cprofile", False) or _detect_cprofile_from_disk(
            row.get("experiment_name", ""),
            row.get("datasets"),
            db_path
        ),
        axis=1
    )

    # Select key columns for display
    display_cols = [
        "run_id",
        "execution_datetime",
        "total_time_s",
        "datasets",
        "num_gpus",
        "profiling_cprofile",
        "profiling_cuml_accel_profile",
        "system.hostname",
    ]
    available_cols = [c for c in display_cols if c in df.columns]

    df_display = df[available_cols].head(limit)

    if as_json:
        click.echo(df_display.to_json(orient="records", indent=2))
    else:
        # Truncate long strings for display
        if "run_id" in df_display.columns:
            df_display = df_display.copy()
            df_display["run_id"] = df_display["run_id"].str[:12] + "..."

        with pd.option_context(
            "display.max_columns", None,
            "display.width", 200,
            "display.max_colwidth", 40,
        ):
            click.echo(df_display.to_string(index=False))

        if len(df) > limit:
            click.echo(f"\n... showing {limit} of {len(df)} runs (use -n to show more)")


@cli.command()
@click.argument("run_id")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def timings(ctx, run_id: str, as_json: bool):
    """Show timing breakdown for a specific run.

    RUN_ID can be a partial match (prefix).
    """
    # Load all timings and filter by run_id prefix
    df = load_benchmark_timings(db_path=ctx.obj["db_path"])

    if df.empty:
        click.echo("No timing data found.")
        return

    # Filter by run_id prefix
    mask = df["run_id"].str.startswith(run_id)
    df_filtered = df[mask]

    if df_filtered.empty:
        click.echo(f"No timings found for run_id starting with '{run_id}'")
        return

    # Select timing columns for display
    display_cols = ["stage", "time_ms", "time_s", "timestamp"]
    available_cols = [c for c in display_cols if c in df_filtered.columns]

    df_display = df_filtered[available_cols]

    if as_json:
        click.echo(df_display.to_json(orient="records", indent=2))
    else:
        click.echo(f"Timings for run: {df_filtered['run_id'].iloc[0]}")
        click.echo(f"Experiment: {df_filtered['experiment_name'].iloc[0]}")
        click.echo("-" * 60)
        with pd.option_context("display.max_columns", None, "display.width", 200):
            click.echo(df_display.to_string(index=False))


@cli.command()
@click.argument("run_id")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option("--category", "-c", default=None, help="Filter by category (system, cuda, python, git)")
@click.pass_context
def info(ctx, run_id: str, as_json: bool, category: str | None):
    """Show full metadata for a specific run.

    RUN_ID can be a partial match (prefix).
    """
    df = load_benchmark_runs(db_path=ctx.obj["db_path"])

    if df.empty:
        click.echo("No benchmark runs found.")
        return

    # Filter by run_id prefix
    mask = df["run_id"].str.startswith(run_id)
    df_filtered = df[mask]

    if df_filtered.empty:
        click.echo(f"No run found with run_id starting with '{run_id}'")
        return

    if len(df_filtered) > 1:
        click.echo(f"Multiple runs match '{run_id}'. Please be more specific:")
        for rid in df_filtered["run_id"].tolist():
            click.echo(f"  - {rid}")
        return

    row = df_filtered.iloc[0].to_dict()

    # Filter by category if specified
    if category:
        prefix = f"{category}."
        row = {k: v for k, v in row.items() if k.startswith(prefix) or k in ["run_id", "experiment_name"]}

    if as_json:
        # Convert to JSON-serializable format
        for k, v in row.items():
            if pd.isna(v):
                row[k] = None
        click.echo(json.dumps(row, indent=2, default=str))
    else:
        click.echo(f"Run ID: {row.get('run_id', 'N/A')}")
        click.echo(f"Experiment: {row.get('experiment_name', 'N/A')}")
        click.echo("=" * 60)

        # Group by prefix
        groups: dict[str, dict] = {}
        for k, v in sorted(row.items()):
            if "." in k:
                prefix, key = k.split(".", 1)
                groups.setdefault(prefix, {})[key] = v
            else:
                groups.setdefault("general", {})[k] = v

        for group_name, group_data in sorted(groups.items()):
            if category and group_name != category and group_name != "general":
                continue
            click.echo(f"\n[{group_name}]")
            for k, v in sorted(group_data.items()):
                # Truncate very long values
                v_str = str(v)
                if len(v_str) > 100:
                    v_str = v_str[:100] + "..."
                click.echo(f"  {k}: {v_str}")


@cli.command()
@click.pass_context
def tables(ctx):
    """List all tables in the database."""
    import sqlite3

    with sqlite3.connect(ctx.obj["db_path"]) as conn:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

    if not tables:
        click.echo("No tables found in database.")
        return

    click.echo("Tables in database:")
    for table in tables:
        # Get row count
        with sqlite3.connect(ctx.obj["db_path"]) as conn:
            cursor = conn.execute(f"SELECT COUNT(*) FROM '{table}'")
            count = cursor.fetchone()[0]
        click.echo(f"  - {table} ({count} rows)")


@cli.command()
@click.argument("table_name")
@click.option("--limit", "-n", default=10, type=int, help="Max rows to show")
@click.pass_context
def query(ctx, table_name: str, limit: int):
    """Show raw data from a specific table."""
    import sqlite3

    try:
        with sqlite3.connect(ctx.obj["db_path"]) as conn:
            df = pd.read_sql(f"SELECT * FROM '{table_name}' LIMIT {limit}", conn)
    except pd.io.sql.DatabaseError as e:
        click.echo(f"Error querying table '{table_name}': {e}", err=True)
        return

    if df.empty:
        click.echo(f"Table '{table_name}' is empty.")
        return

    with pd.option_context(
        "display.max_columns", 10,
        "display.width", 200,
        "display.max_colwidth", 30,
    ):
        click.echo(df.to_string(index=False))

    click.echo(f"\nColumns: {', '.join(df.columns.tolist())}")


@cli.command()
@click.argument("run_id")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def results(ctx, run_id: str, as_json: bool):
    """Show model results for a specific run.

    RUN_ID can be a partial match (prefix).
    Displays the model results stored from end_to_end_results.model_results.
    """
    df = load_benchmark_runs(db_path=ctx.obj["db_path"])

    if df.empty:
        click.echo("No benchmark runs found.")
        return

    # Filter by run_id prefix
    mask = df["run_id"].str.startswith(run_id)
    df_filtered = df[mask]

    if df_filtered.empty:
        click.echo(f"No run found with run_id starting with '{run_id}'")
        return

    if len(df_filtered) > 1:
        click.echo(f"Multiple runs match '{run_id}'. Please be more specific:")
        for rid in df_filtered["run_id"].tolist():
            click.echo(f"  - {rid}")
        return

    row = df_filtered.iloc[0]

    # Get datasets and results_json
    datasets = row.get("datasets", "N/A")
    results_json = row.get("results_json", None)

    if as_json:
        output = {
            "run_id": row["run_id"],
            "experiment_name": row.get("experiment_name", "N/A"),
            "datasets": json.loads(datasets) if datasets and datasets != "N/A" else None,
            "results": json.loads(results_json) if results_json else None,
        }
        click.echo(json.dumps(output, indent=2, default=str))
    else:
        click.echo(f"Run ID: {row['run_id']}")
        click.echo(f"Experiment: {row.get('experiment_name', 'N/A')}")
        click.echo(f"Datasets: {datasets}")
        click.echo("=" * 60)

        if results_json:
            try:
                results_dict = json.loads(results_json)
                # Convert to DataFrame for nice display
                df_results = pd.DataFrame(results_dict)
                with pd.option_context(
                    "display.max_rows", None,
                    "display.max_columns", None,
                    "display.width", 200,
                    "display.max_colwidth", 50,
                ):
                    click.echo("Model Results:")
                    click.echo(df_results.to_string())
            except (json.JSONDecodeError, ValueError) as e:
                click.echo(f"Error parsing results: {e}")
                click.echo(f"Raw results: {results_json[:500]}...")
        else:
            click.echo("No results data found for this run.")


@cli.command()
@click.option("--experiment", "-e", default=None, help="Filter by experiment name")
@click.option("--stage", "-s", default="model_fit", help="Timing stage to aggregate (default: model_fit)")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option(
    "--include-profiled",
    is_flag=True,
    default=False,
    help="Include runs with cProfile or cuml.accel profiling enabled (excluded by default).",
)
@click.option(
    "--no-infer-gpu-count",
    is_flag=True,
    default=False,
    help="Don't infer num_gpus from cuda_device_count when num_gpus is not set.",
)
@click.pass_context
def aggregate(ctx, experiment: str | None, stage: str, as_json: bool, include_profiled: bool, no_infer_gpu_count: bool):
    """Aggregate timing results across multiple runs using median.

    Shows median timing for the specified stage grouped by dataset and GPU count.
    Also includes median time_train_s and time_infer_s from model results.

    By default, runs with cProfile or cuml.accel --profile enabled are excluded
    to avoid skewed timing results. Use --include-profiled to include them.
    """
    # Load runs and timings
    df_runs = load_benchmark_runs(experiment_name=experiment, db_path=ctx.obj["db_path"])
    df_timings = load_benchmark_timings(experiment_name=experiment, db_path=ctx.obj["db_path"])

    if df_runs.empty or df_timings.empty:
        click.echo("No benchmark data found.")
        return

    # Filter timings for the specified stage
    df_stage = df_timings[df_timings["stage"] == stage].copy()

    # Filter out profiled runs unless explicitly included
    if not include_profiled:
        df_stage = _parse_profiling_flags(df_stage)
        profiled_mask = df_stage["profiling_cprofile"] | df_stage["profiling_cuml_accel_profile"]
        excluded_count = profiled_mask.sum()
        df_stage = df_stage[~profiled_mask]

        if excluded_count > 0:
            click.echo(f"Note: Excluded {excluded_count} profiled run(s). Use --include-profiled to include them.")

    if df_stage.empty:
        click.echo(f"No timing data found for stage '{stage}'.")
        return

    # Use num_gpus from stage_metadata (experiment config), rename for clarity
    if "stage_metadata.num_gpus" in df_stage.columns:
        df_stage = df_stage.rename(columns={"stage_metadata.num_gpus": "num_gpus"})
        # Convert to numeric (may be stored as string due to SQLite/pandas type handling)
        df_stage["num_gpus"] = pd.to_numeric(df_stage["num_gpus"], errors="coerce")

    # Infer num_gpus from cuda.device_count if not set (enabled by default)
    if not no_infer_gpu_count and "num_gpus" in df_stage.columns:
        df_stage = _infer_num_gpus_from_cuda_device_count(df_stage, df_runs)

    # Extract time_train_s and time_infer_s from results_json
    results_data = []
    for _, row in df_runs.iterrows():
        results_json = row.get("results_json")
        if results_json:
            try:
                results_dict = json.loads(results_json)
                # results_dict is in column-oriented format, extract values
                time_train_values = results_dict.get("time_train_s", {})
                time_infer_values = results_dict.get("time_infer_s", {})
                # Each key is a row index, get all values
                for idx in time_train_values.keys():
                    results_data.append({
                        "run_id": row["run_id"],
                        "time_train_s": time_train_values.get(idx),
                        "time_infer_s": time_infer_values.get(idx),
                    })
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

    df_results = pd.DataFrame(results_data) if results_data else pd.DataFrame()

    # Merge with runs to get datasets
    df_runs_subset = df_runs[["run_id", "datasets"]].copy()
    df_merged = df_stage.merge(df_runs_subset, on="run_id", how="left")

    # Merge results timing data
    if not df_results.empty:
        df_merged = df_merged.merge(df_results, on="run_id", how="left")

    # Determine groupby columns based on what's available
    groupby_cols = ["datasets"]
    if "num_gpus" in df_merged.columns:
        groupby_cols.append("num_gpus")

    # Build aggregation dict
    agg_dict = {
        "median_time_s": ("time_s", "median"),
        "count": ("time_s", "count"),
    }
    if "time_train_s" in df_merged.columns:
        agg_dict["median_time_train_s"] = ("time_train_s", "median")
    if "time_infer_s" in df_merged.columns:
        agg_dict["median_time_infer_s"] = ("time_infer_s", "median")

    # Group by datasets and num_gpus, compute median
    df_agg = (
        df_merged.groupby(groupby_cols, dropna=False)
        .agg(**agg_dict)
        .reset_index()
    )

    # Sort by datasets
    df_agg = df_agg.sort_values("datasets")

    # Rename for display
    df_agg = df_agg.rename(columns={"median_time_s": f"{stage}_time_s"})

    if as_json:
        click.echo(df_agg.to_json(orient="records", indent=2))
    else:
        with pd.option_context(
            "display.max_columns", None,
            "display.width", 200,
            "display.max_colwidth", 50,
        ):
            click.echo(df_agg.to_string(index=False))


# Known dataset sizes from OpenML (rows, columns)
DATASET_SIZES = {
    '["anneal"]': (598, 38),
    '["APSFailure"]': (50666, 170),
    '["credit-g"]': (666, 20),
    '["customer_satisfaction_in_airline"]': (86586, 21),
    '["diabetes"]': (512, 8),
}


@cli.command()
@click.option("--experiment", "-e", default=None, help="Filter by experiment name")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option(
    "--include-profiled",
    is_flag=True,
    default=False,
    help="Include runs with cProfile or cuml.accel profiling enabled (excluded by default).",
)
@click.option(
    "--no-infer-gpu-count",
    is_flag=True,
    default=False,
    help="Don't infer num_gpus from cuda_device_count when num_gpus is not set.",
)
@click.pass_context
def speedup(ctx, experiment: str | None, as_json: bool, include_profiled: bool, no_infer_gpu_count: bool):
    """Compare GPU timing speedup against CPU baseline (num_gpus=0).

    Shows speedup ratios for time_train_s and time_infer_s from model results,
    grouped by dataset and GPU count. A speedup > 1 means the GPU is faster.

    By default, runs with cProfile or cuml.accel --profile enabled are excluded
    to avoid skewed timing results. Use --include-profiled to include them.
    """
    # Load runs and timings
    df_runs = load_benchmark_runs(experiment_name=experiment, db_path=ctx.obj["db_path"])
    df_timings = load_benchmark_timings(experiment_name=experiment, db_path=ctx.obj["db_path"])

    if df_runs.empty or df_timings.empty:
        click.echo("No benchmark data found.")
        return

    # Use model_fit stage for filtering profiled runs and getting num_gpus
    df_stage = df_timings[df_timings["stage"] == "model_fit"].copy()

    # Filter out profiled runs unless explicitly included
    if not include_profiled:
        df_stage = _parse_profiling_flags(df_stage)
        profiled_mask = df_stage["profiling_cprofile"] | df_stage["profiling_cuml_accel_profile"]
        excluded_count = profiled_mask.sum()
        df_stage = df_stage[~profiled_mask]

        if excluded_count > 0:
            click.echo(f"Note: Excluded {excluded_count} profiled run(s). Use --include-profiled to include them.")

    if df_stage.empty:
        click.echo("No timing data found.")
        return

    # Use num_gpus from stage_metadata (experiment config), rename for clarity
    if "stage_metadata.num_gpus" in df_stage.columns:
        df_stage = df_stage.rename(columns={"stage_metadata.num_gpus": "num_gpus"})

    if "num_gpus" not in df_stage.columns:
        click.echo("No num_gpus data found in timing metadata.")
        return

    # Convert num_gpus to numeric (may be stored as string due to SQLite/pandas type handling)
    df_stage["num_gpus"] = pd.to_numeric(df_stage["num_gpus"], errors="coerce")

    # Infer num_gpus from cuda.device_count if not set (enabled by default)
    if not no_infer_gpu_count:
        df_stage = _infer_num_gpus_from_cuda_device_count(df_stage, df_runs)

    # Extract time_train_s and time_infer_s from results_json
    results_data = []
    for _, row in df_runs.iterrows():
        results_json = row.get("results_json")
        if results_json:
            try:
                results_dict = json.loads(results_json)
                time_train_values = results_dict.get("time_train_s", {})
                time_infer_values = results_dict.get("time_infer_s", {})
                for idx in time_train_values.keys():
                    results_data.append({
                        "run_id": row["run_id"],
                        "time_train_s": time_train_values.get(idx),
                        "time_infer_s": time_infer_values.get(idx),
                    })
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

    if not results_data:
        click.echo("No model results data found.")
        return

    df_results = pd.DataFrame(results_data)

    # Merge with runs to get datasets, and with stage to get num_gpus
    # Use inner merge with df_stage_subset to exclude profiled runs (already filtered out of df_stage)
    df_runs_subset = df_runs[["run_id", "datasets"]].copy()
    df_stage_subset = df_stage[["run_id", "num_gpus"]].copy()

    df_merged = df_results.merge(df_runs_subset, on="run_id", how="left")
    df_merged = df_merged.merge(df_stage_subset, on="run_id", how="inner")

    # Group by datasets and num_gpus, compute median
    df_agg = (
        df_merged.groupby(["datasets", "num_gpus"], dropna=False)
        .agg(
            median_time_train_s=("time_train_s", "median"),
            median_time_infer_s=("time_infer_s", "median"),
            count=("time_train_s", "count"),
        )
        .reset_index()
    )

    # Extract baseline (num_gpus == 0) timing for each dataset
    df_baseline = df_agg[df_agg["num_gpus"] == 0][["datasets", "median_time_train_s", "median_time_infer_s"]].copy()
    df_baseline = df_baseline.rename(columns={
        "median_time_train_s": "baseline_train_s",
        "median_time_infer_s": "baseline_infer_s",
    })

    if df_baseline.empty:
        click.echo("No baseline (num_gpus=0) data found. Cannot compute speedup.")
        return

    # Merge baseline timing back to all rows
    df_speedup = df_agg.merge(df_baseline, on="datasets", how="left")

    # Calculate speedup (baseline / gpu_time)
    df_speedup["speedup_train"] = df_speedup["baseline_train_s"] / df_speedup["median_time_train_s"]
    df_speedup["speedup_infer"] = df_speedup["baseline_infer_s"] / df_speedup["median_time_infer_s"]

    # Filter out baseline rows (speedup would always be 1.0)
    df_speedup = df_speedup[df_speedup["num_gpus"] != 0].copy()

    if df_speedup.empty:
        click.echo("No GPU runs found to compare against baseline.")
        return

    # Add dataset size information
    df_speedup["rows"] = df_speedup["datasets"].map(lambda x: DATASET_SIZES.get(x, (None, None))[0])
    df_speedup["cols"] = df_speedup["datasets"].map(lambda x: DATASET_SIZES.get(x, (None, None))[1])

    # Sort by datasets and num_gpus
    df_speedup = df_speedup.sort_values(["datasets", "num_gpus"])

    # Rename for display
    df_speedup = df_speedup.rename(columns={
        "median_time_train_s": "time_train_s",
        "median_time_infer_s": "time_infer_s",
    })

    # Select columns for display
    display_cols = [
        "datasets", "rows", "cols", "num_gpus",
        "baseline_train_s", "speedup_train",
        "baseline_infer_s", "speedup_infer",
        "count",
    ]
    df_display = df_speedup[display_cols]

    if as_json:
        click.echo(df_display.to_json(orient="records", indent=2))
    else:
        with pd.option_context(
            "display.max_columns", None,
            "display.width", 250,
            "display.max_colwidth", 50,
            "display.float_format", "{:.2f}".format,
        ):
            click.echo(df_display.to_string(index=False))


if __name__ == "__main__":
    cli()
