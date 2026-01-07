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


@cli.command()
@click.option("--experiment", "-e", default=None, help="Filter by experiment name")
@click.option("--limit", "-n", default=20, type=int, help="Max rows to show")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def runs(ctx, experiment: str | None, limit: int, as_json: bool):
    """List benchmark runs with summary info."""
    df = load_benchmark_runs(experiment_name=experiment, db_path=ctx.obj["db_path"])

    if df.empty:
        click.echo("No benchmark runs found.")
        return

    # Select key columns for display
    display_cols = [
        "run_id",
        "experiment_name",
        "execution_datetime",
        "total_time_s",
        "datasets",
        "system.hostname",
        "cuda.cuda_device_count",
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


if __name__ == "__main__":
    cli()
