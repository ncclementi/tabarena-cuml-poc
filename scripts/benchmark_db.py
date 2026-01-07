"""Database utilities for storing benchmark results in SQLite."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from benchmark_timer import BenchmarkTimer


# Default database filename
DEFAULT_DB_NAME = "benchmark_results.db"


def get_database_path() -> Path:
    """Return path to benchmark_results.db in project root.

    Returns
    -------
    Path
        Path to the SQLite database file.
    """
    # Navigate from scripts/ to project root
    scripts_dir = Path(__file__).parent
    project_root = scripts_dir.parent
    return project_root / DEFAULT_DB_NAME


@contextmanager
def get_database_connection(db_path: Path | None = None):
    """Context manager for SQLite database connection.

    Parameters
    ----------
    db_path : Path | None
        Path to the database file. If None, uses the default path
        returned by get_database_path().

    Yields
    ------
    sqlite3.Connection
        SQLite database connection.
    """
    if db_path is None:
        db_path = get_database_path()

    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


def save_experiment_results(
    timer: BenchmarkTimer,
    results: dict[str, Any],
    datasets: list[str] | None = None,
    db_path: Path | None = None,
) -> None:
    """Store experiment results with timing and metadata in SQLite.

    Parameters
    ----------
    timer : BenchmarkTimer
        Timer instance containing timing data and environment metadata.
    results : dict[str, Any]
        Experiment results to store (will be JSON serialized).
    datasets : list[str] | None
        List of dataset names used in the experiment.
    db_path : Path | None
        Path to the database file. If None, uses default path.

    Notes
    -----
    Creates two tables:
    - benchmark_runs: One row per experiment run with metadata and serialized results
    - benchmark_timings: One row per timed stage, linked by run_id
    """
    # Record total time before saving
    timer.record_total_time()

    with get_database_connection(db_path) as conn:
        # Save timing data
        timer.to_sql(conn, table="benchmark_timings")

        # Save experiment run summary
        run_summary = {
            "run_id": timer.run_id,
            "experiment_name": timer.experiment_name,
            "datasets": json.dumps(datasets) if datasets else None,
            "results_json": json.dumps(results, default=str),
            **timer.context,
        }

        # Add aggregated timing info
        df_timings = timer.to_df()
        if not df_timings.empty:
            total_row = df_timings[df_timings["stage"] == "total"]
            if not total_row.empty:
                run_summary["total_time_s"] = total_row["time_s"].iloc[0]
                run_summary["total_time_ms"] = total_row["time_ms"].iloc[0]

        # Serialize any list/dict values for SQLite compatibility
        for key, value in run_summary.items():
            if isinstance(value, (list, dict)):
                run_summary[key] = json.dumps(value, default=str)

        df_run = pd.DataFrame([run_summary])
        _save_dataframe_with_schema_evolution(df_run, conn, "benchmark_runs")

        conn.commit()

    print(f"Results saved to {get_database_path() if db_path is None else db_path}")


def _save_dataframe_with_schema_evolution(
    df: pd.DataFrame,
    conn: sqlite3.Connection,
    table: str,
) -> None:
    """Save DataFrame to SQLite with automatic schema evolution.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    conn : sqlite3.Connection
        SQLite database connection.
    table : str
        Name of the table.
    """
    if len(df) == 0:
        return

    try:
        df.to_sql(table, conn, if_exists="append", index=False)
    except sqlite3.OperationalError as error:
        if "has no column" in str(error):
            # Schema evolution: merge with existing data
            df_existing = pd.read_sql(f"SELECT * FROM '{table}'", conn)
            df = pd.concat([df_existing, df], ignore_index=True, sort=False)
            df.to_sql(table, conn, if_exists="replace", index=False)
        else:
            raise


def load_benchmark_runs(
    experiment_name: str | None = None,
    db_path: Path | None = None,
) -> pd.DataFrame:
    """Load benchmark run summaries from the database.

    Parameters
    ----------
    experiment_name : str | None
        Filter by experiment name. If None, loads all runs.
    db_path : Path | None
        Path to the database file. If None, uses default path.

    Returns
    -------
    pd.DataFrame
        DataFrame containing benchmark run summaries.
    """
    with get_database_connection(db_path) as conn:
        try:
            if experiment_name:
                query = f"SELECT * FROM benchmark_runs WHERE experiment_name = ?"
                return pd.read_sql(query, conn, params=(experiment_name,))
            else:
                return pd.read_sql("SELECT * FROM benchmark_runs", conn)
        except pd.io.sql.DatabaseError:
            return pd.DataFrame()


def load_benchmark_timings(
    run_id: str | None = None,
    experiment_name: str | None = None,
    db_path: Path | None = None,
) -> pd.DataFrame:
    """Load benchmark timing data from the database.

    Parameters
    ----------
    run_id : str | None
        Filter by specific run ID.
    experiment_name : str | None
        Filter by experiment name.
    db_path : Path | None
        Path to the database file. If None, uses default path.

    Returns
    -------
    pd.DataFrame
        DataFrame containing timing data.
    """
    with get_database_connection(db_path) as conn:
        try:
            query = "SELECT * FROM benchmark_timings WHERE 1=1"
            params = []

            if run_id:
                query += " AND run_id = ?"
                params.append(run_id)
            if experiment_name:
                query += " AND experiment_name = ?"
                params.append(experiment_name)

            return pd.read_sql(query, conn, params=params if params else None)
        except pd.io.sql.DatabaseError:
            return pd.DataFrame()
