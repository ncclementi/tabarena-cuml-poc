"""Timer class for benchmarking experiments with timing instrumentation."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from time import perf_counter_ns
from typing import Any
from uuid import uuid4

import pandas as pd

from benchmark_util import collect_all_metadata


def _with_prefix(prefix: str, mapping: dict) -> dict:
    """Add a prefix to all keys in a mapping."""
    return {f"{prefix}{key}": value for key, value in mapping.items()}


def _serialize_for_sqlite(value: Any) -> Any:
    """Serialize non-primitive types to JSON strings for SQLite storage."""
    if isinstance(value, (list, dict)):
        return json.dumps(value, default=str)
    return value


@dataclass
class BenchmarkTimer:
    """Timer for benchmarking experiments with metadata collection.

    Collects environment metadata at initialization and provides a context manager
    for timing specific stages of an experiment.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment being timed.
    metadata : dict
        Additional metadata to include with each timing record.

    Attributes
    ----------
    run_id : str
        Unique identifier for this benchmark run.
    context : dict
        Environment metadata collected at initialization.
    timings : list
        List of timing records for each timed stage.

    Examples
    --------
    >>> timer = BenchmarkTimer(experiment_name="my_experiment")
    >>> with timer.time("model_fit"):
    ...     # training code here
    ...     pass
    >>> with timer.time("evaluation"):
    ...     # evaluation code here
    ...     pass
    >>> print(timer.summary())
    """

    experiment_name: str
    metadata: dict = field(default_factory=dict)
    run_id: str = field(default_factory=lambda: uuid4().hex)
    context: dict = field(default_factory=dict, init=False)
    timings: list = field(default_factory=list, init=False)
    _start_time: float = field(default=0.0, init=False)

    def __post_init__(self):
        """Collect environment metadata and record start time."""
        self._start_time = perf_counter_ns()
        self.context = collect_all_metadata()

    @contextmanager
    def time(self, stage: str, metadata: dict | None = None):
        """Time a specific stage of the experiment.

        Parameters
        ----------
        stage : str
            Name of the stage being timed (e.g., 'fit', 'predict', 'evaluate').
        metadata : dict | None
            Additional metadata specific to this stage. Overrides class-level
            metadata for duplicate keys.

        Yields
        ------
        None
            Context manager that records timing on exit.
        """
        # Merge class-level metadata with stage-specific metadata
        stage_metadata = deepcopy(self.metadata)
        if metadata:
            stage_metadata.update(metadata)

        start = perf_counter_ns()
        try:
            yield
        finally:
            elapsed_ns = perf_counter_ns() - start
            self.timings.append({
                "stage": stage,
                "time_ns": elapsed_ns,
                "time_ms": elapsed_ns / 1e6,
                "time_s": elapsed_ns / 1e9,
                "timestamp": datetime.now().isoformat(),
                **_with_prefix("stage_metadata.", stage_metadata),
            })

    def record_total_time(self):
        """Record the total elapsed time since timer initialization."""
        total_ns = perf_counter_ns() - self._start_time
        self.timings.append({
            "stage": "total",
            "time_ns": total_ns,
            "time_ms": total_ns / 1e6,
            "time_s": total_ns / 1e9,
            "timestamp": datetime.now().isoformat(),
        })

    def to_df(self) -> pd.DataFrame:
        """Convert timings and context to a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with one row per timed stage, including all context
            metadata as columns.
        """
        table = deepcopy(self.timings)
        for row in table:
            row["experiment_name"] = self.experiment_name
            row["run_id"] = self.run_id
            row.update(self.context)
            # Serialize non-primitive types for SQLite compatibility
            for key, value in row.items():
                row[key] = _serialize_for_sqlite(value)

        return pd.DataFrame(table)

    def summary(self) -> str:
        """Return a human-readable summary of timing results.

        Returns
        -------
        str
            Formatted string showing stage timings.
        """
        df = self.to_df()
        if df.empty:
            return "No timings recorded."

        cols = ["timestamp", "experiment_name", "stage", "time_ms", "time_s"]
        available_cols = [c for c in cols if c in df.columns]
        to_print = df[available_cols].copy()
        return to_print.to_string(index=False)

    def to_json(self) -> str:
        """Convert results to JSON string.

        Returns
        -------
        str
            JSON representation of the timing data.
        """
        return self.to_df().to_json(orient="records", indent=2)

    def to_sql(self, conn, table: str = "benchmark_timings"):
        """Store timing results in a SQLite database.

        Parameters
        ----------
        conn : sqlite3.Connection
            SQLite database connection.
        table : str
            Name of the table to store results in.

        Notes
        -----
        If the table exists but has different columns, the table schema
        will be updated to accommodate new columns.
        """
        df = self.to_df()

        if len(df) == 0:
            return  # nothing to store

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
