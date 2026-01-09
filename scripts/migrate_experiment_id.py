#!/usr/bin/env python
"""One-off migration script to backfill experiment_id from experiment_name.

This script extracts experiment_id from experiment_name (format: ${experiment_id}_${dataset_name})
and updates the stage_metadata.experiment_id column in the benchmark_timings table.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

# Add scripts directory to path for local module imports
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

from benchmark_db import get_database_path


def extract_experiment_id(experiment_name: str, datasets_json: str | None) -> str | None:
    """Extract experiment_id from experiment_name.

    The experiment_name format is: ${experiment_id}_${dataset_name}
    
    Parameters
    ----------
    experiment_name : str
        Full experiment name (e.g., "d5c9b4244ef14e0480a3c6815c08e803_customer_satisfaction_in_airline")
    datasets_json : str | None
        JSON string containing list of datasets (e.g., '["customer_satisfaction_in_airline"]')
    
    Returns
    -------
    str | None
        Extracted experiment_id, or None if extraction fails
    """
    if not experiment_name:
        return None
    
    # Try to use datasets to find the suffix
    if datasets_json:
        try:
            datasets = json.loads(datasets_json)
            if datasets and len(datasets) > 0:
                dataset = datasets[0]
                suffix = f"_{dataset}"
                if experiment_name.endswith(suffix):
                    return experiment_name[:-len(suffix)]
        except (json.JSONDecodeError, TypeError, IndexError):
            pass
    
    # Fallback: assume experiment_id is a 32-char hex string at the start
    # Split on first underscore and check if first part looks like a UUID
    if "_" in experiment_name:
        first_part = experiment_name.split("_", 1)[0]
        # Check if it looks like a hex UUID (32 chars, all hex)
        if len(first_part) == 32 and all(c in "0123456789abcdef" for c in first_part.lower()):
            return first_part
    
    return None


def migrate_experiment_ids(db_path: Path | None = None, dry_run: bool = False) -> None:
    """Migrate experiment_id values in the database.
    
    Parameters
    ----------
    db_path : Path | None
        Path to database. If None, uses default.
    dry_run : bool
        If True, only show what would be done without making changes.
    """
    if db_path is None:
        db_path = get_database_path()
    
    print(f"Database: {db_path}")
    print(f"Dry run: {dry_run}")
    print("=" * 60)
    
    conn = sqlite3.connect(db_path)
    
    try:
        # Check if the column exists
        cursor = conn.execute("PRAGMA table_info(benchmark_timings)")
        columns = [row[1] for row in cursor.fetchall()]
        
        has_experiment_id_col = "stage_metadata.experiment_id" in columns
        print(f"Column 'stage_metadata.experiment_id' exists: {has_experiment_id_col}")
        
        # Load timings data
        df_timings = pd.read_sql("SELECT * FROM benchmark_timings", conn)
        print(f"Total rows in benchmark_timings: {len(df_timings)}")
        
        if df_timings.empty:
            print("No data to migrate.")
            return
        
        # Load runs data to get datasets for each run_id
        df_runs = pd.read_sql("SELECT run_id, datasets FROM benchmark_runs", conn)
        run_to_datasets = dict(zip(df_runs["run_id"], df_runs["datasets"]))
        
        # Find rows missing experiment_id
        if has_experiment_id_col:
            missing_mask = df_timings["stage_metadata.experiment_id"].isna()
        else:
            missing_mask = pd.Series([True] * len(df_timings))
        
        missing_count = missing_mask.sum()
        print(f"Rows missing experiment_id: {missing_count}")
        
        if missing_count == 0:
            print("Nothing to migrate!")
            return
        
        # Extract experiment_id for missing rows
        updates = []
        for idx, row in df_timings[missing_mask].iterrows():
            experiment_name = row.get("experiment_name", "")
            run_id = row.get("run_id", "")
            datasets_json = run_to_datasets.get(run_id)
            
            experiment_id = extract_experiment_id(experiment_name, datasets_json)
            
            if experiment_id:
                updates.append({
                    "idx": idx,
                    "run_id": run_id,
                    "experiment_name": experiment_name,
                    "experiment_id": experiment_id,
                })
        
        print(f"Successfully extracted experiment_id for {len(updates)} rows")
        
        # Show sample of updates
        print("\nSample updates:")
        for update in updates[:5]:
            print(f"  {update['experiment_name'][:50]}... -> {update['experiment_id']}")
        if len(updates) > 5:
            print(f"  ... and {len(updates) - 5} more")
        
        if dry_run:
            print("\nDry run complete. No changes made.")
            return
        
        # Apply updates
        print("\nApplying updates...")
        
        # Add column if it doesn't exist
        if not has_experiment_id_col:
            print("Adding column 'stage_metadata.experiment_id'...")
            conn.execute('ALTER TABLE benchmark_timings ADD COLUMN "stage_metadata.experiment_id" TEXT')
        
        # Update rows
        for update in updates:
            conn.execute(
                'UPDATE benchmark_timings SET "stage_metadata.experiment_id" = ? WHERE rowid = ?',
                (update["experiment_id"], update["idx"] + 1)  # rowid is 1-based
            )
        
        conn.commit()
        print(f"Successfully updated {len(updates)} rows.")
        
    finally:
        conn.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate experiment_id values in benchmark database")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--db", type=Path, default=None, help="Path to database file")
    
    args = parser.parse_args()
    
    migrate_experiment_ids(db_path=args.db, dry_run=args.dry_run)
