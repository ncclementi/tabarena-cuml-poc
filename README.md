# tabarena-cuml-poc
Benchmarking sklearn estimators accelerated via cuml accel 

## Setup

To set up the environment in a reproducible way, run the provided setup script:

```bash
./setup.sh
```

**Requirements:**
- CUDA 13 must be installed and `nvcc` must be in your PATH
- The script will automatically install `uv` if not already present

**What the script does:**
1. Checks for and installs `uv` if needed
2. Creates a Python 3.12 virtual environment
3. Verifies CUDA 13 is available
4. Installs PyTorch with CUDA 13 support
5. Clones/updates AutoGluon from https://github.com/csadorf/autogluon
6. Clones/updates TabArena from https://github.com/csadorf/tabarena
7. (optional) Builds and installs treelite from source
8. Installs cuML-cu13 version 25.12.00

After setup completes, activate the environment:
```bash
source .venv/bin/activate
```

## Running the cuML-accelerated benchmark quickstart

To test cuML-accelerated estimators with TabArena, run the quickstart script:

```bash
./quickstart.sh
```

This will run the cuML Random Forest benchmark on a few small datasets.
You can modify the script to experiment with other models or datasets as needed.

### Running benchmarks

Run benchmarks on multiple datasets using the `run.sh` script:

```bash
# Run without profiling (GPU-accelerated by default if GPUs are present)
./run.sh

# Run with cuml.accel profiling
./run.sh --cuml-profile

# Run with cProfile profiling
./run.sh --cprofile

# Run with both profiling modes
./run.sh --cuml-profile --cprofile

# Run on CPU only (no cuml.accel)
./run.sh --num-gpus 0

# Use a custom experiment ID
./run.sh --experiment-id my_experiment_001
```

Results are saved to `results/<experiment-id>/` and cProfile files (when enabled) to `cprofiles/<experiment-id>/`.

### Show benchmark results

Benchmark results are stored in two locations:

1. **On disk** (`results/<experiment-id>/`): Raw output files for each dataset
   - `<dataset>_output.txt` - Full console output from the benchmark run
   - `<dataset>_results.txt` - Extracted results section (metrics and timing)

2. **SQLite database** (`benchmark_results.db`): Structured data for analysis
   - `benchmark_runs` table - One row per run with metadata, config, and results JSON
   - `benchmark_timings` table - Detailed timing breakdown per stage

Use `scripts/show_results.py` to query and analyze results from the database:

```bash
# List all benchmark runs
./scripts/show_results.py runs

# Show results for a specific run (partial run_id prefix works)
./scripts/show_results.py results 23eccace

# Show timing breakdown for a run
./scripts/show_results.py timings 23eccace

# Show full metadata for a run
./scripts/show_results.py info 23eccace
```

#### Aggregating results across runs

The `aggregate` command computes median timings grouped by dataset and GPU count, making it easy to compare CPU vs GPU performance:

```bash
./scripts/show_results.py aggregate
```

Example output:
```
                            datasets  num_gpus  model_fit_time_s  count  median_time_train_s  median_time_infer_s
                      ["APSFailure"]       0.0         28.175330      1            23.699634             0.667471
                      ["APSFailure"]       1.0         17.705651      1            10.998923             0.668610
                          ["anneal"]       0.0          6.633581      1             3.224594             0.393610
                          ["anneal"]       1.0          6.279709      1             2.617248             0.276223
```

The `num_gpus` column shows results for CPU-only runs (`0.0`), GPU runs (`1.0`, `2.0`), and when no "num_gpus" argument was provided (`NaN`). This allows you to quickly see the speedup achieved by GPU acceleration.

Additional options:
```bash
# Filter by experiment name
./scripts/show_results.py aggregate -e my_experiment

# Output as JSON
./scripts/show_results.py aggregate --json
```

#### Computing GPU speedup

The `speedup` command compares GPU timing against CPU baseline (num_gpus=0) to show performance differences:

```bash
./scripts/show_results.py speedup
```

Example output:
```
                                     datasets   rows   cols  num_gpus  baseline_train_s  speedup_train  baseline_infer_s  speedup_infer  count
                               ["APSFailure"]  50666    170       1.0             23.70           2.15              0.67           1.00      1
                                   ["anneal"]    598     38       1.0              3.22           1.23              0.39           1.42      1
  ["customer_satisfaction_in_airline"]         86586     21       1.0             45.12           3.50              1.20           1.80      1
```

A speedup > 1 means the GPU run was faster than the CPU baseline. The command also displays dataset dimensions (rows, cols) to help correlate speedup with dataset size.

Additional options:
```bash
# Use minimum instead of median for aggregation
./scripts/show_results.py speedup --agg min

# Filter by experiment name
./scripts/show_results.py speedup -e my_experiment

# Include profiled runs (excluded by default to avoid skewed results)
./scripts/show_results.py speedup --include-profiled

# Output as JSON
./scripts/show_results.py speedup --json
```

#### Relationship between disk and database results

The disk files (`results/`) contain the raw output for debugging and manual inspection, while the database (`benchmark_results.db`) stores the same data in a structured format for programmatic analysis. Both are created during the same benchmark run—the database is updated via `benchmark_db.save_experiment_results()` at the end of each experiment.


## TODO: 
- [x] Setup script to install everything in a reproducible way using uv 
    - [x] cuml from nightly, specific version
    - [x] autogluon from https://github.com/csadorf/autogluon main
    - [x] tabarena from https://github.com/csadorf/tabarena main
- [ ] Separately create a test for installation for LR, KNN, RF
    - [ ] one simple dataset that runs with pure autogluon cpu 
    - [ ] one that runs with cuml accel POC 
