#!/bin/bash

# Quickstart script for running TabArena with cuML acceleration
# This script copies the quickstart Python script to the tabarena benchmarking
# directory and runs it with cuML acceleration enabled.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Verify tabarena is properly installed (not just the directory)
if ! python -c "from tabarena.benchmark.experiment import AGModelBagExperiment" 2>/dev/null; then
    echo "ERROR: tabarena is not properly installed."
    echo ""
    echo "Please run setup.sh first:"
    echo "  ./setup.sh"
    exit 1
fi

# Copy the quickstart script to the tabarena benchmarking directory
cp "$SCRIPT_DIR/run_tabarena_rf_experiment.py" "$SCRIPT_DIR/tabarena/examples/benchmarking/"

# Run the quickstart with cuML acceleration
cd "$SCRIPT_DIR/tabarena/examples/benchmarking"
python -m cuml.accel run_tabarena_rf_experiment.py

# Return to original directory
cd "$SCRIPT_DIR"
