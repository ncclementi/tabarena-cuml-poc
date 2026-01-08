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

# Run the tabarena rf experiment with cuML acceleration
python -m cuml.accel "$SCRIPT_DIR/scripts/run_tabarena_rf_experiment.py"

# Return to original directory
cd "$SCRIPT_DIR"
