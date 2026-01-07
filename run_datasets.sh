#!/bin/bash
set -eu

# Default values
CUML_PROFILE=false
CPROFILE=false
EXPERIMENT_ID=""

# Usage function
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --cuml-profile       Enable cuml.accel profiling (--profile flag)"
    echo "  --cprofile           Enable cProfile profiling (saves .prof files)"
    echo "  --experiment-id ID   Use a custom experiment ID (default: auto-generated timestamp)"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                              # Run without profiling"
    echo "  $0 --cuml-profile               # Run with cuml.accel profiling"
    echo "  $0 --cprofile                   # Run with cProfile profiling"
    echo "  $0 --cuml-profile --cprofile    # Run with both profiling modes"
    echo "  $0 --experiment-id my_exp_001   # Run with custom experiment ID"
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cuml-profile)
            CUML_PROFILE=true
            shift
            ;;
        --cprofile)
            CPROFILE=true
            shift
            ;;
        --experiment-id)
            EXPERIMENT_ID="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Generate experiment ID if not provided (32-character UUID4)
if [ -z "$EXPERIMENT_ID" ]; then
    EXPERIMENT_ID=$(uuidgen | tr -d '-' | tr '[:upper:]' '[:lower:]')
fi

# Define the datasets to run
datasets=("anneal" "credit-g" "diabetes" "APSFailure" "customer_satisfaction_in_airline")

# Base directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BENCHMARK_SCRIPT="$SCRIPT_DIR/scripts/run_tabarena_rf_experiment.py"

# Create experiment-specific output directories
RESULTS_DIR="$SCRIPT_DIR/results/${EXPERIMENT_ID}"
PROFILES_DIR="$SCRIPT_DIR/cprofiles/${EXPERIMENT_ID}"

# Create directories
mkdir -p "$RESULTS_DIR"
if [ "$CPROFILE" = true ]; then
    mkdir -p "$PROFILES_DIR"
fi

# Display experiment configuration
echo "=========================================="
echo "Experiment Configuration:"
echo "  Experiment ID: $EXPERIMENT_ID"
echo "  cuml.accel --profile: $CUML_PROFILE"
echo "  cProfile: $CPROFILE"
echo "  Results directory: $RESULTS_DIR"
if [ "$CPROFILE" = true ]; then
    echo "  Profiles directory: $PROFILES_DIR"
fi
echo "=========================================="
echo ""

# Loop through each dataset
for dataset in "${datasets[@]}"; do
    echo "=========================================="
    echo "Running benchmark for dataset: $dataset"
    echo "=========================================="

    OUTPUT_FILE="$RESULTS_DIR/${dataset}_output.txt"
    EXPERIMENT_NAME="${EXPERIMENT_ID}_${dataset}"

    # Build profiling metadata flags for Python script
    PROFILE_FLAGS=""
    if [ "$CUML_PROFILE" = true ]; then
        PROFILE_FLAGS="$PROFILE_FLAGS --cuml-profile"
    fi
    if [ "$CPROFILE" = true ]; then
        PROFILE_FLAGS="$PROFILE_FLAGS --cprofile"
    fi

    # Build the command based on profiling options
    if [ "$CPROFILE" = true ]; then
        PROFILE_FILE="$PROFILES_DIR/${dataset}.prof"
        if [ "$CUML_PROFILE" = true ]; then
            # Both cProfile and cuml.accel profiling
            CMD="python -m cProfile -o $PROFILE_FILE -m cuml.accel --profile $BENCHMARK_SCRIPT --datasets $dataset --experiment-name $EXPERIMENT_NAME$PROFILE_FLAGS"
        else
            # Only cProfile
            CMD="python -m cProfile -o $PROFILE_FILE -m cuml.accel $BENCHMARK_SCRIPT --datasets $dataset --experiment-name $EXPERIMENT_NAME$PROFILE_FLAGS"
        fi
    else
        if [ "$CUML_PROFILE" = true ]; then
            # Only cuml.accel profiling
            CMD="python -m cuml.accel --profile $BENCHMARK_SCRIPT --datasets $dataset --experiment-name $EXPERIMENT_NAME$PROFILE_FLAGS"
        else
            # No profiling
            CMD="python -m cuml.accel $BENCHMARK_SCRIPT --datasets $dataset --experiment-name $EXPERIMENT_NAME$PROFILE_FLAGS"
        fi
    fi

    echo "Running: $CMD"
    eval "$CMD" 2>&1 | tee "$OUTPUT_FILE"

    if [ "$CPROFILE" = true ]; then
        echo "Profile saved to: $PROFILE_FILE"
    fi

    # Extract the results section (from "Results:" to the end)
    RESULTS_FILE="$RESULTS_DIR/${dataset}_results.txt"
    echo "Extracting results to: $RESULTS_FILE"

    if grep -q "Results:" "$OUTPUT_FILE"; then
        sed -n '/Results:/,$p' "$OUTPUT_FILE" > "$RESULTS_FILE"
        echo "Results extracted successfully for $dataset"
    else
        echo "Warning: Could not find 'Results:' section in output for $dataset"
    fi

    # Clean up: delete AutogluonModels and experiments folders
    echo "Cleaning up AutogluonModels and experiments folders..."
    if [ -d "$SCRIPT_DIR/AutogluonModels" ]; then
        rm -rf "$SCRIPT_DIR/AutogluonModels"
        echo "Deleted AutogluonModels folder"
    fi

    if [ -d "$SCRIPT_DIR/experiments" ]; then
        rm -rf "$SCRIPT_DIR/experiments"
        echo "Deleted experiments folder"
    fi

    echo "Completed benchmark for dataset: $dataset"
    echo ""
done

echo "=========================================="
echo "All benchmarks completed!"
echo "Experiment ID: $EXPERIMENT_ID"
echo "Results saved in: $RESULTS_DIR"
if [ "$CPROFILE" = true ]; then
    echo "Profiles saved in: $PROFILES_DIR"
fi
echo "=========================================="
