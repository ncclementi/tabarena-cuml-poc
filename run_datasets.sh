#!/bin/bash
set -eu

# Default values
CUML_PROFILE=false
CPROFILE=false
EXPERIMENT_ID=""
NUM_GPUS=""

# Usage function
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --cuml-profile       Enable cuml.accel profiling (--profile flag)"
    echo "  --cprofile           Enable cProfile profiling (saves .prof files)"
    echo "  --experiment-id ID   Use a custom experiment ID (default: auto-generated timestamp)"
    echo "  --num-gpus N         Number of GPUs to use (0 for CPU-only, omit for default)"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                              # Run without profiling"
    echo "  $0 --cuml-profile               # Run with cuml.accel profiling"
    echo "  $0 --cprofile                   # Run with cProfile profiling"
    echo "  $0 --cuml-profile --cprofile    # Run with both profiling modes"
    echo "  $0 --experiment-id my_exp_001   # Run with custom experiment ID"
    echo "  $0 --num-gpus 0                 # Run on CPU only (no cuml.accel)"
    echo "  $0 --num-gpus 1                 # Run with 1 GPU"
    echo ""
    echo "Note: --num-gpus 0 and --cuml-profile cannot be used together."
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
        --num-gpus)
            NUM_GPUS="$2"
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

# Validate incompatible options
if [ "$CUML_PROFILE" = true ] && [ "$NUM_GPUS" = "0" ]; then
    echo "Error: --num-gpus 0 and --cuml-profile cannot be used together."
    echo "cuml.accel profiling requires GPU acceleration (num_gpus > 0)."
    exit 1
fi

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
if [ -n "$NUM_GPUS" ]; then
    echo "  num_gpus: $NUM_GPUS"
else
    echo "  num_gpus: (default)"
fi
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

    # Build command incrementally
    CMD="python"

    # Add cProfile module if enabled
    if [ "$CPROFILE" = true ]; then
        PROFILE_FILE="$PROFILES_DIR/${dataset}.prof"
        CMD="$CMD -m cProfile -o $PROFILE_FILE"
    fi

    # Add cuml.accel module if GPU mode (NUM_GPUS != 0)
    if [ "$NUM_GPUS" != "0" ]; then
        CMD="$CMD -m cuml.accel"
        [ "$CUML_PROFILE" = true ] && CMD="$CMD --profile"
    fi

    # Add script path
    CMD="$CMD $BENCHMARK_SCRIPT"

    # Add script arguments
    CMD="$CMD --datasets $dataset --experiment-name $EXPERIMENT_NAME"
    [ -n "$NUM_GPUS" ] && CMD="$CMD --num-gpus $NUM_GPUS"
    [ "$CPROFILE" = true ] && CMD="$CMD --metadata cprofile=true"
    [ "$CUML_PROFILE" = true ] && CMD="$CMD --metadata cuml_accel_profile=true"

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
