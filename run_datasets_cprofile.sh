#!/bin/bash

# Define the datasets to run
datasets=("anneal" "credit-g" "diabetes" "APSFailure" "customer_satisfaction_in_airline")

# Base directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BENCHMARK_SCRIPT="$SCRIPT_DIR/run_quickstart_tabarena_cuml.py"
RESULTS_DIR="$SCRIPT_DIR/results_per_dataset_cprofile"
PROFILES_DIR="$SCRIPT_DIR/cprofiles_per_dataset"

# Create results and profiles directories if they don't exist
mkdir -p "$RESULTS_DIR"
mkdir -p "$PROFILES_DIR"

# Loop through each dataset
for dataset in "${datasets[@]}"; do
    echo "=========================================="
    echo "Running cProfile benchmark for dataset: $dataset"
    echo "=========================================="
    
    # Run the benchmark with cProfile using CLI options
    PROFILE_FILE="$PROFILES_DIR/profile_${dataset}.prof"
    OUTPUT_FILE="$RESULTS_DIR/${dataset}_output.txt"
    echo "Running: python -m cProfile -o $PROFILE_FILE -m cuml.accel $BENCHMARK_SCRIPT --datasets $dataset --experiment-name test_rf_model_gpu_$dataset"
    python -m cProfile -o "$PROFILE_FILE" -m cuml.accel "$BENCHMARK_SCRIPT" \
        --datasets "$dataset" \
        --experiment-name "test_rf_model_gpu_$dataset" 2>&1 | tee "$OUTPUT_FILE"
    
    echo "Profile saved to: $PROFILE_FILE"
    
    # Extract the results section (from "Results:" to the end)
    RESULTS_FILE="$RESULTS_DIR/${dataset}_results.txt"
    echo "Extracting results to: $RESULTS_FILE"
    
    # Find the line number where "Results:" appears and extract from there to the end
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
    
    echo "Completed cProfile benchmark for dataset: $dataset"
    echo ""
done

echo "=========================================="
echo "All cProfile benchmarks completed!"
echo "Profiles saved in: $PROFILES_DIR"
echo "Results saved in: $RESULTS_DIR"
echo "=========================================="

