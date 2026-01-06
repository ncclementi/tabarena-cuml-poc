#!/bin/bash

# Define the datasets to run
datasets=("anneal" "credit-g" "diabetes" "APSFailure" "customer_satisfaction_in_airline")

# Base directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BENCHMARK_SCRIPT="$SCRIPT_DIR/scripts/run_tabarena_rf_experiment.py"
RESULTS_DIR="$SCRIPT_DIR/results_per_dataset"

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Loop through each dataset
for dataset in "${datasets[@]}"; do
    echo "=========================================="
    echo "Running benchmark for dataset: $dataset"
    echo "=========================================="
    
    # Run the benchmark with cuml.accel profiling using CLI options
    OUTPUT_FILE="$RESULTS_DIR/${dataset}_output.txt"
    echo "Running: python -m cuml.accel --profile $BENCHMARK_SCRIPT --datasets $dataset --experiment-name test_rf_model_gpu_$dataset"
    python -m cuml.accel --profile "$BENCHMARK_SCRIPT" \
        --datasets "$dataset" \
        --experiment-name "test_rf_model_gpu_$dataset" 2>&1 | tee "$OUTPUT_FILE"
    
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
    
    # Clean up: delete AutoGluonModels and experiments folders
    echo "Cleaning up AutoGluonModels and experiments folders..."
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
echo "Results saved in: $RESULTS_DIR"
echo "=========================================="

