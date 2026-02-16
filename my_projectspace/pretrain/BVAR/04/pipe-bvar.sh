#!/bin/bash
#!/usr/bin/env bash
# This file is a part of a Master Thesis titled:
# "Bayesian VAR vs. Temporal Convolutional Networks: A Simulation Study
# on Forecasting Accuracy and Uncertainty Calibration"
# Universidad Carlos III de Madrid (UC3M), 2026
#
# FILENAME: pipe-bvar.sh 
# # AUTHOR: Zeno S. alias Zeno Fontaine
# DATE: 2026-02-07
#
# LICENSE: GNU GPLv3 for code
#          Creative Commons for the Master thesis
# NOTE:
#   -- simple loop through all datasets, calling bvar_model.py for each.
#
#===============================================================

set -e  

# ========== ARGUMENT PARSING ==========
LAGS="4"
PLOTS=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --lags)
            LAGS="$2"
            shift 2
            ;;
        --plots)
            PLOTS=true
            shift 1
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --lags L [--plots]"
            exit 1
            ;;
    esac
done

# Error check
if [ -z "$LAGS" ]; then
    echo "Error: Missing required parameter --lags"
    echo "Usage: $0 --lags L [--plots]"
    echo "Example: $0 --lags 6 --plots"
    exit 1
fi

echo "=========================================="
echo "BVAR Training Pipeline"
echo "=========================================="
echo "Configuration:"
echo "  Lags:   $LAGS"
echo "  Plots:  $PLOTS"
echo "=========================================="

# ========== CREATE RESULTS DIRECTORY ==========

mkdir -p RESULTS
echo "Results will be saved to: RESULTS/"
echo ""

# ========== PROCESS ALL DATASETS ==========

total_datasets=0
completed=0

# Find all CSV files (excluding RESULTS directory)
for csv_file in excitement_*/*.csv; do
    # Skip if no files found
    [ -e "$csv_file" ] || continue
    
    total_datasets=$((total_datasets + 1))
done

echo "Found $total_datasets datasets to process"
echo ""

for csv_file in excitement_*/*.csv; do
    [ -e "$csv_file" ] || continue
    
    # Extract folder and dataset name
    folder=$(dirname "$csv_file")
    folder_name=$(basename "$folder")
    dataset=$(basename "$csv_file" .csv)
    
    output="RESULTS/${folder_name}_${dataset}.csv"
    
    completed=$((completed + 1))
    echo "[$completed/$total_datasets] Processing: $csv_file"
    echo "  → Output: $output"
    
# ========== RUN BVAR MODEL=====================

    if [ "$PLOTS" = true ]; then
        python bvar_model.py \
            --input "$csv_file" \
            --output "$output" \
            --lags "$LAGS" \
            --plots
    else
        python bvar_model.py \
            --input "$csv_file" \
            --output "$output" \
            --lags "$LAGS"
    fi
    
    echo "  ✓ Completed"
    echo ""
done

echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo "Processed: $completed datasets"
echo "Results in: RESULTS/"
echo ""
echo "To compute metrics, run your external metrics script:"
echo "  python compute_metrics.py RESULTS/*.csv"
echo "=========================================="
