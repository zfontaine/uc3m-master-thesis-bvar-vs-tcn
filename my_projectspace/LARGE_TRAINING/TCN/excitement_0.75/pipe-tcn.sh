#!/bin/bash
#!/usr/bin/env bash
# This file is a part of a Master Thesis titled:
# "Bayesian VAR vs. Temporal Convolutional Networks: A Simulation Study
# on Forecasting Accuracy and Uncertainty Calibration"
# Universidad Carlos III de Madrid (UC3M), 2026
#
# FILENAME: pipe-tcn.sh 
# # AUTHOR: Zeno S. alias Zeno Fontaine
# DATE: 2026-02-07
#
# LICENSE: GNU GPLv3 for code
#          Creative Commons for the Master thesis
# NOTE:
#   -- Simple loop through all datasets, calling tcn_model.py for each.
#
#===============================================================

set -e  # Exit on error

# ========== ARGUMENT PARSING ==========
LAYERS="3"
CHANNELS="64"
DROPOUT="0.1"
LOOKBACK=50  # Default
PLOTS=false  # Default: no plots
SHOCKPROPPLUS=false  # Default: no shock propagation analysis

while [[ $# -gt 0 ]]; do
    case $1 in
        --layers)
            LAYERS="$2"
            shift 2
            ;;
        --channels)
            CHANNELS="$2"
            shift 2
            ;;
        --dropout)
            DROPOUT="$2"
            shift 2
            ;;
        --lookback)
            LOOKBACK="$2"
            shift 2
            ;;
        --plots)
            PLOTS=true
            shift 1
            ;;
        --shockpropplus)
            SHOCKPROPPLUS=true
            shift 1
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --layers L --channels C --dropout D [--lookback LB] [--plots] [--shockpropplus]"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [ -z "$LAYERS" ] || [ -z "$CHANNELS" ] || [ -z "$DROPOUT" ]; then
    echo "Error: Missing required parameters"
    echo "Usage: $0 --layers L --channels C --dropout D [--lookback LB] [--plots] [--shockpropplus]"
    echo "Example: $0 --layers 3 --channels 32 --dropout 0.2 --plots --shockpropplus"
    exit 1
fi

echo "=========================================="
echo "TCN Training Pipeline"
echo "=========================================="
echo "Configuration:"
echo "  Layers:        $LAYERS"
echo "  Channels:      $CHANNELS"
echo "  Dropout:       $DROPOUT"
echo "  Lookback:      $LOOKBACK"
echo "  Plots:         $PLOTS"
echo "  ShockPropPlus: $SHOCKPROPPLUS"
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

# Process each dataset
for csv_file in excitement_*/*.csv; do
    # Skip if no files found
    [ -e "$csv_file" ] || continue
    
    # Extract folder and dataset name
    folder=$(dirname "$csv_file")
    folder_name=$(basename "$folder")
    dataset=$(basename "$csv_file" .csv)
    
    # Output filename: RESULTS/excitement_0.50_d1.csv
    output="RESULTS/${folder_name}_${dataset}.csv"
    
    completed=$((completed + 1))
    echo "[$completed/$total_datasets] Processing: $csv_file"
    echo "  → Output: $output"
    
    # Build command with optional flags
    CMD="python tcn_model.py \
        --input \"$csv_file\" \
        --output \"$output\" \
        --layers $LAYERS \
        --channels $CHANNELS \
        --dropout $DROPOUT \
        --lookback $LOOKBACK"
    
    if [ "$PLOTS" = true ]; then
        CMD="$CMD --plots"
    fi
    
    if [ "$SHOCKPROPPLUS" = true ]; then
        CMD="$CMD --shockpropplus"
    fi
    
    # Execute command
    eval $CMD
    
    echo "  ✓ Completed"
    echo ""
done

echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo "Processed: $completed datasets"
echo "Results in: RESULTS/"
echo ""

# Show what was generated
if [ "$SHOCKPROPPLUS" = true ]; then
    echo "Files generated per dataset:"
    echo "  - *_d*.csv            (predictions + quantiles)"
    echo "  - *_d*_shockprop.csv  (shock propagation data)"
    echo "  - *_d*_shockprop.png  (shock propagation plot)"
    echo "  - *_d*_uncertainty.csv (uncertainty time series)"
    echo "  - *_d*_uncertainty.png (uncertainty regime plot)"
    if [ "$PLOTS" = true ]; then
        echo "  - *_d*_diagnostics.png (diagnostic plots)"
    fi
else
    echo "Files generated per dataset:"
    echo "  - *_d*.csv            (predictions + quantiles)"
    if [ "$PLOTS" = true ]; then
        echo "  - *_d*_diagnostics.png (diagnostic plots)"
    fi
fi
echo ""

echo "To compute metrics, run your external metrics script:"
echo "  python compute_metrics.py RESULTS/*.csv"
echo "=========================================="
