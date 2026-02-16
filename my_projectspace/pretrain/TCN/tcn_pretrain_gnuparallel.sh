#!/bin/bash
#!/usr/bin/env bash
# This file is a part of a Master Thesis titled:
# "Bayesian VAR vs. Temporal Convolutional Networks: A Simulation Study
# on Forecasting Accuracy and Uncertainty Calibration"
# Universidad Carlos III de Madrid (UC3M), 2026
#
# FILENAME: tcn_pretrain_gnuparallel.sh
# AUTHOR: Zeno S. alias Zeno Fontaine
# DATE: 2026-02-07
#
# LICENSE: GNU GPLv3 for code
#          Creative Commons for the Master thesis
#
#===============================================================
set -e

echo "=========================================="
echo "TCN Pretraining - Parallel Grid Search"
echo "=========================================="
echo "Grid: layers={2,3,4} × channels={16,32,64} × dropout={0.1,0.2,0.3}"
echo "Total configs: 27"
echo "Parallelization: 2 jobs per config"
echo "=========================================="

# VERIFY DATAFOLDER EXISTS!
if [ ! -d "pretrain_data" ]; then
    echo "Error: pretrain_data/ directory not found!"
    echo "Please create it with structure:"
    echo "  pretrain_data/excitement_0.25/*.csv"
    echo "  pretrain_data/excitement_0.50/*.csv"
    echo "  pretrain_data/excitement_0.75/*.csv"
    exit 1
fi

n_files=$(find pretrain_data -name "*.csv" | wc -l)
echo "Found $n_files input CSV files"

if [ $n_files -eq 0 ]; then
    echo "Error: No CSV files found in pretrain_data/"
    exit 1
fi

echo "=========================================="
echo ""

mkdir -p RESULTS

config_num=1

# =====================================================
# GNU PARALLEL SECTION
# =====================================================

# Nested loops for Cartesian product--IMPORTANT TO ENABLE ALL 27 CONFIGS!
for layers in 2 3 4; do
  for channels in 16 32 64; do
    for dropout in 0.1 0.2 0.3; do
      
      dropout_fmt=$(echo $dropout | sed 's/\./d/')
      
      # Create config-specific folder
      config_dir="RESULTS/config$(printf "%02d" $config_num)_L${layers}_C${channels}_D${dropout_fmt}"
      mkdir -p "$config_dir"
      
      echo ""
      echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      echo "Config $config_num/27: layers=$layers, channels=$channels, dropout=$dropout"
      echo "  Output: $config_dir/"
      echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
      
# =====================================================
# IMPORTANT! The below code:
#   -- Run parallel for all input CSVs with this configuration
#   -- Extract excitement level from path to preserve in output filename
# =====================================================

      parallel --jobs 2 --bar --eta \
        'excitement=$(basename $(dirname {})); \
         python tcn_model.py \
           --input {} \
           --output '"$config_dir"'/${excitement}_{/.}.csv \
           --layers '"$layers"' \
           --channels '"$channels"' \
           --dropout '"$dropout"' \
           --plots' \
        ::: pretrain_data/*/*.csv
      
      echo "  ✓ Config $config_num complete"
      
      ((config_num++))
      
    done
  done
done

echo ""
echo "=========================================="
echo "✓ All 27 configurations complete!"
echo "=========================================="
echo ""
echo "Output structure:"
echo "  RESULTS/"
echo "    ├── config01_L2_C16_D0d1/"
echo "    │   ├── excitement_0.25_d1.csv"
echo "    │   ├── excitement_0.25_d2.csv"
echo "    │   ├── ..."
echo "    │   ├── excitement_0.50_d1.csv"
echo "    │   └── excitement_0.75_d5.csv"
echo "    ├── config02_L2_C16_D0d2/"
echo "    └── ..."
echo ""
echo "Next steps:"
echo "  1. Verify results: ls -R RESULTS/ | head -50"
echo "  2. Run aggregator:"
echo "     cd RESULTS"
echo "     python ../grand_aggregator.py --folders config01 config02 ... config27 --output TCN_pretraining_all_metrics.csv"
echo "  3. Run selector:"
echo "     python pretraining_selector.py --input TCN_pretraining_all_metrics.csv --output TCN_pretraining_summary.csv"
echo ""
