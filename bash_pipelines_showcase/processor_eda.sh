#!/bin/bash
#!/usr/bin/env bash
# This file is a part of a Master Thesis titled:
# "Bayesian VAR vs. Temporal Convolutional Networks: A Simulation Study
# on Forecasting Accuracy and Uncertainty Calibration"
# Universidad Carlos III de Madrid (UC3M), 2026
#
# FILENAME: processor_eda.sh
# AUTHOR: Zeno S. alias Zeno Fontaine
# DATE: 2026-02-07
#
# LICENSE: GNU GPLv3 for code
#          Creative Commons for the Master thesis
#
#===============================================================




"""
processor_eda.sh - Orchestrate EDA pipeline for all datasets in current directory

This script:
1. Creates/cleans RESULTS/ directory
2. Generates header for eda_stat.csv
3. Processes all .csv files with eda.py
4. Runs aggregate.py to generate final statistics
"""

# ==========================================================
# STAGE 1: results directory
# ==========================================================
if [ -d "RESULTS" ]; then
    echo "Cleaning existing RESULTS directory..."
    rm -rf RESULTS
fi
mkdir RESULTS

# ==========================================================
# STAGE 2: header 
# ==========================================================
echo "Creating eda_stat.csv with header..."
echo "mean_var1,mean_var2,mean_var3,median_var1,median_var2,median_var3,std_var1,std_var2,std_var3,iqr_var1,iqr_var2,iqr_var3,min_var1,min_var2,min_var3,max_var1,max_var2,max_var3,skew_var1,skew_var2,skew_var3,kurt_var1,kurt_var2,kurt_var3,mean_correlation,JB_pval_var1,JB_pval_var2,JB_pval_var3" > RESULTS/eda_stat.csv

# COUNT TOTAL CSV FILES!
total_files=$(ls -1 *.csv 2>/dev/null | wc -l)

if [ $total_files -eq 0 ]; then
    echo "Error: No .csv files found in current directory"
    exit 1
fi

echo "Found $total_files CSV files to process"
echo ""

# ==========================================================
# STAGE 3: PROCESS CSV FILES 
# ==========================================================
counter=0
for file in *.csv; do
    counter=$((counter + 1))
    echo "Processing file $counter/$total_files: $file"
    python3 eda.py "$file" >> RESULTS/eda_stat.csv
    
    # Check if eda.py succeeded
    if [ $? -ne 0 ]; then
        echo "Error processing $file - stopping pipeline"
        exit 1
    fi
done

echo ""
echo "All datasets processed successfully!"
echo "Running aggregate.py to generate final statistics..."

#=====================================================================
# FINAL: RUN AGGREGATION
#=====================================================================

python3 aggregate.py RESULTS/eda_stat.csv > RESULTS/stat_final.csv

if [ $? -eq 0 ]; then
    echo ""
    echo "Pipeline complete! Results saved in RESULTS/"
    echo "  - eda_stat.csv: Raw statistics for all datasets"
    echo "  - stat_final.csv: Aggregated final statistics"
else
    echo "Error running aggregate.py"
    exit 1
fi
