#!/usr/bin/env python3
# This file is a part of a Master Thesis titled:
# "Bayesian VAR vs. Temporal Convolutional Networks: A Simulation Study 
# on Forecasting Accuracy and Uncertainty Calibration"
#
# Institution: Universidad Carlos III de Madrid (UC3M)
# Year: 2026
#
# FILENAME: eda.py
# AUTHOR: Zeno S. alias Zeno Fontaine
# DATE: 2026-02-06
#
# INPUT: Single dataset CSV file with columns:
#        time, var1, var2, var3 (3 asset price series)
#
# OUTPUT: Comma-separated line to stdout containing 28 statistics:
#         - Basic statistics × 3 variables: mean, median, std, IQR, min, max (18 values)
#         - Distribution statistics × 3 variables: skewness, kurtosis (6 values)
#         - Mean correlation (1 value)
#         - Jarque-Bera p-values × 3 variables (3 values)
#
# FILE DESCRIPTION: exploratory-data-analysis, descriptive-statistics
# Computes descriptive statistics for a single generated dataset. This is
# the first stage in the EDA pipeline: eda.py processes individual datasets
# → aggregate.py combines statistics across all datasets to produce summary
# tables for data validation and regime characterization.
#
# Configurable parameters:
#   - None (standard statistical measures)
#
# LICENSE: GNU GPLv3 for code
#          Creative Commons for the Master thesis
#
# NOTE: Anthropic AI was used for enhancing readability of 
# this file. AI was used to generate better comments, write
# comprehensive descriptions, and rename variables. However,
# the design and code writing was done by the author, where
# AI-based tools were only used for debugging.
#
#===============================================================

import sys
import pandas as pd
import numpy as np
from scipy import stats


def compute_statistics(csv_path):
    """
    Compute all statistics for a single dataset.
    
    Returns a list of 28 values in the order:
    mean_var1, mean_var2, mean_var3,
    median_var1, median_var2, median_var3,
    std_var1, std_var2, std_var3,
    iqr_var1, iqr_var2, iqr_var3,
    min_var1, min_var2, min_var3,
    max_var1, max_var2, max_var3,
    skew_var1, skew_var2, skew_var3,
    kurt_var1, kurt_var2, kurt_var3,
    mean_correlation,
    JB_pval_var1, JB_pval_var2, JB_pval_var3
    """
    df = pd.read_csv(csv_path)
    
    var1 = df.iloc[:, 1].values
    var2 = df.iloc[:, 2].values
    var3 = df.iloc[:, 3].values
    
    variables = [var1, var2, var3]
    results = []
    
    for var in variables:
        results.append(np.mean(var))
    
    for var in variables:
        results.append(np.median(var))
    
    for var in variables:
        results.append(np.std(var, ddof=1))
    
    for var in variables:
        q75, q25 = np.percentile(var, [75, 25])
        results.append(q75 - q25)
    
    for var in variables:
        results.append(np.min(var))
    
    for var in variables:
        results.append(np.max(var))
    
    for var in variables:
        results.append(stats.skew(var))
    
    for var in variables:
        results.append(stats.kurtosis(var))
    
    corr_12 = np.corrcoef(var1, var2)[0, 1]
    corr_13 = np.corrcoef(var1, var3)[0, 1]
    corr_23 = np.corrcoef(var2, var3)[0, 1]
    mean_corr = np.mean([corr_12, corr_13, corr_23])
    results.append(mean_corr)
    
    for var in variables:
        jb_stat, jb_pval = stats.jarque_bera(var)
        results.append(jb_pval)
    
    return results


def main():
    if len(sys.argv) != 2:
        print("Usage: python eda.py <path_to_csv>", file=sys.stderr)
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    try:
        stats_list = compute_statistics(csv_path)
        
        output_line = ','.join([f"{val:.8f}" for val in stats_list])
        print(output_line)
        
    except Exception as e:
        print(f"Error processing {csv_path}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
