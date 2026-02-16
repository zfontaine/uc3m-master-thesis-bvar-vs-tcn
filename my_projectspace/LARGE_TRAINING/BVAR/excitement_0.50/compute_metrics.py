#!/usr/bin/env python3
# This file is a part of a Master Thesis titled:
# "Bayesian VAR vs. Temporal Convolutional Networks: A Simulation Study 
# on Forecasting Accuracy and Uncertainty Calibration"
#
# Institution: Universidad Carlos III de Madrid (UC3M)
# Year: 2026
#
# FILENAME: compute_metrics.py
# AUTHOR: Zeno S. alias Zeno Fontaine
# DATE: 2026-02-06
#
# INPUT: Single forecast CSV file with columns:
#        time, var1_actual, var1_pred, var1_q2.5, var1_q5, var1_q10, 
#        var1_q20, var1_q50, var1_q80, var1_q90, var1_q95, var1_q97.5
#        (same pattern for var2, var3)
#        Optional header lines starting with # (e.g., # runtime: 138.24)
#
# OUTPUT: Comma-separated line to stdout containing 24 metrics:
#         - RMSE (3 values: var1, var2, var3)
#         - MAE (3 values: var1, var2, var3)
#         - Coverage rates for 80%, 90%, 95% PIs (9 values: 3 intervals × 3 vars)
#         - Interval widths for 80%, 90%, 95% PIs (9 values: 3 intervals × 3 vars)
#
# FILE DESCRIPTION: evaluation-metrics, forecasting
# Computes forecast evaluation metrics for a single dataset. This is the
# first stage in the evaluation pipeline: compute_metrics.py processes
# individual CSVs, grand_aggregator.py aggregates across many CSVs, and
# pretraining_selector.py creates summary tables for model selection.
#
# Configurable parameters:
#   - None (metric definitions are standard)
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


def compute_metrics(csv_path):
    """
    Compute all forecast evaluation metrics for a single dataset.
    
    Returns a list of 24 values in order:
    RMSE_var1, RMSE_var2, RMSE_var3,
    MAE_var1, MAE_var2, MAE_var3,
    Coverage80_var1, Coverage80_var2, Coverage80_var3,
    Coverage90_var1, Coverage90_var2, Coverage90_var3,
    Coverage95_var1, Coverage95_var2, Coverage95_var3,
    Width80_var1, Width80_var2, Width80_var3,
    Width90_var1, Width90_var2, Width90_var3,
    Width95_var1, Width95_var2, Width95_var3
    """
    df = pd.read_csv(csv_path, comment='#')
    
    results = []
    
    for var_num in [1, 2, 3]:
        var_prefix = f'var{var_num}'
        
        actual = df[f'{var_prefix}_actual'].values
        pred = df[f'{var_prefix}_pred'].values
        q2_5 = df[f'{var_prefix}_q2.5'].values
        q5 = df[f'{var_prefix}_q5'].values
        q10 = df[f'{var_prefix}_q10'].values
        q90 = df[f'{var_prefix}_q90'].values
        q95 = df[f'{var_prefix}_q95'].values
        q97_5 = df[f'{var_prefix}_q97.5'].values
        
        if var_num == 1:
            var_data = {'actual': actual, 'pred': pred,
                       'q2_5': q2_5, 'q5': q5, 'q10': q10,
                       'q90': q90, 'q95': q95, 'q97_5': q97_5}
            all_vars = [var_data]
        else:
            var_data = {'actual': actual, 'pred': pred,
                       'q2_5': q2_5, 'q5': q5, 'q10': q10,
                       'q90': q90, 'q95': q95, 'q97_5': q97_5}
            all_vars.append(var_data)
    
    for var_data in all_vars:
        rmse = np.sqrt(np.mean((var_data['actual'] - var_data['pred'])**2))
        results.append(rmse)
    
    for var_data in all_vars:
        mae = np.mean(np.abs(var_data['actual'] - var_data['pred']))
        results.append(mae)
    
    for var_data in all_vars:
        covered = (var_data['actual'] >= var_data['q10']) & (var_data['actual'] <= var_data['q90'])
        coverage_80 = covered.mean()
        results.append(coverage_80)
    
    for var_data in all_vars:
        covered = (var_data['actual'] >= var_data['q5']) & (var_data['actual'] <= var_data['q95'])
        coverage_90 = covered.mean()
        results.append(coverage_90)
    
    for var_data in all_vars:
        covered = (var_data['actual'] >= var_data['q2_5']) & (var_data['actual'] <= var_data['q97_5'])
        coverage_95 = covered.mean()
        results.append(coverage_95)
    
    for var_data in all_vars:
        width_80 = np.mean(var_data['q90'] - var_data['q10'])
        results.append(width_80)
    
    for var_data in all_vars:
        width_90 = np.mean(var_data['q95'] - var_data['q5'])
        results.append(width_90)
    
    for var_data in all_vars:
        width_95 = np.mean(var_data['q97_5'] - var_data['q2_5'])
        results.append(width_95)
    
    return results


def main():
    if len(sys.argv) != 2:
        print("Usage: python compute_metrics.py <path_to_forecast_csv>", file=sys.stderr)
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    try:
        metrics_list = compute_metrics(csv_path)
        
        output_line = ','.join([f"{val:.8f}" for val in metrics_list])
        print(output_line)
        
    except Exception as e:
        print(f"Error processing {csv_path}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
