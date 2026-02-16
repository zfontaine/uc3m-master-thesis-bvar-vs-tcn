#!/usr/bin/env python3
# This file is a part of a Master Thesis titled:
# "Bayesian VAR vs. Temporal Convolutional Networks: A Simulation Study 
# on Forecasting Accuracy and Uncertainty Calibration"
#
# Institution: Universidad Carlos III de Madrid (UC3M)
# Year: 2026
#
# FILENAME: pretraining_selector.py
# AUTHOR: Zeno S. alias Zeno Fontaine
# DATE: 2026-02-06
#
# INPUT: CSV file from grand_aggregator.py containing aggregated metrics
#        with columns: model_name, n_datasets, runtime_total, DIC_mean,
#        RMSE_var1_mean, RMSE_var1_std, MAE_var1_mean, Coverage95_var1_mean,
#        Width95_var1_mean, etc. (48 metric columns + metadata)
#
# OUTPUT: CSV file with summary table for model selection containing:
#         - Metadata: model_name, runtime_total, DIC_mean (BVAR only), data_tag_dist (BVAR only)
#         - Derived metrics: RMSE_overall_mean, MAE_overall_mean, Coverage95_overall_mean,
#           Width95_overall_mean, RMSE_MAE_ratio, plus individual variable means and stds
#         Float format: 5 decimal places
#
# FILE DESCRIPTION: model-selection, evaluation-metrics, forecasting
# Processes output from grand_aggregator.py and creates summary table
# for model selection during pretraining phase. This is the final stage
# in the evaluation pipeline: compute_metrics.py processes individual CSVs
# → grand_aggregator.py aggregates across many CSVs → pretraining_selector.py
# creates summary tables with derived metrics for manual model selection.
#
# Configurable parameters:
#   - --bvar: Include BVAR-specific columns (DIC_mean, data_tag_dist)
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

import argparse
import sys
import pandas as pd
import numpy as np


def compute_derived_metrics(df):
    """
    Compute derived metrics from grand_aggregator.py output.
    
    Returns DataFrame with additional derived columns for overall metrics
    (means and stds across var1, var2, var3) and RMSE/MAE ratio.
    """
    df = df.copy()
    
    if all(col in df.columns for col in ['RMSE_var1_mean', 'RMSE_var2_mean', 'RMSE_var3_mean']):
        df['RMSE_overall_mean'] = df[['RMSE_var1_mean', 'RMSE_var2_mean', 'RMSE_var3_mean']].mean(axis=1)
    
    if all(col in df.columns for col in ['RMSE_var1_std', 'RMSE_var2_std', 'RMSE_var3_std']):
        df['RMSE_overall_std'] = df[['RMSE_var1_std', 'RMSE_var2_std', 'RMSE_var3_std']].mean(axis=1)
    
    if all(col in df.columns for col in ['MAE_var1_mean', 'MAE_var2_mean', 'MAE_var3_mean']):
        df['MAE_overall_mean'] = df[['MAE_var1_mean', 'MAE_var2_mean', 'MAE_var3_mean']].mean(axis=1)
    
    if all(col in df.columns for col in ['MAE_var1_std', 'MAE_var2_std', 'MAE_var3_std']):
        df['MAE_overall_std'] = df[['MAE_var1_std', 'MAE_var2_std', 'MAE_var3_std']].mean(axis=1)
    
    if all(col in df.columns for col in ['Coverage95_var1_mean', 'Coverage95_var2_mean', 'Coverage95_var3_mean']):
        df['Coverage95_overall_mean'] = df[['Coverage95_var1_mean', 'Coverage95_var2_mean', 'Coverage95_var3_mean']].mean(axis=1)
    
    if all(col in df.columns for col in ['Coverage95_var1_std', 'Coverage95_var2_std', 'Coverage95_var3_std']):
        df['Coverage95_overall_std'] = df[['Coverage95_var1_std', 'Coverage95_var2_std', 'Coverage95_var3_std']].mean(axis=1)
    
    if all(col in df.columns for col in ['Width95_var1_mean', 'Width95_var2_mean', 'Width95_var3_mean']):
        df['Width95_overall_mean'] = df[['Width95_var1_mean', 'Width95_var2_mean', 'Width95_var3_mean']].mean(axis=1)
    
    if all(col in df.columns for col in ['Width95_var1_std', 'Width95_var2_std', 'Width95_var3_std']):
        df['Width95_overall_std'] = df[['Width95_var1_std', 'Width95_var2_std', 'Width95_var3_std']].mean(axis=1)
    
    if 'RMSE_overall_mean' in df.columns and 'MAE_overall_mean' in df.columns:
        df['RMSE_MAE_ratio'] = df['RMSE_overall_mean'] / df['MAE_overall_mean']
    
    return df


def create_summary_table(df, is_bvar=False):
    """
    Create summary table with selected columns for model selection.
    
    BVAR mode includes DIC_mean and data_tag_dist columns.
    TCN mode includes DIC_mean column as NaN for consistency.
    
    Returns DataFrame with selected columns.
    """
    base_cols = ['model_name', 'runtime_total']
    
    if is_bvar:
        bvar_cols = []
        if 'data_tag_dist' in df.columns:
            bvar_cols.append('data_tag_dist')
        if 'DIC_mean' in df.columns:
            bvar_cols.append('DIC_mean')
        
        columns = [base_cols[0]] + bvar_cols + [base_cols[1]]
    else:
        columns = base_cols.copy()
        if 'DIC_mean' in df.columns:
            columns.insert(2, 'DIC_mean')
        else:
            df['DIC_mean'] = np.nan
            columns.insert(2, 'DIC_mean')
    
    metric_cols = [
        'RMSE_overall_mean', 'RMSE_var1_mean', 'RMSE_var2_mean', 'RMSE_var3_mean', 'RMSE_overall_std',
        'MAE_overall_mean', 'MAE_var1_mean', 'MAE_var2_mean', 'MAE_var3_mean', 'MAE_overall_std',
        'Coverage95_overall_mean', 'Coverage95_overall_std',
        'Width95_overall_mean', 'Width95_overall_std',
        'RMSE_MAE_ratio'
    ]
    
    columns.extend(metric_cols)
    
    available_cols = [col for col in columns if col in df.columns]
    
    return df[available_cols]


def main():
    parser = argparse.ArgumentParser(
        description='Pretraining Selector - Process pretraining results for model selection'
    )
    
    parser.add_argument('--input', type=str, required=True,
                        help='Input CSV from grand_aggregator.py')
    parser.add_argument('--output', type=str, required=True,
                        help='Output summary CSV filename')
    parser.add_argument('--bvar', action='store_true',
                        help='BVAR mode: include DIC_mean and data_tag_dist columns')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Pretraining Selector - Model Selection Summary")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Mode: {'BVAR' if args.bvar else 'TCN'}")
    print("=" * 60)
    print()
    
    try:
        df = pd.read_csv(args.input)
        print(f"✓ Loaded {len(df)} models from input")
    except Exception as e:
        print(f"Error loading input file: {e}", file=sys.stderr)
        sys.exit(1)
    
    print("Computing derived metrics...")
    df = compute_derived_metrics(df)
    
    print("Creating summary table...")
    summary = create_summary_table(df, is_bvar=args.bvar)
    
    print(f"Saving to {args.output}...")
    summary.to_csv(args.output, index=False, float_format='%.5f')
    
    print()
    print("=" * 60)
    print(f"✓ Summary table saved: {args.output}")
    print(f"  Total models: {len(summary)}")
    print(f"  Total columns: {len(summary.columns)}")
    print("=" * 60)
    print()
    
    print("Preview (first 5 rows, key columns only):")
    preview_cols = ['model_name']
    
    if args.bvar and 'DIC_mean' in summary.columns:
        preview_cols.append('DIC_mean')
    
    if 'RMSE_overall_mean' in summary.columns:
        preview_cols.append('RMSE_overall_mean')
    if 'MAE_overall_mean' in summary.columns:
        preview_cols.append('MAE_overall_mean')
    if 'Coverage95_overall_mean' in summary.columns:
        preview_cols.append('Coverage95_overall_mean')
    if 'runtime_total' in summary.columns:
        preview_cols.append('runtime_total')
    
    preview_cols = [col for col in preview_cols if col in summary.columns]
    print(summary[preview_cols].head().to_string(index=False))
    print()
    
    print("Next Steps:")
    if args.bvar:
        print("  1. Inspect the output CSV")
        print("  2. Select optimal lag based on DIC_mean (lower is better)")
        print("  3. Consider RMSE, MAE, and Coverage95 for tie-breaking")
    else:
        print("  1. Inspect the output CSV")
        print("  2. Select optimal config based on RMSE_overall_mean (lower is better)")
        print("  3. Consider Coverage95 and runtime for tie-breaking")
    print()


if __name__ == '__main__':
    main()
