#!/usr/bin/env python3
# This file is a part of a Master Thesis titled:
# "Bayesian VAR vs. Temporal Convolutional Networks: A Simulation Study 
# on Forecasting Accuracy and Uncertainty Calibration"
#
# Institution: Universidad Carlos III de Madrid (UC3M)
# Year: 2026
#
# FILENAME: aggregate.py
# AUTHOR: Zeno S. alias Zeno Fontaine
# DATE: 2026-02-06
#
# INPUT: CSV file (eda_stat.csv) containing statistics from eda.py for all datasets
#        with columns: mean_var1, mean_var2, mean_var3, median_var1, ..., JB_pval_var3
#        One row per dataset
#
# OUTPUT: CSV table to stdout with aggregated statistics:
#         - Rows: statistics (Mean, Median, Std Dev, IQR, Min, Max, Skewness, 
#           Kurtosis, Mean Correlation, JB Passing Rate %, N Observations, N Datasets)
#         - Columns: Variable_1, Variable_2, Variable_3
#         Float format: 5 decimal places (except integer counts)
#
# FILE DESCRIPTION: exploratory-data-analysis, aggregation, descriptive-statistics
# Aggregates statistics across all generated datasets to produce summary tables
# for data validation and regime characterization. This is the second stage in
# the EDA pipeline: eda.py processes individual datasets → aggregate.py combines
# statistics across all datasets to compute means and passing rates for normality tests.
#
# Configurable parameters:
#   - n_observations: Number of timesteps per dataset (default: 6250)
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


def aggregate_statistics(csv_path):
    """
    Aggregate statistics from eda_stat.csv.
    
    Returns DataFrame with rows as statistics and columns as Variable_1, 
    Variable_2, Variable_3.
    """
    df = pd.read_csv(csv_path)
    
    n_datasets = len(df)
    n_observations = 6250
    
    results = {
        'Statistic': [],
        'Variable_1': [],
        'Variable_2': [],
        'Variable_3': []
    }
    
    stat_names = [
        ('Mean', ['mean_var1', 'mean_var2', 'mean_var3']),
        ('Median', ['median_var1', 'median_var2', 'median_var3']),
        ('Standard_Deviation', ['std_var1', 'std_var2', 'std_var3']),
        ('IQR', ['iqr_var1', 'iqr_var2', 'iqr_var3']),
        ('Minimum', ['min_var1', 'min_var2', 'min_var3']),
        ('Maximum', ['max_var1', 'max_var2', 'max_var3']),
        ('Skewness', ['skew_var1', 'skew_var2', 'skew_var3']),
        ('Kurtosis', ['kurt_var1', 'kurt_var2', 'kurt_var3'])
    ]
    
    for stat_name, columns in stat_names:
        results['Statistic'].append(stat_name)
        for col in columns:
            results['Variable_1' if 'var1' in col else 'Variable_2' if 'var2' in col else 'Variable_3'].append(
                df[col].mean()
            )
    
    mean_corr = df['mean_correlation'].mean()
    results['Statistic'].append('Aggregate_Mean_Correlation')
    results['Variable_1'].append(mean_corr)
    results['Variable_2'].append(mean_corr)
    results['Variable_3'].append(mean_corr)
    
    jb_columns = ['JB_pval_var1', 'JB_pval_var2', 'JB_pval_var3']
    results['Statistic'].append('JB_Passing_Rate_Percent')
    for col in jb_columns:
        passing_rate = (df[col] > 0.05).sum() / n_datasets * 100
        results['Variable_1' if 'var1' in col else 'Variable_2' if 'var2' in col else 'Variable_3'].append(
            passing_rate
        )
    
    results['Statistic'].append('N_Observations')
    results['Variable_1'].append(n_observations)
    results['Variable_2'].append(n_observations)
    results['Variable_3'].append(n_observations)
    
    results['Statistic'].append('N_Datasets')
    results['Variable_1'].append(n_datasets)
    results['Variable_2'].append(n_datasets)
    results['Variable_3'].append(n_datasets)
    
    result_df = pd.DataFrame(results)
    
    return result_df


def main():
    if len(sys.argv) != 2:
        print("Usage: python aggregate.py <path_to_eda_stat.csv>", file=sys.stderr)
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    try:
        result_df = aggregate_statistics(csv_path)
        
        output_lines = ['Statistic,Variable_1,Variable_2,Variable_3']
        
        for _, row in result_df.iterrows():
            stat = row['Statistic']
            
            if stat in ['N_Observations', 'N_Datasets']:
                line = f"{stat},{int(row['Variable_1'])},{int(row['Variable_2'])},{int(row['Variable_3'])}"
            else:
                line = f"{stat},{row['Variable_1']:.5f},{row['Variable_2']:.5f},{row['Variable_3']:.5f}"
            
            output_lines.append(line)
        
        print('\n'.join(output_lines))
        
    except Exception as e:
        print(f"Error processing {csv_path}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
