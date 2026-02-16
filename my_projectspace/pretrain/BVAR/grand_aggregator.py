#!/usr/bin/env python3
# This file is a part of a Master Thesis titled:
# "Bayesian VAR vs. Temporal Convolutional Networks: A Simulation Study 
# on Forecasting Accuracy and Uncertainty Calibration"
#
# Institution: Universidad Carlos III de Madrid (UC3M)
# Year: 2026
#
# FILENAME: grand_aggregator.py
# AUTHOR: Zeno S. alias Zeno Fontaine
# DATE: 2026-02-06
#
# INPUT: Model folders containing RESULTS/ subdirectories with forecast CSVs
#        Each CSV has header lines (# runtime: X, # DIC: Y, # data_tag: Z)
#        and forecast data processed by compute_metrics.py
#
# OUTPUT: CSV file with aggregated metrics across all datasets per model:
#         - Metadata: model_name, n_datasets, data_tag_dist, runtime_total
#         - Header aggregates: DIC_mean, DIC_std (mean/std across datasets)
#         - Metric aggregates: RMSE_var1_mean, RMSE_var1_std, etc. (48 columns)
#
# FILE DESCRIPTION: aggregation, evaluation-metrics, forecasting
# Aggregates forecast evaluation metrics across multiple datasets within
# model folders. This is the second stage in the evaluation pipeline:
# compute_metrics.py processes individual CSVs → grand_aggregator.py 
# aggregates across many CSVs → pretraining_selector.py creates summary
# tables for model selection.
#
# Configurable parameters:
#   - --folders: List of model folder names to process
#   - --input_dir: Base directory containing model folders
#   - --compute_metrics: Path to compute_metrics.py script
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
import os
import glob
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict


def extract_headers(csv_path):
    """
    Extract all header tags from CSV file (lines starting with #).
    
    Returns dict mapping header names to values.
    Example: {'runtime': 138.24, 'DIC': -27062.89, 'data_tag': 1}
    """
    headers = {}
    
    with open(csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            if not line.startswith('#'):
                break
            
            if ':' in line:
                line = line.lstrip('#').strip()
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                try:
                    value = float(value)
                except ValueError:
                    pass
                
                headers[key] = value
    
    return headers


def call_compute_metrics(csv_path, compute_metrics_script='compute_metrics.py'):
    """
    Call compute_metrics.py on a single CSV file via subprocess.
    
    Returns list of 24 metric values, or None if failed.
    """
    try:
        result = subprocess.run(
            ['python', compute_metrics_script, csv_path],
            capture_output=True,
            text=True,
            check=True
        )
        
        metrics = [float(x) for x in result.stdout.strip().split(',')]
        
        if len(metrics) != 24:
            print(f"Warning: Expected 24 metrics, got {len(metrics)} for {csv_path}", file=sys.stderr)
            return None
        
        return metrics
    
    except subprocess.CalledProcessError as e:
        print(f"Error processing {csv_path}: {e.stderr}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error processing {csv_path}: {e}", file=sys.stderr)
        return None


def aggregate_data_tag(tag_values):
    """
    Aggregate data_tag values into distribution string.
    
    Returns format: "count0;count1;count2(total)"
    Example: "14;1;0(15)" means 14 datasets with tag 0, 1 with tag 1, 0 with tag 2
    """
    from collections import Counter
    
    counts = Counter(tag_values)
    total = len(tag_values)
    
    count0 = counts.get(0, 0)
    count1 = counts.get(1, 0)
    count2 = counts.get(2, 0)
    
    return f"{count0};{count1};{count2}({total})"


def process_model_folder(folder_path, compute_metrics_script='compute_metrics.py'):
    """
    Process all CSVs in a model folder's RESULTS/ directory.
    
    Returns dict with aggregated results for this folder, or None if failed.
    """
    results_dir = Path(folder_path) / 'RESULTS'
    
    if not results_dir.exists():
        print(f"Warning: RESULTS directory not found: {results_dir}", file=sys.stderr)
        return None
    
    csv_files = sorted(results_dir.glob('*.csv'))
    
    if not csv_files:
        print(f"Warning: No CSV files found in {results_dir}", file=sys.stderr)
        return None
    
    print(f"Processing {len(csv_files)} CSVs in {folder_path}...")
    
    all_headers = defaultdict(list)
    all_metrics = []
    
    for csv_file in csv_files:
        headers = extract_headers(csv_file)
        for key, value in headers.items():
            all_headers[key].append(value)
        
        metrics = call_compute_metrics(csv_file, compute_metrics_script)
        if metrics is not None:
            all_metrics.append(metrics)
    
    if not all_metrics:
        print(f"Error: No valid metrics computed for {folder_path}", file=sys.stderr)
        return None
    
    metrics_array = np.array(all_metrics)
    
    metrics_mean = np.mean(metrics_array, axis=0)
    metrics_std = np.std(metrics_array, axis=0, ddof=1)
    
    aggregated_headers = {}
    
    for key, values in all_headers.items():
        if key == 'runtime':
            aggregated_headers['runtime_total'] = sum(values)
        elif key == 'data_tag':
            aggregated_headers['data_tag_dist'] = aggregate_data_tag(values)
        else:
            aggregated_headers[f'{key}_mean'] = np.mean(values)
            aggregated_headers[f'{key}_std'] = np.std(values, ddof=1)
    
    result = {
        'n_datasets': len(csv_files),
        **aggregated_headers
    }
    
    metric_names = [
        'RMSE_var1', 'RMSE_var2', 'RMSE_var3',
        'MAE_var1', 'MAE_var2', 'MAE_var3',
        'Coverage80_var1', 'Coverage80_var2', 'Coverage80_var3',
        'Coverage90_var1', 'Coverage90_var2', 'Coverage90_var3',
        'Coverage95_var1', 'Coverage95_var2', 'Coverage95_var3',
        'Width80_var1', 'Width80_var2', 'Width80_var3',
        'Width90_var1', 'Width90_var2', 'Width90_var3',
        'Width95_var1', 'Width95_var2', 'Width95_var3'
    ]
    
    for i, name in enumerate(metric_names):
        result[f'{name}_mean'] = metrics_mean[i]
        result[f'{name}_std'] = metrics_std[i]
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description='Grand Aggregator - Universal metric aggregation across model folders'
    )
    
    parser.add_argument('--folders', nargs='+', required=True,
                        help='Model folder names to process (e.g., 03 04 05)')
    parser.add_argument('--input_dir', type=str, default='.',
                        help='Base directory containing model folders (default: current directory)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output CSV filename')
    parser.add_argument('--compute_metrics', type=str, default='compute_metrics.py',
                        help='Path to compute_metrics.py script (default: compute_metrics.py)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.compute_metrics):
        print(f"Error: compute_metrics.py not found at {args.compute_metrics}", file=sys.stderr)
        sys.exit(1)
    
    print("=" * 60)
    print("Grand Aggregator - Universal Metric Aggregation")
    print("=" * 60)
    print(f"Base directory: {args.input_dir}")
    print(f"Model folders: {args.folders}")
    print(f"Output: {args.output}")
    print("=" * 60)
    print()
    
    results = []
    
    for folder in args.folders:
        folder_path = Path(args.input_dir) / folder
        
        if not folder_path.exists():
            print(f"Warning: Folder not found: {folder_path}", file=sys.stderr)
            continue
        
        print(f"\n[Processing folder: {folder}]")
        result = process_model_folder(folder_path, args.compute_metrics)
        
        if result is not None:
            result['model_name'] = folder
            results.append(result)
            print(f"✓ Completed: {folder}")
        else:
            print(f"✗ Failed: {folder}")
    
    if not results:
        print("\nError: No results generated!", file=sys.stderr)
        sys.exit(1)
    
    df = pd.DataFrame(results)
    
    metadata_cols = ['model_name', 'n_datasets', 'data_tag_dist', 'runtime_total']
    header_cols = [col for col in df.columns if col not in metadata_cols and 
                   not any(metric in col for metric in ['RMSE', 'MAE', 'Coverage', 'Width'])]
    metric_cols = [col for col in df.columns if any(metric in col for metric in ['RMSE', 'MAE', 'Coverage', 'Width'])]
    
    column_order = metadata_cols + sorted(header_cols) + sorted(metric_cols)
    column_order = [col for col in column_order if col in df.columns]
    
    df = df[column_order]
    
    df.to_csv(args.output, index=False, float_format='%.8f')
    
    print("\n" + "=" * 60)
    print(f"✓ Results saved to: {args.output}")
    print(f"  Total folders processed: {len(results)}")
    print(f"  Total columns: {len(df.columns)}")
    print("=" * 60)
    
    print("\nSummary:")
    print(df[['model_name', 'n_datasets', 'runtime_total']].to_string(index=False))


if __name__ == '__main__':
    main()
