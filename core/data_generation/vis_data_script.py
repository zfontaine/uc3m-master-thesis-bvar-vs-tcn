#!/usr/bin/env python3
# This file is a part of a Master Thesis titled:
# "Bayesian VAR vs. Temporal Convolutional Networks: A Simulation Study 
# on Forecasting Accuracy and Uncertainty Calibration"
#
# FILENAME: vis_data_script.py
# AUTHOR: Zeno S. alias Zeno Fontaine
# DATE: 2026-02-06
# 
# INPUT: Path to CSV file containing simulated market data
#        (e.g., data_folder/excitement_0.50/d1.csv)
#
# OUTPUT: 2x2 grid visualization showing:
#         (1) Price series over time
#         (2) Returns distribution histograms
#         (3) Rolling volatility (30-period window)
#         (4) Asset correlation heatmap
#         Optionally saved as PNG or displayed interactively
#
# FILE DESCRIPTION: visualization, exploratory-analysis, diagnostics
# Command-line tool for visualizing simulated market datasets. Creates
# comprehensive diagnostic plots to verify data properties: time series
# dynamics, return distributions, volatility clustering, and correlation
# structure. Useful for validating simulation outputs before feeding to
# forecasting models. Includes automatic file discovery and summary statistics.
#
# Configurable parameters:
#   - file: Path to dataset CSV
#   - --save: Flag to save plot instead of displaying
#   - --output: Custom output filename for saved plot
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
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob

# =============================================================================
# VISUALIZATION FUNCTION
# =============================================================================

def visualize_data(filepath):
    """
    Generate 2x2 diagnostic plot grid for a single dataset.
    
    Parameters
    ----------
    filepath : str
        Path to CSV file containing price data.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object with 4 subplots.
    """
    
    df = pd.read_csv(filepath, index_col='time')
    
    # Extract metadata from filepath
    folder = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    excitement = folder.split('_')[-1] if 'excitement' in folder else 'unknown'
    
    # Create subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Dataset: {filename} (Excitement: {excitement})', 
                 fontsize=14, fontweight='bold')
    
    # ========== PLOT 1: PRICE SERIES ==========
    ax1 = axes[0, 0]
    for col in df.columns:
        ax1.plot(df.index, df[col], label=col, alpha=0.8, linewidth=1)
    ax1.set_title('Price Series Over Time')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ========== PLOT 2: RETURNS DISTRIBUTION ==========
    ax2 = axes[0, 1]
    returns = df.pct_change().dropna()
    for i, col in enumerate(returns.columns):
        ax2.hist(returns[col], bins=50, alpha=0.5, label=col)
    ax2.set_title('Returns Distribution')
    ax2.set_xlabel('Return')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # ========== PLOT 3: ROLLING VOLATILITY ==========
    ax3 = axes[1, 0]
    rolling_vol = returns.rolling(window=30).std()
    for col in rolling_vol.columns:
        ax3.plot(rolling_vol.index, rolling_vol[col], label=col, alpha=0.8, linewidth=1)
    ax3.set_title('Rolling Volatility (30-period)')
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Volatility')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # ========== PLOT 4: CORRELATION HEATMAP ==========
    ax4 = axes[1, 1]
    corr_matrix = returns.corr()
    im = ax4.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax4.set_xticks(range(len(corr_matrix.columns)))
    ax4.set_yticks(range(len(corr_matrix.columns)))
    ax4.set_xticklabels(corr_matrix.columns)
    ax4.set_yticklabels(corr_matrix.columns)
    ax4.set_title('Asset Correlation Matrix')
    
    # Annotate correlation values
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            ax4.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                    ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im, ax=ax4)
    
    plt.tight_layout()
    
    # Add summary statistics box
    stats_text = f"Summary Statistics:\n"
    stats_text += f"Mean Returns: {returns.mean().mean():.4f}\n"
    stats_text += f"Mean Volatility: {returns.std().mean():.4f}\n"
    stats_text += f"Max Drawdown: {((df / df.cummax() - 1).min().min()):.2%}"
    
    fig.text(0.02, 0.02, stats_text, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    return fig

# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def main():
    """
    Parse arguments and execute visualization.
    """
    parser = argparse.ArgumentParser(description='Visualize simulated market data')
    parser.add_argument('file', help='Path to CSV file (e.g., data_folder/excitement_0.50/d1.csv)')
    parser.add_argument('--save', action='store_true', help='Save plot instead of displaying')
    parser.add_argument('--output', type=str, default=None, help='Output filename for saved plot')
    
    args = parser.parse_args()
    
    # ========== FILE DISCOVERY ==========
    if not os.path.exists(args.file):
        # Try pattern matching with glob
        matches = glob(f"**/{args.file}", recursive=True)
        if not matches:
            matches = glob(f"data_folder/*/{args.file}")
        
        if matches:
            filepath = matches[0]
            print(f"Found file: {filepath}")
        else:
            print(f"Error: File '{args.file}' not found!")
            print("\nTry one of these formats:")
            print("  python3 vis_data_script.py data_folder/excitement_0.50/d1.csv")
            print("  python3 vis_data_script.py d1.csv  (if in data_folder)")
            return
    else:
        filepath = args.file
    
    print(f"Visualizing: {filepath}")
    
    # ========== GENERATE VISUALIZATION ==========
    fig = visualize_data(filepath)
    
    # ========== OUTPUT HANDLING ==========
    if args.save:
        if args.output:
            output_file = args.output
        else:
            # Auto-generate filename
            base = os.path.splitext(os.path.basename(filepath))[0]
            output_file = f"{base}_visualization.png"
        
        fig.savefig(output_file, dpi=100, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()

if __name__ == "__main__":
    main()
