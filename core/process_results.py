#!/usr/bin/env python3
# This file is a part of a Master Thesis titled:
# "Bayesian VAR vs. Temporal Convolutional Networks:
# A Simulation Study on Forecasting Accuracy and
# Uncertainty Calibration" produced by the author
# at Universidad Carlos III. de Madrid (UC3M) in 2026.
# All rights reserved.
#
# FILENAME: process_results.py
# AUTHOR: Zeno S. alias Zeno Fontaine
# DATE: 2026-02-06
# 
# INPUT: Command-line argument: excitement level (0p25, 0p50, 0p75)
#        Expected files in working directory:
#          - BVAR{excitement}_final.csv (BVAR results)
#          - TCN{excitement}_final.csv (TCN results)
#
# OUTPUT: CSV files:
#           - pandas_master_{excitement}.csv (merged results)
#           - pandas_overall_{excitement}.csv (overall metrics)
#           - pandas_coverage_{excitement}.csv (coverage rates)
#           - pandas_width_{excitement}.csv (interval widths)
#           - pandas_MAE_{excitement}.csv (MAE by variable)
#           - pandas_RMSE_{excitement}.csv (RMSE by variable)
#         LaTeX file:
#           - {excitement}_final.tex (5 formatted tables)
#
# FILE DESCRIPTION: [results-processing, latex-export, model-comparison]
# Post-processing tool for comparing BVAR and TCN forecasting results.
# Loads results CSVs from both models, merges them, separates metrics into
# thematic tables (overall performance, coverage, width, MAE, RMSE), and
# exports to CSV and publication-ready LaTeX format for thesis inclusion.
#
# Configurable parameters:
#   - excitement_level: passed via command line (0p25, 0p50, 0p75)
#   - LaTeX table formatting: precision (.5f for metrics, .2f for runtime)
#   - Coverage/Width levels: [80, 90, 95]
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

import pandas as pd
import sys

# =============================================================================
# DATA LOADING AND PREPROCESSING
#
# Loads BVAR and TCN result CSVs for specified excitement level.
# Preprocesses by removing DIC columns from BVAR, adding data_tag_dist
# column to TCN, renaming model names, and merging into master DataFrame.
# =============================================================================

def load_and_process(excitement_level):
    """
    Load BVAR and TCN CSVs, preprocess, and merge.
    
    Args:
        excitement_level: String like "0p25", "0p50", "0p75"
    
    Returns:
        Merged DataFrame with 2 rows (BVAR_Lag_5, Model_16)
    """
    print(f"Processing excitement level: {excitement_level}")
    print("="*60)
    
    bvar_file = f'BVAR{excitement_level}_final.csv'
    tcn_file = f'TCN{excitement_level}_final.csv'
    
    # ========== 1. LOAD BVAR ==========
    print(f"\n1. Loading {bvar_file}...")
    df_bvar = pd.read_csv(bvar_file)
    print(f"   Shape: {df_bvar.shape}")
    
    dic_cols = [col for col in df_bvar.columns if 'DIC' in col]
    if dic_cols:
        print(f"   Removing DIC columns: {dic_cols}")
        df_bvar = df_bvar.drop(columns=dic_cols)
    
    df_bvar['model_name'] = df_bvar['model_name'].replace('./', 'BVAR_Lag_5')
    print(f"   Renamed model_name: './' → 'BVAR_Lag_5'")
    
    # ========== 2. LOAD TCN ==========
    print(f"\n2. Loading {tcn_file}...")
    df_tcn = pd.read_csv(tcn_file)
    print(f"   Shape: {df_tcn.shape}")
    
    # ALIGNMENT: Insert data_tag_dist after n_datasets for column matching
    n_datasets_idx = df_tcn.columns.get_loc('n_datasets')
    df_tcn.insert(n_datasets_idx + 1, 'data_tag_dist', 'na')
    print(f"   Added data_tag_dist column with 'na'")
    
    df_tcn['model_name'] = df_tcn['model_name'].replace('./', 'Model_16')
    print(f"   Renamed model_name: './' → 'Model_16'")
    
    # ========== 3. MERGE ==========
    print(f"\n3. Merging BVAR and TCN...")
    pandas_master = pd.concat([df_bvar, df_tcn], ignore_index=True)
    print(f"   Master shape: {pandas_master.shape}")
    print(f"   Models: {list(pandas_master['model_name'])}")
    
    return pandas_master

# =============================================================================
# TABLE SEPARATION
#
# Separates master DataFrame into five thematic tables for analysis:
# 1. Overall performance (runtime, overall RMSE/MAE, RMSE/MAE ratio)
# 2. Coverage metrics (80%, 90%, 95% prediction intervals)
# 3. Width metrics (80%, 90%, 95% interval widths)
# 4. MAE by variable (3 variables × mean/std)
# 5. RMSE by variable (3 variables × mean/std)
# =============================================================================

def separate_tables(pandas_master):
    """
    Separate master DataFrame into thematic tables.
    
    Args:
        pandas_master: Merged DataFrame
    
    Returns:
        Dictionary with separated tables
    """
    print(f"\n4. Separating into thematic tables...")
    print("="*60)
    
    tables = {}
    
    # ========== pandas_overall ==========
    overall_cols = ['model_name', 'data_tag_dist', 'runtime_total']
    tables['pandas_overall'] = pandas_master[overall_cols].copy()
    
    # AGGREGATE: Overall mean across 3 variables
    tables['pandas_overall']['RMSE_overall_mean'] = pandas_master[[
        'RMSE_var1_mean', 'RMSE_var2_mean', 'RMSE_var3_mean'
    ]].mean(axis=1)
    
    tables['pandas_overall']['RMSE_overall_std'] = pandas_master[[
        'RMSE_var1_std', 'RMSE_var2_std', 'RMSE_var3_std'
    ]].mean(axis=1)
    
    tables['pandas_overall']['MAE_overall_mean'] = pandas_master[[
        'MAE_var1_mean', 'MAE_var2_mean', 'MAE_var3_mean'
    ]].mean(axis=1)
    
    tables['pandas_overall']['MAE_overall_std'] = pandas_master[[
        'MAE_var1_std', 'MAE_var2_std', 'MAE_var3_std'
    ]].mean(axis=1)
    
    # RATIO: RMSE/MAE indicates error distribution shape
    tables['pandas_overall']['RMSE_MAE_ratio'] = (
        tables['pandas_overall']['RMSE_overall_mean'] / 
        tables['pandas_overall']['MAE_overall_mean']
    )
    
    print(f"\npandas_overall: {tables['pandas_overall'].shape[1]} columns")
    print(f"  Base columns: {overall_cols}")
    print(f"  Calculated metrics: RMSE_overall_mean, RMSE_overall_std, MAE_overall_mean, MAE_overall_std, RMSE_MAE_ratio")
    
    # ========== pandas_coverage ==========
    coverage_cols = ['model_name'] + [col for col in pandas_master.columns if 'Coverage' in col]
    tables['pandas_coverage'] = pandas_master[coverage_cols].copy()
    print(f"\npandas_coverage: {tables['pandas_coverage'].shape[1]} columns")
    print(f"  Total coverage metrics: {len(coverage_cols) - 1}")
    
    # ========== pandas_width ==========
    width_cols = ['model_name'] + [col for col in pandas_master.columns if 'Width' in col]
    tables['pandas_width'] = pandas_master[width_cols].copy()
    print(f"\npandas_width: {tables['pandas_width'].shape[1]} columns")
    print(f"  Total width metrics: {len(width_cols) - 1}")
    
    # ========== pandas_MAE ==========
    mae_cols = ['model_name'] + [col for col in pandas_master.columns if 'MAE' in col]
    tables['pandas_MAE'] = pandas_master[mae_cols].copy()
    print(f"\npandas_MAE: {tables['pandas_MAE'].shape[1]} columns")
    print(f"  Total MAE metrics: {len(mae_cols) - 1}")
    
    # ========== pandas_RMSE ==========
    rmse_cols = ['model_name'] + [col for col in pandas_master.columns if 'RMSE' in col]
    tables['pandas_RMSE'] = pandas_master[rmse_cols].copy()
    print(f"\npandas_RMSE: {tables['pandas_RMSE'].shape[1]} columns")
    print(f"  Total RMSE metrics: {len(rmse_cols) - 1}")
    
    return tables

# =============================================================================
# LATEX EXPORT
#
# Exports five publication-ready LaTeX tables to single .tex file:
# 1. Overall Performance (horizontal layout)
# 2. Coverage Metrics (transposed, 80/90/95%)
# 3. Width Metrics (transposed, 80/90/95%)
# 4. RMSE by Variable (horizontal layout)
# 5. MAE by Variable (horizontal layout)
#
# Uses booktabs package formatting (\toprule, \midrule, \bottomrule).
# =============================================================================

def export_to_latex(tables, excitement_level):
    """
    Export all tables to LaTeX format in single .tex file.
    
    Args:
        tables: Dictionary of separated tables
        excitement_level: String like "0p25", "0p50", "0p75"
    """
    print(f"\n5. Exporting to LaTeX...")
    print("="*60)
    
    exc_readable = excitement_level.replace('p', '.')
    
    output_filename = f'{excitement_level}_final.tex'
    
    with open(output_filename, 'w') as f:
        f.write(f"% Final Results Tables - Excitement Level {exc_readable}\n")
        f.write(f"% Generated automatically\n\n")
        
        # ========== TABLE 1: OVERALL ==========
        f.write("% Table 1: Overall Performance\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{Overall Performance Metrics (Excitement = {exc_readable})}}\n")
        f.write(f"\\label{{tab:overall_{excitement_level}}}\n")
        f.write("\\begin{tabular}{lcccccccc}\n")
        f.write("\\toprule\n")
        
        df = tables['pandas_overall']
        f.write("Model & Data Tag & Runtime & RMSE & RMSE & MAE & MAE & RMSE/ \\\\\n")
        f.write("      &          & (sec)   & (mean) & (std) & (mean) & (std) & MAE \\\\\n")
        f.write("\\midrule\n")
        
        for idx, row in df.iterrows():
            f.write(f"{row['model_name'].replace('_', ' ')} & ")
            f.write(f"{row['data_tag_dist']} & ")
            f.write(f"{row['runtime_total']:.2f} & ")
            f.write(f"{row['RMSE_overall_mean']:.5f} & ")
            f.write(f"{row['RMSE_overall_std']:.5f} & ")
            f.write(f"{row['MAE_overall_mean']:.5f} & ")
            f.write(f"{row['MAE_overall_std']:.5f} & ")
            f.write(f"{row['RMSE_MAE_ratio']:.5f} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")
        
        # ========== TABLE 2: COVERAGE (TRANSPOSED) ==========
        f.write("% Table 2: Coverage Metrics\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{Prediction Interval Coverage Rates (Excitement = {exc_readable})}}\n")
        f.write(f"\\label{{tab:coverage_{excitement_level}}}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        
        df_cov = tables['pandas_coverage']
        models = df_cov['model_name'].values
        f.write(f" & \\multicolumn{{2}}{{c}}{{{models[0].replace('_', ' ')}}} & ")
        f.write(f"\\multicolumn{{2}}{{c}}{{{models[1].replace('_', ' ')}}} \\\\\n")
        f.write("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}\n")
        f.write(" & Mean & SD & Mean & SD \\\\\n")
        f.write("\\midrule\n")
        
        for level in [80, 90, 95]:
            f.write(f"\\multicolumn{{5}}{{l}}{{\\textit{{Coverage {level}\\%}}}} \\\\\n")
            
            for var in [1, 2, 3]:
                col_mean = f'Coverage{level}_var{var}_mean'
                col_std = f'Coverage{level}_var{var}_std'
                
                f.write(f"Variable {var} & ")
                f.write(f"{df_cov.iloc[0][col_mean]:.5f} & ")
                f.write(f"{df_cov.iloc[0][col_std]:.5f} & ")
                f.write(f"{df_cov.iloc[1][col_mean]:.5f} & ")
                f.write(f"{df_cov.iloc[1][col_std]:.5f} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")
        
        # ========== TABLE 3: WIDTH (TRANSPOSED) ==========
        f.write("% Table 3: Width Metrics\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{Prediction Interval Widths (Excitement = {exc_readable})}}\n")
        f.write(f"\\label{{tab:width_{excitement_level}}}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        
        df_width = tables['pandas_width']
        f.write(f" & \\multicolumn{{2}}{{c}}{{{models[0].replace('_', ' ')}}} & ")
        f.write(f"\\multicolumn{{2}}{{c}}{{{models[1].replace('_', ' ')}}} \\\\\n")
        f.write("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}\n")
        f.write(" & Mean & SD & Mean & SD \\\\\n")
        f.write("\\midrule\n")
        
        for level in [80, 90, 95]:
            f.write(f"\\multicolumn{{5}}{{l}}{{\\textit{{Width {level}\\%}}}} \\\\\n")
            
            for var in [1, 2, 3]:
                col_mean = f'Width{level}_var{var}_mean'
                col_std = f'Width{level}_var{var}_std'
                
                f.write(f"Variable {var} & ")
                f.write(f"{df_width.iloc[0][col_mean]:.5f} & ")
                f.write(f"{df_width.iloc[0][col_std]:.5f} & ")
                f.write(f"{df_width.iloc[1][col_mean]:.5f} & ")
                f.write(f"{df_width.iloc[1][col_std]:.5f} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")
        
        # ========== TABLE 4: RMSE ==========
        f.write("% Table 4: RMSE Metrics\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{Root Mean Squared Error by Variable (Excitement = {exc_readable})}}\n")
        f.write(f"\\label{{tab:rmse_{excitement_level}}}\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\toprule\n")
        
        df_rmse = tables['pandas_RMSE']
        f.write(" & \\multicolumn{3}{c}{RMSE (mean)} & \\multicolumn{3}{c}{RMSE (std)} \\\\\n")
        f.write("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}\n")
        f.write(" & Var 1 & Var 2 & Var 3 & Var 1 & Var 2 & Var 3 \\\\\n")
        f.write("\\midrule\n")
        
        for idx, row in df_rmse.iterrows():
            f.write(f"{row['model_name'].replace('_', ' ')} & ")
            f.write(f"{row['RMSE_var1_mean']:.5f} & ")
            f.write(f"{row['RMSE_var2_mean']:.5f} & ")
            f.write(f"{row['RMSE_var3_mean']:.5f} & ")
            f.write(f"{row['RMSE_var1_std']:.5f} & ")
            f.write(f"{row['RMSE_var2_std']:.5f} & ")
            f.write(f"{row['RMSE_var3_std']:.5f} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")
        
        # ========== TABLE 5: MAE ==========
        f.write("% Table 5: MAE Metrics\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{Mean Absolute Error by Variable (Excitement = {exc_readable})}}\n")
        f.write(f"\\label{{tab:mae_{excitement_level}}}\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\toprule\n")
        
        df_mae = tables['pandas_MAE']
        f.write(" & \\multicolumn{3}{c}{MAE (mean)} & \\multicolumn{3}{c}{MAE (std)} \\\\\n")
        f.write("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}\n")
        f.write(" & Var 1 & Var 2 & Var 3 & Var 1 & Var 2 & Var 3 \\\\\n")
        f.write("\\midrule\n")
        
        for idx, row in df_mae.iterrows():
            f.write(f"{row['model_name'].replace('_', ' ')} & ")
            f.write(f"{row['MAE_var1_mean']:.5f} & ")
            f.write(f"{row['MAE_var2_mean']:.5f} & ")
            f.write(f"{row['MAE_var3_mean']:.5f} & ")
            f.write(f"{row['MAE_var1_std']:.5f} & ")
            f.write(f"{row['MAE_var2_std']:.5f} & ")
            f.write(f"{row['MAE_var3_std']:.5f} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"\n✓ Saved: {output_filename}")
    print(f"  Contains 5 LaTeX tables:")
    print(f"    1. Overall Performance")
    print(f"    2. Coverage Metrics (transposed)")
    print(f"    3. Width Metrics (transposed)")
    print(f"    4. RMSE by Variable")
    print(f"    5. MAE by Variable")

# =============================================================================
# MAIN
# =============================================================================

def main():
    if len(sys.argv) != 2:
        print("Usage: python process_results.py <excitement_level>")
        print("Example: python process_results.py 0p50")
        sys.exit(1)
    
    excitement_level = sys.argv[1]
    
    # ========== 1. LOAD AND PROCESS ==========
    pandas_master = load_and_process(excitement_level)
    
    # ========== 2. SEPARATE INTO THEMATIC TABLES ==========
    tables = separate_tables(pandas_master)
    
    # ========== 3. SAVE CSV OUTPUTS ==========
    print(f"\n5. Saving CSV outputs...")
    print("="*60)
    
    pandas_master.to_csv(f'pandas_master_{excitement_level}.csv', index=False)
    print(f"\nSaved: pandas_master_{excitement_level}.csv")
    
    for name, df in tables.items():
        filename = f'{name}_{excitement_level}.csv'
        df.to_csv(filename, index=False)
        print(f"Saved: {filename}")
    
    # ========== 4. EXPORT TO LATEX ==========
    export_to_latex(tables, excitement_level)
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)

if __name__ == '__main__':
    main()
