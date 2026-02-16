#!/usr/bin/env python3
# This file is a part of a Master Thesis titled:
# "Bayesian VAR vs. Temporal Convolutional Networks: A Simulation Study 
# on Forecasting Accuracy and Uncertainty Calibration"
#
# FILENAME: gen_data_script.py
# AUTHOR: Zeno S. alias Zeno Fontaine
# DATE: 2026-02-06
# 
# INPUT: None (configurable parameters below)
#
# OUTPUT: CSV files containing simulated multivariate market data
#         organized by excitement level in data_folder/
#
# FILE DESCRIPTION: data-generation, simulation, market-dynamics
# Generates multiple datasets of simulated multivariate market prices
# across different volatility regimes (excitement levels). Uses the
# stochastic volatility jump diffusion model from simulation.py to create
# reproducible datasets for testing forecasting models. Each excitement
# level gets multiple independent datasets with unique but reproducible
# seeds derived from a master seed.
#
# Configurable parameters:
#   - MASTER_SEED: Root seed for full reproducibility
#   - N_DATASETS: Number of independent datasets per excitement level
#   - N_OBS: Time series length (observations per dataset)
#   - N_VARS: Number of assets/variables per dataset
#   - excitement_levels: List of volatility regimes to simulate
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

import os
import numpy as np
import pandas as pd
from simulation import simulate_multivariate_market

# =============================================================================
# CONFIGURATION
# =============================================================================

MASTER_SEED = 666       # Master seed for reproducibility
N_DATASETS = 5          # Number of datasets per excitement level
N_OBS = 5250            # Number of observations per dataset
N_VARS = 3              # Number of variables/assets
DATA_FOLDER = "data_folder"  # Base directory for all data

excitement_levels = [0.25, 0.5, 0.75]

# =============================================================================
# MAIN DATA GENERATION PIPELINE
# =============================================================================

if __name__ == "__main__":
    # ========== 1. INITIALIZE SEED SYSTEM ==========
    # CRITICAL: Master seed ensures full reproducibility
    np.random.seed(MASTER_SEED)
    
    # Generate unique seeds for each dataset deterministically
    dataset_seeds = np.random.randint(0, 1000000, size=(len(excitement_levels), N_DATASETS))
    
    print(f"Starting data generation...")
    print(f"Master seed: {MASTER_SEED}")
    print(f"Generating {N_DATASETS} datasets for each excitement level")
    print(f"Each dataset: {N_OBS} observations, {N_VARS} assets")
    print("-" * 50)
    
    # ========== 2. GENERATE DATASETS FOR EACH EXCITEMENT LEVEL ==========
    for exc_idx, excitement in enumerate(excitement_levels):
        folder_name = f"excitement_{excitement:.2f}"
        folder_path = os.path.join(DATA_FOLDER, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        print(f"\nGenerating data for excitement level {excitement}...")
        
        # ========== 3. CREATE MULTIPLE DATASETS PER LEVEL ==========
        for dataset_num in range(1, N_DATASETS + 1):
            seed = dataset_seeds[exc_idx, dataset_num - 1]
            
            # Generate synthetic market data
            prices = simulate_multivariate_market(
                n_obs=N_OBS, 
                n_vars=N_VARS, 
                excitement=excitement,
                seed=seed
            )
            
            # Convert to DataFrame with asset labels
            df = pd.DataFrame(
                prices,
                columns=[f"asset_{i+1}" for i in range(N_VARS)]
            )
            df.index.name = 'time'
            
            # ========== 4. SAVE TO CSV ==========
            filename = f"d{dataset_num}.csv"
            filepath = os.path.join(folder_path, filename)
            df.to_csv(filepath)
            
            print(f"  Created {filename} (seed: {seed})")
        
        print(f"  Completed {N_DATASETS} datasets in {folder_path}")
    
    # ========== 5. SUMMARY ==========
    print("\n" + "=" * 50)
    print("Data generation complete!")
    print(f"Total datasets created: {len(excitement_levels) * N_DATASETS}")
    print(f"Location: {DATA_FOLDER}/")
    print("\nFolder structure:")
    for excitement in excitement_levels:
        print(f"  excitement_{excitement:.2f}/")
        print(f"    ├── d1.csv through d{N_DATASETS}.csv")
