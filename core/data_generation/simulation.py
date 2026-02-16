#!/usr/bin/env python3
# This file is a part of a Master Thesis titled:
# "Bayesian VAR vs. Temporal Convolutional Networks: A Simulation Study 
# on Forecasting Accuracy and Uncertainty Calibration"
#
# FILENAME: simulation.py
# AUTHOR: Zeno S. alias Zeno Fontaine
# DATE: 2026-02-06
# 
# INPUT: Parameters for n_obs, n_vars, excitement level, and seed
#
# OUTPUT: NumPy array of shape (n_obs, n_vars) containing simulated price paths
#
# FILE DESCRIPTION: stochastic-volatility, jump-diffusion, market-simulation
# Implements a multivariate stochastic volatility model with correlated jump
# diffusion for simulating realistic market price dynamics. The model combines:
# (1) Ornstein-Uhlenbeck process for time-varying volatility, (2) correlated
# Brownian motion for normal returns, and (3) correlated jump process for
# extreme events. The 'excitement' parameter controls regime intensity by
# scaling jump frequency, volatility persistence, and cross-asset correlations.
# Calibrated for minute-level financial data with stability constraints.
#
# Configurable parameters:
#   - n_obs: Number of time observations to generate
#   - n_vars: Number of correlated assets
#   - excitement: Volatility regime intensity [0=calm, 1=turbulent]
#   - seed: Random seed for reproducibility
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

import numpy as np

# =============================================================================
# MULTIVARIATE STOCHASTIC VOLATILITY SIMULATION
#
# Stochastic volatility follows Ornstein-Uhlenbeck in log space for stability.
# Jump process uses copula-like structure for correlated extreme events.
# Excitement parameter scales jump intensity, vol persistence, and correlations.
# Returns capped to prevent numerical explosions in extreme regimes.
# =============================================================================

def simulate_multivariate_market(n_obs=6000, n_vars=3, excitement=0.5, seed=None):
    """
    Generate multivariate price series with stochastic volatility and jumps.
    
    Parameters
    ----------
    n_obs : int
        Number of observations (default 6000 = 3 weeks minute data).
    n_vars : int
        Number of assets.
    excitement : float
        Regime intensity in [0,1]. Controls jump frequency, volatility
        persistence, and correlations.
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    prices : ndarray of shape (n_obs, n_vars)
        Simulated price paths starting at 100.
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize state variables
    returns = np.zeros((n_obs, n_vars))
    log_volatility = np.zeros((n_vars,))
    
    # Excitement-scaled parameters (minute-level calibration)
    base_jump_prob = 0.0005 + 0.001 * excitement
    vol_of_vol = 0.05 + 0.1 * excitement
    vol_mean_reversion = 0.1 * (1 - 0.5 * excitement)
    jump_correlation = 0.3 + 0.6 * excitement
    
    # Correlation structure
    base_correlation = 0.2 + 0.4 * excitement
    L = generate_cholesky(n_vars, base_correlation)
    
    for t in range(1, n_obs):
        # Update stochastic volatility (Ornstein-Uhlenbeck)
        vol_shocks = np.random.normal(0, vol_of_vol, n_vars)
        log_volatility = (log_volatility * (1 - vol_mean_reversion) + 
                         vol_mean_reversion * 0 +
                         vol_shocks)
        
        # CAP: Prevent volatility explosion/collapse
        log_volatility = np.clip(log_volatility, -2, 2)
        
        current_vol = np.exp(log_volatility/2)
        
        # Generate correlated diffusion component
        minute_scale = 0.001
        independent_normals = np.random.normal(0, 1, n_vars)
        correlated_normals = L @ independent_normals
        diffusion_component = minute_scale * current_vol * correlated_normals
        
        # Correlated jump component
        if np.random.random() < base_jump_prob:
            jump_vol = 0.002 + 0.003 * excitement
            
            # Copula-like jump structure
            common_jump = np.random.normal(0, jump_vol)
            idiosyncratic_jumps = np.random.normal(0, jump_vol, n_vars)
            
            jump_component = (jump_correlation * common_jump + 
                            np.sqrt(1 - jump_correlation**2) * idiosyncratic_jumps)
        else:
            jump_component = np.zeros(n_vars)
        
        # Combine components
        returns[t] = diffusion_component + jump_component
        
        # CAP: Prevent extreme returns from breaking numerical stability
        returns[t] = np.clip(returns[t], -0.05, 0.05)
    
    # Convert to prices (all start at 100)
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    
    return prices

# =============================================================================
# CORRELATION MATRIX HELPER
# =============================================================================

def generate_cholesky(n_vars, correlation):
    """
    Generate Cholesky decomposition for correlation matrix.
    
    Parameters
    ----------
    n_vars : int
        Dimension of correlation matrix.
    correlation : float
        Off-diagonal correlation value.
    
    Returns
    -------
    L : ndarray of shape (n_vars, n_vars)
        Lower triangular Cholesky factor.
    """
    corr_matrix = np.full((n_vars, n_vars), correlation)
    np.fill_diagonal(corr_matrix, 1.0)
    return np.linalg.cholesky(corr_matrix)
