#!/usr/bin/env python3
# This file is a part of a Master Thesis titled:
# "Bayesian VAR vs. Temporal Convolutional Networks:
# A Simulation Study on Forecasting Accuracy and
# Uncertainty Calibration" produced by the author
# at Universidad Carlos III. de Madrid (UC3M) in 2026.
# All rights reserved.
#
# FILENAME: bvar_model.py
# AUTHOR: Zeno S. alias Zeno Fontaine
# DATE: 2026-02-06
# 
# INPUT: CSV file with time series data (6250 observations, 3 variables)
#        Command-line arguments: --input, --output, --lags, --plots, --irfplus
#
# OUTPUT: CSV with predictions and quantiles for test set (5000-6249)
#         Optional: diagnostic plots, IRF plots, FEVD plots
#         Header comments in CSV: runtime, DIC, data_tag
#
# FILE DESCRIPTION: [bayesian-var, forecasting, minnesota-prior, time-series]
# Bayesian Vector Autoregression model with Minnesota prior for multi-asset
# forecasting. Implements stationarity testing via ADF, differencing if needed,
# PyMC-based Bayesian inference with MCMC sampling, and generates forecasts
# with prediction intervals. Includes optional IRF and FEVD analysis.
#
# Configurable parameters:
#   - N_TRAINING, N_FORECAST: dataset split sizes
#   - LAMBDA_PARAM, TAU, DELTA: Minnesota prior hyperparameters
#   - MCMC_DRAWS, MCMC_TUNE, MCMC_CHAINS: MCMC sampler settings
#   - TARGET_ACCEPT, MAX_TREEDEPTH: NUTS sampler tuning
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
import time
import os
import warnings
import numpy as np
import pandas as pd
import pymc as pm
from statsmodels.tsa.stattools import adfuller
from scipy.stats import multivariate_normal

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

N_TRAINING = 5000
N_FORECAST = 1250  

# Minnesota Prior Hyperparameters
LAMBDA_PARAM = 0.2  # Overall tightness
TAU = 10            # Cross-variable shrinkage
DELTA = 1           # Lag decay rate

# MCMC Settings
MCMC_DRAWS = 2000
MCMC_TUNE = 1000
MCMC_CHAINS = 4
TARGET_ACCEPT = 0.9
MAX_TREEDEPTH = 12
SEED_CHOICE = 1964


# =============================================================================
# STATIONARITY HANDLING
# =============================================================================

def check_and_transform_stationarity(data):
    """
    Check stationarity via ADF test and apply differencing if needed.
    
    Returns:
        transformed_data: Stationary data
        data_tag: 0=stationary, 1=first diff, 2=second diff
        diff_info: Dict with last values for integration
    """
    n_vars = data.shape[1]
    data_tag = 0
    diff_info = {'last_values': None, 'second_last_values': None}
    transformed_data = data.copy()
    
    print("  Stationarity test:")
    
    for diff_level in range(3):
        all_stationary = True
        
        for var_idx in range(n_vars):
            series = transformed_data[:, var_idx]
            adf_result = adfuller(series, autolag='AIC')
            adf_pval = adf_result[1]
            is_stationary = adf_pval < 0.05
            
            if diff_level == 0:
                print(f"    Var {var_idx + 1}: ADF p={adf_pval:.4f} {'✓' if is_stationary else '✗'}")
            
            if not is_stationary:
                all_stationary = False
        
        if all_stationary:
            print(f"    → Stationary at diff level {diff_level}")
            break
        elif diff_level < 2:
            if diff_level == 0:
                diff_info['last_values'] = data[-1, :].copy()
                diff_info['second_last_values'] = data[-2, :].copy()
            
            transformed_data = np.diff(transformed_data, axis=0)
            data_tag = diff_level + 1
            print(f"    → Applying differencing (level {diff_level + 1})")
    
    if not all_stationary:
        print("    ⚠ Still non-stationary after 2nd diff, proceeding anyway")
    
    return transformed_data, data_tag, diff_info


def integrate_forecasts(forecast_samples, data_tag, diff_info):
    """
    Integrate differenced forecasts back to levels.
    Handles both first and second order differencing.
    """
    if data_tag == 0:
        return forecast_samples
    
    n_samples, n_forecast, k = forecast_samples.shape
    integrated = forecast_samples.copy()
    
    if data_tag == 1:
        for s in range(n_samples):
            last_value = diff_info['last_values'].copy()
            for h in range(n_forecast):
                integrated[s, h, :] = last_value + forecast_samples[s, h, :]
                last_value = integrated[s, h, :]
    
    elif data_tag == 2:
        # STEP 1: Second diff -> First diff
        for s in range(n_samples):
            last_diff = diff_info['last_values'] - diff_info['second_last_values']
            first_integrated = np.zeros((n_forecast, k))
            for h in range(n_forecast):
                first_integrated[h, :] = last_diff + forecast_samples[s, h, :]
                last_diff = first_integrated[h, :]
            
            # STEP 2: First diff -> Levels
            last_value = diff_info['last_values'].copy()
            for h in range(n_forecast):
                integrated[s, h, :] = last_value + first_integrated[h, :]
                last_value = integrated[s, h, :]
    
    return integrated

# =============================================================================
# MINNESOTA PRIOR CONSTRUCTION
#
# Estimate AR(1) coefficients and residual variances for each variable.
# Prior mean: AR(1) on diagonal of first lag block, zeros elsewhere.
# Prior covariance: own-lag variance scales with LAMBDA_PARAM/lag^DELTA,
# cross-lag variance additionally divided by TAU and scaled by residual variance ratio.
# =============================================================================


def calculate_minnesota_prior(data, lags):
    """
    Calculate Minnesota prior mean and covariance.
    
    Prior structure:
    - Own first lag: AR(1) coefficient from data
    - Cross-variable lags: zero with tight shrinkage
    - Higher lags: decay by 1/lag^delta
    """
    
    k = data.shape[1]
    p = lags
    
    ar1_coeffs = np.zeros(k)
    residual_vars = np.zeros(k)
    
    for i in range(k):
        if len(data) > 1:
            y = data[1:, i]
            X = data[:-1, i].reshape(-1, 1)
            X_with_const = np.column_stack([np.ones(len(X)), X])
            
            try:
                beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
                ar1_coeffs[i] = beta[1]
                residuals = y - X_with_const @ beta
                residual_vars[i] = np.var(residuals)
            except:
                ar1_coeffs[i] = 0.5
                residual_vars[i] = np.var(data[:, i])
        else:
            ar1_coeffs[i] = 0.5
            residual_vars[i] = 1.0
    
    residual_vars = np.maximum(residual_vars, 1e-6)
    
    prior_mean_size = (1 + k * p) * k
    prior_mean = np.zeros(prior_mean_size)
    
    for i in range(k):
        pos = i * (1 + k * p) + 1 + i
        prior_mean[pos] = ar1_coeffs[i]
    
    prior_cov = np.zeros((prior_mean_size, prior_mean_size))
    
    for i in range(k):
        pos = i * (1 + k * p)
        prior_cov[pos, pos] = 10.0 * residual_vars[i]
    
    for lag in range(1, p + 1):
        for i in range(k):
            for j in range(k):
                pos = i * (1 + k * p) + 1 + (lag - 1) * k + j
                
                if i == j:
                    var = (LAMBDA_PARAM / (lag ** DELTA)) ** 2
                else:
                    var = (LAMBDA_PARAM / (lag ** DELTA)) ** 2 * \
                          (residual_vars[i] / residual_vars[j]) / TAU
                
                prior_cov[pos, pos] = var
    
    # ENSURE POSITIVE DEFINITE!
    min_eigenval = np.min(np.linalg.eigvalsh(prior_cov))
    if min_eigenval < 1e-6:
        prior_cov += (1e-6 - min_eigenval) * np.eye(prior_mean_size)
    
    return prior_mean, prior_cov

# =============================================================================
# BVAR MODEL FITTING
# =============================================================================

def fit_bvar_model(data, lags, prior_mean, prior_cov):
    """
    Fit BVAR model using PyMC with Minnesota prior.
    
    Returns:
        model: PyMC model
        trace: Posterior samples
        Y: Response matrix
        X: Design matrix
    """
    n_obs, k = data.shape
    p = lags
    
    Y = data[p:, :]
    X = np.ones((n_obs - p, 1))
    
    for lag in range(1, p + 1):
        X = np.hstack([X, data[p - lag:-lag, :]])
    
    n_samples = Y.shape[0]
    
    print(f"  Model: Y={Y.shape}, X={X.shape}, params={len(prior_mean)}")
    
    with pm.Model() as model:
        beta = pm.MvNormal('beta', mu=prior_mean, cov=prior_cov, 
                          shape=len(prior_mean))
        
        sd_dist = pm.Exponential.dist(1.0, shape=k)
        chol, corr, stds = pm.LKJCholeskyCov(
            'chol_cov', n=k, eta=2.0, sd_dist=sd_dist, compute_corr=True
        )
        
        cov = pm.Deterministic('cov', chol.dot(chol.T))
        
        beta_reshaped = beta.reshape((1 + k * p, k))
        mu = pm.math.dot(X, beta_reshaped)
        
        Y_obs = pm.MvNormal('Y_obs', mu=mu, chol=chol, observed=Y, shape=(n_samples, k))
    
    print(f"  MCMC: {MCMC_DRAWS} draws, {MCMC_TUNE} tune, {MCMC_CHAINS} chains")
    
    with model:
        trace = pm.sample(
            draws=MCMC_DRAWS,
            tune=MCMC_TUNE,
            chains=MCMC_CHAINS,
            cores=min(MCMC_CHAINS, os.cpu_count() or 1),
            progressbar=True,
            target_accept=TARGET_ACCEPT,
            return_inferencedata=True,
            nuts_sampler_kwargs={'max_treedepth': MAX_TREEDEPTH},
            random_seed=SEED_CHOICE
        )
    
    return model, trace, Y, X

# =============================================================================
# DIC CALCULATION
# =============================================================================

def calculate_dic(trace, Y, X, lags):
    """
    Calculate DIC (Deviance Information Criterion).
    
    DIC = -2 * log p(y|theta_mean) + 2 * p_DIC
    where p_DIC is the effective number of parameters.
    """
    k = Y.shape[1]
    p = lags
    
    beta_mean = trace.posterior['beta'].mean(dim=['chain', 'draw']).values
    cov_mean = trace.posterior['cov'].mean(dim=['chain', 'draw']).values
    
    beta_reshaped = beta_mean.reshape((1 + k * p, k))
    mu_mean = X @ beta_reshaped
    
    log_lik_mean = 0
    for i in range(len(Y)):
        log_lik_mean += multivariate_normal.logpdf(Y[i], mu_mean[i], cov_mean)
    
    beta_samples = trace.posterior['beta'].values
    cov_samples = trace.posterior['cov'].values
    
    n_chains, n_draws = beta_samples.shape[:2]
    n_samples = min(n_chains * n_draws, 100)
    
    sample_indices = np.random.choice(n_chains * n_draws, n_samples, replace=False)
    
    log_liks = []
    for idx in sample_indices:
        chain_idx = idx // n_draws
        draw_idx = idx % n_draws
        
        beta_s = beta_samples[chain_idx, draw_idx].reshape((1 + k * p, k))
        cov_s = cov_samples[chain_idx, draw_idx]
        
        mu_s = X @ beta_s
        log_lik_s = 0
        for i in range(len(Y)):
            log_lik_s += multivariate_normal.logpdf(Y[i], mu_s[i], cov_s)
        log_liks.append(log_lik_s)
    
    p_dic = 2 * (log_lik_mean - np.mean(log_liks))
    dic = -2 * log_lik_mean + 2 * p_dic
    
    return dic

# =============================================================================
# VECTORIZED FORECAST GENERATION
#
# Pre-generate all correlated noise using Cholesky decomposition of covariance.
# For each horizon: build design matrix X from lagged window, compute predictions
# via einsum batch multiply, add noise, update rolling window.
# Finally integrate back to levels if data was differenced.
# =============================================================================

def generate_forecasts(trace, train_data, test_actual, lags, data_tag, diff_info):
    """
    Generate forecasts with uncertainty quantification.
    All posterior samples processed simultaneously for each horizon step.
    """
    
    k = train_data.shape[1]
    p = lags
    n_forecast = len(test_actual)
    
    beta_samples = trace.posterior['beta'].values
    cov_samples = trace.posterior['cov'].values
    
    n_chains, n_draws = beta_samples.shape[:2]
    n_samples = n_chains * n_draws
    
    beta_samples = beta_samples.reshape(n_samples, -1)
    cov_samples = cov_samples.reshape(n_samples, k, k)
    
    beta_reshaped = beta_samples.reshape(n_samples, 1 + k * p, k)
    
    forecast_samples = np.zeros((n_samples, n_forecast, k))
    
    current_window = np.tile(train_data[-p:, :], (n_samples, 1, 1))
    
    print("    Pre-generating correlated noise...")
    all_noise = np.zeros((n_samples, n_forecast, k))
    for s in range(n_samples):
        try:
            L = np.linalg.cholesky(cov_samples[s])
            z = np.random.randn(n_forecast, k)
            all_noise[s] = z @ L.T
        except np.linalg.LinAlgError:
            # FALLBACK: Independent noise if covariance not PD
            std = np.sqrt(np.diag(cov_samples[s]))
            all_noise[s] = np.random.randn(n_forecast, k) * std
    
    print("    Generating forecasts (vectorized)...")
    
    for h in range(n_forecast):
        if h % 250 == 0:
            print(f"      Horizon {h}/{n_forecast}...")
        
        X_pred = np.ones((n_samples, 1 + k * p))
        
        for lag in range(1, p + 1):
            start_idx = 1 + (lag - 1) * k
            end_idx = 1 + lag * k
            X_pred[:, start_idx:end_idx] = current_window[:, -lag, :]
        
        mean_pred = np.einsum('si,sik->sk', X_pred, beta_reshaped)
        
        y_pred = mean_pred + all_noise[:, h, :]
        
        forecast_samples[:, h, :] = y_pred
        
        current_window = np.concatenate([
            current_window[:, 1:, :],
            y_pred[:, np.newaxis, :]
        ], axis=1)
    
    print("    Forecasts complete!")
    
    if data_tag > 0:
        print("    Integrating back to levels...")
        forecast_samples = integrate_forecasts(forecast_samples, data_tag, diff_info)
    
    print("    Computing quantiles...")
    quantiles = [0.025, 0.05, 0.10, 0.20, 0.50, 0.80, 0.90, 0.95, 0.975]
    forecast_quantiles = np.percentile(forecast_samples, 
                                       [q * 100 for q in quantiles], axis=0)
    
    forecast_mean = np.mean(forecast_samples, axis=0)
    
    results_dict = {'time': np.arange(N_TRAINING, N_TRAINING + n_forecast)}
    
    for var_idx in range(k):
        var_name = f'var{var_idx + 1}'
        results_dict[f'{var_name}_actual'] = test_actual[:, var_idx]
        results_dict[f'{var_name}_pred'] = forecast_mean[:, var_idx]
        
        for q_idx, q in enumerate(quantiles):
            if q == 0.025:
                q_name = f'{var_name}_q2.5'
            elif q == 0.975:
                q_name = f'{var_name}_q97.5'
            else:
                q_name = f'{var_name}_q{int(q * 100)}'
            results_dict[q_name] = forecast_quantiles[q_idx, :, var_idx]
    
    return pd.DataFrame(results_dict)

# =============================================================================
# IMPULSE RESPONSE FUNCTION AND FEVD
#   FOR IRF
#
#   Extract VAR coefficient matrices A_1, ..., A_p from posterior beta.
#   For each posterior sample: apply Cholesky orthogonalized shock,
#   propagate forward using VAR recursion. Shock at t=0 is the Cholesky
#   column corresponding to shock_var.
#   
#   FOR FEVD
#   [missing]
# =============================================================================

def compute_irf(trace, lags, horizon=40, shock_var=0):
    """
    Compute Impulse Response Function from BVAR posterior.
    Shocks variable by one standard deviation using Cholesky identification.
    
    Returns:
        irf_samples: (n_samples, horizon, k) array of IRF responses
    """
    
    beta_samples = trace.posterior['beta'].values
    cov_samples = trace.posterior['cov'].values
    
    n_chains, n_draws = beta_samples.shape[:2]
    n_samples = n_chains * n_draws
    
    beta_samples = beta_samples.reshape(n_samples, -1)
    cov_samples = cov_samples.reshape(n_samples, cov_samples.shape[-2], cov_samples.shape[-1])
    
    k = cov_samples.shape[-1]
    p = lags
    
    beta_reshaped = beta_samples.reshape(n_samples, 1 + k * p, k)
    
    A_matrices = []
    for lag in range(p):
        start_idx = 1 + lag * k
        end_idx = 1 + (lag + 1) * k
        A_matrices.append(beta_reshaped[:, start_idx:end_idx, :])
    
    irf_samples = np.zeros((n_samples, horizon, k))
    
    print(f"    Computing IRF for {n_samples} posterior samples...")
    
    for s in range(n_samples):
        try:
            L = np.linalg.cholesky(cov_samples[s])
        except np.linalg.LinAlgError:
            # FIX: Add small diagonal if not PD
            cov_fixed = cov_samples[s] + 1e-6 * np.eye(k)
            L = np.linalg.cholesky(cov_fixed)
        
        shock = L[:, shock_var]
        
        irf_history = np.zeros((horizon + p, k))
        
        # INITIAL SHOCK at t=0
        irf_history[p, :] = shock
        
        for t in range(1, horizon):
            response = np.zeros(k)
            for lag in range(p):
                if p + t - 1 - lag >= 0:
                    response += A_matrices[lag][s].T @ irf_history[p + t - 1 - lag, :]
            irf_history[p + t, :] = response
        
        irf_samples[s, :, :] = irf_history[p:p+horizon, :]
    
    print(f"    IRF computation complete!")
    return irf_samples


def compute_fevd(irf_samples, horizon=40):
    """
    Compute Forecast Error Variance Decomposition from IRF.
    
    Returns:
        fevd_mean: (horizon, k) mean FEVD
        fevd_lower: (horizon, k) 16th percentile
        fevd_upper: (horizon, k) 84th percentile
    """
    n_samples, h, k = irf_samples.shape
    
    fevd_samples = np.zeros((n_samples, horizon, k))
    
    for s in range(n_samples):
        cum_sq = np.cumsum(irf_samples[s, :, :] ** 2, axis=0)
        
        total_var = np.sum(cum_sq, axis=1, keepdims=True)
        total_var = np.maximum(total_var, 1e-10)
        
        fevd_samples[s, :, :] = cum_sq / np.sum(cum_sq[-1, :])
    
    fevd_mean = np.mean(fevd_samples, axis=0)
    fevd_lower = np.percentile(fevd_samples, 16, axis=0)
    fevd_upper = np.percentile(fevd_samples, 84, axis=0)
    
    return fevd_mean, fevd_lower, fevd_upper


def create_irf_plot(irf_samples, output_path, horizon=40):
    """
    Create IRF plot showing response of all variables to var1 shock.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    k = irf_samples.shape[2]
    
    irf_mean = np.mean(irf_samples, axis=0)
    irf_68_lower = np.percentile(irf_samples, 16, axis=0)
    irf_68_upper = np.percentile(irf_samples, 84, axis=0)
    irf_95_lower = np.percentile(irf_samples, 2.5, axis=0)
    irf_95_upper = np.percentile(irf_samples, 97.5, axis=0)
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('Impulse Response Function: Shock to Variable 1', fontsize=14, fontweight='bold')
    
    x = np.arange(horizon)
    colors = ['steelblue', 'coral', 'seagreen']
    
    for var_idx in range(k):
        ax = axes[var_idx]
        
        ax.fill_between(x, irf_95_lower[:, var_idx], irf_95_upper[:, var_idx],
                       alpha=0.2, color=colors[var_idx], label='95% CI')
        ax.fill_between(x, irf_68_lower[:, var_idx], irf_68_upper[:, var_idx],
                       alpha=0.4, color=colors[var_idx], label='68% CI')
        ax.plot(x, irf_mean[:, var_idx], color=colors[var_idx], linewidth=2, label='Mean')
        
        ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
        
        ax.set_title(f'Response of var{var_idx + 1}')
        ax.set_xlabel('Horizon (steps)')
        ax.set_ylabel('Response')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_file = output_path.replace('.csv', '_irf.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  IRF plot saved: {plot_file}")
    return plot_file


def create_fevd_plot(fevd_mean, fevd_lower, fevd_upper, output_path, horizon=40):
    """
    Create FEVD plot showing variance decomposition.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    k = fevd_mean.shape[1]
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('Forecast Error Variance Decomposition: Contribution of var1 Shock', 
                 fontsize=14, fontweight='bold')
    
    x = np.arange(horizon)
    colors = ['steelblue', 'coral', 'seagreen']
    
    for var_idx in range(k):
        ax = axes[var_idx]
        
        ax.fill_between(x, fevd_lower[:, var_idx], fevd_upper[:, var_idx],
                       alpha=0.3, color=colors[var_idx], label='68% CI')
        ax.plot(x, fevd_mean[:, var_idx], color=colors[var_idx], linewidth=2, label='Mean')
        
        ax.set_title(f'FEVD of var{var_idx + 1}')
        ax.set_xlabel('Horizon (steps)')
        ax.set_ylabel('Proportion of Variance')
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_file = output_path.replace('.csv', '_fevd.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  FEVD plot saved: {plot_file}")
    return plot_file


def save_irf_data(irf_samples, output_path, horizon=40):
    """
    Save IRF data to CSV.
    """
    k = irf_samples.shape[2]
    
    irf_mean = np.mean(irf_samples, axis=0)
    irf_68_lower = np.percentile(irf_samples, 16, axis=0)
    irf_68_upper = np.percentile(irf_samples, 84, axis=0)
    irf_95_lower = np.percentile(irf_samples, 2.5, axis=0)
    irf_95_upper = np.percentile(irf_samples, 97.5, axis=0)
    
    data = {'horizon': np.arange(horizon)}
    
    for var_idx in range(k):
        var_name = f'var{var_idx + 1}'
        data[f'{var_name}_mean'] = irf_mean[:, var_idx]
        data[f'{var_name}_68lower'] = irf_68_lower[:, var_idx]
        data[f'{var_name}_68upper'] = irf_68_upper[:, var_idx]
        data[f'{var_name}_95lower'] = irf_95_lower[:, var_idx]
        data[f'{var_name}_95upper'] = irf_95_upper[:, var_idx]
    
    df = pd.DataFrame(data)
    csv_file = output_path.replace('.csv', '_irf.csv')
    df.to_csv(csv_file, index=False)
    print(f"  IRF data saved: {csv_file}")


def save_fevd_data(fevd_mean, fevd_lower, fevd_upper, output_path, horizon=40):
    """
    Save FEVD data to CSV.
    """
    k = fevd_mean.shape[1]
    
    data = {'horizon': np.arange(horizon)}
    
    for var_idx in range(k):
        var_name = f'var{var_idx + 1}'
        data[f'{var_name}_mean'] = fevd_mean[:, var_idx]
        data[f'{var_name}_68lower'] = fevd_lower[:, var_idx]
        data[f'{var_name}_68upper'] = fevd_upper[:, var_idx]
    
    df = pd.DataFrame(data)
    csv_file = output_path.replace('.csv', '_fevd.csv')
    df.to_csv(csv_file, index=False)
    print(f"  FEVD data saved: {csv_file}")

# =============================================================================
# DIAGNOSTIC PLOTS
# =============================================================================

def create_diagnostic_plots(trace, forecasts, output_path):
    """
    Create diagnostic plots for MCMC convergence and forecast quality.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import arviz as az
    
    plot_file = output_path.replace('.csv', '_diagnostics.png')
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('BVAR Diagnostics', fontsize=14, fontweight='bold')
    
    beta_samples = trace.posterior['beta'].values
    
    for i in range(min(3, beta_samples.shape[-1])):
        axes[0, 0].plot(beta_samples[0, :, i], alpha=0.7, label=f'β_{i+1}')
    axes[0, 0].set_title('Trace Plot (First 3 Coeffs)')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].legend(fontsize=8)
    
    axes[0, 1].hist(beta_samples[0, :, 1].flatten(), bins=30, alpha=0.7, color='steelblue')
    axes[0, 1].set_title('Posterior: β_1 (First Own Lag)')
    axes[0, 1].set_xlabel('Value')
    
    try:
        rhat_vals = az.rhat(trace)
        if 'beta' in rhat_vals:
            rhat_beta = rhat_vals['beta'].values.flatten()[:10]
            axes[0, 2].bar(range(len(rhat_beta)), rhat_beta, color='steelblue')
            axes[0, 2].axhline(y=1.1, color='r', linestyle='--', label='R-hat = 1.1')
            axes[0, 2].set_title('R-hat (First 10 Params)')
            axes[0, 2].set_xlabel('Parameter Index')
            axes[0, 2].set_ylabel('R-hat')
            axes[0, 2].set_ylim(0.99, 1.15)
            axes[0, 2].legend(fontsize=8)
        else:
            axes[0, 2].text(0.5, 0.5, 'R-hat: beta not found', ha='center', va='center', transform=axes[0, 2].transAxes)
    except Exception as e:
        axes[0, 2].text(0.5, 0.5, f'R-hat error:\n{str(e)[:30]}', ha='center', va='center', transform=axes[0, 2].transAxes)
    
    n_show = min(200, len(forecasts))
    
    for var_idx in range(3):
        ax = axes[1, var_idx]
        var_name = f'var{var_idx + 1}'
        
        ax.plot(forecasts[f'{var_name}_actual'].values[:n_show], 'b-', 
                label='Actual', alpha=0.7, linewidth=1.5)
        ax.plot(forecasts[f'{var_name}_pred'].values[:n_show], 'orange', 
                label='Predicted', alpha=0.7, linewidth=1.5)
        
        ax.fill_between(range(n_show),
                       forecasts[f'{var_name}_q2.5'].values[:n_show],
                       forecasts[f'{var_name}_q97.5'].values[:n_show],
                       alpha=0.2, color='blue', label='95% PI')
        
        ax.set_title(f'{var_name}: Forecast with 95% PI')
        ax.set_xlabel('Time Step')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    for var_idx in range(3):
        ax = axes[2, var_idx]
        var_name = f'var{var_idx + 1}'
        
        errors = forecasts[f'{var_name}_actual'].values - forecasts[f'{var_name}_pred'].values
        colors = ['steelblue', 'coral', 'seagreen']
        ax.hist(errors, bins=50, alpha=0.7, color=colors[var_idx], edgecolor='black')
        ax.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero Error')
        ax.axvline(errors.mean(), color='orange', linestyle='--', 
                  linewidth=2, label=f'Mean: {errors.mean():.3f}')
        ax.set_title(f'{var_name}: Error Distribution')
        ax.set_xlabel('Error (Actual - Predicted)')
        ax.set_ylabel('Frequency')
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Diagnostics saved: {plot_file}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='BVAR Model - Pure Prediction Module')
    
    parser.add_argument('--input', type=str, required=True,
                        help='Input CSV file (6250 observations, 3 assets)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output CSV file (predictions + quantiles)')
    parser.add_argument('--lags', type=int, required=True,
                        help='Number of lags (e.g., 3, 4, 5, 6, 7, 8, 9)')
    
    parser.add_argument('--plots', action='store_true',
                        help='Generate diagnostic plots')
    parser.add_argument('--irfplus', action='store_true',
                        help='Generate IRF and FEVD analysis (shock to var1)')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    print("=" * 60)
    print("BVAR Model with Minnesota Prior")
    print("=" * 60)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Lags:   {args.lags}")
    print("=" * 60)

    # ========== 1. LOAD DATA ==========
    print("\n[1/6] Loading data...")
    df = pd.read_csv(args.input)
    
    if 'time' in df.columns:
        df = df.drop('time', axis=1)
    
    data_values = df.values[:N_TRAINING + N_FORECAST]
    print(f"  Data shape: {data_values.shape}")
    
    # ========== 2. TRAIN/TEST SPLIT ==========
    train_raw = data_values[:N_TRAINING]
    test_actual = data_values[N_TRAINING:N_TRAINING + N_FORECAST]
    
    print(f"  Train: {train_raw.shape}, Test: {test_actual.shape}")
    
    # ========== 3. STATIONARITY CHECK ==========
    print("\n[2/6] Checking stationarity...")
    train_data, data_tag, diff_info = check_and_transform_stationarity(train_raw)
    print(f"  Data tag: {data_tag}")
   
    # ========== 4. FIT MODEL ==========
    print("\n[3/6] Calculating Minnesota prior...")
    prior_mean, prior_cov = calculate_minnesota_prior(train_data, args.lags)
    
    print("\n[4/6] Fitting BVAR model...")
    model, trace, Y, X = fit_bvar_model(train_data, args.lags, prior_mean, prior_cov)
   
    # ========== 5. CALCULATE DIC ==========
    print("\n[5/6] Calculating DIC...")
    dic_value = calculate_dic(trace, Y, X, args.lags)
    print(f"  DIC: {dic_value:.2f}")
    
    # ========== 6. GENERATE FORECASTS ==========
    print("\n[6/6] Generating forecasts...")
    forecasts = generate_forecasts(trace, train_data, test_actual, args.lags, 
                                   data_tag, diff_info)
    
    # ========== SAVE OUTPUT ==========
    runtime = time.time() - start_time
    
    print(f"\nSaving results to {args.output}...")
    print(f"Total runtime: {runtime:.2f} seconds ({runtime/60:.2f} minutes)")
    
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    
    with open(args.output, 'w') as f:
        f.write(f"# runtime: {runtime:.2f}\n")
        f.write(f"# DIC: {dic_value:.2f}\n")
        f.write(f"# data_tag: {data_tag}\n")
        forecasts.to_csv(f, index=False)
    
    print("Done!")
    
    # ========== DIAGNOSTIC PLOTS ==========
    if args.plots:
        print("\nGenerating diagnostic plots...")
        create_diagnostic_plots(trace, forecasts, args.output)
    
    # ========== IRF + FEVD ==========
    if args.irfplus:
        print("\n" + "=" * 60)
        print("IRF + FEVD Analysis (Shock to Variable 1)")
        print("=" * 60)
        
        print("\nComputing IRF...")
        irf_samples = compute_irf(trace, args.lags, horizon=40, shock_var=0)
        
        create_irf_plot(irf_samples, args.output, horizon=40)
        save_irf_data(irf_samples, args.output, horizon=40)
        
        print("\nComputing FEVD...")
        fevd_mean, fevd_lower, fevd_upper = compute_fevd(irf_samples, horizon=40)
        
        create_fevd_plot(fevd_mean, fevd_lower, fevd_upper, args.output, horizon=40)
        save_fevd_data(fevd_mean, fevd_lower, fevd_upper, args.output, horizon=40)
        
        print("\nIRF + FEVD complete!")
    
    # ========== TUI SUMMARY ==========
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Runtime:    {runtime:.2f}s ({runtime/60:.2f}min)")
    print(f"  DIC:        {dic_value:.2f}")
    print(f"  Data tag:   {data_tag}")
    print(f"  Lags:       {args.lags}")
    print(f"  Forecasts:  {len(forecasts)} observations")
    print("=" * 60)


if __name__ == '__main__':
    main()
