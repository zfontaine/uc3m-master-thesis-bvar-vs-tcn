#!/usr/bin/env python3
# This file is a part of a Master Thesis titled:
# "Bayesian VAR vs. Temporal Convolutional Networks:
# A Simulation Study on Forecasting Accuracy and
# Uncertainty Calibration" produced by the author
# at Universidad Carlos III. de Madrid (UC3M) in 2026.
# All rights reserved.
#
# FILENAME: tcn_model.py
# AUTHOR: Zeno S. alias Zeno Fontaine
# DATE: 2026-02-06
# 
# INPUT: CSV file with time series data (6250 observations, 3 variables)
#        Command-line arguments: --input, --output, --layers, --channels, --dropout, --lookback
#        Optional: --shockpropplus, --plots
#
# OUTPUT: CSV with predictions and quantiles for test set (5000-6249)
#         Optional: shock propagation plots/CSVs, uncertainty analysis plots/CSVs
#         Header comments in CSV: runtime
#
# FILE DESCRIPTION: [tcn, deep-learning, mc-dropout, forecasting, time-series]
# Temporal Convolutional Network with MC Dropout for multi-asset forecasting.
# Implements dilated causal convolutions with Bayesian uncertainty quantification.
# Includes shock propagation analysis (TCN analog to IRF) and uncertainty regime
# detection for market event analysis.
#
# Configurable parameters:
#   - layers, channels, dropout: TCN architecture hyperparameters
#   - lookback: sequence length for input windows
#   - epochs, batch_size, lr, patience: training parameters
#   - mc_samples: number of MC Dropout samples for uncertainty
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
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# TCN ARCHITECTURE
# =============================================================================

class TCNBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size,
            padding=padding,
            dilation=dilation
        )
        
        self.chomp = nn.ConstantPad1d((0, -padding), 0)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = self.conv(x)
        # CAUSAL: Remove future timesteps
        out = self.chomp(out[:, :, :x.size(2)])
        out = self.relu(out)
        out = self.dropout(out)
        return out


class TCNBayesian(nn.Module):
    """
    TCN with MC Dropout for Bayesian uncertainty quantification.
    
    Architecture:
        - Multiple dilated convolution layers (dilation = 2^i)
        - Dropout after each layer (stays active during inference)
        - Final linear layer for multi-output prediction
    """
    
    def __init__(self, input_size, output_size, num_layers, num_channels, 
                 kernel_size=3, dropout=0.2):
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            in_ch = input_size if i == 0 else num_channels
            out_ch = num_channels
            
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels, output_size)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, features)
        Returns:
            (batch, output_size)
        """
        x = x.transpose(1, 2)
        
        y = self.network(x)
        
        y = y[:, :, -1]
        
        y = self.linear(y)
        
        return y

# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_sequences(data, lookback=50):
    """
    Create sliding window sequences for training.
    
    Returns:
        X: (n_samples, lookback, n_assets)
        y: (n_samples, n_assets)
    """
    X, y = [], []
    
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i])
    
    return np.array(X), np.array(y)

# =============================================================================
# TRAINING
# =============================================================================

def train_tcn(model, train_loader, epochs=50, lr=0.001, patience=10, device='cuda'):
    """
    Train TCN with early stopping.
    
    Returns:
        trained model, loss_history (list of epoch losses)
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    patience_counter = 0
    loss_history = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}/{epochs}")
            break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    return model, loss_history

# =============================================================================
# MC DROPOUT INFERENCE
# =============================================================================

def mc_dropout_predict(model, X_test, n_samples=100, device='cuda'):
    """
    MC Dropout inference with uncertainty quantification.
    
    Returns:
        predictions: (n_test, n_assets, n_samples)
    """
    # CRITICAL: Keep dropout active for MC sampling!
    model.train()
    
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    predictions = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            y_pred = model(X_test_tensor).cpu().numpy()
            predictions.append(y_pred)
    
    predictions = np.array(predictions).transpose(1, 2, 0)
    
    return predictions


def compute_quantiles(predictions, quantiles=[2.5, 5, 10, 20, 50, 80, 90, 95, 97.5]):
    """
    Compute quantiles from MC samples.
    
    Returns:
        quantiles_dict: {q: (n_test, n_assets)}
    """
    quantiles_dict = {}
    
    for q in quantiles:
        quantiles_dict[f'q{q}'] = np.percentile(predictions, q, axis=2)
    
    return quantiles_dict

# =============================================================================
# SHOCK PROPAGATION ANALYSIS
#
# TCN analog to BVAR Impulse Response Function.
# Inject Cholesky-identified shock and track propagation with MC Dropout uncertainty.
# Matches BVAR methodology: 1 std dev orthogonalized shock, 40-step horizon.
# =============================================================================

def shock_propagation_analysis(model, X_test, y_test, test_prices,
                                shock_var=0, horizon=40, mc_samples=100,
                                output_prefix='', device='cuda'):
    """
    Inject synthetic shock and track propagation with MC Dropout uncertainty.
    
    Output Files:
        - {output_prefix}_shockprop.csv: Shock response data
        - {output_prefix}_shockprop.png: IRF Visualization
        - {output_prefix}_fevd.csv: FEVD data
        - {output_prefix}_fevd.png: FEVD Visualization
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    returns = np.diff(test_prices, axis=0) / test_prices[:-1]
    cov_matrix = np.cov(returns.T)
    
    try:
        L = np.linalg.cholesky(cov_matrix)
    except np.linalg.LinAlgError:
        # FIX: Add small diagonal if not PD
        cov_matrix += 1e-6 * np.eye(3)
        L = np.linalg.cholesky(cov_matrix)
    
    cholesky_shock = L[:, shock_var]
    
    print(f"  Injecting 1 std dev Cholesky shock to var{shock_var+1}...")
    print(f"    Shock magnitudes: var1={cholesky_shock[0]:.4f}, var2={cholesky_shock[1]:.4f}, var3={cholesky_shock[2]:.4f}")
    
    baseline_idx = len(X_test) // 2
    baseline_seq = X_test[baseline_idx:baseline_idx+1].copy()
    baseline_vals = baseline_seq[0, -1, :].copy()
    
    shocked_seq = baseline_seq.copy()
    for i in range(3):
        shocked_seq[0, -1, i] += cholesky_shock[i]
    
    shocked_vals = shocked_seq[0, -1, :].copy()
    
    all_responses = {f'var{i+1}': [] for i in range(3)}
    
    responses = {
        f'var{i+1}': {'mean': [], 'q2.5': [], 'q16': [], 'q84': [], 'q97.5': []} 
        for i in range(3)
    }
    
    model.train()
    
    baseline_seq_copy = baseline_seq.copy()
    
    for h in range(horizon):
        mc_preds_shocked = []
        shocked_tensor = torch.FloatTensor(shocked_seq).to(device)
        
        with torch.no_grad():
            for _ in range(mc_samples):
                pred = model(shocked_tensor).cpu().numpy()
                mc_preds_shocked.append(pred)
        
        mc_preds_shocked = np.array(mc_preds_shocked)
        
        mc_preds_baseline = []
        baseline_tensor = torch.FloatTensor(baseline_seq_copy).to(device)
        
        with torch.no_grad():
            for _ in range(mc_samples):
                pred = model(baseline_tensor).cpu().numpy()
                mc_preds_baseline.append(pred)
        
        mc_preds_baseline = np.array(mc_preds_baseline)
        
        # RESPONSE: Shocked minus baseline prediction
        for i in range(3):
            response_samples = mc_preds_shocked[:, 0, i] - mc_preds_baseline[:, 0, i]
            all_responses[f'var{i+1}'].append(response_samples)
            
            responses[f'var{i+1}']['mean'].append(response_samples.mean())
            responses[f'var{i+1}']['q2.5'].append(np.percentile(response_samples, 2.5))
            responses[f'var{i+1}']['q16'].append(np.percentile(response_samples, 16))
            responses[f'var{i+1}']['q84'].append(np.percentile(response_samples, 84))
            responses[f'var{i+1}']['q97.5'].append(np.percentile(response_samples, 97.5))
        
        shocked_seq = np.roll(shocked_seq, -1, axis=1)
        shocked_seq[0, -1, :] = mc_preds_shocked.mean(axis=0)[0, :]
        
        baseline_seq_copy = np.roll(baseline_seq_copy, -1, axis=1)
        baseline_seq_copy[0, -1, :] = mc_preds_baseline.mean(axis=0)[0, :]
    
    for i in range(3):
        all_responses[f'var{i+1}'] = np.array(all_responses[f'var{i+1}'])
    
    fevd_data = {'horizon': list(range(horizon))}
    
    for h in range(horizon):
        total_variance = 0
        variances = []
        
        for i in range(3):
            var_h = all_responses[f'var{i+1}'][h, :].var()
            variances.append(var_h)
            total_variance += var_h
        
        for i in range(3):
            if total_variance > 0:
                prop = variances[i] / total_variance
            else:
                prop = 1/3
            
            if h == 0:
                fevd_data[f'var{i+1}_prop'] = [prop]
            else:
                fevd_data[f'var{i+1}_prop'].append(prop)
    
    csv_data = {'horizon': list(range(horizon))}
    
    for i in range(3):
        var_name = f'var{i+1}'
        csv_data[f'{var_name}_mean'] = responses[var_name]['mean']
        csv_data[f'{var_name}_q2.5'] = responses[var_name]['q2.5']
        csv_data[f'{var_name}_q16'] = responses[var_name]['q16']
        csv_data[f'{var_name}_q84'] = responses[var_name]['q84']
        csv_data[f'{var_name}_q97.5'] = responses[var_name]['q97.5']
    
    csv_file = f"{output_prefix}_shockprop.csv"
    pd.DataFrame(csv_data).to_csv(csv_file, index=False)
    print(f"  Saved: {csv_file}")
    
    fevd_csv_file = f"{output_prefix}_fevd.csv"
    pd.DataFrame(fevd_data).to_csv(fevd_csv_file, index=False)
    print(f"  Saved: {fevd_csv_file}")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    fig.suptitle(f'Shock Propagation Plot: Shock to Variable {shock_var+1}', 
                 fontsize=14, fontweight='bold')
    
    colors = ['steelblue', 'coral', 'seagreen']
    horizons = list(range(horizon))
    
    for i in range(3):
        ax = axes[i]
        var_name = f'var{i+1}'
        
        ax.plot(horizons, responses[var_name]['mean'], 
                color=colors[i], linewidth=2, label='Mean')
        
        ax.fill_between(horizons,
                        responses[var_name]['q16'],
                        responses[var_name]['q84'],
                        color=colors[i], alpha=0.4, label='68% CI')
        
        ax.fill_between(horizons,
                        responses[var_name]['q2.5'],
                        responses[var_name]['q97.5'],
                        color=colors[i], alpha=0.2, label='95% CI')
        
        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        ax.axvline(0, color='red', linestyle='--', linewidth=2, 
                   label=r'Shock (1$\sigma$)')
        
        ax.set_title(f'Response of {var_name}', fontweight='bold')
        ax.set_xlabel('Horizon (steps)')
        ax.set_ylabel('Response')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    png_file = f"{output_prefix}_shockprop.png"
    plt.savefig(png_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {png_file}")
    
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    
    fig2.suptitle(f'Forecast Error Variance Decomposition: Contribution of var{shock_var+1} Shock', 
                  fontsize=14, fontweight='bold')
    
    for i in range(3):
        ax = axes2[i]
        var_name = f'var{i+1}'
        
        ax.plot(horizons, fevd_data[f'{var_name}_prop'], 
                color=colors[i], linewidth=2, label='Mean')
        
        ax.fill_between(horizons, 0, fevd_data[f'{var_name}_prop'],
                        color=colors[i], alpha=0.3)
        
        ax.set_title(f'FEVD of {var_name}', fontweight='bold')
        ax.set_xlabel('Horizon (steps)')
        ax.set_ylabel('Proportion of Variance')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
    
    plt.tight_layout()
    
    fevd_png_file = f"{output_prefix}_fevd.png"
    plt.savefig(fevd_png_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fevd_png_file}")

# =============================================================================
# UNCERTAINTY REGIME ANALYSIS
# =============================================================================

def uncertainty_regime_analysis(predictions, y_test, test_prices, 
                                 output_prefix='', lookback=50):
    """
    Analyze prediction uncertainty over time and detect market regimes.
    
    Output Files:
        - {output_prefix}_uncertainty.csv: Uncertainty time series
        - {output_prefix}_uncertainty_predictions.png: Predictions plot
        - {output_prefix}_uncertainty_regimes.png: Uncertainty over time plot
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    print(f"  Computing uncertainty metrics...")
    
    pred_std = predictions.std(axis=2)
    pred_mean = predictions.mean(axis=2)
    
    returns = np.diff(test_prices, axis=0) / test_prices[:-1]
    returns = np.vstack([np.zeros((1, 3)), returns])
    
    jump_threshold = 0.01
    jumps = np.abs(returns) > jump_threshold
    jump_times = np.where(jumps.any(axis=1))[0]
    
    window = 20
    rolling_vol = pd.DataFrame(returns).rolling(window, min_periods=1).std().values
    vol_threshold = rolling_vol.mean() + 2 * rolling_vol.std()
    vol_spikes = rolling_vol > vol_threshold
    spike_times = np.where(vol_spikes.any(axis=1))[0]
    
    csv_data = {'time': np.arange(5000, 6250)}
    
    for i in range(3):
        var_name = f'var{i+1}'
        csv_data[f'{var_name}_actual'] = y_test[:, i]
        csv_data[f'{var_name}_pred_mean'] = pred_mean[:, i]
        csv_data[f'{var_name}_pred_std'] = pred_std[:, i]
    
    csv_file = f"{output_prefix}_uncertainty.csv"
    pd.DataFrame(csv_data).to_csv(csv_file, index=False)
    print(f"  Saved: {csv_file}")
    
    fig1, axes1 = plt.subplots(1, 3, figsize=(15, 4))
    fig1.suptitle('Predictions with Uncertainty Bands', fontsize=14, fontweight='bold')
    
    colors = ['steelblue', 'coral', 'seagreen']
    n_show = min(250, len(y_test))
    
    for i in range(3):
        ax = axes1[i]
        var_name = f'var{i+1}'
        
        ax.plot(y_test[:n_show, i], label='Actual', 
                color='black', linewidth=1.5, alpha=0.7)
        ax.plot(pred_mean[:n_show, i], label='Predicted', 
                color=colors[i], linewidth=1.5)
        
        q_lower = pred_mean[:n_show, i] - 2 * pred_std[:n_show, i]
        q_upper = pred_mean[:n_show, i] + 2 * pred_std[:n_show, i]
        ax.fill_between(range(n_show), q_lower, q_upper,
                        color=colors[i], alpha=0.2, label='95% PI')
        
        jump_mask = jumps[:n_show, i]
        if jump_mask.any():
            ax.scatter(np.where(jump_mask)[0], y_test[jump_mask, i],
                      color='red', s=50, marker='x', label='Jumps', zorder=5)
        
        ax.set_title(f'Response of {var_name}', fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Price')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    png_file1 = f"{output_prefix}_uncertainty_predictions.png"
    plt.savefig(png_file1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {png_file1}")
    
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    fig2.suptitle('Uncertainty Regime Detection', fontsize=14, fontweight='bold')
    
    for i in range(3):
        ax = axes2[i]
        var_name = f'var{i+1}'
        
        ax.plot(pred_std[:n_show, i], color=colors[i], linewidth=2, 
                label='Prediction Std Dev')
        
        for jt in jump_times:
            if jt < n_show:
                ax.axvline(jt, color='red', alpha=0.3, linestyle='--', linewidth=1)
        
        for st in spike_times:
            if st < n_show:
                ax.axvline(st, color='orange', alpha=0.3, linestyle='--', linewidth=1)
        
        if len(jump_times) > 0:
            ax.axvline(-1, color='red', alpha=0.5, linestyle='--', 
                      linewidth=1, label='Jump Events')
        if len(spike_times) > 0:
            ax.axvline(-1, color='orange', alpha=0.5, linestyle='--', 
                      linewidth=1, label='Vol Spikes')
        
        ax.set_title(f'Uncertainty: {var_name}', fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Std Dev')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    png_file2 = f"{output_prefix}_uncertainty_regimes.png"
    plt.savefig(png_file2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {png_file2}")
    
    print(f"\n  Summary Statistics:")
    for i in range(3):
        mean_unc = pred_std[:, i].mean()
        jump_unc = pred_std[jumps[:, i], i].mean() if jumps[:, i].any() else 0
        spike_unc = pred_std[vol_spikes[:, i], i].mean() if vol_spikes[:, i].any() else 0
        
        print(f"    var{i+1}: Mean={mean_unc:.4f}, Jump={jump_unc:.4f}, Spike={spike_unc:.4f}")

def set_seed(seed=666):
    """Set all random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='TCN Model - Pure Prediction Module')
    
    parser.add_argument('--input', type=str, required=True,
                        help='Input CSV file (6250 observations, 3 assets)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output CSV file (predictions + quantiles)')
    
    parser.add_argument('--layers', type=int, required=True,
                        help='Number of TCN layers (e.g., 2, 3, 4)')
    parser.add_argument('--channels', type=int, required=True,
                        help='Number of channels per layer (e.g., 16, 32, 64)')
    parser.add_argument('--dropout', type=float, required=True,
                        help='Dropout rate (e.g., 0.1, 0.2, 0.3)')
    parser.add_argument('--lookback', type=int, default=50,
                        help='Lookback window size (default: 50)')
    
    parser.add_argument('--epochs', type=int, default=50,
                        help='Maximum epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (default: 10)')
    parser.add_argument('--mc_samples', type=int, default=100,
                        help='MC Dropout samples (default: 100)')
    
    parser.add_argument('--plots', action='store_true',
                        help='Generate diagnostic plots')
    parser.add_argument('--shockpropplus', action='store_true',
                        help='Generate shock propagation + uncertainty regime analysis')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    set_seed(1964)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # ========== 1. LOAD DATA ==========
    print(f"Loading data from {args.input}...")
    df = pd.read_csv(args.input)
    
    price_cols = [col for col in df.columns if col.startswith('asset_')]
    prices = df[price_cols].values
    
    print(f"Data shape: {prices.shape}")
    
    # ========== 2. TRAIN/TEST SPLIT ==========
    train_prices = prices[:5000]
    test_prices = prices[5000:]
    
    print(f"Train: {train_prices.shape}, Test: {test_prices.shape}")
    
    # ========== 3. PREPARE SEQUENCES ==========
    print(f"Creating sequences with lookback={args.lookback}...")
    
    X_train, y_train = prepare_sequences(train_prices, lookback=args.lookback)
    print(f"Train sequences: X={X_train.shape}, y={y_train.shape}")
    
    full_prices = np.vstack([train_prices[-(args.lookback):], test_prices])
    X_test, y_test = prepare_sequences(full_prices, lookback=args.lookback)
    print(f"Test sequences: X={X_test.shape}, y={y_test.shape}")
    
    # ========== 4. CREATE DATALOADERS ==========
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # ========== 5. INITIALIZE MODEL ==========
    print(f"Initializing TCN: layers={args.layers}, channels={args.channels}, dropout={args.dropout}")
    
    model = TCNBayesian(
        input_size=3,
        output_size=3,
        num_layers=args.layers,
        num_channels=args.channels,
        kernel_size=3,
        dropout=args.dropout
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ========== 6. TRAIN MODEL ==========
    print("Training model...")
    model, loss_history = train_tcn(
        model, 
        train_loader, 
        epochs=args.epochs, 
        lr=args.lr, 
        patience=args.patience,
        device=device
    )
    
    # ========== 7. MC DROPOUT INFERENCE ==========
    print(f"Generating predictions with {args.mc_samples} MC samples...")
    predictions = mc_dropout_predict(model, X_test, n_samples=args.mc_samples, device=device)
    
    print(f"Predictions shape: {predictions.shape}")
    
    # ========== 8. COMPUTE QUANTILES ==========
    print("Computing quantiles...")
    
    pred_mean = predictions.mean(axis=2)
    
    quantiles = compute_quantiles(predictions)
    
    # ========== 9. PREPARE OUTPUT ==========
    print("Preparing output CSV...")
    
    time_indices = np.arange(5000, 6250)
    
    output_data = {'time': time_indices}
    
    for i, asset in enumerate(price_cols):
        var_name = f'var{i+1}'
        
        output_data[f'{var_name}_actual'] = y_test[:, i]
        
        output_data[f'{var_name}_pred'] = pred_mean[:, i]
        
        for q_name, q_values in quantiles.items():
            output_data[f'{var_name}_{q_name}'] = q_values[:, i]
    
    output_df = pd.DataFrame(output_data)
    
    # ========== 10. SAVE OUTPUT ==========
    runtime = time.time() - start_time
    
    print(f"Saving results to {args.output}...")
    print(f"Total runtime: {runtime:.2f} seconds ({runtime/60:.2f} minutes)")
    
    with open(args.output, 'w') as f:
        f.write(f"# runtime: {runtime:.2f}\n")
        output_df.to_csv(f, index=False)
    
    print("Done!")
    
    # ========== 11. SHOCK PROPAGATION + UNCERTAINTY ANALYSIS ==========
    if args.shockpropplus:
        print("\n" + "="*60)
        print("GENERATING SHOCK PROPAGATION + UNCERTAINTY ANALYSIS")
        print("="*60 + "\n")
        
        output_prefix = args.output.replace('.csv', '')
        
        print("1/2: Shock Propagation Analysis...")
        shock_propagation_analysis(
            model=model,
            X_test=X_test,
            y_test=y_test,
            test_prices=test_prices,
            shock_var=0,
            horizon=40,
            mc_samples=args.mc_samples,
            output_prefix=output_prefix,
            device=device
        )
        print()
        
        print("2/2: Uncertainty Regime Detection...")
        uncertainty_regime_analysis(
            predictions=predictions,
            y_test=y_test,
            test_prices=test_prices,
            output_prefix=output_prefix,
            lookback=args.lookback
        )
        print()
        
        print("="*60)
        print("SHOCK PROPAGATION + UNCERTAINTY ANALYSIS COMPLETE")
        print("="*60 + "\n")
    
    # ========== 12. DIAGNOSTIC PLOTS ==========
    if args.plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        print("Generating diagnostic plots...")
        
        plot_file = args.output.replace('.csv', '_diagnostics.png')
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle(f'TCN Diagnostics: L={args.layers} C={args.channels} D={args.dropout}', 
                     fontsize=14, fontweight='bold')
        
        axes[0, 0].plot(loss_history, linewidth=2, color='steelblue')
        axes[0, 0].set_title('Training Loss Curve', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('MSE Loss')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(loss_history[-1], color='r', linestyle='--', 
                          linewidth=1, label=f'Final: {loss_history[-1]:.2f}')
        axes[0, 0].legend(loc='best')
        
        last_losses = loss_history[-10:] if len(loss_history) >= 10 else loss_history
        axes[0, 1].hist(last_losses, bins=min(10, len(last_losses)), 
                       edgecolor='black', alpha=0.7, color='steelblue')
        axes[0, 1].set_title('Final Loss Distribution', fontweight='bold')
        axes[0, 1].set_xlabel('MSE Loss')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(loss_history[-1], color='r', linestyle='--', 
                          linewidth=2, label=f'Final: {loss_history[-1]:.2f}')
        axes[0, 1].legend(loc='best')
        axes[0, 1].grid(True, alpha=0.3)
        
        coverage_stats = []
        for i in range(3):
            in_interval = (y_test[:, i] >= quantiles['q2.5'][:, i]) & \
                         (y_test[:, i] <= quantiles['q97.5'][:, i])
            coverage = in_interval.mean() * 100
            coverage_stats.append(coverage)
        
        axes[0, 2].bar(['var1', 'var2', 'var3'], coverage_stats, 
                      color=['steelblue', 'coral', 'seagreen'], alpha=0.7)
        axes[0, 2].axhline(95, color='r', linestyle='--', linewidth=2, label='Target: 95%')
        axes[0, 2].set_title('95% Prediction Interval Coverage', fontweight='bold')
        axes[0, 2].set_ylabel('Coverage (%)')
        axes[0, 2].set_ylim(80, 100)
        axes[0, 2].legend(loc='best')
        axes[0, 2].grid(True, alpha=0.3, axis='y')
        
        for i, color in enumerate(['blue', 'orange', 'green']):
            ax = axes[1, i]
            ax.scatter(y_test[:, i], pred_mean[:, i], alpha=0.3, s=10, color=color)
            min_val = min(y_test[:, i].min(), pred_mean[:, i].min())
            max_val = max(y_test[:, i].max(), pred_mean[:, i].max())
            ax.plot([min_val, max_val], [min_val, max_val], 
                   'r--', lw=2, label='Perfect Prediction')
            ax.set_title(f'Asset {i+1}: Actual vs Predicted', fontweight='bold')
            ax.set_xlabel('Actual Price')
            ax.set_ylabel('Predicted Price')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        for i, color in enumerate(['steelblue', 'coral', 'lightgreen']):
            ax = axes[2, i]
            errors = y_test[:, i] - pred_mean[:, i]
            ax.hist(errors, bins=50, edgecolor='black', alpha=0.7, color=color)
            ax.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero Error')
            ax.axvline(errors.mean(), color='orange', linestyle='--', 
                      linewidth=2, label=f'Mean: {errors.mean():.3f}')
            ax.set_title(f'Asset {i+1}: Error Distribution', fontweight='bold')
            ax.set_xlabel('Error (Actual - Predicted)')
            ax.set_ylabel('Frequency')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Diagnostic plots saved to: {plot_file}")
        
        for i in range(3):
            errors = y_test[:, i] - pred_mean[:, i]
            in_interval = (y_test[:, i] >= quantiles['q2.5'][:, i]) & \
                         (y_test[:, i] <= quantiles['q97.5'][:, i])
            coverage = in_interval.mean() * 100
            uncertainty_std = predictions[:, i, :].std(axis=1).mean()
            
            print(f"  Asset {i+1}: Coverage={coverage:.1f}%, Mean Error={errors.mean():.4f}, Uncertainty={uncertainty_std:.4f}")

if __name__ == '__main__':
    main()
