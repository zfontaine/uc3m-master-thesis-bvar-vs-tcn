> **Note:** this is a mirror. The original repository for this thesis is at
> <https://gitlab.com/zfontaine/uc3m-master-thesis-bvar-vs-tcn>

---
*last edit: 2026-10-17*

*Author's Alias: Zeno Fontaine*

*Thesis Supervisors: Juan Miguel Marín Diazaraque and María Helena Lopes Moreira Da Veiga*

**UPDATE: 2026-10-17**

_On 27th February 2026, the thesis associated with this repository received a provisional
grade of **10/10** from its defense committee and was nominated for Honours
(*Matrícula de Honor*). Spanish university rules limit Honours to at most 5% of
students, which for this degree meant a single award. For this reason, all nominated theses were
reviewed together by the faculty before grades were finalised. The review concluded on
9th October 2026: the grade of **10/10** was confirmed as final, and Honours were not
awarded to this thesis._

# UC3M Master Thesis: Bayesian VAR vs. Temporal Convolutional Networks: A Simulation Study on Forecasting Accuracy and Uncertainty Calibration

## THESIS ABSTRACT

This work aims to assess whether traditional Bayesian Vector Autoregression(BVAR)
methods remain competitive, in terms of forecasting and uncertainty quantification, given
recent advancements in deep learning algorithms for sequential data. To above question
is explored via a simulation study that compares two statistical frameworks in their most
common architectures: BVAR with Minnesota prior against Temporal Convolutional Net-
works (TCN) with Monte Carlo Dropout. The simulation study generates 315 datasets,
each containing 6,250 observations across three variables, based on a stochastic model
that aims to simulate minute-level trading data with three regimes that are modelled using
a proposed Excitement parameter. The design combines the following elements: Stochas-
tic Volatility via Ornstein-Uhlenbeck process, Jump-Diffusion, Cholesky-based Correla-
tion Structure, and Regime-Dependent Correlations via Excitement parameter. The study
utilised a two-stage design: the first stage found the best-performing architecture of each
model family across 15 datasets, the second stage compared the two best-perforimng ar-
chitectures across 300 datasets.

The results suggest that, despite being the older framework, BVAR dominated in ac-
curacy (27-39% better RMSE, 38-53% better MAE), uncertainty calibration, and inter-
val width (TCN intervals approximately 2.7 times wider). BVAR achieved near-nominal
coverage whilst TCN remained severely overconservative due to MC Dropout being a
Bayesian approximation method. Despite raw performance, the results are more nuanced.
While BVAR remains competitive under suitable conditions and in cases where calibrated
uncertainty is non-negotiable, TCN was approximately 9 times faster which makes it more
suitable when computational speed and deployment over much larger datasets without
strong structural assumptions takes priority.

**Keywords:**  Bayesian VAR, Simulation Study, Minnesota Prior, Temporal Convolu-
tional Networks, Monte Carlo Dropout, High-Frequency Forecasting, Forecasting Accu-
racy, Uncertainty Callibration

## Description

**PLEASE NOTE: MY ACTUAL WORKSPACE CAN BE SEEN IN "MY_PROJECTSPACE." IT
IS HIGHLY UNLIKELY THAT THAT FOLDER CAN BE AUTOMATICALLY RUN ON ANY OTHER
SYSTEM. FOR REPLICABILITY, THIS REPOSITORY ALSO INCLUDES GUIDES AND WAYS
TO SUCCESSFULLY REDO THE SIMULATION.**

There are four steps that need to be carried out to replicate the
simulation study:

1. Generate simulated data using code in **data_generation** folder,
where 15 datasets are created using *42* seed and 300 using *666* seed.
2. Carry out EDA on the datasets using contents of the **EDA** and
**metrics_calculation** folders.
3. Write **Bash pipelines** or alternatives that correspond to your
operating system. A general guide can be found in the *bash_pipeline_showcase*
folder.
4. Use bash pipelines to carry out **Pretraining** and **Forecasting Test**
stages using corresponding *model* codes.
Calculate metrics using code in **metrics_calculation** and **process_metrics.py**.

This is the author's official repository for the master's thesis. The thesis PDF is
a redacted version from which personally identifiable details have been removed,
including the author's legal name on the front cover and dedication page. The thesis
and code were produced under the author's established publishing alias,
Zeno Fontaine, which appears in the original document and is retained here.
The author holds full copyright to all materials in this repository.

The code is licensed under the GNU GPLv3 licence, while the published work,
master thesis, and all graphics associated with it are licensed under Creative Commons Attribution-NonCommercial-NoDerivatives 4.0
International (CC BY-NC-ND 4.0) (https://creativecommons.org/licenses/by-nc-nd/4.0/). 

If you use this work in your research, please cite:

```
@mastersthesis{fontaine2026bayesian,
  author  = {Fontaine, Zeno and Marín Diazaraque, Juan Miguel and Lopes Moreira Da Veiga, María Helena},
  title   = {Bayesian VAR vs. Temporal Convolutional Networks: A Simulation Study on Forecasting Accuracy and Uncertainty Calibration},
  school  = {Universidad Carlos III de Madrid},
  year    = {2026},
  type    = {Master's thesis},
  URL     = LINK TO THIS REPOSITORY
}
```


## Bash Pipelines Showcase Directory

The directory contains a collection of bash files
that were used for orchestration, parallelisation and pipes.
These files can be found in **my_projectspace** directory
which can be used as inspiration for writing bash files for
replicating the simulation study. The files are:

1. **bvar_pretrain_simpleparallel** (basic parallelisation which runs two models and must be set manually)
2. **orchestration_final_bvar** (final parallelisation script for BVAR)
3. **orchestration_final_tcn** (final parallelisation script for TCN)
4. **pipe-bvar** (pipe used by BVAR to run bvar_model.py on multiple models)
5. **pipe-tcn** (pipe used by TCN to run tcn_model.py  on multiple models)
6. **processor_eda** (orchestration script for running EDA in current directory)
7. **tcn_pretrain_gnuparallel**  (TCN pretraining pipeline using gnuparallel)


##  Core Files

### bvar_model.py

Bayesian VAR implementation with
Minnesota prior for multivariate dataset

#### Features

- Minnesota prior with hyperparameters (λ, τ, δ)
- Automatic stationarity testing and differencing (ADF test)
- MCMC inference via PyMC (NUTS sampler)
- Vectorised multi-step ahead forecasting for faster computation
- Uncertainty quantification (9 quantile levels)
- DIC calculation for model comparison
- Optional diagnostics: IRF, FEVD, and convergence plots

#### Usage

```bash
python bvar_model.py \
    --input data/excitement_0.50/d1.csv \
    --output results/excitement_0.50_d1.csv \
    --lags 6 \
    --plots \
    --irfplus
```

#### Arguments

- `--input`: CSV file 
- `--output`: Output CSV path
- `--lags`: Number of lags 
- `--plots`: Generate diagnostic plots (optional)
- `--irfplus`: Generate IRF/FEVD analysis (optional)

#### Output

**CSV columns:**
- `time`: Time index
- `varN_actual`: Actual values
- `varN_pred`: Predicted values (posterior mean)
- `varN_q2.5` through `varN_q97.5`: Prediction quantiles

**CSV header:**
- `# runtime`: Execution time (seconds)
- `# DIC`: Deviance Information Criterion
- `# data_tag`: Differencing applied (0=none, 1=first-order, 2=second-order)

**Additional outputs (if `--plots`):**
- `*_diagnostics.png`: Graphical diagnostic plots

**Additional outputs (if `--irfplus`):**
- `*_irf.png`, `*_irf.csv`: Impulse Response Functions
- `*_fevd.png`, `*_fevd.csv`: Forecast Error Variance Decomposition

#### Configuration

Edit constants at top of file:

```python
N_TRAINING = 5000       # Training set size
N_FORECAST = 1250       # Forecast horizon

LAMBDA_PARAM = 0.2      # Overall prior tightness
TAU = 10                # Cross-variable shrinkage
DELTA = 1               # Lag decay rate

MCMC_DRAWS = 2000       # MCMC samples
MCMC_CHAINS = 4         # Parallel chains
```
### tcn_model.py

Temporal Convolutional Network with MC Dropout for multivariate dataset.

#### Features

- Dilated causal convolutions with exponential receptive field growth
- MC Dropout for Bayesian approximation of uncertainty quantification
- Automatic early stopping 
- Multi-step ahead forecasting with quantile prediction
- Shock propagation analysis (TCN analogue to IRF)
- Uncertainty regime detection 
- Optional diagnostics

#### Usage

```bash
python tcn_model.py \
    --input data/excitement_0.50/d1.csv \
    --output results/excitement_0.50_d1_tcn.csv \
    --layers 4 \
    --channels 64 \
    --dropout 0.2 \
    --lookback 50 \
    --plots \
    --shockpropplus
```
#### Arguments

- `--input`: CSV file 
- `--output`: Output CSV path
- `--layers`: Number of TCN layers 
- `--channels`: Channels per layer 
- `--dropout`: Dropout rate for MC sampling 
- `--lookback`: Sequence length for input windows 
- `--epochs`: Maximum training epochs 
- `--batch_size`: Training batch size 
- `--lr`: Learning rate 
- `--patience`: Early stopping patience 
- `--mc_samples`: MC Dropout samples for inference 
- `--plots`: Generate diagnostic plots (optional)
- `--shockpropplus`: Generate shock propagation and uncertainty analysis (optional)

#### Output

**CSV columns:**
- `time`: Time index
- `varN_actual`: Actual values
- `varN_pred`: Predicted values (MC Dropout mean)
- `varN_q2.5` through `varN_q97.5`: Prediction quantiles

**CSV header:**
- `# runtime`: Execution time (seconds)

**Additional outputs (if `--plots`):**
- `*_diagnostics.png`: Training loss, coverage, prediction scatter, error distributions

**Additional outputs (if `--shockpropplus`):**
- `*_shock_prop.png`, `*_shock_prop.csv`: Shock propagation responses
- `*_uncertainty_regimes.png`, `*_uncertainty_regimes.csv`: High uncertainty periods

#### Configuration

Edit constants or pass arguments:
```python
# Architecture
layers = 4              # Number of dilated conv layers
channels = 64           # Channels per layer
dropout = 0.2           # MC Dropout rate
lookback = 50           # Input sequence length

# Training
epochs = 100            # Maximum epochs
batch_size = 64         # Training batch size
lr = 0.001              # Learning rate
patience = 10           # Early stopping patience

# Inference
mc_samples = 100        # MC Dropout samples for uncertainty
```

### data_generation folder
**Contains 3 files:**

1. **`simulation.py`—** 
Core stochastic volatility model with jump diffusion. Generates
multivariate price series using Ornstein–Uhlenbeck volatility process
and correlated jumps. The `excitement` parameter controls regime intensity.

2. **`gen_data_script.py`—** 
Batch data generation pipeline. Creates
multiple datasets across different excitement
levels (0.25, 0.5, 0.75) with reproducible seeding.
Outputs CSV files organised by volatility regime in
`data_folder/excitement_X.XX/`.

3. **`vis_data_script.py`—**
Command-line visualisation tool
for diagnostic plots. Generates 2×2
grid showing price series, return
distributions, rolling volatility,
and correlation heatmaps. Run with `python3 vis_data_script.py <path_to_csv>`.

**Usage:** Run `gen_data_script.py` to create datasets,
then use `vis_data_script.py` to inspect data properties before forecasting experiments.

### metrics_calculation folder

**Contains 3 files in a sequential pipeline:**

1. **compute_metrics.py —**
Standalone metrics calculator for individual forecast CSVs.
Computes 24 evaluation metrics: RMSE, MAE, coverage rates
(80%, 90%, 95% prediction intervals), and interval widths across
all three variables. Outputs comma-separated values to stdout.
Run with python3 compute_metrics.py <path_to_forecast_csv>.
2. **grand_aggregator.py —**
Universal aggregation engine across model folders.
Calls compute_metrics.py via subprocess on all CSVs
in each model's RESULTS/ directory. Aggregates mean
and standard deviation of metrics across datasets,
plus runtime totals and header metadata (DIC, data_tag distribution).
Outputs single CSV with one row per model folder containing
48 metric columns plus metadata.
3. **pretraining_selector.py —**
Model selection summary generator for pretraining phase.
Processes grand_aggregator.py output to compute derived metrics
(overall means across variables, RMSE/MAE ratio). Creates compact
summary tables with 5 decimal precision. Supports --bvar flag to
include BVAR-specific columns (DIC_mean, data_tag_dist) or --tcn
mode for neural network configurations.

### EDA folder

**Contains 2 files in a sequential pipeline:**

1. **eda.py —**
Descriptive statistics calculator for individual datasets. Computes 28 statistics per dataset:
basic measures (mean, median, std, IQR, min, max); distribution properties (skewness, kurtosis);
mean correlation across variable pairs; and Jarque–Bera normality
test p-values for all three variables. Outputs comma-separated values
to stdout. Run with python3 eda.py <path_to_csv>.
2. **aggregate.py —**
Cross-dataset aggregator for summary statistics. Reads eda_stat.csv containing
statistics from all datasets, computes means across datasets for each statistic,
calculates Jarque–Bera passing rates (percentage of datasets where p > 0.05), and outputs
formatted table with rows as statistics and columns as variables. Float precision: 5 decimal places.

### process_results.py

Post-processing tool for comparing BVAR and TCN forecasting results.

#### Features

- Loads and merges BVAR and TCN result CSVs for single excitement level
- Separates metrics into five tables (overall, coverage, width, MAE, RMSE)
- Calculates aggregate performance metrics across variables
- Exports to CSV format for further analysis
- Generates publication-ready LaTeX tables with booktabs formatting
- Produces five formatted tables in single .tex file

#### Usage

```bash
python process_results.py 0p25
python process_results.py 0p50
python process_results.py 0p75
```

#### Arguments

- Positional argument: excitement level (0p25, 0p50, or 0p75)

#### Input Files

Expected in working directory:

- `BVAR{excitement}_final.csv`: BVAR aggregated results
- `TCN{excitement}_final.csv`: TCN aggregated results

#### Output

**CSV files:**
- `pandas_master_{excitement}.csv`: Merged results (2 models)
- `pandas_overall_{excitement}.csv`: Overall metrics with RMSE/MAE ratio
- `pandas_coverage_{excitement}.csv`: Coverage rates (80%, 90%, 95%)
- `pandas_width_{excitement}.csv`: Interval widths (80%, 90%, 95%)
- `pandas_MAE_{excitement}.csv`: MAE by variable (mean/std)
- `pandas_RMSE_{excitement}.csv`: RMSE by variable (mean/std)

**LaTeX file:**
- `{excitement}_final.tex`: Five formatted tables ready for thesis inclusion
  - Table 1: Overall Performance (runtime, RMSE, MAE, ratio)
  - Table 2: Coverage Metrics 
  - Table 3: Width Metrics 
  - Table 4: RMSE by Variable
  - Table 5: MAE by Variable

#### Configuration

Edit directly in `export_to_latex()` function:
```python
# Numeric precision
runtime: .2f        # 2 decimal places
metrics: .5f        # 5 decimal places

# Coverage/Width levels
levels = [80, 90, 95]  # Prediction interval levels
```

## Requirements (2026-02-06)

- Python Version: 3.13
- numpy==2.3.5
- pandas==2.3.3
- scipy==1.16.3
- pymc==5.27.0
- statsmodels==0.14.6
- matplotlib==3.10.8
- torch==2.9.1

###  NVIDIA CUDA 12 Libraries (installed for enabling torch+CUDA)

**These are dependencies linked to torch and listed for reference:**

- nvidia-cublas-cu12==12.8.4.1
- nvidia-cuda-cupti-cu12==12.8.93
- nvidia-cuda-nvrtc-cu12==12.8.90
- nvidia-cuda-runtime-cu12==12.8.90
- nvidia-cudnn-cu12==9.10.2.21
- nvidia-cufft-cu12==11.3.3.83
- nvidia-cufile-cu12==1.13.1.3
- nvidia-curand-cu12==10.3.9.90
- nvidia-cusolver-cu12==11.7.3.90
- nvidia-cusparse-cu12==12.5.8.93
- nvidia-cusparselt-cu12==0.7.1
- nvidia-nccl-cu12==2.275
- nvidia-nvjitlink-cu12==12.8.93
- nvidia-nvshmem-cu12==3.3.20
- nvidia-nvtx-cu12==12.8.90
