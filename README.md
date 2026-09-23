# Option-Implied Regime-Switching Portfolio Optimization

Quantitative portfolio management framework that formulates asset allocation as a sequential regime-switching mean-variance optimization problem across 2,339 S&P 500 equities (2008–2026).

The policy infers latent market stress regimes using a Gaussian Hidden Markov Model (HMM) driven by forward-looking option indicators (Variance Risk Premium and VIX term structure), denoises large-scale covariance matrices via Randomized Singular Value Decomposition (SVD), and enforces an $L_1$-norm turnover penalty to manage transaction costs alongside individual position limits.

---

## Repository Structure

```text
.
├── data/                       # Market data and generated artifacts
│   ├── cboe_skew.xlsx          # Raw CBOE SKEW index data
│   ├── riskfreerate.xlsx       # 3-Month Treasury bill risk-free rate
│   ├── spx_xndx.xlsx           # SPX and NDX total return indices
│   ├── spybidask.xlsx          # Daily SPY bid-ask spread
│   ├── vix_daily.xlsx          # CBOE VIX index
│   ├── vix_vix3m_ratio_daily.xlsx # VIX / VIX3M term structure data
│   ├── 01_cleaned_macro.csv    # Merged macro indicators and VRP features
│   ├── 02_regimes_data.csv     # Expanding-window HMM inferred regimes
│   ├── table_01_regime_characteristics.csv # Regime statistics summary
│   └── table_02_performance_metrics.csv   # Out-of-sample backtest metrics
├── scripts/                    # Core quantitative research pipeline
│   ├── data_prep.py            # Clean raw Excel datasets & compute returns
│   ├── hmm_regimes.py          # Expanding-window Gaussian HMM regime model
│   ├── covariance.py           # Regime-conditioned Randomized SVD covariance
│   ├── portfolio_opt.py        # Weekly L1-penalized OSQP portfolio optimizer
│   ├── benchmarks.py           # Unpenalized MVO & 1/N equal-weight baselines
│   ├── new_benchmarks.py       # 60/40 & VIX threshold heuristic comparisons
│   ├── transaction_cost_sensitivity.py # Turnover vs. transaction cost stress tests
│   ├── plotting.py             # Visualizations & summary tables
│   └── regime_data.py          # Standalone regime exploration utility
├── notebooks/                  # Interactive Jupyter notebooks
│   └── exploration.ipynb       # Interactive regimes & portfolio walkthrough
├── report/                     # Academic report & LaTeX publication
│   ├── main.tex                # Complete paper source
│   ├── references.bib          # Bibliography
│   ├── main.pdf                # Compiled 19-page report
│   └── fig_01_*.png ...        # High-resolution figures (fig_01 to fig_06)
├── build_report.py             # Automated report compilation engine
├── build_report.sh             # Executable shell wrapper for PDF generation
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mmy32/Option-Implied-Regime-Switching-Portfolio-Optimization.git
   cd Option-Implied-Regime-Switching-Portfolio-Optimization
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Quantitative Pipeline

Execute the pipeline in sequential order:

```bash
# 1. Preprocess macro and equity returns
python scripts/data_prep.py

# 2. Fit expanding-window Hidden Markov Model to identify market regimes
python scripts/hmm_regimes.py

# 3. Clean empirical covariance matrices using Randomized SVD
python scripts/covariance.py

# 4. Run weekly L1-penalized dynamic portfolio optimization
python scripts/portfolio_opt.py

# 5. Evaluate benchmarks and generate report figures
python scripts/benchmarks.py
python scripts/new_benchmarks.py
python scripts/transaction_cost_sensitivity.py
python scripts/plotting.py
```

---

## Compiling the Report PDF

A single command synchronizes all figures, resolves citations, and compiles `report/main.tex` into a 19-page PDF:

```bash
./build_report.sh --clean
```

Or via Python:
```bash
python build_report.py --clean
```

---

## Key Empirical Results (2008–2026 Out-of-Sample)

| Strategy | Ann. Return | Ann. Volatility | Sharpe Ratio | Max Drawdown |
| :--- | :---: | :---: | :---: | :---: |
| **Dynamic $L_1$ Portfolio** | **37.09%** | **23.72%** | **1.56** | **-31.4%** |
| Unpenalized Mean-Variance | 47.58% | 32.17% | 1.48 | -44.2% |
| Static 60/40 Equity-Bond | 7.91% | 12.43% | 0.64 | -35.2% |
| 1/N Equal Weight Benchmark | 8.38% | 20.84% | 0.40 | -55.8% |
