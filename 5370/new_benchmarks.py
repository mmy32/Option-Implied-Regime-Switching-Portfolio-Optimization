import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('bmh')

data_dir = '/mnt/c/Users/Owner/5370'
regimes_path = os.path.join(data_dir, '02_regimes_data.csv')
returns_path = os.path.join(data_dir, '01_stock_returns.csv')
weights_path = os.path.join(data_dir, '04_portfolio_weights.csv')

macro_df = pd.read_csv(regimes_path, index_col='Date', parse_dates=True)
stock_returns = pd.read_csv(returns_path, index_col='Date', parse_dates=True)
weights_df = pd.read_csv(weights_path, index_col='Date', parse_dates=True)

aligned_dates = weights_df.index.intersection(macro_df.index).intersection(stock_returns.index)
macro_df = macro_df.loc[aligned_dates]
weights_df = weights_df.loc[aligned_dates]
stock_returns = stock_returns.loc[aligned_dates]

active_mask = ~np.isnan(stock_returns.values)
n_active = active_mask.sum(axis=1)
eq_weights = np.where(active_mask, 1.0 / n_active[:, None], 0.0)

macro_df['L1_Ret'] = (weights_df.values * np.nan_to_num(stock_returns.values)).sum(axis=1)
macro_df['Equal_Weight_Ret'] = (eq_weights * np.nan_to_num(stock_returns.values)).sum(axis=1)

daily_bond_yield = 0.04 / 252
macro_df['60_40_Ret'] = (0.60 * macro_df['Equal_Weight_Ret']) + (0.40 * daily_bond_yield)


if 'VIX_Close' in macro_df.columns:
    vix_series = macro_df['VIX_Close']
else:
    vix_series = pd.Series(20, index=macro_df.index)

macro_df['VIX_Heuristic_Ret'] = np.where(
    vix_series > 30, 
    macro_df['Equal_Weight_Ret'] * 0.5, 
    macro_df['Equal_Weight_Ret']
)

cumulative_L1 = (1 + macro_df['L1_Ret']).cumprod()
cumulative_EQ = (1 + macro_df['Equal_Weight_Ret']).cumprod()
cumulative_6040 = (1 + macro_df['60_40_Ret']).cumprod()
cumulative_VIX = (1 + macro_df['VIX_Heuristic_Ret']).cumprod()

plt.figure(figsize=(14, 8))
plt.plot(cumulative_L1.index, cumulative_L1, color='blue', label='Dynamic L1 Portfolio', linewidth=2)
plt.plot(cumulative_EQ.index, cumulative_EQ, color='gray', label='1/N Equity Benchmark', alpha=0.7)
plt.plot(cumulative_6040.index, cumulative_6040, color='green', label='60/40 Equity-Bond', alpha=0.7)
plt.plot(cumulative_VIX.index, cumulative_VIX, color='orange', label='VIX Threshold Heuristic', alpha=0.7)

plt.yscale('log')
plt.title('Expanded Baselines Comparison (Log Scale)')
plt.ylabel('Cumulative Wealth Multiplier')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(data_dir, 'fig_05_expanded_baselines.png'))
