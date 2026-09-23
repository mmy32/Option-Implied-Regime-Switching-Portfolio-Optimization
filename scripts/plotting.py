import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set professional plotting style
plt.style.use('bmh')

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
data_dir = os.path.join(project_root, 'data')
report_dir = os.path.join(project_root, 'report')
regimes_path = os.path.join(data_dir, '02_regimes_data.csv')
weights_path = os.path.join(data_dir, '04_portfolio_weights.csv')
returns_path = os.path.join(data_dir, '01_stock_returns.csv')

print("Loading data for visualization...")
macro_df = pd.read_csv(regimes_path, index_col='Date', parse_dates=True)
weights_df = pd.read_csv(weights_path, index_col='Date', parse_dates=True)
stock_returns = pd.read_csv(returns_path, index_col='Date', parse_dates=True)

# Align dates to ensure clean plotting
aligned_dates = weights_df.index.intersection(macro_df.index).intersection(stock_returns.index)
macro_df = macro_df.loc[aligned_dates]
weights_df = weights_df.loc[aligned_dates]
stock_returns = stock_returns.loc[aligned_dates]

# Reconstruct L1 Portfolio Returns from weights
l1_daily_rets = (weights_df.values * np.nan_to_num(stock_returns.values)).sum(axis=1)
macro_df['L1_Port_Ret'] = l1_daily_rets

# Reconstruct 1/N Equal Weight Returns for baseline comparison
active_mask = ~np.isnan(stock_returns.values)
n_active = active_mask.sum(axis=1)
eq_weights = np.where(active_mask, 1.0 / n_active[:, None], 0.0)
macro_df['Equal_Weight_Ret'] = (eq_weights * np.nan_to_num(stock_returns.values)).sum(axis=1)

print("Generating Plot 1: Market Regimes and VIX Term Structure...")
fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()
ax1.plot(macro_df.index, macro_df['VIX_VIX3M_Ratio'], color='navy', alpha=0.7, label='VIX/VIX3M Ratio')
ax2.fill_between(macro_df.index, 0, macro_df['regime'], color='red', alpha=0.2, label='Stress Regime (State 1)')
ax1.set_xlabel('Date')
ax1.set_ylabel('VIX/VIX3M Ratio')
ax2.set_ylabel('Regime State (0 = Calm, 1 = Stress)')
ax1.set_title('HMM Inferred Market Regimes vs Volatility Term Structure')
fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
plt.tight_layout()
plt.savefig(os.path.join(data_dir, 'fig_01_market_regimes.png'))
plt.savefig(os.path.join(report_dir, 'fig_01_market_regimes.png'))
plt.close()

print("Generating Plot 2: Portfolio Drawdowns...")
def calculate_drawdown(returns):
    cum_ret = (1 + returns).cumprod()
    running_max = cum_ret.cummax()
    drawdown = (cum_ret - running_max) / running_max
    return drawdown

l1_drawdown = calculate_drawdown(macro_df['L1_Port_Ret'])
eq_drawdown = calculate_drawdown(macro_df['Equal_Weight_Ret'])

plt.figure(figsize=(12, 6))
plt.fill_between(l1_drawdown.index, l1_drawdown, 0, color='blue', alpha=0.5, label='Dynamic L1 Portfolio')
plt.fill_between(eq_drawdown.index, eq_drawdown, 0, color='gray', alpha=0.3, label='1/N Benchmark')
plt.title('Historical Drawdowns: L1 Portfolio vs Market Benchmark')
plt.ylabel('Drawdown Percentage')
plt.xlabel('Date')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(data_dir, 'fig_02_drawdowns.png'))
plt.savefig(os.path.join(report_dir, 'fig_02_drawdowns.png'))
plt.close()

print("Generating Plot 3: 252-Day Rolling Volatility...")
rolling_vol_l1 = macro_df['L1_Port_Ret'].rolling(window=252).std() * np.sqrt(252)
rolling_vol_eq = macro_df['Equal_Weight_Ret'].rolling(window=252).std() * np.sqrt(252)

plt.figure(figsize=(12, 6))
plt.plot(rolling_vol_l1.index, rolling_vol_l1, color='blue', label='Dynamic L1 Portfolio Volatility')
plt.plot(rolling_vol_eq.index, rolling_vol_eq, color='gray', label='1/N Benchmark Volatility')
plt.title('Rolling 1-Year Annualized Volatility')
plt.ylabel('Annualized Volatility')
plt.xlabel('Date')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(data_dir, 'fig_03_rolling_volatility.png'))
plt.savefig(os.path.join(report_dir, 'fig_03_rolling_volatility.png'))
plt.close()

print("Generating Plot 4: Capital Allocation Stability (Top 10 Assets)...")
# Find the top 10 assets by average weight to plot an area chart
avg_weights = weights_df.mean().sort_values(ascending=False)
top_10_tickers = avg_weights.head(10).index
top_10_weights = weights_df[top_10_tickers]

plt.figure(figsize=(12, 6))
plt.stackplot(top_10_weights.index, top_10_weights.T, labels=top_10_tickers, alpha=0.8)
plt.title('Portfolio Weight Stability: Top 10 Allocated Assets over Time')
plt.ylabel('Capital Allocation (Weight)')
plt.xlabel('Date')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig(os.path.join(data_dir, 'fig_04_weight_allocations.png'))
plt.savefig(os.path.join(report_dir, 'fig_04_weight_allocations.png'))
plt.close()

print("Generating Table 1: Regime Characteristic Summary...")
regime_stats = macro_df.groupby('regime')[['VIX_VIX3M_Ratio', 'vrp', 'Bid_Ask_Spread']].mean()
regime_stats.index = ['Regime 0 (Calm)', 'Regime 1 (Stress)']
regime_stats.columns = ['Avg VIX Term Structure', 'Avg Variance Risk Premium', 'Avg SPY Bid-Ask Spread']
regime_stats.to_csv(os.path.join(data_dir, 'table_01_regime_characteristics.csv'))

print("Generating Table 2: Comprehensive Performance Metrics...")
def get_metrics(returns):
    ann_ret = np.mean(returns) * 252
    ann_vol = np.std(returns) * np.sqrt(252)
    sharpe = ann_ret / ann_vol
    max_dd = calculate_drawdown(returns).min()
    return pd.Series({
        'Annualized Return': f"{ann_ret*100:.2f}%",
        'Annualized Volatility': f"{ann_vol*100:.2f}%",
        'Sharpe Ratio': f"{sharpe:.2f}",
        'Max Drawdown': f"{max_dd*100:.2f}%"
    })

metrics_df = pd.DataFrame({
    'Dynamic L1 Portfolio': get_metrics(macro_df['L1_Port_Ret']),
    '1/N Equal Weight': get_metrics(macro_df['Equal_Weight_Ret'])
}).T
metrics_df.to_csv(os.path.join(data_dir, 'table_02_performance_metrics.csv'))

print("All plots and tables successfully generated and saved to /mnt/c/Users/Owner/5370")