import os
import pandas as pd
import numpy as np
import pickle
import cvxpy as cp
import matplotlib.pyplot as plt
from tqdm import tqdm

data_dir = '/mnt/c/Users/Owner/5370'
regimes_path = os.path.join(data_dir, '02_regimes_data.csv')
params_path = os.path.join(data_dir, '03_regime_params.pkl')
returns_path = os.path.join(data_dir, '01_stock_returns.csv')

macro_df = pd.read_csv(regimes_path, index_col='Date')
stock_returns = pd.read_csv(returns_path, index_col='Date')

# Date alignment
macro_df.index = pd.to_datetime(macro_df.index, errors='coerce').normalize()
stock_returns.index = pd.to_datetime(stock_returns.index, errors='coerce').normalize()

if macro_df.index.tz is not None:
    macro_df.index = macro_df.index.tz_localize(None)
if stock_returns.index.tz is not None:
    stock_returns.index = stock_returns.index.tz_localize(None)

macro_df = macro_df[macro_df.index.notnull()]
stock_returns = stock_returns[stock_returns.index.notnull()]
macro_df = macro_df[~macro_df.index.duplicated(keep='first')]
stock_returns = stock_returns[~stock_returns.index.duplicated(keep='first')]

aligned_dates = macro_df.index.intersection(stock_returns.index)
macro_df = macro_df.loc[aligned_dates]
stock_returns = stock_returns.loc[aligned_dates]

with open(params_path, 'rb') as f:
    regime_params = pickle.load(f)

gamma = 2.0  
n_assets = len(stock_returns.columns)

# 1. Calculate 1/N Equal Weight Benchmark
equal_weight_returns = []
for date in aligned_dates:
    daily_returns = stock_returns.loc[date].values
    active_mask = ~np.isnan(daily_returns)
    n_active = np.sum(active_mask)
    
    if n_active > 0:
        w_eq = np.zeros(n_assets)
        w_eq[active_mask] = 1.0 / n_active
        port_ret = np.dot(w_eq, np.nan_to_num(daily_returns))
    else:
        port_ret = 0.0
    equal_weight_returns.append(port_ret)

macro_df['1/N_Ret'] = equal_weight_returns

# 2. Calculate Unpenalized Benchmark 
unpenalized_returns = []
w_prev = np.zeros(n_assets)
rebalance_freq = 5 

for i, date in enumerate(tqdm(aligned_dates, desc="Optimizing Unpenalized")):
    row = macro_df.loc[date]
    daily_returns = stock_returns.loc[date].values
    
    if i % rebalance_freq == 0:
        current_regime = int(row['regime'])
        mu_k = regime_params[current_regime]['mu']
        Sigma_k = regime_params[current_regime]['cleaned_cov']
        
        active_mask = ~np.isnan(daily_returns)
        lambda_pen = row['Bid_Ask_Spread']
        
        w_t = cp.Variable(n_assets)
        expected_return = w_t.T @ mu_k
        risk_penalty = (gamma / 2) * cp.quad_form(w_t, Sigma_k)
        
        objective = cp.Maximize(expected_return - risk_penalty)
        
        constraints = [
            cp.sum(w_t) == 1, 
            w_t >= 0,
            w_t[~active_mask] == 0 
        ]
        
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.OSQP) 
            if problem.status in ["infeasible", "unbounded", None]:
                w_optimal = w_prev
            else:
                w_optimal = w_t.value
        except Exception:
            w_optimal = w_prev
    else:
        w_drift = w_prev * (1 + np.nan_to_num(daily_returns))
        sum_drift = np.sum(w_drift)
        w_optimal = w_drift / sum_drift if sum_drift > 0 else w_prev
        
    turnover = np.sum(np.abs(w_optimal - w_prev))
    cost = turnover * lambda_pen if i % rebalance_freq == 0 else 0
    
    port_ret = np.dot(w_optimal, np.nan_to_num(daily_returns)) - cost
    unpenalized_returns.append(port_ret)
    w_prev = w_optimal

macro_df['Unpenalized_Ret'] = unpenalized_returns

plt.figure(figsize=(14, 8))

macro_df['L1_Penalized_Ret'] = pd.read_csv(os.path.join(data_dir, '04_portfolio_weights.csv'), index_col=0).sum(axis=1) # Temporary dummy for plot alignment, we will use the actual returns
cumulative_L1 = (1 + macro_df['Port_Ret']).cumprod() if 'Port_Ret' in macro_df.columns else (1 + pd.Series(unpenalized_returns)).cumprod() # Safety fallback

L1_returns = pd.read_csv(regimes_path, index_col='Date')['Port_Ret'] if 'Port_Ret' in pd.read_csv(regimes_path).columns else pd.read_csv(os.path.join(data_dir, '01_cleaned_macro.csv'))['SPX_Ret'] # Needs exact data tracking
weights_df = pd.read_csv(os.path.join(data_dir, '04_portfolio_weights.csv'), index_col=0)
aligned_returns = stock_returns.loc[weights_df.index]
l1_daily_rets = (weights_df.values * np.nan_to_num(aligned_returns.values)).sum(axis=1)

cumulative_L1 = (1 + l1_daily_rets).cumprod()
cumulative_eq = (1 + macro_df['1/N_Ret']).cumprod()
cumulative_unp = (1 + macro_df['Unpenalized_Ret']).cumprod()

plt.plot(weights_df.index, cumulative_L1, color='blue', label='Dynamic L1 Portfolio', linewidth=2)
plt.plot(weights_df.index, cumulative_unp, color='red', label='Unpenalized Mean-Variance (High Turnover)', alpha=0.7)
plt.plot(weights_df.index, cumulative_eq, color='gray', label='1/N Equal Weight Benchmark', alpha=0.7)

plt.yscale('log')
plt.title('Regime-Switching Strategy vs. Benchmarks (Log Scale)')
plt.ylabel('Cumulative Wealth Multiplier (Log Scale)')
plt.xlabel('Date')
plt.grid(True, alpha=0.3, which="both", ls="--")
plt.legend()
plt.tight_layout()

plot_path = os.path.join(data_dir, 'benchmark_comparison_plot.png')
plt.savefig(plot_path)

# Print Comparative Metrics
def print_metrics(name, rets):
    ann_ret = np.mean(rets) * 252
    ann_vol = np.std(rets) * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    print(f"{name} -> Ann. Ret: {ann_ret*100:.2f}% | Ann. Vol: {ann_vol*100:.2f}% | Sharpe: {sharpe:.2f}")

print("\n--- Strategy Comparison ---")
print_metrics("Dynamic L1 Portfolio", l1_daily_rets)
print_metrics("Unpenalized Mean-Variance", unpenalized_returns)
print_metrics("1/N Equal Weight Benchmark", equal_weight_returns)