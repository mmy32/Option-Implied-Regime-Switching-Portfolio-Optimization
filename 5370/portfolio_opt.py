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

print(f"\n DIAGNOSTICS")
print(f"Macro dates range: {macro_df.index.min()} to {macro_df.index.max()} ({len(macro_df)} days)")
print(f"Stock dates range: {stock_returns.index.min()} to {stock_returns.index.max()} ({len(stock_returns)} days)")
print(f"Total overlapping days: {len(aligned_dates)}\n")

macro_df = macro_df.loc[aligned_dates]
stock_returns = stock_returns.loc[aligned_dates]

with open(params_path, 'rb') as f:
    regime_params = pickle.load(f)

gamma = 2.0  
n_assets = len(stock_returns.columns)
asset_columns = stock_returns.columns

optimal_weights_history = []
portfolio_returns = []
w_prev = np.zeros(n_assets)


rebalance_freq = 5  # Rebalance weekly

for i, date in enumerate(tqdm(aligned_dates, desc="Optimizing Portfolio")):
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
        transaction_cost_penalty = lambda_pen * cp.norm(w_t - w_prev, 1)
        
        objective = cp.Maximize(expected_return - risk_penalty - transaction_cost_penalty)
        
        constraints = [
            cp.sum(w_t) == 1, 
            w_t >= 0,
            w_t[~active_mask] == 0, 
            w_t <= 0.05
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

    optimal_weights_history.append(w_optimal)
    
    port_ret = np.dot(w_optimal, np.nan_to_num(daily_returns))
    portfolio_returns.append(port_ret)
    
    w_prev = w_optimal

macro_df['Port_Ret'] = portfolio_returns
excess_returns = macro_df['Port_Ret'] - macro_df['Daily_RF']
annualized_return = np.mean(macro_df['Port_Ret']) * 252
annualized_vol = np.std(macro_df['Port_Ret']) * np.sqrt(252)
sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

print("\n--- Out-of-Sample Performance (S&P 500 Universe) ---")
print(f"Annualized Return: {annualized_return * 100:.2f}%")
print(f"Annualized Volatility: {annualized_vol * 100:.2f}%")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

weights_df = pd.DataFrame(optimal_weights_history, index=aligned_dates, columns=asset_columns)
weights_output_path = os.path.join(data_dir, '04_portfolio_weights.csv')
weights_df.to_csv(weights_output_path)

plt.figure(figsize=(12, 6))
cumulative_returns = (1 + macro_df['Port_Ret']).cumprod()
plt.plot(cumulative_returns.index, cumulative_returns, color='blue', label='Dynamic L1 Portfolio')
plt.title('Cumulative Return of Regime-Switching S&P 500 Strategy')
plt.ylabel('Cumulative Wealth (Multiplier)')
plt.xlabel('Date')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plot_path = os.path.join(data_dir, 'cumulative_return_plot.png')
plt.savefig(plot_path)
