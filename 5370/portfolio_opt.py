import os
import pandas as pd
import numpy as np
import pickle
import cvxpy as cp
import matplotlib.pyplot as plt

data_dir = '/mnt/c/Users/Owner/5370'
regimes_path = os.path.join(data_dir, '02_regimes_data.csv')
params_path = os.path.join(data_dir, '03_regime_params.pkl')

df = pd.read_csv(regimes_path, parse_dates=['Date'], index_col='Date')
with open(params_path, 'rb') as f:
    regime_params = pickle.load(f)

gamma = 2.0  
n_assets = len(regime_params[0]['mu'])

optimal_weights_history = []
portfolio_returns = []
w_prev = np.ones(n_assets) / n_assets

asset_columns = ['SPX_Ret', 'NDX_Ret']

print("Starting dynamic optimization backtest...")

for date, row in df.iterrows():
    current_regime = row['regime']
    mu_k = regime_params[current_regime]['mu']
    Sigma_k = regime_params[current_regime]['cleaned_cov']
    
    # Dynamic transaction cost penalty based on daily bid-ask spread
    lambda_pen = row['Bid_Ask_Spread'] 
    
    w_t = cp.Variable(n_assets)
    expected_return = w_t.T @ mu_k
    risk_penalty = (gamma / 2) * cp.quad_form(w_t, Sigma_k)
    transaction_cost_penalty = lambda_pen * cp.norm(w_t - w_prev, 1)
    
    objective = cp.Maximize(expected_return - risk_penalty - transaction_cost_penalty)
    constraints = [cp.sum(w_t) == 1, w_t >= 0]
    
    problem = cp.Problem(objective, constraints)
    
    try:
        problem.solve(solver=cp.ECOS)
        w_optimal = w_t.value
    except Exception:
        w_optimal = w_prev
        
    optimal_weights_history.append(w_optimal)
    
    # Calculate daily return of the portfolio
    daily_asset_returns = np.array([row['SPX_Ret'], row['NDX_Ret']])
    port_ret = np.dot(w_optimal, daily_asset_returns)
    portfolio_returns.append(port_ret)
    
    w_prev = w_optimal

df['Port_Ret'] = portfolio_returns

# Evaluation Metrics
excess_returns = df['Port_Ret'] - df['Daily_RF']
annualized_return = np.mean(df['Port_Ret']) * 252
annualized_vol = np.std(df['Port_Ret']) * np.sqrt(252)
sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

print("\n--- Out-of-Sample Performance ---")
print(f"Annualized Return: {annualized_return * 100:.2f}%")
print(f"Annualized Volatility: {annualized_vol * 100:.2f}%")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

weights_df = pd.DataFrame(optimal_weights_history, index=df.index, columns=asset_columns)
weights_output_path = os.path.join(data_dir, '04_portfolio_weights.csv')
weights_df.to_csv(weights_output_path)

plt.figure(figsize=(12, 6))
plt.stackplot(weights_df.index, weights_df.T, labels=weights_df.columns, alpha=0.8)
plt.title('Dynamic Portfolio Allocations (SPX vs NDX)')
plt.ylabel('Portfolio Weight')
plt.legend(loc='upper left')
plt.tight_layout()

plot_path = os.path.join(data_dir, 'allocation_plot.png')
plt.savefig(plot_path)
print(f"Allocation plot saved to {plot_path}")