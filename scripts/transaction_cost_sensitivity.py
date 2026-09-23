import os
import pandas as pd
import numpy as np
import pickle
import cvxpy as cp
import matplotlib.pyplot as plt
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
data_dir = os.path.join(project_root, 'data')
report_dir = os.path.join(project_root, 'report')
regimes_path = os.path.join(data_dir, '02_regimes_data.csv')
params_path = os.path.join(data_dir, '03_regime_params.pkl')
returns_path = os.path.join(data_dir, '01_stock_returns.csv')

macro_df = pd.read_csv(regimes_path, index_col='Date', parse_dates=True)
stock_returns = pd.read_csv(returns_path, index_col='Date', parse_dates=True)
with open(params_path, 'rb') as f:
    regime_params = pickle.load(f)

test_dates = macro_df.index[macro_df.index > '2023-01-01']
macro_df = macro_df.loc[test_dates]
stock_returns = stock_returns.loc[test_dates]

gamma = 2.0
n_assets = len(stock_returns.columns)
multipliers = [1, 5, 10]
turnover_results = []

for mult in multipliers:
    print(f"Running sensitivity test with transaction cost multiplier: {mult}x")
    w_prev = np.zeros(n_assets)
    total_turnover = 0
    
    for i, date in enumerate(tqdm(test_dates)):
        row = macro_df.loc[date]
        daily_returns = stock_returns.loc[date].values
        
        if i % 5 == 0:
            current_regime = int(row['regime'])
            mu_k = regime_params[current_regime]['mu']
            Sigma_k = regime_params[current_regime]['cleaned_cov']
            active_mask = ~np.isnan(daily_returns)
            active_indices = np.where(active_mask)[0]
            n_act = len(active_indices)
            
            # Change transaction cost penalty for sensitivity
            lambda_pen = row['Bid_Ask_Spread'] * mult
            
            if n_act > 0:
                w_sub = cp.Variable(n_act)
                w_prev_sub = w_prev[active_indices]
                mu_sub = mu_k[active_indices]
                Sigma_sub = Sigma_k[np.ix_(active_indices, active_indices)]
                
                expected_return = w_sub.T @ mu_sub
                risk_penalty = (gamma / 2) * cp.quad_form(w_sub, Sigma_sub)
                tc_penalty = lambda_pen * cp.norm(w_sub - w_prev_sub, 1)
                
                objective = cp.Maximize(expected_return - risk_penalty - tc_penalty)
                constraints = [cp.sum(w_sub) == 1, w_sub >= 0, w_sub <= 0.05]
                
                problem = cp.Problem(objective, constraints)
                try:
                    problem.solve(solver=cp.OSQP)
                    if problem.status not in ["infeasible", "unbounded", None] and w_sub.value is not None:
                        w_optimal = np.zeros(n_assets)
                        w_optimal[active_indices] = w_sub.value
                    else:
                        w_optimal = w_prev
                except:
                    w_optimal = w_prev
            else:
                w_optimal = w_prev
                
            total_turnover += np.sum(np.abs(w_optimal - w_prev))
        else:
            w_drift = w_prev * (1 + np.nan_to_num(daily_returns))
            sum_drift = np.sum(w_drift)
            w_optimal = w_drift / sum_drift if sum_drift > 0 else w_prev
            
        w_prev = w_optimal
        
    avg_weekly_turnover = total_turnover / (len(test_dates) / 5)
    turnover_results.append(avg_weekly_turnover)

plt.figure(figsize=(8, 5))
plt.bar([f"{m}x" for m in multipliers], turnover_results, color=['blue', 'orange', 'red'])
plt.title('Stress Test: Average Weekly Turnover vs. Transaction Costs')
plt.xlabel('Bid-Ask Spread Multiplier')
plt.ylabel('Average Weekly Turnover (%)')
plt.savefig(os.path.join(data_dir, 'fig_06_sensitivity.png'))
plt.savefig(os.path.join(report_dir, 'fig_06_sensitivity.png'))
