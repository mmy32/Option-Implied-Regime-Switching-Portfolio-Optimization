import os
import pandas as pd
import numpy as np
import pickle

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
data_dir = os.path.join(project_root, 'data')
report_dir = os.path.join(project_root, 'report')
regimes_path = os.path.join(data_dir, '02_regimes_data.csv')
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

macro_df = macro_df.loc[aligned_dates]
stock_returns = stock_returns.loc[aligned_dates]

returns_filled = stock_returns.fillna(0)
asset_columns = stock_returns.columns

def randomized_svd(A, rank, n_oversamples=10, n_iter=2):
    M, N = A.shape
    Omega = np.random.randn(N, rank + n_oversamples)
    Y = A @ Omega
    for _ in range(n_iter):
        Y = A @ (A.T @ Y)
    Q, _ = np.linalg.qr(Y)
    B = Q.T @ A
    U_tilde, S, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ U_tilde
    return U[:, :rank], S[:rank], Vt[:rank, :]

regimes = macro_df['regime'].unique()
regime_params = {}

for k in regimes:
    regime_dates = macro_df[macro_df['regime'] == k].index
    regime_data = returns_filled.loc[regime_dates]
    
    mu_k = regime_data.mean().values
    empirical_cov = regime_data.cov().values
    
    target_rank = min(10, len(asset_columns) - 1)
    
    U, S, Vt = randomized_svd(empirical_cov, rank=target_rank)
    cleaned_cov = U @ np.diag(S) @ U.T
    np.fill_diagonal(cleaned_cov, np.diag(empirical_cov))
        
    regime_params[k] = {
        'mu': mu_k,
        'cleaned_cov': cleaned_cov
    }

output_path = os.path.join(data_dir, '03_regime_params.pkl')
with open(output_path, 'wb') as f:
    pickle.dump(regime_params, f)

