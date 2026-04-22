import os
import pandas as pd
import numpy as np
import pickle

data_dir = '/mnt/c/Users/Owner/5370'
input_path = os.path.join(data_dir, '02_regimes_data.csv')
df = pd.read_csv(input_path, parse_dates=['Date'], index_col='Date')

asset_columns = ['SPX_Ret', 'NDX_Ret']

def randomized_svd(A, rank, n_oversamples=5, n_iter=2):
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

regimes = df['regime'].unique()
regime_params = {}

for k in regimes:
    regime_data = df[df['regime'] == k][asset_columns]
    mu_k = regime_data.mean().values
    empirical_cov = regime_data.cov().values
    
    target_rank = 1  # Reduced rank since we only have 2 assets
    
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

print(f"Cleaned covariance parameters saved to {output_path}")