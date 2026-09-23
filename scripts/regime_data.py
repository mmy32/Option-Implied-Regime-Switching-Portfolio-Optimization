import os
import pandas as pd
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
data_dir = os.path.join(project_root, 'data')
report_dir = os.path.join(project_root, 'report')
all_files = os.listdir(data_dir)

vix_vix3m_path = next((os.path.join(data_dir, f) for f in all_files if 'vix_vix3m' in f.lower() and f.endswith('.xlsx')), None)
vix_path = next((os.path.join(data_dir, f) for f in all_files if 'vix_daily' in f.lower() and f.endswith('.xlsx')), None)
spx_path = next((os.path.join(data_dir, f) for f in all_files if 'spx' in f.lower() and f.endswith('.xlsx')), None)

vix_vix3m = pd.read_excel(vix_vix3m_path, skiprows=6, header=None, usecols=[0, 12], 
                        names=['Date', 'VIX_VIX3M_Ratio'], parse_dates=['Date'], index_col='Date')

vix = pd.read_excel(vix_path, skiprows=6, header=None, usecols=[0, 1], 
                  names=['Date', 'VIX_Close'], parse_dates=['Date'], index_col='Date')

spx = pd.read_excel(spx_path, skiprows=6, header=None, usecols=[0, 1], 
                  names=['Date', 'SPX_Close'], parse_dates=['Date'], index_col='Date')

df = pd.concat([vix_vix3m, vix, spx], axis=1).dropna()

df['log_ret'] = np.log(df['SPX_Close'] / df['SPX_Close'].shift(1))
df['realized_var'] = df['log_ret'].rolling(window=21).var() * 252
df['implied_var'] = (df['VIX_Close'] / 100) ** 2
df['vrp'] = df['implied_var'] - df['realized_var']
df = df.dropna()

# Fit the HMM
X = df[['VIX_VIX3M_Ratio', 'vrp']].values

model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=1000, tol=0.01, random_state=42)
model.fit(X)

df['regime'] = model.predict(X)

# Plot the SPX overlayed with Regimes
plt.figure(figsize=(14, 7))

# Plot the SPX Close Price
plt.plot(df.index, df['SPX_Close'], color='black', label='SPX Close Price', linewidth=1)

# Highlight Regime 1
plt.fill_between(df.index, df['SPX_Close'].min(), df['SPX_Close'].max(), 
                 where=(df['regime'] == 1), color='red', alpha=0.3, label='Regime 1')

# Highlight Regime 0 
plt.fill_between(df.index, df['SPX_Close'].min(), df['SPX_Close'].max(), 
                 where=(df['regime'] == 0), color='green', alpha=0.3, label='Regime 0')

plt.title('S&P 500 Price overlayed with HMM Inferred Regimes')
plt.xlabel('Date')
plt.ylabel('SPX Price')
plt.legend()
plt.tight_layout()

windows_dir = os.path.dirname(os.path.abspath(__file__))
report_dir = os.path.abspath(os.path.join(windows_dir, '..', 'report'))
plot_path = os.path.join(windows_dir, 'regimes_plot.png')


plt.savefig(plot_path)



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

def get_regime_parameters(df, regime_col='regime', rank=1):
    """
    Calculates the expected returns and cleaned covariance matrix for each regime.
    """
    regimes = df[regime_col].unique()
    regime_params = {}
    
    features = ['log_ret', 'vrp', 'VIX_VIX3M_Ratio'] 
    
    for k in regimes:
        regime_data = df[df[regime_col] == k][features]
        
        # Expected Returns
        mu_k = regime_data.mean().values
        
        # Empirical Covariance 
        empirical_cov = regime_data.cov().values
        
        target_rank = min(rank, len(features) - 1) 
        
        if target_rank > 0:
            U, S, Vt = randomized_svd(empirical_cov, rank=target_rank)
            
            cleaned_cov = U @ np.diag(S) @ U.T
            
            np.fill_diagonal(cleaned_cov, np.diag(empirical_cov))
        else:
            cleaned_cov = empirical_cov
            
        regime_params[k] = {
            'mu': mu_k,
            'empirical_cov': empirical_cov,
            'cleaned_cov': cleaned_cov
        }
        
    return regime_params

regime_parameters = get_regime_parameters(df, rank=1)

# Display results
for k, params in regime_parameters.items():
    print(f"\n--- Regime {k} ---")
    print("Expected Returns (mu_k):")
    print(params['mu'])
    print("\nCleaned Covariance Matrix (Sigma_k_clean):")
    print(params['cleaned_cov'])