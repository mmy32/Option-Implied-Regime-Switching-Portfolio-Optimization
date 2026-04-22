import os
import pandas as pd
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt

# Get the directory where the script is running
current_dir = os.getcwd()
all_files = os.listdir(current_dir)

# Dynamically find the files based on partial names
vix_vix3m_path = next((f for f in all_files if 'vix_vix3m' in f.lower() and f.endswith('.xlsx')), None)
vix_path = next((f for f in all_files if 'vix_daily' in f.lower() and f.endswith('.xlsx')), None)
spx_path = next((f for f in all_files if 'spx' in f.lower() and f.endswith('.xlsx')), None)

# 1. Load the data using read_excel
vix_vix3m = pd.read_excel(vix_vix3m_path, skiprows=6, header=None, usecols=[0, 12], 
                        names=['Date', 'VIX_VIX3M_Ratio'], parse_dates=['Date'], index_col='Date')

vix = pd.read_excel(vix_path, skiprows=6, header=None, usecols=[0, 1], 
                  names=['Date', 'VIX_Close'], parse_dates=['Date'], index_col='Date')

spx = pd.read_excel(spx_path, skiprows=6, header=None, usecols=[0, 1], 
                  names=['Date', 'SPX_Close'], parse_dates=['Date'], index_col='Date')

# 2. Clean and align data
df = pd.concat([vix_vix3m, vix, spx], axis=1).dropna()

# 3. Calculate Realized Variance and Variance Risk Premium
df['log_ret'] = np.log(df['SPX_Close'] / df['SPX_Close'].shift(1))
df['realized_var'] = df['log_ret'].rolling(window=21).var() * 252
df['implied_var'] = (df['VIX_Close'] / 100) ** 2
df['vrp'] = df['implied_var'] - df['realized_var']
df = df.dropna()

# 4. Fit the Hidden Markov Model
X = df[['VIX_VIX3M_Ratio', 'vrp']].values

# Using n_init and adjusting tol slightly to help with the convergence warning
model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=1000, tol=0.01, random_state=42)
model.fit(X)

df['regime'] = model.predict(X)

# 5. Plot the SPX overlayed with Regimes
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

# Save the plot directly to your Windows directory using the WSL mount path
windows_dir = '/mnt/c/Users/Owner/5370'
plot_path = os.path.join(windows_dir, 'regimes_plot.png')

# Create the directory if it doesn't exist just in case
os.makedirs(windows_dir, exist_ok=True)

plt.savefig(plot_path)
print(f"Plot successfully saved to your Windows folder at: {plot_path}")

import numpy as np
import pandas as pd

def randomized_svd(A, rank, n_oversamples=5, n_iter=2):
    """
    Computes the randomized SVD of matrix A.
    """
    M, N = A.shape
    
    # 1. Generate a random Gaussian matrix
    Omega = np.random.randn(N, rank + n_oversamples)
    
    # 2. Form the sketch of A
    Y = A @ Omega
    
    # 3. Perform power iterations to separate the singular values
    for _ in range(n_iter):
        Y = A @ (A.T @ Y)
        
    # 4. Find an orthogonal basis for the sketch
    Q, _ = np.linalg.qr(Y)
    
    # 5. Project A onto this lower-dimensional space
    B = Q.T @ A
    
    # 6. Compute the deterministic SVD on the small matrix B
    U_tilde, S, Vt = np.linalg.svd(B, full_matrices=False)
    
    # 7. Recover the left singular vectors of A
    U = Q @ U_tilde
    
    # Truncate to the desired target rank
    return U[:, :rank], S[:rank], Vt[:rank, :]

def get_regime_parameters(df, regime_col='regime', rank=1):
    """
    Calculates the expected returns and cleaned covariance matrix for each regime.
    """
    regimes = df[regime_col].unique()
    regime_params = {}
    
    # Select only the numerical features for covariance
    # When you add stock constituents, replace this with your asset columns
    features = ['log_ret', 'vrp', 'VIX_VIX3M_Ratio'] 
    
    for k in regimes:
        # Isolate data for regime k
        regime_data = df[df[regime_col] == k][features]
        
        # 1. Expected Returns (mu_k)
        mu_k = regime_data.mean().values
        
        # 2. Empirical Covariance (Sigma_k)
        empirical_cov = regime_data.cov().values
        
        # 3. Cleaned Covariance (Sigma_k_clean) via Randomized SVD
        # Using a target rank. For a large stock portfolio, you might use rank 3 to 10 (market, size, value factors)
        target_rank = min(rank, len(features) - 1) 
        
        if target_rank > 0:
            U, S, Vt = randomized_svd(empirical_cov, rank=target_rank)
            
            # Reconstruct the cleaned matrix: U * S * U^T (since covariance is symmetric)
            cleaned_cov = U @ np.diag(S) @ U.T
            
            # Common quantitative adjustment: restore the original empirical variances to the diagonal
            np.fill_diagonal(cleaned_cov, np.diag(empirical_cov))
        else:
            cleaned_cov = empirical_cov
            
        regime_params[k] = {
            'mu': mu_k,
            'empirical_cov': empirical_cov,
            'cleaned_cov': cleaned_cov
        }
        
    return regime_params

# Assuming 'df' is the dataframe from your previous script
# Execute the parameter extraction
regime_parameters = get_regime_parameters(df, rank=1)

# Display the results
for k, params in regime_parameters.items():
    print(f"\n--- Regime {k} ---")
    print("Expected Returns (mu_k):")
    print(params['mu'])
    print("\nCleaned Covariance Matrix (Sigma_k_clean):")
    print(params['cleaned_cov'])