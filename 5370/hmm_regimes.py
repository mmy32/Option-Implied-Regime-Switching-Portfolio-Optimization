import os
import pandas as pd
import numpy as np
from hmmlearn import hmm
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

data_dir = '/mnt/c/Users/Owner/5370'
input_path = os.path.join(data_dir, '01_cleaned_macro.csv')

print("Loading macro data...")
df = pd.read_csv(input_path, index_col='Date', parse_dates=True)

features = ['VIX_VIX3M_Ratio', 'vrp']
X = df[features].values

initial_train_days = 756

# We retrain the model every 21 trading days (1 month) to keep execution time fast
retrain_freq = 21

predicted_regimes = np.full(len(df), np.nan)

print(f"Starting out-of-sample expanding window HMM. Total days: {len(df)}")
print(f"Initial training window: {initial_train_days} days. Retraining every {retrain_freq} days.")

current_model = None

for t in tqdm(range(initial_train_days, len(df)), desc="HMM Expanding Window"):
    if (t - initial_train_days) % retrain_freq == 0:
        X_train = X[:t]
        
        # Instantiate a fresh model to prevent memory bleed
        model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=100, random_state=42)
        
        try:
            model.fit(X_train)
            
            # We must force State 1 to ALWAYS be the "Stress" regime.
            if model.means_[0, 0] > model.means_[1, 0]:
                model.means_ = model.means_[[1, 0]]
                model.covars_ = model.covars_[[1, 0]]
                model.transmat_ = model.transmat_[[1, 0], :][:, [1, 0]]
                model.startprob_ = model.startprob_[[1, 0]]
                
            current_model = model
        except Exception:
            pass
            
    # Predict the regime for the current day t using our working model
    if current_model is not None:
        X_eval = X[:t+1]
        try:
            _, states = current_model.decode(X_eval)
            predicted_regimes[t] = states[-1]
        except Exception:
            pass

df['regime'] = predicted_regimes

# Drop the initial 3-year training window since we cannot trade it out-of-sample
out_of_sample_df = df.dropna(subset=['regime']).copy()

output_path = os.path.join(data_dir, '02_regimes_data.csv')
out_of_sample_df.to_csv(output_path)

print(f"Strictly out-of-sample regimes saved! Ready for covariance cleaning on {len(out_of_sample_df)} days.")