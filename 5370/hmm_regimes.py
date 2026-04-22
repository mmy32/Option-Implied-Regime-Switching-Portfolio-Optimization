import os
import pandas as pd
from hmmlearn import hmm

data_dir = '/mnt/c/Users/Owner/5370'
input_path = os.path.join(data_dir, '01_cleaned_data.csv')
df = pd.read_csv(input_path, parse_dates=['Date'], index_col='Date')

X = df[['VIX_VIX3M_Ratio', 'vrp', 'Skew']].values

model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=1000, tol=0.01, random_state=42)
model.fit(X)

df['regime'] = model.predict(X)
probs = model.predict_proba(X)
df['prob_regime_0'] = probs[:, 0]
df['prob_regime_1'] = probs[:, 1]

output_path = os.path.join(data_dir, '02_regimes_data.csv')
df.to_csv(output_path)
print(f"Regimes inferred and saved to {output_path}")