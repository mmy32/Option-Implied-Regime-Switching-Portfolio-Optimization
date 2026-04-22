import os
import pandas as pd
import numpy as np

data_dir = '/mnt/c/Users/Owner/5370'

# File paths
vix_vix3m_path = os.path.join(data_dir, 'vix_vix3m_ratio_daily.xlsx')
vix_path = os.path.join(data_dir, 'vix_daily.xlsx')
spx_xndx_path = os.path.join(data_dir, 'spx_xndx.xlsx')
skew_path = os.path.join(data_dir, 'cboe_skew.xlsx')
spybidask_path = os.path.join(data_dir, 'spybidask.xlsx')
rf_path = os.path.join(data_dir, 'riskfreerate.xlsx')

# Load data
vix_vix3m = pd.read_excel(vix_vix3m_path, skiprows=6, header=None, usecols=[0, 12], names=['Date', 'VIX_VIX3M_Ratio'], parse_dates=['Date'], index_col='Date')
vix = pd.read_excel(vix_path, skiprows=6, header=None, usecols=[0, 1], names=['Date', 'VIX_Close'], parse_dates=['Date'], index_col='Date')
prices = pd.read_excel(spx_xndx_path, skiprows=6, header=None, usecols=[0, 1, 2], names=['Date', 'SPX_Close', 'NDX_Close'], parse_dates=['Date'], index_col='Date')
skew = pd.read_excel(skew_path, skiprows=6, header=None, usecols=[0, 1], names=['Date', 'Skew'], parse_dates=['Date'], index_col='Date')

# Read Bid and Ask columns (0, 1, 2) and compute proportional spread
bidask = pd.read_excel(spybidask_path, skiprows=6, header=None, usecols=[0, 1, 2], names=['Date', 'Bid', 'Ask'], parse_dates=['Date'], index_col='Date')
bidask['Bid_Ask_Spread'] = (bidask['Ask'] - bidask['Bid']) / ((bidask['Ask'] + bidask['Bid']) / 2)

# Skip the blank first column in riskfreerate
rf_rate = pd.read_excel(rf_path, skiprows=6, header=None, usecols=[1, 2], names=['Date', 'RF_Rate'], parse_dates=['Date'], index_col='Date')

dfs = [vix_vix3m, vix, prices, skew, bidask[['Bid_Ask_Spread']], rf_rate]
cleaned_dfs = []

for d in dfs:
    # Drop missing dates
    d = d[d.index.notnull()]
    # Remove duplicate dates, keeping only the first instance
    d = d[~d.index.duplicated(keep='first')]
    d.index = pd.to_datetime(d.index, errors='coerce')
    d = d[d.index.notnull()]
    cleaned_dfs.append(d)

df = pd.concat(cleaned_dfs, axis=1).dropna()

# Asset returns
df['SPX_Ret'] = np.log(df['SPX_Close'] / df['SPX_Close'].shift(1))
df['NDX_Ret'] = np.log(df['NDX_Close'] / df['NDX_Close'].shift(1))

# Variance Risk Premium calculation
df['realized_var'] = df['SPX_Ret'].rolling(window=21).var() * 252
df['implied_var'] = (df['VIX_Close'] / 100) ** 2
df['vrp'] = df['implied_var'] - df['realized_var']

df['RF_Rate'] = pd.to_numeric(df['RF_Rate'], errors='coerce')
df['Daily_RF'] = df['RF_Rate'] / 100 / 252

df = df.dropna()
output_path = os.path.join(data_dir, '01_cleaned_data.csv')
df.to_csv(output_path)
print(f"Data prepped and saved to {output_path}")