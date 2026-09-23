import os
import pandas as pd
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
data_dir = os.path.join(project_root, 'data')

vix_vix3m_path = os.path.join(data_dir, 'vix_vix3m_ratio_daily.xlsx')
vix_path = os.path.join(data_dir, 'vix_daily.xlsx')
spx_xndx_path = os.path.join(data_dir, 'spx_xndx.xlsx')
skew_path = os.path.join(data_dir, 'cboe_skew.xlsx')
rf_path = os.path.join(data_dir, 'riskfreerate.xlsx')
spybidask_path = os.path.join(data_dir, 'spybidask.xlsx')
stocks_path = os.path.join(data_dir, 'spx_coys.xlsx')

macro_output_path = os.path.join(data_dir, '01_cleaned_macro.csv')
if not os.path.exists(macro_output_path):
    print("Processing macro indicators...")
    vix_vix3m = pd.read_excel(vix_vix3m_path, skiprows=6, header=None, usecols=[0, 12], names=['Date', 'VIX_VIX3M_Ratio'], parse_dates=['Date'], index_col='Date')
    vix = pd.read_excel(vix_path, skiprows=6, header=None, usecols=[0, 1], names=['Date', 'VIX_Close'], parse_dates=['Date'], index_col='Date')
    spx = pd.read_excel(spx_xndx_path, skiprows=6, header=None, usecols=[0, 1], names=['Date', 'SPX_Close'], parse_dates=['Date'], index_col='Date')
    skew = pd.read_excel(skew_path, skiprows=6, header=None, usecols=[0, 1], names=['Date', 'Skew'], parse_dates=['Date'], index_col='Date')
    rf_rate = pd.read_excel(rf_path, skiprows=6, header=None, usecols=[1, 2], names=['Date', 'RF_Rate'], parse_dates=['Date'], index_col='Date')

    bidask = pd.read_excel(spybidask_path, skiprows=6, header=None, usecols=[0, 1, 2], names=['Date', 'Bid', 'Ask'], parse_dates=['Date'], index_col='Date')
    bidask['Bid_Ask_Spread'] = (bidask['Ask'] - bidask['Bid']) / ((bidask['Ask'] + bidask['Bid']) / 2)

    dfs = [vix_vix3m, vix, spx, skew, rf_rate, bidask[['Bid_Ask_Spread']]]
    cleaned_dfs = []

    for d in dfs:
        d = d[d.index.notnull()]
        d = d[~d.index.duplicated(keep='first')]
        d.index = pd.to_datetime(d.index, errors='coerce')
        d = d[d.index.notnull()]
        cleaned_dfs.append(d)

    macro_df = pd.concat(cleaned_dfs, axis=1).dropna()

    macro_df['SPX_Ret'] = np.log(macro_df['SPX_Close'] / macro_df['SPX_Close'].shift(1))
    macro_df['realized_var'] = macro_df['SPX_Ret'].rolling(window=21).var() * 252
    macro_df['implied_var'] = (macro_df['VIX_Close'] / 100) ** 2
    macro_df['vrp'] = macro_df['implied_var'] - macro_df['realized_var']
    macro_df['RF_Rate'] = pd.to_numeric(macro_df['RF_Rate'], errors='coerce')
    macro_df['Daily_RF'] = macro_df['RF_Rate'] / 100 / 252
    macro_df = macro_df.dropna()

    macro_df.to_csv(macro_output_path)
else:
    print("Found existing 01_cleaned_macro.csv, skipping macro extraction.")

raw_data = pd.read_excel(stocks_path, header=None)

stock_series_list = []


for start_col in range(10, raw_data.shape[1], 7):
    # Safety check
    if start_col + 4 >= raw_data.shape[1]:
        break
        
    ticker = str(raw_data.iloc[0, start_col]).strip()
    if ticker == 'nan' or not ticker:
        continue
        
    # Extract Dates (start_col) and PX_LAST (start_col + 4) starting from Row 3 (index 2)
    raw_dates = raw_data.iloc[2:, start_col].values
    raw_prices = raw_data.iloc[2:, start_col + 4].values
    
    # Clean the dates for this stock using vectorized parsing
    dt_series = pd.Series(raw_dates)
    is_num = pd.to_numeric(dt_series, errors='coerce')
    cleaned_dates = pd.Series(pd.NaT, index=dt_series.index)
    valid_num = (is_num >= 30000) & (is_num <= 60000)
    if valid_num.any():
        cleaned_dates[valid_num] = pd.to_datetime(is_num[valid_num], unit='D', origin='1899-12-30', errors='coerce')
    remain = ~valid_num & dt_series.notnull()
    if remain.any():
        cleaned_dates[remain] = pd.to_datetime(dt_series[remain], errors='coerce')
    cleaned_dates[(cleaned_dates < '2000-01-01') | (cleaned_dates > '2030-01-01')] = pd.NaT
            
    # Build a clean series for the stock
    s = pd.Series(raw_prices, index=cleaned_dates.values, name=ticker)
    
    # Drop empty rows and duplicate dates just for this stock
    s = s[s.index.notnull()]
    s = s[~s.index.duplicated(keep='first')]
    
    stock_series_list.append(s)

print(f"Extracted {len(stock_series_list)} individual assets. Merging timelines...")

price_df = pd.concat(stock_series_list, axis=1)
price_df.index = pd.DatetimeIndex(price_df.index).normalize()
price_df.index.name = 'Date'

price_df = price_df.apply(pd.to_numeric, errors='coerce')
stock_returns = np.log(price_df / price_df.shift(1))

output_path = os.path.join(data_dir, '01_stock_returns.csv')
stock_returns.to_csv(output_path)

