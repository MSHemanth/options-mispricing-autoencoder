import yfinance as yf
import pandas as pd
from datetime import date
from config import MARKET_PARAMS
import os
from data_loader import fetch_option_chain  # ✅ Correct function import

def preprocess_options_data(df, min_volume=1, require_bid_ask=True, otm_only=False):
    # Remove options with zero volume
    df = df[df['volume'] >= min_volume]

    # Remove options with 0.0 bid/ask (illiquid contracts)
    if require_bid_ask:
        df = df[(df['bid'] > 0) & (df['ask'] > 0)]

    # Optional: Filter to only Out-of-the-Money (OTM) options
    if otm_only:
        df = df[~df['inTheMoney']]

    # Drop unnecessary columns
    drop_cols = ['contractSize', 'currency']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    return df.reset_index(drop=True)

def save_to_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[INFO] Saved cleaned options data to: {path}")

def main():
    symbol = MARKET_PARAMS['symbol']
    expiry = MARKET_PARAMS['expiry_date'].strftime('%Y-%m-%d')
    print(f"[INFO] Using expiry: {expiry}")

    raw_df = fetch_option_chain(symbol, expiry)  # ✅ Pass correct arguments
    print(f"[INFO] Raw options count: {len(raw_df)}")

    cleaned_df = preprocess_options_data(raw_df,
                                         min_volume=1,
                                         require_bid_ask=True,
                                         otm_only=True)
    print(f"[INFO] Filtered options count: {len(cleaned_df)}")
    print(cleaned_df.head())

    # Optional: Save cleaned data
    save_path = "./data/processed/cleaned_options.csv"
    save_to_csv(cleaned_df, save_path)

if __name__ == "__main__":
    main()
