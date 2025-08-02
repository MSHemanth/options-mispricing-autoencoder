import yfinance as yf
import pandas as pd
from datetime import date, datetime
from src.config import MARKET_PARAMS
import os
import numpy as np

def fetch_option_chain(symbol, expiry):
    ticker = yf.Ticker(symbol)
    opt_chain = ticker.option_chain(expiry)
    return opt_chain.calls  # We're only using Call options

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
    all_cleaned_data = []

    for expiry_date in MARKET_PARAMS['expiry_dates']:
        expiry = expiry_date.strftime('%Y-%m-%d')
        print(f"[INFO] Processing expiry: {expiry}")

        # Fetch option chain for current expiry
        raw_df = fetch_option_chain(symbol, expiry)
        print(f"[INFO] Raw options count for {expiry}: {len(raw_df)}")

        # Preprocess and annotate expiry
        cleaned_df = preprocess_options_data(raw_df,
                                             min_volume=1,
                                             require_bid_ask=True,
                                             otm_only=True)
        cleaned_df['expiry'] = expiry  # Add expiry as a new column
        print(f"[INFO] Filtered options count for {expiry}: {len(cleaned_df)}")

        all_cleaned_data.append(cleaned_df)

    # Combine all expiries
    if all_cleaned_data:
        combined_df = pd.concat(all_cleaned_data, ignore_index=True)
        print(f"[INFO] Total cleaned options across expiries: {len(combined_df)}")
        print(combined_df.head())

        # Save to a single CSV
        save_path = "./data/processed/cleaned_options.csv"
        save_to_csv(combined_df, save_path)
    else:
        print("[WARNING] No data collected. Exiting.")

def compute_time_to_expiry(expiry_date_str: str) -> float:
    """Returns time to expiry in years."""
    expiry_date = pd.to_datetime(expiry_date_str)
    today = datetime.today()
    return max((expiry_date - today).days / 365.0, 0.001)

def prepare_dataset_from_csv(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads cleaned options data and prepares feature matrix X and target y.

    Args:
        csv_path: Path to CSV file with options data.
    Returns:
        X: Feature matrix (numpy array)
        y: Target vector (mid price)
    """
    df = pd.read_csv(csv_path)

    # Ensure required columns exist
    required_cols = ['strike', 'bid', 'ask']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Compute mid-price (target)
    df['mid_price'] = (df['bid'] + df['ask']) / 2

    # Add features
    underlying_price = MARKET_PARAMS['underlying_price']
    expiry_date_str = MARKET_PARAMS['expiry_date'].strftime('%Y-%m-%d')

    df['underlying'] = underlying_price
    df['moneyness'] = df['underlying'] / df['strike']
    df['time_to_expiry'] = compute_time_to_expiry(expiry_date_str)

    # Define feature columns
    feature_cols = ['strike', 'underlying', 'moneyness', 'time_to_expiry']
    X = df[feature_cols].values
    y = df['mid_price'].values

    return X, y
if __name__ == "__main__":
    main()
