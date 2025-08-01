import pandas as pd
from datetime import datetime
from config import DATA_PATH, MARKET_PARAMS
import yfinance as yf
from pathlib import Path



def load_cleaned_data(path="./data/processed/cleaned_options.csv"):
    return pd.read_csv(path)


def get_spot_price(symbol):
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="1d")
    return hist['Close'].iloc[-1] if not hist.empty else None


def compute_mid_price(row):
    return (row['ask'] + row['bid']) / 2


def compute_time_to_expiry(expiry_date):
    if expiry_date is None:
        return None
    today = datetime.today()
    return (expiry_date - today).days / 365.0


import re
from datetime import datetime

def extract_expiry(symbol):
    # US format: AAPL250801C00205000
    match = re.search(r'(\d{6})[CP]', symbol)
    if match:
        try:
            return datetime.strptime(match.group(1), "%y%m%d")
        except ValueError:
            print(f"[WARN] Failed to parse expiry for: {symbol}")
            return None
    else:
        print(f"[WARN] No expiry found in: {symbol}")
        return None




def engineer_features(df, spot_price):
    if spot_price is None:
        raise ValueError("Spot price is None. Cannot compute moneyness.")

    df['moneyness'] = (df['strike'] - spot_price) / spot_price
    df['iv_oi_ratio'] = df['impliedVolatility'] / (df['openInterest'] + 1e-5)
    df['volume_oi_ratio'] = df['volume'] / (df['openInterest'] + 1e-5)
    df['mid_price'] = df.apply(compute_mid_price, axis=1)
    df['expiry'] = df['contractSymbol'].apply(extract_expiry)
    df['time_to_expiry'] = df['expiry'].apply(compute_time_to_expiry)

    return df


def get_feature_matrix(df):
    return df[['strike', 'impliedVolatility', 'moneyness', 'time_to_expiry',
               'mid_price', 'volume', 'openInterest']].values


if __name__ == "__main__":
    df = load_cleaned_data().copy()
    spot_price = get_spot_price(MARKET_PARAMS['symbol'])
    print(f"[INFO] Using spot price: {spot_price}")

    df_feat = engineer_features(df, spot_price)
    print(f"[INFO] Extracted expiry dates for {df_feat['expiry'].notnull().sum()} contracts")

    # Optional cleaning step
    df_feat = df_feat[df_feat['expiry'].notnull()]
    df_feat = df_feat[df_feat['mid_price'].notnull()]

    output_csv = str(Path(DATA_PATH) / "features.csv")
    df_feat.to_csv(output_csv, index=False)
    print(f"[INFO] Features saved to {output_csv}")

    X = get_feature_matrix(df_feat)
    print("[INFO] Feature matrix shape:", X.shape)
    print("[INFO] First 5 rows:\n", X[:5])

