# config.py

from datetime import date

# Paths
DATA_PATH = "./data/processed/"
MODEL_SAVE_PATH = "./results/model_weights/"
REPORTS_PATH = "./results/reports/"

# Ticker configuration
TICKER = "AAPL"

# Option chain parameters
OPTION_PARAMS = {
    "option_type": "calls",  # "calls" or "puts"
    "expiry_index": 0,       # Index of expiry from yfinance
}

# Optional: date range for reports/archival
DATE_RANGE = {
    "start": date(2023, 12, 1),
    "end": date(2023, 12, 28),
}
MARKET_PARAMS = {
    'symbol': 'AAPL',                    # Use 'NIFTY' if using NSE-based source
    'start': date(2025, 7, 1),           # Historical data fetch start
    'end': date(2025, 8, 1),             # Historical data fetch end
    'option_type': 'call',              # 'call' or 'put'
    'strike_price': None,               # Optional: Set a fixed strike to filter
    'expiry_date': date(2025, 8, 1),     # Target expiry
}
