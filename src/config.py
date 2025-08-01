from datetime import date

# Paths
DATA_PATH = "./data/processed/"
MODEL_SAVE_PATH = "./results/model_weights/"
REPORTS_PATH = "./results/reports/"

# Ticker configuration (used for sources like yfinance)
TICKER = "TSLA"  # NVIDIA ticker

# Option chain parameters (for yfinance)
OPTION_PARAMS = {
    "option_type": "calls",  # "calls" or "puts"
    "expiry_index": 0,       # Only used if not specifying expiry directly
}

# Date range for fetching historical price data and reports
DATE_RANGE = {
    "start": date(2025, 7, 1),
    "end": date(2025, 8, 1),
}

# Main market data fetch configuration
MARKET_PARAMS = {
    'symbol': 'TSLA',  # Dow Jones Industrial Average
    'start': DATE_RANGE["start"],
    'end': DATE_RANGE["end"],
    'option_type': 'call',           # 'call' or 'put'
    'strike_price': None,            # Optional: set a fixed strike to filter
    'expiry_dates': [                # 🔁 List of expiry dates to process
        date(2025, 8, 1),
        date(2025, 8, 8),
        date(2025, 8, 15),
        date(2025, 8, 22),
        date(2025, 8, 29),
    ]
}
