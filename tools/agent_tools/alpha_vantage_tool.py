"""Tool for interacting with the Alpha Vantage API."""

import os
import time
from alpha_vantage.fundamentaldata import FundamentalData
from alpha_vantage.timeseries import TimeSeries
from langchain_core.tools import tool
import pandas as pd

# IMPORTANT: Set your Alpha Vantage API key as an environment variable
# You can get a free key from: https://www.alphavantage.co/support/#api-key
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "YOUR_API_KEY") # Replace YOUR_API_KEY or set env var

# Global client initialization (consider rate limiting for free tier - 5 calls/min)
fd = FundamentalData(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')

# Rate limiting helper (very basic)
# A more robust solution would use a library like 'ratelimit'
_last_api_call_time = 0
_api_call_interval = 15 # Slightly more than 12 seconds for 5 calls/min limit

def _rate_limit_pause():
    """Pauses execution if called too frequently."""
    global _last_api_call_time
    now = time.time()
    elapsed = now - _last_api_call_time
    if elapsed < _api_call_interval:
        pause_time = _api_call_interval - elapsed
        print(f"Alpha Vantage rate limit: Pausing for {pause_time:.2f} seconds...")
        time.sleep(pause_time)
    _last_api_call_time = time.time()

@tool("get_company_overview", return_direct=False)
def get_company_overview(symbol: str):
    """Fetches company overview data (description, sector, industry, P/E, EPS, etc.) from Alpha Vantage."""
    _rate_limit_pause()
    try:
        data, _ = fd.get_company_overview(symbol=symbol)
        # Convert DataFrame to a more readable string format
        overview_str = f"Company Overview for {symbol}:\n"
        for index, value in data.iloc[0].items(): # Assuming single row DataFrame
            overview_str += f"- {index}: {value}\n"
        return overview_str
    except Exception as e:
        return f"Error fetching company overview for {symbol} from Alpha Vantage: {e}"

@tool("get_income_statement", return_direct=False)
def get_income_statement(symbol: str):
    """Fetches the latest annual income statement data from Alpha Vantage."""
    _rate_limit_pause()
    try:
        data, _ = fd.get_income_statement_annual(symbol=symbol)
        latest_income = data.iloc[:, 0] # Get the most recent column (latest year)
        income_str = f"Latest Annual Income Statement for {symbol} (Alpha Vantage):\n"
        for index, value in latest_income.items():
            # Simple formatting for currency
            val_str = f"${float(value)/1e6:.2f}M" if pd.notna(value) and float(value) > 1000 else str(value)
            income_str += f"- {index}: {val_str}\n"
        return income_str
    except Exception as e:
        return f"Error fetching income statement for {symbol} from Alpha Vantage: {e}"

# Add similar tools for get_balance_sheet_annual and get_cash_flow_annual if needed

@tool("get_latest_quote", return_direct=False)
def get_latest_quote(symbol: str):
    """Fetches the latest quote (price, volume, change) from Alpha Vantage."""
    _rate_limit_pause()
    try:
        data, _ = ts.get_quote_endpoint(symbol=symbol)
        quote_str = f"Latest Quote for {symbol} (Alpha Vantage):\n"
        for key, value in data.items():
             # Clean up keys from API response (e.g., '01. symbol' -> 'symbol')
             clean_key = key.split('. ')[1].replace(' ', '_') if '. ' in key else key
             quote_str += f"- {clean_key}: {value}\n"
        return quote_str
    except Exception as e:
        return f"Error fetching latest quote for {symbol} from Alpha Vantage: {e}"

# Example Usage (optional)
if __name__ == '__main__':
    if ALPHA_VANTAGE_API_KEY == "YOUR_API_KEY":
        print("Please set your ALPHA_VANTAGE_API_KEY environment variable or replace YOUR_API_KEY in the script.")
    else:
        symbol = "IBM"
        print(get_company_overview.invoke(symbol))
        print("\n")
        print(get_income_statement.invoke(symbol))
        print("\n")
        print(get_latest_quote.invoke(symbol)) 