"""Fetch the last 10 years of Federal Funds Rate (FEDFUNDS) from FRED."""

import os
import sys
from datetime import date, timedelta

import pandas as pd
import requests

FRED_API_KEY = os.environ.get("FRED_API_KEY")
if not FRED_API_KEY:
    sys.exit("Error: set the FRED_API_KEY environment variable before running.")

SERIES_ID = "FEDFUNDS"
END_DATE = date.today()
START_DATE = END_DATE - timedelta(days=365 * 10)

url = "https://api.stlouisfed.org/fred/series/observations"
params = {
    "series_id": SERIES_ID,
    "api_key": FRED_API_KEY,
    "file_type": "json",
    "observation_start": START_DATE.isoformat(),
    "observation_end": END_DATE.isoformat(),
}

response = requests.get(url, params=params, timeout=10)
response.raise_for_status()

data = response.json()
observations = data.get("observations", [])
if not observations:
    sys.exit("No observations returned.")

df = pd.DataFrame(observations)[["date", "value"]]
df["date"] = pd.to_datetime(df["date"])
df["value"] = pd.to_numeric(df["value"], errors="coerce")
df = df.rename(columns={"value": "fed_funds_rate_%"})
df = df.set_index("date").sort_index()

print(f"Federal Funds Rate — last 10 years ({START_DATE} to {END_DATE})")
print(f"Total observations: {len(df)}\n")
print(df.head())
