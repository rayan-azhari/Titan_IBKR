import os
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from research.orb.run_orb_oos_databento import download_databento_1yr

SP100_TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
    "META",
    "BRK-B",
    "LLY",
    "AVGO",
    "V",
    "JPM",
    "XOM",
    "UNH",
    "WMT",
    "MA",
    "JNJ",
    "PG",
    "HD",
    "ORCL",
    "COST",
    "ABBV",
    "CVX",
    "CRM",
    "BAC",
    "MRK",
    "KO",
    "NFLX",
    "AMD",
    "PEP",
    "LIN",
    "TMO",
    "ADBE",
    "WFC",
    "CSCO",
    "MCD",
    "ABT",
    "TMUS",
    "INTU",
    "IBM",
    "GE",
    "CAT",
    "QCOM",
    "TXN",
    "AMAT",
    "DHR",
    "VZ",
    "PFE",
    "PM",
    "ISRG",
    "NOW",
    "COP",
    "BA",
    "SPGI",
    "HON",
    "AMGN",
    "RTX",
    "UNP",
    "LOW",
    "INTC",
    "SYK",
    "GS",
    "NEE",
    "ELV",
    "BLK",
    "TJX",
    "PGR",
    "AXP",
    "MDT",
    "C",
    "LMT",
    "UBER",
    "VRTX",
    "CB",
    "REGN",
    "MMC",
    "ADI",
    "BSX",
    "CI",
    "CVS",
    "ZTS",
    "T",
    "FI",
    "SLB",
    "MDLZ",
    "MO",
    "BKNG",
    "GILD",
    "EOG",
    "BDX",
    "SO",
    "CME",
    "NOC",
    "CSX",
    "ITW",
    "DUK",
    "CL",
    "ICE",
    "FDX",
    "USB",
    "PNC",
]


def main():
    cache_dir = "data/databento"
    os.makedirs(cache_dir, exist_ok=True)

    downloaded = 0
    total = len(SP100_TICKERS)

    for ticker in SP100_TICKERS:
        cache_file = os.path.join(cache_dir, f"{ticker}_1yr_5m.csv")
        if os.path.exists(cache_file):
            print(f"Skipping {ticker}, already downloaded.")
            downloaded += 1
            continue

        print(f"[{downloaded + 1}/{total}] Downloading data for {ticker}...")
        try:
            download_databento_1yr(ticker)
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")

        # Optional delay to avoid any potential rate spikes
        time.sleep(0.5)
        downloaded += 1

    print("\nDownload process completed.")


if __name__ == "__main__":
    main()
