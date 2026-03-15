import json

import pandas as pd

df = pd.read_csv("data/databento/optimization_results.csv")
target_tickers = ["UNH", "WMT", "AMAT", "TXN", "CRM", "CSCO", "INTC"]

stats = []

for ticker in target_tickers:
    ticker_df = df[df["ticker"] == ticker]
    if ticker_df.empty:
        continue
    best = ticker_df.sort_values("score", ascending=False).iloc[0]

    total_trades = int(best["is_trades"] + best["oos_trades"])
    ret_is = best["is_return"] / 100.0
    ret_oos = best["oos_return"] / 100.0
    total_ret = ((1 + ret_is) * (1 + ret_oos)) - 1

    # Simulating $100k starting capital with 1% risk geometry
    final_equity = 100000 * (1 + total_ret)

    stats.append(
        {
            "Ticker": ticker,
            "Total Trades": total_trades,
            "Total Return (%)": round(total_ret * 100, 2),
            "Final Equity": round(final_equity, 2),
            "Starting Equity": 100000,
        }
    )

print(json.dumps(stats, indent=2))
