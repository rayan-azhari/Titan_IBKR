# Quant Portfolio Strategy: Multi-Layered Approach

This multi-layered approach—combining **Equity Co-integration**, **FX Carry**, and **Cross-Asset Spreads**—is precisely how professional quant portfolios are constructed. It is not about finding one perfect strategy; it is about layering uncorrelated, mathematically sound edges.

Let us critically examine the three proposed paths and determine the most effective immediate build.

---

## 1. The Cross-Asset Spread: USD/CAD vs. WTI Crude

This is the standout recommendation. It perfectly bridges your domain expertise in energy with a robust statistical arbitrage framework.

- **The Mechanism:** The Canadian economy is heavily reliant on crude oil exports. Therefore, the value of the Canadian Dollar (CAD) is intrinsically linked to the price of West Texas Intermediate (WTI) crude. When WTI rises, CAD generally appreciates (driving USD/CAD down).
- **The Edge:** This relationship is not a mere statistical artifact; it is grounded in global trade flows. When the spread between USD/CAD and WTI diverges significantly from its historical norm, it is highly likely to mean-revert, offering a powerful, non-directional trading opportunity.
- **The Implementation:** You can pull historical daily or hourly data for both instruments. You will then apply the **Johansen test** to confirm co-integration and construct a **Z-score model** to generate entry and exit signals.

---

## 2. The FX Carry Trade with Momentum Filter

This is a classic, slow-burn strategy that exploits interest rate differentials.

- **The Mechanism:** You buy a currency with a high interest rate and short a currency with a low interest rate, capturing the "carry" (the swap paid by the broker) while holding the position.
- **The Edge:** Central bank interest rates are public knowledge, but the momentum filter prevents you from holding a high-yielding currency that is in a structural downtrend (where capital depreciation would wipe out the yield).
- **The Implementation:** This requires a weekly script that scrapes central bank rates, ranks the G10 currencies, applies a simple momentum filter (e.g., price > 50-day SMA), and executes the rebalancing via the IBKR API. It is incredibly capital-efficient for a £10k account.

---

## 3. The Monday Gap Fade

This is a fascinating, niche anomaly based on liquidity mechanics.

- **The Mechanism:** FX markets close over the weekend. If significant news breaks, the market opens on Sunday night with a "gap" in price. Because liquidity is exceptionally thin during the Asian open, these gaps are often over-extended and tend to "fill" (revert to the Friday close) as European and US liquidity enters the market on Monday.
- **The Edge:** This is a purely structural inefficiency. You are betting that the initial weekend panic or exuberance will be smoothed out by larger institutional flows.
- **The Implementation:** This is highly automatable. The script simply checks the Sunday open against the Friday close. If the gap exceeds a defined threshold (e.g., > 0.5% ATR), it executes a trade targeting the Friday closing price, with a hard time-based exit before the end of the London session.

---

## Immediate Build: Dukascopy Ingestion & USD/CAD vs. WTI

Building the **Dukascopy data pipeline** is significantly faster and cheaper than constructing a full tick database for equities. It provides a clean, immediate win and allows you to start writing actual backtesting code in `vectorbt`. Furthermore, the USD/CAD vs. WTI spread leverages your energy background perfectly.

### Development Sprint Blueprint

1.  **The Data Ingestion Engine:** We will write a Python script using the `dukascopy` or `ddata` libraries to programmatically download, clean, and format 10 years of hourly OHLCV data for USD/CAD and a proxy for WTI (e.g., the USO ETF or a CFD, depending on what Dukascopy offers that aligns with IBKR).
2.  **The Co-integration Engine:** We will build a Jupyter Notebook (integrated into your VS Code environment) that takes this data, runs the **Augmented Dickey-Fuller** and **Johansen tests** using `statsmodels`, and calculates the rolling hedge ratio and Z-score.
3.  **The vectorbt Backtest:** We will feed the Z-score signals into a `vectorbt` portfolio simulation, rigorously accounting for IBKR's specific margin requirements, commissions, and swap rates for both legs of the spread.