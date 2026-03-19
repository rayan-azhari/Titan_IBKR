# Innovative Approaches for IC Signal Discovery

The current IC pipeline is a very solid, institutional-grade foundation. By using Spearman Rank Correlation and ICIR across multiple horizons, you are already filtering out 90% of the noise that trips up retail algorithmic traders.

However, the current pipeline assumes that the relationship between an indicator and future returns is mostly **monotonic** (higher signal = higher return) and **unconditional** (it works the same way all the time).

To find stronger, more elusive alpha, you need to look for signals that are non-linear, regime-dependent, or structurally different. Here are the most effective, innovative approaches you could take to upgrade your signal discovery:

---

## 1. From Linear IC to Mutual Information & SHAP (Non-Linearity)
Spearman IC only measures monotonic relationships. However, some of the best signals are non-linear. For instance, an RSI of 50 vs 60 might have zero predictive difference, but 85 vs 95 might have massive predictive power.

*   **Mutual Information (MI):** Instead of IC, compute the Mutual Information score between your signals and forward returns. MI captures any relationship, linear or non-linear, and is standard in modern ML feature selection.
*   **SHAP Values via LightGBM:** Train a shallow Gradient Boosting model (like LightGBM) to predict forward returns using your 52 signals. Then, use SHAP (SHapley Additive exPlanations) to see which features the tree actually relies on. Trees are exceptional at finding non-linear "sweet spots" that IC misses completely.

## 2. Regime-Conditional Information Coefficients
A signal often has an IC of 0.0 over 10 years, not because it's noise, but because it has an IC of +0.08 in trending markets and -0.08 in ranging markets.

*   **Conditioning:** Split your dataset into regimes (e.g., Volatility > 70th percentile vs. Volatility < 30th percentile, or using a Hidden Markov Model).
*   **Conditional IC:** Compute the IC for your signals only within those specific regimes. You will frequently find that "useless" indicators suddenly show massive predictive power when filtered by a regime (e.g., Mean Reversion group signals when ADX is very low).

## 3. Fractional Differentiation (Preserving Memory)
Currently in Group E (Acceleration), you are using `.diff(1)` to make indicators stationary. Standard first-order differencing makes data stationary but destroys its "memory" (the long-term context of where the price came from).

*   **The Upgrade:** Use **Fractional Differencing** (popularized by Marcos Lopez de Prado). Instead of differencing by 1.0 (which destroys memory) or 0.0 (which is non-stationary price), you difference by a fraction like 0.4. This achieves stationarity (required for models) while retaining the maximum possible long-term memory. It is considered a massive edge in quant finance.

## 4. Cross-Asset & Relative Signals (The "Pairs" Component)
Forex is a highly interconnected graph. EUR/USD doesn't move in a vacuum; it moves because the USD index is moving, or because EUR/GBP is exhibiting flow.

*   **Lead-Lag Correlation:** Don't just compute the IC of EUR/USD signals on EUR/USD returns. Compute the IC of GBP/USD or USD/JPY momentum signals on EUR/USD forward returns.
*   **Cross-Sectional Momentum:** Instead of absolute momentum (e.g., `roc_10` > 0), use relative rank across all 6 pairs (e.g., rank(EUR/USD `roc_10`) against the other 5 pairs).

## 5. Microstructural Data
Your current signals use open, high, low, close. If you have access to higher-resolution data from Databento (e.g., tick data, M1 data, or order-book data), you can build microstructural features that have way higher ICs than daily Moving Averages.

*   **Volume-Weighted metrics:** VWAP deviations.
*   **Order Flow Imbalance (OFI):** If bid-ask volume data is available, tracking the pressure differential is highly predictive at horizons `h=1` or `h=5`.
*   **Intrabar Volatility:** Measuring the exact path the price took to get from Open to Close, rather than just using the 4 summary points.

## 6. Genetic Programming / Symbolic Regression
Instead of manually typing formulas for `trend_vol_adj`, you can have an algorithm discover formulas mechanically.

Using libraries like `gplearn` (Genetic Programming), you provide raw inputs (Open, High, Low, Close, Volume) and mathematical operators (+, -, *, /, log, sin, rolling_mean).
The algorithm iteratively "breeds" millions of complex, bizarre indicator formulas, measures their IC against forward returns, and keeps the ones that survive out-of-sample. It often finds incredibly unintuitive but robust mathematical relationships that humans wouldn't think of.

---

> [!TIP]
> **Recommendation on where to start:** If you want the highest ROI for your time, I highly recommend implementing **Regime-Conditional ICs**. You already have Volatility (Group D) signals. Simply write a script that splits your data where `adx_14 > 25` (Trending) vs `adx_14 < 20` (Ranging) and recalculates the leaderboard. It is the easiest way to unlock alpha from the 52 signals you already have programmed.