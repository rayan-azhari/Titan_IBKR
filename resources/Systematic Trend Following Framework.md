Trend following acts as the primary engine within Rob Carver’s core systematic methodology, contributing approximately 60% of the overall risk in his automated futures trading system
. Rather than relying on a single, fragile signal, Carver’s framework builds robustness by applying multiple variations of trend following alongside strict risk management, continuous forecasting, and diversification rules.
Implementation via EWMAC and Breakouts Within the core methodology, trend following is primarily executed using the Exponentially Weighted Moving Average Crossover (EWMAC) rule, which is run simultaneously across multiple speeds (fast, medium, and slow lookbacks) to capture trends of varying lengths
. Carver supplements this with a "Breakout" rule, which functions similarly to a stochastic oscillator by measuring where a price sits within its recent historical maximum and minimum range
. Because different trend indicators (like EWMAC and Breakouts) are highly but not perfectly correlated, combining them provides a small but valuable layer of internal diversification
.
Furthermore, Carver applies trend following not just to individual instruments, but also to synthetic asset classes (Aggregate Momentum) and normalized price series, stripping out noise to allow the trend filters to work on cleaner data
.
Continuous and Volatility-Scaled Forecasts A defining characteristic of Carver's core methodology is that trend-following signals are continuous rather than binary
. Instead of merely triggering a static "buy" or "sell" alert, the rules calculate the distance between moving averages or ranges to generate a dynamic forecast
.
To make these signals perfectly comparable across vastly different assets—from corn to equity indices—the forecasts are volatility-normalized (made scale-free) and multiplied by a scalar so that their expected average absolute value is exactly 10, with an absolute cap of +/-20
. This continuous scaling ensures the system naturally scales up positions when trends are strong, and mathematically scales them down when trends are weak or when market volatility spikes
.
Performance Characteristics and Regime Sensitivity Trend following is a positive skew strategy, meaning it typically experiences many small losses (a slow "bleed") offset by occasional, massive gains during extended market moves
. However, Carver notes that trend-following returns tend to be negatively autocorrelated, meaning the strategy struggles significantly in choppy, sideways markets
.
For instance, Carver's system saw spectacular trend-following performance during the bond rallies of 2014, but the strategy performed poorly during the whipsaw markets of 2011–2013
. Furthermore, his meta-prediction research shows that trend following—especially the slowest rule variations—performs particularly poorly in rising interest rate environments. This is because the conflicting forces of positive carry and falling prices create a highly choppy total return series that generates continuous false signals
.
The Necessity of Diversification and Cost Control Because of these inherent weaknesses, Carver's core methodology dictates that trend following must never be traded in isolation
. He pairs his trend-following models with a Carry rule (which profits from the shape of the futures curve), cross-sectional Mean Reversion (which bets against short-term asset outperformance), and a systematic Short Volatility bias
. These non-trend rules act as crucial "insurance policies" that generate returns when trends break down
.
Finally, because fast trend-following rules can generate excessive turnover, Carver's methodology mitigates execution costs through position inertia or "buffering"
. The system will only execute a trade if the theoretical optimal position deviates from the current actual position by a set threshold (e.g., 10%), ensuring that the trend-following engine does not bleed capital through over-trading
.