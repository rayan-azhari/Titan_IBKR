Volatility targeting serves as the primary risk-management and position-sizing mechanism within Rob Carver's core methodology, acting as the bridge between theoretical trading signals and actual capital allocation. By targeting a specific level of volatility, the system ensures that risk is evenly distributed across diverse asset classes and continuously adapts to changing market conditions.
The Mechanics of Position Sizing In Carver's framework, a position size is essentially determined by the formula: Position ~ Forecast × Volatility Target / Natural Volatility of the Instrument
.
Scale-Free Forecasts: Before targeting portfolio risk, individual trading rule forecasts (like the continuous trend-following signals discussed previously) are volatility-normalised by dividing the raw signal by the instrument's recent price volatility
. This strips out the instrument's specific units, meaning a forecast of +20 indicates a "strong buy" equally for S&P 500 futures as it does for Eurodollars
.
Targeting Expected Risk: The trader sets a long-term annualised risk target (e.g., 25%), which is divided by 16 (the approximate square root of 256 business days) to find the target daily cash risk
.
EWMA Over SMA: To measure the underlying instrument's volatility for these adjustments, Carver relies on an Exponentially Weighted Moving Average (EWMA)—typically using a 36-day center of mass—rather than a Simple Moving Average (SMA). He rejects SMAs because they are "super jumpy" as large returns abruptly enter and exit the moving window
.
Account-Level Volatility Targeting and "Half-Kelly" Volatility targeting also dictates how total portfolio capital is managed during drawdowns and winning streaks. Carver defines the rule that "if you lose 1% of your bankroll, then you should cut your risk by 1%" specifically as volatility targeting
. To find the optimal baseline risk target, Carver uses the Kelly criterion but strongly advocates for trading at "Half-Kelly"
. Because a trader can never be certain of their true expected Sharpe Ratio (due to the risk of overfitting or changing market conditions), running at a full Kelly optimal target can easily lead to catastrophic over-leveraging if the true Sharpe Ratio turns out to be lower than expected
.
The Danger of "Pure Evil" Low Volatility Because the volatility targeting formula dictates that position sizes must increase when an instrument's natural volatility decreases, instruments with exceptionally low volatility become highly dangerous
. Carver refers to low volatility as "pure evil" because it forces the system to apply enormous leverage to hit the desired risk target
. This extreme leverage exposes the portfolio to massive fat-tail events (kurtosis)—such as a central bank unexpectedly moving pegged interest rates. For this exact reason, Carver's methodology dictates that traders must actively avoid or stop trading contracts when their volatility drops too low, such as short-end German bonds (Schatz) or Swiss Francs during the peg
.
Ultimately, volatility targeting equalises the expected risk across the portfolio, though Carver notes that realised volatility will naturally fluctuate 50% of the time above and 50% of the time below this target as trading signals strengthen or weaken
.
NotebookLM can be inaccurate; please double-check its responses.