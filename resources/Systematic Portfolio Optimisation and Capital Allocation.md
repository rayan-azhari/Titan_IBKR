Within Rob Carver's core methodology, portfolio optimisation is the mathematical bridge that translates raw trading signals into actual capital allocations across different instruments and trading rules
,
. The primary objective is to maximize the portfolio's expected risk-adjusted returns (Sharpe ratio) while strictly controlling for the noise and uncertainty inherent in historical market data
,
.
The Rejection of Naive Markowitz Optimisation Carver completely rejects standard "one-period" or naive Markowitz mean-variance optimisation
,
. Because the naive method assumes that your inputs for expected returns and correlations are 100% accurate, it inevitably produces highly unstable and extreme portfolio weights that look great in a backtest but fail in live trading
,
. Furthermore, he demonstrates that using a short "rolling window" (e.g., looking at only the past one to five years of data) exacerbates this instability, leading to massive performance degradation out-of-sample
,
.
Carver's Preferred Optimisation Methods To deal with data uncertainty, Carver's methodology relies on three alternative methods:
Bootstrapping (The Primary Engine): This is Carver’s preferred quantitative method
. Bootstrapping is a non-parametric process that randomly samples historical returns with replacement—specifically using "block bootstrapping" to maintain the serial correlation of the markets
,
,
. By running hundreds of optimisations on these random samples and averaging the resulting weights, the system produces highly stable, non-extreme allocations that naturally account for data noise
,
.
Shrinkage: A Bayesian method that takes estimated Sharpe ratios and correlations and "shrinks" them toward a baseline prior (such as assuming all assets have equal expected Sharpe ratios and identical correlations)
,
. While it is computationally faster than bootstrapping, Carver cautions that it is harder to get the exact shrinkage factor right, requiring you to shrink mean estimates heavily when data is scarce
,
.
Handcrafting: A simple, heuristic-based method that groups assets manually based on correlations without complex algorithms, often yielding robust results that rival sophisticated computer optimisations
,
,
,
.
The Two-Stage Allocation Process Within the systematic framework, optimisation is cleanly separated into two distinct layers
:
Forecast Weights (Trading Rules): The system first determines how much weight to give each trading rule variation (e.g., combining fast EWMAC, slow EWMAC, and Carry rules)
. To achieve statistical significance, Carver strongly recommends pooling the pre-cost returns of all instruments together to create a massive, multi-century dataset
,
,
. Crucially, optimisation at this level must explicitly account for trading costs
,
. Carver's favored approach is to apply a "cost ceiling" to eliminate overly expensive, fast-trading rules from consideration, and then optimise the remaining rules based on their gross returns before adjusting the final weights to penalise the more expensive rules
,
.
Instrument Weights (Capital Allocation): Next, the system allocates capital across the different instrument subsystems
. For this, Carver prefers an expanding window (anchored fitting) rather than a rolling window
,
. This means using all available historical data from the start of the dataset up to the present moment, gradually incorporating new data as time passes
. If an instrument has a very short history, the bootstrapping algorithm naturally acts as a conservative Bayesian filter, pushing its weight toward a safe, average allocation until more data is gathered
.
Volatility Normalisation and Diversification Multipliers Before any optimisation occurs, the returns of the assets or rules are volatility-normalised, effectively equalising their expected risk and turning the covariance matrix into a correlation matrix
,
.
Because the optimised portfolio will contain assets and rules that are not perfectly correlated, the overall portfolio volatility will naturally drop below the trader's desired target
,
. To counteract this "loss" of volatility, the methodology applies Diversification Multipliers—specifically the Forecast Diversification Multiplier (FDM) and the Instrument Diversification Multiplier (IDM)
,
. These multipliers scale the positions back up, ensuring the trader fully reaps the mathematical benefits of diversification by achieving a higher expected return for the exact same level of target risk
,
,
.