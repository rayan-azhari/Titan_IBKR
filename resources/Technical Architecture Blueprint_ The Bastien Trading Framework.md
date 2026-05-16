# Technical Architecture Blueprint: The Bastien Trading Framework

# Technical Architecture Blueprint: The Bastien Trading Framework

## 1. Architectural Philosophy and Design Principles

The Bastien Framework is predicated on a "Universal" approach to quantitative strategy architecture. The system operates on the core axiom that price action is the ultimate ground truth, necessitating an ingestion layer that makes zero assumptions about the underlying asset classes. The architecture prioritizes the mathematical properties of price movement—specifically divergence and convergence—over economic factor identification.

### Core Design Axioms
\*   \*\*Price Universality:\*\* The methodology assumes that trend-following mechanics are invariant across commodities, equities, bonds, and FX. The algorithmic core remains constant; only the ticker changes.
\*   \*\*Long Volatility Nature of Trend:\*\* Trend following is architected as a "Long Volatility" strategy. It seeks to capture range extension and volatility expansion (e.g., Donchian channel expansion) rather than mean reversion.
\*   \*\*System-Based Diversification:\*\* The framework rejects traditional instrument-based diversification as "intellectually unsatisfying." Diversity is enforced through the interaction of distinct system logics (e.g., combining trend following with relative value and market neutral crypto) rather than simply increasing ticker count.
\*   \*\*Rejection of Correlation-Based Risk:\*\* Traditional correlation metrics are discarded as "meaningless." In the Bastien Framework, correlation is viewed as a fickle metric that only functions when not needed and inevitably converges to 1.0 during "liquidation events," exactly when diversification is most critical.

## 2. The Global Strategy Hierarchy (Structural Overview)

The architecture utilizes a modular stack designed for industrial-scale reliability. The "Global Strategy" serves as the command apex, managing four primary sub-pillars:

\*   \*\*Global Strategy\*\*
    \*   \*\*CTA Component:\*\* A multi-asset pillar executing both idiosyncratic (single-instrument) and relative value (spread-based) trend following.
    \*   \*\*Tactical Asset Allocation (TAA):\*\* A long-only momentum engine designed to capture capital flows between major asset classes.
    \*   \*\*Relative Value Volatility:\*\* A market-neutral pillar exploiting spreads between short-term and mid-term volatility.
    \*   \*\*Market Neutral Crypto:\*\* A relative value momentum strategy focusing on the top 15 liquid crypto-assets with intraday execution capabilities.

## 3. The CTA Sub-Strategy Architecture

The CTA component is organized as a "Strategy Tree." Signal and risk flow from the broad asset layer down to the "Leaf" nodes (specific parameter sets) to ensure maximum ensemble stability.

1.  \*\*Asset Class Layer:\*\* The ingestion layer buckets assets into four idiosyncratic domains: Commodities, Equities, Bonds, and FX.
2.  \*\*Strategy Style Layer:\*\* Assets are processed via two distinct styles: \*\*Pure Idiosyncratic Trend\*\* (tracking single instruments) and \*\*Relative Value Trend\*\* (tracking the performance of spreads between instruments).
3.  \*\*Model Layer:\*\* Specific algorithmic implementations (e.g., breakout or momentum models) are deployed within each style.
4.  \*\*Parameter Set Layer (The Leaf Nodes):\*\* To mitigate "Single Point of Failure" risk, every model utilizes an ensemble of varied parameter sets. No individual parameter set is permitted to dominate the signal.

## 4. Bottom-Up Dynamic Risk Allocation Engine

The framework implements a Hierarchical Risk Parity (HRP) engine. Unlike the static capital allocation seen in "Turtle-style" systems, this engine prioritizes the \*\*Stability of Risk\*\*. It functions like a "flock of birds," where positions and weights expand and contract in unison based on real-time volatility data.

### Technical Aggregation Process
Risk is allocated as a "budget" rather than a cash percentage. The process begins at the Leaf nodes (Parameter Sets), where individual risk profiles are generated. These profiles are aggregated upward through the hierarchy. At every level, the system rebalances to ensure no single position or sub-strategy dominates the global portfolio. If a strategy's realized risk increases, the HRP engine automatically starves it of budget to maintain a constant risk profile for the total fund.

### Risk Management: Bastien Framework vs. Traditional Trend Following

| Risk Metric | Bastien Framework | Traditional Trend Following (e.g., Turtles) |
| :--- | :--- | :--- |
| \*\*Primary Unit\*\* | Risk Budget (Volatility-Adjusted) | Capital/Cash Percentage (e.g., 25 bps) |
| \*\*Adjustment Frequency\*\* | Real-time / Event-driven (Every new tick) | Static (At entry or infrequent intervals) |
| \*\*Position Dominance\*\* | Prevented via Hierarchical Parity | Allowed (Profits "run" until one trade dominates) |

## 5. Core Strategy Modules: Logic and Implementation

### Module A: The CTA Component
This module captures divergence by applying identical trend-following logic to idiosyncratic instruments and spreads. By trading the Relative Value Trend (spreads), the system extracts alpha while stripping away systematic market risk.

### Module B: Tactical Asset Allocation (TAA)
The TAA module utilizes a "Follow the Money" logic. It operates on the architectural assumption that "Big Money" (pension funds) has minimum exposure requirements and cannot exit to cash entirely. 
\*   \*\*Logic:\*\* It identifies which asset classes are receiving these mandatory flows.
\*   \*\*Execution:\*\* If an asset class trend is positive, it allocates; if negative, it defaults to cash. In 2022, this logic successfully avoided the simultaneous collapse of equities and bonds by identifying negative momentum and shifting to Commodities or Cash.

### Module C: Relative Value Volatility
A Market Neutral strategy targeting the VXX (short-term) and VXZ (mid-term) relationship.
\*   \*\*Logic:\*\* It applies Trend Following logic to the spread itself.
\*   \*\*Tail Risk Catch:\*\* While primarily short-volatility through decay capture, the system triggers a "tail risk catch" (shifting rapidly to long-vol) specifically when the long position in the volatility strategy exceeds a predefined threshold. This protected the framework during the 2020 volatility spike.

### Module D: Market Neutral Crypto
This module applies the framework's core momentum logic to the top 15 cryptocurrencies. 
\*   \*\*Logic:\*\* It executes relative value momentum on pairs (e.g., BTC/ETH). 
\*   \*\*Frequency:\*\* Unlike the daily ETF models, this includes an hourly intraday component to capture higher-velocity crypto trends.

## 6. Universal Trend-Following Model Constraints

To prevent overfitting and ensure architectural longevity, all models must adhere to four non-negotiable constraints:

1.  \*\*Non-Binary Execution:\*\* Rejection of "on/off" step functions. The system uses "Shades of Gray," scaling positions smoothly. The probability of being "full long" is near zero, as it requires extreme, low-volatility persistence.
2.  \*\*Optimization Surface Smoothness:\*\* Models are rejected if they do not exist on a wide "plateau." If a minor parameter shift (e.g., 20-day to 19-day lookback) creates a performance "cliff," the model is discarded as overfitted.
3.  \*\*Regime Agnosticism:\*\* Predictive regime identification is rejected. The system uses \*\*Drawdown Control\*\* as a proxy for regime. If a model is in a drawdown, it is de-facto in the "wrong" regime, and the HRP engine scales it back.
4.  \*\*Instrument Limitation:\*\* The system trades \*\*exactly 20 high-capacity ETFs\*\*. This is a fixed operational constraint for a "one-man show" and purposefully avoids the idiosyncratic risks of single stocks (e.g., CEO tweets or corporate actions).

## 7. Technical Stack and Operational Methodology

The infrastructure reflects a "control freak" philosophy, prioritizing total automation and the rejection of proprietary, black-box software.

### System Specifications
\*   \*\*Operating System:\*\* 100% Linux-based open-source environment (strict rejection of Microsoft/Proprietary OS).
\*   \*\*Programming Languages:\*\* A hybrid stack utilizing \*\*C++\*\* for the high-performance core engine and \*\*Python\*\* (Pandas/NumPy) for research, vectorized data processing, and new architecture.
\*   \*\*Execution:\*\* 100% automated; discretionary intervention is prohibited. The system utilizes proprietary execution algorithms, such as custom-coded TWAPs.
\*   \*\*Backtesting Engine:\*\* Proprietary and modular. Designed for rapid iteration, capable of running a full-history backtest (early 2000s to present) in under 10 seconds.

## 8. Model Decay and Maintenance Protocols

The framework treats alpha decay as a budgetary function. The system does not require a human to "kill" a model; the HRP engine naturally de-allocates failing logic.

### Drawdown Response Sequence
In the event of performance inconsistency, the system executes the following logic-gate sequence:

1.  \*\*IF\*\* \`Current\_Drawdown\` > \`Historical\_Threshold\_Fast\`:
    \*   \*\*THEN\*\* Identify as "Market Shock/Regime Shift."
    \*   \*\*ACTION\*\* Trigger immediate budget contraction via the HRP engine.
2.  \*\*IF\*\* \`Current\_Drawdown\` > \`Historical\_Threshold\_Slow\`:
    \*   \*\*THEN\*\* Identify as "Alpha Decay."
    \*   \*\*ACTION\*\* Systematically reduce model weight.
3.  \*\*IF\*\* \`Model\_Weight\` < \`Minimum\_Allocation\_Threshold\`:
    \*   \*\*THEN\*\* Flag model for architectural review.
    \*   \*\*ACTION\*\* Natural budget starvation trends weight toward zero, neutralizing the impact of the decaying alpha on the total portfolio.