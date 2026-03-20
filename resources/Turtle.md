Listed directory strategies

The best way to build a **Turtle Trading Strategy** within the Titan-IBKR framework is to strictly follow the **Modular Architecture** outlined in your `AGENTS.md` file. The Turtle rules (Donchian breakouts for entries and ATR/"N" for position sizing) fit perfectly into a quantitative, rule-based paradigm.

Here is the step-by-step roadmap for building the Turtle Strategy "The Titan Way":

### 1. The Research Phase (`research/ic_analysis` or custom VectorBT Lab)
Before writing execution code, you should validate whether the 20-day and 55-day structural breakouts still offer a statistical edge on your target instruments today.
*   **Leverage the IC Pipeline**: You already have the building blocks in the pipeline we just audited! In `phase1_sweep.py`, we compute `donchian_pos_20`, `donchian_pos_55` (Group F: Structural) and `norm_atr_14` (Group D). 
*   **Run a Sweep**: You can run `uv run python research/ic_analysis/phase1_sweep.py --instrument SPY --timeframe D` to check the unconditional and regime-conditioned Information Coefficient (IC) of these Donchian breakouts.
*   **VectorBT Pro Loop**: If the IC is viable, build a rapid prototype in `research/strategies/turtle_vbt.py` using VectorBT Pro to test the ATR-based position sizing and trailing stops over decades of data.

### 2. Core Logic (`titan/strategies/turtle/`)
Once validated, the strategy moves to Layer 1 (Core Logic). This directory must contain reusable, production-grade, strict-typed classes.

*   **Config (`turtle_config.py`)**: Create a dataclass or Pydantic model inheriting from Nautilus configuration.
    ```python
    from nautilus_trader.config import StrategyConfig

    class TurtleConfig(StrategyConfig):
        fast_breakout: int = 20
        slow_breakout: int = 55
        atr_period: int = 20
        risk_per_trade_pct: float = 0.01  # 1% risk rule
        pyramiding: int = 4  # Max units per market
    ```
*   **Strategy Class (`turtle_strategy.py`)**: Create the Nautilus `Strategy` implementation. It should only contain trading logic, utilizing strict typing (`decimal.Decimal` for prices) and absolute imports (`from titan.models...`).
    *   **Indicators**: Use native `nautilus_trader.indicators` like `DonchianChannel` and `AverageTrueRange` to avoid calculating these manually tick-by-tick. 
    *   **Position Sizing logic ("N")**: When `on_bar` triggers a 20-day breakout, calculate the absolute unit size: `(Account Equity * 1%) / (ATR * Point Value)`.
    *   **Risk Control**: Issue stop-loss orders exactly `2N` below the entry price, per classic Turtle rules.

### 3. Application State (`config/`)
All strategy parameters should be completely decoupled from the code. Create a `config/turtle.toml` file.
```toml
[strategy]
fast_breakout = 20
slow_breakout = 55
atr_period = 20
risk_per_trade_pct = 0.01

[universe]
instruments = ["SPY", "QQQ", "GLD"]
```

### 4. The Execution Layer (`scripts/`)
Finally, build CLI entry points for operators. This is Layer 3, and should *never* contain internal trading logic.
*   **Backtesting (`scripts/run_turtle_backtest.py`)**: A script that parses the `turtle.toml` config, loads daily Parquet data from `data/`, sets up the Nautilus `BacktestEngine`, adds the `TurtleStrategy`, and outputs the tearsheet.
*   **Live Trading (`scripts/run_live_turtle.py`)**: An equivalent script that hooks into the Interactive Brokers (`IBKR`) adapter to run the strategy live in a `LiveExecutionEngine`.

### Key System Guidelines Check
When building this strategy, remember the core Titan tenets:
1.  **Strict Typing**: Ensure all quantities and prices utilize Nautilus-native types or `decimal.Decimal` when executing.
2.  **No Scripts in Core**: Keep all logic strictly class-based in `titan/`.
3.  **Sanity Check Look-Aheads**: The 20-day high is determined using the *previous* 20 closing/high bars. Ensure the Nautilus Donchian indicator is reading closed bars to prevent the classic backtesting optical illusion.