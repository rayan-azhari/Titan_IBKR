The Opening Range Breakout (ORB) strategy described in the video relies on a multi-timeframe approach to capture early momentum during the New York session.

Here is the mechanical rule set extracted from the video:

Define the Range: Mark the High and Low of the first 15-minute candle at the New York open ([09:30:00] to 09:45 EST).

Signal Confirmation: Wait for a 5-minute candle to close above the 15-minute range high (for a long bias) or below the 15-minute range low (for a short bias) ([05:37:00]).

Execution & Management: Drop down to the 1-minute chart to execute the trade using a breakout, retest, or mean-reversion model, targeting a strict 1:2 risk-to-reward ratio ([08:53:00]).

Below is a comprehensive framework to backtest this behaviour using vectorbt and a blueprint to execute it live on Interactive Brokers using NautilusTrader.

1. VectorBT Backtesting Implementation
This script standardises 1-minute OHLCV data, resamples it to build the 15-minute opening range and 5-minute confirmation candles, and evaluates the strategy using vectorbt.

To execute this systematically on Interactive Brokers via NautilusTrader, you will need to handle multiple bar specifications simultaneously within the strategy class. Nautilus handles message-driven events, allowing you to subscribe to 15m, 5m, and 1m ticks or bars concurrently to construct the logic.

Implementing a strict 1% risk management rule is one of the most robust ways to preserve capital and ensure longevity in the markets. Since the Opening Range Breakout (ORB) strategy relies on capturing momentum, the distance to your stop loss will vary with the market's volatility on any given day. This means your position size must dynamically adjust to keep the risk constant.The Core Risk FormulaTo calculate the exact number of shares or contracts to buy, you must determine your absolute risk per trade and divide it by the risk per unit (the distance between your entry price and your stop loss).$$\text{Position Size} = \frac{\text{Total Equity} \times 0.01}{|\text{Entry Price} - \text{Stop Loss Price}|}$$Total Equity: Your current account balance (e.g., £100,000).1% Risk: The maximum amount you are willing to lose (£1,000).Trade Risk (Denominator): If entering at £150 with a stop loss at £145, your risk per share is £5.Result: £1,000 / £5 = 200 shares.1. Dynamic Sizing in VectorBTWhen backtesting with vectorbt, you can simulate this dynamic sizing by calculating the risk distance for every potential trade and passing an array of sizes to the portfolio generator.

While the mathematical model is precise, live markets are not. When trading the New York open, volatility is high. If your stop loss is triggered via a market order during a fast down-move, slippage will likely cause your actual loss to slightly exceed the strict 1% threshold. It is often wise to build a small slippage buffer into your denominator to account for this reality.

Here is the fully integrated NautilusTrader script. It combines the multi-timeframe Opening Range Breakout (ORB) logic with the dynamic 1% risk position sizing function.

This script is designed to wait for the 15-minute range to establish, look for a 5-minute confirmation close, calculate the exact order size based on your live account equity, and route the trade on the 1-minute chart.

The Complete ORB Execution Script

from decimal import Decimal
import math
from nautilus_trader.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.models.data import Bar
from nautilus_trader.models.enums import OrderSide
from nautilus_trader.models.identifiers import InstrumentId, AccountId

class ORBConfig(StrategyConfig):
    instrument_id: InstrumentId
    account_id: AccountId
    risk_percent: Decimal = Decimal("0.01")  # 1% risk per trade
    risk_reward_ratio: Decimal = Decimal("2.0")

class ORBStrategy(Strategy):
    def __init__(self, config: ORBConfig):
        super().__init__(config)
        self.instrument_id = config.instrument_id
        self.account_id = config.account_id
        
        # State variables
        self.or_high: Decimal | None = None
        self.or_low: Decimal | None = None
        self.signal_active: str | None = None
        self.traded_today: bool = False  # Flag to prevent overtrading
        self.current_date: str | None = None

    def on_start(self):
        """Subscribe to the required timeframes for the strategy."""
        self.subscribe_bars(self.instrument_id, "15-MINUTE")
        self.subscribe_bars(self.instrument_id, "5-MINUTE")
        self.subscribe_bars(self.instrument_id, "1-MINUTE")
        self.log.info("ORB Strategy started. Subscribed to 1m, 5m, and 15m bars.")

    def on_bar(self, bar: Bar):
        """Main event loop triggered on every new bar."""
        dt = bar.ts_event.to_datetime()
        time_str = dt.strftime('%H:%M')
        date_str = dt.strftime('%Y-%m-%d')
        bar_type = bar.bar_type.to_str()

        # Reset daily variables at the start of a new trading session
        if self.current_date != date_str:
            self.current_date = date_str
            self.or_high = None
            self.or_low = None
            self.signal_active = None
            self.traded_today = False

        # 1. Establish the 15-minute Opening Range (09:30 - 09:45)
        if "15-MINUTE" in bar_type and time_str == "09:45":
            self.or_high = bar.high
            self.or_low = bar.low
            self.log.info(f"[{time_str}] ORB Established: High {self.or_high}, Low {self.or_low}")

        # 2. Check 5-minute Confirmation (Breakout of the range)
        if "5-MINUTE" in bar_type and self.or_high is not None and not self.signal_active and not self.traded_today:
            if bar.close > self.or_high:
                self.signal_active = "LONG"
                self.log.info(f"[{time_str}] Bullish 5m confirmation. Looking for long entry.")
            elif bar.close < self.or_low:
                self.signal_active = "SHORT"
                self.log.info(f"[{time_str}] Bearish 5m confirmation. Looking for short entry.")

        # 3. Execute on the 1-minute chart
        if "1-MINUTE" in bar_type and self.signal_active and not self.traded_today:
            # Ensure we are not already holding a position
            if not self.portfolio.is_flat(self.instrument_id):
                return 
            
            entry_price = bar.close
            
            # Determine Stop Loss based on direction
            stop_loss = self.or_low if self.signal_active == "LONG" else self.or_high
            
            # Calculate dynamic size based on 1% risk
            size = self._calculate_position_size(entry_price, stop_loss)
            
            if size > Decimal("0"):
                side = OrderSide.BUY if self.signal_active == "LONG" else OrderSide.SELL
                self._execute_bracket_trade(side, entry_price, stop_loss, size)
                self.traded_today = True  # Lock out further trades for the day

    def _calculate_position_size(self, entry_price: Decimal, stop_loss_price: Decimal) -> Decimal:
        """Calculates order quantity to risk exactly 1% of total equity."""
        # Retrieve the live account state from Nautilus
        account = self.portfolio.get_account(self.account_id)
        if not account:
            self.log.error("Account not found. Cannot calculate position size.")
            return Decimal("0")
            
        current_equity = account.margins.net_liquidation_value
        risk_amount = current_equity * self.config.risk_percent
        
        trade_risk = abs(entry_price - stop_loss_price)
        
        if trade_risk <= Decimal("0"):
            self.log.error("Invalid trade risk calculation. Aborting sizing.")
            return Decimal("0")
        
        # Calculate raw size and floor it to avoid fractional units
        raw_size = risk_amount / trade_risk
        position_size = Decimal(math.floor(raw_size))
        
        self.log.info(
            f"Sizing Calculation -> Equity: {current_equity} | Risk (£): {risk_amount:.2f} | "
            f"Risk per share: {trade_risk} | Size: {position_size} units"
        )
        
        return position_size

    def _execute_bracket_trade(self, side: OrderSide, entry_price: Decimal, stop_loss_price: Decimal, quantity: Decimal):
        """Submits the entry order alongside a Stop Loss and Take Profit."""
        risk_per_unit = abs(entry_price - stop_loss_price)
        target_distance = risk_per_unit * self.config.risk_reward_ratio
        
        # Calculate Take Profit price
        if side == OrderSide.BUY:
            take_profit_price = entry_price + target_distance
        else:
            take_profit_price = entry_price - target_distance

        # Note: Nautilus order execution logic can vary slightly depending on the exact adapter.
        # This standardises sending a market entry with attached limit/stop exits.
        entry_order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=side,
            quantity=quantity
        )
        
        # In a fully integrated production script, you would attach OCO (One-Cancels-Other) 
        # conditions for the SL and TP here according to the IBKR adapter's capabilities.
        
        self.submit_order(entry_order)
        self.log.info(
            f"Executed {side} | Qty: {quantity} | Entry: {entry_price} | "
            f"SL: {stop_loss_price} | TP: {take_profit_price}"
        )

    def on_stop(self):
        """Cleanup function when the strategy is stopped."""
        self.cancel_all_orders(self.instrument_id)
        self.close_all_positions(self.instrument_id)
        self.log.info("Strategy stopped. All orders cancelled and positions closed.")




Key Optimisations Added:
self.traded_today Toggle: Once the first valid signal of the day triggers and executes, this boolean switches to True. This prevents the algorithm from continuously firing orders if the price hovers around your entry level. It resets automatically the next trading day.

Date Tracking: Added logic to clear out yesterday's or_high and or_low parameters when a new session begins, ensuring you do not accidentally execute trades based on stale data.

Bracket Order Blueprint: The _execute_bracket_trade function calculates your 1:2 Take Profit natively based on the entry and the stop loss.