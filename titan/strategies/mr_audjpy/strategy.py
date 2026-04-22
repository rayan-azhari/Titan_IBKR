"""mr_audjpy/strategy.py -- AUD/JPY MR Champion Strategy.

Intraday mean reversion on AUD/JPY using VWAP deviation grid entries
with Donchian-position multi-scale confluence as regime filter.

Post-remediation champion config (April 2026 audit):
  vwap_anchor=24, regime_filter=conf_donchian_pos_20,
  tier_grid=conservative, is_bars=32000, oos_bars=8000

  Pre-fix research claimed Sharpe +4.64 at vwap_anchor=46; both numbers
  were inflated by the sqrt(252)-on-H1 bug. On the corrected harness:

    vwap_anchor=24  Sharpe +0.53  DD -38%  (OPTIMAL)
    vwap_anchor=36  Sharpe +0.48  DD -47%
    vwap_anchor=46  Sharpe +0.32  DD -54%  (old "champion", now 4th/6)
    vwap_anchor=60  Sharpe +0.30  DD -61%
    vwap_anchor=72  Sharpe +0.11  DD -63%

  Full analysis: directives/Autoresearch Agent Loop 2026-04-21.md iter-5.

Regime filter: donchian_pos_20 at H1/H4/D/W scales must DISAGREE
  (scales don't agree on direction = ranging = allow MR entries).
  donchian_pos_20 = (close - 20-bar low) / (20-bar high - 20-bar low) - 0.5
  Positive = upper half of range = trending up.
  Negative = lower half = trending down.

Signal:
    1. Compute rolling VWAP with 24-bar anchor (~1 trading day on H1).
    2. Track price deviation from VWAP.
    3. Compute rolling percentile bands (95/98/99/99.9 = conservative grid).
    4. Enter tiered grid [1,2,4,8] when deviation > percentile threshold.
    5. Regime gate: donchian_pos_20 at H1/H4/D/W scales must DISAGREE.
    6. Session filter: 07:00-12:00 UTC entries only.
    7. Exit: 50% reversion TP or 21:00 UTC hard close.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.datetime import unix_nanos_to_dt
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Quantity
from nautilus_trader.trading.strategy import Strategy

from titan.research.metrics import BARS_PER_YEAR, ewm_vol_last
from titan.risk.portfolio_allocator import portfolio_allocator
from titan.risk.portfolio_risk_manager import portfolio_risk_manager
from titan.risk.strategy_equity import (
    StrategyEquityTracker,
    convert_notional_to_units,
    report_equity_and_check,
    split_fx_pair,
)

# Donchian period at H1 scale; higher scales multiply this
DONCHIAN_PERIOD = 20
# H1-bar equivalents per scale (H1=1, H4=4, D=24, W=120)
DONCHIAN_SCALES: dict[str, int] = {"H1": 1, "H4": 4, "D": 24, "W": 120}

TIERS_PCT = [0.95, 0.98, 0.99, 0.999]  # conservative grid
TIER_SIZES = [1, 2, 4, 8]


class MRAUDJPYConfig(StrategyConfig):
    """Configuration for AUD/JPY MR champion strategy."""

    instrument_id: str  # "AUD/JPY.IDEALPRO"
    bar_type_h1: str  # "AUD/JPY.IDEALPRO-1-HOUR-MID-EXTERNAL"
    ticker: str = "AUD_JPY"
    vwap_anchor: int = 24  # Post-audit champion: 24-bar (~1 day H1). Beats 46 on corrected harness.
    pct_window: int = 500  # Rolling percentile window
    reversion_pct: float = 0.50  # Exit at 50% reversion toward VWAP
    ny_close_hour: int = 21  # Hard close at 21:00 UTC
    entry_start_hour: int = 7  # Session entry start
    entry_end_hour: int = 12  # Session entry end
    vol_target_pct: float = 0.08  # Annualised vol target (8%)
    ewma_span: int = 20
    max_leverage: float = 2.0  # Paper trade: 2× (scale to 7× after validation)
    warmup_bars: int = 3000  # Enough for W-scale donchian (2400 bars)
    initial_equity: float = 10_000.0  # Seed capital for this strategy (base ccy)
    base_ccy: str = "USD"  # Portfolio accounting currency
    # FX rate from the instrument's quote ccy to base ccy. For AUD/JPY the
    # quote is JPY, so this is JPY->USD (~0.0067). For AUD/USD it's 1.0.
    # MUST be updated in config or overridden at runtime for non-USD quotes.
    fx_rate_quote_to_base: float = 1.0
    quote_ccy: str = "USD"  # Overridden at runtime for FX pairs


class MRAUDJPYStrategy(Strategy):
    """AUD/JPY MR with donchian-position multi-scale regime filter."""

    def __init__(self, config: MRAUDJPYConfig) -> None:
        super().__init__(config)
        self.instrument_id = InstrumentId.from_str(config.instrument_id)
        self.bar_type_h1 = BarType.from_str(config.bar_type_h1)

        self._closes: list[float] = []
        self._highs: list[float] = []
        self._lows: list[float] = []
        self._prm_id: str = ""
        self._equity_tracker: StrategyEquityTracker | None = None

        # Resolve quote currency from the instrument symbol (AUD/JPY -> JPY).
        pair = split_fx_pair(config.instrument_id)
        self._base_unit_ccy = pair[0] if pair else None  # trading units (AUD)
        self._quote_ccy = pair[1] if pair else config.quote_ccy  # JPY or USD
        self._fx_rate_quote_to_base = float(config.fx_rate_quote_to_base)

        # Position state
        self._tiers_hit: set[int] = set()

    def on_start(self) -> None:
        self._prm_id = f"mr_audjpy_{self.config.ticker}"

        # Fail fast: for a non-USD-quoted pair (e.g. AUD/JPY where quote is JPY),
        # the config default ``fx_rate_quote_to_base = 1.0`` would silently size
        # as if JPY == USD. Require an explicit override or refuse to start.
        if (
            self._quote_ccy != self.config.base_ccy
            and abs(self._fx_rate_quote_to_base - 1.0) < 1e-12
        ):
            raise ValueError(
                f"mr_audjpy: quote_ccy={self._quote_ccy!r} != "
                f"base_ccy={self.config.base_ccy!r} but fx_rate_quote_to_base "
                f"is still the default 1.0. Set an explicit rate in the config "
                f"(e.g. fx_rate_quote_to_base=0.0067 for JPY->USD) — refusing "
                f"to silently assume parity."
            )

        portfolio_risk_manager.register_strategy(self._prm_id, self.config.initial_equity)
        self._equity_tracker = StrategyEquityTracker(
            prm_id=self._prm_id,
            initial_equity=self.config.initial_equity,
            base_ccy=self.config.base_ccy,
        )

        self._warmup()
        self.subscribe_bars(self.bar_type_h1)
        self.log.info(
            f"MR AUD/JPY champion started | vwap_anchor={self.config.vwap_anchor}"
            f" | pct_window={self.config.pct_window}"
            f" | max_leverage={self.config.max_leverage}x"
            f" | warmup_bars={len(self._closes)}"
        )

    def _warmup(self) -> None:
        project_root = Path(__file__).resolve().parents[3]
        path = project_root / "data" / f"{self.config.ticker}_H1.parquet"
        if not path.exists():
            self.log.warning(f"Warmup data missing: {path}")
            return
        df = pd.read_parquet(path).sort_index()
        if "timestamp" in df.columns:
            df = df.set_index(pd.to_datetime(df["timestamp"], utc=True))
        tail = df.tail(self.config.warmup_bars)
        for _, row in tail.iterrows():
            self._closes.append(float(row["close"]))
            self._highs.append(float(row.get("high", row["close"])))
            self._lows.append(float(row.get("low", row["close"])))
        self.log.info(f"  {self.config.ticker}: {len(self._closes)} H1 bars loaded")

    def on_stop(self) -> None:
        self.cancel_all_orders(self.instrument_id)
        self.close_all_positions(self.instrument_id)
        self.log.info("MR AUD/JPY stopped -- flat.")

    def on_bar(self, bar: Bar) -> None:
        if bar.bar_type != self.bar_type_h1:
            return

        px = float(bar.close)
        self._closes.append(px)
        self._highs.append(float(bar.high))
        self._lows.append(float(bar.low))

        # Trim buffers
        max_keep = self.config.warmup_bars + 2000
        if len(self._closes) > max_keep:
            self._closes = self._closes[-max_keep:]
            self._highs = self._highs[-max_keep:]
            self._lows = self._lows[-max_keep:]

        ts = unix_nanos_to_dt(bar.ts_event)
        hour = ts.hour

        # Portfolio risk check (per-strategy equity + explicit timestamp)
        _, halted = report_equity_and_check(self, self._prm_id, bar, tracker=self._equity_tracker)
        if halted:
            self.log.warning("Portfolio kill switch -- flattening.")
            self._flatten()
            return

        portfolio_allocator.tick(now=ts.date())

        # Need enough bars for W-scale donchian (2400) + pct_window (500)
        min_bars = DONCHIAN_PERIOD * DONCHIAN_SCALES["W"] + self.config.pct_window
        if len(self._closes) < min_bars:
            return

        # VWAP deviation
        anchor = self.config.vwap_anchor
        vwap = float(np.mean(self._closes[-anchor:]))
        deviation = (px - vwap) / max(abs(vwap), 1e-8)
        abs_dev = abs(deviation)

        # Rolling percentile levels (conservative grid)
        devs = [
            abs(
                (self._closes[i] - float(np.mean(self._closes[max(0, i - anchor) : i])))
                / max(abs(float(np.mean(self._closes[max(0, i - anchor) : i]))), 1e-8)
            )
            for i in range(len(self._closes) - self.config.pct_window, len(self._closes))
        ]
        dev_arr = np.array(devs)
        levels = [float(np.percentile(dev_arr, p * 100)) for p in TIERS_PCT]

        # Check existing position
        positions = self.cache.positions(instrument_id=self.instrument_id)
        position = positions[-1] if positions else None
        has_pos = position and position.is_open

        # Exit checks
        if has_pos:
            if hour >= self.config.ny_close_hour:
                self._flatten()
                self.log.info(f"NY CLOSE: flatten @ {px:.5f}")
                return
            if levels and abs_dev < levels[0] * (1 - self.config.reversion_pct):
                self._flatten()
                self.log.info(
                    f"REVERSION TP: dev={abs_dev:.6f}"
                    f" < threshold={levels[0] * (1 - self.config.reversion_pct):.6f}"
                )
                return

        # Entry checks: session filter + regime gate
        in_session = self.config.entry_start_hour <= hour < self.config.entry_end_hour
        if not in_session:
            return

        ranging = self._check_donchian_regime()
        if not ranging:
            return

        # Tiered grid entries
        if not has_pos:
            self._tiers_hit = set()

        for tier_idx, (level, size) in enumerate(zip(levels, TIER_SIZES)):
            if tier_idx in self._tiers_hit:
                continue
            if abs_dev > level:
                direction = OrderSide.SELL if deviation > 0 else OrderSide.BUY
                units = self._compute_size(px, size)
                if units > 0:
                    qty = Quantity.from_int(units)
                    order = self.order_factory.market(
                        instrument_id=self.instrument_id,
                        order_side=direction,
                        quantity=qty,
                        time_in_force=TimeInForce.GTC,
                    )
                    self.submit_order(order)
                    self._tiers_hit.add(tier_idx)
                    self.log.info(
                        f"TIER {tier_idx + 1} {'SELL' if direction == OrderSide.SELL else 'BUY'}"
                        f" {qty} @ ~{px:.5f} | dev={abs_dev:.6f} > level={level:.6f}"
                    )

    def _check_donchian_regime(self) -> bool:
        """Return True if Donchian-position scales DISAGREE (ranging = allow MR).

        donchian_pos_20 = (close - N-bar low) / (N-bar high - N-bar low) - 0.5
        Compute at H1 (N=20), H4 (N=80), D (N=480), W (N=2400).
        Block entries when all scales agree (trending).
        """
        close_arr = np.array(self._closes)
        high_arr = np.array(self._highs)
        low_arr = np.array(self._lows)

        signs = []
        for label, mult in DONCHIAN_SCALES.items():
            w = DONCHIAN_PERIOD * mult
            if len(close_arr) < w:
                continue
            lo = float(low_arr[-w:].min())
            hi = float(high_arr[-w:].max())
            rng = hi - lo
            if rng < 1e-8:
                continue
            don_pos = (close_arr[-1] - lo) / rng - 0.5
            signs.append(np.sign(don_pos))

        if len(signs) < 2:
            return True  # Insufficient data — allow entries

        all_pos = all(s > 0 for s in signs)
        all_neg = all(s < 0 for s in signs)
        return not (all_pos or all_neg)  # Disagree = ranging = allow MR

    def _flatten(self) -> None:
        self.close_all_positions(self.instrument_id)
        self._tiers_hit = set()

    def _compute_size(self, price: float, tier_mult: int) -> int:
        """Vol-targeted sizing per tier, capped at max_leverage.

        Bars are H1, so annualisation is ``252 * 24``. Goes through the
        shared ``titan.research.metrics.ewm_vol_last`` so the same math is
        used in backtest and live (no drift).
        """
        if len(self._closes) < 60 or price <= 0:
            return 0
        if self._equity_tracker is None:
            return 0

        # Use *this strategy's* equity, not whole-account NLV.
        equity = self._equity_tracker.current_equity()
        if equity <= 0:
            return 0

        rets = pd.Series(self._closes[-60:]).pct_change().dropna()
        if len(rets) < 10:
            return 0
        # span -> RiskMetrics lambda:  alpha = 2/(span+1),  lambda = 1-alpha
        span = self.config.ewma_span
        rm_lambda = (span - 1.0) / (span + 1.0)
        ann_vol = ewm_vol_last(
            rets,
            lam=rm_lambda,
            periods_per_year=BARS_PER_YEAR["H1"],
        )
        if ann_vol <= 0:
            return 0

        notional = equity * (self.config.vol_target_pct / ann_vol)
        notional = min(notional, equity * self.config.max_leverage)

        alloc = portfolio_allocator.get_weight(self._prm_id)
        notional *= alloc * portfolio_risk_manager.scale_factor

        # FX-aware unit conversion -- AUD/JPY price is JPY-per-AUD so we must
        # convert a USD notional into AUD units via an explicit JPY->USD rate.
        fx_rate = self._fx_rate_quote_to_base if self._quote_ccy != self.config.base_ccy else None
        base_units = convert_notional_to_units(
            notional_base=notional,
            price=price,
            quote_ccy=self._quote_ccy,
            base_ccy=self.config.base_ccy,
            fx_rate_quote_to_base=fx_rate,
        )
        return max(0, base_units * tier_mult // max(sum(TIER_SIZES), 1))

    def on_position_closed(self, event) -> None:
        self._tiers_hit = set()
        if self._equity_tracker is not None:
            try:
                # realized_pnl is in the instrument's quote ccy (JPY for AUD/JPY)
                pnl_quote = float(event.realized_pnl.as_double())
                fx = self._fx_rate_quote_to_base if self._quote_ccy != self.config.base_ccy else 1.0
                self._equity_tracker.on_position_closed(pnl_quote, fx_to_base=fx)
            except Exception as e:
                self.log.warning(f"tracker on_position_closed failed: {e}")
        self.log.info(f"CLOSED: PnL={event.realized_pnl}")

    def on_order_rejected(self, event) -> None:
        self.log.error(f"REJECTED: {event.client_order_id} -- {event.reason}")
