# -------------------------------------------------------------------------------------------------
# Multi-Timeframe Momentum Mean Reversion Strategy for Nautilus Trader
# 多时间框架动量均值回归策略
# -------------------------------------------------------------------------------------------------

from decimal import Decimal
from typing import Dict, Optional

import pandas as pd

from nautilus_trader.common.enums import LogColor
from nautilus_trader.config import PositiveFloat
from nautilus_trader.config import PositiveInt
from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.correctness import PyCondition
from nautilus_trader.core.data import Data
from nautilus_trader.core.message import Event
from nautilus_trader.indicators.average.ema import ExponentialMovingAverage
from nautilus_trader.indicators.atr import AverageTrueRange
from nautilus_trader.indicators.bollinger_bands import BollingerBands
from nautilus_trader.indicators.macd import MovingAverageConvergenceDivergence
from nautilus_trader.indicators.rsi import RelativeStrengthIndex
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import TimeInForce
from nautilus_trader.model.enums import TriggerType
from nautilus_trader.model.events import OrderFilled
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity
from nautilus_trader.model.orders import LimitOrder
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.model.orders import StopMarketOrder
from nautilus_trader.trading.strategy import Strategy

class MTMMRConfig(StrategyConfig, frozen=True):
    """
    Configuration for Multi-Timeframe Momentum Mean Reversion Strategy.

    Parameters
    ----------
    instrument_id : InstrumentId
        The instrument ID for the strategy.
    long_bar_type : BarType
        The long timeframe bar type (15-minute).
    medium_bar_type : BarType
        The medium timeframe bar type (5-minute).
    short_bar_type : BarType
        The short timeframe bar type (1-minute).
    trade_size : Decimal
        The base position size per trade.
    max_position_risk : PositiveFloat, default 0.01
        Maximum risk per position as fraction of account balance.
    ema_fast_period : PositiveInt, default 21
        The fast EMA period.
    ema_slow_period : PositiveInt, default 50
        The slow EMA period.
    macd_fast_period : PositiveInt, default 12
        The MACD fast period.
    macd_slow_period : PositiveInt, default 26
        The MACD slow period.
    bb_period : PositiveInt, default 20
        The Bollinger Bands period.
    bb_std : PositiveFloat, default 2.0
        The Bollinger Bands standard deviation.
    rsi_period : PositiveInt, default 14
        The RSI period.
    atr_period : PositiveInt, default 14
        The ATR period.
    rsi_oversold : PositiveFloat, default 30.0
        The RSI oversold threshold.
    rsi_overbought : PositiveFloat, default 70.0
        The RSI overbought threshold.
    stop_loss_atr_multiple : PositiveFloat, default 2.0
        The stop loss ATR multiple.
    take_profit_1_atr_multiple : PositiveFloat, default 1.5
        The first take profit ATR multiple.
    take_profit_2_atr_multiple : PositiveFloat, default 3.0
        The second take profit ATR multiple.
    """

    instrument_id: InstrumentId
    long_bar_type: BarType
    medium_bar_type: BarType
    short_bar_type: BarType
    trade_size: Decimal
    max_position_risk: PositiveFloat = 0.01
    ema_fast_period: PositiveInt = 21
    ema_slow_period: PositiveInt = 50
    macd_fast_period: PositiveInt = 12
    macd_slow_period: PositiveInt = 26
    bb_period: PositiveInt = 20
    bb_std: PositiveFloat = 2.0
    rsi_period: PositiveInt = 14
    atr_period: PositiveInt = 14
    rsi_oversold: PositiveFloat = 30.0
    rsi_overbought: PositiveFloat = 70.0
    stop_loss_atr_multiple: PositiveFloat = 2.0
    take_profit_1_atr_multiple: PositiveFloat = 1.5
    take_profit_2_atr_multiple: PositiveFloat = 3.0

class MTMMRStrategy(Strategy):
    """
    Multi-Timeframe Momentum Mean Reversion Strategy.

    This strategy combines trend following and mean reversion approaches across
    multiple timeframes to identify high-probability trading opportunities.

    Strategy Logic:
    1. Long-term trend analysis (15-min): Determine overall market direction
    2. Medium-term momentum (5-min): Confirm trend strength
    3. Short-term entry (1-min): Find precise entry points using mean reversion

    Entry Conditions:
    - Long: Uptrend + momentum confirmation + oversold bounce from BB lower band
    - Short: Downtrend + momentum confirmation + overbought rejection from BB upper band

    Risk Management:
    - Dynamic position sizing based on ATR
    - Stop loss at 2x ATR
    - Partial profit taking at 1.5x and 3x ATR
    """

    def __init__(self, config: MTMMRConfig) -> None:
        self.configue_parameters(config)
        super().__init__(config)

        self.instrument: Optional[Instrument] = None

        # Long timeframe indicators (15-min)
        self.long_ema_fast = ExponentialMovingAverage(config.ema_fast_period)
        self.long_ema_slow = ExponentialMovingAverage(config.ema_slow_period)

        # Medium timeframe indicators (5-min)
        self.medium_macd = MovingAverageConvergenceDivergence(
            config.macd_fast_period,
            config.macd_slow_period,
        )

        # Short timeframe indicators (1-min)
        self.short_bb = BollingerBands(config.bb_period, config.bb_std)
        self.short_rsi = RelativeStrengthIndex(config.rsi_period)
        self.short_atr = AverageTrueRange(config.atr_period)

        # Strategy state
        self.long_trend_direction: Optional[int] = None  # 1 for up, -1 for down, None for neutral
        self.medium_momentum_confirmed: bool = False
        self.current_position_side: Optional[OrderSide] = None
        self.stop_loss_order: Optional[StopMarketOrder] = None
        self.take_profit_orders: Dict[str, LimitOrder] = {}

    def on_start(self) -> None:
        """Actions to be performed on strategy start."""
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(f"Could not find instrument for {self.config.instrument_id}")
            self.stop()
            return

        # Register indicators for different timeframes
        #long bar
        self.register_indicator_for_bars(self.config.long_bar_type, self.long_ema_fast)
        self.register_indicator_for_bars(self.config.long_bar_type, self.long_ema_slow)
        # medium bar
        self.register_indicator_for_bars(self.config.medium_bar_type, self.medium_macd)
        # short bar
        self.register_indicator_for_bars(self.config.short_bar_type, self.short_bb)
        self.register_indicator_for_bars(self.config.short_bar_type, self.short_rsi)
        self.register_indicator_for_bars(self.config.short_bar_type, self.short_atr)

        # Request historical data
        self.request_bars(
            self.config.long_bar_type,
            start=self._clock.utc_now() - pd.Timedelta(days=1),
        )
        self.request_bars(
            self.config.medium_bar_type,
            start=self._clock.utc_now() - pd.Timedelta(days=1),
        )
        self.request_bars(
            self.config.short_bar_type,
            start=self._clock.utc_now() - pd.Timedelta(days=1),
        )

        # Subscribe to live data
        self.subscribe_bars(self.config.long_bar_type)
        self.subscribe_bars(self.config.medium_bar_type)
        self.subscribe_bars(self.config.short_bar_type)

        # 订阅交易情况与市场价信息
        self.subscribe_trade_ticks(self.config.instrument_id)
        self.subscribe_quote_ticks(self.config.instrument_id)

    def on_bar(self, bar: Bar) -> None:
        """Actions to be performed when receiving a bar."""
        self.log.info(f"Received bar: {bar.bar_type} - {bar.close}", LogColor.CYAN)

        # Update trend analysis based on bar type
        if bar.bar_type == self.config.long_bar_type:
            self._update_long_trend()
        elif bar.bar_type == self.config.medium_bar_type:
            self._update_medium_momentum()
        elif bar.bar_type == self.config.short_bar_type:
            self._check_entry_signals(bar)

    def _update_long_trend(self) -> None:
        """Update long-term trend direction."""
        if not self.long_ema_fast.initialized or not self.long_ema_slow.initialized:
            self.log.info(f"long_trend_indicators not initialized fully yet", LogColor.RED)
            return

        if self.long_ema_fast.value > self.long_ema_slow.value:
            self.long_trend_direction = 1  # Uptrend
            self.log.info("Long-term trend: UPTREND", LogColor.GREEN)
        elif self.long_ema_fast.value < self.long_ema_slow.value:
            self.long_trend_direction = -1  # Downtrend
            self.log.info("Long-term trend: DOWNTREND", LogColor.GREEN)
        else:
            self.long_trend_direction = None  # Neutral
            self.log.info("Long-term trend: NEUTRAL", LogColor.GREEN)

    def _update_medium_momentum(self) -> None:
        """Update medium-term momentum confirmation."""
        if not self.medium_macd.initialized:
            self.log.info("medium indicators not initialized fully yet", LogColor.RED)
            return

        macd_value = self.medium_macd.value

        if self.long_trend_direction == 1:
            # In uptrend, look for bullish momentum (MACD above zero)
            self.medium_momentum_confirmed = macd_value > 0
        elif self.long_trend_direction == -1:
            # In downtrend, look for bearish momentum (MACD below zero)
            self.medium_momentum_confirmed = macd_value < 0
        else:
            self.medium_momentum_confirmed = False

        self.log.info(
            f"Medium momentum confirmed: {self.medium_momentum_confirmed} "
            f"(MACD: {float(macd_value):.4f})",
            LogColor.GREEN,
        )

    def _check_entry_signals(self, bar: Bar) -> None:
        """Check for entry signals on short timeframe."""
        if not self._all_indicators_ready():
            self.log.info("indicators are not initialized fully please wait...", LogColor.RED)
            return

        # Don't enter new positions if already in one
        if not self.portfolio.is_flat(self.config.instrument_id):
            self.log.info(f"Already in position, skipping entry checks for {self.config.instrument_id} this time", LogColor.YELLOW)
            return

        current_price = bar.close
        bb_upper = self.short_bb.upper
        bb_lower = self.short_bb.lower
        bb_middle = self.short_bb.middle
        rsi_value = self.short_rsi.value

        self.log.info(
            f"Market state - Price: {current_price}, BB: [{float(bb_lower):.4f}, {float(bb_middle):.4f}, {float(bb_upper):.4f}], "
            f"RSI: {float(rsi_value):.2f}, Trend: {self.long_trend_direction}, Momentum: {self.medium_momentum_confirmed}",
            LogColor.CYAN,
        )

        # Long entry conditions
        if (self.long_trend_direction == 1 and
                self.medium_momentum_confirmed and
                current_price <= bb_lower and
                rsi_value <= self.config.rsi_oversold):

            self.log.info("LONG ENTRY SIGNAL DETECTED!", LogColor.BLUE)
            self._enter_long(current_price)

        # Short entry conditions
        elif (self.long_trend_direction == -1 and
              self.medium_momentum_confirmed and
              current_price >= bb_upper and
              rsi_value >= self.config.rsi_overbought):

            self.log.info("SHORT ENTRY SIGNAL DETECTED!", LogColor.BLUE)
            self._enter_short(current_price)

    def _all_indicators_ready(self) -> bool:
        """Check if all indicators are initialized."""
        return (self.long_ema_fast.initialized and
                self.long_ema_slow.initialized and
                self.medium_macd.initialized and
                self.short_bb.initialized and
                self.short_rsi.initialized and
                self.short_atr.initialized)

    def _enter_long(self, entry_price: float) -> None:
        """Enter a long position."""
        if not self.instrument:
            return

        # Calculate position size based on ATR and risk management
        atr_value = self.short_atr.value
        risk_amount = self.portfolio.account(self.config.instrument_id.venue).balance() * self.config.max_position_risk
        stop_distance = atr_value * self.config.stop_loss_atr_multiple
        position_size = min(
            float(self.config.trade_size),
            risk_amount / stop_distance if stop_distance > 0 else float(self.config.trade_size)
        )

        # Entry order
        entry_order = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.BUY,
            quantity=self.instrument.make_qty(position_size),
            time_in_force=TimeInForce.GTC,
        )

        self.submit_order(entry_order)
        self.current_position_side = OrderSide.BUY

        # Set stop loss and take profit orders
        self._set_risk_management_orders(entry_price, OrderSide.BUY, position_size)

    def _enter_short(self, entry_price: float) -> None:
        """Enter a short position."""
        if not self.instrument:
            return

        # Calculate position size based on ATR and risk management
        atr_value = self.short_atr.value
        risk_amount = self.portfolio.account(self.config.instrument_id.venue).balance() * self.config.max_position_risk
        stop_distance = atr_value * self.config.stop_loss_atr_multiple
        position_size = min(
            float(self.config.trade_size),
            risk_amount / stop_distance if stop_distance > 0 else float(self.config.trade_size)
        )

        # Entry order
        entry_order = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.SELL,
            quantity=self.instrument.make_qty(position_size),
            time_in_force=TimeInForce.GTC,
        )

        self.submit_order(entry_order)
        self.current_position_side = OrderSide.SELL

        # Set stop loss and take profit orders
        self._set_risk_management_orders(entry_price, OrderSide.SELL, position_size)

    def _set_risk_management_orders(self, entry_price: float, side: OrderSide, position_size: float) -> None:
        """Set stop loss and take profit orders."""
        if not self.instrument:
            return

        atr_value = self.short_atr.value

    def on_stop(self):
        """Actions to be performed when strategy is stopped."""
        # Cancel all open orders
        self.cancel_all_orders(self.config.instrument_id)

        # Note: For spot trading, we typically don't close all positions
        # as they can be held long-term. Only close if explicitly required.

        # Unsubscribe from data
        self.unsubscribe_bars(self.config.long_bar_type)
        self.unsubscribe_bars(self.config.medium_bar_type)
        self.unsubscribe_bars(self.config.short_bar_type)
        self.unsubscribe_quote_ticks(self.config.instrument_id)
        self.unsubscribe_trade_ticks(self.config.instrument_id)

        self.log.info("MTMMR Strategy stopped", LogColor.YELLOW)

    def on_reset(self):
        """Reset all indicators and state."""
        # Reset indicators
        self.long_ema_fast.reset()
        self.long_ema_slow.reset()
        self.medium_macd.reset()
        self.short_bb.reset()
        self.short_rsi.reset()
        self.short_atr.reset()
        # Reset strategy state
        self.long_trend_direction: Optional[int] = None
        self.medium_momentum_confirmed: bool = False
        self.current_position_side: Optional[OrderSide] = None
        self.stop_loss_order: Optional[StopMarketOrder] = None
        self.take_profit_orders: Dict[str, LimitOrder] = {}

    def on_save(self) -> dict[str, bytes]:
        """Save strategy state."""
        return {}

    def on_load(self, state: dict[str, bytes]) -> None:
        """Load strategy state."""
        pass

    def on_dispose(self) -> None:
        """Cleanup resources."""
        pass

    def configue_parameters(self, config: MTMMRConfig) -> None:
        """Configure strategy parameters."""
        PyCondition.is_true(
            config.ema_fast_period < config.ema_slow_period,
            f"{config.ema_fast_period=} must be less than {config.ema_slow_period=}",
        )
        PyCondition.is_true(
            config.macd_fast_period < config.macd_slow_period,
            f"{config.macd_fast_period=} must be less than {config.macd_slow_period=}",
        )
        PyCondition.is_true(
            config.rsi_oversold < config.rsi_overbought,
            f"{config.rsi_oversold=} must be less than {config.rsi_overbought=}",
        )