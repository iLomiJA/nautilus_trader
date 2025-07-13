# -------------------------------------------------------------------------------------------------
# Multi-Level Dynamic Allocation Strategy for Spot Trading
# 多层级动态配置现货交易策略
# -------------------------------------------------------------------------------------------------

from decimal import Decimal
from typing import Dict, Optional
from enum import Enum

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
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import TimeInForce
from nautilus_trader.model.events import OrderFilled
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.objects import Price
from nautilus_trader.model.objects import Quantity
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.model.orders import StopMarketOrder
from nautilus_trader.trading.strategy import Strategy

class PositionLevel(Enum):
    """仓位层级枚举"""
    CORE = "CORE"           # 核心仓位 (40%)
    TACTICAL = "TACTICAL"   # 战术仓位 (30%)
    OPPORTUNITY = "OPPORTUNITY"  # 机会仓位 (20%)
    CASH = "CASH"          # 现金储备 (10%)

class MarketState(Enum):
    """市场状态枚举"""
    BULL = "BULL"          # 牛市
    BEAR = "BEAR"          # 熊市
    SIDEWAYS = "SIDEWAYS"  # 震荡市

class MDASConfig(StrategyConfig, frozen=True):
    """
    Configuration for Multi-Level Dynamic Allocation Strategy.
    
    Parameters
    ----------
    instrument_id : InstrumentId
        The instrument ID for the strategy.
    daily_bar_type : BarType
        The daily timeframe bar type.
    h4_bar_type : BarType
        The 4-hour timeframe bar type.
    h1_bar_type : BarType
        The 1-hour timeframe bar type.
    total_capital : Decimal
        Total capital allocated to this strategy.
    core_allocation : PositiveFloat, default 0.40
        Core position allocation ratio (40%).
    tactical_allocation : PositiveFloat, default 0.30
        Tactical position allocation ratio (30%).
    opportunity_allocation : PositiveFloat, default 0.20
        Opportunity position allocation ratio (20%).
    cash_allocation : PositiveFloat, default 0.10
        Cash reserve allocation ratio (10%).
    max_single_trade_ratio : PositiveFloat, default 0.05
        Maximum single trade ratio (5% of total capital).
    ema_short_period : PositiveInt, default 20
        Short EMA period.
    ema_medium_period : PositiveInt, default 50
        Medium EMA period.
    ema_long_period : PositiveInt, default 200
        Long EMA period.
    rsi_period : PositiveInt, default 14
        RSI period.
    bb_period : PositiveInt, default 20
        Bollinger Bands period.
    bb_std : PositiveFloat, default 2.0
        Bollinger Bands standard deviation.
    atr_period : PositiveInt, default 14
        ATR period.
    rsi_oversold : PositiveFloat, default 30.0
        RSI oversold threshold.
    rsi_overbought : PositiveFloat, default 70.0
        RSI overbought threshold.
    core_stop_loss : PositiveFloat, default 0.20
        Core position stop loss ratio (20%).
    tactical_stop_loss : PositiveFloat, default 0.10
        Tactical position stop loss ratio (10%).
    opportunity_stop_loss : PositiveFloat, default 0.05
        Opportunity position stop loss ratio (5%).
    """
    
    instrument_id: InstrumentId
    daily_bar_type: BarType
    h4_bar_type: BarType
    h1_bar_type: BarType
    total_capital: Decimal
    core_allocation: PositiveFloat = 0.40
    tactical_allocation: PositiveFloat = 0.30
    opportunity_allocation: PositiveFloat = 0.20
    cash_allocation: PositiveFloat = 0.10
    max_single_trade_ratio: PositiveFloat = 0.05
    ema_short_period: PositiveInt = 20
    ema_medium_period: PositiveInt = 50
    ema_long_period: PositiveInt = 200
    rsi_period: PositiveInt = 14
    bb_period: PositiveInt = 20
    bb_std: PositiveFloat = 2.0
    atr_period: PositiveInt = 14
    rsi_oversold: PositiveFloat = 30.0
    rsi_overbought: PositiveFloat = 70.0
    core_stop_loss: PositiveFloat = 0.20
    tactical_stop_loss: PositiveFloat = 0.10
    opportunity_stop_loss: PositiveFloat = 0.05


class MDASStrategy(Strategy):
    """
    Multi-Level Dynamic Allocation Strategy for Spot Trading.
    
    This strategy implements a multi-level position allocation approach specifically
    designed for spot trading, where only long positions are possible.
    
    Strategy Features:
    1. Multi-level capital allocation (Core 40%, Tactical 30%, Opportunity 20%, Cash 10%)
    2. Multi-timeframe analysis (Daily, 4H, 1H)
    3. Dynamic position sizing based on market conditions
    4. Risk management with different stop-loss levels for each position type
    5. Trend following combined with mean reversion opportunities
    
    Position Levels:
    - Core (40%): Long-term trend following, daily timeframe
    - Tactical (30%): Medium-term swing trading, 4-hour timeframe  
    - Opportunity (20%): Short-term mean reversion, 1-hour timeframe
    - Cash (10%): Reserve for extreme opportunities
    """
    
    def __init__(self, config: MDASConfig) -> None:
        # Validate allocation ratios sum to 1.0
        total_allocation = (config.core_allocation + config.tactical_allocation + 
                          config.opportunity_allocation + config.cash_allocation)
        PyCondition.is_true(
            abs(total_allocation - 1.0) < 0.01,
            f"Total allocation must equal 1.0, got {total_allocation}",
        )
        
        super().__init__(config)
        
        self.instrument: Optional[Instrument] = None
        
        # Daily timeframe indicators (for core positions)
        self.daily_ema_short = ExponentialMovingAverage(config.ema_short_period)
        self.daily_ema_medium = ExponentialMovingAverage(config.ema_medium_period)
        self.daily_ema_long = ExponentialMovingAverage(config.ema_long_period)
        self.daily_rsi = RelativeStrengthIndex(config.rsi_period)
        
        # 4-hour timeframe indicators (for tactical positions)
        self.h4_ema_short = ExponentialMovingAverage(config.ema_short_period)
        self.h4_ema_medium = ExponentialMovingAverage(config.ema_medium_period)
        self.h4_macd = MovingAverageConvergenceDivergence(12, 26)
        self.h4_rsi = RelativeStrengthIndex(config.rsi_period)
        
        # 1-hour timeframe indicators (for opportunity positions)
        self.h1_bb = BollingerBands(config.bb_period, config.bb_std)
        self.h1_rsi = RelativeStrengthIndex(config.rsi_period)
        self.h1_atr = AverageTrueRange(config.atr_period)
        
        # Strategy state
        self.market_state: Optional[MarketState] = None
        self.position_levels: Dict[PositionLevel, Decimal] = {
            PositionLevel.CORE: Decimal("0"),
            PositionLevel.TACTICAL: Decimal("0"),
            PositionLevel.OPPORTUNITY: Decimal("0"),
            PositionLevel.CASH: config.total_capital * Decimal(str(config.cash_allocation)),
        }
        
        # Stop loss orders tracking
        self.stop_orders: Dict[PositionLevel, Optional[StopMarketOrder]] = {
            PositionLevel.CORE: None,
            PositionLevel.TACTICAL: None,
            PositionLevel.OPPORTUNITY: None,
        }
        
        # Last trade prices for stop loss calculation
        self.last_trade_prices: Dict[PositionLevel, Optional[float]] = {
            PositionLevel.CORE: None,
            PositionLevel.TACTICAL: None,
            PositionLevel.OPPORTUNITY: None,
        }
        
    def on_start(self) -> None:
        """Actions to be performed on strategy start."""
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(f"Could not find instrument for {self.config.instrument_id}")
            self.stop()
            return
            
        # Register indicators for different timeframes
        # Daily indicators
        self.register_indicator_for_bars(self.config.daily_bar_type, self.daily_ema_short)
        self.register_indicator_for_bars(self.config.daily_bar_type, self.daily_ema_medium)
        self.register_indicator_for_bars(self.config.daily_bar_type, self.daily_ema_long)
        self.register_indicator_for_bars(self.config.daily_bar_type, self.daily_rsi)
        
        # 4-hour indicators
        self.register_indicator_for_bars(self.config.h4_bar_type, self.h4_ema_short)
        self.register_indicator_for_bars(self.config.h4_bar_type, self.h4_ema_medium)
        self.register_indicator_for_bars(self.config.h4_bar_type, self.h4_macd)
        self.register_indicator_for_bars(self.config.h4_bar_type, self.h4_rsi)
        
        # 1-hour indicators
        self.register_indicator_for_bars(self.config.h1_bar_type, self.h1_bb)
        self.register_indicator_for_bars(self.config.h1_bar_type, self.h1_rsi)
        self.register_indicator_for_bars(self.config.h1_bar_type, self.h1_atr)
        
        # Request historical data
        self.request_bars(
            self.config.daily_bar_type,
            start=self._clock.utc_now() - pd.Timedelta(days=3),  # 1 year for daily
        )
        self.request_bars(
            self.config.h4_bar_type,
            start=self._clock.utc_now() - pd.Timedelta(days=1),   # 2 months for 4H
        )
        self.request_bars(
            self.config.h1_bar_type,
            start=self._clock.utc_now() - pd.Timedelta(days=1),   # 2 weeks for 1H
        )
        
        # Subscribe to live data
        self.subscribe_bars(self.config.daily_bar_type)
        self.subscribe_bars(self.config.h4_bar_type)
        self.subscribe_bars(self.config.h1_bar_type)

        self.subscribe_quote_ticks(self.config.instrument_id)
        self.subscribe_trade_ticks(self.config.instrument_id)

        self.log.info("MDAS Strategy started with multi-level allocation", LogColor.GREEN)
        self._log_position_status()
        
    def on_bar(self, bar: Bar) -> None:
        """Actions to be performed when receiving a bar."""
        self.log.info(f"Received bar: {bar.bar_type} - {bar.close}", LogColor.CYAN)

        # Update market state and check signals based on bar type
        if bar.bar_type == self.config.daily_bar_type:
            self.log.info(repr(bar), LogColor.CYAN)
            self._update_market_state()
            self._check_core_signals(bar)
        elif bar.bar_type == self.config.h4_bar_type:
            self.log.info(repr(bar), LogColor.CYAN)
            self._check_tactical_signals(bar)
        elif bar.bar_type == self.config.h1_bar_type:
            self.log.info(repr(bar), LogColor.CYAN)
            self._check_opportunity_signals(bar)
            
    def _update_market_state(self) -> None:
        """Update overall market state based on daily indicators."""
        if not self._daily_indicators_ready():
            return
        print(f"It is update_market_state")
        ema_short = self.daily_ema_short.value
        ema_medium = self.daily_ema_medium.value
        ema_long = self.daily_ema_long.value
        
        # Determine market state
        if ema_short > ema_medium > ema_long:
            self.market_state = MarketState.BULL
            self.log.info("Market State: BULL MARKET", LogColor.GREEN)
        elif ema_short < ema_medium < ema_long:
            self.market_state = MarketState.BEAR
            self.log.info("Market State: BEAR MARKET", LogColor.RED)
        else:
            self.market_state = MarketState.SIDEWAYS
            self.log.info("Market State: SIDEWAYS MARKET", LogColor.YELLOW)
            
    def _check_core_signals(self, bar: Bar) -> None:
        """Check for core position signals (long-term, daily timeframe)."""
        self.log.info(f"It is check_core_signals", LogColor.BLUE)

        if not self._daily_indicators_ready():
            self.log.info(f"daily_indicators_not_ready", LogColor.RED)
            return

        current_price = bar.close
        ema_medium = self.daily_ema_medium.value
        ema_long = self.daily_ema_long.value
        rsi = self.daily_rsi.value
        
        # Core position logic: Long-term trend following
        core_target_value = self.config.total_capital * Decimal(str(self.config.core_allocation))
        current_core_value = self._get_position_value(PositionLevel.CORE, current_price)

        # self._execute_buy(PositionLevel.CORE, current_price, "Core trend following")
        # 仅作测试用
        # self._execute_sell(PositionLevel.CORE, current_price, "Core trend deterioration")
        
        # Buy signal for core position
        if (self.market_state == MarketState.BULL and
            ema_medium > ema_long and
            current_price <= ema_medium * 1.02 and  # Price near EMA50
            rsi < 60 and  # Not overbought
            current_core_value < core_target_value * Decimal("0.8")):  # Not fully allocated

            self.log.info(f"Execute BUY in check_core_signals", LogColor.MAGENTA)
            self._execute_buy(PositionLevel.CORE, current_price, "Core trend following")
            
        # Reduce core position if trend deteriorates
        elif (self.market_state == MarketState.BEAR and
              current_core_value > core_target_value * Decimal("0.2")):

            self.log.info(f"Execute SELL in check_core_signals", LogColor.MAGENTA)
            self._execute_sell(PositionLevel.CORE, current_price, "Core trend deterioration")

        else:
            self.log.info(f"No Core Signals Continue......", LogColor.YELLOW)
            
    def _check_tactical_signals(self, bar: Bar) -> None:
        """Check for tactical position signals (medium-term, 4-hour timeframe)."""
        self.log.info(f"It is check_tactical_signals", LogColor.BLUE)

        if not self._h4_indicators_ready():
            self.log.info(f"h4_indicators_not_ready", LogColor.RED)
            return
            
        current_price = bar.close
        ema_short = self.h4_ema_short.value
        ema_medium = self.h4_ema_medium.value
        macd = self.h4_macd.value
        rsi = self.h4_rsi.value
        
        tactical_target_value = self.config.total_capital * Decimal(str(self.config.tactical_allocation))
        current_tactical_value = self._get_position_value(PositionLevel.TACTICAL, current_price)

        # Buy signal for tactical position
        if (ema_short > ema_medium and
            macd > 0 and
                40 < rsi < 70 and
            current_tactical_value < tactical_target_value * Decimal("0.8")):

            self.log.info(f"Execute BUY in check_tactical_signals", LogColor.MAGENTA)
            self._execute_buy(PositionLevel.TACTICAL, current_price, "Tactical momentum")
            
        # Sell signal for tactical position
        elif ((ema_short < ema_medium or
              macd < 0 or
              rsi > 75) and
              (current_tactical_value > tactical_target_value * Decimal("0.2"))):
                self.log.info(f"Execute SELL in check_tactical_signals", LogColor.MAGENTA)
                self._execute_sell(PositionLevel.TACTICAL, current_price, "Tactical exit")

        else:
            self.log.info(f"No Tactical signals Continue......", LogColor.YELLOW)

    def _check_opportunity_signals(self, bar: Bar) -> None:
        """Check for opportunity position signals (short-term, 1-hour timeframe)."""
        self.log.info(f"It is check_opportunity_signals", LogColor.BLUE)
        if not self._h1_indicators_ready():
            self.log.info(f"h1_indicators_not_ready", LogColor.RED)
            return

        current_price = bar.close
        bb_lower = self.h1_bb.lower
        bb_middle = self.h1_bb.middle
        bb_upper = self.h1_bb.upper
        rsi = self.h1_rsi.value
        
        opportunity_target_value = self.config.total_capital * Decimal(str(self.config.opportunity_allocation))
        current_opportunity_value = self._get_position_value(PositionLevel.OPPORTUNITY, current_price)

        # self._execute_buy(PositionLevel.OPPORTUNITY, current_price, "Opportunity oversold bounce")
        #仅测试用
        # self._execute_sell(PositionLevel.OPPORTUNITY, current_price, "Opportunity profit taking")

        # Buy signal for opportunity position (mean reversion)
        if (current_price <= bb_lower and
            rsi <= self.config.rsi_oversold and
            current_opportunity_value < opportunity_target_value * Decimal("0.8")):

            self.log.info(f"Execute BUY in check_opportunity_signals", LogColor.MAGENTA)
            self._execute_buy(PositionLevel.OPPORTUNITY, current_price, "Opportunity oversold bounce")
            
        # Sell signal for opportunity position
        elif (current_price >= bb_middle or
              rsi >= self.config.rsi_overbought) and current_opportunity_value > opportunity_target_value * Decimal("0.1"):

                self.log.info(f"Execute SELL in check_opportunity_signals", LogColor.MAGENTA)
                self._execute_sell(PositionLevel.OPPORTUNITY, current_price, "Opportunity profit taking")

        else:
            self.log.info(f"No opportunity signal Continue......", LogColor.YELLOW)

    def _execute_buy(self, level: PositionLevel, price: float, reason: str) -> None:
        """Execute a buy order for the specified position level."""
        if not self.instrument:
            return
            
        # Calculate trade size
        max_trade_value = self.config.total_capital * Decimal(str(self.config.max_single_trade_ratio))
        available_cash = self.position_levels[PositionLevel.CASH]

        trade_value = self.instrument.make_price(min(max_trade_value, available_cash))
        min_notional = self.instrument.make_price(self.instrument.min_notional)

        if trade_value < min_notional:
            self.log.warning(f"Trade value {trade_value} below minimum notional {min_notional}", LogColor.YELLOW)
            return

        trade_quantity = trade_value / Decimal(str(price))
        
        # Create and submit buy order
        order = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.BUY,
            quantity=self.instrument.make_qty(trade_quantity),
            time_in_force=TimeInForce.GTC,
            tags=[level.value],  # Tag order with position level
        )
        
        self.submit_order(order)
        
        # Update position tracking
        self.position_levels[PositionLevel.CASH] -= trade_value
        self.last_trade_prices[level] = price
        
        self.log.info(
            f"{level.value} BUY: {trade_quantity:.6f} @ PRICE: {float(price):.4f} - Reason: {reason}",
            LogColor.GREEN,
        )
        
        # Set stop loss
        self._set_stop_loss(level, price, trade_quantity)
        
    def _execute_sell(self, level: PositionLevel, price: float, reason: str) -> None:
        """Execute a sell order for the specified position level."""
        if not self.instrument:
            return
            
        # Get current position for this level
        # position = self.portfolio(self.config.instrument_id)
        if self.portfolio.is_flat(self.config.instrument_id):
            self.log.info(f"No position to sell for {level.value}", LogColor.YELLOW)
            return
            
        # Calculate sell quantity (partial sell based on level allocation)
        level_allocation = getattr(self.config, f"{level.value.lower()}_allocation")
        total_quantity = self.portfolio.net_position(self.config.instrument_id)
        sell_ratio = Decimal("0.3")  # Sell 30% of level allocation
        sell_quantity = self.instrument.make_qty(total_quantity * Decimal(str(level_allocation)) * sell_ratio)
        min_qty = self.instrument.make_qty(self.instrument.min_quantity)

        if sell_quantity < min_qty:
            self.log.warning(f"Sell Quantity {sell_quantity} below minimum qty", LogColor.YELLOW)
            return
            
        # Create and submit sell order
        order = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.SELL,
            quantity=self.instrument.make_qty(sell_quantity),
            time_in_force=TimeInForce.GTC,
            tags=[level.value],
        )
        
        self.submit_order(order)
        
        # Update cash position
        sell_value = sell_quantity * Decimal(str(price))
        self.position_levels[PositionLevel.CASH] += sell_value
        
        self.log.info(
            f"{level.value} SELL: {float(sell_quantity):.6f} @ Price: {float(price):.4f} - Reason: {reason}",
            LogColor.GREEN,
        )
        
    def _set_stop_loss(self, level: PositionLevel, entry_price: float, quantity: Decimal) -> None:
        """Set stop loss order for the position level."""
        if not self.instrument:
            return
            
        # Get stop loss ratio for this level
        if level == PositionLevel.CORE:
            stop_ratio = self.config.core_stop_loss
        elif level == PositionLevel.TACTICAL:
            stop_ratio = self.config.tactical_stop_loss
        else:  # OPPORTUNITY
            stop_ratio = self.config.opportunity_stop_loss
            
        stop_price = entry_price * (1 - stop_ratio)
        
        # Cancel existing stop loss for this level
        if self.stop_orders[level]:
            self.cancel_order(self.stop_orders[level])
            
        # Create new stop loss order
        stop_order = self.order_factory.stop_market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.SELL,
            quantity=self.instrument.make_qty(quantity),
            trigger_price=self.instrument.make_price(stop_price),
            time_in_force=TimeInForce.GTC,
            reduce_only=True,
            tags=[f"{level.value}_STOP"],
        )
        
        self.submit_order(stop_order)
        self.stop_orders[level] = stop_order
        
        self.log.info(
            f"Stop loss set for {level.value}: {float(stop_price):.4f} stop_ratio: (-{stop_ratio*100:.1f}%)",
            LogColor.MAGENTA,
        )
        
    def _get_position_value(self, level: PositionLevel, current_price: float) -> Decimal:
        """Get the current value of a position level."""
        # position = self.portfolio(self.config.instrument_id)
        if self.portfolio.is_flat(self.config.instrument_id):
            return Decimal("0")
            
        # Estimate level allocation (simplified)
        level_allocation = getattr(self.config, f"{level.value.lower()}_allocation")
        total_value = self.portfolio.net_position(self.config.instrument_id) * Decimal(str(current_price))
        return total_value * Decimal(str(level_allocation))
        
    def _daily_indicators_ready(self) -> bool:
        """Check if daily indicators are ready."""
        return (self.daily_ema_short.initialized and
                self.daily_ema_medium.initialized and
                self.daily_ema_long.initialized and
                self.daily_rsi.initialized)
                
    def _h4_indicators_ready(self) -> bool:
        """Check if 4-hour indicators are ready."""
        return (self.h4_ema_short.initialized and
                self.h4_ema_medium.initialized and
                self.h4_macd.initialized and
                self.h4_rsi.initialized)
                
    def _h1_indicators_ready(self) -> bool:
        """Check if 1-hour indicators are ready."""
        return (self.h1_bb.initialized and
                self.h1_rsi.initialized and
                self.h1_atr.initialized)
                
    def _log_position_status(self) -> None:
        """Log current position status."""
        self.log.info("=== Position Status ===", LogColor.BLUE)
        for level, value in self.position_levels.items():
            percentage = (value / self.config.total_capital) * 100
            self.log.info(f"{level.value}: {value:.2f} USDT ({percentage:.1f}%)", LogColor.BLUE)
        self.log.info("=====================", LogColor.BLUE)
        
    def on_event(self, event: Event) -> None:
        """Handle order fill events."""
        if isinstance(event, OrderFilled):
            self.log.info(
                f"Order filled: {event.order_side} {event.last_qty} @ {event.last_px}",
                LogColor.GREEN,
            )
            self._log_position_status()
            
    def on_data(self, data: Data) -> None:
        """Handle custom data."""
        pass
        
    def on_stop(self) -> None:
        """Actions to be performed when strategy is stopped."""
        # Cancel all open orders
        self.cancel_all_orders(self.config.instrument_id)
        
        # Note: For spot trading, we typically don't close all positions
        # as they can be held long-term. Only close if explicitly required.
        
        # Unsubscribe from data
        self.unsubscribe_bars(self.config.daily_bar_type)
        self.unsubscribe_bars(self.config.h4_bar_type)
        self.unsubscribe_bars(self.config.h1_bar_type)
        self.unsubscribe_quote_ticks(self.config.instrument_id)
        self.unsubscribe_trade_ticks(self.config.instrument_id)

        self.log.info("MDAS Strategy stopped", LogColor.YELLOW)
        
    def on_reset(self) -> None:
        """Reset all indicators and state."""
        # Reset indicators
        self.daily_ema_short.reset()
        self.daily_ema_medium.reset()
        self.daily_ema_long.reset()
        self.daily_rsi.reset()
        self.h4_ema_short.reset()
        self.h4_ema_medium.reset()
        self.h4_macd.reset()
        self.h4_rsi.reset()
        self.h1_bb.reset()
        self.h1_rsi.reset()
        self.h1_atr.reset()
        
        # Reset strategy state
        self.market_state = None
        self.position_levels = {
            PositionLevel.CORE: Decimal("0"),
            PositionLevel.TACTICAL: Decimal("0"),
            PositionLevel.OPPORTUNITY: Decimal("0"),
            PositionLevel.CASH: self.config.total_capital * Decimal(str(self.config.cash_allocation)),
        }
        self.stop_orders = {level: None for level in [PositionLevel.CORE, PositionLevel.TACTICAL, PositionLevel.OPPORTUNITY]}
        self.last_trade_prices = {level: None for level in [PositionLevel.CORE, PositionLevel.TACTICAL, PositionLevel.OPPORTUNITY]}

    # def on_quote_tick(self, tick: QuoteTick) -> None:
    #     """
    #     Actions to be performed when the strategy is running and receives a quote tick.
    #
    #     Parameters
    #     ----------
    #     tick : QuoteTick
    #         The tick received.
    #
    #     """
    #     # For debugging (must add a subscription)
    #     self.log.info(repr(tick), LogColor.GREEN)
    #
    # def on_trade_tick(self, tick: TradeTick) -> None:
    #     """
    #     Actions to be performed when the strategy is running and receives a trade tick.
    #
    #     Parameters
    #     ----------
    #     tick : TradeTick
    #         The tick received.
    #
    #     """
    #     # For debugging (must add a subscription)
    #     self.log.info(repr(tick), LogColor.CYAN)

    def on_save(self) -> dict[str, bytes]:
        """Save strategy state."""
        return {}
        
    def on_load(self, state: dict[str, bytes]) -> None:
        """Load strategy state."""
        pass
        
    def on_dispose(self) -> None:
        """Cleanup resources."""
        pass

    def configue_parameters(self, config: MDASConfig) -> None:
        """Configure strategy parameters."""
        PyCondition.is_true(
            config.ema_short_period < config.ema_medium_period,
            f"{config.ema_short_period=} must be less than {config.ema_medium_period=}",
        )
        PyCondition.is_true(
            config.ema_medium_period < config.ema_long_period,
            f"{config.ema_medium_period=} must be less than {config.ema_long_period=}",
        )
        PyCondition.is_true(
            config.macd < config.macd_slow_period,
            f"{config.macd_fast_period=} must be less than {config.macd_slow_period=}",
        )
        PyCondition.is_true(
            config.rsi_oversold < config.rsi_overbought,
            f"{config.rsi_oversold=} must be less than {config.rsi_overbought=}",
        )