# -------------------------------------------------------------------------------------------------
#  Copyright (C) 2015-2025 Nautech Systems Pty Ltd. All rights reserved.
#  https://nautechsystems.io
#
#  Licensed under the GNU Lesser General Public License Version 3.0 (the "License");
#  You may not use this file except in compliance with the License.
#  You may obtain a copy of the License at https://www.gnu.org/licenses/lgpl-3.0.en.html
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -------------------------------------------------------------------------------------------------

import asyncio
from collections import defaultdict
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any, Union
import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from nautilus_trader.common.actor import ActorConfig
from nautilus_trader.common.component import TimeEvent

from nautilus_trader.common.enums import LogColor
from nautilus_trader.config import PositiveFloat
from nautilus_trader.config import PositiveInt
from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.data import Data
from nautilus_trader.core.message import Event
from nautilus_trader.model.objects import Quantity
from nautilus_trader.model.objects import Price
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import OrderStatus
from nautilus_trader.model.enums import PositionSide
from nautilus_trader.model.enums import TimeInForce
from nautilus_trader.model.events import OrderFilled
from nautilus_trader.model.events import PositionClosed
from nautilus_trader.model.events import PositionOpened
from nautilus_trader.model.identifiers import ClientId
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import ComponentId
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.model.orders import Order
from nautilus_trader.model.position import Position
from nautilus_trader.trading.strategy import Strategy

from triangular_arbitrage_agents import TriangularArbitrageAgentManager
from triangular_arbitrage_agents import TripletSelectionAgent
from triangular_arbitrage_agents import MarketSituationAgent
from triangular_arbitrage_agents import ParameterOptimizationAgent

from llm_enhanced_agents import (
    LLMEnhancedTripletSelectionAgent,
    LLMEnhancedMarketSituationAgent, 
    LLMEnhancedParameterOptimizationAgent,
)


@dataclass
class GetTripletSelectionAlysis(Event):
    """ Get alysis from TripletSelection agent."""
    analysis: Dict[str, Any]
    Topic: str = "triplet_selection"

@dataclass
class GetMarketSituationAlysis(Event):
    """ Get alysis from MarketSituation agent."""
    analysis: Dict[str, Any]
    Topic: str = "market_situation"

@dataclass
class GetParameterOptimizationAlysis(Event):
    """ Get alysis from ParameterOptimization agent."""
    analysis: Dict[str, Any]
    Topic: str = "parameter_optimization"

class TriangularArbitrageConfig(StrategyConfig, frozen=True):
    """
    Configuration for ``TriangularArbitrageStrategy`` instances.
    
    Parameters
    ----------
    base_currency : str
        Base currency for triangular arbitrage (e.g., "BTC", "ETH", "USDT").
    initial_triplet : Tuple[str, str, str]
        Initial triplet for triangular arbitrage (base, currency1, currency2).
    candidate_currencies : List[str]
        List of candidate currencies for triangular arbitrage.
    trade_size : Decimal
        The base position size per trade.
    min_profit_threshold : Decimal
        The minimum profit threshold for executing triangular arbitrage.
    max_position_size : Decimal
        Maximum position size limit.
    max_drawdown : Decimal
        Maximum allowed drawdown before pausing trading.
    position_timeout_mins : int
        Timeout in minutes for closing positions.
    transaction_cost_bps : int
        Transaction cost in basis points per trade.
    enable_llm_agents : bool
        Whether to enable LLM-enhanced intelligent agents.
    enable_triplet_selection : bool
        Whether to enable intelligent triplet selection.
    enable_market_situation_awareness : bool
        Whether to enable market situation awareness.
    enable_parameter_optimization : bool
        Whether to enable parameter optimization.
    triplet_selection_agent_config : Dict[str, Any]
        Configuration for triplet selection agent.
    market_situation_agent_config : Dict[str, Any]
        Configuration for market situation agent.
    parameter_optimization_agent_config : Dict[str, Any]
        Configuration for parameter optimization agent.
    triplet_selection_interval_mins : int
        Interval for triplet selection updates.
    market_situation_check_interval_mins : int
        Interval for market situation checks.
    parameter_optimization_interval_mins : int
        Interval for parameter optimization updates.
    client_id : ClientId
        Client ID for spot trading.
    """

    base_currency: str
    trade_size: Decimal
    initial_triplet: Tuple[str, str, str] | None = None
    candidate_currencies: List[str] | None = None

    min_profit_threshold: Decimal = Decimal("0.002")  # 0.2%
    max_position_size: Decimal = Decimal("10.0")
    max_drawdown: Decimal = Decimal("0.05")
    position_timeout_mins: int = 5
    transaction_cost_bps: int = 10  # 0.1%
    enable_llm_agents: bool = True
    triplet_selection_agent_config: Dict[str, Any] | None = None
    market_situation_agent_config: Dict[str, Any] | None = None
    parameter_optimization_agent_config: Dict[str, Any] | None = None
    agent_interval_mins: int = 60
    client_id: ClientId | None = None

class TriangularArbitrageStrategy(Strategy):
    """
    Advanced triangular arbitrage strategy with intelligent agents.
    
    This strategy:
    1. Identifies arbitrage opportunities between three currencies (A->B->C->A)
    2. Executes simultaneous trades to capture price discrepancies
    3. Uses intelligent agents for triplet selection, market awareness, and optimization
    4. Implements sophisticated risk management and transaction cost analysis
    
    Parameters
    ----------
    config : TriangularArbitrageConfig
        The configuration for the strategy instance.
    """

    def __init__(self, config: TriangularArbitrageConfig) -> None:
        super().__init__(config)
        
        # Current parameters (can be updated by agents)
        self.current_trade_size = config.trade_size
        self.current_min_profit_threshold = config.min_profit_threshold
        self.current_max_position_size = config.max_position_size
        self.current_position_timeout_mins = config.position_timeout_mins
        self.current_transaction_cost_bps = config.transaction_cost_bps
        
        # Triplet management
        self.current_triplet: Optional[Tuple[str, str, str]] = config.initial_triplet
        self.triplet_instruments: Dict[str, Instrument] = {}
        self.available_triplets: List[Tuple[str, str, str]] = []
        
        # Market data
        self.quote_ticks: Dict[str, QuoteTick] = {}
        self.price_history: Dict[str, List[Dict]] = defaultdict(list)
        
        # Position tracking
        self.active_arbitrage_positions: Dict[str, Dict] = {}
        self.position_timers: Dict[str, pd.Timestamp] = {}
        
        # Performance tracking
        self.trade_history: List[Dict] = []
        self.total_pnl = Decimal("0.0")
        self.max_drawdown_current = Decimal("0.0")
        self.high_water_mark = Decimal("0.0")
        self.successful_arbitrages = 0
        self.failed_arbitrages = 0
        
        # Risk management
        self.is_paused = False
        self.pause_reason = ""
        
        # Intelligent agents (support both traditional and LLM-enhanced)
        self.agent_manager = TriangularArbitrageAgentManager(self)
        self.llm_agents: Dict[str, Any] = {}
        self.llm_agents_analysis: Dict[str, Any] = {}
        
        # Arbitrage analysis
        self.arbitrage_opportunities_history = []
        self.last_arbitrage_scan = None
        
    def on_start(self) -> None:
        """Actions to be performed on strategy start."""
        self.log.info("Starting triangular arbitrage strategy", LogColor.GREEN)
        
        # Generate available triplets
        self._generate_available_triplets()
        
        # Initialize with first available triplet
        if self.available_triplets:
            self.current_triplet = self.available_triplets[0]
            self._initialize_triplet_instruments()
            self._subscribe_to_triplet_data()
        else:
            self.log.error("No valid triplets found")
            self.stop()
            return
        
        # Initialize intelligent agents
        self._initialize_llm_agents()

        # Subscribe to agent analysis messages
        self._subscribe_to_agent_messages()
        
        # Start monitoring timers
        self._start_monitoring_timers()

        self.clock.set_timer(
            name="llm_analysis",
            interval=pd.Timedelta(seconds=40),
            callback=self._get_llm_analysis,
        )
        
    def _generate_available_triplets(self) -> None:
        """Generate all possible triangular arbitrage triplets."""
        base = self.config.base_currency
        
        # Use provided candidate currencies or derive from initial triplet
        if self.config.candidate_currencies:
            candidates = self.config.candidate_currencies
        elif self.config.initial_triplet:
            # Extract candidate currencies from initial triplet
            _, currency1, currency2 = self.config.initial_triplet
            candidates = [currency1, currency2]
        else:
            # Default candidates
            candidates = ["BTC", "ETH", "BNB", "ADA", "DOT"]
        
        # Generate all possible triplets of the form (base, currency1, currency2)
        # This creates arbitrage paths like USDT -> BTC -> ETH -> USDT
        for combo in itertools.combinations(candidates, 2):
            triplet = (base, combo[0], combo[1])
            self.available_triplets.append(triplet)
            
        self.log.info(f"Generated {len(self.available_triplets)} possible arbitrage triplets")
        
    def _initialize_triplet_instruments(self) -> None:
        """Initialize instruments for current triplet."""
        if not self.current_triplet:
            return
            
        base, currency1, currency2 = self.current_triplet
        
        # Create instrument IDs for the three pairs
        pair1_id = InstrumentId.from_str(f"{currency1}{base}.BINANCE")  # BTC/USDT
        pair2_id = InstrumentId.from_str(f"{currency2}{base}.BINANCE")  # ETH/USDT
        pair3_id = InstrumentId.from_str(f"{currency2}{currency1}.BINANCE")  # ETH/BTC
        
        # Get instruments from cache
        self.triplet_instruments = {
            "pair1": self.cache.instrument(pair1_id),  # BTC/USDT
            "pair2": self.cache.instrument(pair2_id),  # ETH/USDT
            "pair3": self.cache.instrument(pair3_id),  # ETH/BTC
        }
        
        # Check if all instruments are available
        if not all(self.triplet_instruments.values()):
            self.log.error(f"Could not find all instruments for triplet {self.current_triplet}")
            return
            
        self.log.info(f"Initialized triplet: {base}->{currency1}->{currency2}->{base}")
        
    def _subscribe_to_triplet_data(self) -> None:
        """Subscribe to market data for current triplet."""
        if not self.triplet_instruments:
            return
            
        for pair_key, instrument in self.triplet_instruments.items():
            if instrument:
                self.subscribe_quote_ticks(instrument.id)
                self.subscribe_trade_ticks(instrument.id)
                self.subscribe_bars(BarType.from_str(f"{instrument.id}-1-MINUTE-LAST-EXTERNAL"))
                self.request_bars(
                    BarType.from_str(f"{instrument.id}-1-MINUTE-LAST-EXTERNAL"),
                    start=self._clock.utc_now() - pd.Timedelta(days=1),
                )
                
    def _unsubscribe_from_triplet_data(self) -> None:
        """Unsubscribe from market data for current triplet."""
        if not self.triplet_instruments:
            return
            
        for pair_key, instrument in self.triplet_instruments.items():
            if instrument:
                self.unsubscribe_quote_ticks(instrument.id)
                self.unsubscribe_trade_ticks(instrument.id)
                self.unsubscribe_bars(BarType.from_str(f"{instrument.id}-1-MINUTE-LAST-EXTERNAL"))
        
    def _initialize_llm_agents(self) -> None:
        """Initialize LLM-enhanced intelligent agents."""
        if self.config.enable_llm_agents:
            self.log.info("Initializing LLM-enhanced intelligent agents", LogColor.CYAN)
            if self.config.triplet_selection_agent_config:
                triplet_agent = LLMEnhancedTripletSelectionAgent(
                    actor_config=ActorConfig(component_id=ComponentId("triplet_selection_agent")),
                    config=self.config.triplet_selection_agent_config,
                    msgbus=self.msgbus,
                    cache=self.cache,
                    clock=self.clock,
                )
                self.llm_agents["triplet_selection"] = triplet_agent

            if self.config.market_situation_agent_config:
                market_agent = LLMEnhancedMarketSituationAgent(
                    actor_config=ActorConfig(component_id=ComponentId("market_situation_agent")),
                    config=self.config.market_situation_agent_config,
                    msgbus=self.msgbus,
                    cache=self.cache,
                    clock=self.clock,
                )
                self.llm_agents["market_situation"] = market_agent

            if self.config.parameter_optimization_agent_config:
                param_agent = LLMEnhancedParameterOptimizationAgent(
                    actor_config=ActorConfig(component_id=ComponentId("parameter_optimization_agent")),
                    config=self.config.parameter_optimization_agent_config,
                    msgbus=self.msgbus,
                    cache=self.cache,
                    clock=self.clock,
                )
                self.llm_agents["parameter_optimization"] = param_agent
        
    def _subscribe_to_agent_messages(self) -> None:
        """Subscribe to LLM agent analysis messages."""
        if "triplet_selection" in self.llm_agents:
            self.msgbus.subscribe(
                topic=GetTripletSelectionAlysis.Topic,
                handler=self.on_agent_analysis,
            )
            
        if "market_situation" in self.llm_agents:
            self.msgbus.subscribe(
                topic=GetMarketSituationAlysis.Topic,
                handler=self.on_agent_analysis,
            )
            
        if "parameter_optimization" in self.llm_agents:
            self.msgbus.subscribe(
                topic=GetParameterOptimizationAlysis.Topic,
                handler=self.on_agent_analysis,
            )

    def _get_llm_analysis(self, event: TimeEvent) -> Optional[Dict[str, Any]]:
        """Get LLM analysis for a specific topic."""
        self.log.info(f"Get LLM analysis, Canceling the timers for trade scanning, position timeout check, and risk management check", LogColor.CYAN)

        self.clock.cancel_timer("arbitrage_scanner")
        self.clock.cancel_timer("position_timeout_check")
        self.clock.cancel_timer("risk_management_check")

        triplet_selection_analysis = {}
        market_situation_analysis = {}
        parameter_optimization_analysis = {}
        for agent_topic, agent in self.llm_agents.items():
            if agent_topic == "triplet_selection":
                triplet_selection_analysis = agent.analyze()
            elif agent_topic == "market_situation":
                market_situation_analysis = agent.analyze()
            elif agent_topic == "parameter_optimization":
                parameter_optimization_analysis = agent.analyze()

        if triplet_selection_analysis:
            event = GetTripletSelectionAlysis(triplet_selection_analysis)
            self.msgbus.publish(GetTripletSelectionAlysis.Topic, event)
        if market_situation_analysis:
            event = GetMarketSituationAlysis(market_situation_analysis)
            self.msgbus.publish(GetMarketSituationAlysis.Topic, event)
        if parameter_optimization_analysis:
            event = GetParameterOptimizationAlysis(parameter_optimization_analysis)
            self.msgbus.publish(GetParameterOptimizationAlysis.Topic, event)

        self._start_monitoring_timers()

    def on_agent_analysis(self, event: Union[GetTripletSelectionAlysis, GetMarketSituationAlysis, GetParameterOptimizationAlysis])-> None:
        """Handle LLM agent analysis messages."""
        self.log.info(f"Received analysis from {event.Topic}", LogColor.CYAN)

        # Store analysis for later processing
        self.llm_agents_analysis[event.Topic] = event.analysis

        # Dispatch to appropriate handler based on topic
        if event.Topic == "triplet_selection":
            self._handle_triplet_selection_analysis(event.analysis)
        elif event.Topic == "market_situation":
            self._handle_market_situation_analysis(event.analysis)
        elif event.Topic == "parameter_optimization":
            self._handle_parameter_optimization_analysis(event.analysis)

    def _handle_triplet_selection_analysis(self, analysis: Dict[str, Any]) -> None:
        """Handle triplet selection analysis from LLM agent."""
        self.log.info("Received triplet selection analysis from LLM agent")
        
        recommendations = analysis.get("recommendations", {})
        best_triplet = recommendations.get("best_triplet")
        
        if best_triplet and best_triplet != str(self.current_triplet):
            try:
                # Parse triplet string back to tuple
                import ast
                new_triplet = ast.literal_eval(best_triplet)
                if isinstance(new_triplet, tuple) and len(new_triplet) == 3 and all(isinstance(x, str) for x in new_triplet):
                    self.update_triplet(new_triplet)
            except Exception as e:
                self.log.error(f"Error parsing triplet recommendation: {e}")
                
    def _handle_market_situation_analysis(self, analysis: Dict[str, Any]) -> None:
        """Handle market situation analysis from LLM agent."""
        self.log.info("Received market situation analysis from LLM agent")
        
        recommendations = analysis.get("recommendations", {})
        trading_activity = recommendations.get("trading_activity", "normal")

        if trading_activity == "pause":
            self.is_paused = True
            self.pause_reason = "llm_market_analysis"
            self.log.warning("Trading paused due to LLM market analysis")
        elif trading_activity == "normal" and self.pause_reason == "llm_market_analysis":
            self.is_paused = False
            self.pause_reason = ""
            self.log.info("Trading resumed after LLM market analysis")
            
        # Apply position size and profit threshold multipliers
        if "position_size_multiplier" in recommendations:
            multiplier = recommendations["position_size_multiplier"]
            self.current_trade_size = self.config.trade_size * Decimal(str(multiplier))
            self.log.info(f"Adjusted trade size to {self.current_trade_size} based on LLM analysis")
            
        if "profit_threshold_multiplier" in recommendations:
            multiplier = recommendations["profit_threshold_multiplier"]
            self.current_min_profit_threshold = self.config.min_profit_threshold * Decimal(str(multiplier))
            self.log.info(f"Adjusted profit threshold to {self.current_min_profit_threshold} based on LLM analysis")

        self.adjust_for_market_conditions(analysis)
    
    def _handle_parameter_optimization_analysis(self, analysis: Dict[str, Any]) -> None:
        """Handle parameter optimization analysis from LLM agent."""
        self.log.info("Received parameter optimization analysis from LLM agent")
        
        recommendations = analysis.get("recommendations", {})
        optimized_params = analysis.get("optimized_parameters", {})
        
        # Apply parameter optimizations with confidence weighting
        confidence = analysis.get("overall_confidence", 0.7)
        
        if confidence > 0.6:  # Only apply if confident enough
            self.update_parameters(optimized_params)
        
    def _start_monitoring_timers(self) -> None:
        """Start monitoring timers."""
        self.clock.set_timer(
            name="arbitrage_scanner",
            interval=pd.Timedelta(seconds=2),  # 2 seconds
            callback=self._scan_arbitrage_opportunities,
        )
        
        self.clock.set_timer(
            name="position_timeout_check",
            interval=pd.Timedelta(seconds=10),  # 10 seconds
            callback=self._check_position_timeouts,
        )
        
        self.clock.set_timer(
            name="risk_management_check",
            interval=pd.Timedelta(seconds=30),  # 30 seconds
            callback=self._risk_management_check,
        )
        
    def on_quote_tick(self, tick: QuoteTick) -> None:
        """Handle incoming quote ticks."""
        self.quote_ticks[str(tick.instrument_id)] = tick
        
        # Store price history for analysis
        self.price_history[str(tick.instrument_id)].append({
            "timestamp": tick.ts_event,
            "bid": float(tick.bid_price),
            "ask": float(tick.ask_price),
            "mid": float(tick.bid_price.as_decimal() + tick.ask_price.as_decimal() / Decimal(2)),
            "spread": float(tick.ask_price - tick.bid_price),
        })
        
        # Keep only recent history (last 1000 ticks)
        if len(self.price_history[str(tick.instrument_id)]) > 1000:
            self.price_history[str(tick.instrument_id)] = self.price_history[str(tick.instrument_id)][-1000:]
            
    def on_trade_tick(self, tick: TradeTick) -> None:
        """Handle incoming trade ticks."""
        self.log.debug(f"Trade tick: {tick.instrument_id} @ {tick.price} size {tick.size}")
        
    def _scan_arbitrage_opportunities(self, event: TimeEvent) -> None:
        """Scan for triangular arbitrage opportunities."""
        self.log.info(f"Scanning for triangular arbitrage opportunities name: {event.name}", LogColor.CYAN)
        if self.is_paused or not self.current_triplet:
            return
            
        opportunity = self._calculate_triangular_arbitrage_opportunity()
        if opportunity and opportunity["net_profit_rate"] >= self.current_min_profit_threshold:
            self.log.info(f"Found arbitrage opportunity: {opportunity['path']}, "
                          f"profit rate: {opportunity['net_profit_rate']:.4f}", LogColor.BLUE)
            self._execute_triangular_arbitrage(opportunity)
            
    def _calculate_triangular_arbitrage_opportunity(self) -> Optional[Dict]:
        """Calculate triangular arbitrage opportunity for current triplet."""
        if not self.current_triplet or not self.triplet_instruments:
            self.log.warning(f"Current triplet or instruments not initialized")
            return None
            
        # Get current quotes for all three pairs
        quotes = {}
        for pair_key, instrument in self.triplet_instruments.items():
            if instrument:
                quote = self.quote_ticks.get(str(instrument.id))
                if not quote:
                    self.log.warning(f"No quote for instrument {instrument.id}")
                    return None
                quotes[pair_key] = quote
                
        if len(quotes) != 3:
            self.log.warning(f"quotes do not match expected triplet size: 3")
            return None
            
        base, currency1, currency2 = self.current_triplet
        
        # Calculate arbitrage paths
        # Path 1: base -> currency1 -> currency2 -> base
        path1_opportunity = self._calculate_arbitrage_path(
            quotes, "forward", base, currency1, currency2
        )
        
        # Path 2: base -> currency2 -> currency1 -> base
        path2_opportunity = self._calculate_arbitrage_path(
            quotes, "reverse", base, currency2, currency1
        )
        
        # Return the more profitable opportunity
        if path1_opportunity and path2_opportunity:
            if path1_opportunity["net_profit_rate"] > path2_opportunity["net_profit_rate"]:
                self.log.info(f"finding opportunity and Path 1 is more profitable", LogColor.GREEN)
                return path1_opportunity
            else:
                self.log.info(f"finding opportunity and Path 2 is more profitable", LogColor.GREEN)
                return path2_opportunity
        elif path1_opportunity:
            self.log.info(f"Only find Path1 opportunity", LogColor.GREEN)
            return path1_opportunity
        elif path2_opportunity:
            self.log.info(f"Only find Path2 opportunity", LogColor.GREEN)
            return path2_opportunity
        else:
            self.log.warning(f"No path1 opportunity or path2 opportunity found")
            return None
            
    def _calculate_arbitrage_path(self, quotes: Dict, direction: str, base: str, 
                                  currency1: str, currency2: str) -> Optional[Dict]:
        """Calculate arbitrage opportunity for a specific path."""
        try:
            # Transaction cost rate
            cost_rate = self.current_transaction_cost_bps / 10000.0
            
            if direction == "forward":
                # Path: base -> currency1 -> currency2 -> base
                # Step 1: Buy currency1 with base (use ask price)
                pair1_quote = quotes["pair1"]  # currency1/base
                rate1 = 1.0 / float(pair1_quote.ask_price)  # base -> currency1
                
                # Step 2: Buy currency2 with currency1 (use ask price)
                pair3_quote = quotes["pair3"]  # currency2/currency1
                rate2 = 1.0 / float(pair3_quote.ask_price)  # currency1 -> currency2
                
                # Step 3: Sell currency2 for base (use bid price)
                pair2_quote = quotes["pair2"]  # currency2/base
                rate3 = float(pair2_quote.bid_price)  # currency2 -> base
                
                # Calculate final amount after all conversions
                final_amount = rate1 * rate2 * rate3
                
                # Account for transaction costs (3 trades)
                final_amount_after_costs = final_amount * ((1 - cost_rate) ** 3)
                
                path_info = {
                    "direction": "forward",
                    "path": f"{base}->{currency1}->{currency2}->{base}",
                    "rates": [rate1, rate2, rate3],
                    "pair_prices": [float(pair1_quote.ask_price), float(pair3_quote.ask_price), float(pair2_quote.bid_price)],
                }
                
            else:  # reverse
                # Path: base -> currency2 -> currency1 -> base
                # Step 1: Buy currency2 with base (use ask price)
                pair2_quote = quotes["pair2"]  # currency2/base
                rate1 = 1.0 / float(pair2_quote.ask_price)  # base -> currency2
                
                # Step 2: Buy currency1 with currency2 (use bid price)
                pair3_quote = quotes["pair3"]  # currency2/currency1
                rate2 = float(pair3_quote.bid_price)  # currency2 -> currency1
                
                # Step 3: Sell currency1 for base (use bid price)
                pair1_quote = quotes["pair1"]  # currency1/base
                rate3 = float(pair1_quote.bid_price)  # currency1 -> base
                
                # Calculate final amount after all conversions
                final_amount = rate1 * rate2 * rate3
                
                # Account for transaction costs (3 trades)
                final_amount_after_costs = final_amount * ((1 - cost_rate) ** 3)
                
                path_info = {
                    "direction": "reverse",
                    "path": f"{base}->{currency2}->{currency1}->{base}",
                    "rates": [rate1, rate2, rate3],
                    "pair_prices": [float(pair2_quote.ask_price), float(pair3_quote.bid_price), float(pair1_quote.bid_price)],
                }
                
            # Calculate profit
            gross_profit_rate = final_amount - 1.0
            net_profit_rate = final_amount_after_costs - 1.0
            
            if net_profit_rate > 0:
                return {
                    "triplet": self.current_triplet,
                    "timestamp": self.clock.timestamp_ns(),
                    "gross_profit_rate": gross_profit_rate,
                    "net_profit_rate": net_profit_rate,
                    "final_amount": final_amount,
                    "final_amount_after_costs": final_amount_after_costs,
                    "transaction_cost_rate": cost_rate,
                    "quotes": quotes,
                    **path_info
                }
            else:
                self.log.info(f"{direction} path net_profit_rate is negative: {net_profit_rate:.4f}, ", LogColor.BLUE)
                return None
                
        except Exception as e:
            self.log.error(f"Error calculating arbitrage path: {e}")
            return None
            
    def _execute_triangular_arbitrage(self, opportunity: Dict) -> None:
        """Execute triangular arbitrage opportunity."""
        if not self.triplet_instruments:
            return
            
        # Check position limits
        current_exposure = self._get_current_exposure()
        if current_exposure + self.current_trade_size > self.current_max_position_size:
            self.log.warning("Position limit reached, skipping arbitrage")
            return
            
        self.log.info(f"Executing triangular arbitrage: {opportunity['path']}, "
                     f"expected profit: {opportunity['net_profit_rate']:.4f}", LogColor.YELLOW)
        
        try:
            # Generate unique position ID
            position_id = f"arb_{self.clock.timestamp_ns()}"
            
            # Execute the three trades simultaneously
            orders = self._create_arbitrage_orders(opportunity)
            
            if len(orders) == 3:
                # Submit all orders
                self.log.info(f"Submitting orders for position {position_id}", LogColor.CYAN)
                for order in orders:
                    self.submit_order(order, client_id=self.config.client_id)
                    self.log.info(f"Order submitted: {order}", LogColor.BLUE)
                # Track the arbitrage position
                self._track_arbitrage_position(position_id, opportunity, orders)
                
            else:
                self.log.error("Failed to create all required orders for arbitrage")
                
        except Exception as e:
            self.log.error(f"Error executing triangular arbitrage: {e}")
            self.failed_arbitrages += 1

    def _create_arbitrage_orders(self, opportunity: Dict) -> List[Order]:
        """Create the three orders required for triangular arbitrage."""
        orders = []
        direction = opportunity["direction"]
        base, currency1, currency2 = opportunity["triplet"]

        try:
            if direction == "forward":
                # Path: base -> currency1 -> currency2 -> base
                pair1_instrument = self.triplet_instruments["pair1"]
                pair3_instrument = self.triplet_instruments["pair3"]
                pair2_instrument = self.triplet_instruments["pair2"]
                qty_list = self._calculate_qty(direction, opportunity)

                # Order 1: Buy currency1 with base (pair1: currency1/base)
                if pair1_instrument:
                    order1 = self.order_factory.market(
                        instrument_id=pair1_instrument.id,
                        order_side=OrderSide.BUY,
                        quantity=qty_list[0],
                        time_in_force=TimeInForce.GTC,
                    )
                    orders.append(order1)

                # Order 2: Buy currency2 with currency1 (pair3: currency2/currency1)
                # Calculate quantity in currency1 terms

                if pair3_instrument:
                    order2 = self.order_factory.market(
                        instrument_id=pair3_instrument.id,
                        order_side=OrderSide.BUY,
                        quantity=qty_list[1],
                        time_in_force=TimeInForce.GTC,
                    )
                    orders.append(order2)

                # Order 3: Sell currency2 for base (pair2: currency2/base)
                if pair2_instrument:
                    # Calculate quantity in currency2 terms
                    order3 = self.order_factory.market(
                        instrument_id=pair2_instrument.id,
                        order_side=OrderSide.SELL,
                        quantity=qty_list[2],
                        time_in_force=TimeInForce.GTC,
                    )
                    orders.append(order3)

            else:  # reverse
                # Path: base -> currency2 -> currency1 -> base
                qty_list = self._calculate_qty(direction, opportunity)
                # Order 1: Buy currency2 with base (pair2: currency2/base)
                pair2_instrument = self.triplet_instruments["pair2"]
                pair3_instrument = self.triplet_instruments["pair3"]
                pair1_instrument = self.triplet_instruments["pair1"]
                if pair2_instrument:
                    order1 = self.order_factory.market(
                        instrument_id=pair2_instrument.id,
                        order_side=OrderSide.BUY,
                        quantity=qty_list[0],
                        time_in_force=TimeInForce.GTC,
                    )
                    orders.append(order1)

                # Order 2: Sell currency2 for currency1 (pair3: currency2/currency1)

                # Calculate quantity in currency2 terms
                if pair3_instrument:
                    order2 = self.order_factory.market(
                        instrument_id=pair3_instrument.id,
                        order_side=OrderSide.SELL,
                        quantity=qty_list[1],
                        time_in_force=TimeInForce.GTC,
                    )
                    orders.append(order2)

                # Order 3: Sell currency1 for base (pair1: currency1/base)

                if pair1_instrument:
                    # Calculate quantity in currency1 terms
                    order3 = self.order_factory.market(
                        instrument_id=pair1_instrument.id,
                        order_side=OrderSide.SELL,
                        quantity=qty_list[2],
                        time_in_force=TimeInForce.GTC,
                    )
                    orders.append(order3)

        except Exception as e:
            self.log.error(f"Error creating arbitrage orders: {e}")
            return []

        return orders

    def _calculate_qty(self, direction: str, opportunity: Dict) -> list[Quantity]:
        qty_list = []

        if direction == "forward":
            current_trade_size = Quantity(self.current_trade_size, 8)
            currency1_quantity = Quantity(self.current_trade_size / Decimal(opportunity["rates"][0]), 8)
            currency2_quantity = Quantity(currency1_quantity.as_decimal() / Decimal(opportunity["rates"][1]), 8)

            raw_qty = [current_trade_size, currency1_quantity, currency2_quantity]

            pair1_instrument = self.triplet_instruments["pair1"]
            pair3_instrument = self.triplet_instruments["pair3"]
            pair2_instrument = self.triplet_instruments["pair2"]

            instruments = [pair1_instrument, pair3_instrument, pair2_instrument]
            maximums = [pair1_instrument.max_quantity, pair3_instrument.max_quantity, pair2_instrument.max_quantity] #Quantity
            minimums = [pair1_instrument.min_quantity, pair3_instrument.min_quantity, pair2_instrument.min_quantity]

            for i in range(3):
                pair_price = Price(opportunity["pair_prices"][i], 8)
                pair_min_notional = instruments[i].min_notional
                pair_max_notional = instruments[i].max_notional
                pair_notional = instruments[i].notional_value(raw_qty[i], pair_price)

                pair_qty = raw_qty[i] if (pair_min_notional < pair_notional < pair_max_notional) and (
                            minimums[i] < raw_qty[i] < maximums[i]) else Quantity(
                    (pair_min_notional.as_decimal() + Decimal(5.0) / pair_price.as_decimal()), 8)

                qty_list.append(pair_qty)

        else:
            current_trade_size = Quantity(self.current_trade_size, 8)
            currency2_quantity = Quantity(self.current_trade_size / Decimal(opportunity["rates"][0]), 8)
            currency1_quantity = Quantity(currency2_quantity.as_decimal() * Decimal(opportunity["rates"][1]), 8)

            raw_qty = [current_trade_size, currency2_quantity, currency1_quantity]

            pair2_instrument = self.triplet_instruments["pair2"]
            pair3_instrument = self.triplet_instruments["pair3"]
            pair1_instrument = self.triplet_instruments["pair1"]

            instruments = [pair2_instrument, pair3_instrument, pair1_instrument]
            maximums = [pair2_instrument.max_quantity, pair3_instrument.max_quantity, pair1_instrument.max_quantity]  # Quantity
            minimums = [pair2_instrument.min_quantity, pair3_instrument.min_quantity, pair1_instrument.min_quantity]

            for i in range(len(opportunity["pair_prices"])):
                pair_price = Price(opportunity["pair_prices"][i], 8)
                pair_min_notional = instruments[i].min_notional
                pair_max_notional = instruments[i].max_notional
                pair_notional = instruments[i].notional_value(raw_qty[i], pair_price)

                pair_qty = raw_qty[i] if (pair_min_notional < pair_notional < pair_max_notional) and (
                        minimums[i] < raw_qty[i] < maximums[i]) else Quantity(
                    (pair_min_notional.as_decimal() + Decimal(5.0) / pair_price.as_decimal()), 8)

                qty_list.append(pair_qty)

        min_qty = min(qty_list)
        self.log.info(f"min qty is {min_qty}", LogColor.BLUE)
        a = [min_qty, min_qty, min_qty]
        return a

    def _track_arbitrage_position(self, position_id: str, opportunity: Dict, orders: List[Order]) -> None:
        """Track arbitrage position for monitoring."""
        self.active_arbitrage_positions[position_id] = {
            "opportunity": opportunity,
            "orders": orders,
            "open_time": self.clock.timestamp_ns(),
            "expected_profit": opportunity["net_profit_rate"] * float(self.current_trade_size),
            "status": "pending",
        }
        
        # Set timeout for monitoring
        timeout_time = self.clock.utc_now() + pd.Timedelta(minutes=self.current_position_timeout_mins)
        self.position_timers[position_id] = timeout_time
        
        self.log.info(f"Tracking arbitrage position: {position_id}")
        
    def _check_position_timeouts(self, event: TimeEvent) -> None:
        """Check for position timeouts."""
        self.log.info(f"Checking for position timeouts", LogColor.CYAN)
        current_time = self.clock.utc_now()
        
        for position_id, timeout_time in list(self.position_timers.items()):
            if current_time >= timeout_time:
                self.log.warning(f"Position timeout for {position_id}")
                self._close_arbitrage_position(reason="timeout")

    def _close_all_positions(self, reason: str = "manual") -> None:
        """Close all active arbitrage positions."""
        self.log.info(f"Closing all positions due to {reason}", LogColor.RED)

        for pair_key, instrument in self.triplet_instruments.items():
            # Close each position
            self.close_all_positions(
                instrument_id=instrument.id,
            )

        # Clear active positions and timers
        self.active_arbitrage_positions.clear()
        self.position_timers.clear()

        self.log.info("All positions closed successfully")

    def _close_arbitrage_position(self, reason: str = "manual") -> None:
        """Close arbitrage position."""

        for pair_key, instrument in self.triplet_instruments.items():
                try:
                    if instrument:
                        # Cancel any open orders for this position
                        self.cancel_all_orders(instrument_id=instrument.id, client_id=self.config.client_id)
                        self.log.info(f"Closed {instrument.id} arbitrage position due to {reason}")

                except Exception as e:
                    self.log.error(f"Error closing {instrument.id} arbitrage position: {e}")
            
    def _get_current_exposure(self) -> Decimal:
        """Get current total exposure."""
        return Decimal(str(len(self.active_arbitrage_positions))) * self.current_trade_size
        
    def _risk_management_check(self, event: TimeEvent) -> None:
        """Perform risk management checks."""
        self.log.info(f"Performing risk management checks", LogColor.CYAN)

        # Check drawdown
        if self.max_drawdown_current >= self.config.max_drawdown:
            self.log.warning(f"Maximum drawdown reached: {self.max_drawdown_current:.4f}")
            self.is_paused = True
            self.pause_reason = "max_drawdown"
            self._close_all_positions("risk_management")
            
        # Check position concentration
        if len(self.active_arbitrage_positions) > 5:
            self.log.warning("Too many concurrent arbitrage positions")
            self.is_paused = True
            self.pause_reason = "position_concentration"
            
    def on_order_filled(self, event: OrderFilled) -> None:
        """Handle order filled events."""
        self.log.info(f"Order filled: {event.order_side} {event.last_qty} @ {event.last_px}")
        
        # Update performance tracking
        self._update_performance_metrics(event)
        
        # Check if this completes an arbitrage
        self._check_arbitrage_completion(event)
        
    def _update_performance_metrics(self, event: OrderFilled) -> None:
        """Update performance metrics."""
        # Simple PnL calculation
        pnl_impact = event.last_px * event.last_qty
        if event.order_side == OrderSide.SELL:
            self.total_pnl += pnl_impact
        else:
            self.total_pnl -= pnl_impact
            
        # Update drawdown tracking
        if self.total_pnl > self.high_water_mark:
            self.high_water_mark = self.total_pnl
            self.max_drawdown_current = Decimal("0.0")
        else:
            if self.high_water_mark > 0:
                self.max_drawdown_current = (self.high_water_mark - self.total_pnl) / self.high_water_mark
                
    def _check_arbitrage_completion(self, event: OrderFilled) -> None:
        """Check if an arbitrage sequence is completed."""
        # Find the arbitrage position this order belongs to
        for position_id, position_info in self.active_arbitrage_positions.items():
            for order in position_info["orders"]:
                if order.client_order_id == event.client_order_id:
                    # Update order status
                    order.status = OrderStatus.FILLED
                    
                    # Check if all orders are filled
                    all_filled = all(o.status == OrderStatus.FILLED for o in position_info["orders"])
                    
                    if all_filled:
                        self.successful_arbitrages += 1
                        self.log.info(f"Arbitrage completed successfully: {position_id}", LogColor.GREEN)
                        
                        # Record trade history
                        self.trade_history.append({
                            "position_id": position_id,
                            "timestamp": self.clock.timestamp_ns(),
                            "triplet": position_info["opportunity"]["triplet"],
                            "expected_profit": position_info["expected_profit"],
                            "status": "completed",
                        })
                        
                        # Clean up
                        self._close_arbitrage_position(position_id, "completed")
                    
                    return
                    
    # Agent integration methods
    def update_triplet(self, new_triplet: Tuple[str, str, str]) -> None:
        """Update trading triplet based on agent recommendation."""
        if new_triplet != self.current_triplet:
            self.log.info(f"Updating triplet from {self.current_triplet} to {new_triplet}")
            
            # Close existing positions
            self._close_arbitrage_position("triplet_change")
            
            # Unsubscribe from old data
            self._unsubscribe_from_triplet_data()
            
            # Update to new triplet
            self.current_triplet = new_triplet
            self._initialize_triplet_instruments()
            self._subscribe_to_triplet_data()
            
    def update_parameters(self, new_params: Dict[str, Any]) -> None:
        """Update strategy parameters based on agent recommendation."""
        if "min_profit_threshold" in new_params:
            self.current_min_profit_threshold = Decimal(str(new_params["min_profit_threshold"]))
            self.log.info(f"Updated min profit threshold to {self.current_min_profit_threshold}")
            
        if "trade_size" in new_params:
            self.current_trade_size = Decimal(str(new_params["trade_size"]))
            self.log.info(f"Updated trade size to {self.current_trade_size}")
            
        if "transaction_cost_bps" in new_params:
            self.current_transaction_cost_bps = int(new_params["transaction_cost_bps"])
            self.log.info(f"Updated transaction cost to {self.current_transaction_cost_bps} bps")
            
    def adjust_for_market_conditions(self, market_analysis: Dict[str, Any]) -> None:
        """Adjust strategy based on market situation analysis."""
        sentiment = market_analysis.get("overall_sentiment", "neutral")
        volatility = market_analysis.get("volatility_level", "normal")
        
        if sentiment == "bearish" and volatility == "high":
            # Reduce position size in bearish high-volatility conditions
            self.current_trade_size = self.config.trade_size * Decimal("0.5")
            self.log.info("Reduced trade size due to bearish sentiment and high volatility")
            
        elif sentiment == "bullish" and volatility == "low":
            # Increase position size in bullish low-volatility conditions
            self.current_trade_size = self.config.trade_size * Decimal("1.5")
            self.log.info("Increased trade size due to bullish sentiment and low volatility")
            
        elif volatility == "extreme":
            # Pause trading in extreme volatility
            self.is_paused = True
            self.pause_reason = "extreme_volatility"
            self.log.warning("Paused trading due to extreme market volatility")
            
    def on_stop(self) -> None:
        """Actions to be performed when the strategy is stopped."""
        self.log.info("Stopping triangular arbitrage strategy", LogColor.YELLOW)
        
        # Stop all timers (both trade and LLM)
        self.clock.cancel_timers()
        
        # Close all positions
        self._close_arbitrage_position("strategy_stop")
        
        # Unsubscribe from data
        self._unsubscribe_from_triplet_data()
        
        # Log final performance
        success_rate = self.successful_arbitrages / max(1, self.successful_arbitrages + self.failed_arbitrages)
        self.log.info(f"Final stats - PnL: {self.total_pnl:.4f}, "
                     f"Success rate: {success_rate:.2%}, "
                     f"Max drawdown: {self.max_drawdown_current:.4f}")
        
    def on_reset(self) -> None:
        """Actions to be performed when the strategy is reset."""
        self.active_arbitrage_positions.clear()
        self.position_timers.clear()
        self.trade_history.clear()
        self.total_pnl = Decimal("0.0")
        self.max_drawdown_current = Decimal("0.0")
        self.high_water_mark = Decimal("0.0")
        self.successful_arbitrages = 0
        self.failed_arbitrages = 0
        self.is_paused = False
        self.pause_reason = ""
        
    def on_save(self) -> dict[str, bytes]:
        """Save strategy state."""
        return {
            "total_pnl": str(self.total_pnl).encode(),
            "max_drawdown": str(self.max_drawdown_current).encode(),
            "high_water_mark": str(self.high_water_mark).encode(),
            "successful_arbitrages": str(self.successful_arbitrages).encode(),
            "failed_arbitrages": str(self.failed_arbitrages).encode(),
        }
        
    def on_load(self, state: dict[str, bytes]) -> None:
        """Load strategy state."""
        if "total_pnl" in state:
            self.total_pnl = Decimal(state["total_pnl"].decode())
        if "max_drawdown" in state:
            self.max_drawdown_current = Decimal(state["max_drawdown"].decode())
        if "high_water_mark" in state:
            self.high_water_mark = Decimal(state["high_water_mark"].decode())
        if "successful_arbitrages" in state:
            self.successful_arbitrages = int(state["successful_arbitrages"].decode())
        if "failed_arbitrages" in state:
            self.failed_arbitrages = int(state["failed_arbitrages"].decode())
            
    def on_dispose(self) -> None:
        """Cleanup resources."""
        self.price_history.clear()
        self.quote_ticks.clear()
        self.arbitrage_opportunities_history.clear()