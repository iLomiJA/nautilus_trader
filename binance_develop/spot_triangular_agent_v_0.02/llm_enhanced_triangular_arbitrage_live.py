#!/usr/bin/env python3
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

"""
Example configuration and usage of LLM-enhanced intelligent agents
for cryptocurrency arbitrage strategies.
"""

import os
from decimal import Decimal

from nautilus_trader.common.actor import ActorConfig
from nautilus_trader.adapters.binance import BinanceAccountType
from nautilus_trader.adapters.binance import BinanceDataClientConfig
from nautilus_trader.adapters.binance import BinanceExecClientConfig
from nautilus_trader.adapters.binance import BinanceLiveDataClientFactory
from nautilus_trader.adapters.binance import BinanceLiveExecClientFactory
from nautilus_trader.adapters.sandbox.config import SandboxExecutionClientConfig
from nautilus_trader.adapters.sandbox.factory import SandboxLiveExecClientFactory
from nautilus_trader.cache.config import CacheConfig
from nautilus_trader.config import InstrumentProviderConfig
from nautilus_trader.config import LiveExecEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.config import TradingNodeConfig
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.identifiers import ClientId
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.model.identifiers import ComponentId
from nautilus_trader.model.venues import Venue

from triangular_arbitrage_strategy import TriangularArbitrageStrategy
from triangular_arbitrage_strategy import TriangularArbitrageConfig
from llm_enhanced_agents import LLMEnhancedTripletSelectionAgent, LLMEnhancedMarketSituationAgent, LLMEnhancedParameterOptimizationAgent

LLM_CONFIG_CLAUDE = {
    "enabled": True,
    "provider": "openai",
    "model": "deepseek-chat",  # Change to your preferred model
    "api_key": "sk-1b13c026918e4a13994ce4f0ef682b83",
    "base_url": "https://api.deepseek.com",  # Ollama default
    "timeout": 30,
    "max_tokens": 4000,
    "temperature": 0.1,
    "requests_per_minute": 60,
}

LLM_CONFIG_OPENAI = {
    "enabled": True,
    "provider": "openai",
    "model": "deepseek-chat",
    "api_key": "sk-1b13c026918e4a13994ce4f0ef682b83",
    "base_url": "https://api.deepseek.com",  # Ollama default
    "timeout": 30,
    "max_tokens": 4000,
    "temperature": 0.1,
    "requests_per_minute": 60,
}

LLM_CONFIG_LOCAL = {
    "enabled": True,
    "provider": "openai",
    "model": "deepseek-chat",
    "api_key": "sk-1b13c026918e4a13994ce4f0ef682b83",
    "base_url": "https://api.deepseek.com",  # Ollama default
    "timeout": 60,
    "max_tokens": 4000,
    "temperature": 0.1,
    "requests_per_minute": 120,
}
# Disable LLM for testing/fallback
LLM_CONFIG_DISABLED = {
    "enabled": False,
}

# Choose your LLM configuration
SELECTED_LLM_CONFIG = LLM_CONFIG_CLAUDE  # Change this to your preferred LLM

# Agent Configurations
TRIPLET_SELECTION_AGENT_CONFIG = {
    "update_interval_secs": 300,  # 5 minutes
    "base_currency": "USDT",
    "candidate_currencies": [
        "BTC", "ETH", "BNB", "ADA", "DOT", "LINK", 
        "SOL", "MATIC", "AVAX", "ATOM", "UNI", "AAVE"
    ],
    "volume_threshold": 2000000,  # $2M daily volume
    "volatility_window": 24,  # 24 hours
    "llm_config": SELECTED_LLM_CONFIG,
    "use_llm_fallback": True,
    "max_history_size": 100,
}

MARKET_SITUATION_AGENT_CONFIG = {
    "update_interval_secs": 180,  # 3 minutes
    "search_interval_hours": 1,
    "news_sources": [
        "coindesk", "cointelegraph", "cryptonews", 
        "decrypt", "theblock", "coinbase"
    ],
    "sentiment_threshold": 0.3,
    "llm_config": SELECTED_LLM_CONFIG,
    "use_llm_fallback": True,
    "max_history_size": 50,
}

PARAMETER_OPTIMIZATION_AGENT_CONFIG = {
    "update_interval_secs": 600,  # 10 minutes
    "optimization_window": 100,  # Number of trades to analyze
    "risk_tolerance": 0.03,  # 3% risk tolerance
    "llm_config": SELECTED_LLM_CONFIG,
    "use_llm_fallback": True,
    "max_history_size": 200,
}


# Configure the trading node
config_node = TradingNodeConfig(
    trader_id=TraderId("LLM-ARBITRAGE-001"),
    logging=LoggingConfig(
        log_level="INFO",
        use_pyo3=True,
    ),
    exec_engine=LiveExecEngineConfig(
        reconciliation=True,
        snapshot_orders=True,
        snapshot_positions=True,
        snapshot_positions_interval_secs=10.0,
    ),
    cache=CacheConfig(
        timestamps_as_iso8601=True,
        buffer_interval_ms=100,
        flush_on_start=False,
    ),
    data_clients={
        "BINANCE": BinanceDataClientConfig(
            venue=Venue("BINANCE"),
            api_key="DMyKrxqC9NQpBazQkAeRgVZxGTWn3LSo8UhszWh0tHarodT7Jk3QhLQD438flegZ",
            api_secret="1ql0IIC7gnLDJDX6eQOkCCBL8vdj9cz5aPMGjF6h7O6ftAHEuO6cRdxxzQhzeDeR",
            account_type=BinanceAccountType.SPOT,
            base_url_http=None,
            base_url_ws=None,
            us=False,
            testnet=False,  # Set to True for testing
            instrument_provider=InstrumentProviderConfig(load_all=True),
        ),
    },
#     exec_clients={
#         "BINANCE": BinanceExecClientConfig(
#             venue=Venue("BINANCE"),
#             api_key=None,  # 'BINANCE_API_KEY' env var
#             api_secret=None,  # 'BINANCE_API_SECRET' env var
#             account_type=BinanceAccountType.SPOT,
#             base_url_http=None,
#             base_url_ws=None,
#             us=False,
#             testnet=False,  # Set to True for testing
#             instrument_provider=InstrumentProviderConfig(load_all=True),
#             max_retries=3,
#             retry_delay_initial_ms=1_000,
#             retry_delay_max_ms=10_000,
#         ),
#     },
#     timeout_connection=30.0,
#     timeout_reconciliation=10.0,
#     timeout_portfolio=10.0,
#     timeout_disconnection=10.0,
#     timeout_post_stop=5.0,
# )
    exec_clients={
        "BINANCE": SandboxExecutionClientConfig(
            venue="BINANCE",
            account_type="CASH",
            starting_balances=["100 USDT", "0.01 ETH"],
        ),
    },
    timeout_connection=20.0,
    timeout_reconciliation=5.0,
    timeout_portfolio=5.0,
    timeout_disconnection=5.0,
    timeout_post_stop=3.0,
)

# Instantiate the node
node = TradingNode(config=config_node)

# Configure the triangular arbitrage strategy with LLM enhancement
strat_config = TriangularArbitrageConfig(
    # Base currency for triplets
    base_currency="USDT",
    
    # Initial triplet - will be dynamically optimized by LLM agents
    initial_triplet=("USDT", "BTC", "ETH"),
    
    # Trading parameters - will be dynamically adjusted by LLM agents
    trade_size=Decimal("0.02"),
    min_profit_threshold=Decimal("0.001"),  # 0.1%
    max_position_size=Decimal("10.0"),
    
    # Risk management
    max_drawdown=Decimal("0.05"),  # 5%
    position_timeout_mins=5,
    
    # LLM-enhanced intelligent agents
    enable_llm_agents=True,
    triplet_selection_agent_config=TRIPLET_SELECTION_AGENT_CONFIG,
    market_situation_agent_config=MARKET_SITUATION_AGENT_CONFIG,
    parameter_optimization_agent_config=PARAMETER_OPTIMIZATION_AGENT_CONFIG,
    
    # Client ID
    client_id=ClientId("BINANCE"),
)

# Instantiate the strategy
strategy = TriangularArbitrageStrategy(config=strat_config)

# Add the strategy to the node
node.trader.add_strategy(strategy)

# Register client factories
node.add_data_client_factory("BINANCE", BinanceLiveDataClientFactory)
node.add_exec_client_factory("BINANCE", SandboxLiveExecClientFactory)

# Build the node
node.build()


def print_configuration_summary():
    """Print a summary of the current configuration."""
    print("\n🤖 LLM-Enhanced Triangular Arbitrage Strategy Configuration")
    print("=" * 60)
    
    print(f"LLM Provider: {SELECTED_LLM_CONFIG['provider']}")
    print(f"LLM Model: {SELECTED_LLM_CONFIG.get('model', 'N/A')}")
    print(f"LLM Enabled: {SELECTED_LLM_CONFIG['enabled']}")
    
    print(f"\nBase Currency: {strat_config.base_currency}")
    print(f"Initial Triplet: {strat_config.initial_triplet}")
    print(f"Trade Size: {strat_config.trade_size}")
    print(f"Min Profit Threshold: {strat_config.min_profit_threshold}")
    
    print(f"\nAgent Update Intervals:")
    print(f"  - Triplet Selection: {TRIPLET_SELECTION_AGENT_CONFIG['update_interval_secs']}s")
    print(f"  - Market Situation: {MARKET_SITUATION_AGENT_CONFIG['update_interval_secs']}s")
    print(f"  - Parameter Optimization: {PARAMETER_OPTIMIZATION_AGENT_CONFIG['update_interval_secs']}s")
    
    print(f"\nCandidate Currencies: {len(TRIPLET_SELECTION_AGENT_CONFIG['candidate_currencies'])}")
    print(f"News Sources: {len(MARKET_SITUATION_AGENT_CONFIG['news_sources'])}")
    
    print("=" * 60)


# Run the trading node
if __name__ == "__main__":
    print_configuration_summary()

    
    print("\n🚀 Starting LLM-Enhanced Triangular Arbitrage Strategy...")
    print("Press Ctrl+C to stop")
    
    try:
        node.run()
    except KeyboardInterrupt:
        print("\n⏹️  Stopping strategy...")
    finally:
        node.dispose()
        print("✅ Strategy stopped and resources cleaned up.")