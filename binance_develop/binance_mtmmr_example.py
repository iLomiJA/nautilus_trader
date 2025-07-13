#!/usr/bin/env python3
# -------------------------------------------------------------------------------------------------
# Binance Multi-Timeframe Momentum Mean Reversion Strategy Example
# 币安多时间框架动量均值回归策略示例
# -------------------------------------------------------------------------------------------------

from decimal import Decimal

from nautilus_trader.adapters.binance import BINANCE
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
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import TraderId

# Import our custom strategy
from mtmmr_strategy import MTMMRConfig, MTMMRStrategy

# *** 这是一个示例策略，请在实盘交易前充分测试和优化参数 ***
# *** THIS IS AN EXAMPLE STRATEGY, PLEASE TEST AND OPTIMIZE BEFORE LIVE TRADING ***

def create_trading_node():
    """Create and configure the trading node."""
    
    # Configure the trading node
    config_node = TradingNodeConfig(
        trader_id=TraderId("MTMMR-001"),
        logging=LoggingConfig(
            log_level="INFO",
            log_level_file="DEBUG",
            log_file_format="json",
        ),
        exec_engine=LiveExecEngineConfig(
            reconciliation=True,
            reconciliation_lookback_mins=1440,
            filter_position_reports=True,
            # Enable snapshots for production
            snapshot_orders=True,
            snapshot_positions=True,
            snapshot_positions_interval_secs=10.0,
        ),
        cache=CacheConfig(
            database=None,  # Use Redis for production
            timestamps_as_iso8601=True,
            flush_on_start=False,
        ),
        data_clients={
            BINANCE: BinanceDataClientConfig(
                api_key="DMyKrxqC9NQpBazQkAeRgVZxGTWn3LSo8UhszWh0tHarodT7Jk3QhLQD438flegZ", # Will use 'BINANCE_API_KEY' env var
                api_secret="1ql0IIC7gnLDJDX6eQOkCCBL8vdj9cz5aPMGjF6h7O6ftAHEuO6cRdxxzQhzeDeR",  # Will use 'BINANCE_API_SECRET' env var
                account_type=BinanceAccountType.MARGIN,  # Futures trading
                base_url_http=None,  # Use default endpoint
                base_url_ws=None,  # Use default endpoint
                us=False,  # Not Binance US
                testnet=False,  # Use testnet for testing, set to False for live trading
                instrument_provider=InstrumentProviderConfig(load_all=True),
                update_instruments_interval_mins=60,
                use_agg_trade_ticks=False,  # Use raw trade ticks
            ),
        },
        exec_clients={
            BINANCE: SandboxExecutionClientConfig(
                venue="BINANCE",
                account_type="MARGIN",
                starting_balances=["10_000 USDT", "0.01 ETH"],
            ),
        },
        # exec_clients={
        #     BINANCE: BinanceExecClientConfig(
        #         api_key=None,  # Will use 'BINANCE_API_KEY' env var
        #         api_secret=None,  # Will use 'BINANCE_API_SECRET' env var
        #         account_type=BinanceAccountType.USDT_FUTURE,  # Futures trading
        #         base_url_http=None,  # Use default endpoint
        #         base_url_ws=None,  # Use default endpoint
        #         us=False,  # Not Binance US
        #         testnet=True,  # Use testnet for testing, set to False for live trading
        #         instrument_provider=InstrumentProviderConfig(load_all=True),
        #         use_gtd=True,
        #         use_reduce_only=True,
        #         use_position_ids=True,  # Use Binance position IDs
        #         use_trade_lite=False,
        #         treat_expired_as_canceled=False,
        #         recv_window_ms=5000,
        #         max_retries=3,
        #         retry_delay_initial_ms=1000,
        #         retry_delay_max_ms=10000,
        #         # Futures specific settings
        #         futures_leverages=None,  # Will use default leverage
        #         futures_margin_types=None,  # Will use default margin type
        #         listen_key_ping_max_failures=3,
        #     ),
        # },
        timeout_connection=30.0,
        timeout_reconciliation=10.0,
        timeout_portfolio=10.0,
        timeout_disconnection=10.0,
        timeout_post_stop=5.0,
    )
    
    return TradingNode(config=config_node)


def create_strategy_config():
    """Create the strategy configuration."""
    
    # Define the instrument to trade
    instrument_id = InstrumentId.from_str("ETHUSDT-PERP.BINANCE")
    
    # Define bar types for different timeframes
    long_bar_type = BarType.from_str("ETHUSDT-PERP.BINANCE-15-MINUTE-LAST-EXTERNAL")
    medium_bar_type = BarType.from_str("ETHUSDT-PERP.BINANCE-5-MINUTE-LAST-EXTERNAL")
    short_bar_type = BarType.from_str("ETHUSDT-PERP.BINANCE-1-MINUTE-LAST-EXTERNAL")
    
    # Configure the strategy
    strategy_config = MTMMRConfig(
        instrument_id=instrument_id,
        external_order_claims=[instrument_id],  # Claim external orders for this instrument
        long_bar_type=long_bar_type,
        medium_bar_type=medium_bar_type,
        short_bar_type=short_bar_type,
        trade_size=Decimal("0.01"),  # Base position size (adjust based on account size)
        max_position_risk=0.01,  # Risk 1% of account per trade
        
        # Technical indicator parameters
        ema_fast_period=21,
        ema_slow_period=50,
        macd_fast_period=12,
        macd_slow_period=26,
        bb_period=20,
        bb_std=2.0,
        rsi_period=14,
        atr_period=14,
        
        # Signal thresholds
        rsi_oversold=30.0,
        rsi_overbought=70.0,
        
        # Risk management parameters
        stop_loss_atr_multiple=2.0,
        take_profit_1_atr_multiple=1.5,
        take_profit_2_atr_multiple=3.0,
        
        # Strategy metadata
        order_id_tag="MTMMR",
        oms_type="HEDGING",  # Use hedging mode for futures
    )
    
    return strategy_config

def main():
    """Main function to run the strategy."""
    
    # Create trading node
    node = create_trading_node()
    
    # Create strategy configuration
    strategy_config = create_strategy_config()
    
    # Instantiate the strategy
    strategy = MTMMRStrategy(config=strategy_config)
    
    # Add strategy to the node
    node.trader.add_strategy(strategy)
    
    # Register client factories
    node.add_data_client_factory(BINANCE, BinanceLiveDataClientFactory)
    node.add_exec_client_factory(BINANCE, SandboxLiveExecClientFactory)
    
    # Build the node
    node.build()
    
    print("=" * 80)
    print("Multi-Timeframe Momentum Mean Reversion Strategy")
    print("多时间框架动量均值回归策略")
    print("=" * 80)
    print(f"Instrument: {strategy_config.instrument_id}")
    print(f"Long Timeframe: {strategy_config.long_bar_type}")
    print(f"Medium Timeframe: {strategy_config.medium_bar_type}")
    print(f"Short Timeframe: {strategy_config.short_bar_type}")
    print(f"Base Trade Size: {strategy_config.trade_size}")
    print(f"Max Risk per Trade: {strategy_config.max_position_risk * 100}%")
    print("=" * 80)
    print("Strategy is starting... Press Ctrl+C to stop.")
    print("=" * 80)
    
    return node


if __name__ == "__main__":
    # Create and run the trading node
    trading_node = main()
    
    try:
        # Run the strategy
        trading_node.run()
    except KeyboardInterrupt:
        print("\nShutting down strategy...")
    finally:
        # Clean shutdown
        trading_node.dispose()
        print("Strategy stopped successfully.")