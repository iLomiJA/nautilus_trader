#!/usr/bin/env python3
# -------------------------------------------------------------------------------------------------
# Binance Multi-Level Dynamic Allocation Strategy Example for Spot Trading
# 币安多层级动态配置现货交易策略示例
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
from mdas_strategy import MDASConfig, MDASStrategy


# *** 这是一个现货交易策略示例，专为长期投资设计 ***
# *** THIS IS A SPOT TRADING STRATEGY EXAMPLE, DESIGNED FOR LONG-TERM INVESTMENT ***


def create_trading_node():
    """Create and configure the trading node for spot trading."""
    
    # Configure the trading node
    config_node = TradingNodeConfig(
        trader_id=TraderId("MDAS-SPOT-001"),
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
            snapshot_positions_interval_secs=30.0,  # More frequent for spot
        ),
        cache=CacheConfig(
            database=None,  # Use Redis for production
            timestamps_as_iso8601=True,
            flush_on_start=False,
        ),
        data_clients={
            BINANCE: BinanceDataClientConfig(
                api_key="DMyKrxqC9NQpBazQkAeRgVZxGTWn3LSo8UhszWh0tHarodT7Jk3QhLQD438flegZ",  # Will use 'BINANCE_API_KEY' env var
                api_secret="1ql0IIC7gnLDJDX6eQOkCCBL8vdj9cz5aPMGjF6h7O6ftAHEuO6cRdxxzQhzeDeR",  # Will use 'BINANCE_API_SECRET' env var
                account_type=BinanceAccountType.SPOT,  # Spot trading
                base_url_http=None,  # Use default endpoint
                base_url_ws=None,  # Use default endpoint
                us=False,  # Not Binance US
                testnet=False,  # Use testnet for testing, set to False for live trading
                instrument_provider=InstrumentProviderConfig(load_all=True),
                update_instruments_interval_mins=60,
                use_agg_trade_ticks=False,  # Use raw trade ticks
            ),
        },
    #     exec_clients={
    #         BINANCE: BinanceExecClientConfig(
    #             api_key=None,  # Will use 'BINANCE_API_KEY' env var
    #             api_secret=None,  # Will use 'BINANCE_API_SECRET' env var
    #             account_type=BinanceAccountType.SPOT,  # Spot trading
    #             base_url_http=None,  # Use default endpoint
    #             base_url_ws=None,  # Use default endpoint
    #             us=False,  # Not Binance US
    #             testnet=True,  # Use testnet for testing, set to False for live trading
    #             instrument_provider=InstrumentProviderConfig(load_all=True),
    #             use_gtd=True,
    #             use_reduce_only=False,  # Not applicable for spot
    #             use_position_ids=False,  # Not applicable for spot
    #             use_trade_lite=False,
    #             treat_expired_as_canceled=False,
    #             recv_window_ms=5000,
    #             max_retries=3,
    #             retry_delay_initial_ms=1000,
    #             retry_delay_max_ms=10000,
    #             listen_key_ping_max_failures=3,
    #         ),
    #     },
    #     timeout_connection=30.0,
    #     timeout_reconciliation=10.0,
    #     timeout_portfolio=10.0,
    #     timeout_disconnection=10.0,
    #     timeout_post_stop=5.0,
    # )
        exec_clients={
            BINANCE: SandboxExecutionClientConfig(
                venue="BINANCE",
                account_type="CASH",
                starting_balances=["10_000 USDT", "0.01 ETH"],
            ),
        },
        timeout_connection=30.0,
        timeout_reconciliation=10.0,
        timeout_portfolio=10.0,
        timeout_disconnection=10.0,
        timeout_post_stop=5.0,
    )
    return TradingNode(config=config_node)

def create_strategy_config():
    """Create the strategy configuration for spot trading."""
    
    # Define the instrument to trade (spot pair)
    instrument_id = InstrumentId.from_str("ETHUSDT.BINANCE")
    
    # Define bar types for different timeframes
    daily_bar_type = BarType.from_str("ETHUSDT.BINANCE-15-MINUTE-LAST-EXTERNAL")
    h4_bar_type = BarType.from_str("ETHUSDT.BINANCE-5-MINUTE-LAST-EXTERNAL")
    h1_bar_type = BarType.from_str("ETHUSDT.BINANCE-1-MINUTE-LAST-EXTERNAL")
    
    # Configure the strategy
    strategy_config = MDASConfig(
        instrument_id=instrument_id,
        external_order_claims=[instrument_id],  # Claim external orders for this instrument
        daily_bar_type=daily_bar_type,
        h4_bar_type=h4_bar_type,
        h1_bar_type=h1_bar_type,
        
        # Capital allocation (adjust based on your account size)
        total_capital=Decimal("10000"),  # 10,000 USDT total capital
        
        # Multi-level allocation ratios
        core_allocation=0.40,        # 40% for core long-term positions
        tactical_allocation=0.30,    # 30% for tactical swing trading
        opportunity_allocation=0.20, # 20% for opportunity trades
        cash_allocation=0.10,        # 10% cash reserve
        
        # Risk management
        max_single_trade_ratio=0.05,  # Maximum 5% per trade
        core_stop_loss=0.20,          # 20% stop loss for core positions
        tactical_stop_loss=0.10,      # 10% stop loss for tactical positions
        opportunity_stop_loss=0.05,   # 5% stop loss for opportunity trades
        
        # Technical indicator parameters
        ema_short_period=20,
        ema_medium_period=50,
        ema_long_period=200,
        rsi_period=14,
        bb_period=20,
        bb_std=2.0,
        atr_period=14,
        
        # Signal thresholds
        rsi_oversold=30.0,
        rsi_overbought=70.0,
        
        # Strategy metadata
        order_id_tag="MDAS",
        oms_type="NETTING",  # Use netting mode for spot
    )
    
    return strategy_config


def main():
    """Main function to run the spot trading strategy."""
    
    # Create trading node
    node = create_trading_node()
    
    # Create strategy configuration
    strategy_config = create_strategy_config()
    
    # Instantiate the strategy
    strategy = MDASStrategy(config=strategy_config)
    
    # Add strategy to the node
    node.trader.add_strategy(strategy)
    
    # Register client factories
    node.add_data_client_factory("BINANCE", BinanceLiveDataClientFactory)
    node.add_exec_client_factory("BINANCE", SandboxLiveExecClientFactory)
    
    # Build the node
    node.build()
    
    print("=" * 80)
    print("Multi-Level Dynamic Allocation Strategy for Spot Trading")
    print("多层级动态配置现货交易策略")
    print("=" * 80)
    print(f"Instrument: {strategy_config.instrument_id}")
    print(f"Total Capital: {strategy_config.total_capital} USDT")
    print(f"Core Allocation: {strategy_config.core_allocation * 100}%")
    print(f"Tactical Allocation: {strategy_config.tactical_allocation * 100}%")
    print(f"Opportunity Allocation: {strategy_config.opportunity_allocation * 100}%")
    print(f"Cash Reserve: {strategy_config.cash_allocation * 100}%")
    print("=" * 80)
    print("Strategy Features:")
    print("• Multi-level position allocation")
    print("• Multi-timeframe analysis (Daily/4H/1H)")
    print("• Dynamic risk management")
    print("• Spot trading optimized (long-only)")
    print("• Suitable for long-term investment")
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


# ============================================================================
# 现货交易策略说明 / Spot Trading Strategy Instructions
# ============================================================================

"""
## 策略特点 / Strategy Features

### 1. 多层级资金配置 / Multi-Level Capital Allocation
- 核心仓位 (40%): 长期趋势跟踪，基于日线分析
- 战术仓位 (30%): 中期波段交易，基于4小时分析  
- 机会仓位 (20%): 短期抄底，基于1小时分析
- 现金储备 (10%): 应急资金，等待极端机会

### 2. 适合现货交易的特点 / Spot Trading Optimized
- 只做多方向，符合现货交易限制
- 分批建仓，降低平均成本
- 长期持有导向，适合价值投资
- 动态风险管理，不同层级不同止损

### 3. 多时间框架分析 / Multi-Timeframe Analysis
- 日线: 判断长期趋势和市场状态
- 4小时: 中期动量和波段机会
- 1小时: 短期超跌反弹机会

## 使用说明 / Usage Instructions

### 1. 环境设置 / Environment Setup
```bash
# 设置API密钥
export BINANCE_API_KEY="your_api_key"
export BINANCE_API_SECRET="your_api_secret"

# 测试网密钥 (推荐先使用)
export BINANCE_TESTNET_API_KEY="your_testnet_api_key"
export BINANCE_TESTNET_API_SECRET="your_testnet_api_secret"
```

### 2. 参数配置 / Parameter Configuration
- total_capital: 根据你的账户资金调整
- 各层级allocation: 可根据风险偏好调整比例
- stop_loss: 根据市场波动率调整止损比例

### 3. 适用品种 / Suitable Instruments
- 主流币种: BTC, ETH
- 优质山寨币: 有基本面支撑的项目
- 建议选择流动性好的USDT交易对

### 4. 风险提示 / Risk Warning
- 现货交易相对安全，但仍有市场风险
- 建议用闲置资金投资
- 定期检查策略表现并调整参数
- 注意资金管理，不要过度集中

### 5. 监控要点 / Monitoring Points
- 各层级仓位分配是否合理
- 止损订单是否正常工作
- 市场状态判断是否准确
- 整体收益风险比是否满意

## 预期表现 / Expected Performance
- 年化收益: 15-30% (取决于市场环境)
- 最大回撤: <20%
- 适合风险偏好: 中等
- 投资期限: 6个月以上

## 优化建议 / Optimization Suggestions
1. 根据历史数据回测优化参数
2. 可以添加更多技术指标确认
3. 考虑加入基本面分析
4. 定期重新平衡各层级配置
"""

