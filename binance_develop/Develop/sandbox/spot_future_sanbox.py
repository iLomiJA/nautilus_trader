import asyncio
import os
from decimal import Decimal

from nautilus_trader.adapters.binance.common.enums import BinanceAccountType
from nautilus_trader.adapters.binance.config import BinanceDataClientConfig
from nautilus_trader.adapters.sandbox.config import SandboxExecutionClientConfig
from nautilus_trader.adapters.binance.factories import BinanceLiveDataClientFactory
from nautilus_trader.adapters.sandbox.factory import SandboxLiveExecClientFactory
from nautilus_trader.config import CacheConfig
from nautilus_trader.config import InstrumentProviderConfig
from nautilus_trader.config import LiveExecEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.config import TradingNodeConfig
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.identifiers import ClientId
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.model.identifiers import Venue

from spot_future_strategy import SpotFutureStrategyConfig as sf_config
from spot_future_strategy import SpotFutureStrategy as sf_strategy

async def main() -> None:

    config_node = TradingNodeConfig(
        trader_id=TraderId("TESTER-001"),
        logging=LoggingConfig(
            log_level="INFO",
            log_colors=True,
            use_pyo3=True,
        ),
        exec_engine=LiveExecEngineConfig(
            reconciliation=True,
            reconciliation_lookback_mins=1440,
            filter_position_reports=True,
        ),
        cache=CacheConfig(
            timestamps_as_iso8601=True,
            flush_on_start=False
        ),
        data_clients={
            "BINANCE_FUTURES": BinanceDataClientConfig(
                venue=Venue("BINANCE_FUTURES"),
                api_key="DMyKrxqC9NQpBazQkAeRgVZxGTWn3LSo8UhszWh0tHarodT7Jk3QhLQD438flegZ",#os.environ["BINANCE_API_KEY"],
                api_secret="1ql0IIC7gnLDJDX6eQOkCCBL8vdj9cz5aPMGjF6h7O6ftAHEuO6cRdxxzQhzeDeR",#os.environ["BINANCE_API_SECRET"],
                account_type=BinanceAccountType.USDT_FUTURE,
                base_url_http=None,  # Override with custom endpoint
                base_url_ws=None,  # Override with custom endpoint
                us=False,  # If client is for Binance US
                testnet=False,  # If client uses the testnet
                instrument_provider=InstrumentProviderConfig(load_all=True),
            ),
            "BINANCE_SPOT": BinanceDataClientConfig(
                venue=Venue("BINANCE_SPOT"),
                api_key="DMyKrxqC9NQpBazQkAeRgVZxGTWn3LSo8UhszWh0tHarodT7Jk3QhLQD438flegZ",
                api_secret="1ql0IIC7gnLDJDX6eQOkCCBL8vdj9cz5aPMGjF6h7O6ftAHEuO6cRdxxzQhzeDeR",
                account_type=BinanceAccountType.SPOT,
                base_url_http=None,  # Override with custom endpoint
                base_url_ws=None,  # Override with custom endpoint
                us=False,  # If client is for Binance US
                testnet=False,  # If client uses the testnet
                instrument_provider=InstrumentProviderConfig(load_all=True),
            ),
        },
        exec_clients={
            "BINANCE_FUTURES": SandboxExecutionClientConfig(
                venue="BINANCE_FUTURES",
                account_type="MARGIN",
                starting_balances=["10_000 USDT", "0.01 BTC"],
                default_leverage=Decimal("5"),
            ),
            "BINANCE_SPOT": SandboxExecutionClientConfig(
                venue="BINANCE_SPOT",
                account_type="CASH",
                starting_balances=["10_000 USDT", "0.01 BTC"],
            )
        },
        timeout_connection=30.0,
        timeout_reconciliation=10.0,
        timeout_portfolio=10.0,
        timeout_disconnection=10.0,
        timeout_post_stop=5.0,
    )

    node = TradingNode(config=config_node)

    start_config = sf_config(
        future_client_id=ClientId("BINANCE_FUTURES"),
        future_instrument_id=InstrumentId.from_str("BTCUSDT-PERP.BINANCE_FUTURES"),
        spot_instrument_id=InstrumentId.from_str("BTCUSDT.BINANCE_SPOT"),
    )

    strategy = sf_strategy(
        config=start_config,
    )

    node.trader.add_strategy(strategy)

    # Register your client factories with the node (can take user-defined factories)
    node.add_data_client_factory("BINANCE_FUTURES", BinanceLiveDataClientFactory)
    node.add_data_client_factory("BINANCE_SPOT", BinanceLiveDataClientFactory)
    node.add_exec_client_factory("BINANCE_FUTURES", SandboxLiveExecClientFactory)
    node.add_exec_client_factory("BINANCE_SPOT", SandboxLiveExecClientFactory)
    node.build()

    try:
        await node.run_async()
    finally:
        await node.stop()
        await asyncio.sleep(1)
        node.dispose()


if __name__ == "__main__":
    asyncio.run(main())