import pandas as pd
from decimal import Decimal

from nautilus_trader.common.enums import LogColor
from nautilus_trader.config import PositiveInt
from nautilus_trader.config import StrategyConfig
from nautilus_trader.core.correctness import PyCondition
from nautilus_trader.core.data import Data
from nautilus_trader.core.message import Event
from nautilus_trader.indicators.average.ema import ExponentialMovingAverage
from nautilus_trader.indicators.rsi import RelativeStrengthIndex
from nautilus_trader.model.book import OrderBook
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import BarType
from nautilus_trader.model.data import OrderBookDeltas
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.enums import TimeInForce
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.model.objects import Quantity
from nautilus_trader.model.orders import MarketOrder
from nautilus_trader.trading.strategy import Strategy

from nautilus_trader.adapters.binance import BINANCE
from nautilus_trader.adapters.binance import BinanceAccountType
from nautilus_trader.adapters.binance import BinanceDataClientConfig
from nautilus_trader.adapters.binance import BinanceExecClientConfig
from nautilus_trader.adapters.binance import BinanceLiveDataClientFactory
from nautilus_trader.adapters.binance import BinanceLiveExecClientFactory
from nautilus_trader.config import InstrumentProviderConfig
from nautilus_trader.config import LiveExecEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.config import TradingNodeConfig
from nautilus_trader.live.node import TradingNode
from nautilus_trader.model.data import BarType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import TraderId

class EMACrossConfig(StrategyConfig, frozen=True):
    """
    Configuration for ``EMACross`` instances.

    Parameters
    ----------
    instrument_id : InstrumentId
        The instrument ID for the strategy.
    bar_type : BarType
        The bar type for the strategy.
    trade_size : Decimal
        The position size per trade.
    fast_ema_period : int, default 10
        The fast EMA period.
    slow_ema_period : int, default 20
        The slow EMA period.
    subscribe_quote_ticks : bool, default False (Live) 实时信息（买一、卖一价）
        If quotes should be subscribed to.
    subscribe_trade_ticks : bool, default True (Live) 实时信息 （实际交易信息，如成交量、价格等）
        If trades should be subscribed to.
    request_bars : bool, default True (historical) 历史数据
        If historical bars should be requested on strategy start.
    unsubscribe_data_on_stop : bool, default True
        If live data feeds should be unsubscribed on strategy stop.
    order_quantity_precision : int, optional 订单数量精度，代表订单数量的小数位数
        The quantity precision for strategy market orders.
    order_time_in_force : TimeInForce, optional 订单有效期
        The time in force for strategy market orders.
    close_positions_on_stop : bool, default True
        If all open positions should be closed on strategy stop.
    reduce_only_on_stop : bool, default True 是否平仓市场订单只能允许减仓
        If position closing market orders on stop should be reduce-only.
    """

    instrument_id: InstrumentId
    bar_type: BarType
    trade_size: Decimal
    fast_ema_period: PositiveInt = 10
    slow_ema_period: PositiveInt = 20
    subscribe_quote_ticks: bool = False
    subscribe_trade_ticks: bool = True
    request_bars: bool = True
    unsubscribe_data_on_stop: bool = True
    order_quantity_precision: int | None = None
    order_time_in_force: TimeInForce | None = None
    close_positions_on_stop: bool = True
    reduce_only_on_stop: bool = True


class EMACross(Strategy):
    """
    A simple moving average cross example strategy.

    When the fast EMA crosses the slow EMA then enter a position at the market
    in that direction.

    Parameters
    ----------
    config : EMACrossConfig
        The configuration for the instance.

    Raises
    ------
    ValueError
        If `config.fast_ema_period` is not less than `config.slow_ema_period`.

    """

    def __init__(self, config: EMACrossConfig) -> None:
        PyCondition.is_true(
            config.fast_ema_period < config.slow_ema_period,
            "{config.fast_ema_period=} must be less than {config.slow_ema_period=}",
        ) #自带的条件检查函数
        super().__init__(config)

        self.instrument: Instrument = None

        # Create the indicators for the strategy
        self.fast_ema = ExponentialMovingAverage(config.fast_ema_period)
        self.slow_ema = ExponentialMovingAverage(config.slow_ema_period)

    def on_start(self) -> None:
        """
        Actions to be performed on strategy start.
        """
        self.instrument = self.cache.instrument(self.config.instrument_id)
        if self.instrument is None:
            self.log.error(f"Could not find instrument for {self.config.instrument_id}")
            self.stop()
            return

        # Register the indicators for updating
        #为我们设定好的Bar类型注册了快慢线值的指标，后续在新的bar接受的时候会自动的更新这两个参数的值，我们不需要手动调用更新方法
        self.register_indicator_for_bars(self.config.bar_type, self.fast_ema)
        self.register_indicator_for_bars(self.config.bar_type, self.slow_ema)

        # Get historical data
        if self.config.request_bars:
            self.request_bars(
                self.config.bar_type,
                start=self._clock.utc_now() - pd.Timedelta(days=1),
            )

        # self.request_quote_ticks(self.config.instrument_id)
        # self.request_trade_ticks(self.config.instrument_id)

        # Subscribe to real-time data
        self.subscribe_bars(self.config.bar_type)

        if self.config.subscribe_quote_ticks:
            self.subscribe_quote_ticks(self.config.instrument_id)
        if self.config.subscribe_trade_ticks:
            self.subscribe_trade_ticks(self.config.instrument_id)

        # self.subscribe_order_book_deltas(self.config.instrument_id, depth=20)  # For debugging
        # self.subscribe_order_book_at_interval(self.config.instrument_id, depth=20)  # For debugging

    def on_instrument(self, instrument: Instrument) -> None:
        """
        Actions to be performed when the strategy is running and receives an instrument.
        在策略开始并接受到了一个新的交易品种时执行的操作。
        Parameters
        ----------
        instrument : Instrument
            The instrument received.

        """
        # For debugging (must add a subscription)
        # self.log.info(repr(instrument), LogColor.CYAN)

    def on_order_book_deltas(self, deltas: OrderBookDeltas) -> None:
        """
        Actions to be performed when the strategy is running and receives order book
        deltas.
        在接受到了order book deltas时执行操作，order book deltas是订单簿差值，计算方法是将特点时间段内订单簿中所有
        买单总量减去卖单总量得到的一个值。
        Parameters
        ----------
        deltas : OrderBookDeltas
            The order book deltas received.

        """
        # For debugging (must add a subscription)
        # self.log.info(repr(deltas), LogColor.CYAN)

    def on_order_book(self, order_book: OrderBook) -> None:
        """
        Actions to be performed when the strategy is running and receives an order book.
        在接受到了order book时执行的操作。
        Parameters
        ----------
        order_book : OrderBook
            The order book received.

        """
        # For debugging (must add a subscription)
        # self.log.info(repr(order_book), LogColor.CYAN)

    def on_quote_tick(self, tick: QuoteTick) -> None:
        """
        Actions to be performed when the strategy is running and receives a quote tick.

        Parameters
        ----------
        tick : QuoteTick
            The tick received.

        """
        # For debugging (must add a subscription)
        self.log.info(repr(tick), LogColor.GREEN)

    def on_trade_tick(self, tick: TradeTick) -> None:
        """
        Actions to be performed when the strategy is running and receives a trade tick.

        Parameters
        ----------
        tick : TradeTick
            The tick received.

        """
        # For debugging (must add a subscription)
        self.log.info(repr(tick), LogColor.CYAN)

    def on_bar(self, bar: Bar) -> None:
        """
        Actions to be performed when the strategy is running and receives a bar.

        Parameters
        ----------
        bar : Bar
            The bar received.
        """
        self.log.info(repr(bar), LogColor.CYAN)

        # Check if indicators ready
        if not self.indicators_initialized():
            self.log.info(
                f"Waiting for indicators to warm up [{self.cache.bar_count(self.config.bar_type)}]",
                color=LogColor.BLUE,
            )
            return  # Wait for indicators to warm up...

        if bar.is_single_price():
            self._log.warning("Bar OHLC is single price; implies no market information")
            return

        # BUY LOGIC
        # if self.fast_ema.value >= self.slow_ema.value:
        if self.portfolio.is_flat(self.config.instrument_id): #检查持仓是否为零
            self.buy()
        elif self.portfolio.is_net_short(self.config.instrument_id): #检查持仓是否为净空头
            self.close_all_positions(self.config.instrument_id) #先平仓后买入
            self.buy()
        else:
            self.buy()
        # # SELL LOGIC
        # elif self.fast_ema.value < self.slow_ema.value:
        #     if self.portfolio.is_flat(self.config.instrument_id):
        #         self.sell()
        #     elif self.portfolio.is_net_long(self.config.instrument_id):
        #         self.close_all_positions(self.config.instrument_id)
        #         self.sell()

    def buy(self) -> None:
        """
        Users simple buy method (example).
        """
        order: MarketOrder = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.BUY,
            quantity=self.create_order_qty(),
            time_in_force=self.config.order_time_in_force or TimeInForce.GTC,
        )

        self.submit_order(order)

    def sell(self) -> None:
        """
        Users simple sell method (example).
        """
        order: MarketOrder = self.order_factory.market(
            instrument_id=self.config.instrument_id,
            order_side=OrderSide.SELL,
            quantity=self.create_order_qty(),
            time_in_force=self.config.order_time_in_force or TimeInForce.GTC,
        )

        self.submit_order(order)

    def create_order_qty(self) -> Quantity:
        if self.config.order_quantity_precision is not None:
            return Quantity(self.config.trade_size, self.config.order_quantity_precision)
        else:
            return self.instrument.make_qty(self.config.trade_size)

    def on_data(self, data: Data) -> None:
        """
        Actions to be performed when the strategy is running and receives data.

        Parameters
        ----------
        data : Data
            The data received.

        """

    def on_event(self, event: Event) -> None:
        """
        Actions to be performed when the strategy is running and receives an event.

        Parameters
        ----------
        event : Event
            The event received.

        """

    def on_stop(self) -> None:
        """
        Actions to be performed when the strategy is stopped.
        """
        self.cancel_all_orders(self.config.instrument_id)
        if self.config.close_positions_on_stop:
            self.close_all_positions(
                instrument_id=self.config.instrument_id,
                reduce_only=self.config.reduce_only_on_stop,
            )

        if self.config.unsubscribe_data_on_stop:
            self.unsubscribe_bars(self.config.bar_type)

        if self.config.unsubscribe_data_on_stop and self.config.subscribe_quote_ticks:
            self.unsubscribe_quote_ticks(self.config.instrument_id)

        if self.config.unsubscribe_data_on_stop and self.config.subscribe_trade_ticks:
            self.unsubscribe_trade_ticks(self.config.instrument_id)

        # self.unsubscribe_order_book_deltas(self.config.instrument_id)
        # self.unsubscribe_order_book_at_interval(self.config.instrument_id)

    def on_reset(self) -> None:
        """
        Actions to be performed when the strategy is reset.
        """
        # Reset indicators here
        self.fast_ema.reset()
        self.slow_ema.reset()

    def on_save(self) -> dict[str, bytes]:
        """
        Actions to be performed when the strategy is saved.

        Create and return a state dictionary of values to be saved.

        Returns
        -------
        dict[str, bytes]
            The strategy state dictionary.

        """
        return {}

    def on_load(self, state: dict[str, bytes]) -> None:
        """
        Actions to be performed when the strategy is loaded.

        Saved state values will be contained in the give state dictionary.

        Parameters
        ----------
        state : dict[str, bytes]
            The strategy state dictionary.

        """

    def on_dispose(self) -> None:
        """
        Actions to be performed when the strategy is disposed.

        Cleanup any resources used by the strategy here.

        """

# *** THIS IS A TEST STRATEGY WITH NO ALPHA ADVANTAGE WHATSOEVER. ***
# *** IT IS NOT INTENDED TO BE USED TO TRADE LIVE WITH REAL MONEY. ***


# Configure the trading node
config_node = TradingNodeConfig(
    trader_id=TraderId("TESTER-001"),
    logging=LoggingConfig(log_level="INFO"),
    exec_engine=LiveExecEngineConfig(
        reconciliation=True,
        reconciliation_lookback_mins=1440,
        # snapshot_orders=True,
        # snapshot_positions=True,
        # snapshot_positions_interval_secs=5.0,
    ),
    # cache=CacheConfig(
    #     database=DatabaseConfig(),
    #     buffer_interval_ms=100,
    # ),
    # message_bus=MessageBusConfig(
    #     database=DatabaseConfig(),
    #     encoding="json",
    #     streams_prefix="quoters",
    #     use_instance_id=False,
    #     timestamps_as_iso8601=True,
    #     # types_filter=[QuoteTick],
    #     autotrim_mins=1,
    #     heartbeat_interval_secs=1,
    # ),
    data_clients={
        BINANCE: BinanceDataClientConfig(
            api_key="DMyKrxqC9NQpBazQkAeRgVZxGTWn3LSo8UhszWh0tHarodT7Jk3QhLQD438flegZ",  # 'BINANCE_API_KEY' env var
            api_secret="1ql0IIC7gnLDJDX6eQOkCCBL8vdj9cz5aPMGjF6h7O6ftAHEuO6cRdxxzQhzeDeR",  # 'BINANCE_API_SECRET' env var
            account_type=BinanceAccountType.SPOT,
            base_url_http=None,  # Override with custom endpoint
            base_url_ws=None,  # Override with custom endpoint
            us=False,  # If client is for Binance US
            testnet=False,  # If client uses the testnet
            instrument_provider=InstrumentProviderConfig(load_all=True),
        ),
    },
    exec_clients={
        BINANCE: BinanceExecClientConfig(
            api_key="DMyKrxqC9NQpBazQkAeRgVZxGTWn3LSo8UhszWh0tHarodT7Jk3QhLQD438flegZ",  # 'BINANCE_API_KEY' env var
            api_secret="1ql0IIC7gnLDJDX6eQOkCCBL8vdj9cz5aPMGjF6h7O6ftAHEuO6cRdxxzQhzeDeR",  # 'BINANCE_API_SECRET' env var
            account_type=BinanceAccountType.SPOT,
            base_url_http=None,  # Override with custom endpoint
            base_url_ws=None,  # Override with custom endpoint
            us=False,  # If client is for Binance US
            testnet=False,  # If client uses the testnet
            instrument_provider=InstrumentProviderConfig(load_all=True),
            max_retries=3,
            retry_delay_initial_ms=1_000,
            retry_delay_max_ms=10_000,
        ),
    },
    timeout_connection=30.0,
    timeout_reconciliation=10.0,
    timeout_portfolio=10.0,
    timeout_disconnection=10.0,
    timeout_post_stop=5.0,
)

# Instantiate the node with a configuration
node = TradingNode(config=config_node)

# Configure your strategy
strat_config = EMACrossConfig(
    instrument_id=InstrumentId.from_str("ETHUSDT.BINANCE"),
    external_order_claims=[InstrumentId.from_str("ETHUSDT.BINANCE")],
    bar_type=BarType.from_str("ETHUSDT.BINANCE-1-MINUTE-LAST-EXTERNAL"),
    fast_ema_period=10,
    slow_ema_period=20,
    trade_size=Decimal("0.0020"),
    order_id_tag="001",
)
# Instantiate your strategy
strategy = EMACross(config=strat_config)

# Add your strategies and modules
node.trader.add_strategy(strategy)

# Register your client factories with the node (can take user-defined factories)
node.add_data_client_factory(BINANCE, BinanceLiveDataClientFactory)
node.add_exec_client_factory(BINANCE, BinanceLiveExecClientFactory)
node.build()


# Stop and dispose of the node with SIGINT/CTRL+C
if __name__ == "__main__":
    try:
        node.run()
    finally:
        node.dispose()
