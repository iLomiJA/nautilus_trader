import json

from nautilus_trader.model.identifiers import ClientId
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.identifiers import TraderId
from nautilus_trader.model.identifiers import Venue
from nautilus_trader.model.instruments import Instrument
from nautilus_trader.core.data import Data
from nautilus_trader.adapters.binance.futures.types import BinanceFuturesMarkPriceUpdate
from nautilus_trader.model.data import Bar
from nautilus_trader.model.data import DataType
from nautilus_trader.model.data import QuoteTick
from nautilus_trader.model.data import TradeTick
from nautilus_trader.common.enums import LogColor
from nautilus_trader.trading.config import StrategyConfig
from nautilus_trader.trading import Strategy


class SpotFutureStrategyConfig(StrategyConfig, frozen=True):
    future_client_id: ClientId
    future_instrument_id: InstrumentId
    spot_instrument_id: InstrumentId

class SpotFutureStrategy(Strategy):
    def __init__(self, config: SpotFutureStrategyConfig) -> None:
        super().__init__(config)

        self.future_instrument: Instrument | None = None
        self.spot_instrument: Instrument | None = None
        self.future_client_id = config.future_client_id

    def on_start(self) -> None:
        self.futures_instrument = self.cache.instrument(self.futures_instrument_id)
        if self.futures_instrument is None:
            self.log.error(
                f"Could not find instrument for {self.config.futures_instrument_id}"
                f"\nPossible instruments: {self.cache.instrument_ids()}",
            )
            self.stop()
            return
        self.spot_instrument = self.cache.instrument(self.spot_instrument_id)
        if self.spot_instrument is None:
            self.log.error(
                f"Could not find instrument for {self.config.spot_instrument_id}"
                f"\nPossible instruments: {self.cache.instrument_ids()}",
            )
            self.stop()
            return

        account = self.portfolio.account(venue=self.futures_instrument.venue)
        balance = {str(currency):str(balance) for currency, balance in account.balance().items()}
        self.log.info(f"Futures Blance: {json.dumps(balance, indent=4)}", LogColor.GREEN)
        account = self.portfolio.account(venue=self.spot_instrument.venue)
        balance = {str(currency):str(balance) for currency, balance in account.balance().items()}
        self.log.info(f"Spot Blance: {json.dumps(balance, indent=4)}", LogColor.GREEN)

        #订阅实时数据
        self.subscribe_quote_ticks(self.futures_instrument)
        self.subscribe_quote_ticks(self.spot_instrument)
        self.subscribe_data(
            data_type=DataType(
                BinanceFuturesMarkPriceUpdate,
                metadata={"instrument_id": self.futures_instrument.id}
            ),
            client_id=self.future_client_id
        )

        def on_data(self, data: Data) -> None:
            self.log.info(repr(data), LogColor.CYAN)

        def on_quote_tick(self, tick: QuoteTick) -> None:
            self.log.info(repr(tick), LogColor.CYAN)

        def on_trade_tick(self, tick: TradeTick) -> None:
            self.log.info(repr(tick), LogColor.CYAN)

        def on_bar(self, bar: Bar) -> None:
            self.log.info(repr(bar), LogColor.CYAN)

        def on_stop(self) -> None:
            # Unsubscribe from data
            self.unsubscribe_quote_ticks(self.config.futures_instrument_id)
            self.unsubscribe_quote_ticks(self.config.spot_instrument_id)