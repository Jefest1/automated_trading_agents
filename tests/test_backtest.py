from __future__ import annotations

import unittest

from trading_agent.backtest import Backtester, Candle
from trading_agent.core.config import AppConfig


def kline(open_: float, high: float, low: float, close: float, *, t: int) -> list:
    return [t, str(open_), str(high), str(low), str(close), "1000.0", t + 1]


def flat_series(price: float = 100.0, count: int = 30) -> list[list]:
    return [kline(price, price * 1.0001, price * 0.9999, price, t=i) for i in range(count)]


class BacktesterTest(unittest.TestCase):
    def setUp(self) -> None:
        # These backtester tests hand-craft candles for a 1%/1.5% bracket; pin it
        # explicitly so they exercise the TP/SL LOGIC independent of the production
        # swing-config defaults (4% stop / 6% TP).
        self.config = AppConfig()
        self.config.risk.stop_loss_pct = 0.01
        self.config.risk.take_profit_pct = 0.015

    def test_flat_market_produces_no_trades(self) -> None:
        result = Backtester(self.config).run("BTCUSDT", flat_series())
        self.assertEqual(result.trades, [])
        self.assertEqual(result.realized_pnl, 0.0)
        self.assertFalse(result.summary()["beats_wait_always"])

    def test_momentum_spike_opens_and_take_profits(self) -> None:
        # Candle 5 jumps +1% (100 bps momentum -> edge well above 30 bps), the
        # next candle dips to fill the limit, then a wick runs through TP while
        # closes stay flat (so no follow-on signal fires).
        rows = flat_series(100.0, 5)
        rows.append(kline(100.0, 101.2, 100.0, 101.0, t=5))  # decision candle
        rows.append(kline(101.0, 101.1, 100.5, 101.0, t=6))  # fills limit ~100.99
        rows.append(kline(101.0, 104.0, 100.5, 101.0, t=7))  # +1.5% TP wick
        rows.extend(kline(101.0, 101.05, 100.95, 101.0, t=8 + i) for i in range(3))

        result = Backtester(self.config).run("BTCUSDT", rows)

        self.assertEqual(len(result.trades), 1)
        trade = result.trades[0]
        self.assertEqual(trade.exit_reason, "TAKE_PROFIT")
        self.assertAlmostEqual(
            trade.exit_price, trade.entry_price * (1 + self.config.risk.take_profit_pct), places=6
        )
        self.assertGreater(trade.realized_pnl, 0.0)
        # fees on both sides at 10 bps of ~100 USD notional
        self.assertGreater(trade.fees, 0.15)

    def test_stop_loss_has_priority_when_both_touched(self) -> None:
        rows = flat_series(100.0, 5)
        rows.append(kline(100.0, 101.2, 100.0, 101.0, t=5))  # decision candle
        rows.append(kline(101.0, 101.1, 100.5, 101.0, t=6))  # entry fills
        # one violent candle touches both stop (-1%) and target (+1.5%)
        rows.append(kline(101.0, 105.0, 99.0, 101.0, t=7))
        rows.extend(kline(101.0, 101.05, 100.95, 101.0, t=8 + i) for i in range(3))

        result = Backtester(self.config).run("BTCUSDT", rows)

        self.assertEqual(len(result.trades), 1)
        self.assertEqual(result.trades[0].exit_reason, "STOP_LOSS")
        self.assertLess(result.trades[0].realized_pnl, 0.0)

    def test_unfilled_entry_expires(self) -> None:
        # Spike then gap up and hold: the limit at the old bid never fills and
        # the series ends before any later proposal can fill either.
        rows = flat_series(100.0, 5)
        rows.append(kline(100.0, 101.2, 100.0, 101.0, t=5))
        rows.extend(kline(105.0, 105.5, 104.5, 105.0, t=6 + i) for i in range(3))

        result = Backtester(self.config).run("BTCUSDT", rows)

        self.assertGreaterEqual(result.proposals, 1)
        self.assertEqual(result.trades, [])

    def test_candle_parsing(self) -> None:
        candle = Candle.from_kline([123, "1.5", "2.0", "1.0", "1.8", "42.0", 456])
        self.assertEqual(candle.open_time, 123)
        self.assertEqual(candle.high, 2.0)
        self.assertEqual(candle.volume, 42.0)

    def test_buy_hold_benchmark_reflects_series_drift(self) -> None:
        rows = flat_series(100.0, 2) + [kline(110.0, 110.1, 109.9, 110.0, t=2)]
        result = Backtester(self.config).run("BTCUSDT", rows)
        self.assertGreater(result.buy_hold_pnl, 9.0)  # +10% on 100 USD minus fees


if __name__ == "__main__":
    unittest.main()
