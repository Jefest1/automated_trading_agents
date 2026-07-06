from __future__ import annotations

import unittest

from trading_agent.core.models import FillRecord, Side
from trading_agent.core.pnl import round_trip_pnl, split_symbol


def fill(price: float, qty: float, commission: float, asset: str, *, is_exit: bool) -> FillRecord:
    return FillRecord(
        order_id="ord_1",
        symbol="BTCUSDT",
        side=Side.SELL if is_exit else Side.BUY,
        price=price,
        qty=qty,
        quote_qty=price * qty,
        commission=commission,
        commission_asset=asset,
        is_exit=is_exit,
    )


class RoundTripPnlTest(unittest.TestCase):
    def test_quote_asset_commissions_are_exact(self) -> None:
        entry = [fill(100.0, 1.0, 0.1, "USDT", is_exit=False)]
        exit_ = [fill(110.0, 1.0, 0.11, "USDT", is_exit=True)]
        result = round_trip_pnl(entry, exit_, base_asset="BTC", quote_asset="USDT")
        self.assertAlmostEqual(result.realized_pnl, 110.0 - 100.0 - 0.21)
        self.assertFalse(result.estimated)
        self.assertAlmostEqual(result.commission_total_quote, 0.21)

    def test_base_asset_commission_valued_at_fill_price(self) -> None:
        entry = [fill(100.0, 1.0, 0.001, "BTC", is_exit=False)]  # 0.001 BTC @ 100 = 0.1 USDT
        exit_ = [fill(110.0, 1.0, 0.0, "USDT", is_exit=True)]
        result = round_trip_pnl(entry, exit_, base_asset="BTC", quote_asset="USDT")
        self.assertAlmostEqual(result.realized_pnl, 10.0 - 0.1)
        self.assertFalse(result.estimated)

    def test_bnb_commission_flags_estimate(self) -> None:
        entry = [fill(100.0, 1.0, 0.005, "BNB", is_exit=False)]
        exit_ = [fill(110.0, 1.0, 0.0, "USDT", is_exit=True)]
        result = round_trip_pnl(entry, exit_, base_asset="BTC", quote_asset="USDT")
        self.assertAlmostEqual(result.realized_pnl, 10.0)  # BNB fee excluded, flagged
        self.assertTrue(result.estimated)
        self.assertAlmostEqual(result.unconverted_commissions["BNB"], 0.005)

    def test_bnb_commission_converted_with_supplied_price(self) -> None:
        entry = [fill(100.0, 1.0, 0.005, "BNB", is_exit=False)]  # 0.005 BNB @ 600 = 3 USDT
        exit_ = [fill(110.0, 1.0, 0.0, "USDT", is_exit=True)]
        result = round_trip_pnl(
            entry, exit_, base_asset="BTC", quote_asset="USDT", conversion_prices={"BNB": 600.0}
        )
        self.assertAlmostEqual(result.realized_pnl, 10.0 - 3.0)
        self.assertAlmostEqual(result.commission_total_quote, 3.0)
        self.assertTrue(result.estimated)  # converted at an approximate price
        self.assertEqual(result.unconverted_commissions, {})

    def test_partial_fills_aggregate(self) -> None:
        entry = [
            fill(100.0, 0.4, 0.04, "USDT", is_exit=False),
            fill(99.0, 0.6, 0.06, "USDT", is_exit=False),
        ]
        exit_ = [fill(105.0, 1.0, 0.105, "USDT", is_exit=True)]
        result = round_trip_pnl(entry, exit_, base_asset="BTC", quote_asset="USDT")
        entry_quote = 100.0 * 0.4 + 99.0 * 0.6
        self.assertAlmostEqual(result.realized_pnl, 105.0 - entry_quote - 0.205)
        self.assertAlmostEqual(result.entry.avg_price, entry_quote / 1.0)

    def test_partial_exit_realizes_only_sold_portion(self) -> None:
        # A TP1 scale-out (40% of the entry) sold higher: realize PnL ONLY on the
        # sold portion, NOT the full entry cost against a partial exit (which used
        # to produce a large phantom loss, e.g. -56 here / -19 on the live SOL).
        entry = [fill(100.0, 1.0, 0.10, "USDT", is_exit=False)]
        exit_ = [fill(110.0, 0.4, 0.044, "USDT", is_exit=True)]
        result = round_trip_pnl(entry, exit_, base_asset="BTC", quote_asset="USDT")
        # 0.4 sold @110 (44) − cost basis 0.4×100 (40) − fees (0.04 entry-prorated + 0.044) = 3.916
        self.assertAlmostEqual(result.realized_pnl, 44.0 - 40.0 - (0.04 + 0.044))
        self.assertGreater(result.realized_pnl, 0.0)
        self.assertAlmostEqual(result.commission_total_quote, 0.084)

    def test_tiered_exits_reconcile_to_full_close(self) -> None:
        # Cumulative realized after each tier (round_trip_pnl is recomputed from
        # ALL fills so far) must equal selling those quantities at average cost,
        # and the full close equals the sum of the per-tier realizations.
        entry = [fill(100.0, 1.0, 0.10, "USDT", is_exit=False)]
        tp1 = fill(110.0, 0.4, 0.044, "USDT", is_exit=True)
        tp2 = fill(120.0, 0.3, 0.036, "USDT", is_exit=True)
        runner = fill(130.0, 0.3, 0.039, "USDT", is_exit=True)

        after_tp1 = round_trip_pnl(entry, [tp1], base_asset="BTC", quote_asset="USDT")
        after_tp2 = round_trip_pnl(entry, [tp1, tp2], base_asset="BTC", quote_asset="USDT")
        full = round_trip_pnl(entry, [tp1, tp2, runner], base_asset="BTC", quote_asset="USDT")

        self.assertAlmostEqual(after_tp1.realized_pnl, 44.0 - 40.0 - (0.04 + 0.044))   # 3.916
        self.assertAlmostEqual(after_tp2.realized_pnl, 80.0 - 70.0 - (0.07 + 0.08))    # 9.85
        # Full close: cost basis is the entire entry (exit_fraction == 1.0).
        self.assertAlmostEqual(full.realized_pnl, 119.0 - 100.0 - (0.10 + 0.119))      # 18.781
        # Monotonic banking; cumulative never regresses.
        self.assertLess(after_tp1.realized_pnl, after_tp2.realized_pnl)
        self.assertLess(after_tp2.realized_pnl, full.realized_pnl)


class SplitSymbolTest(unittest.TestCase):
    def test_common_quotes(self) -> None:
        self.assertEqual(split_symbol("BTCUSDT"), ("BTC", "USDT"))
        self.assertEqual(split_symbol("ethbtc"), ("ETH", "BTC"))
        self.assertEqual(split_symbol("SOLUSDC"), ("SOL", "USDC"))


if __name__ == "__main__":
    unittest.main()
