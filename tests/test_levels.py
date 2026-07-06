from __future__ import annotations

import unittest

from trading_agent.core import levels
from trading_agent.core.levels import Candle


def _c(open_time: int, o: float, h: float, l: float, cl: float, v: float = 100.0) -> Candle:
    return Candle(open_time=open_time, open=o, high=h, low=l, close=cl, volume=v)


class FractalPivotTest(unittest.TestCase):
    def test_detects_swing_high_and_low(self) -> None:
        # A clear peak at index 2 and trough at index 6.
        candles = [
            _c(0, 10, 11, 9, 10),
            _c(1, 10, 12, 10, 11),
            _c(2, 11, 20, 11, 15),  # swing high (20)
            _c(3, 15, 16, 13, 14),
            _c(4, 14, 14, 12, 13),
            _c(5, 13, 13, 8, 9),
            _c(6, 9, 9, 2, 5),  # swing low (2)
            _c(7, 5, 8, 5, 7),
            _c(8, 7, 10, 7, 9),
        ]
        highs, lows = levels.fractal_pivots(candles, left=2, right=2)
        self.assertIn(20.0, highs)
        self.assertIn(2.0, lows)


class VolumeByPriceTest(unittest.TestCase):
    def test_high_volume_node_surfaces(self) -> None:
        # Most volume concentrated around price ~50.
        candles = [_c(i, 50, 51, 49, 50, v=1000.0) for i in range(10)]
        candles += [_c(10 + i, 80, 81, 79, 80, v=10.0) for i in range(5)]
        nodes = levels.volume_by_price(candles, bins=20)
        self.assertTrue(nodes)
        # The strongest node should sit near 50, not near 80.
        strongest = max(nodes, key=lambda n: n[1])
        self.assertLess(abs(strongest[0] - 50.0), 5.0)


class FibTest(unittest.TestCase):
    def test_fib_levels_between_low_and_high(self) -> None:
        fib = levels.fib_retracement(100.0, 0.0)
        self.assertAlmostEqual(fib["0.5"], 50.0)
        self.assertAlmostEqual(fib["0.618"], 38.2)

    def test_inverted_swing_returns_empty(self) -> None:
        self.assertEqual(levels.fib_retracement(10.0, 20.0), {})


class RoundLevelTest(unittest.TestCase):
    def test_step_scales_with_price(self) -> None:
        self.assertGreater(levels.round_number_step(64000), levels.round_number_step(74))
        nearby = levels.round_levels(74.0)
        self.assertTrue(any(abs(x - 74.0) <= levels.round_number_step(74.0) for x in nearby))


class ClusterTest(unittest.TestCase):
    def test_nearby_levels_merge_and_sum_strength(self) -> None:
        lv = [
            levels.PriceLevel(100.0, "swing_low", "1d", 2.0),
            levels.PriceLevel(100.2, "hvn", "1d", 3.0),
            levels.PriceLevel(130.0, "swing_high", "1d", 2.0),
        ]
        zones = levels.cluster_levels(lv, current_price=120.0, tolerance_pct=0.01)
        # The two ~100 levels merge into one support zone; 130 is a resistance zone.
        support = [z for z in zones if z.side == "support"]
        resistance = [z for z in zones if z.side == "resistance"]
        self.assertEqual(len(support), 1)
        self.assertEqual(support[0].touches, 2)
        self.assertAlmostEqual(support[0].strength, 5.0)
        self.assertTrue(resistance)


class RegimeTest(unittest.TestCase):
    def test_uptrend_when_price_above_stacked_emas(self) -> None:
        closes = [float(i) for i in range(1, 80)]  # steadily rising
        self.assertEqual(levels.classify_regime(closes, closes), "uptrend")

    def test_downtrend_when_price_below_stacked_emas(self) -> None:
        closes = [float(i) for i in range(80, 1, -1)]  # steadily falling
        self.assertEqual(levels.classify_regime(closes, closes), "downtrend")

    def test_short_series_is_range(self) -> None:
        self.assertEqual(levels.classify_regime([1.0, 2.0, 3.0]), "range")


class BuildLevelMapTest(unittest.TestCase):
    def test_splits_support_and_resistance_around_price(self) -> None:
        daily = [_c(i, 90 + i, 92 + i, 88 + i, 91 + i, v=500.0) for i in range(60)]
        candles_by_tf = {"1d": daily, "1w": daily, "1h": daily}
        price = daily[-1].close
        lmap = levels.build_level_map("BTCUSDT", candles_by_tf, price)
        self.assertEqual(lmap.symbol, "BTCUSDT")
        for zone in lmap.support_zones:
            self.assertLess(zone.high, price)
        for zone in lmap.resistance_zones:
            self.assertGreater(zone.low, price)

    def test_empty_input_is_safe(self) -> None:
        lmap = levels.build_level_map("ETHUSDT", {}, 1000.0)
        self.assertEqual(lmap.support_zones, [])
        self.assertEqual(lmap.resistance_zones, [])
        self.assertEqual(lmap.regime, "range")


if __name__ == "__main__":
    unittest.main()
