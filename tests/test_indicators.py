from __future__ import annotations

import unittest

from trading_agent.utils import indicators


class SmaEmaTest(unittest.TestCase):
    def test_sma_known_vector(self) -> None:
        out = indicators.sma([1.0, 2.0, 3.0, 4.0, 5.0], 3)
        self.assertEqual(out[:2], [None, None])
        self.assertAlmostEqual(out[2], 2.0)
        self.assertAlmostEqual(out[3], 3.0)
        self.assertAlmostEqual(out[4], 4.0)

    def test_ema_seeds_with_sma_then_smooths(self) -> None:
        values = [22.27, 22.19, 22.08, 22.17, 22.18, 22.13, 22.23, 22.43, 22.24, 22.29]
        out = indicators.ema(values, 10)
        self.assertEqual(out[:9], [None] * 9)
        self.assertAlmostEqual(out[9], sum(values) / 10)

    def test_latest_skips_none(self) -> None:
        self.assertEqual(indicators.latest([None, 1.0, None]), 1.0)
        self.assertIsNone(indicators.latest([None, None]))


class RsiTest(unittest.TestCase):
    def test_rsi_all_gains_is_100(self) -> None:
        values = [float(i) for i in range(1, 20)]
        out = indicators.rsi(values, 14)
        self.assertAlmostEqual(out[-1], 100.0)

    def test_rsi_classic_wilder_vector(self) -> None:
        # Wilder's worked example data (14-period), first RSI ~70.46.
        values = [
            44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42,
            45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28,
        ]
        out = indicators.rsi(values, 14)
        self.assertIsNotNone(out[14])
        self.assertAlmostEqual(out[14], 70.46, delta=0.1)


class MacdTest(unittest.TestCase):
    def test_macd_line_is_fast_minus_slow(self) -> None:
        values = [float(i) for i in range(1, 60)]
        macd_line, signal_line, histogram = indicators.macd(values)
        self.assertIsNotNone(macd_line[-1])
        self.assertIsNotNone(signal_line[-1])
        self.assertAlmostEqual(histogram[-1], macd_line[-1] - signal_line[-1])


class AtrTest(unittest.TestCase):
    def test_atr_constant_range(self) -> None:
        highs = [110.0] * 20
        lows = [100.0] * 20
        closes = [105.0] * 20
        out = indicators.atr(highs, lows, closes, 14)
        self.assertAlmostEqual(out[-1], 10.0)


class BollingerTest(unittest.TestCase):
    def test_constant_series_collapses_bands(self) -> None:
        values = [50.0] * 25
        middle, upper, lower = indicators.bollinger(values)
        self.assertAlmostEqual(middle[-1], 50.0)
        self.assertAlmostEqual(upper[-1], 50.0)
        self.assertAlmostEqual(lower[-1], 50.0)


class SparklineTest(unittest.TestCase):
    def test_sparkline_shape(self) -> None:
        spark = indicators.ascii_sparkline([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(len(spark), 4)
        self.assertEqual(spark[0], "▁")
        self.assertEqual(spark[-1], "█")

    def test_sparkline_downsamples(self) -> None:
        spark = indicators.ascii_sparkline([float(i) for i in range(200)], width=40)
        self.assertEqual(len(spark), 40)

    def test_sparkline_empty_and_flat(self) -> None:
        self.assertEqual(indicators.ascii_sparkline([]), "")
        self.assertEqual(indicators.ascii_sparkline([5.0, 5.0]), "▁▁")


if __name__ == "__main__":
    unittest.main()
