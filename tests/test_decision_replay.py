from __future__ import annotations

import unittest

from trading_agent.backtest import Candle
from trading_agent.core.config import AppConfig, ExitConfig
from trading_agent.decision_replay import replay_recorded_decisions


def candle(low: float, high: float, close: float = 0.0) -> Candle:
    return Candle(
        open=high,
        high=high,
        low=low,
        close=close or high,
        volume=1.0,
        open_time=0,
    )


def buy_record(
    decision_id: str,
    symbol: str = "BTCUSDT",
    *,
    limit_price: float = 100.0,
    quantity: float = 1.0,
    executed: bool = True,
    gate_approved: bool = True,
) -> dict:
    return {
        "id": decision_id,
        "created_at": "2026-06-12T00:00:00+00:00",
        "action": "BUY",
        "symbol": symbol,
        "gate_approved": int(gate_approved),
        "executed_order_id": "ord_x" if executed else None,
        "payload": {
            "action": "BUY",
            "symbol": symbol,
            "limit_price": limit_price,
            "quantity": quantity,
            "stop_loss_pct": 0.01,
            "take_profit_pct": 0.015,
        },
    }


class DecisionReplayTest(unittest.TestCase):
    def test_take_profit_is_a_win(self) -> None:
        # Legacy single-leg replay (tiered scale-out covered separately below).
        config = AppConfig(exits=ExitConfig(enabled=False))
        rec = buy_record("dec_tp")
        # candle0 fills the limit; candle1 reaches take-profit (>=101.5).
        window = {"dec_tp": [candle(low=99.0, high=100.0, close=100.0), candle(low=100.5, high=102.0, close=101.5)]}
        result = replay_recorded_decisions([rec], window, config)
        self.assertEqual(result.buy_decisions, 1)
        self.assertEqual(result.filled, 1)
        self.assertEqual(result.wins, 1)
        self.assertGreater(result.realized_pnl, 0.0)
        self.assertEqual(result.trades[0].exit_reason, "TAKE_PROFIT")

    def test_tiered_replay_scales_out_and_trails(self) -> None:
        # Default tiered exits: a strong up-move banks TP1 (+3%), TP2 (+6%), and
        # TP3 (+10%), then the runner rides a trailing stop out on the pullback.
        config = AppConfig()  # exits.enabled defaults True
        rec = buy_record("dec_tiered")
        window = {
            "dec_tiered": [
                candle(low=99.0, high=100.0, close=100.0),    # fills entry @100
                candle(low=100.5, high=103.2, close=103.0),   # TP1 (103) banks -> stop 100
                candle(low=102.0, high=106.1, close=106.0),   # TP2 (106) banks -> stop 103
                candle(low=104.0, high=110.2, close=110.0),   # TP3 (110) banks -> stop 106; runner active
                candle(low=108.0, high=114.0, close=113.5),   # runner high-water 114 -> trail 110.58
                candle(low=110.0, high=112.0, close=110.5),   # trail stop (114*0.97=110.58) hit
            ]
        }
        result = replay_recorded_decisions([rec], window, config)
        self.assertEqual(result.filled, 1)
        self.assertEqual(result.wins, 1)
        self.assertGreater(result.realized_pnl, 0.0)
        self.assertEqual(result.trades[0].exit_reason, "TRAIL_STOP")

    def test_stop_loss_is_a_loss(self) -> None:
        # Legacy single-leg replay using the decision's 1% stop fallback, so the
        # candle design is independent of the production swing-config defaults.
        config = AppConfig(exits=ExitConfig(enabled=False))
        rec = buy_record("dec_sl")
        # candle0 fills; candle1 breaks the stop (<=99).
        window = {"dec_sl": [candle(low=100.0, high=100.5, close=100.0), candle(low=98.0, high=100.0, close=98.5)]}
        result = replay_recorded_decisions([rec], window, config)
        self.assertEqual(result.filled, 1)
        self.assertEqual(result.losses, 1)
        self.assertLess(result.realized_pnl, 0.0)
        self.assertEqual(result.trades[0].exit_reason, "STOP_LOSS")

    def test_unfilled_when_limit_never_reached(self) -> None:
        config = AppConfig()
        rec = buy_record("dec_unfilled", limit_price=100.0)
        window = {"dec_unfilled": [candle(low=101.0, high=103.0), candle(low=102.0, high=104.0)]}
        result = replay_recorded_decisions([rec], window, config)
        self.assertEqual(result.buy_decisions, 1)
        self.assertEqual(result.filled, 0)
        self.assertEqual(result.trades[0].exit_reason, "UNFILLED")
        self.assertEqual(result.realized_pnl, 0.0)

    def test_wait_decisions_are_counted_but_not_traded(self) -> None:
        config = AppConfig()
        wait = {
            "id": "dec_wait",
            "created_at": "2026-06-12T00:00:00+00:00",
            "action": "WAIT",
            "symbol": "ETHUSDT",
            "gate_approved": None,
            "executed_order_id": None,
            "payload": {"action": "WAIT", "symbol": "ETHUSDT"},
        }
        result = replay_recorded_decisions([wait], {}, config)
        self.assertEqual(result.decisions_total, 1)
        self.assertEqual(result.buy_decisions, 0)
        self.assertEqual(result.trades, [])

    def test_missing_window_is_reported_not_crashed(self) -> None:
        config = AppConfig()
        rec = buy_record("dec_nodata")
        result = replay_recorded_decisions([rec], {}, config)
        self.assertEqual(result.buy_decisions, 1)
        self.assertEqual(result.filled, 0)
        self.assertIn("no price window", result.trades[0].note)

    def test_aggregates_across_symbols(self) -> None:
        config = AppConfig()
        records = [buy_record("dec_tp", "BTCUSDT"), buy_record("dec_sl", "ETHUSDT", limit_price=100.0)]
        windows = {
            "dec_tp": [candle(low=99.0, high=100.0, close=100.0), candle(low=100.5, high=102.0, close=101.5)],
            "dec_sl": [candle(low=100.0, high=100.5, close=100.0), candle(low=98.0, high=100.0, close=98.5)],
        }
        result = replay_recorded_decisions(records, windows, config)
        self.assertEqual(result.filled, 2)
        self.assertEqual(result.wins, 1)
        self.assertEqual(result.losses, 1)
        self.assertIn("BTCUSDT", result.per_symbol_pnl)
        self.assertIn("ETHUSDT", result.per_symbol_pnl)
        summary = result.summary()
        self.assertEqual(summary["win_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
