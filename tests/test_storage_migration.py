from __future__ import annotations

import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from trading_agent.core.storage import Store

V1_ORDERS = """
CREATE TABLE orders (
    id TEXT PRIMARY KEY,
    proposal_id TEXT NOT NULL,
    mode TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    order_type TEXT NOT NULL,
    price REAL NOT NULL,
    quantity REAL NOT NULL,
    take_profit_price REAL NOT NULL,
    stop_loss_price REAL NOT NULL,
    status TEXT NOT NULL,
    opened_at TEXT NOT NULL,
    closed_at TEXT,
    entry_fee REAL NOT NULL,
    exit_fee REAL NOT NULL,
    realized_pnl REAL NOT NULL,
    exit_price REAL
);
"""


class StorageMigrationTest(unittest.TestCase):
    def test_v1_database_gains_exchange_columns_and_new_tables(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.sqlite3"
            conn = sqlite3.connect(path)
            conn.executescript(V1_ORDERS)
            conn.execute(
                "INSERT INTO orders VALUES ('ord_1', 'tp_1', 'paper', 'BTCUSDT', 'BUY', "
                "'SPOT_LIMIT_ENTRY', 100.0, 1.0, 105.0, 95.0, 'ENTRY_OPEN', "
                "'2026-01-01T00:00:00+00:00', NULL, 0.0, 0.0, 0.0, NULL)"
            )
            conn.commit()
            conn.close()

            with Store(path) as store:
                columns = {
                    row["name"] for row in store.conn.execute("PRAGMA table_info(orders)").fetchall()
                }
                self.assertIn("exit_reason", columns)
                self.assertIn("exchange_order_id", columns)
                self.assertIn("exchange_status", columns)
                self.assertIn("executed_qty", columns)
                self.assertIn("pnl_estimated", columns)
                tables = {
                    row["name"]
                    for row in store.conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
                self.assertIn("fills", tables)
                self.assertIn("supervisor_decisions", tables)
                # the migrated legacy row still loads through the dataclass path
                orders = store.open_positions()
                self.assertEqual(orders[0].id, "ord_1")
                self.assertEqual(orders[0].executed_qty, 0.0)
                self.assertIsNone(orders[0].exchange_order_id)


if __name__ == "__main__":
    unittest.main()
