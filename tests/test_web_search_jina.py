from __future__ import annotations

import json
import unittest

from trading_agent.utils import web_search


class JinaWebSearchTest(unittest.TestCase):
    def setUp(self) -> None:
        self._orig = {
            name: getattr(web_search, name)
            for name in ("_jina_search", "_ddg_text", "_jina_reader")
        }

    def tearDown(self) -> None:
        for name, fn in self._orig.items():
            setattr(web_search, name, fn)

    def test_search_prefers_jina(self) -> None:
        web_search._jina_search = lambda q, n: [  # type: ignore[assignment]
            {"title": "T", "url": "https://x", "snippet": "s", "date": ""}
        ]
        web_search._ddg_text = lambda q, n: (_ for _ in ()).throw(AssertionError("DDG should not run"))  # type: ignore[assignment]
        rows = web_search.run_web_search("bitcoin etf flows")
        self.assertEqual(rows, [{"title": "T", "url": "https://x", "snippet": "s"}])

    def test_search_falls_back_to_ddg_when_jina_empty(self) -> None:
        web_search._jina_search = lambda q, n: []  # type: ignore[assignment]
        web_search._ddg_text = lambda q, n: [{"title": "D", "url": "https://d", "snippet": "ds"}]  # type: ignore[assignment]
        rows = web_search.run_web_search("solana")
        self.assertEqual(rows[0]["url"], "https://d")

    def test_search_falls_back_to_ddg_when_jina_raises(self) -> None:
        def boom(q, n):
            raise RuntimeError("jina down")

        web_search._jina_search = boom  # type: ignore[assignment]
        web_search._ddg_text = lambda q, n: [{"title": "D", "url": "https://d", "snippet": "ds"}]  # type: ignore[assignment]
        rows = web_search.run_web_search("eth")
        self.assertEqual(rows[0]["title"], "D")

    def test_fetch_url_prefers_jina_reader(self) -> None:
        web_search._jina_reader = lambda url, n: "clean markdown body"  # type: ignore[assignment]
        out = web_search.run_fetch_url("https://example.com/article")
        self.assertEqual(out, "clean markdown body")

    def test_tool_wrapper_returns_ok_json(self) -> None:
        web_search._jina_search = lambda q, n: [{"title": "T", "url": "https://x", "snippet": "s", "date": ""}]  # type: ignore[assignment]
        payload = json.loads(web_search.web_search.invoke({"query": "btc"}))
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["results"][0]["url"], "https://x")


if __name__ == "__main__":
    unittest.main()
