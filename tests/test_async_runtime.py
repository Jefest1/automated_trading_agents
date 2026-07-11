from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from trading_agent.core.config import AppConfig, Settings
from trading_agent.core.storage import Store
from trading_agent.graph import SupervisorRuntime
from trading_agent.graph.checkpointer import CHECKPOINT_DB_FILENAME, CheckpointerFactory
from trading_agent.graph import deep_agent as deep_agent_module


class TinyCheckpointState(TypedDict):
    value: int


def tiny_checkpoint_graph() -> StateGraph:
    graph = StateGraph(TinyCheckpointState)

    def increment(state: TinyCheckpointState) -> TinyCheckpointState:
        return {"value": state["value"] + 1}

    graph.add_node("increment", increment)
    graph.add_edge(START, "increment")
    graph.add_edge("increment", END)
    return graph


class AsyncRuntimeTest(unittest.IsolatedAsyncioTestCase):
    async def test_arun_once_completes_and_creates_checkpoint_db(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = AppConfig(home=str(root), database_path=str(root / "agent.sqlite3"))
            with Store(config.database_path) as store:
                runtime = SupervisorRuntime(config, store, settings=Settings())
                result = await runtime.arun_once(cycle=1)
                summary = store.summary()

            self.assertEqual(result.cycle, 1)
            self.assertEqual(summary["latest_agent_run"]["status"], "completed")
            self.assertTrue((root / CHECKPOINT_DB_FILENAME).exists())

    async def test_checkpointer_factory_round_trips_per_call(self) -> None:
        with TemporaryDirectory() as tmp:
            factory = CheckpointerFactory(tmp)
            # Two separate opens must not share loop-bound state but hit one file.
            async with factory.open() as saver:
                self.assertIsNotNone(saver)
            async with factory.open() as saver:
                self.assertIsNotNone(saver)
            self.assertTrue(factory.path.exists())

    async def test_sqlite_checkpointer_persists_async_graph_state(self) -> None:
        with TemporaryDirectory() as tmp:
            factory = CheckpointerFactory(tmp)
            config = {"configurable": {"thread_id": "thread-async"}}
            async with factory.open() as saver:
                graph = tiny_checkpoint_graph().compile(checkpointer=saver)
                await graph.ainvoke({"value": 41}, config)

            async with factory.open() as saver:
                graph = tiny_checkpoint_graph().compile(checkpointer=saver)
                snapshot = await graph.aget_state(config)

            self.assertEqual(snapshot.values["value"], 42)


class SyncCheckpointerTest(unittest.TestCase):
    def test_sqlite_checkpointer_persists_sync_graph_state(self) -> None:
        with TemporaryDirectory() as tmp:
            factory = CheckpointerFactory(tmp)
            config = {"configurable": {"thread_id": "thread-sync"}}
            with factory.open_sync() as saver:
                graph = tiny_checkpoint_graph().compile(checkpointer=saver)
                graph.invoke({"value": 9}, config)

            with factory.open_sync() as saver:
                graph = tiny_checkpoint_graph().compile(checkpointer=saver)
                snapshot = graph.get_state(config)

            self.assertEqual(snapshot.values["value"], 10)

    def test_runtime_passes_active_sqlite_checkpointer_to_deep_agent(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = AppConfig(home=str(root), database_path=str(root / "agent.sqlite3"))
            captured: dict[str, object] = {}
            original = deep_agent_module.build_deep_agent

            def fake_build_deep_agent(**kwargs):  # type: ignore[no-untyped-def]
                captured.update(kwargs)
                return object()

            with Store(config.database_path) as store:
                runtime = SupervisorRuntime(config, store, settings=Settings())
                runtime._deep_agent_model = lambda: object()  # type: ignore[method-assign]
                runtime.subagent_specs = lambda tools=None: []  # type: ignore[method-assign]
                runtime._skill_source_paths = lambda: []  # type: ignore[method-assign]
                deep_agent_module.build_deep_agent = fake_build_deep_agent  # type: ignore[assignment]
                try:
                    with runtime.checkpointer_factory.open_sync() as checkpointer:
                        runtime._active_checkpointer = checkpointer
                        runtime.build_deep_agent([])
                        self.assertIs(captured["checkpointer"], checkpointer)
                finally:
                    runtime._active_checkpointer = None
                    deep_agent_module.build_deep_agent = original  # type: ignore[assignment]


class ThreadRotationTest(unittest.TestCase):
    def test_pre_sqlite_thread_id_is_rotated_once(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = AppConfig(home=str(root), database_path=str(root / "agent.sqlite3"))
            with Store(config.database_path) as store:
                store.set_setting("agent_thread_id", "thread_legacy")
                runtime = SupervisorRuntime(config, store, settings=Settings())
                first = runtime._resolve_thread_id(None)
                second = runtime._resolve_thread_id(None)

            self.assertNotEqual(first, "thread_legacy")
            self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()


class TransientNetworkErrorTest(unittest.TestCase):
    def test_detects_wrapped_read_error(self) -> None:
        from trading_agent.graph.nodes import _is_transient_network_error

        class ReadError(Exception):
            pass

        inner = ReadError("connection dropped")
        outer = RuntimeError("stream failed")
        outer.__cause__ = inner
        self.assertTrue(_is_transient_network_error(outer))

    def test_non_network_error_is_not_transient(self) -> None:
        from trading_agent.graph.nodes import _is_transient_network_error

        self.assertFalse(_is_transient_network_error(ValueError("bad decision json")))
