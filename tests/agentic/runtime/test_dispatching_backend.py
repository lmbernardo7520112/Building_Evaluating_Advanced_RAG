"""Contract tests for public DispatchingRetrievalBackend."""

import unittest
from typing import Any
from unittest.mock import MagicMock

from raglab.agentic.contracts import ToolObservation
from raglab.agentic.enums import InvocationStatus
from raglab.agentic.runtime.retrieval_tool_adapter import RetrievalToolAdapter


def _get_dispatching_backend_cls() -> type:
    """Import and return the target class at test execution time."""
    from raglab.agentic.runtime.dispatching_backend import (
        DispatchingRetrievalBackend,
    )

    return DispatchingRetrievalBackend


class TestDispatchingRetrievalBackendContract(unittest.TestCase):
    """Test public DispatchingRetrievalBackend contract."""

    def setUp(self) -> None:
        self.mock_adapter_baseline = MagicMock(spec=RetrievalToolAdapter)
        self.obs_baseline = ToolObservation(
            invocation_id="inv_baseline_01",
            status=InvocationStatus.EXECUTED,
            passage_ids=("ps_1",),
            document_ids=("doc_1",),
            ranks=(1,),
            scores=(0.95,),
            content_hashes=("abc",),
            retrieval_config_hash="hash1",
            latency_ms=12.5,
        )
        self.mock_adapter_baseline.retrieve.return_value = self.obs_baseline

        self.mock_adapter_window = MagicMock(spec=RetrievalToolAdapter)
        self.obs_window = ToolObservation(
            invocation_id="inv_window_01",
            status=InvocationStatus.EXECUTED,
            passage_ids=("ps_2",),
            document_ids=("doc_2",),
            ranks=(1,),
            scores=(0.98,),
            content_hashes=("def",),
            retrieval_config_hash="hash2",
            latency_ms=18.3,
        )
        self.mock_adapter_window.retrieve.return_value = self.obs_window

        self.adapters = {
            "retrieve_baseline": self.mock_adapter_baseline,
            "retrieve_sentence_window_rerank": self.mock_adapter_window,
        }

    def _make_dispatcher(self) -> Any:
        cls = _get_dispatching_backend_cls()
        return cls(self.adapters)

    def test_01_dispatch_baseline_invokes_baseline_adapter_only(self) -> None:
        dispatcher = self._make_dispatcher()
        obs = dispatcher.retrieve(
            query="test query baseline",
            strategy="baseline",
            top_k=3,
        )

        self.assertEqual(obs, self.obs_baseline)
        self.mock_adapter_baseline.retrieve.assert_called_once_with(
            query="test query baseline",
            strategy="baseline",
            top_k=3,
        )
        self.mock_adapter_window.retrieve.assert_not_called()

    def test_02_dispatch_sentence_window_rerank_invokes_its_adapter_only(
        self,
    ) -> None:
        dispatcher = self._make_dispatcher()
        obs = dispatcher.retrieve(
            query="compare A and B",
            strategy="sentence_window_rerank",
            top_k=5,
        )

        self.assertEqual(obs, self.obs_window)
        self.mock_adapter_window.retrieve.assert_called_once_with(
            query="compare A and B",
            strategy="sentence_window_rerank",
            top_k=5,
        )
        self.mock_adapter_baseline.retrieve.assert_not_called()

    def test_03_query_strategy_top_k_forwarded_correctly(self) -> None:
        dispatcher = self._make_dispatcher()
        dispatcher.retrieve(
            query="exact question",
            strategy="baseline",
            top_k=10,
        )

        self.mock_adapter_baseline.retrieve.assert_called_once_with(
            query="exact question",
            strategy="baseline",
            top_k=10,
        )

    def test_04_unknown_strategy_rejected_fail_closed(self) -> None:
        dispatcher = self._make_dispatcher()

        with self.assertRaises((ValueError, KeyError)):
            dispatcher.retrieve(
                query="some query",
                strategy="unknown_strategy",
                top_k=3,
            )

        self.mock_adapter_baseline.retrieve.assert_not_called()
        self.mock_adapter_window.retrieve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
