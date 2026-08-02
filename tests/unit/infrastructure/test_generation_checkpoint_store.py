"""Tests for GenerationCheckpointStore — offline, no credentials."""

from __future__ import annotations

import json
import pytest


class TestGenerationCheckpointStore:
    def test_initially_empty(self, tmp_path):
        from raglab.infrastructure.persistence.generation_checkpoint_store import (
            GenerationCheckpointStore,
        )
        store = GenerationCheckpointStore(run_id="test_run", store_dir=tmp_path)
        assert store.completed_count() == 0
        assert store.is_completed("q_dev_01", "F0_baseline") is False

    def test_mark_and_check_completed(self, tmp_path):
        from raglab.infrastructure.persistence.generation_checkpoint_store import (
            GenerationCheckpointStore,
        )
        store = GenerationCheckpointStore(run_id="test_run", store_dir=tmp_path)
        store.mark_completed("q_dev_01", "F0_baseline", abstained=False, citation_count=2)
        assert store.is_completed("q_dev_01", "F0_baseline") is True
        assert store.is_completed("q_dev_02", "F0_baseline") is False

    def test_persists_to_disk(self, tmp_path):
        from raglab.infrastructure.persistence.generation_checkpoint_store import (
            GenerationCheckpointStore,
        )
        store = GenerationCheckpointStore(run_id="persist_run", store_dir=tmp_path)
        store.mark_completed("q_dev_01", "W0_sentence_window", abstained=False, citation_count=3)

        # Reload
        store2 = GenerationCheckpointStore(run_id="persist_run", store_dir=tmp_path)
        assert store2.is_completed("q_dev_01", "W0_sentence_window") is True
        assert store2.completed_count() == 1

    def test_different_run_ids_isolated(self, tmp_path):
        from raglab.infrastructure.persistence.generation_checkpoint_store import (
            GenerationCheckpointStore,
        )
        store_a = GenerationCheckpointStore(run_id="run_a", store_dir=tmp_path)
        store_a.mark_completed("q_dev_01", "F0_baseline", abstained=False, citation_count=1)

        store_b = GenerationCheckpointStore(run_id="run_b", store_dir=tmp_path)
        assert store_b.is_completed("q_dev_01", "F0_baseline") is False

    def test_no_credentials_in_checkpoint_file(self, tmp_path):
        from raglab.infrastructure.persistence.generation_checkpoint_store import (
            GenerationCheckpointStore,
        )
        store = GenerationCheckpointStore(run_id="secure_run", store_dir=tmp_path)
        store.mark_completed("q_dev_01", "F0_baseline", abstained=False, citation_count=2)

        ckpt_files = list(tmp_path.glob("*.json"))
        assert len(ckpt_files) == 1
        content = ckpt_files[0].read_text()
        assert "GEMINI_API_KEY" not in content
        assert "API_KEY" not in content

    def test_schema_version_in_file(self, tmp_path):
        from raglab.infrastructure.persistence.generation_checkpoint_store import (
            GenerationCheckpointStore,
        )
        store = GenerationCheckpointStore(run_id="schema_run", store_dir=tmp_path)
        store.mark_completed("q_dev_01", "S0_sentence_anchor", abstained=True, citation_count=0)

        ckpt_files = list(tmp_path.glob("*.json"))
        data = json.loads(ckpt_files[0].read_text())
        assert data["schema"] == "slice4_v2"
        assert data["run_id"] == "schema_run"

    def test_incompatible_schema_rejected(self, tmp_path):
        from raglab.infrastructure.persistence.generation_checkpoint_store import (
            GenerationCheckpointStore,
        )
        ckpt_file = tmp_path / "slice4_gen_checkpoint_incompat.json"
        ckpt_file.write_text(json.dumps({"schema": "slice4_v1", "run_id": "incompat", "completed": {}}))

        with pytest.raises(ValueError, match="INCOMPATIBLE_CHECKPOINT_SCHEMA"):
            GenerationCheckpointStore(run_id="incompat", store_dir=tmp_path)

    def test_abstained_recorded(self, tmp_path):
        from raglab.infrastructure.persistence.generation_checkpoint_store import (
            GenerationCheckpointStore,
        )
        store = GenerationCheckpointStore(run_id="abstain_run", store_dir=tmp_path)
        store.mark_completed("q_test_04", "F0_baseline", abstained=True, citation_count=0)
        pairs = store.completed_pairs()
        assert "q_test_04::F0_baseline" in pairs

    def test_completed_pairs_sorted(self, tmp_path):
        from raglab.infrastructure.persistence.generation_checkpoint_store import (
            GenerationCheckpointStore,
        )
        store = GenerationCheckpointStore(run_id="sorted_run", store_dir=tmp_path)
        store.mark_completed("q_dev_02", "F0_baseline", abstained=False, citation_count=1)
        store.mark_completed("q_dev_01", "F0_baseline", abstained=False, citation_count=2)
        pairs = store.completed_pairs()
        assert pairs == sorted(pairs)
