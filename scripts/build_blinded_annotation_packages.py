"""Blinded Annotation Package & Adjudication Template Builder (Gate B1).

Generates:
- benchmarks/ground_truth/v2/annotation_packages/annotator_a/{development,test}.jsonl
- benchmarks/ground_truth/v2/annotation_packages/annotator_b/{development,test}.jsonl
- benchmarks/ground_truth/v2/annotation_packages/package_manifest.json
- benchmarks/ground_truth/v2/adjudication_template.jsonl

Strictly blinded, offline, and reproducible. Holdout questions remain sealed!
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

# Ensure src and benchmarks are on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT / "benchmarks") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "benchmarks"))

from run_slice4_benchmark import ACTIVE_QUESTIONS  # noqa: E402

SCHEMA_VERSION = "2.0.0"
PACKAGE_VERSION = "2.0.0"


def load_passage_registry(registry_file: Path) -> list[dict[str, Any]]:
    """Load passage registry entries from JSONL."""
    if not registry_file.exists():
        raise FileNotFoundError(f"Passage registry not found at {registry_file}")

    entries = []
    with registry_file.open("r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                entries.append(json.loads(line_str))
    return entries


def build_candidate_pool_for_question(
    question: dict[str, Any],
    all_passages: list[dict[str, Any]],
    passages_by_page: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Assemble candidate pool combining relevant page passages & negative controls."""  # noqa: E501
    qid = question["qid"]
    rel_pages = set(question.get("relevant_pages", []))

    candidate_passages: list[dict[str, Any]] = []
    used_ids: set[str] = set()

    # 1. Add passages matching relevant pages
    for page_num in sorted(rel_pages):
        for ps in passages_by_page.get(page_num, []):
            ps_id = ps["passage_id"]
            if ps_id not in used_ids:
                used_ids.add(ps_id)
                candidate_passages.append(
                    {
                        "passage_id": ps_id,
                        "page_number": ps["page_number"],
                        "text": ps["text"],
                        "relevance_grade": None,
                        "evidence_role": None,
                        "annotation_notes": "",
                    }
                )

    # 2. Add negative control passages (deterministically selected)
    non_rel_pages = [p for p in passages_by_page if p not in rel_pages]
    neg_seed = int(
        hashlib.sha256(f"neg_control:{qid}".encode()).hexdigest()[:8], 16
    )
    rng = random.Random(neg_seed)  # noqa: S311

    available_negatives = []
    for p in non_rel_pages:
        available_negatives.extend(passages_by_page[p])

    if available_negatives:
        num_negs = max(2, min(5, len(available_negatives)))
        selected_negs = rng.sample(available_negatives, k=num_negs)
        for ps in selected_negs:
            ps_id = ps["passage_id"]
            if ps_id not in used_ids:
                used_ids.add(ps_id)
                candidate_passages.append(
                    {
                        "passage_id": ps_id,
                        "page_number": ps["page_number"],
                        "text": ps["text"],
                        "relevance_grade": None,
                        "evidence_role": None,
                        "annotation_notes": "",
                    }
                )

    # Deterministic shuffling seed for candidate order (blinded order)
    shuffle_seed = int(
        hashlib.sha256(f"blind_order:{qid}".encode()).hexdigest()[:8], 16
    )
    shuffle_rng = random.Random(shuffle_seed)  # noqa: S311
    shuffle_rng.shuffle(candidate_passages)

    return candidate_passages


def build_annotation_packages(
    registry_dir: Path,
    output_dir: Path,
    annotators: tuple[str, ...] = ("annotator_a", "annotator_b"),
) -> tuple[Path, Path]:
    """Build blinded annotation packages for Annotator A and B, plus adjudication template."""  # noqa: E501
    registry_file = registry_dir / "passage_registry.jsonl"
    registry_dir / "passage_registry_manifest.json"

    passages = load_passage_registry(registry_file)
    registry_sha256 = hashlib.sha256(registry_file.read_bytes()).hexdigest()

    # Index passages by page number
    passages_by_page: dict[int, list[dict[str, Any]]] = {}
    for ps in passages:
        p_num = ps["page_number"]
        passages_by_page.setdefault(p_num, []).append(ps)

    # Filter out sealed holdout questions strictly!
    active_non_holdout = [q for q in ACTIVE_QUESTIONS if "holdout" not in q["qid"]]

    packages_dir = output_dir / "annotation_packages"
    packages_dir.mkdir(parents=True, exist_ok=True)

    adjudication_items: list[dict[str, Any]] = []

    for annotator_id in annotators:
        ann_dir = packages_dir / annotator_id
        ann_dir.mkdir(parents=True, exist_ok=True)

        for split in ("development", "test"):
            split_questions = [
                q for q in active_non_holdout if q.get("split", "development") == split
            ]
            records: list[dict[str, Any]] = []

            for q in split_questions:
                qid = q["qid"]
                candidates = build_candidate_pool_for_question(
                    q, passages, passages_by_page
                )

                rec = {
                    "question_id": qid,
                    "question_text": q["query"],
                    "answerability": None,
                    "unanswerable_reason": None,
                    "candidate_passages": candidates,
                    "evidence_sets": [],
                    "gold_answer": None,
                    "gold_supporting_passage_ids": [],
                    "annotator_id": annotator_id,
                    "annotation_status": "PENDING",
                }
                records.append(rec)

                # Populate adjudication template items (once per candidate)
                if annotator_id == annotators[0]:
                    for cand in candidates:
                        adjudication_items.append(
                            {
                                "question_id": qid,
                                "passage_id": cand["passage_id"],
                                "annotator_a_grade": None,
                                "annotator_b_grade": None,
                                "adjudicated_grade": None,
                                "adjudication_reason": "",
                                "adjudicator_id": "",
                                "adjudication_status": "PENDING",
                            }
                        )

            jsonl_file = ann_dir / f"{split}.jsonl"
            lines = [json.dumps(r, ensure_ascii=False) for r in records]
            jsonl_file.write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )

    # Package Manifest
    pkg_manifest_file = packages_dir / "package_manifest.json"
    manifest_data = {
        "schema_version": SCHEMA_VERSION,
        "package_version": PACKAGE_VERSION,
        "annotators": list(annotators),
        "splits": ["development", "test"],
        "question_count": len(active_non_holdout),
        "passage_registry_sha256": registry_sha256,
        "candidate_sources": ["page_match_pool", "negative_controls"],
        "unavailable_sources": ["CANDIDATE_SOURCE_NOT_AVAILABLE_OFFLINE"],
        "holdout_sealed": True,
        "created_by": "deterministic_offline_builder",
        "network_used": False,
        "api_used": False,
    }
    pkg_manifest_file.write_text(
        json.dumps(manifest_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # Adjudication Template File
    adj_file = output_dir / "adjudication_template.jsonl"
    adj_lines = [json.dumps(it, ensure_ascii=False) for it in adjudication_items]
    adj_file.write_text(
        "\n".join(adj_lines) + ("\n" if adj_lines else ""), encoding="utf-8"
    )

    return pkg_manifest_file, adj_file


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build blinded annotation packages for Gate B1"
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=_REPO_ROOT / "benchmarks" / "ground_truth" / "v2",
        help="Root ground truth v2 directory",
    )
    args = parser.parse_args()

    pkg_man, adj_file = build_annotation_packages(
        registry_dir=args.root_dir,
        output_dir=args.root_dir,
    )
    print("BLINDED ANNOTATION PACKAGES BUILT SUCCESSFULLY")
    print(f"Package Manifest: {pkg_man}")
    print(f"Adjudication Template: {adj_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
