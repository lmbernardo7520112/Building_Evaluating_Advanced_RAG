# ruff: noqa: E501
"""Local Offline Human Annotation Interface for Gate B (RAGLab v7).
Provides a secure, offline, local-only web interface (stdlib-only) for
Annotators A and B to evaluate passages without editing JSONL manually.

SECURITY & GOVERNANCE:
- Runs strictly on 127.0.0.1 (localhost loopback). Non-loopback binding is rejected.
- Zero external dependencies — uses Python standard library http.server.
- Zero network calls, zero APIs, zero LLMs, zero telemetry, zero external assets/CDNs.
- Strict blinding: silver predictions, ranks, scores, gold answers, and holdout are hidden.
- Atomic persistence with fsync + replace to prevent data corruption.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import os
import secrets
import sys
import tempfile
import urllib.parse
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Final

_REPO_ROOT = Path(__file__).resolve().parents[1]

PROTOCOL_VERSION: Final[str] = "raglab_v7_slice4_v3"
SCHEMA_VERSION: Final[str] = "3.0.0"
HOLDOUT_QIDS: Final[frozenset[str]] = frozenset({"q_holdout_01", "q_holdout_02"})

# Blinding forbidden fields in queue items
BLINDING_FORBIDDEN_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "confidence",
        "reasoning",
        "supporting_span",
        "judge_model",
        "judge_provider",
        "judge_id",
        "label_source",
        "strategy",
        "retrieval_rank",
        "retrieval_score",
        "pre_rerank_rank",
        "post_rerank_rank",
        "selected_by_reranker",
        "dropped_by_reranker",
        "relevant_pages",
        "gold_answer",
    }
)

VALID_GRADES: Final[frozenset[int]] = frozenset({0, 1, 2, 3})
VALID_ROLES: Final[frozenset[str]] = frozenset(
    {"NEGATIVE_CONTROL", "CONTEXTUAL", "SUPPORTING", "PRIMARY"}
)

DEFAULT_ROLE_FOR_GRADE: Final[dict[int, str]] = {
    0: "NEGATIVE_CONTROL",
    1: "CONTEXTUAL",
    2: "SUPPORTING",
    3: "PRIMARY",
}

logger = logging.getLogger("annotate_human_queue")


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write_jsonl(target_path: Path, records: list[dict[str, Any]]) -> None:
    """Write records to target_path atomically using tmp file + fsync + replace."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path_str = tempfile.mkstemp(
        dir=target_path.parent, prefix=f".tmp_{target_path.name}_"
    )
    tmp_path = Path(tmp_path_str)

    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, target_path)

        # Sync parent directory if supported
        try:
            parent_fd = os.open(str(target_path.parent), os.O_RDONLY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        except OSError:
            pass  # Parent directory fsync not supported on all OS/filesystems
    except Exception:
        if tmp_path.exists():
            with contextlib.suppress(OSError):
                tmp_path.unlink()
        raise


def load_questions_file(questions_file: Path) -> dict[str, str]:
    """Load authoritative questions file and return mapping of question_id -> question_text."""
    if not questions_file.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_file}")

    content = questions_file.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Questions file is empty: {questions_file}")

    raw_items: list[dict[str, Any]] = []

    try:
        data = json.loads(content)
        if isinstance(data, dict) and "questions" in data:
            raw_items = data["questions"]
        elif isinstance(data, list):
            raw_items = data
        else:
            raise ValueError(
                "Root object must contain 'questions' list or be a list of objects"
            )
    except json.JSONDecodeError:
        # Fallback to JSONL
        for line_num, line in enumerate(content.splitlines(), 1):
            if not line.strip():
                continue
            try:
                raw_items.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON/JSONL in questions file at line {line_num}: {exc}"
                ) from exc

    questions_map: dict[str, str] = {}
    seen_qids: set[str] = set()

    for item in raw_items:
        if not isinstance(item, dict):
            raise ValueError("Question record must be a JSON object")

        qid = str(item.get("qid") or item.get("question_id") or "").strip()
        qtext = str(item.get("question") or item.get("query") or "").strip()

        if not qid:
            raise ValueError("Question record missing 'qid' or 'question_id'")
        if qid in seen_qids:
            raise ValueError(f"Duplicate question_id '{qid}' in questions file")
        seen_qids.add(qid)

        if not qtext:
            raise ValueError(f"Question text is empty for question_id '{qid}'")

        questions_map[qid] = qtext

    return questions_map


def load_and_validate_queue(
    queue_file: Path,
    annotator_id: str,
    questions_map: dict[str, str],
) -> list[dict[str, Any]]:
    """Load and validate human review queue file fail-closed."""
    if not queue_file.exists():
        raise FileNotFoundError(f"Queue file not found: {queue_file}")

    lines = queue_file.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"Queue file is empty: {queue_file}")

    queue_items: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for line_num, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in queue file at line {line_num}: {exc}"
            ) from exc

        if not isinstance(item, dict):
            raise ValueError(f"Queue line {line_num} must be a JSON object")

        qid = str(item.get("question_id", "")).strip()
        ps_id = str(item.get("passage_id", "")).strip()

        if not qid or not ps_id:
            raise ValueError(f"Queue line {line_num} missing question_id or passage_id")

        if qid in HOLDOUT_QIDS or "holdout" in qid.lower():
            raise ValueError(f"HOLDOUT VIOLATION: item '{qid}' in queue file")

        pair = (qid, ps_id)
        if pair in seen_pairs:
            raise ValueError(f"Duplicate pair {pair} in queue file")
        seen_pairs.add(pair)

        # Check identity compatibility
        item_ann = item.get("annotator_id")
        if item_ann and item_ann != annotator_id:
            raise ValueError(
                f"Identity mismatch: queue item specifies annotator_id '{item_ann}', "
                f"but session started with '{annotator_id}'"
            )

        # Verify blinding fields absence
        for field in BLINDING_FORBIDDEN_FIELDS:
            if field in item:
                raise ValueError(
                    f"BLINDING VIOLATION: Forbidden field '{field}' present in queue item {pair}"
                )

        # Check question_id exists in questions_map
        if qid not in questions_map:
            raise ValueError(
                f"Question ID '{qid}' from queue file not found in questions file"
            )

        queue_items.append(item)

    # Check filename vs annotator_id compatibility
    if annotator_id == "annotator_a" and "annotator_b" in queue_file.name:
        raise ValueError(
            "Incompatible CLI arguments: annotator_a specified with annotator_b queue file"
        )
    if annotator_id == "annotator_b" and "annotator_a" in queue_file.name:
        raise ValueError(
            "Incompatible CLI arguments: annotator_b specified with annotator_a queue file"
        )

    return queue_items


def validate_annotation_payload(
    relevance_grade: Any,
    evidence_role: Any,
    supporting_span_human: str,
    annotation_notes: str,
    passage_text: str,
) -> tuple[int, str, str, str]:
    """Validate annotation payload server-side fail-closed."""
    try:
        grade = int(relevance_grade)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Invalid relevance_grade '{relevance_grade}'. Must be integer."
        ) from exc

    if grade not in VALID_GRADES:
        raise ValueError(f"relevance_grade {grade} invalid. Must be in {{0, 1, 2, 3}}.")

    role = str(evidence_role or "").strip().upper()
    if role not in VALID_ROLES:
        raise ValueError(
            f"evidence_role '{role}' invalid. Must be one of {sorted(VALID_ROLES)}."
        )

    # Role / Grade compatibility check
    expected_role = DEFAULT_ROLE_FOR_GRADE[grade]
    notes = annotation_notes.strip()

    if role != expected_role and not notes:
        raise ValueError(
            f"Non-standard evidence_role '{role}' for relevance_grade {grade} "
            f"(expected '{expected_role}'). Non-empty justification in annotation_notes is required."
        )

    span = supporting_span_human.strip()
    if span and span not in passage_text:
        raise ValueError(
            "supporting_span_human is not an exact literal substring of passage_text"
        )

    return grade, role, span, notes


class AnnotationSession:
    """Manages state, atomic persistence, and resume for an active annotation session."""

    def __init__(
        self,
        annotator_id: str,
        queue_file: Path,
        questions_file: Path,
        output_file: Path,
        manifest_file: Path | None = None,
    ) -> None:
        self.annotator_id = annotator_id
        self.queue_file = queue_file
        self.questions_file = questions_file
        self.output_file = output_file
        self.manifest_file = manifest_file

        self.queue_sha = sha256_file(queue_file)
        self.questions_sha = sha256_file(questions_file)
        self.manifest_sha = (
            sha256_file(manifest_file)
            if manifest_file and manifest_file.exists()
            else ""
        )

        self.questions_map = load_questions_file(questions_file)
        self.queue_items = load_and_validate_queue(
            queue_file, annotator_id, self.questions_map
        )

        self.annotated_records: dict[tuple[str, str], dict[str, Any]] = {}
        self.unlocked_keys: set[tuple[str, str]] = set()

        self._load_existing_output()

    def _load_existing_output(self) -> None:
        """Load and validate existing output file if present."""
        if not self.output_file.exists():
            return

        lines = self.output_file.read_text(encoding="utf-8").splitlines()
        for line_num, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Corrupted existing output file at line {line_num}: {exc}"
                ) from exc

            rec_ann = rec.get("annotator_id")
            if rec_ann and rec_ann != self.annotator_id:
                raise ValueError(
                    f"Existing output file belongs to annotator '{rec_ann}', "
                    f"cannot resume with '{self.annotator_id}'"
                )

            rec_q_sha = rec.get("queue_file_sha256")
            if rec_q_sha and rec_q_sha != self.queue_sha:
                raise ValueError(
                    "Existing output file was generated from a different queue file version"
                )

            rec_qs_sha = rec.get("questions_file_sha256")
            if rec_qs_sha and rec_qs_sha != self.questions_sha:
                raise ValueError(
                    "Existing output file was generated from a different questions file version"
                )

            qid = str(rec.get("question_id", ""))
            ps_id = str(rec.get("passage_id", ""))
            if qid and ps_id:
                self.annotated_records[(qid, ps_id)] = rec

    def get_progress(self) -> dict[str, Any]:
        total = len(self.queue_items)
        completed = sum(
            1
            for rec in self.annotated_records.values()
            if rec.get("status") == "COMPLETED"
        )
        return {
            "annotator_id": self.annotator_id,
            "total_items": total,
            "completed_items": completed,
            "remaining_items": total - completed,
            "percent": round((completed / total) * 100, 1) if total > 0 else 100.0,
        }

    def get_item_data(self, index: int) -> dict[str, Any]:
        if not (0 <= index < len(self.queue_items)):
            raise IndexError(
                f"Item index {index} out of range [0, {len(self.queue_items) - 1}]"
            )

        item = self.queue_items[index]
        qid = item["question_id"]
        ps_id = item["passage_id"]
        pair = (qid, ps_id)

        existing = self.annotated_records.get(pair, {})
        is_completed = existing.get("status") == "COMPLETED"
        is_unlocked = pair in self.unlocked_keys

        return {
            "index": index,
            "total": len(self.queue_items),
            "question_id": qid,
            "question_text": self.questions_map[qid],
            "passage_id": ps_id,
            "page_number": item["page_number"],
            "passage_text": item["text"],
            "existing_annotation": {
                "relevance_grade": existing.get("relevance_grade"),
                "evidence_role": existing.get("evidence_role"),
                "supporting_span_human": existing.get("supporting_span_human", ""),
                "annotation_notes": existing.get("annotation_notes", ""),
                "status": existing.get("status", "PENDING"),
            },
            "is_completed": is_completed,
            "is_unlocked": is_unlocked,
            "read_only": is_completed and not is_unlocked,
        }

    def unlock_edit(self, index: int) -> None:
        if not (0 <= index < len(self.queue_items)):
            raise IndexError(f"Index {index} out of range")
        item = self.queue_items[index]
        pair = (item["question_id"], item["passage_id"])
        self.unlocked_keys.add(pair)

    def save_annotation(
        self,
        index: int,
        relevance_grade: Any,
        evidence_role: Any,
        supporting_span_human: str,
        annotation_notes: str,
    ) -> dict[str, Any]:
        if not (0 <= index < len(self.queue_items)):
            raise IndexError(f"Index {index} out of range")

        item = self.queue_items[index]
        qid = item["question_id"]
        ps_id = item["passage_id"]
        pair = (qid, ps_id)

        existing = self.annotated_records.get(pair, {})
        if existing.get("status") == "COMPLETED" and pair not in self.unlocked_keys:
            raise ValueError(
                "Item is marked COMPLETED and locked. Confirm edit before modifying."
            )

        grade, role, span, notes = validate_annotation_payload(
            relevance_grade=relevance_grade,
            evidence_role=evidence_role,
            supporting_span_human=supporting_span_human,
            annotation_notes=annotation_notes,
            passage_text=item["text"],
        )

        now_utc = datetime.now(UTC).isoformat()

        rec = {
            "schema_version": SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "annotator_id": self.annotator_id,
            "queue_file_sha256": self.queue_sha,
            "questions_file_sha256": self.questions_sha,
            "routing_manifest_sha256": self.manifest_sha,
            "question_id": qid,
            "passage_id": ps_id,
            "page_number": item["page_number"],
            "relevance_grade": grade,
            "evidence_role": role,
            "supporting_span_human": span,
            "annotation_notes": notes,
            "annotated_at_utc": now_utc,
            "status": "COMPLETED",
        }

        # Update in-memory dict
        self.annotated_records[pair] = rec

        # Atomic write of all current records preserving queue order
        records_to_write: list[dict[str, Any]] = []
        for q_item in self.queue_items:
            q_pair = (q_item["question_id"], q_item["passage_id"])
            if q_pair in self.annotated_records:
                records_to_write.append(self.annotated_records[q_pair])

        atomic_write_jsonl(self.output_file, records_to_write)
        self.unlocked_keys.discard(pair)

        return rec


# ─────────────────────────────────────────────────────────────────
# HTML / CSS / JS Template (Embedded, Escaped, Self-Contained)
# ─────────────────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>  <!-- noqa: E501 -->
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAGLab v7 — Anotação Humana (Gate B)</title>
    <style>
        :root {{
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --text-color: #f8fafc;
            --text-muted: #94a3b8;
            --accent-color: #38bdf8;
            --accent-hover: #0284c7;
            --border-color: #334155;
            --success-color: #22c55e;
            --warning-color: #eab308;
            --danger-color: #ef4444;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
            padding: 20px;
        }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 24px;
        }}
        .title-area h1 {{ font-size: 1.5rem; color: var(--accent-color); }}
        .title-area span {{ font-size: 0.875rem; color: var(--text-muted); }}
        .progress-bar-container {{
            width: 100%;
            background-color: var(--border-color);
            border-radius: 9999px;
            height: 10px;
            overflow: hidden;
            margin: 12px 0 24px 0;
        }}
        .progress-bar {{
            height: 100%;
            background-color: var(--accent-color);
            width: 0%;
            transition: width 0.3s ease;
        }}
        .card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        .card-header {{
            font-weight: 600;
            color: var(--accent-color);
            margin-bottom: 8px;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .content-box {{
            background-color: #0f172a;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 16px;
            font-size: 1rem;
            white-space: pre-wrap;
            word-break: break-word;
            max-height: 350px;
            overflow-y: auto;
        }}
        .form-group {{ margin-bottom: 16px; }}
        label {{ display: block; font-weight: 600; margin-bottom: 6px; font-size: 0.9rem; }}
        select, input[type="text"], textarea {{
            width: 100%;
            padding: 10px;
            background-color: #0f172a;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            color: var(--text-color);
            font-size: 0.95rem;
        }}
        select:focus, input:focus, textarea:focus {{
            outline: none;
            border-color: var(--accent-color);
        }}
        .nav-buttons {{
            display: flex;
            justify-content: space-between;
            margin-top: 24px;
        }}
        button {{
            padding: 10px 20px;
            border-radius: 6px;
            border: none;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s;
        }}
        .btn-primary {{ background-color: var(--accent-color); color: #0f172a; }}
        .btn-primary:hover {{ background-color: var(--accent-hover); }}
        .btn-secondary {{ background-color: var(--border-color); color: var(--text-color); }}
        .btn-warning {{ background-color: var(--warning-color); color: #0f172a; }}
        .status-badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: bold;
        }}
        .status-completed {{ background-color: var(--success-color); color: #0f172a; }}
        .status-pending {{ background-color: var(--warning-color); color: #0f172a; }}
        .alert {{
            padding: 12px;
            border-radius: 6px;
            margin-bottom: 16px;
            display: none;
        }}
        .alert-error {{ background-color: rgba(239, 68, 68, 0.2); border: 1px solid var(--danger-color); color: #fca5a5; }}
        .alert-success {{ background-color: rgba(34, 197, 94, 0.2); border: 1px solid var(--success-color); color: #86efac; }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="title-area">
                <h1>RAGLab v7 — Painel de Anotação Humana (Gate B)</h1>
                <span>Anotador: <strong id="lbl-annotator">--</strong></span>
            </div>
            <div>
                <span id="lbl-status-badge" class="status-badge status-pending">PENDING</span>
            </div>
        </header>

        <div id="alert-box" class="alert"></div>

        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span>Progresso: <strong id="lbl-progress-text">0 / 0 (0%)</strong></span>
            <span>Item <strong id="lbl-current-index">0</strong> de <strong id="lbl-total-count">0</strong></span>
        </div>
        <div class="progress-bar-container">
            <div id="progress-bar" class="progress-bar"></div>
        </div>

        <div class="card">
            <div class="card-header">PERGUNTA (qid: <span id="lbl-qid">--</span>)</div>
            <div id="txt-query" class="content-box">Carregando pergunta...</div>
        </div>

        <div class="card">
            <div class="card-header">EVIDÊNCIA DOCUMENTAL (passage_id: <span id="lbl-psid">--</span> | pág: <span id="lbl-page">--</span>)</div>
            <div id="txt-passage" class="content-box">Carregando passagem...</div>
        </div>

        <form id="form-annotation" class="card">
            <div class="form-group">
                <label for="sel-grade">Grau de Relevância (0 a 3) *</label>
                <select id="sel-grade" required>
                    <option value="">-- Selecione o grau --</option>
                    <option value="0">0 — Irrelevante (Negative Control)</option>
                    <option value="1">1 — Informação contextual ou marginal</option>
                    <option value="2">2 — Evidência útil, mas insuficiente isoladamente</option>
                    <option value="3">3 — Evidência diretamente suficiente / central</option>
                </select>
            </div>

            <div class="form-group">
                <label for="sel-role">Papel da Evidência (evidence_role) *</label>
                <select id="sel-role" required>
                    <option value="NEGATIVE_CONTROL">NEGATIVE_CONTROL (Padrão para Grau 0)</option>
                    <option value="CONTEXTUAL">CONTEXTUAL (Padrão para Grau 1)</option>
                    <option value="SUPPORTING">SUPPORTING (Padrão para Grau 2)</option>
                    <option value="PRIMARY">PRIMARY (Padrão para Grau 3)</option>
                </select>
            </div>

            <div class="form-group">
                <label for="txt-span">Trecho Literal de Suporte (supporting_span_human) — Opcional</label>
                <input type="text" id="txt-span" placeholder="Cole aqui o trecho literal exato extraído do texto da passagem acima...">
            </div>

            <div class="form-group">
                <label for="txt-notes">Notas / Justificativa (annotation_notes) *</label>
                <textarea id="txt-notes" rows="3" placeholder="Obrigatório caso o Papel divirja do padrão do Grau..."></textarea>
            </div>

            <div style="display: flex; gap: 12px; align-items: center;">
                <button type="submit" id="btn-save" class="btn-primary">Salvar e Avançar</button>
                <button type="button" id="btn-unlock" class="btn-warning" style="display: none;">Confirmar Desbloqueio para Edição</button>
            </div>
        </form>

        <div class="nav-buttons">
            <button id="btn-prev" class="btn-secondary">&larr; Anterior</button>
            <button id="btn-next" class="btn-secondary">Próximo &rarr;</button>
        </div>
    </div>

    <script>
        const SESSION_TOKEN = "{token}";
        let currentIndex = 0;
        let totalCount = 0;
        let isReadOnly = false;
        let rawPassageText = "";

        const defaultRoles = {{ 0: "NEGATIVE_CONTROL", 1: "CONTEXTUAL", 2: "SUPPORTING", 3: "PRIMARY" }};

        function showAlert(msg, isError = false) {{
            const box = document.getElementById("alert-box");
            box.innerText = msg;
            box.className = "alert " + (isError ? "alert-error" : "alert-success");
            box.style.display = "block";
            setTimeout(() => {{ box.style.display = "none"; }}, 4000);
        }}

        async function fetchState() {{
            const res = await fetch("/api/state");
            const data = await res.json();
            document.getElementById("lbl-annotator").innerText = data.annotator_id;
            totalCount = data.total_items;
            document.getElementById("lbl-total-count").innerText = totalCount;
            document.getElementById("lbl-progress-text").innerText = `${{data.completed_items}} / ${{data.total_items}} (${{data.percent}}%)`;
            document.getElementById("progress-bar").style.width = `${{data.percent}}%`;
            return data;
        }}

        async function loadItem(index) {{
            if (index < 0 || index >= totalCount) return;
            currentIndex = index;
            document.getElementById("lbl-current-index").innerText = currentIndex + 1;

            const res = await fetch(`/api/item?index=${{index}}`);
            if (!res.ok) {{
                showAlert("Erro ao carregar item.", true);
                return;
            }}
            const data = await res.json();
            rawPassageText = data.passage_text;

            document.getElementById("lbl-qid").innerText = data.question_id;
            document.getElementById("txt-query").innerText = data.question_text;
            document.getElementById("lbl-psid").innerText = data.passage_id;
            document.getElementById("lbl-page").innerText = data.page_number;
            document.getElementById("txt-passage").innerText = data.passage_text;

            const existing = data.existing_annotation;
            const badge = document.getElementById("lbl-status-badge");
            badge.innerText = existing.status;
            badge.className = "status-badge " + (existing.status === "COMPLETED" ? "status-completed" : "status-pending");

            document.getElementById("sel-grade").value = existing.relevance_grade !== null ? existing.relevance_grade : "";
            document.getElementById("sel-role").value = existing.evidence_role || "NEGATIVE_CONTROL";
            document.getElementById("txt-span").value = existing.supporting_span_human || "";
            document.getElementById("txt-notes").value = existing.annotation_notes || "";

            isReadOnly = data.read_only;
            setFormState(isReadOnly);
        }}

        function setFormState(readOnly) {{
            document.getElementById("sel-grade").disabled = readOnly;
            document.getElementById("sel-role").disabled = readOnly;
            document.getElementById("txt-span").disabled = readOnly;
            document.getElementById("txt-notes").disabled = readOnly;
            document.getElementById("btn-save").style.display = readOnly ? "none" : "inline-block";
            document.getElementById("btn-unlock").style.display = readOnly ? "inline-block" : "none";
        }}

        document.getElementById("sel-grade").addEventListener("change", (e) => {{
            const g = parseInt(e.target.value);
            if (!isNaN(g) && defaultRoles[g]) {{
                document.getElementById("sel-role").value = defaultRoles[g];
            }}
        }});

        document.getElementById("btn-unlock").addEventListener("click", async () => {{
            if (confirm("Este item já está CONCLUÍDO. Deseja realmente destravar para edição?")) {{
                const res = await fetch("/api/confirm_edit", {{
                    method: "POST",
                    headers: {{
                        "Content-Type": "application/json",
                        "X-Session-Token": SESSION_TOKEN
                    }},
                    body: JSON.stringify({{ index: currentIndex }})
                }});
                if (res.ok) {{
                    setFormState(false);
                    showAlert("Item destravado para edição.");
                }}
            }}
        }});

        document.getElementById("form-annotation").addEventListener("submit", async (e) => {{
            e.preventDefault();
            const grade = document.getElementById("sel-grade").value;
            const role = document.getElementById("sel-role").value;
            const span = document.getElementById("txt-span").value.trim();
            const notes = document.getElementById("txt-notes").value.trim();

            if (grade === "") {{
                showAlert("Selecione o Grau de Relevância.", true);
                return;
            }}

            if (span && !rawPassageText.includes(span)) {{
                showAlert("Erro: O Trecho Literal digitado não existe exatamente no texto da passagem!", true);
                return;
            }}

            const payload = {{
                index: currentIndex,
                relevance_grade: parseInt(grade),
                evidence_role: role,
                supporting_span_human: span,
                annotation_notes: notes
            }};

            const res = await fetch("/api/annotate", {{
                method: "POST",
                headers: {{
                    "Content-Type": "application/json",
                    "X-Session-Token": SESSION_TOKEN
                }},
                body: JSON.stringify(payload)
            }});

            const data = await res.json();
            if (!res.ok) {{
                showAlert(data.error || "Erro ao salvar anotação.", true);
                return;
            }}

            showAlert("Anotação salva com sucesso!");
            await fetchState();
            if (currentIndex < totalCount - 1) {{
                loadItem(currentIndex + 1);
            }} else {{
                loadItem(currentIndex);
            }}
        }});

        document.getElementById("btn-prev").addEventListener("click", () => {{
            if (currentIndex > 0) loadItem(currentIndex - 1);
        }});

        document.getElementById("btn-next").addEventListener("click", () => {{
            if (currentIndex < totalCount - 1) loadItem(currentIndex + 1);
        }});

        // Init
        (async () => {{
            const state = await fetchState();
            // Find first pending item or start at 0
            let firstPending = 0;
            for (let i = 0; i < state.total_items; i++) {{
                const itemRes = await fetch(`/api/item?index=${{i}}`);
                const itemData = await itemRes.json();
                if (itemData.existing_annotation.status !== "COMPLETED") {{
                    firstPending = i;
                    break;
                }}
            }}
            loadItem(firstPending);
        }})();
    </script>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────────
# HTTP Request Handler (Localhost Loopback Server Only)
# ─────────────────────────────────────────────────────────────────


class AnnotationHTTPRequestHandler(BaseHTTPRequestHandler):
    """Custom HTTP Request Handler enforcing localhost security and explicit routes."""

    session: AnnotationSession
    session_token: str

    def log_message(self, format_str: str, *args: Any) -> None:
        """Sanitize HTTP access logs — never print full annotation text or tokens."""
        clean_args = [str(a)[:50] for a in args]
        logger.info(
            "%s - - [%s] %s",
            self.address_string(),
            self.log_date_time_string(),
            format_str % tuple(clean_args),
        )

    def _send_security_headers(
        self, status_code: int = 200, content_type: str = "application/json"
    ) -> None:
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'",
        )
        self.end_headers()

    def _validate_host_and_origin(self) -> bool:
        host = self.headers.get("Host", "")
        if not (host.startswith("127.0.0.1:") or host.startswith("localhost:")):
            self._send_security_headers(403, "application/json")
            self.wfile.write(
                json.dumps({"error": f"Forbidden Host header '{host}'"}).encode("utf-8")
            )
            return False

        if self.command == "POST":
            origin = self.headers.get("Origin", "")
            if origin and not (
                origin.startswith("http://127.0.0.1:")
                or origin.startswith("http://localhost:")
            ):
                self._send_security_headers(403, "application/json")
                self.wfile.write(
                    json.dumps({"error": f"Forbidden Origin header '{origin}'"}).encode(
                        "utf-8"
                    )
                )
                return False
        return True

    def _validate_token(self) -> bool:
        token = self.headers.get("X-Session-Token")
        if not token or not secrets.compare_digest(token, self.session_token):
            self._send_security_headers(401, "application/json")
            self.wfile.write(
                json.dumps({"error": "Unauthorized: Invalid session token"}).encode(
                    "utf-8"
                )
            )
            return False
        return True

    def do_GET(self) -> None:
        if not self._validate_host_and_origin():
            return

        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query = urllib.parse.parse_qs(parsed_url.query)

        if path == "/":
            self._send_security_headers(200, "text/html; charset=utf-8")
            rendered_html = HTML_TEMPLATE.format(token=self.session_token)
            self.wfile.write(rendered_html.encode("utf-8"))
            return

        if path == "/api/state":
            self._send_security_headers(200, "application/json")
            data = self.session.get_progress()
            self.wfile.write(json.dumps(data).encode("utf-8"))
            return

        if path == "/api/item":
            index_str = query.get("index", ["0"])[0]
            try:
                index = int(index_str)
                data = self.session.get_item_data(index)
                self._send_security_headers(200, "application/json")
                self.wfile.write(json.dumps(data).encode("utf-8"))
            except (ValueError, IndexError) as exc:
                self._send_security_headers(400, "application/json")
                self.wfile.write(json.dumps({"error": str(exc)}).encode("utf-8"))
            return

        self._send_security_headers(404, "application/json")
        self.wfile.write(json.dumps({"error": "Not Found"}).encode("utf-8"))

    def do_POST(self) -> None:
        if not self._validate_host_and_origin() or not self._validate_token():
            return

        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path

        content_len_header = self.headers.get("Content-Length", "0")
        try:
            content_len = int(content_len_header)
        except ValueError:
            content_len = 0

        if content_len > 1_048_576:
            self._send_security_headers(413, "application/json")
            self.wfile.write(
                json.dumps({"error": "Payload Too Large (max 1MB)"}).encode("utf-8")
            )
            return

        body_bytes = self.rfile.read(content_len)
        try:
            payload = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
        except json.JSONDecodeError:
            self._send_security_headers(400, "application/json")
            self.wfile.write(json.dumps({"error": "Invalid JSON body"}).encode("utf-8"))
            return

        if path == "/api/annotate":
            try:
                index = int(payload.get("index", -1))
                rec = self.session.save_annotation(
                    index=index,
                    relevance_grade=payload.get("relevance_grade"),
                    evidence_role=payload.get("evidence_role"),
                    supporting_span_human=str(
                        payload.get("supporting_span_human") or ""
                    ),
                    annotation_notes=str(payload.get("annotation_notes") or ""),
                )
                self._send_security_headers(200, "application/json")
                self.wfile.write(
                    json.dumps({"status": "SUCCESS", "record": rec}).encode("utf-8")
                )
            except Exception as exc:
                self._send_security_headers(400, "application/json")
                self.wfile.write(json.dumps({"error": str(exc)}).encode("utf-8"))
            return

        if path == "/api/confirm_edit":
            try:
                index = int(payload.get("index", -1))
                self.session.unlock_edit(index)
                self._send_security_headers(200, "application/json")
                self.wfile.write(json.dumps({"status": "UNLOCKED"}).encode("utf-8"))
            except Exception as exc:
                self._send_security_headers(400, "application/json")
                self.wfile.write(json.dumps({"error": str(exc)}).encode("utf-8"))
            return

        self._send_security_headers(404, "application/json")
        self.wfile.write(json.dumps({"error": "Not Found"}).encode("utf-8"))

    def do_PUT(self) -> None:
        self._send_security_headers(405, "application/json")
        self.wfile.write(json.dumps({"error": "Method Not Allowed"}).encode("utf-8"))

    def do_DELETE(self) -> None:
        self._send_security_headers(405, "application/json")
        self.wfile.write(json.dumps({"error": "Method Not Allowed"}).encode("utf-8"))

    def do_PATCH(self) -> None:
        self._send_security_headers(405, "application/json")
        self.wfile.write(json.dumps({"error": "Method Not Allowed"}).encode("utf-8"))


def create_annotation_server(
    session: AnnotationSession,
    host: str = "127.0.0.1",
    port: int = 8501,
) -> tuple[HTTPServer, str]:
    if host not in ("127.0.0.1", "localhost"):
        msg = f"SECURITY VIOLATION: Refusing external host binding '{host}'."
        raise ValueError(msg)

    token = secrets.token_urlsafe(32)

    class CustomHandler(AnnotationHTTPRequestHandler):
        pass

    CustomHandler.session = session
    CustomHandler.session_token = token

    server = HTTPServer((host, port), CustomHandler)
    return server, token


def main() -> int:
    parser = argparse.ArgumentParser(description="Human Annotation Interface (Gate B)")
    parser.add_argument(
        "--annotator-id",
        required=True,
        choices=["annotator_a", "annotator_b"],
        help="Annotator identity",
    )
    parser.add_argument(
        "--queue-file",
        type=Path,
        required=True,
        help="Path to blinded queue file",
    )
    parser.add_argument(
        "--questions-file",
        type=Path,
        required=True,
        help="Path to authoritative questions file",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Path to output work file",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host binding",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port number",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open browser",
    )
    args = parser.parse_args()

    if args.host not in ("127.0.0.1", "localhost"):
        msg = f"ERROR: External host binding '{args.host}' is forbidden."
        print(msg, file=sys.stderr)
        return 1

    try:
        manifest_file = args.queue_file.parent / "routing_manifest.json"
        session = AnnotationSession(
            annotator_id=args.annotator_id,
            queue_file=args.queue_file,
            questions_file=args.questions_file,
            output_file=args.output_file,
            manifest_file=manifest_file if manifest_file.exists() else None,
        )

        server, token = create_annotation_server(
            session, host=args.host, port=args.port
        )
        url = f"http://{args.host}:{args.port}/?token={token}"

        print("=" * 65)
        print("OFFLINE HUMAN ANNOTATION INTERFACE (GATE B)")
        print("=" * 65)
        print(f"Annotator ID: {args.annotator_id}")
        print(f"Queue File:   {args.queue_file}")
        print(f"Questions:    {args.questions_file}")
        print(f"Output Work:  {args.output_file}")
        print(f"Server URL:   {url}")
        print("=" * 65)
        print("Press Ctrl+C to terminate session safely.")

        if not args.no_browser:
            import contextlib
            import webbrowser

            with contextlib.suppress(Exception):
                webbrowser.open(url)

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nSession terminated safely by user.")
        finally:
            server.server_close()

        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
