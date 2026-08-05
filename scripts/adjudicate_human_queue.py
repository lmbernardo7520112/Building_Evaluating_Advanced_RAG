# ruff: noqa: E501, E741, W291
"""Offline Localhost HTTP Interface for Blinded Human Adjudication (Gate B).

CLI:
  python scripts/adjudicate_human_queue.py \\
    --adjudicator-id ADJUDICATOR_ID \\
    --queue-file PATH \\
    --questions-file PATH \\
    --output-file PATH \\
    [--port PORT] \\
    [--no-browser]

Bound strictly to localhost 127.0.0.1 using Python standard library http.server.
Enforces mandatory reasoning for all items, exact literal span validation for grade > 0,
empty span for grade 0, confirmation before editing completed items, and atomic persistence.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import tempfile
import urllib.parse
import webbrowser
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Final

PROTOCOL_VERSION: Final[str] = "raglab_v7_slice4_v3"
SCHEMA_VERSION: Final[str] = "3.0.0"
HOLDOUT_QIDS: Final[frozenset[str]] = frozenset({"q_holdout_01", "q_holdout_02"})

VALID_GRADES: Final[frozenset[int]] = frozenset({0, 1, 2, 3})
VALID_ROLES: Final[frozenset[str]] = frozenset(
    {"NEGATIVE_CONTROL", "CONTEXTUAL", "SUPPORTING", "PRIMARY"}
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write_jsonl(target_path: Path, items: list[dict[str, Any]]) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path_str = tempfile.mkstemp(
        dir=target_path.parent, prefix=f".tmp_{target_path.name}_"
    )
    tmp_path = Path(tmp_path_str)

    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, target_path)

        with contextlib.suppress(OSError):
            parent_fd = os.open(str(target_path.parent), os.O_RDONLY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
    except Exception:
        if tmp_path.exists():
            with contextlib.suppress(OSError):
                tmp_path.unlink()
        raise


class AdjudicationSession:
    """Manages adjudication queue, progress state, and atomic saving."""

    def __init__(
        self,
        adjudicator_id: str,
        queue_path: Path,
        questions_path: Path,
        work_path: Path,
    ) -> None:
        self.adjudicator_id = adjudicator_id
        self.queue_path = queue_path
        self.questions_path = questions_path
        self.work_path = work_path

        self._load_questions()
        self._load_and_initialize_queue()

    def _load_questions(self) -> None:
        data = json.loads(self.questions_path.read_text(encoding="utf-8"))
        self.questions_dict: dict[str, dict[str, Any]] = {}
        for q in data.get("questions", []):
            qid = q.get("qid") or q.get("question_id")
            if qid:
                self.questions_dict[qid] = q

    def _load_and_initialize_queue(self) -> None:
        lines = [
            line
            for line in self.queue_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not lines:
            raise ValueError(f"Queue file is empty: {self.queue_path}")

        self.queue_items: list[dict[str, Any]] = []
        for line in lines:
            rec = json.loads(line)
            qid = str(rec.get("question_id", "")).strip()

            if qid in HOLDOUT_QIDS or "holdout" in qid.lower():
                raise ValueError(
                    f"HOLDOUT VIOLATION: item '{qid}' found in adjudication queue"
                )

            self.queue_items.append(rec)

        # Resume state if work_path exists
        self.work_map: dict[tuple[str, str], dict[str, Any]] = {}
        if self.work_path.exists():
            w_lines = [
                l
                for l in self.work_path.read_text(encoding="utf-8").splitlines()
                if l.strip()
            ]
            for l in w_lines:
                w_rec = json.loads(l)
                pair = (w_rec["question_id"], w_rec["passage_id"])
                self.work_map[pair] = w_rec

    def get_progress(self) -> dict[str, Any]:
        total = len(self.queue_items)
        completed = 0
        for item in self.queue_items:
            pair = (item["question_id"], item["passage_id"])
            w_rec = self.work_map.get(pair)
            if w_rec and w_rec.get("status") == "COMPLETED":
                completed += 1
        return {
            "adjudicator_id": self.adjudicator_id,
            "total_items": total,
            "completed_items": completed,
            "percent_complete": round((completed / total) * 100, 1)
            if total > 0
            else 100.0,
        }

    def get_item(self, index: int) -> dict[str, Any] | None:
        if index < 0 or index >= len(self.queue_items):
            return None

        raw_item = self.queue_items[index]
        pair = (raw_item["question_id"], raw_item["passage_id"])
        qid = raw_item["question_id"]
        q_obj = self.questions_dict.get(qid, {})

        # Merge with existing work if present
        current_work = self.work_map.get(pair, {})

        merged = {
            "index": index,
            "total": len(self.queue_items),
            "question_id": qid,
            "question_text": q_obj.get("question")
            or q_obj.get("text")
            or raw_item.get("question_text", ""),
            "is_abstention_question": q_obj.get("is_abstention") is True
            or q_obj.get("question_type") == "abstention",
            "passage_id": raw_item["passage_id"],
            "page_number": raw_item.get("page_number", 0),
            "passage_text": raw_item.get("passage_text") or raw_item.get("text", ""),
            "adjudication_reasons": raw_item.get("adjudication_reasons", []),
            "reviewer_1_grade": raw_item.get("reviewer_1_grade"),
            "reviewer_2_grade": raw_item.get("reviewer_2_grade"),
            "reviewer_1_role": raw_item.get("reviewer_1_role"),
            "reviewer_2_role": raw_item.get("reviewer_2_role"),
            "adjudicated_grade": current_work.get("adjudicated_grade"),
            "adjudicated_role": current_work.get("adjudicated_role"),
            "adjudication_reasoning": current_work.get("adjudication_reasoning", ""),
            "supporting_span_human": current_work.get("supporting_span_human", ""),
            "status": current_work.get("status", "PENDING"),
        }

        return merged

    def save_adjudication(
        self,
        index: int,
        grade: int,
        role: str,
        reasoning: str,
        span: str,
        force_edit: bool = False,
    ) -> dict[str, Any]:
        item = self.get_item(index)
        if item is None:
            raise ValueError(f"Item index {index} out of bounds")

        pair = (item["question_id"], item["passage_id"])
        existing = self.work_map.get(pair)

        if existing and existing.get("status") == "COMPLETED" and not force_edit:
            raise PermissionError("ITEM_ALREADY_COMPLETED_NEEDS_CONFIRMATION")

        # Validate inputs
        if grade not in VALID_GRADES:
            raise ValueError(f"Invalid grade '{grade}'. Must be in 0, 1, 2, 3")

        role_clean = role.strip().upper()
        if role_clean not in VALID_ROLES:
            raise ValueError(
                f"Invalid role '{role_clean}'. Must be in {sorted(VALID_ROLES)}"
            )

        reasoning_clean = reasoning.strip()
        if not reasoning_clean:
            raise ValueError("Adjudication reasoning is mandatory and cannot be empty")

        span_clean = span.strip()
        passage_text = item["passage_text"]

        if grade > 0:
            if not span_clean:
                raise ValueError(
                    f"Literal supporting span is mandatory when grade > 0 (grade={grade})"
                )
            if span_clean not in passage_text:
                raise ValueError(
                    "Literal supporting span is not an exact substring of the passage text"
                )
        else:
            if span_clean:
                raise ValueError("Supporting span must be empty when grade == 0")

        # Construct saved record
        rec = {
            "schema_version": SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "adjudicator_id": self.adjudicator_id,
            "question_id": item["question_id"],
            "passage_id": item["passage_id"],
            "page_number": item["page_number"],
            "reviewer_1_grade": item["reviewer_1_grade"],
            "reviewer_2_grade": item["reviewer_2_grade"],
            "reviewer_1_role": item["reviewer_1_role"],
            "reviewer_2_role": item["reviewer_2_role"],
            "adjudicated_grade": grade,
            "adjudicated_role": role_clean,
            "adjudication_reasoning": reasoning_clean,
            "supporting_span_human": span_clean,
            "status": "COMPLETED",
            "updated_at_utc": datetime.now(UTC).isoformat(),
        }

        self.work_map[pair] = rec

        # Atomic persistence of full work file
        ordered_records: list[dict[str, Any]] = []
        for q_item in self.queue_items:
            q_pair = (q_item["question_id"], q_item["passage_id"])
            if q_pair in self.work_map:
                ordered_records.append(self.work_map[q_pair])

        atomic_write_jsonl(self.work_path, ordered_records)
        return rec


class AdjudicationHTTPHandler(BaseHTTPRequestHandler):
    """HTTP Handler serving HTML SPA and REST API endpoints."""

    session: AdjudicationSession

    def _send_json(self, data: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str, status: int = HTTPStatus.OK) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self._send_html(HTML_INTERFACE)
            return

        if path == "/api/progress":
            self._send_json(self.session.get_progress())
            return

        if path.startswith("/api/item/"):
            with contextlib.suppress(ValueError):
                idx = int(path.split("/")[-1])
                item = self.session.get_item(idx)
                if item is not None:
                    self._send_json(item)
                    return
            self._send_json({"error": "Item not found"}, status=HTTPStatus.NOT_FOUND)
            return

        self._send_json({"error": "Not Found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/save":
            content_len = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_len)
            try:
                payload = json.loads(body_bytes.decode("utf-8"))
                res = self.session.save_adjudication(
                    index=int(payload["index"]),
                    grade=int(payload["grade"]),
                    role=str(payload["role"]),
                    reasoning=str(payload.get("reasoning", "")),
                    span=str(payload.get("span", "")),
                    force_edit=bool(payload.get("force_edit", False)),
                )
                self._send_json({"status": "SUCCESS", "record": res})
            except PermissionError as exc:
                self._send_json(
                    {"error": str(exc), "code": "NEEDS_CONFIRMATION"},
                    status=HTTPStatus.CONFLICT,
                )
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        self._send_json({"error": "Not Found"}, status=HTTPStatus.NOT_FOUND)


HTML_INTERFACE: Final[str] = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAGLab v7 - Adjudicação Humana Cegada (Gate B)</title>
    <style>
        :root {
            --bg: #0f172a;
            --card-bg: #1e293b;
            --text: #f8fafc;
            --muted: #94a3b8;
            --accent: #38bdf8;
            --accent-hover: #0284c7;
            --border: #334155;
            --success: #22c55e;
            --warning: #f59e0b;
            --danger: #ef4444;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background: var(--bg); color: var(--text); padding: 20px; line-height: 1.5; }
        .container { max-width: 1000px; margin: 0 auto; }
        header { display: flex; justify-content: space-between; align-items: center; padding-bottom: 20px; border-bottom: 1px solid var(--border); margin-bottom: 20px; }
        h1 { font-size: 1.5rem; color: var(--accent); }
        .progress-bar-bg { background: var(--border); border-radius: 999px; height: 10px; width: 200px; overflow: hidden; }
        .progress-bar-fill { background: var(--accent); height: 100%; transition: width 0.3s; }
        .card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-bottom: 20px; }
        .badge { display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: bold; text-transform: uppercase; margin-margin-right: 8px; margin-bottom: 8px; }
        .badge-disagreement { background: rgba(245, 158, 11, 0.2); color: var(--warning); border: 1px solid var(--warning); }
        .badge-abstention { background: rgba(56, 189, 248, 0.2); color: var(--accent); border: 1px solid var(--accent); }
        .badge-completed { background: rgba(34, 197, 94, 0.2); color: var(--success); border: 1px solid var(--success); }
        .reviewers-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 15px 0; background: rgba(0,0,0,0.2); padding: 15px; border-radius: 6px; }
        .reviewer-box { border-left: 3px solid var(--accent); padding-left: 10px; }
        .passage-box { background: #0f172a; border: 1px solid var(--border); padding: 15px; border-radius: 6px; font-family: monospace; white-space: pre-wrap; word-break: break-word; max-height: 250px; overflow-y: auto; margin: 10px 0; user-select: text; }
        .guidance-box { background: rgba(245, 158, 11, 0.1); border-left: 4px solid var(--warning); padding: 12px; font-size: 0.9rem; margin-bottom: 15px; }
        form { display: grid; gap: 15px; }
        label { font-weight: bold; font-size: 0.9rem; color: var(--muted); }
        select, textarea, input[type="text"] { background: #0f172a; border: 1px solid var(--border); color: var(--text); padding: 10px; border-radius: 6px; font-size: 1rem; width: 100%; }
        textarea { resize: vertical; min-height: 80px; }
        .btn-group { display: flex; justify-content: space-between; gap: 10px; margin-top: 10px; }
        button { background: var(--accent); color: #000; font-weight: bold; border: none; padding: 12px 20px; border-radius: 6px; cursor: pointer; transition: background 0.2s; }
        button:hover { background: var(--accent-hover); color: #fff; }
        button.btn-secondary { background: var(--border); color: var(--text); }
        button.btn-secondary:hover { background: #475569; }
        .error-msg { background: rgba(239, 68, 68, 0.2); border: 1px solid var(--danger); color: var(--danger); padding: 10px; border-radius: 6px; display: none; margin-bottom: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>Gate B - Adjudicação Humana</h1>
                <div id="session-info" style="color: var(--muted); font-size: 0.85rem;">Carregando...</div>
            </div>
            <div>
                <div style="font-size: 0.85rem; color: var(--muted); text-align: right; margin-bottom: 4px;" id="progress-text">0 / 0 (0%)</div>
                <div class="progress-bar-bg"><div class="progress-bar-fill" id="progress-bar" style="width: 0%;"></div></div>
            </div>
        </header>

        <div id="error-banner" class="error-msg"></div>

        <div class="card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div id="badges-container"></div>
                <div style="color: var(--muted); font-size: 0.85rem;" id="item-counter">Item 0 de 0</div>
            </div>

            <div style="margin-top: 10px;">
                <label>PERGUNTA (<span id="qid-text"></span>)</label>
                <div id="question-text" style="font-size: 1.1rem; font-weight: 500; margin-top: 4px; color: #fff;"></div>
            </div>

            <div id="abstention-guidance" class="guidance-box" style="display: none;">
                <strong>⚠️ Auditoria Estrutural de Abstenção:</strong><br>
                <em>"Esta passagem contém informação suficiente ou materialmente útil para responder a esta pergunta específica? Similaridade temática genérica não é relevância."</em>
            </div>

            <div class="reviewers-grid">
                <div class="reviewer-box">
                    <div style="font-size: 0.8rem; color: var(--muted);">REVISOR 1 (Anonimizado)</div>
                    <div>Grau: <strong id="rev1-grade">-</strong> | Papel: <span id="rev1-role">-</span></div>
                </div>
                <div class="reviewer-box">
                    <div style="font-size: 0.8rem; color: var(--muted);">REVISOR 2 (Anonimizado)</div>
                    <div>Grau: <strong id="rev2-grade">-</strong> | Papel: <span id="rev2-role">-</span></div>
                </div>
            </div>

            <div>
                <div style="display: flex; justify-content: space-between;">
                    <label>PASSAGEM DO DOCUMENTO (Página <span id="page-num"></span>)</label>
                    <button type="button" class="btn-secondary" style="padding: 2px 8px; font-size: 0.75rem;" onclick="copySelectedSpan()">Copiar Trecho Selecionado</button>
                </div>
                <div class="passage-box" id="passage-text"></div>
            </div>
        </div>

        <div class="card">
            <h2 style="font-size: 1.1rem; margin-bottom: 15px;">Decisão do Adjudicador</h2>
            <form id="adj-form" onsubmit="submitForm(event)">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                    <div>
                        <label for="grade-select">GRAU ADJUDICADO (0 a 3) *</label>
                        <select id="grade-select" required onchange="onGradeChange()">
                            <option value="">-- Selecione o Grau --</option>
                            <option value="0">0 - Irrelevante / Ruído</option>
                            <option value="1">1 - Relevância Fraca / Contextual</option>
                            <option value="2">2 - Relevante / Suporte Parcial</option>
                            <option value="3">3 - Altamente Relevante / Suporte Direto</option>
                        </select>
                    </div>
                    <div>
                        <label for="role-select">PAPEL ADJUDICADO *</label>
                        <select id="role-select" required>
                            <option value="NEGATIVE_CONTROL">NEGATIVE_CONTROL (Grau 0)</option>
                            <option value="CONTEXTUAL">CONTEXTUAL (Grau 1)</option>
                            <option value="SUPPORTING">SUPPORTING (Grau 2)</option>
                            <option value="PRIMARY">PRIMARY (Grau 3)</option>
                        </select>
                    </div>
                </div>

                <div>
                    <label for="reasoning-input">JUSTIFICATIVA DA ADJUDICAÇÃO * (Obrigatório)</label>
                    <textarea id="reasoning-input" required placeholder="Explique os motivos objetivos da sua decisão final de grau e papel..."></textarea>
                </div>

                <div>
                    <label for="span-input">TRECHO SUPORTE LITERAL (Obrigatório se Grau > 0; Vazio se Grau = 0)</label>
                    <input type="text" id="span-input" placeholder="Selecione o texto na caixa acima e clique em 'Copiar Trecho Selecionado'">
                </div>

                <div class="btn-group">
                    <button type="button" class="btn-secondary" onclick="navItem(-1)">← Anterior</button>
                    <button type="submit" id="save-btn">Salvar Adjudicação</button>
                    <button type="button" class="btn-secondary" onclick="navItem(1)">Próximo →</button>
                </div>
            </form>
        </div>
    </div>

    <script>
        let currentIndex = 0;
        let currentItem = null;
        let totalItems = 0;

        async function loadProgress() {
            const res = await fetch('/api/progress');
            const data = await res.json();
            totalItems = data.total_items;
            document.getElementById('session-info').innerText = `Adjudicador: ${data.adjudicator_id}`;
            document.getElementById('progress-text').innerText = `${data.completed_items} / ${totalItems} (${data.percent_complete}%)`;
            document.getElementById('progress-bar').style.width = `${data.percent_complete}%`;
        }

        async function loadItem(index) {
            hideError();
            const res = await fetch(`/api/item/${index}`);
            if (!res.ok) return;
            currentItem = await res.json();
            currentIndex = index;

            document.getElementById('item-counter').innerText = `Item ${index + 1} de ${totalItems}`;
            document.getElementById('qid-text').innerText = currentItem.question_id;
            document.getElementById('question-text').innerText = currentItem.question_text;
            document.getElementById('page-num').innerText = currentItem.page_number;
            document.getElementById('passage-text').innerText = currentItem.passage_text;

            document.getElementById('rev1-grade').innerText = currentItem.reviewer_1_grade;
            document.getElementById('rev1-role').innerText = currentItem.reviewer_1_role;
            document.getElementById('rev2-grade').innerText = currentItem.reviewer_2_grade;
            document.getElementById('rev2-role').innerText = currentItem.reviewer_2_role;

            const badges = document.getElementById('badges-container');
            badges.innerHTML = '';
            (currentItem.adjudication_reasons || []).forEach(r => {
                const b = document.createElement('span');
                b.className = r === 'disagreement' ? 'badge badge-disagreement' : 'badge badge-abstention';
                b.innerText = r === 'disagreement' ? 'Divergência de Grau' : 'Auditoria de Abstenção';
                badges.appendChild(b);
            });

            if (currentItem.status === 'COMPLETED') {
                const b = document.createElement('span');
                b.className = 'badge badge-completed';
                b.innerText = 'Concluído';
                badges.appendChild(b);
            }

            document.getElementById('abstention-guidance').style.display = 
                (currentItem.adjudication_reasons || []).includes('structural_abstention_audit') ? 'block' : 'none';

            // Form inputs
            document.getElementById('grade-select').value = currentItem.adjudicated_grade !== null ? currentItem.adjudicated_grade : '';
            document.getElementById('role-select').value = currentItem.adjudicated_role || 'NEGATIVE_CONTROL';
            document.getElementById('reasoning-input').value = currentItem.adjudication_reasoning || '';
            document.getElementById('span-input').value = currentItem.supporting_span_human || '';

            onGradeChange();
            await loadProgress();
        }

        function onGradeChange() {
            const gradeVal = document.getElementById('grade-select').value;
            const roleSelect = document.getElementById('role-select');
            const spanInput = document.getElementById('span-input');

            if (gradeVal === '0') {
                roleSelect.value = 'NEGATIVE_CONTROL';
                spanInput.value = '';
                spanInput.disabled = true;
            } else if (gradeVal === '1') {
                if (roleSelect.value === 'NEGATIVE_CONTROL') roleSelect.value = 'CONTEXTUAL';
                spanInput.disabled = false;
            } else if (gradeVal === '2') {
                if (roleSelect.value === 'NEGATIVE_CONTROL' || roleSelect.value === 'CONTEXTUAL') roleSelect.value = 'SUPPORTING';
                spanInput.disabled = false;
            } else if (gradeVal === '3') {
                roleSelect.value = 'PRIMARY';
                spanInput.disabled = false;
            }
        }

        function copySelectedSpan() {
            const sel = window.getSelection().toString().trim();
            if (sel) {
                document.getElementById('span-input').value = sel;
            }
        }

        function showError(msg) {
            const b = document.getElementById('error-banner');
            b.innerText = msg;
            b.style.display = 'block';
        }

        function hideError() {
            document.getElementById('error-banner').style.display = 'none';
        }

        async function submitForm(e, forceEdit = false) {
            e.preventDefault();
            hideError();

            const gradeVal = document.getElementById('grade-select').value;
            if (gradeVal === '') {
                showError('Por favor, selecione um Grau.');
                return;
            }

            const payload = {
                index: currentIndex,
                grade: parseInt(gradeVal, 10),
                role: document.getElementById('role-select').value,
                reasoning: document.getElementById('reasoning-input').value,
                span: document.getElementById('span-input').value,
                force_edit: forceEdit
            };

            const res = await fetch('/api/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await res.json();
            if (!res.ok) {
                if (data.code === 'NEEDS_CONFIRMATION') {
                    if (confirm('Este item já foi concluído anteriormente. Deseja sobrescrever a decisão?')) {
                        submitForm(e, true);
                    }
                } else {
                    showError(data.error || 'Erro ao salvar adjudicação.');
                }
                return;
            }

            await loadItem(currentIndex < totalItems - 1 ? currentIndex + 1 : currentIndex);
        }

        function navItem(dir) {
            const target = currentIndex + dir;
            if (target >= 0 && target < totalItems) {
                loadItem(target);
            }
        }

        window.onload = async () => {
            await loadProgress();
            await loadItem(0);
        };
    </script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Offline HTTP Interface for Blinded Human Adjudication (Gate B)"
    )
    parser.add_argument(
        "--adjudicator-id", type=str, required=True, help="ID of the human adjudicator"
    )
    parser.add_argument(
        "--queue-file",
        type=Path,
        required=True,
        help="Path to adjudication queue JSONL",
    )
    parser.add_argument(
        "--questions-file",
        type=Path,
        required=True,
        help="Path to controlled questions JSON",
    )
    parser.add_argument(
        "--output-file", type=Path, required=True, help="Path to output work file"
    )
    parser.add_argument(
        "--port", type=int, default=8503, help="Port to bind server (default 8503)"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open web browser",
    )
    args = parser.parse_args()

    try:
        session = AdjudicationSession(
            adjudicator_id=args.adjudicator_id,
            queue_path=args.queue_file,
            questions_path=args.questions_file,
            work_path=args.output_file,
        )

        AdjudicationHTTPHandler.session = session

        server = HTTPServer(("127.0.0.1", args.port), AdjudicationHTTPHandler)
        url = f"http://127.0.0.1:{args.port}"

        print("==================================================================")
        print(" RAGLAB v7 - INTERFACE LOCAL OFFLINE DE ADJUDICAÇÃO HUMANA (GATE B)")
        print("==================================================================")
        print(f" Servidor ativo em: {url}")
        print(f" Adjudicador ID:   {args.adjudicator_id}")
        print(f" Arquivo Fila:     {args.queue_file}")
        print(f" Arquivo Trabalho: {args.output_file}")
        print(" Pressione Ctrl+C para encerrar o servidor.")
        print("==================================================================")

        if not args.no_browser:
            webbrowser.open(url)

        server.serve_forever()
        return 0
    except KeyboardInterrupt:
        print("\nServidor encerrado com sucesso.")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
