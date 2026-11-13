"""SQLite WAL persistence engine for Tapes, Steps, MCP Spans, and Biopsy Reports."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from forktape.models.tape import (
    AnomalyFlag,
    BiopsyReport,
    MCPSpan,
    OverridePatch,
    StepTokenBreakdown,
    TapeBundle,
    TapeRecord,
    TapeStatus,
    TapeStep,
    ToolCallInvocation,
)

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA foreign_keys=ON;
PRAGMA busy_timeout=15000;

CREATE TABLE IF NOT EXISTS tapes (
    tape_id TEXT PRIMARY KEY,
    parent_tape_id TEXT,
    fork_from_step INTEGER,
    fork_label TEXT,
    title TEXT NOT NULL,
    agent_command TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'running',
    failure_reason TEXT,
    step_count INTEGER NOT NULL DEFAULT 0,
    total_prompt_tokens INTEGER NOT NULL DEFAULT 0,
    total_cached_tokens INTEGER NOT NULL DEFAULT 0,
    total_completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_cost_usd REAL NOT NULL DEFAULT 0.0,
    total_latency_ms REAL NOT NULL DEFAULT 0.0,
    tags_json TEXT NOT NULL DEFAULT '[]',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tapes_parent ON tapes(parent_tape_id);
CREATE INDEX IF NOT EXISTS idx_tapes_created ON tapes(created_at DESC);

CREATE TABLE IF NOT EXISTS steps (
    step_id TEXT PRIMARY KEY,
    tape_id TEXT NOT NULL REFERENCES tapes(tape_id) ON DELETE CASCADE,
    step_index INTEGER NOT NULL,
    provider TEXT NOT NULL DEFAULT 'openai',
    model TEXT NOT NULL DEFAULT 'gpt-4o',
    canonical_hash TEXT NOT NULL DEFAULT '',
    token_breakdown_json TEXT NOT NULL DEFAULT '{}',
    prefix_cache_ratio REAL NOT NULL DEFAULT 1.0,
    prefix_break_offset INTEGER,
    prefix_break_snippet TEXT,
    raw_request_json TEXT NOT NULL DEFAULT '{}',
    raw_response_json TEXT NOT NULL DEFAULT '{}',
    assembled_text TEXT NOT NULL DEFAULT '',
    reasoning_text TEXT NOT NULL DEFAULT '',
    tool_calls_json TEXT NOT NULL DEFAULT '[]',
    override_patch_json TEXT,
    replayed INTEGER NOT NULL DEFAULT 0,
    ttft_ms REAL NOT NULL DEFAULT 0.0,
    duration_ms REAL NOT NULL DEFAULT 0.0,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    cached_prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0.0,
    anomaly_flags_json TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    UNIQUE(tape_id, step_index)
);

CREATE INDEX IF NOT EXISTS idx_steps_tape ON steps(tape_id, step_index);
CREATE INDEX IF NOT EXISTS idx_steps_hash ON steps(canonical_hash);

CREATE TABLE IF NOT EXISTS mcp_spans (
    span_id TEXT PRIMARY KEY,
    tape_id TEXT NOT NULL REFERENCES tapes(tape_id) ON DELETE CASCADE,
    step_index INTEGER NOT NULL DEFAULT 1,
    server_name TEXT NOT NULL DEFAULT 'default',
    transport TEXT NOT NULL DEFAULT 'stdio',
    method TEXT NOT NULL DEFAULT 'tools/call',
    tool_name TEXT NOT NULL DEFAULT '',
    arguments_json TEXT NOT NULL DEFAULT '{}',
    result_json TEXT,
    is_error INTEGER NOT NULL DEFAULT 0,
    replayed INTEGER NOT NULL DEFAULT 0,
    duration_ms REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_mcp_tape ON mcp_spans(tape_id, step_index);

CREATE TABLE IF NOT EXISTS biopsy_reports (
    report_id TEXT PRIMARY KEY,
    tape_id TEXT NOT NULL UNIQUE REFERENCES tapes(tape_id) ON DELETE CASCADE,
    report_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


class TapeStore:
    """Thread-safe SQLite WAL persistence layer for ForkTape."""

    def __init__(self, db_path: str | Path = ".forktape/tapes.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA_SQL)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> TapeStore:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Tapes CRUD
    # ------------------------------------------------------------------
    def create_tape(self, tape: TapeRecord) -> TapeRecord:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO tapes (
                    tape_id, parent_tape_id, fork_from_step, fork_label,
                    title, agent_command, status, failure_reason,
                    step_count, total_prompt_tokens, total_cached_tokens,
                    total_completion_tokens, total_cost_usd, total_latency_ms,
                    tags_json, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tape.tape_id,
                    tape.parent_tape_id,
                    tape.fork_from_step,
                    tape.fork_label,
                    tape.title,
                    tape.agent_command,
                    tape.status.value,
                    tape.failure_reason,
                    tape.step_count,
                    tape.total_prompt_tokens,
                    tape.total_cached_tokens,
                    tape.total_completion_tokens,
                    tape.total_cost_usd,
                    tape.total_latency_ms,
                    json.dumps(tape.tags, ensure_ascii=False),
                    json.dumps(tape.metadata, ensure_ascii=False),
                    tape.created_at,
                ),
            )
            self._conn.commit()
        return tape

    def update_tape_status(
        self,
        tape_id: str,
        status: TapeStatus,
        failure_reason: str | None = None,
    ) -> None:
        with self._lock:
            if status == TapeStatus.SUCCESS and failure_reason is None:
                self._conn.execute(
                    "UPDATE tapes SET status = ?, failure_reason = NULL WHERE tape_id = ?",
                    (status.value, tape_id),
                )
            else:
                self._conn.execute(
                    "UPDATE tapes SET status = ?, failure_reason = COALESCE(?, failure_reason) WHERE tape_id = ?",
                    (status.value, failure_reason, tape_id),
                )
            self._conn.commit()

    def refresh_tape_Rollups(self, tape_id: str) -> TapeRecord | None:
        """Recompute aggregate token, cost, latency, and step counts from steps."""
        with self._lock:
            row = self._conn.execute(
                """
                SELECT
                    COUNT(*) AS cnt,
                    COALESCE(SUM(prompt_tokens), 0) AS p_tok,
                    COALESCE(SUM(cached_prompt_tokens), 0) AS c_tok,
                    COALESCE(SUM(completion_tokens), 0) AS o_tok,
                    COALESCE(SUM(cost_usd), 0.0) AS cost,
                    COALESCE(SUM(duration_ms), 0.0) AS lat
                FROM steps
                WHERE tape_id = ?
                """,
                (tape_id,),
            ).fetchone()
            if row is not None:
                self._conn.execute(
                    """
                    UPDATE tapes
                    SET step_count = ?,
                        total_prompt_tokens = ?,
                        total_cached_tokens = ?,
                        total_completion_tokens = ?,
                        total_cost_usd = ?,
                        total_latency_ms = ?
                    WHERE tape_id = ?
                    """,
                    (
                        int(row["cnt"]),
                        int(row["p_tok"]),
                        int(row["c_tok"]),
                        int(row["o_tok"]),
                        round(float(row["cost"]), 6),
                        round(float(row["lat"]), 2),
                        tape_id,
                    ),
                )
                self._conn.commit()
        return self.get_tape(tape_id)

    def get_tape(self, tape_id: str) -> TapeRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM tapes WHERE tape_id = ?", (tape_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_tape(row)

    def list_tapes(
        self,
        status: str | None = None,
        limit: int = 100,
    ) -> list[TapeRecord]:
        with self._lock:
            if status:
                rows = self._conn.execute(
                    "SELECT * FROM tapes WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM tapes ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [self._row_to_tape(r) for r in rows]

    def get_children_tapes(self, parent_tape_id: str) -> list[TapeRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM tapes WHERE parent_tape_id = ? ORDER BY created_at ASC",
                (parent_tape_id,),
            ).fetchall()
        return [self._row_to_tape(r) for r in rows]

    def delete_tape(self, tape_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM tapes WHERE tape_id = ?", (tape_id,))
            self._conn.commit()
            return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Steps CRUD
    # ------------------------------------------------------------------
    def append_step(self, step: TapeStep) -> TapeStep:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO steps (
                    step_id, tape_id, step_index, provider, model, canonical_hash,
                    token_breakdown_json, prefix_cache_ratio, prefix_break_offset,
                    prefix_break_snippet, raw_request_json, raw_response_json,
                    assembled_text, reasoning_text, tool_calls_json,
                    override_patch_json, replayed, ttft_ms, duration_ms,
                    prompt_tokens, cached_prompt_tokens, completion_tokens,
                    cost_usd, anomaly_flags_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    step.step_id,
                    step.tape_id,
                    step.step_index,
                    step.provider,
                    step.model,
                    step.canonical_hash,
                    step.token_breakdown.model_dump_json(),
                    step.prefix_cache_ratio,
                    step.prefix_break_offset,
                    step.prefix_break_snippet,
                    json.dumps(step.raw_request, ensure_ascii=False),
                    json.dumps(step.raw_response, ensure_ascii=False),
                    step.assembled_text,
                    step.reasoning_text,
                    json.dumps([tc.model_dump() for tc in step.tool_calls], ensure_ascii=False),
                    step.override_patch.model_dump_json() if step.override_patch else None,
                    1 if step.replayed else 0,
                    step.ttft_ms,
                    step.duration_ms,
                    step.prompt_tokens,
                    step.cached_prompt_tokens,
                    step.completion_tokens,
                    step.cost_usd,
                    json.dumps([f.model_dump() for f in step.anomaly_flags], ensure_ascii=False),
                    step.created_at,
                ),
            )
            self._conn.commit()
        self.refresh_tape_Rollups(step.tape_id)
        return step

    def update_step_anomalies(self, tape_id: str, step_index: int, flags: list[AnomalyFlag]) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE steps SET anomaly_flags_json = ? WHERE tape_id = ? AND step_index = ?",
                (json.dumps([f.model_dump() for f in flags], ensure_ascii=False), tape_id, step_index),
            )
            self._conn.commit()

    def get_steps(self, tape_id: str) -> list[TapeStep]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM steps WHERE tape_id = ? ORDER BY step_index ASC",
                (tape_id,),
            ).fetchall()
        return [self._row_to_step(r) for r in rows]

    def get_step(self, tape_id: str, step_index: int) -> TapeStep | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM steps WHERE tape_id = ? AND step_index = ?",
                (tape_id, step_index),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_step(row)

    def get_max_step_index(self, tape_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(step_index), 0) AS max_idx FROM steps WHERE tape_id = ?",
                (tape_id,),
            ).fetchone()
        return int(row["max_idx"]) if row is not None else 0

    # ------------------------------------------------------------------
    # MCP Spans CRUD
    # ------------------------------------------------------------------
    def append_mcp_span(self, span: MCPSpan) -> MCPSpan:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO mcp_spans (
                    span_id, tape_id, step_index, server_name, transport,
                    method, tool_name, arguments_json, result_json,
                    is_error, replayed, duration_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    span.span_id,
                    span.tape_id,
                    span.step_index,
                    span.server_name,
                    span.transport,
                    span.method,
                    span.tool_name,
                    json.dumps(span.arguments, ensure_ascii=False),
                    json.dumps(span.result, ensure_ascii=False),
                    1 if span.is_error else 0,
                    1 if span.replayed else 0,
                    span.duration_ms,
                    span.created_at,
                ),
            )
            self._conn.commit()
        return span

    def get_mcp_spans(self, tape_id: str, step_index: int | None = None) -> list[MCPSpan]:
        with self._lock:
            if step_index is not None:
                rows = self._conn.execute(
                    "SELECT * FROM mcp_spans WHERE tape_id = ? AND step_index = ? ORDER BY created_at ASC",
                    (tape_id, step_index),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM mcp_spans WHERE tape_id = ? ORDER BY step_index ASC, created_at ASC",
                    (tape_id,),
                ).fetchall()
        return [self._row_to_mcp_span(r) for r in rows]

    # ------------------------------------------------------------------
    # Biopsy Reports CRUD
    # ------------------------------------------------------------------
    def save_biopsy_report(self, report: BiopsyReport) -> BiopsyReport:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO biopsy_reports (report_id, tape_id, report_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    report.report_id,
                    report.tape_id,
                    report.model_dump_json(),
                    report.created_at,
                ),
            )
            self._conn.commit()
        return report

    def get_biopsy_report(self, tape_id: str) -> BiopsyReport | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT report_json FROM biopsy_reports WHERE tape_id = ?",
                (tape_id,),
            ).fetchone()
        if row is None:
            return None
        return BiopsyReport.model_validate_json(row["report_json"])

    # ------------------------------------------------------------------
    # Portable Bundle Export / Import (.ftk.json)
    # ------------------------------------------------------------------
    def export_bundle(self, tape_id: str) -> TapeBundle:
        tape = self.get_tape(tape_id)
        if tape is None:
            raise KeyError(f"Tape '{tape_id}' not found")
        return TapeBundle(
            tape=tape,
            steps=self.get_steps(tape_id),
            mcp_spans=self.get_mcp_spans(tape_id),
            biopsy_report=self.get_biopsy_report(tape_id),
        )

    def export_bundle_file(self, tape_id: str, output_path: str | Path) -> Path:
        bundle = self.export_bundle(tape_id)
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
        return out

    def import_bundle(self, bundle: TapeBundle) -> TapeRecord:
        self.create_tape(bundle.tape)
        for step in bundle.steps:
            self.append_step(step)
        for span in bundle.mcp_spans:
            self.append_mcp_span(span)
        if bundle.biopsy_report is not None:
            self.save_biopsy_report(bundle.biopsy_report)
        refreshed = self.refresh_tape_Rollups(bundle.tape.tape_id)
        return refreshed or bundle.tape

    def import_bundle_file(self, bundle_path: str | Path) -> TapeRecord:
        raw = Path(bundle_path).read_text(encoding="utf-8")
        bundle = TapeBundle.model_validate_json(raw)
        return self.import_bundle(bundle)

    # ------------------------------------------------------------------
    # Row Deserializers
    # ------------------------------------------------------------------
    @staticmethod
    def _row_to_tape(row: sqlite3.Row) -> TapeRecord:
        return TapeRecord(
            tape_id=row["tape_id"],
            parent_tape_id=row["parent_tape_id"],
            fork_from_step=row["fork_from_step"],
            fork_label=row["fork_label"],
            title=row["title"],
            agent_command=row["agent_command"],
            status=TapeStatus(row["status"]),
            failure_reason=row["failure_reason"],
            step_count=int(row["step_count"]),
            total_prompt_tokens=int(row["total_prompt_tokens"]),
            total_cached_tokens=int(row["total_cached_tokens"]),
            total_completion_tokens=int(row["total_completion_tokens"]),
            total_cost_usd=float(row["total_cost_usd"]),
            total_latency_ms=float(row["total_latency_ms"]),
            tags=json.loads(row["tags_json"] or "[]"),
            metadata=json.loads(row["metadata_json"] or "{}"),
            created_at=row["created_at"],
        )

    @staticmethod
    def _row_to_step(row: sqlite3.Row) -> TapeStep:
        tb_raw = json.loads(row["token_breakdown_json"] or "{}")
        tc_raw: list[dict[str, Any]] = json.loads(row["tool_calls_json"] or "[]")
        af_raw: list[dict[str, Any]] = json.loads(row["anomaly_flags_json"] or "[]")
        patch_json = row["override_patch_json"]
        return TapeStep(
            step_id=row["step_id"],
            tape_id=row["tape_id"],
            step_index=int(row["step_index"]),
            provider=row["provider"],
            model=row["model"],
            canonical_hash=row["canonical_hash"],
            token_breakdown=StepTokenBreakdown.model_validate(tb_raw),
            prefix_cache_ratio=float(row["prefix_cache_ratio"]),
            prefix_break_offset=row["prefix_break_offset"],
            prefix_break_snippet=row["prefix_break_snippet"],
            raw_request=json.loads(row["raw_request_json"] or "{}"),
            raw_response=json.loads(row["raw_response_json"] or "{}"),
            assembled_text=row["assembled_text"],
            reasoning_text=row["reasoning_text"],
            tool_calls=[ToolCallInvocation.model_validate(x) for x in tc_raw],
            override_patch=OverridePatch.model_validate_json(patch_json) if patch_json else None,
            replayed=bool(row["replayed"]),
            ttft_ms=float(row["ttft_ms"]),
            duration_ms=float(row["duration_ms"]),
            prompt_tokens=int(row["prompt_tokens"]),
            cached_prompt_tokens=int(row["cached_prompt_tokens"]),
            completion_tokens=int(row["completion_tokens"]),
            cost_usd=float(row["cost_usd"]),
            anomaly_flags=[AnomalyFlag.model_validate(x) for x in af_raw],
            created_at=row["created_at"],
        )

    @staticmethod
    def _row_to_mcp_span(row: sqlite3.Row) -> MCPSpan:
        return MCPSpan(
            span_id=row["span_id"],
            tape_id=row["tape_id"],
            step_index=int(row["step_index"]),
            server_name=row["server_name"],
            transport=row["transport"],
            method=row["method"],
            tool_name=row["tool_name"],
            arguments=json.loads(row["arguments_json"] or "{}"),
            result=json.loads(row["result_json"]) if row["result_json"] is not None else None,
            is_error=bool(row["is_error"]),
            replayed=bool(row["replayed"]),
            duration_ms=float(row["duration_ms"]),
            created_at=row["created_at"],
        )
