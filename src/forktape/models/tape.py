"""Pydantic v2 domain schemas for Tapes, Steps, MCP Spans, Overrides, and Biopsy Reports."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


class TapeStatus(StrEnum):
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    LOOP_ABORTED = "loop_aborted"
    DIVERGED = "diverged"


class AnomalyFlag(BaseModel):
    code: str
    severity: str = "high"  # "critical" | "high" | "medium" | "low"
    step_index: int
    title: str
    detail: str
    evidence: dict[str, Any] = Field(default_factory=dict)


class StepTokenBreakdown(BaseModel):
    system_tokens: int = 0
    tools_schema_tokens: int = 0
    history_tokens: int = 0
    tool_output_tokens: int = 0

    @property
    def total(self) -> int:
        return (
            self.system_tokens
            + self.tools_schema_tokens
            + self.history_tokens
            + self.tool_output_tokens
        )


class OverridePatch(BaseModel):
    """Counterfactual modification applied at or after the fork point."""

    label: str | None = None
    system_prompt_replace: str | None = None
    system_prompt_append: str | None = None
    swap_model: str | None = None
    prune_tools_top_k: int | None = None
    compact_context_max_chars: int | None = None
    stabilize_prefix_cache: bool = False
    override_temperature: float | None = None
    mock_tool_outputs: dict[str, Any] = Field(default_factory=dict)
    dry_run_write_tools: bool = False

    def is_empty(self) -> bool:
        return (
            self.system_prompt_replace is None
            and self.system_prompt_append is None
            and self.swap_model is None
            and self.prune_tools_top_k is None
            and self.compact_context_max_chars is None
            and not self.stabilize_prefix_cache
            and self.override_temperature is None
            and not self.mock_tool_outputs
            and not self.dry_run_write_tools
        )


class MCPSpan(BaseModel):
    span_id: str = Field(default_factory=lambda: new_id("mcp"))
    tape_id: str
    step_index: int = 1
    server_name: str = "default"
    transport: str = "stdio"  # "stdio" | "http"
    method: str = "tools/call"
    tool_name: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    is_error: bool = False
    replayed: bool = False
    duration_ms: float = 0.0
    created_at: str = Field(default_factory=utc_now_iso)


class ToolCallInvocation(BaseModel):
    id: str = ""
    name: str
    arguments: dict[str, Any] | str = Field(default_factory=dict)


class TapeStep(BaseModel):
    step_id: str = Field(default_factory=lambda: new_id("stp"))
    tape_id: str
    step_index: int
    provider: str = "openai"  # "openai" | "anthropic" | "mock"
    model: str = "gpt-4o"
    canonical_hash: str = ""
    token_breakdown: StepTokenBreakdown = Field(default_factory=StepTokenBreakdown)
    prefix_cache_ratio: float = 1.0
    prefix_break_offset: int | None = None
    prefix_break_snippet: str | None = None
    raw_request: dict[str, Any] = Field(default_factory=dict)
    raw_response: dict[str, Any] = Field(default_factory=dict)
    assembled_text: str = ""
    reasoning_text: str = ""
    tool_calls: list[ToolCallInvocation] = Field(default_factory=list)
    override_patch: OverridePatch | None = None
    replayed: bool = False
    ttft_ms: float = 0.0
    duration_ms: float = 0.0
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    anomaly_flags: list[AnomalyFlag] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now_iso)


class TapeRecord(BaseModel):
    tape_id: str = Field(default_factory=lambda: new_id("tape"))
    parent_tape_id: str | None = None
    fork_from_step: int | None = None
    fork_label: str | None = None
    title: str = "Untitled Agent Run"
    agent_command: str = ""
    status: TapeStatus = TapeStatus.RUNNING
    failure_reason: str | None = None
    step_count: int = 0
    total_prompt_tokens: int = 0
    total_cached_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def cache_hit_rate(self) -> float:
        if self.total_prompt_tokens <= 0:
            return 0.0
        return round(self.total_cached_tokens / self.total_prompt_tokens, 4)


class AblationRunResult(BaseModel):
    experiment_id: str = Field(default_factory=lambda: new_id("exp"))
    hypothesis_name: str
    operator: str  # "prune_tools_bm25" | "compact_context" | "stabilize_prefix_cache" | "zero_temp_probe"
    step_index: int
    forked_tape_id: str | None = None
    patch_applied: OverridePatch
    before_summary: str
    after_summary: str
    resolved_failure: bool = False
    prompt_tokens_before: int = 0
    prompt_tokens_after: int = 0
    token_reduction_pct: float = 0.0
    cache_ratio_before: float = 0.0
    cache_ratio_after: float = 0.0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    evidence_notes: list[str] = Field(default_factory=list)


class BiopsyReport(BaseModel):
    report_id: str = Field(default_factory=lambda: new_id("bio"))
    tape_id: str
    suspected_drift_step: int
    root_cause_category: str
    confidence: float = 0.85
    executive_summary: str
    heuristic_findings: list[AnomalyFlag] = Field(default_factory=list)
    ablation_runs: list[AblationRunResult] = Field(default_factory=list)
    verified_fix_recommendation: str = ""
    recommended_patch: OverridePatch | None = None
    generated_pytest_code: str = ""
    total_verification_cost_usd: float = 0.0
    total_verification_time_ms: float = 0.0
    created_at: str = Field(default_factory=utc_now_iso)


class StepDiffEntry(BaseModel):
    step_index: int
    relation: str  # "identical_replay" | "fork_point_modified" | "diverged_continuation" | "only_left" | "only_right"
    left_model: str | None = None
    right_model: str | None = None
    left_tools_called: list[str] = Field(default_factory=list)
    right_tools_called: list[str] = Field(default_factory=list)
    left_args_preview: str = ""
    right_args_preview: str = ""
    left_text_preview: str = ""
    right_text_preview: str = ""
    left_prompt_tokens: int = 0
    right_prompt_tokens: int = 0
    left_cache_ratio: float = 0.0
    right_cache_ratio: float = 0.0
    left_duration_ms: float = 0.0
    right_duration_ms: float = 0.0


class BranchComparison(BaseModel):
    left_tape: TapeRecord
    right_tape: TapeRecord
    divergence_step: int | None = None
    token_delta: int = 0
    token_delta_pct: float = 0.0
    cost_delta_usd: float = 0.0
    latency_delta_ms: float = 0.0
    replayed_steps_saved_ms: float = 0.0
    replayed_tokens_saved: int = 0
    cache_hit_rate_left: float = 0.0
    cache_hit_rate_right: float = 0.0
    step_diffs: list[StepDiffEntry] = Field(default_factory=list)


class TapeBundle(BaseModel):
    """Self-contained portable `.ftk.json` bundle for offline sharing and CI fixtures."""

    schema_version: str = "1.0"
    tape: TapeRecord
    steps: list[TapeStep] = Field(default_factory=list)
    mcp_spans: list[MCPSpan] = Field(default_factory=list)
    biopsy_report: BiopsyReport | None = None
