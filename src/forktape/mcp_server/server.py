"""MCP Server (`forktape mcp`) enabling Claude Code, Cursor, and Windsurf to debug broken Agents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    from mcp.server.mcpserver import MCPServer
except ImportError:  # pragma: no cover - fallback for mcp 1.x
    from mcp.server.fastmcp import FastMCP as MCPServer  # type: ignore[no-redef]

from forktape.compiler.pytest_gen import compile_tape_to_pytest_code, export_tape_to_pytest_file
from forktape.config import ForkTapeConfig
from forktape.doctor.graph import run_counterfactual_biopsy
from forktape.engine.diff import compare_tapes
from forktape.engine.replayer import ReplayEngine
from forktape.models.db import TapeStore
from forktape.models.tape import OverridePatch


def build_mcp_server(
    store: TapeStore | None = None,
    config: ForkTapeConfig | None = None,
) -> MCPServer:
    """Construct the ForkTape MCP server with 6 agent-native debugging tools."""
    cfg = config or ForkTapeConfig()
    tape_store = store or TapeStore(cfg.db_path)
    engine = ReplayEngine(tape_store, cfg)

    mcp = MCPServer(
        "ForkTape",
        instructions=(
            "Use ForkTape tools to inspect failed AI agent trajectories, run zero-cost counterfactual "
            "forks at specific steps, verify root causes via LangGraph ablation biopsy, and export "
            "deterministic pytest regression tests."
        ),
    )

    @mcp.tool()
    def list_tapes(status: str = "", limit: int = 20) -> str:
        """List recorded agent trajectory tapes with token usage, KV cache hit rate, and status."""
        tapes = tape_store.list_tapes(status=status or None, limit=limit)
        summary = [
            {
                "tape_id": t.tape_id,
                "parent_tape_id": t.parent_tape_id,
                "fork_from_step": t.fork_from_step,
                "title": t.title,
                "status": t.status.value,
                "failure_reason": t.failure_reason,
                "step_count": t.step_count,
                "total_tokens": t.total_tokens,
                "cache_hit_rate": t.cache_hit_rate,
                "total_cost_usd": t.total_cost_usd,
            }
            for t in tapes
        ]
        return json.dumps({"count": len(summary), "tapes": summary}, indent=2, ensure_ascii=False)

    @mcp.tool()
    def inspect_tape_timeline(tape_id: str) -> str:
        """Inspect step-by-step timeline, tool calls, token breakdown, and KV cache ratios of a tape."""
        tape = tape_store.get_tape(tape_id)
        if tape is None:
            return json.dumps({"error": f"Tape '{tape_id}' not found"})
        steps = tape_store.get_steps(tape_id)
        mcp_spans = tape_store.get_mcp_spans(tape_id)
        timeline: list[dict[str, Any]] = []
        for s in steps:
            timeline.append(
                {
                    "step_index": s.step_index,
                    "model": s.model,
                    "replayed": s.replayed,
                    "prompt_tokens": s.prompt_tokens,
                    "token_breakdown": s.token_breakdown.model_dump(),
                    "prefix_cache_ratio": s.prefix_cache_ratio,
                    "duration_ms": s.duration_ms,
                    "tool_calls": [tc.model_dump() for tc in s.tool_calls],
                    "assembled_text_preview": s.assembled_text[:200],
                    "anomalies": [a.code for a in s.anomaly_flags],
                }
            )
        return json.dumps(
            {
                "tape": tape.model_dump(),
                "steps": timeline,
                "mcp_span_count": len(mcp_spans),
            },
            indent=2,
            ensure_ascii=False,
        )

    @mcp.tool()
    def inspect_step_detail(tape_id: str, step_index: int) -> str:
        """Inspect full prompt context, registered tool count, and raw output of a specific step."""
        step = tape_store.get_step(tape_id, step_index)
        if step is None:
            return json.dumps({"error": f"Step {step_index} not found in tape '{tape_id}'"})
        tools = step.raw_request.get("tools") or []
        return json.dumps(
            {
                "tape_id": tape_id,
                "step_index": step_index,
                "model": step.model,
                "registered_tool_count": len(tools),
                "registered_tool_names": [
                    (t.get("function") or t).get("name", "") for t in tools[:50]
                ],
                "token_breakdown": step.token_breakdown.model_dump(),
                "prefix_cache_ratio": step.prefix_cache_ratio,
                "prefix_break_snippet": step.prefix_break_snippet,
                "messages": step.raw_request.get("messages", []),
                "tool_calls": [tc.model_dump() for tc in step.tool_calls],
                "assembled_text": step.assembled_text,
            },
            indent=2,
            ensure_ascii=False,
        )

    @mcp.tool()
    async def fork_and_ablate_step(
        tape_id: str,
        step_index: int,
        prune_tools_top_k: int = 0,
        compact_context_max_chars: int = 0,
        stabilize_prefix_cache: bool = False,
        system_prompt_append: str = "",
        swap_model: str = "",
    ) -> str:
        """Time-travel fork `tape_id` at `step_index` (replaying steps 1..K-1 at 0 token cost) with counterfactual patches."""
        patch = OverridePatch(
            label=f"MCP Fork @ Step {step_index}",
            prune_tools_top_k=prune_tools_top_k if prune_tools_top_k > 0 else None,
            compact_context_max_chars=(
                compact_context_max_chars if compact_context_max_chars > 0 else None
            ),
            stabilize_prefix_cache=stabilize_prefix_cache,
            system_prompt_append=system_prompt_append or None,
            swap_model=swap_model or None,
            override_temperature=0.0,
        )
        child_tape, forked_step = await engine.fork_step_standalone(
            source_tape_id=tape_id,
            step_index=step_index,
            patch=patch,
        )
        diff = compare_tapes(tape_store, tape_id, child_tape.tape_id)
        return json.dumps(
            {
                "forked_tape_id": child_tape.tape_id,
                "forked_step_tool_calls": [tc.model_dump() for tc in forked_step.tool_calls],
                "forked_step_text": forked_step.assembled_text,
                "prompt_tokens_after": forked_step.prompt_tokens,
                "prefix_cache_ratio_after": forked_step.prefix_cache_ratio,
                "replayed_tokens_saved": diff.replayed_tokens_saved,
                "replayed_steps_saved_ms": diff.replayed_steps_saved_ms,
            },
            indent=2,
            ensure_ascii=False,
        )

    @mcp.tool()
    async def run_biopsy_doctor(tape_id: str) -> str:
        """Run the LangGraph Counterfactual Biopsy Doctor to prove the root cause of a failed tape."""
        report = await run_counterfactual_biopsy(tape_store, engine, tape_id)
        return report.model_dump_json(indent=2)

    @mcp.tool()
    def export_pytest_regression(tape_id: str, output_path: str = "") -> str:
        """Compile a failed tape into a deterministic `pytest` regression test suite."""
        if output_path:
            py_file, bundle_file = export_tape_to_pytest_file(
                tape_store, tape_id, Path(output_path)
            )
            return json.dumps(
                {
                    "exported": True,
                    "pytest_file": str(py_file),
                    "bundle_fixture": str(bundle_file),
                },
                indent=2,
            )
        code = compile_tape_to_pytest_code(tape_store, tape_id)
        return json.dumps({"tape_id": tape_id, "pytest_code": code}, indent=2)

    return mcp
