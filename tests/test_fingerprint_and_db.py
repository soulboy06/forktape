"""Unit tests for Phase 1: Canonical Fingerprinting, KV Cache LCP Analyzer, and SQLite TapeStore."""

from __future__ import annotations

from pathlib import Path

from forktape.config import estimate_cost_usd
from forktape.engine.fingerprint import (
    analyze_prefix_cache_continuity,
    canonical_request_hash,
    compute_structural_similarity,
    estimate_step_token_breakdown,
    redact_payload,
)
from forktape.models.db import TapeStore
from forktape.models.tape import (
    BiopsyReport,
    MCPSpan,
    TapeRecord,
    TapeStatus,
    TapeStep,
    ToolCallInvocation,
)


def test_canonical_hash_ignores_timestamps_uuids_and_random_tool_call_ids() -> None:
    req_run_1 = {
        "model": "gpt-4o",
        "stream": True,
        "messages": [
            {
                "role": "system",
                "content": "You are an order assistant. Current time: 2026-11-14T02:00:01Z. Trace: 550e8400-e29b-41d4-a716-446655440000",
            },
            {"role": "user", "content": "Check order ORD-102"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_random_alpha_991",
                        "type": "function",
                        "function": {"name": "get_order", "arguments": '{"b": 2, "a": 1}'},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_random_alpha_991",
                "content": '{"status": "shipped"}',
            },
        ],
    }

    req_run_2 = {
        "model": "gpt-4o",
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": "You are an order assistant. Current time: 2026-11-15T18:42:59Z. Trace: 123e4567-e89b-12d3-a456-426614174000",
            },
            {"role": "user", "content": "Check order ORD-102"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_completely_different_777",
                        "type": "function",
                        "function": {"name": "get_order", "arguments": '{"a":1,"b":2}'},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_completely_different_777",
                "content": '{"status": "shipped"}',
            },
        ],
    }

    assert canonical_request_hash(req_run_1) == canonical_request_hash(req_run_2)
    assert compute_structural_similarity(req_run_1, req_run_2) == 1.0


def test_secret_redaction_strips_keys_and_headers() -> None:
    dirty = {
        "Authorization": "Bearer sk-proj-1234567890abcdefghijklmnop",
        "nested": {
            "note": "Using token ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456 and sk-ant-api03-999999999999999999",
        },
    }
    cleaned = redact_payload(dirty)
    assert cleaned["Authorization"] == "<REDACTED_HEADER>"
    assert "sk-proj" not in cleaned["nested"]["note"]
    assert "ghp_" not in cleaned["nested"]["note"]
    assert "sk-ant" not in cleaned["nested"]["note"]


def test_prefix_cache_continuity_detects_early_timestamp_thrashing() -> None:
    tools = [
        {"type": "function", "function": {"name": f"tool_{i}", "description": "desc " * 30}}
        for i in range(15)
    ]
    # Case 1: Good static prefix
    step1_good = {
        "tools": tools,
        "messages": [
            {"role": "system", "content": "Static system instructions " * 40},
            {"role": "user", "content": "Find refund policy"},
        ],
    }
    step2_good = {
        "tools": tools,
        "messages": [
            {"role": "system", "content": "Static system instructions " * 40},
            {"role": "user", "content": "Find refund policy"},
            {"role": "assistant", "content": "Checking policy..."},
            {"role": "user", "content": "Apply it to order #1"},
        ],
    }
    good_analysis = analyze_prefix_cache_continuity(step1_good, step2_good)
    assert good_analysis.ratio == 1.0
    assert good_analysis.break_offset is None

    # Case 2: Shuffled tools order breaks KV cache at the very start!
    step2_broken = {
        "tools": list(reversed(tools)),
        "messages": step2_good["messages"],
    }
    bad_analysis = analyze_prefix_cache_continuity(step1_good, step2_broken)
    assert bad_analysis.ratio < 0.15
    assert bad_analysis.break_offset is not None
    assert bad_analysis.break_snippet is not None


def test_token_breakdown_and_sqlite_bundle_roundtrip(tmp_path: Path) -> None:
    db_file = tmp_path / "test_tapes.db"
    store = TapeStore(db_file)

    parent = TapeRecord(tape_id="tape_root01", title="Order Agent Run", tags=["ecommerce"])
    store.create_tape(parent)

    req = {
        "model": "gpt-4o",
        "tools": [{"type": "function", "function": {"name": "lookup_order", "description": "Fetch order"}}],
        "messages": [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Order status?"},
            {"role": "tool", "tool_call_id": "c1", "content": '{"order": 123}'},
        ],
    }
    breakdown = estimate_step_token_breakdown(req)
    assert breakdown.system_tokens > 0
    assert breakdown.tools_schema_tokens > 0
    assert breakdown.history_tokens > 0
    assert breakdown.tool_output_tokens > 0

    cost = estimate_cost_usd("gpt-4o", prompt_tokens=2000, cached_prompt_tokens=1500, completion_tokens=200)
    assert cost > 0

    step = TapeStep(
        tape_id="tape_root01",
        step_index=1,
        model="gpt-4o",
        canonical_hash=canonical_request_hash(req),
        token_breakdown=breakdown,
        raw_request=req,
        raw_response={"id": "chatcmpl-1"},
        tool_calls=[ToolCallInvocation(id="c1", name="lookup_order", arguments={"id": 123})],
        prompt_tokens=2000,
        cached_prompt_tokens=1500,
        completion_tokens=200,
        cost_usd=cost,
        duration_ms=420.5,
    )
    store.append_step(step)
    store.append_mcp_span(
        MCPSpan(
            tape_id="tape_root01",
            step_index=1,
            server_name="orders-mcp",
            tool_name="lookup_order",
            arguments={"id": 123},
            result={"order": 123, "state": "paid"},
            duration_ms=18.2,
        )
    )
    store.update_tape_status("tape_root01", TapeStatus.FAILED, "Wrong refund amount")

    child = TapeRecord(
        tape_id="tape_child01",
        parent_tape_id="tape_root01",
        fork_from_step=1,
        fork_label="Prune Tools Top-5",
        title="Forked Order Agent",
        status=TapeStatus.SUCCESS,
    )
    store.create_tape(child)

    report = BiopsyReport(
        tape_id="tape_root01",
        suspected_drift_step=1,
        root_cause_category="TOOL_SCHEMA_OVERLOAD",
        executive_summary="Verified via ablation.",
    )
    store.save_biopsy_report(report)

    refreshed = store.get_tape("tape_root01")
    assert refreshed is not None
    assert refreshed.step_count == 1
    assert refreshed.total_prompt_tokens == 2000
    assert refreshed.cache_hit_rate == 0.75
    assert len(store.get_children_tapes("tape_root01")) == 1

    bundle_path = tmp_path / "exported.ftk.json"
    store.export_bundle_file("tape_root01", bundle_path)
    store.close()

    # Import into fresh DB
    store2 = TapeStore(tmp_path / "fresh.db")
    imported = store2.import_bundle_file(bundle_path)
    assert imported.tape_id == "tape_root01"
    assert len(store2.get_steps("tape_root01")) == 1
    assert len(store2.get_mcp_spans("tape_root01")) == 1
    assert store2.get_biopsy_report("tape_root01") is not None
    store2.close()
