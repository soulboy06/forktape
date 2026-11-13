"""Pre-built showcase failure tapes for zero-API-key `uvx forktape demo` experience."""

from __future__ import annotations

import json
from typing import Any

from forktape.config import estimate_cost_usd
from forktape.doctor.graph import run_counterfactual_biopsy
from forktape.engine.fingerprint import (
    analyze_prefix_cache_continuity,
    canonical_request_hash,
    estimate_step_token_breakdown,
)
from forktape.engine.replayer import ReplayEngine
from forktape.models.db import TapeStore
from forktape.models.tape import (
    MCPSpan,
    TapeRecord,
    TapeStatus,
    TapeStep,
    ToolCallInvocation,
)


def _make_tool_schema(name: str, description: str, props: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": props, "required": list(props.keys())},
        },
    }


def _record_synthetic_step(
    store: TapeStore,
    tape_id: str,
    step_index: int,
    model: str,
    req: dict[str, Any],
    prev_req: dict[str, Any] | None,
    tool_calls: list[ToolCallInvocation],
    text: str,
    duration_ms: float,
    cached_override: int | None = None,
) -> TapeStep:
    breakdown = estimate_step_token_breakdown(req)
    p_tok = breakdown.total
    cache = analyze_prefix_cache_continuity(prev_req, req)
    c_tok = cached_override if cached_override is not None else (
        int(p_tok * 0.85) if prev_req and cache.ratio >= 0.90 else 0
    )
    o_tok = 68
    raw_resp = {
        "id": f"chatcmpl-demo-{tape_id}-{step_index}",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text or None,
                    "tool_calls": [
                        {
                            "id": tc.id or f"call_{step_index}",
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                            },
                        }
                        for tc in tool_calls
                    ]
                    if tool_calls
                    else None,
                },
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": p_tok,
            "completion_tokens": o_tok,
            "total_tokens": p_tok + o_tok,
            "prompt_tokens_details": {"cached_tokens": c_tok},
        },
    }
    step = TapeStep(
        tape_id=tape_id,
        step_index=step_index,
        provider="openai",
        model=model,
        canonical_hash=canonical_request_hash(req),
        token_breakdown=breakdown,
        prefix_cache_ratio=cache.ratio,
        prefix_break_offset=cache.break_offset,
        prefix_break_snippet=cache.break_snippet,
        raw_request=req,
        raw_response=raw_resp,
        assembled_text=text,
        tool_calls=tool_calls,
        ttft_ms=round(duration_ms * 0.28, 1),
        duration_ms=duration_ms,
        prompt_tokens=p_tok,
        cached_prompt_tokens=c_tok,
        completion_tokens=o_tok,
        cost_usd=estimate_cost_usd(model, p_tok, c_tok, o_tok),
    )
    return store.append_step(step)


async def seed_demo_showcase(store: TapeStore) -> list[str]:
    """Populate 4 realistic multi-step failure scenarios + pre-computed biopsy & fixed fork branches."""
    existing = {t.tape_id for t in store.list_tapes(limit=200)}
    seeded_ids: list[str] = []
    engine = ReplayEngine(store)

    # =========================================================================
    # Scenario 1: 42-Tool MCP Context Tax -> Wrong Refund Tool Selected at Step 5
    # =========================================================================
    tid1 = "tape_demo_mcp_overload"
    if tid1 not in existing:
        t1 = TapeRecord(
            tape_id=tid1,
            title="E-Commerce Support Agent — 42 MCP Tools Schema Overload",
            agent_command="python agents/order_support_langgraph.py --order ORD-8841",
            status=TapeStatus.FAILED,
            failure_reason="Step 5 invoked `legacy_internal_admin_wipe_order` instead of `issue_partial_refund` due to 42-tool schema dilution.",
            tags=["mcp-context-tax", "langgraph", "production-incident"],
        )
        store.create_tape(t1)

        tools_42 = [
            _make_tool_schema(
                f"erp_warehouse_sync_{i}",
                f"Synchronize ERP pallet inventory manifest shard #{i} across regional warehouses and customs queues.",
                {"warehouse_code": {"type": "string"}, "shard_id": {"type": "integer"}},
            )
            for i in range(40)
        ]
        tools_42.insert(
            12,
            _make_tool_schema(
                "issue_partial_refund",
                "Issue a customer partial order refund for damaged items on a verified ORD-XXXX order.",
                {"order_id": {"type": "string"}, "refund_amount": {"type": "number"}},
            ),
        )
        tools_42.append(
            _make_tool_schema(
                "legacy_internal_admin_wipe_order",
                "DEPRECATED internal admin ledger reset — never call for standard customer refunds.",
                {"order_id": {"type": "string"}},
            )
        )

        msgs: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": "You are an autonomous e-commerce refund resolution agent. Verify the order, check item damage policy, and issue an accurate partial refund.",
            },
            {
                "role": "user",
                "content": "Customer reports cracked screen protector in order ORD-8841 ($29.00 item). Issue partial refund of $29.00.",
            },
        ]
        prev_r: dict[str, Any] | None = None

        steps_plan = [
            ("lookup_order_details", {"order_id": "ORD-8841"}, '{"order_id":"ORD-8841","total":149.0,"item":"Screen Protector","item_price":29.0,"status":"delivered"}', 1840.0),
            ("verify_damage_claim", {"order_id": "ORD-8841"}, '{"photo_verified":true,"eligible_partial_refund":29.0}', 1620.0),
            ("check_customer_tier", {"order_id": "ORD-8841"}, '{"tier":"VIP","auto_approve_under":50.0}', 1510.0),
            ("calculate_tax_adjustment", {"amount": 29.0}, '{"net_refund":29.0}', 1490.0),
        ]

        for s_idx, (t_name, t_args, mcp_out, dur) in enumerate(steps_plan, start=1):
            req = {"model": "gpt-4o", "tools": tools_42, "messages": json.loads(json.dumps(msgs))}
            tc = ToolCallInvocation(id=f"call_s{s_idx}", name=t_name, arguments=t_args)
            _record_synthetic_step(store, tid1, s_idx, "gpt-4o", req, prev_r, [tc], "", dur)
            store.append_mcp_span(
                MCPSpan(
                    tape_id=tid1,
                    step_index=s_idx,
                    server_name="shopify-orders-mcp",
                    tool_name=t_name,
                    arguments=t_args,
                    result=json.loads(mcp_out),
                    duration_ms=85.0,
                )
            )
            prev_r = req
            msgs.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": t_name, "arguments": json.dumps(t_args)}}],
                }
            )
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": mcp_out})

        # Step 5: Faulty tool selection under 42-tool schema overload!
        req5 = {"model": "gpt-4o", "tools": tools_42, "messages": json.loads(json.dumps(msgs))}
        bad_tc = ToolCallInvocation(
            id="call_s5_bad",
            name="legacy_internal_admin_wipe_order",
            arguments={"order_id": "ORD-8841"},
        )
        _record_synthetic_step(store, tid1, 5, "gpt-4o", req5, prev_r, [bad_tc], "", 2420.0)
        await run_counterfactual_biopsy(store, engine, tid1)
        seeded_ids.append(tid1)

    # =========================================================================
    # Scenario 2: 18,000-Char Raw JSON Bloat -> Context Middle Loss Policy Breach
    # =========================================================================
    tid2 = "tape_demo_context_rot"
    if tid2 not in existing:
        t2 = TapeRecord(
            tape_id=tid2,
            title="FinOps Cloud Expense Agent — 18k JSON Context Middle Loss",
            agent_command="python agents/finops_approver.py --invoice INV-4402",
            status=TapeStatus.FAILED,
            failure_reason="Step 4 violated the $500 auto-approval cap after Step 2 injected 18,400 chars of uncompacted telemetry JSON.",
            tags=["context-middle-loss", "finops", "policy-breach"],
        )
        store.create_tape(t2)

        fin_tools = [
            _make_tool_schema("fetch_invoice_telemetry", "Fetch cloud billing line items", {"invoice_id": {"type": "string"}}),
            _make_tool_schema("approve_payout", "Approve invoice payout respecting $500 cap", {"invoice_id": {"type": "string"}, "amount": {"type": "number"}}),
        ]
        msgs2: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": "STRICT POLICY: Maximum autonomous payout cap is $500. Any invoice above $500 MUST be capped or escalated.",
            },
            {"role": "user", "content": "Audit cloud invoice INV-4402 and process compliant payout."},
        ]
        r2_1 = {"model": "gpt-4o", "tools": fin_tools, "messages": json.loads(json.dumps(msgs2))}
        tc2_1 = ToolCallInvocation(id="call_f1", name="fetch_invoice_telemetry", arguments={"invoice_id": "INV-4402"})
        _record_synthetic_step(store, tid2, 1, "gpt-4o", r2_1, None, [tc2_1], "", 980.0)

        giant_blob = json.dumps(
            {
                "invoice_id": "INV-4402",
                "requested_amount": 2400.0,
                "raw_k8s_metrics": [f"pod-metric-bucket-{i:03d}-" + ("x" * 180) for i in range(85)],
                "summary": "Requested $2400; policy cap applies.",
            }
        )
        msgs2.append({"role": "assistant", "content": None, "tool_calls": [{"id": "call_f1", "type": "function", "function": {"name": "fetch_invoice_telemetry", "arguments": '{"invoice_id":"INV-4402"}'}}]})
        msgs2.append({"role": "tool", "tool_call_id": "call_f1", "content": giant_blob})

        r2_2 = {"model": "gpt-4o", "tools": fin_tools, "messages": json.loads(json.dumps(msgs2))}
        _record_synthetic_step(store, tid2, 2, "gpt-4o", r2_2, r2_1, [], "Reviewing telemetry breakdown...", 1750.0)
        msgs2.append({"role": "assistant", "content": "Reviewing telemetry breakdown..."})
        msgs2.append({"role": "user", "content": "Proceed with final action for INV-4402."})

        r2_3 = {"model": "gpt-4o", "tools": fin_tools, "messages": json.loads(json.dumps(msgs2))}
        bad_payout = ToolCallInvocation(
            id="call_f3",
            name="approve_payout",
            arguments={"invoice_id": "INV-4402", "amount": 2400.0, "exceeds_policy": True},
        )
        _record_synthetic_step(store, tid2, 3, "gpt-4o", r2_3, r2_2, [bad_payout], "", 2190.0)
        await run_counterfactual_biopsy(store, engine, tid2)
        seeded_ids.append(tid2)

    # =========================================================================
    # Scenario 3: Silent Empty Retrieval -> Fabricated Incident ID Hallucination
    # =========================================================================
    tid3 = "tape_demo_empty_retrieval"
    if tid3 not in existing:
        t3 = TapeRecord(
            tape_id=tid3,
            title="SRE On-Call RCA Agent — Empty Log Query Entity Hallucination",
            agent_command="python agents/sre_oncall_bot.py --service checkout-api",
            status=TapeStatus.FAILED,
            failure_reason="Step 3 fabricated non-existent incident `INC-9042` and triggered rollback after `query_datadog_incidents` returned `[]`.",
            tags=["hallucination", "empty-retrieval", "sre-agent"],
        )
        store.create_tape(t3)

        sre_tools = [
            _make_tool_schema("query_datadog_incidents", "Query active Datadog incidents", {"service": {"type": "string"}}),
            _make_tool_schema("rollback_production_deployment", "Rollback production service deployment", {"incident_id": {"type": "string"}, "service": {"type": "string"}}),
        ]
        msgs3: list[dict[str, Any]] = [
            {"role": "system", "content": "You are an SRE incident responder for checkout-api."},
            {"role": "user", "content": "Check if checkout-api has an active P1 incident and remediate if confirmed."},
        ]
        r3_1 = {"model": "claude-3-7-sonnet", "tools": sre_tools, "messages": json.loads(json.dumps(msgs3))}
        tc3_1 = ToolCallInvocation(id="call_sre1", name="query_datadog_incidents", arguments={"service": "checkout-api"})
        _record_synthetic_step(store, tid3, 1, "claude-3-7-sonnet", r3_1, None, [tc3_1], "", 1120.0)

        msgs3.append({"role": "assistant", "content": None, "tool_calls": [{"id": "call_sre1", "type": "function", "function": {"name": "query_datadog_incidents", "arguments": '{"service":"checkout-api"}'}}]})
        msgs3.append({"role": "tool", "tool_call_id": "call_sre1", "content": '{"results": []}'})

        r3_2 = {"model": "claude-3-7-sonnet", "tools": sre_tools, "messages": json.loads(json.dumps(msgs3))}
        hallucinated_tc = ToolCallInvocation(
            id="call_sre2",
            name="rollback_production_deployment",
            arguments={"incident_id": "INC-9042", "service": "checkout-api"},
        )
        _record_synthetic_step(
            store,
            tid3,
            2,
            "claude-3-7-sonnet",
            r3_2,
            r3_1,
            [hallucinated_tc],
            "Rolling back checkout-api for incident INC-9042.",
            1640.0,
        )
        await run_counterfactual_biopsy(store, engine, tid3)
        seeded_ids.append(tid3)

    # =========================================================================
    # Scenario 4: Prefix KV Cache Thrashing (0% Cache Hit Across Multi-Turn Coding Loop)
    # =========================================================================
    tid4 = "tape_demo_kv_cache_thrash"
    if tid4 not in existing:
        t4 = TapeRecord(
            tape_id=tid4,
            title="Repo Coding Agent — Prefix KV Cache Thrashing & Latency Spike",
            agent_command="python agents/repo_refactor_agent.py",
            status=TapeStatus.FAILED,
            failure_reason="Dynamic per-second timestamp in System Prompt header + shuffled tool schemas broke Prefix KV Cache (3% LCP), inflating cost 4.2x.",
            tags=["kv-cache-thrashing", "cost-optimization", "context-engineering"],
        )
        store.create_tape(t4)

        code_tools_a = [
            _make_tool_schema("read_source_file", "Read repository file contents", {"path": {"type": "string"}}),
            _make_tool_schema("search_ast_symbols", "Search AST function definitions", {"symbol": {"type": "string"}}),
            _make_tool_schema("run_unit_tests", "Run pytest suite", {"target": {"type": "string"}}),
        ]
        code_tools_b = list(reversed(code_tools_a))

        m4_1 = [
            {
                "role": "system",
                "content": "Current time: 2026-11-14T02:10:01.104Z\nTrace ID: req_99182\n" + ("You are a senior Python refactoring agent. Follow PEP8 and preserve public interfaces. " * 25),
            },
            {"role": "user", "content": "Inspect auth middleware token validation."},
        ]
        r4_1 = {"model": "gpt-4o", "tools": code_tools_a, "messages": m4_1}
        _record_synthetic_step(
            store, tid4, 1, "gpt-4o", r4_1, None,
            [ToolCallInvocation(id="c4_1", name="read_source_file", arguments={"path": "src/auth.py"})],
            "", 1320.0, cached_override=0
        )

        m4_2 = [
            {
                "role": "system",
                "content": "Current time: 2026-11-14T02:10:04.882Z\nTrace ID: req_77410\n" + ("You are a senior Python refactoring agent. Follow PEP8 and preserve public interfaces. " * 25),
            },
            {"role": "user", "content": "Inspect auth middleware token validation."},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c4_1", "type": "function", "function": {"name": "read_source_file", "arguments": '{"path":"src/auth.py"}'}}]},
            {"role": "tool", "tool_call_id": "c4_1", "content": "def verify_jwt(token: str) -> dict: ..."},
        ]
        r4_2 = {"model": "gpt-4o", "tools": code_tools_b, "messages": m4_2}
        _record_synthetic_step(
            store, tid4, 2, "gpt-4o", r4_2, r4_1,
            [ToolCallInvocation(id="c4_2", name="run_unit_tests", arguments={"target": "tests/test_auth.py"})],
            "", 2380.0, cached_override=0
        )
        await run_counterfactual_biopsy(store, engine, tid4)
        seeded_ids.append(tid4)

    return seeded_ids
