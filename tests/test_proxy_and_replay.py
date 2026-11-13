"""Integration tests for Phase 2: Dual-Protocol Proxy, SSE Assembly, VCR Fast-Forward, BM25 Patching, and MCP Sandboxing."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from forktape.config import ForkTapeConfig
from forktape.engine.patcher import apply_override_patch
from forktape.engine.replayer import ReplayEngine
from forktape.interceptor.llm_proxy import create_app
from forktape.interceptor.mcp_bridge import MCPInterceptor
from forktape.models.db import TapeStore
from forktape.models.tape import OverridePatch


def _build_mock_upstream_llm() -> tuple[FastAPI, dict[str, Any]]:
    """In-memory mock OpenAI + Anthropic server tracking hit counts and received payloads."""
    state: dict[str, Any] = {"openai_hits": 0, "anthropic_hits": 0, "last_tools_count": 0}
    upstream = FastAPI()

    @upstream.post("/v1/chat/completions", response_model=None)
    async def mock_chat(req: Request) -> Response:
        body = await req.json()
        state["openai_hits"] += 1
        state["last_tools_count"] = len(body.get("tools") or [])
        hit_no = state["openai_hits"]
        is_stream = bool(body.get("stream", False))

        if is_stream:
            async def sse_gen() -> AsyncIterator[bytes]:
                chunks = [
                    {
                        "id": f"chatcmpl-up-{hit_no}",
                        "model": body.get("model", "gpt-4o"),
                        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                    },
                    {
                        "id": f"chatcmpl-up-{hit_no}",
                        "model": body.get("model", "gpt-4o"),
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": f"call_up_{hit_no}",
                                            "type": "function",
                                            "function": {"name": "issue_refund", "arguments": '{"order_id":'},
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                    },
                    {
                        "id": f"chatcmpl-up-{hit_no}",
                        "model": body.get("model", "gpt-4o"),
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {"arguments": f' "ORD-{hit_no}"}}' },
                                        }
                                    ]
                                },
                                "finish_reason": "tool_calls",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 1200,
                            "completion_tokens": 45,
                            "prompt_tokens_details": {"cached_tokens": 800},
                        },
                    },
                ]
                for ch in chunks:
                    yield f"data: {json.dumps(ch)}\n\n".encode()
                yield b"data: [DONE]\n\n"

            return StreamingResponse(sse_gen(), media_type="text/event-stream")

        return JSONResponse(
            {
                "id": f"chatcmpl-up-{hit_no}",
                "object": "chat.completion",
                "model": body.get("model", "gpt-4o"),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"Completed live step #{hit_no} with {state['last_tools_count']} tools",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 900,
                    "completion_tokens": 30,
                    "prompt_tokens_details": {"cached_tokens": 600},
                },
            }
        )

    @upstream.post("/v1/messages")
    async def mock_anthropic(req: Request) -> StreamingResponse:
        body = await req.json()
        state["anthropic_hits"] += 1

        async def anth_gen() -> AsyncIterator[bytes]:
            evs = [
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_anth_1",
                        "model": body.get("model", "claude-3-7-sonnet"),
                        "usage": {"input_tokens": 300, "cache_read_input_tokens": 900},
                    },
                },
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "tool_use", "id": "toolu_01", "name": "lookup_ticket"},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": '{"ticket_id": "T-9"}'},
                },
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use"},
                    "usage": {"output_tokens": 28},
                },
            ]
            for e in evs:
                yield f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode()

        return StreamingResponse(anth_gen(), media_type="text/event-stream")

    return upstream, state


@pytest.mark.asyncio
async def test_record_sse_then_vcr_fast_forward_and_live_bm25_fork(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    upstream_app, upstream_state = _build_mock_upstream_llm()
    upstream_transport = httpx.ASGITransport(app=upstream_app)

    real_async_client = httpx.AsyncClient

    def patched_async_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs.setdefault("transport", upstream_transport)
        return real_async_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", patched_async_client)

    store = TapeStore(tmp_path / "proxy_test.db")
    cfg = ForkTapeConfig(
        db_path=tmp_path / "proxy_test.db",
        openai_upstream="http://mock-upstream/v1",
        anthropic_upstream="http://mock-upstream",
    )
    app = create_app(store=store, config=cfg)

    many_tools = [
        {
            "type": "function",
            "function": {
                "name": "issue_refund" if i == 7 else f"unrelated_erp_tool_{i}",
                "description": (
                    "Issue partial or full customer order refund"
                    if i == 7
                    else f"Manage warehouse forklift telemetry #{i}"
                ),
                "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}},
            },
        }
        for i in range(25)
    ]

    async with real_async_client(
        transport=httpx.ASGITransport(app=app), base_url="http://forktape-local"
    ) as client:
        # 1. Start RECORD session and execute 3 steps (Steps 1 & 2 streaming SSE, Step 3 JSON)
        rec = (await client.post("/api/sessions/record", json={"title": "Refund Run #1"})).json()
        sid = rec["session_id"]

        for step_no in (1, 2):
            r = await client.post(
                "/v1/chat/completions",
                headers={"X-ForkTape-Session": sid},
                json={
                    "model": "gpt-4o",
                    "stream": True,
                    "tools": many_tools,
                    "messages": [
                        {"role": "system", "content": "Current time: 2026-11-14T01:00:00Z"},
                        {"role": "user", "content": f"Refund order ORD-100 step {step_no}"},
                    ],
                },
            )
            assert r.status_code == 200
            assert r.headers["x-forktape-mode"] == "live-record"
            assert "ORD-" in r.text

        r3 = await client.post(
            "/v1/chat/completions",
            headers={"X-ForkTape-Session": sid},
            json={
                "model": "gpt-4o",
                "stream": False,
                "tools": many_tools,
                "messages": [
                    {"role": "system", "content": "Current time: 2026-11-14T01:00:05Z"},
                    {"role": "user", "content": "Please issue refund for order ORD-100 now"},
                ],
            },
        )
        assert r3.status_code == 200
        await client.post(f"/api/sessions/{sid}/complete", json={"status": "failed"})
        assert upstream_state["openai_hits"] == 3

        # 2. Start FORK session at Step 3 with BM25 tool pruning (25 tools -> Top 4!)
        fork_sess = (
            await client.post(
                "/api/sessions/replay",
                json={
                    "source_tape_id": sid,
                    "fork_from_step": 3,
                    "override_patch": {"label": "BM25 Top-4", "prune_tools_top_k": 4},
                    "strict_matching": True,
                },
            )
        ).json()
        fork_sid = fork_sess["session_id"]

        # Steps 1 & 2 MUST hit VCR replay (even with brand-new timestamps!) -> 0 new upstream calls!
        for step_no in (1, 2):
            rv = await client.post(
                "/v1/chat/completions",
                headers={"X-ForkTape-Session": fork_sid},
                json={
                    "model": "gpt-4o",
                    "stream": True,
                    "tools": many_tools,
                    "messages": [
                        {"role": "system", "content": "Current time: 2026-12-31T23:59:59Z"},
                        {"role": "user", "content": f"Refund order ORD-100 step {step_no}"},
                    ],
                },
            )
            assert rv.status_code == 200
            assert rv.headers["x-forktape-mode"] == "vcr-replay"
            assert "data: [DONE]" in rv.text

        assert upstream_state["openai_hits"] == 3  # Zero upstream calls during Steps 1 & 2!

        # Step 3 MUST switch to live-fork AND prune 25 tools down to 4 via BM25!
        rf3 = await client.post(
            "/v1/chat/completions",
            headers={"X-ForkTape-Session": fork_sid},
            json={
                "model": "gpt-4o",
                "stream": False,
                "tools": many_tools,
                "messages": [
                    {"role": "system", "content": "Current time: 2026-12-31T23:59:59Z"},
                    {"role": "user", "content": "Please issue refund for order ORD-100 now"},
                ],
            },
        )
        assert rf3.status_code == 200
        assert rf3.headers["x-forktape-mode"] == "live-fork"
        assert upstream_state["openai_hits"] == 4
        assert upstream_state["last_tools_count"] == 4
        assert "4 tools" in rf3.json()["choices"][0]["message"]["content"]

        # 3. Verify Side-by-Side Branch Diff endpoint
        diff_data = (await client.get(f"/api/tapes/{sid}/diff/{fork_sid}")).json()
        assert diff_data["divergence_step"] == 3
        assert diff_data["replayed_tokens_saved"] > 0

        # 4. Also verify Anthropic SSE streaming recording works cleanly
        anth_rec = (await client.post("/api/sessions/record", json={"title": "Claude Run"})).json()
        ra = await client.post(
            "/v1/messages",
            headers={"X-ForkTape-Session": anth_rec["session_id"]},
            json={
                "model": "claude-3-7-sonnet",
                "stream": True,
                "messages": [{"role": "user", "content": "Check ticket T-9"}],
            },
        )
        assert ra.status_code == 200
        anth_steps = store.get_steps(anth_rec["tape_id"])
        assert len(anth_steps) == 1
        assert anth_steps[0].tool_calls[0].name == "lookup_ticket"
        assert anth_steps[0].cached_prompt_tokens == 900

    store.close()


def test_mcp_zero_side_effect_vcr_and_dry_run_write_sandbox(tmp_path: Path) -> None:
    store = TapeStore(tmp_path / "mcp_test.db")
    engine = ReplayEngine(store)
    mcp = MCPInterceptor(engine)

    # Record run: live DB mutation occurs once
    rec_sess = engine.sessions.start_record_session(title="MCP Write Run")
    rec_sess.current_step_index = 1
    live_rpc_req = {
        "jsonrpc": "2.0",
        "id": 101,
        "method": "tools/call",
        "params": {"name": "create_invoice", "arguments": {"customer": "ACME", "amount": 500}},
    }
    live_rpc_resp = {
        "jsonrpc": "2.0",
        "id": 101,
        "result": {"content": [{"type": "text", "text": '{"invoice_id": "INV-88"}'}], "isError": False},
    }
    assert mcp.try_vcr_or_sandbox_intercept(rec_sess, "billing-mcp", live_rpc_req) is None
    mcp.record_live_mcp_call(rec_sess, "billing-mcp", live_rpc_req, live_rpc_resp, duration_ms=42.0)

    # Fork run at step 2 with dry_run_write_tools=True
    fork_sess = engine.sessions.start_replay_or_fork_session(
        source_tape_id=rec_sess.tape_id,
        fork_from_step=2,
        override_patch=OverridePatch(dry_run_write_tools=True),
    )

    # Step 1 (< fork_from_step=2): Intercepted by VCR! Downstream billing server NOT touched!
    fork_sess.current_step_index = 1
    replayed_rpc = mcp.try_vcr_or_sandbox_intercept(fork_sess, "billing-mcp", live_rpc_req)
    assert replayed_rpc is not None
    assert "INV-88" in json.dumps(replayed_rpc["result"])

    # Step 2 (>= fork_from_step=2): Destructive `delete_customer_record` blocked by Dry-Run Sandbox!
    fork_sess.current_step_index = 2
    dangerous_req = {
        "jsonrpc": "2.0",
        "id": 102,
        "method": "tools/call",
        "params": {"name": "delete_customer_record", "arguments": {"customer_id": "CUST-1"}},
    }
    sandboxed_rpc = mcp.try_vcr_or_sandbox_intercept(fork_sess, "billing-mcp", dangerous_req)
    assert sandboxed_rpc is not None
    assert "_forktape_dry_run_blocked" in json.dumps(sandboxed_rpc["result"])

    # Verify middle-context compaction preserves structure
    bloated_req = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "Current time: 2026-11-14T02:00:00Z\nKeep rules."},
            {"role": "user", "content": "Summarize logs"},
            {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"rows": [f"item_{i}" * 50 for i in range(40)]})},
            {"role": "user", "content": "Final check"},
        ],
    }
    patched = apply_override_patch(
        bloated_req,
        OverridePatch(compact_context_max_chars=500, stabilize_prefix_cache=True),
    )
    assert "Current time:" not in patched["messages"][0]["content"]
    assert len(patched["messages"][2]["content"]) < len(bloated_req["messages"][2]["content"])
    store.close()
