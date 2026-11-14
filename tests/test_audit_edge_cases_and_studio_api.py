"""Adversarial regression tests for Principal Audit edge cases and Web Studio E2E API."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import httpx
from httpx import ASGITransport

from forktape.compiler.pytest_gen import compile_tape_to_pytest_code
from forktape.config import ForkTapeConfig
from forktape.demo.sample_tapes import seed_demo_showcase
from forktape.doctor.heuristics import _safe_float
from forktape.engine.fingerprint import (
    canonical_request_hash,
    redact_payload,
    redact_secrets_in_text,
)
from forktape.engine.patcher import apply_override_patch
from forktape.engine.replayer import ReplayEngine
from forktape.interceptor.llm_proxy import create_app
from forktape.interceptor.sse_assembler import (
    AnthropicSSEAssembler,
    synthesize_anthropic_sse_chunks,
)
from forktape.models.db import TapeStore
from forktape.models.tape import MCPSpan, OverridePatch, TapeRecord, TapeStatus
from forktape.testing import TapeReplayHarness


async def test_streaming_upstream_429_passthrough_and_split_utf8_sse_chunks(
    tmp_path: Path,
) -> None:
    """Verify 4xx streaming errors don't crash or advance step_index, and 2-byte UTF-8 SSE decodes cleanly."""
    attempt_counter = {"count": 0}

    async def raw_http_sse_handler(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            content_length = 0
            while True:
                line = await reader.readline()
                if not line or line in (b"\r\n", b"\n"):
                    break
                if line.lower().startswith(b"content-length:"):
                    content_length = int(line.decode("ascii").split(":", 1)[1].strip())
            if content_length > 0:
                await reader.readexactly(content_length)

            attempt_counter["count"] += 1
            if attempt_counter["count"] == 1:
                err_body = json.dumps(
                    {"error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}}
                ).encode("utf-8")
                writer.write(
                    b"HTTP/1.1 429 Too Many Requests\r\n"
                    b"Content-Type: application/json\r\n"
                    b"Connection: close\r\n"
                    + f"Content-Length: {len(err_body)}\r\n\r\n".encode("ascii")
                    + err_body
                )
                await writer.drain()
                return

            sse_events = [
                {
                    "id": "chatcmpl-utf8",
                    "model": "gpt-4o-mini",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": "你好🚀世界",
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_bad_args",
                                        "function": {
                                            "name": "ping_service",
                                            "arguments": '"not-a-json-object"',
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                },
                {
                    "id": "chatcmpl-utf8",
                    "model": "gpt-4o-mini",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                    "usage": {"prompt_tokens": 24, "completion_tokens": 9, "total_tokens": 33},
                },
            ]
            raw_payload = (
                "".join(f"data: {json.dumps(ev, ensure_ascii=False)}\n\n" for ev in sse_events)
                + "data: [DONE]\n\n"
            ).encode("utf-8")

            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/event-stream\r\n"
                b"Connection: close\r\n"
                + f"Content-Length: {len(raw_payload)}\r\n\r\n".encode("ascii")
            )
            await writer.drain()

            # Deliberately split multi-byte UTF-8 CJK & emoji across 2-byte TCP frames
            for idx in range(0, len(raw_payload), 2):
                writer.write(raw_payload[idx : idx + 2])
                await writer.drain()
                await asyncio.sleep(0.0003)
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(raw_http_sse_handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]

    try:
        cfg = ForkTapeConfig(
            db_path=tmp_path / "utf8_proxy.db",
            openai_upstream=f"http://127.0.0.1:{port}/v1",
        )
        with TapeStore(cfg.db_path) as store:
            engine = ReplayEngine(store, cfg)
            app = create_app(store=store, config=cfg, replay_engine=engine)

            async with httpx.AsyncClient(
                transport=ASGITransport(app=app), base_url="http://forktape.local"
            ) as client:
                rec = (
                    await client.post("/api/sessions/record", json={"title": "UTF8 + 429 Retry"})
                ).json()
                sid = rec["session_id"]
                tape_id = rec["tape_id"]

                # Reject non-http(s) X-ForkTape-Upstream SSRF attempt
                bad_up = await client.post(
                    f"/s/{sid}/v1/chat/completions",
                    headers={"X-ForkTape-Upstream": "file:///etc/passwd"},
                    json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]},
                )
                assert bad_up.status_code == 400

                # Attempt 1: Upstream returns 429 during stream=True
                r1 = await client.post(
                    f"/s/{sid}/v1/chat/completions",
                    json={
                        "model": "gpt-4o-mini",
                        "stream": True,
                        "messages": [{"role": "user", "content": "测试重试"}],
                    },
                )
                assert r1.status_code == 429
                assert "Rate limit exceeded" in r1.text
                assert store.get_steps(tape_id) == []

                # Attempt 2: Retry succeeds and streams 2-byte split UTF-8 SSE frames
                r2 = await client.post(
                    f"/s/{sid}/v1/chat/completions",
                    json={
                        "model": "gpt-4o-mini",
                        "stream": True,
                        "messages": [{"role": "user", "content": "测试重试"}],
                    },
                )
                assert r2.status_code == 200
                assert "你好🚀世界" in r2.text
                assert "\ufffd" not in r2.text

                steps = store.get_steps(tape_id)
                assert len(steps) == 1
                assert steps[0].step_index == 1
                assert steps[0].assembled_text == "你好🚀世界"
                assert steps[0].tool_calls[0].name == "ping_service"
                assert steps[0].tool_calls[0].arguments == '"not-a-json-object"'
    finally:
        server.close()
        await server.wait_closed()


def test_anthropic_sse_thinking_and_patcher_list_blocks() -> None:
    """Verify Anthropic thinking/signature SSE roundtrip and system/tool_result list patching."""
    assembler = AnthropicSSEAssembler()
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_anth_01",
                "model": "claude-3-7-sonnet-20250219",
                "usage": {"input_tokens": 120, "cache_read_input_tokens": 64},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": "Step 1: ", "signature": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "check refund cap."},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "sig_verified_abc"},
        },
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_01",
                "name": "lookup_policy",
                "input": {"tier": "vip"},
            },
        },
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 42},
        },
    ]
    for ev in events:
        assembler.feed_line(f"data: {json.dumps(ev)}")

    built, text, reasoning, tcs = assembler.build_response()
    assert text == ""
    assert reasoning == "Step 1: check refund cap."
    assert built["content"][0]["signature"] == "sig_verified_abc"
    assert len(tcs) == 1
    assert tcs[0].arguments == {"tier": "vip"}

    # Re-synthesize SSE chunks for VCR replay and re-assemble
    synthesized_chunks = synthesize_anthropic_sse_chunks(built)
    replay_assembler = AnthropicSSEAssembler()
    for ch in synthesized_chunks:
        for line in ch.decode("utf-8").splitlines():
            replay_assembler.feed_line(line)
    rebuilt, _, replay_reasoning, replay_tcs = replay_assembler.build_response()
    assert replay_reasoning == "Step 1: check refund cap."
    assert rebuilt["content"][0]["signature"] == "sig_verified_abc"
    assert replay_tcs[0].arguments == {"tier": "vip"}

    # Test Anthropic apply_override_patch with list-based system & tool_result blocks
    stale_blob = "X" * 1600
    anth_req: dict[str, Any] = {
        "model": "claude-3-7-sonnet-20250219",
        "system": [
            {
                "type": "text",
                "text": "Current time: 2026-11-14T03:00:00Z. Help the customer.",
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [
            {"role": "user", "content": "Refund order ORD-9 for $500"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_hist",
                        "name": "legacy_audit_logger",
                        "input": {"order": "ORD-9"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_hist",
                        "content": [{"type": "text", "text": stale_blob}],
                    }
                ],
            },
            {"role": "assistant", "content": "Checking policy next."},
            {"role": "user", "content": "Please issue the refund now."},
        ],
        "tools": [
            {"name": "issue_refund", "description": "Refund an order", "input_schema": {}},
            {"name": "legacy_audit_logger", "description": "Unrelated audit", "input_schema": {}},
            {"name": "weather_forecast", "description": "Check weather", "input_schema": {}},
            {"name": "k8s_pod_restart", "description": "Restart pod", "input_schema": {}},
        ],
    }
    patched = apply_override_patch(
        anth_req,
        OverridePatch(
            system_prompt_append="Always verify VIP cap.",
            stabilize_prefix_cache=True,
            prune_tools_top_k=1,
            compact_context_max_chars=300,
        ),
    )
    # Anthropic forbids role='system' inside messages!
    assert all(m.get("role") != "system" for m in patched["messages"])
    assert "<ISO_TS>" in patched["system"][0]["text"]
    assert "Help the customer." in patched["system"][0]["text"]
    assert "Always verify VIP cap." in patched["system"][-1]["text"]
    kept_names = {t["name"] for t in patched["tools"]}
    assert "issue_refund" in kept_names
    assert "legacy_audit_logger" in kept_names  # Retained because referenced in tool_use history!
    compacted_tool_text = patched["messages"][2]["content"][0]["content"][0]["text"]
    assert "ForkTape compacted" in compacted_tool_text


async def test_cross_process_session_hydration_and_stateful_mcp_vcr(tmp_path: Path) -> None:
    """Verify SQLite cross-process session hydration, duplicate MCP call ordering, and wipe_* sandbox."""
    cfg = ForkTapeConfig(db_path=tmp_path / "cross_proc.db")
    with TapeStore(cfg.db_path) as store1:
        await seed_demo_showcase(store1)
        engine1 = ReplayEngine(store1, cfg)
        sess = engine1.sessions.start_replay_or_fork_session(
            source_tape_id="tape_demo_empty_retrieval",
            fork_from_step=2,
            override_patch=OverridePatch(dry_run_write_tools=True),
            title="Cross Process Fork",
        )
        # Record two distinct MCP spans with identical args + tools/list on step 1
        store1.append_mcp_span(
            MCPSpan(
                tape_id="tape_demo_empty_retrieval",
                step_index=1,
                server_name="queue_srv",
                method="tools/call",
                tool_name="pop_job",
                arguments={"queue": "default"},
                result="job-1",
            )
        )
        store1.append_mcp_span(
            MCPSpan(
                tape_id="tape_demo_empty_retrieval",
                step_index=1,
                server_name="queue_srv",
                method="tools/call",
                tool_name="pop_job",
                arguments={"queue": "default"},
                result="job-2",
            )
        )
        store1.append_mcp_span(
            MCPSpan(
                tape_id="tape_demo_empty_retrieval",
                step_index=1,
                server_name="queue_srv",
                method="tools/list",
                tool_name="tools/list",
                arguments={},
                result={"tools": [{"name": "pop_job"}]},
            )
        )

    # Simulate separate OS process (`forktape mcp-wrap`) opening the same SQLite DB
    with TapeStore(cfg.db_path) as store2:
        engine2 = ReplayEngine(store2, cfg)
        hydrated = engine2.sessions.resolve_session(sess.session_id)
        assert hydrated.tape_id == sess.tape_id
        assert hydrated.source_tape_id == "tape_demo_empty_retrieval"
        assert hydrated.fork_from_step == 2
        assert hydrated.mode == "fork"

        app = create_app(store=store2, config=cfg, replay_engine=engine2)
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://forktape.local"
        ) as client:
            # 1st call gets job-1, 2nd identical call gets job-2 (stateful consumed_mcp_spans!)
            c1 = await client.post(
                f"/s/{sess.session_id}/mcp/queue_srv",
                json={
                    "jsonrpc": "2.0",
                    "id": 10,
                    "method": "tools/call",
                    "params": {"name": "pop_job", "arguments": {"queue": "default"}},
                },
            )
            c2 = await client.post(
                f"/s/{sess.session_id}/mcp/queue_srv",
                json={
                    "jsonrpc": "2.0",
                    "id": 11,
                    "method": "tools/call",
                    "params": {"name": "pop_job", "arguments": {"queue": "default"}},
                },
            )
            assert c1.json()["result"]["content"][0]["text"] == "job-1"
            assert c2.json()["result"]["content"][0]["text"] == "job-2"

            # tools/list replayed from recorded span
            tlist = await client.post(
                f"/s/{sess.session_id}/mcp/queue_srv",
                json={"jsonrpc": "2.0", "id": 12, "method": "tools/list", "params": {}},
            )
            assert tlist.json()["result"]["tools"][0]["name"] == "pop_job"

            # Advance session into forked step 2 and verify wipe_* dry-run sandbox
            hydrated.current_step_index = 2
            sandboxed = await client.post(
                f"/s/{sess.session_id}/mcp/queue_srv",
                json={
                    "jsonrpc": "2.0",
                    "id": 13,
                    "method": "tools/call",
                    "params": {"name": "wipe_production_table", "arguments": {"table": "users"}},
                },
            )
            sandbox_payload = json.loads(sandboxed.json()["result"]["content"][0]["text"])
            assert sandbox_payload["_forktape_dry_run_blocked"] is True


def test_secret_redaction_precision_and_uuidv7_fingerprinting() -> None:
    """Ensure modern tokens are redacted without mangling code snippets or JSON Schemas."""
    # Construct mock token strings dynamically so GitHub Push Protection does not flag test fixtures
    raw_secrets = " ".join(
        [
            "OpenAI=" + "sk-" + "svcacct-" + ("A1b2" * 6),
            "Google=" + "AIza" + ("SyMock0123456789abcdefghij123456789"),
            "HF=" + "hf_" + ("aB1" * 11),
            "Slack=" + "xox" + "b-" + ("1" * 12) + "-" + ("2" * 12) + "-mockSlackTok",
            "AWS=" + "ASIA" + ("0123456789ABCDEF"),
        ]
    )
    redacted = redact_secrets_in_text(raw_secrets)
    assert "<REDACTED_API_KEY>" in redacted
    assert "<REDACTED_GOOGLE_KEY>" in redacted
    assert "<REDACTED_HF_TOKEN>" in redacted
    assert "<REDACTED_SLACK_TOKEN>" in redacted
    assert "<REDACTED_AWS_KEY>" in redacted

    # Legitimate Python code and JSON Schema properties must NOT be mangled
    code_snippet = 'password = os.environ.get("DB_PASS")\ntoken = request.headers.get("token")'
    assert redact_secrets_in_text(code_snippet) == code_snippet

    schema_payload = {
        "Authorization": "Bearer super_secret_header_tok",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "configure_auth",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "authorization": {"type": "string", "description": "OAuth mode"},
                            "cookie": {"type": "boolean"},
                        },
                    },
                },
            }
        ],
    }
    clean = redact_payload(schema_payload)
    assert clean["Authorization"] == "<REDACTED_HEADER>"
    props = clean["tools"][0]["function"]["parameters"]["properties"]
    assert props["authorization"] == {"type": "string", "description": "OAuth mode"}
    assert props["cookie"] == {"type": "boolean"}

    # UUIDv7 normalization in canonical fingerprint
    fp_a = canonical_request_hash(
        {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Trace 01932b6c-7e80-7abc-8def-0123456789ab"}],
        }
    )
    fp_b = canonical_request_hash(
        {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "Trace 01932b6d-1111-7fff-9abc-fedcba987654"}],
        }
    )
    assert fp_a == fp_b


def test_pytest_compiler_adversarial_escaping_and_safe_float(tmp_path: Path) -> None:
    """Verify triple-quote injection safety, step-1 docstring, numeric tape_id, and $600 parsing."""
    assert _safe_float("$600") == 600.0
    assert _safe_float("$1,250.50 USD") == 1250.5
    assert _safe_float("invalid") == 0.0

    with TapeStore(tmp_path / "escape.db") as store:
        tape = store.create_tape(
            TapeRecord(
                tape_id="999",
                title="Adversarial '''single''' and \"\"\"double\"\"\" \\n quotes",
                status=TapeStatus.FAILED,
            )
        )
        engine = ReplayEngine(store)
        engine.build_and_store_live_step(
            tape_id=tape.tape_id,
            step_index=1,
            provider="openai",
            raw_request={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": " Break '''this''' and \"\"\"that\"\"\" "}],
            },
            raw_response={
                "id": "chatcmpl-1",
                "model": "gpt-4o",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
            prev_raw_request=None,
            ttft_ms=5.0,
            duration_ms=10.0,
        )
        code = compile_tape_to_pytest_code(store, "999")
        # Generated code MUST parse & compile as valid Python AST!
        compiled_ast = compile(code, "<generated_pytest>", "exec")
        assert compiled_ast is not None
        assert "async def test_regression_t_999_" in code
        assert "Step 1 executes directly with the counterfactual patch" in code
        assert TapeReplayHarness is not None


async def test_web_studio_spa_and_full_rest_api_e2e(tmp_path: Path) -> None:
    """End-to-end test of Web Studio static bundle + every /api/* endpoint."""
    cfg = ForkTapeConfig(db_path=tmp_path / "studio_e2e.db")
    with TapeStore(cfg.db_path) as store:
        await seed_demo_showcase(store)

        app = create_app(store=store, config=cfg)
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://studio.local"
        ) as client:
            # 1. Serve React 19 Studio index.html & built JS/CSS assets
            idx_resp = await client.get("/")
            assert idx_resp.status_code == 200
            assert '<div id="root"></div>' in idx_resp.text
            js_match = re.search(r'src="(/assets/index-[^"]+\.js)"', idx_resp.text)
            css_match = re.search(r'href="(/assets/index-[^"]+\.css)"', idx_resp.text)
            assert js_match is not None and css_match is not None
            assert (await client.get(js_match.group(1))).status_code == 200
            assert (await client.get(css_match.group(1))).status_code == 200

            # 2. Health & Tapes list
            health = (await client.get("/api/health")).json()
            assert health["status"] == "ok"
            assert health["tape_count"] >= 4

            tapes = (await client.get("/api/tapes")).json()["tapes"]
            assert len(tapes) >= 4

            # 3. Tape detail
            detail = (await client.get("/api/tapes/tape_demo_mcp_overload")).json()
            assert detail["tape"]["tape_id"] == "tape_demo_mcp_overload"
            assert len(detail["steps"]) == 5
            assert len(detail["heuristic_findings"]) >= 1

            # 4. Step validation (step_index=0 -> 400) & valid Time-Travel Fork
            bad_fork = await client.post(
                "/api/tapes/tape_demo_mcp_overload/fork",
                json={"step_index": 0, "patch": {"prune_tools_top_k": 5}},
            )
            assert bad_fork.status_code == 400

            good_fork = await client.post(
                "/api/tapes/tape_demo_mcp_overload/fork",
                json={"step_index": 5, "patch": {"prune_tools_top_k": 5}},
            )
            assert good_fork.status_code == 200
            fork_data = good_fork.json()
            child_id = fork_data["forked_tape"]["tape_id"]
            assert fork_data["forked_tape"]["status"] == "success"

            # 5. Branch diff
            diff_resp = await client.get(f"/api/tapes/tape_demo_mcp_overload/diff/{child_id}")
            assert diff_resp.status_code == 200
            assert diff_resp.json()["token_delta"] < 0

            # 6. Doctor biopsy & Pytest export (including 400 on 0-step tape)
            store.create_tape(TapeRecord(tape_id="empty_zero", title="Empty", status=TapeStatus.FAILED))
            assert (await client.post("/api/tapes/empty_zero/doctor")).status_code == 400
            assert (await client.get("/api/tapes/empty_zero/export-pytest")).status_code == 400

            doc_resp = await client.post("/api/tapes/tape_demo_empty_retrieval/doctor")
            assert doc_resp.status_code == 200
            doc_json = doc_resp.json()
            assert any(r["resolved_failure"] for r in doc_json["ablation_runs"])
            assert doc_json["recommended_patch"] is not None

            py_resp = await client.get("/api/tapes/tape_demo_empty_retrieval/export-pytest")
            assert py_resp.status_code == 200
            assert "TapeReplayHarness" in py_resp.json()["pytest_code"]

            # 7. Bundle export filename sanitization, delete, and import roundtrip
            store.create_tape(
                TapeRecord(tape_id="unsafe!name@1", title="Unsafe", status=TapeStatus.SUCCESS)
            )
            b_unsafe = await client.get("/api/tapes/unsafe!name@1/bundle")
            assert b_unsafe.status_code == 200
            assert 'filename="unsafe_name_1.ftk.json"' in b_unsafe.headers["content-disposition"]

            bundle_resp = await client.get("/api/tapes/tape_demo_kv_cache_thrash/bundle")
            assert bundle_resp.status_code == 200
            bundle_json = bundle_resp.json()

            del_resp = await client.delete("/api/tapes/tape_demo_kv_cache_thrash")
            assert del_resp.status_code == 200
            assert (await client.get("/api/tapes/tape_demo_kv_cache_thrash")).status_code == 404

            imp_resp = await client.post("/api/tapes/import", json=bundle_json)
            assert imp_resp.status_code == 200
            assert (await client.get("/api/tapes/tape_demo_kv_cache_thrash")).status_code == 200


async def test_deepseek_v4_1_reasoning_auto_heal_and_long_thread_heuristics(tmp_path: Path) -> None:
    """Verify proxy auto-heals stripped reasoning_content for DeepSeek V4.1 and avoids log false-positives."""
    from forktape.doctor.heuristics import _is_tool_output_error, scan_tape_anomalies
    from forktape.models.tape import TapeStep, ToolCallInvocation

    # 1. Healthy observability JSON containing 'ERROR'/'failed' inside log strings must NOT be flagged as tool error
    healthy_log_json = json.dumps(
        {"status": "ok", "pod_logs": ["2026-11-14 ERROR pool timeout", "2 replicas failed probe"]}
    )
    assert _is_tool_output_error(healthy_log_json) is False
    assert _is_tool_output_error('{"status": "rejected", "error": "cap exceeded"}') is True

    # 2. Intra-step parallel duplicate read calls must NOT trigger multi-step TOOL_CALL_LOOP
    t_rec = TapeRecord(tape_id="parallel_ok", title="Parallel", status=TapeStatus.SUCCESS)
    s_par = TapeStep(
        tape_id="parallel_ok",
        step_index=1,
        raw_request={"model": "deepseek-flash", "messages": [{"role": "user", "content": "check"}]},
        raw_response={},
        tool_calls=[
            ToolCallInvocation(id="c1", name="read_metric", arguments={"k": "cpu"}),
            ToolCallInvocation(id="c2", name="read_metric", arguments={"k": "cpu"}),
        ],
    )
    assert [f for f in scan_tape_anomalies(t_rec, [s_par]) if f.code == "TOOL_CALL_LOOP"] == []

    # 3. Mock DeepSeek V4.1 upstream strictly enforcing `reasoning_content` on assistant tool_calls history
    seen_payloads: list[dict[str, Any]] = []

    async def strict_ds_handler(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            content_length = 0
            while True:
                line = await reader.readline()
                if not line or line in (b"\r\n", b"\n"):
                    break
                if line.lower().startswith(b"content-length:"):
                    content_length = int(line.decode("ascii").split(":", 1)[1].strip())
            raw_body = await reader.readexactly(content_length) if content_length else b"{}"
            req_json = json.loads(raw_body.decode("utf-8"))
            seen_payloads.append(req_json)

            # Enforce DeepSeek V4.1 thinking-mode invariant on Turn 2+
            for m in req_json.get("messages") or []:
                if m.get("role") == "assistant" and m.get("tool_calls") and "reasoning_content" not in m:
                    err = b'{"error":{"message":"The `reasoning_content` in the thinking mode must be passed back"}}'
                    writer.write(
                        b"HTTP/1.1 400 Bad Request\r\nContent-Type: application/json\r\n"
                        + f"Content-Length: {len(err)}\r\n\r\n".encode("ascii")
                        + err
                    )
                    await writer.drain()
                    return

            if len(seen_payloads) == 1:
                resp_obj = {
                    "id": "ds-1",
                    "model": "deepseek-flash",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "reasoning_content": "Need to inspect shard-01 first.",
                                "tool_calls": [
                                    {
                                        "id": "call_ds_01",
                                        "type": "function",
                                        "function": {"name": "probe", "arguments": '{"shard":"01"}'},
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 120, "completion_tokens": 25, "prompt_tokens_details": {"cached_tokens": 0}},
                }
            else:
                resp_obj = {
                    "id": "ds-2",
                    "model": "deepseek-flash",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "All healthy."},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 180, "completion_tokens": 10, "prompt_tokens_details": {"cached_tokens": 128}},
                }
            out_bytes = json.dumps(resp_obj).encode("utf-8")
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                + f"Content-Length: {len(out_bytes)}\r\n\r\n".encode("ascii")
                + out_bytes
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    srv = await asyncio.start_server(strict_ds_handler, "127.0.0.1", 0)
    port = srv.sockets[0].getsockname()[1]
    try:
        cfg = ForkTapeConfig(
            db_path=tmp_path / "ds_heal.db",
            openai_upstream=f"http://127.0.0.1:{port}/v1",
        )
        with TapeStore(cfg.db_path) as store:
            engine = ReplayEngine(store, cfg)
            app = create_app(store=store, config=cfg, replay_engine=engine)
            async with httpx.AsyncClient(
                transport=ASGITransport(app=app), base_url="http://forktape.local"
            ) as client:
                sid = (await client.post("/api/sessions/record", json={"title": "DS V4.1 Heal"})).json()["session_id"]
                r1 = await client.post(
                    f"/s/{sid}/v1/chat/completions",
                    json={"model": "deepseek-flash", "messages": [{"role": "user", "content": "check shard"}]},
                )
                assert r1.status_code == 200
                # Client strips reasoning_content on Turn 2 (classic OpenAI SDK pitfall!)
                r2 = await client.post(
                    f"/s/{sid}/v1/chat/completions",
                    json={
                        "model": "deepseek-flash",
                        "messages": [
                            {"role": "user", "content": "check shard"},
                            {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_ds_01",
                                        "type": "function",
                                        "function": {"name": "probe", "arguments": '{"shard":"01"}'},
                                    }
                                ],
                            },
                            {"role": "tool", "tool_call_id": "call_ds_01", "content": '{"status":"ok"}'},
                        ],
                    },
                )
                assert r2.status_code == 200
                assert seen_payloads[1]["messages"][1]["reasoning_content"] == "Need to inspect shard-01 first."
                steps = store.get_steps(sid)
                assert steps[0].cached_prompt_tokens == 0  # Preserved exact 0 server cache telemetry
                assert steps[1].cached_prompt_tokens == 128
    finally:
        srv.close()
        await srv.wait_closed()
