"""FastAPI Dual-Protocol Proxy (OpenAI SSE + Anthropic SSE + MCP HTTP) & Web Studio API."""

from __future__ import annotations

import codecs
import re
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from forktape import __version__
from forktape.config import ForkTapeConfig
from forktape.engine.diff import compare_tapes
from forktape.engine.patcher import apply_override_patch
from forktape.engine.replayer import ReplayEngine, TrajectoryDivergenceError
from forktape.interceptor.mcp_bridge import MCPInterceptor
from forktape.interceptor.sse_assembler import (
    AnthropicSSEAssembler,
    OpenAISSEAssembler,
    synthesize_anthropic_sse_chunks,
    synthesize_openai_sse_chunks,
)
from forktape.models.db import TapeStore
from forktape.models.tape import OverridePatch, TapeBundle, TapeStatus

SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


class StartRecordPayload(BaseModel):
    title: str = "Recorded Agent Run"
    agent_command: str = ""
    tape_id: str | None = None
    tags: list[str] = Field(default_factory=list)


class StartReplayPayload(BaseModel):
    source_tape_id: str
    fork_from_step: int | None = None
    override_patch: OverridePatch | None = None
    strict_matching: bool = False
    auto_fork_on_divergence: bool = True
    title: str | None = None


class CompleteSessionPayload(BaseModel):
    status: TapeStatus = TapeStatus.SUCCESS
    failure_reason: str | None = None


class StandaloneForkPayload(BaseModel):
    step_index: int
    patch: OverridePatch = Field(default_factory=OverridePatch)
    prefer_live_upstream: bool = False


def _validate_http_url(url: str, header_name: str) -> str:
    trimmed = url.strip()
    if not (trimmed.startswith("http://") or trimmed.startswith("https://")):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {header_name}: URL must begin with http:// or https://",
        )
    return trimmed


def _forward_upstream_headers(request: Request) -> dict[str, str]:
    forwarded: dict[str, str] = {}
    for k, v in request.headers.items():
        kl = k.lower()
        if kl in ("authorization", "x-api-key", "anthropic-version", "anthropic-beta", "api-key"):
            forwarded[k] = v
    forwarded["Content-Type"] = "application/json"
    return forwarded


def create_app(
    store: TapeStore | None = None,
    config: ForkTapeConfig | None = None,
    replay_engine: ReplayEngine | None = None,
) -> FastAPI:
    cfg = config or ForkTapeConfig()
    tape_store = store or TapeStore(cfg.db_path)
    engine = replay_engine or ReplayEngine(tape_store, cfg)
    mcp_interceptor = MCPInterceptor(engine)

    app = FastAPI(
        title="ForkTape Proxy & Time-Travel Studio",
        version=__version__,
        description="Deterministic time-travel forking and counterfactual biopsy for AI Agents.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.store = tape_store
    app.state.config = cfg
    app.state.engine = engine

    # ------------------------------------------------------------------
    # 1. OpenAI-Compatible Chat Completions Endpoint (SSE & JSON)
    # ------------------------------------------------------------------
    async def _handle_openai_chat(request: Request, path_session_id: str | None = None) -> Response:
        body: dict[str, Any] = await request.json()
        is_stream = bool(body.get("stream", False))
        session_hdr = path_session_id or request.headers.get("x-forktape-session")
        session = engine.sessions.resolve_session(session_hdr)

        with session.lock:
            step_idx = session.current_step_index + 1
            prev_req = session.prev_raw_request

            # A) Attempt <3ms Deterministic VCR Fast-Forward
            try:
                vcr_step = engine.lookup_vcr_step(session, step_idx, body)
            except TrajectoryDivergenceError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc

            if vcr_step is not None:
                t_vcr = time.perf_counter()
                replayed_step = engine.record_replayed_step_into_session(
                    session,
                    vcr_step,
                    replay_duration_ms=max(0.4, (time.perf_counter() - t_vcr) * 1000.0),
                )
                resp_headers = {
                    "X-ForkTape-Tape-Id": session.tape_id,
                    "X-ForkTape-Step": str(step_idx),
                    "X-ForkTape-Mode": "vcr-replay",
                }
                if is_stream:
                    chunks = synthesize_openai_sse_chunks(replayed_step.raw_response)

                    async def replay_sse_gen() -> AsyncIterator[bytes]:
                        for ch in chunks:
                            yield ch

                    return StreamingResponse(
                        replay_sse_gen(),
                        media_type="text/event-stream",
                        headers=resp_headers,
                    )
                return JSONResponse(content=replayed_step.raw_response, headers=resp_headers)

        # B) Live Record or Fork Execution
        applied_patch: OverridePatch | None = None
        outbound_body = body
        if (
            session.override_patch is not None
            and not session.override_patch.is_empty()
            and (session.fork_from_step is None or step_idx >= session.fork_from_step)
        ):
            applied_patch = session.override_patch
            outbound_body = apply_override_patch(body, applied_patch)

        upstream_override = request.headers.get("x-forktape-upstream")
        if upstream_override:
            upstream_base = _validate_http_url(upstream_override, "X-ForkTape-Upstream").rstrip("/")
        else:
            upstream_base = cfg.openai_upstream.rstrip("/")
        target_url = f"{upstream_base}/chat/completions"
        up_headers = _forward_upstream_headers(request)

        if is_stream:
            t0 = time.perf_counter()
            client = httpx.AsyncClient(timeout=120.0)
            up_req = client.build_request("POST", target_url, json=outbound_body, headers=up_headers)
            try:
                up_resp = await client.send(up_req, stream=True)
            except httpx.HTTPError as exc:
                await client.aclose()
                raise HTTPException(status_code=502, detail=f"Upstream connection error: {exc}") from exc

            if up_resp.status_code >= 400:
                err_bytes = await up_resp.aread()
                content_type = up_resp.headers.get("content-type", "application/json")
                status_code = up_resp.status_code
                await up_resp.aclose()
                await client.aclose()
                return Response(content=err_bytes, status_code=status_code, media_type=content_type)

            with session.lock:
                session.current_step_index = max(session.current_step_index + 1, step_idx)
                step_idx = session.current_step_index
                prev_req = session.prev_raw_request
                session.prev_raw_request = outbound_body

            assembler = OpenAISSEAssembler(fallback_model=str(outbound_body.get("model", "gpt-4o")))

            async def live_sse_stream() -> AsyncIterator[bytes]:
                first_token_ms: float | None = None
                utf8_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
                buffer = ""
                try:
                    async for raw_bytes in up_resp.aiter_bytes():
                        if first_token_ms is None:
                            first_token_ms = (time.perf_counter() - t0) * 1000.0
                        buffer += utf8_decoder.decode(raw_bytes, final=False)
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            assembler.feed_line(line)
                        yield raw_bytes
                    buffer += utf8_decoder.decode(b"", final=True)
                    if buffer.strip():
                        assembler.feed_line(buffer)

                    total_ms = (time.perf_counter() - t0) * 1000.0
                    built_resp, _, _, _ = assembler.build_response()
                    engine.build_and_store_live_step(
                        tape_id=session.tape_id,
                        step_index=step_idx,
                        provider="openai",
                        raw_request=outbound_body,
                        raw_response=built_resp,
                        prev_raw_request=prev_req,
                        ttft_ms=first_token_ms or total_ms,
                        duration_ms=total_ms,
                        override_patch=applied_patch,
                    )
                finally:
                    await up_resp.aclose()
                    await client.aclose()

            return StreamingResponse(
                live_sse_stream(),
                media_type="text/event-stream",
                headers={
                    "X-ForkTape-Tape-Id": session.tape_id,
                    "X-ForkTape-Step": str(step_idx),
                    "X-ForkTape-Mode": "live-fork" if applied_patch else "live-record",
                },
            )

        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=120.0) as client:
            up_resp = await client.post(target_url, json=outbound_body, headers=up_headers)
            if up_resp.status_code >= 400:
                return Response(
                    content=up_resp.content,
                    status_code=up_resp.status_code,
                    media_type=up_resp.headers.get("content-type", "application/json"),
                )
            resp_json = up_resp.json()
        total_ms = (time.perf_counter() - t0) * 1000.0

        with session.lock:
            session.current_step_index = max(session.current_step_index + 1, step_idx)
            step_idx = session.current_step_index
            prev_req = session.prev_raw_request
            session.prev_raw_request = outbound_body

        engine.build_and_store_live_step(
            tape_id=session.tape_id,
            step_index=step_idx,
            provider="openai",
            raw_request=outbound_body,
            raw_response=resp_json,
            prev_raw_request=prev_req,
            ttft_ms=total_ms * 0.4,
            duration_ms=total_ms,
            override_patch=applied_patch,
        )
        return JSONResponse(
            content=resp_json,
            headers={
                "X-ForkTape-Tape-Id": session.tape_id,
                "X-ForkTape-Step": str(step_idx),
                "X-ForkTape-Mode": "live-fork" if applied_patch else "live-record",
            },
        )

    @app.post("/v1/chat/completions", response_model=None)
    async def proxy_openai_chat_completions(request: Request) -> Response:
        return await _handle_openai_chat(request)

    @app.post("/s/{session_id}/v1/chat/completions", response_model=None)
    async def proxy_scoped_openai_chat_completions(session_id: str, request: Request) -> Response:
        return await _handle_openai_chat(request, path_session_id=session_id)

    # ------------------------------------------------------------------
    # 2. Anthropic Messages Endpoint (SSE & JSON)
    # ------------------------------------------------------------------
    async def _handle_anthropic_messages(request: Request, path_session_id: str | None = None) -> Response:
        body: dict[str, Any] = await request.json()
        is_stream = bool(body.get("stream", False))
        session_hdr = path_session_id or request.headers.get("x-forktape-session")
        session = engine.sessions.resolve_session(session_hdr)

        with session.lock:
            step_idx = session.current_step_index + 1
            prev_req = session.prev_raw_request

            try:
                vcr_step = engine.lookup_vcr_step(session, step_idx, body)
            except TrajectoryDivergenceError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc

            if vcr_step is not None:
                replayed_step = engine.record_replayed_step_into_session(session, vcr_step)
                resp_headers = {
                    "X-ForkTape-Tape-Id": session.tape_id,
                    "X-ForkTape-Step": str(step_idx),
                    "X-ForkTape-Mode": "vcr-replay",
                }
                if is_stream:
                    chunks = synthesize_anthropic_sse_chunks(replayed_step.raw_response)

                    async def anth_replay_gen() -> AsyncIterator[bytes]:
                        for ch in chunks:
                            yield ch

                    return StreamingResponse(
                        anth_replay_gen(),
                        media_type="text/event-stream",
                        headers=resp_headers,
                    )
                return JSONResponse(content=replayed_step.raw_response, headers=resp_headers)

        applied_patch: OverridePatch | None = None
        outbound_body = body
        if (
            session.override_patch is not None
            and not session.override_patch.is_empty()
            and (session.fork_from_step is None or step_idx >= session.fork_from_step)
        ):
            applied_patch = session.override_patch
            outbound_body = apply_override_patch(body, applied_patch)

        upstream_override = request.headers.get("x-forktape-upstream")
        if upstream_override:
            upstream_base = _validate_http_url(upstream_override, "X-ForkTape-Upstream").rstrip("/")
        else:
            upstream_base = cfg.anthropic_upstream.rstrip("/")
        target_url = f"{upstream_base}/messages" if upstream_base.endswith("/v1") else f"{upstream_base}/v1/messages"
        up_headers = _forward_upstream_headers(request)

        if is_stream:
            t0 = time.perf_counter()
            client = httpx.AsyncClient(timeout=120.0)
            up_req = client.build_request("POST", target_url, json=outbound_body, headers=up_headers)
            try:
                up_resp = await client.send(up_req, stream=True)
            except httpx.HTTPError as exc:
                await client.aclose()
                raise HTTPException(status_code=502, detail=f"Upstream connection error: {exc}") from exc

            if up_resp.status_code >= 400:
                err_bytes = await up_resp.aread()
                content_type = up_resp.headers.get("content-type", "application/json")
                status_code = up_resp.status_code
                await up_resp.aclose()
                await client.aclose()
                return Response(content=err_bytes, status_code=status_code, media_type=content_type)

            with session.lock:
                session.current_step_index = max(session.current_step_index + 1, step_idx)
                step_idx = session.current_step_index
                prev_req = session.prev_raw_request
                session.prev_raw_request = outbound_body

            assembler = AnthropicSSEAssembler(
                fallback_model=str(outbound_body.get("model", "claude-3-7-sonnet"))
            )

            async def live_anth_sse() -> AsyncIterator[bytes]:
                first_token_ms: float | None = None
                utf8_decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
                buffer = ""
                try:
                    async for raw_bytes in up_resp.aiter_bytes():
                        if first_token_ms is None:
                            first_token_ms = (time.perf_counter() - t0) * 1000.0
                        buffer += utf8_decoder.decode(raw_bytes, final=False)
                        while "\n" in buffer:
                            line, buffer = buffer.split("\n", 1)
                            assembler.feed_line(line)
                        yield raw_bytes
                    buffer += utf8_decoder.decode(b"", final=True)
                    if buffer.strip():
                        assembler.feed_line(buffer)

                    total_ms = (time.perf_counter() - t0) * 1000.0
                    built_resp, _, _, _ = assembler.build_response()
                    engine.build_and_store_live_step(
                        tape_id=session.tape_id,
                        step_index=step_idx,
                        provider="anthropic",
                        raw_request=outbound_body,
                        raw_response=built_resp,
                        prev_raw_request=prev_req,
                        ttft_ms=first_token_ms or total_ms,
                        duration_ms=total_ms,
                        override_patch=applied_patch,
                    )
                finally:
                    await up_resp.aclose()
                    await client.aclose()

            return StreamingResponse(
                live_anth_sse(),
                media_type="text/event-stream",
                headers={
                    "X-ForkTape-Tape-Id": session.tape_id,
                    "X-ForkTape-Step": str(step_idx),
                    "X-ForkTape-Mode": "live-fork" if applied_patch else "live-record",
                },
            )

        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=120.0) as client:
            up_resp = await client.post(target_url, json=outbound_body, headers=up_headers)
            if up_resp.status_code >= 400:
                return Response(
                    content=up_resp.content,
                    status_code=up_resp.status_code,
                    media_type=up_resp.headers.get("content-type", "application/json"),
                )
            resp_json = up_resp.json()
        total_ms = (time.perf_counter() - t0) * 1000.0

        with session.lock:
            session.current_step_index = max(session.current_step_index + 1, step_idx)
            step_idx = session.current_step_index
            prev_req = session.prev_raw_request
            session.prev_raw_request = outbound_body

        engine.build_and_store_live_step(
            tape_id=session.tape_id,
            step_index=step_idx,
            provider="anthropic",
            raw_request=outbound_body,
            raw_response=resp_json,
            prev_raw_request=prev_req,
            ttft_ms=total_ms * 0.4,
            duration_ms=total_ms,
            override_patch=applied_patch,
        )
        return JSONResponse(
            content=resp_json,
            headers={
                "X-ForkTape-Tape-Id": session.tape_id,
                "X-ForkTape-Step": str(step_idx),
                "X-ForkTape-Mode": "live-fork" if applied_patch else "live-record",
            },
        )

    @app.post("/v1/messages", response_model=None)
    async def proxy_anthropic_messages(request: Request) -> Response:
        return await _handle_anthropic_messages(request)

    @app.post("/s/{session_id}/v1/messages", response_model=None)
    async def proxy_scoped_anthropic_messages(session_id: str, request: Request) -> Response:
        return await _handle_anthropic_messages(request, path_session_id=session_id)

    # ------------------------------------------------------------------
    # 3. MCP HTTP JSON-RPC Proxy Endpoint
    # ------------------------------------------------------------------
    async def _handle_http_mcp(
        server_name: str, request: Request, path_session_id: str | None = None
    ) -> JSONResponse:
        rpc_body: dict[str, Any] = await request.json()
        sid = path_session_id or request.headers.get("x-forktape-session")
        session = engine.sessions.resolve_session(sid)
        upstream_url = request.headers.get("x-mcp-upstream")
        if not upstream_url:
            intercepted = mcp_interceptor.try_vcr_or_sandbox_intercept(
                session, server_name, rpc_body, transport="http"
            )
            if intercepted is not None:
                return JSONResponse(content=intercepted)
            raise HTTPException(status_code=400, detail="Missing X-MCP-Upstream header for live MCP call")

        validated_upstream = _validate_http_url(upstream_url, "X-MCP-Upstream")
        resp = await mcp_interceptor.forward_http_mcp(
            session=session,
            server_name=server_name,
            upstream_url=validated_upstream,
            rpc_request=rpc_body,
        )
        return JSONResponse(content=resp)

    @app.post("/mcp/{server_name}")
    async def proxy_http_mcp(server_name: str, request: Request) -> JSONResponse:
        return await _handle_http_mcp(server_name, request)

    @app.post("/s/{session_id}/mcp/{server_name}")
    async def proxy_scoped_http_mcp(
        session_id: str, server_name: str, request: Request
    ) -> JSONResponse:
        return await _handle_http_mcp(server_name, request, path_session_id=session_id)

    # ------------------------------------------------------------------
    # 4. Control Plane & Studio REST API
    # ------------------------------------------------------------------
    @app.get("/api/health")
    async def api_health() -> dict[str, Any]:
        tapes = tape_store.list_tapes(limit=500)
        return {
            "status": "ok",
            "version": __version__,
            "tape_count": len(tapes),
            "db_path": str(tape_store.db_path),
        }

    @app.post("/api/sessions/record")
    async def api_start_record(payload: StartRecordPayload) -> dict[str, Any]:
        state = engine.sessions.start_record_session(
            title=payload.title,
            agent_command=payload.agent_command,
            tape_id=payload.tape_id,
            tags=payload.tags,
        )
        return {"session_id": state.session_id, "tape_id": state.tape_id, "mode": state.mode}

    @app.post("/api/sessions/replay")
    async def api_start_replay(payload: StartReplayPayload) -> dict[str, Any]:
        try:
            state = engine.sessions.start_replay_or_fork_session(
                source_tape_id=payload.source_tape_id,
                fork_from_step=payload.fork_from_step,
                override_patch=payload.override_patch,
                strict_matching=payload.strict_matching,
                auto_fork_on_divergence=payload.auto_fork_on_divergence,
                title=payload.title,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            "session_id": state.session_id,
            "tape_id": state.tape_id,
            "source_tape_id": state.source_tape_id,
            "fork_from_step": state.fork_from_step,
            "mode": state.mode,
        }

    @app.post("/api/sessions/{session_id}/complete")
    async def api_complete_session(session_id: str, payload: CompleteSessionPayload) -> dict[str, Any]:
        updated = engine.sessions.complete_session(
            session_id=session_id,
            status=payload.status,
            failure_reason=payload.failure_reason,
        )
        if updated is None:
            raise HTTPException(status_code=404, detail=f"Tape/session '{session_id}' not found")
        return updated.model_dump()

    @app.get("/api/tapes")
    async def api_list_tapes(
        status: str | None = Query(default=None),
        limit: int = Query(default=100, ge=1, le=500),
    ) -> dict[str, Any]:
        tapes = tape_store.list_tapes(status=status, limit=limit)
        items = []
        for t in tapes:
            d = t.model_dump()
            d["total_tokens"] = t.total_tokens
            d["cache_hit_rate"] = t.cache_hit_rate
            items.append(d)
        return {"tapes": items}

    @app.get("/api/tapes/{tape_id}")
    async def api_get_tape_detail(tape_id: str) -> dict[str, Any]:
        tape = tape_store.get_tape(tape_id)
        if tape is None:
            raise HTTPException(status_code=404, detail=f"Tape '{tape_id}' not found")
        steps = tape_store.get_steps(tape_id)
        mcp_spans = tape_store.get_mcp_spans(tape_id)
        children = tape_store.get_children_tapes(tape_id)
        report = tape_store.get_biopsy_report(tape_id)

        try:
            from forktape.doctor.heuristics import scan_tape_anomalies

            findings = scan_tape_anomalies(tape, steps, mcp_spans)
        except ImportError:
            findings = []

        tape_dict = tape.model_dump()
        tape_dict["total_tokens"] = tape.total_tokens
        tape_dict["cache_hit_rate"] = tape.cache_hit_rate

        return {
            "tape": tape_dict,
            "steps": [s.model_dump() for s in steps],
            "mcp_spans": [sp.model_dump() for sp in mcp_spans],
            "children": [c.model_dump() for c in children],
            "heuristic_findings": [f.model_dump() for f in findings],
            "biopsy_report": report.model_dump() if report else None,
        }

    @app.delete("/api/tapes/{tape_id}")
    async def api_delete_tape(tape_id: str) -> dict[str, Any]:
        deleted = tape_store.delete_tape(tape_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"Tape '{tape_id}' not found")
        return {"deleted": True, "tape_id": tape_id}

    @app.post("/api/tapes/{tape_id}/fork")
    async def api_fork_step(tape_id: str, payload: StandaloneForkPayload) -> dict[str, Any]:
        try:
            child_tape, forked_step = await engine.fork_step_standalone(
                source_tape_id=tape_id,
                step_index=payload.step_index,
                patch=payload.patch,
                prefer_live_upstream=payload.prefer_live_upstream,
            )
            comparison = compare_tapes(tape_store, tape_id, child_tape.tape_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "forked_tape": child_tape.model_dump(),
            "forked_step": forked_step.model_dump(),
            "comparison": comparison.model_dump(),
        }

    @app.get("/api/tapes/{tape_id}/diff/{other_id}")
    async def api_diff_tapes(tape_id: str, other_id: str) -> dict[str, Any]:
        try:
            comparison = compare_tapes(tape_store, tape_id, other_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return comparison.model_dump()

    @app.post("/api/tapes/{tape_id}/doctor")
    async def api_run_doctor(tape_id: str) -> dict[str, Any]:
        from forktape.doctor.graph import run_counterfactual_biopsy

        try:
            report = await run_counterfactual_biopsy(tape_store, engine, tape_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return report.model_dump()

    @app.get("/api/tapes/{tape_id}/export-pytest")
    async def api_export_pytest(tape_id: str) -> dict[str, Any]:
        from forktape.compiler.pytest_gen import compile_tape_to_pytest_code

        try:
            code = compile_tape_to_pytest_code(tape_store, tape_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"tape_id": tape_id, "pytest_code": code}

    @app.get("/api/tapes/{tape_id}/bundle")
    async def api_export_bundle(tape_id: str) -> Response:
        try:
            bundle = tape_store.export_bundle(tape_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        safe_fname = SAFE_FILENAME_RE.sub("_", tape_id).strip("._") or "tape"
        return Response(
            content=bundle.model_dump_json(indent=2),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{safe_fname}.ftk.json"'},
        )

    @app.post("/api/tapes/import")
    async def api_import_bundle(bundle: TapeBundle) -> dict[str, Any]:
        imported = tape_store.import_bundle(bundle)
        return imported.model_dump()

    # ------------------------------------------------------------------
    # 5. Mount Embedded React Web Studio (Single-Binary Zero-Node UX)
    # ------------------------------------------------------------------
    static_dir = Path(__file__).resolve().parent.parent / "static_ui"
    if static_dir.exists() and (static_dir / "index.html").exists():
        assets_dir = static_dir / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="studio_assets")

        @app.get("/", include_in_schema=False)
        async def serve_studio_index() -> FileResponse:
            return FileResponse(static_dir / "index.html")

    return app
