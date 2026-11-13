"""Transparent MCP JSON-RPC Interceptor (`stdio` pipe wrapper & HTTP proxy) with VCR sandboxing."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from typing import Any

import httpx

from forktape.engine.fingerprint import canonical_request_hash, redact_payload
from forktape.engine.replayer import ReplayEngine, SessionState
from forktape.models.tape import MCPSpan

DESTRUCTIVE_TOOL_PREFIXES = (
    "delete_",
    "drop_",
    "remove_",
    "destroy_",
    "truncate_",
    "wipe_",
    "rollback_",
    "purge_",
    "send_email",
    "send_message",
    "transfer_",
    "charge_",
    "issue_full_wipe",
    "exec_shell",
    "run_bash",
)


def is_destructive_tool(tool_name: str) -> bool:
    lower = tool_name.lower()
    return any(lower.startswith(p) or p in lower for p in DESTRUCTIVE_TOOL_PREFIXES)


class MCPInterceptor:
    """Intercepts MCP JSON-RPC 2.0 calls for recording, zero-side-effect VCR replay, and write sandboxing."""

    def __init__(self, replay_engine: ReplayEngine) -> None:
        self.engine = replay_engine

    def _effective_step_index(self, session: SessionState) -> int:
        db_max = self.engine.store.get_max_step_index(session.tape_id)
        return max(1, session.current_step_index, db_max)

    def try_vcr_or_sandbox_intercept(
        self,
        session: SessionState,
        server_name: str,
        rpc_request: dict[str, Any],
        transport: str = "stdio",
    ) -> dict[str, Any] | None:
        """Return a mocked/replayed JSON-RPC response if step is replaying or sandboxed, else None."""
        method = str(rpc_request.get("method") or "")
        rpc_id = rpc_request.get("id")
        params = rpc_request.get("params") or {}

        if method not in ("tools/call", "tools/list", "resources/read"):
            return None

        tool_name = str(params.get("name") or params.get("uri") or method)
        arguments = (
            params.get("arguments")
            if isinstance(params.get("arguments"), dict)
            else (params if isinstance(params, dict) else {})
        )
        step_idx = self._effective_step_index(session)

        # 1. Explicit mock override in OverridePatch
        patch = session.override_patch
        if (
            method == "tools/call"
            and patch
            and tool_name in patch.mock_tool_outputs
            and not session.is_replaying_step(step_idx)
        ):
            mock_val = patch.mock_tool_outputs[tool_name]
            self.engine.store.append_mcp_span(
                MCPSpan(
                    tape_id=session.tape_id,
                    step_index=step_idx,
                    server_name=server_name,
                    transport=transport,
                    method=method,
                    tool_name=tool_name,
                    arguments=redact_payload(arguments),
                    result=mock_val,
                    replayed=True,
                    duration_ms=0.3,
                )
            )
            return {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(mock_val, ensure_ascii=False)}],
                    "isError": False,
                },
            }

        # 2. Deterministic VCR Replay for steps prior to fork_from_step (Zero DB/API Side Effects!)
        if session.source_tape_id and session.is_replaying_step(step_idx):
            arg_hash = canonical_request_hash({"method": method, "tool": tool_name, "args": arguments})
            candidates = self.engine.store.get_mcp_spans(session.source_tape_id)
            unconsumed = [sp for sp in candidates if sp.span_id not in session.consumed_mcp_spans]
            pool = unconsumed or candidates

            matched: MCPSpan | None = None
            # Prefer exact (step_index, tool_name, arg_hash) match first
            for sp in pool:
                sp_hash = canonical_request_hash({"method": sp.method, "tool": sp.tool_name, "args": sp.arguments})
                if sp.step_index == step_idx and sp.tool_name == tool_name and sp_hash == arg_hash:
                    matched = sp
                    break
            if matched is None:
                for sp in pool:
                    sp_hash = canonical_request_hash({"method": sp.method, "tool": sp.tool_name, "args": sp.arguments})
                    if sp.tool_name == tool_name and sp_hash == arg_hash:
                        matched = sp
                        break
            if matched is None:
                matched = next((sp for sp in pool if sp.tool_name == tool_name and sp.method == method), None)

            if matched is not None:
                session.consumed_mcp_spans.add(matched.span_id)
                self.engine.store.append_mcp_span(
                    MCPSpan(
                        tape_id=session.tape_id,
                        step_index=step_idx,
                        server_name=server_name,
                        transport=transport,
                        method=method,
                        tool_name=tool_name,
                        arguments=redact_payload(arguments),
                        result=matched.result,
                        is_error=matched.is_error,
                        replayed=True,
                        duration_ms=0.5,
                    )
                )
                result_payload = matched.result
                if method != "tools/call" or (isinstance(result_payload, dict) and "content" in result_payload):
                    rpc_result = result_payload
                else:
                    rpc_result = {
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    result_payload
                                    if isinstance(result_payload, str)
                                    else json.dumps(result_payload, ensure_ascii=False)
                                ),
                            }
                        ],
                        "isError": matched.is_error,
                    }
                return {"jsonrpc": "2.0", "id": rpc_id, "result": rpc_result}

        # 3. Dry-Run Write Guardrail during live forks
        if method == "tools/call" and patch and patch.dry_run_write_tools and is_destructive_tool(tool_name):
            sandbox_msg = {
                "_forktape_dry_run_blocked": True,
                "tool": tool_name,
                "intercepted_arguments": redact_payload(arguments),
                "note": "Blocked destructive side-effect during counterfactual fork.",
            }
            self.engine.store.append_mcp_span(
                MCPSpan(
                    tape_id=session.tape_id,
                    step_index=step_idx,
                    server_name=server_name,
                    transport=transport,
                    method=method,
                    tool_name=tool_name,
                    arguments=redact_payload(arguments),
                    result=sandbox_msg,
                    is_error=False,
                    replayed=False,
                    duration_ms=0.2,
                )
            )
            return {
                "jsonrpc": "2.0",
                "id": rpc_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(sandbox_msg, ensure_ascii=False)}],
                    "isError": False,
                },
            }

        return None

    def record_live_mcp_call(
        self,
        session: SessionState,
        server_name: str,
        rpc_request: dict[str, Any],
        rpc_response: dict[str, Any],
        duration_ms: float,
        transport: str = "stdio",
    ) -> MCPSpan | None:
        method = str(rpc_request.get("method") or "")
        if method not in ("tools/call", "resources/read", "tools/list"):
            return None
        params = rpc_request.get("params") or {}
        tool_name = str(params.get("name") or params.get("uri") or method)
        arguments = params.get("arguments") if isinstance(params.get("arguments"), dict) else params
        is_error = "error" in rpc_response or bool((rpc_response.get("result") or {}).get("isError"))
        result_val = rpc_response.get("error") if "error" in rpc_response else rpc_response.get("result")

        span = MCPSpan(
            tape_id=session.tape_id,
            step_index=self._effective_step_index(session),
            server_name=server_name,
            transport=transport,
            method=method,
            tool_name=tool_name,
            arguments=redact_payload(arguments),
            result=redact_payload(result_val),
            is_error=is_error,
            replayed=False,
            duration_ms=round(duration_ms, 2),
        )
        return self.engine.store.append_mcp_span(span)

    async def forward_http_mcp(
        self,
        session: SessionState,
        server_name: str,
        upstream_url: str,
        rpc_request: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        intercepted = self.try_vcr_or_sandbox_intercept(
            session, server_name, rpc_request, transport="http"
        )
        if intercepted is not None:
            return intercepted

        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(upstream_url, json=rpc_request, headers=headers or {})
            resp.raise_for_status()
            rpc_response = resp.json()
        dur_ms = (time.perf_counter() - t0) * 1000.0
        self.record_live_mcp_call(
            session, server_name, rpc_request, rpc_response, dur_ms, transport="http"
        )
        return rpc_response


async def run_stdio_mcp_bridge(
    replay_engine: ReplayEngine,
    session_id: str | None,
    server_name: str,
    command: list[str],
) -> int:
    """Transparently wrap a child MCP `stdio` server process for recording & VCR replay."""
    interceptor = MCPInterceptor(replay_engine)
    proc = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=sys.stderr,
        limit=16 * 1024 * 1024,
    )
    assert proc.stdin is not None and proc.stdout is not None

    pending_requests: dict[Any, tuple[dict[str, Any], float]] = {}

    async def pump_client_to_server() -> None:
        loop = asyncio.get_running_loop()
        while True:
            line = await loop.run_in_executor(None, sys.stdin.buffer.readline)
            if not line:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
                break
            try:
                rpc_req = json.loads(line.decode("utf-8"))
            except ValueError:
                try:
                    proc.stdin.write(line)
                    await proc.stdin.drain()
                except (BrokenPipeError, ConnectionResetError):
                    break
                continue

            session = replay_engine.sessions.resolve_session(session_id)
            intercepted = interceptor.try_vcr_or_sandbox_intercept(
                session, server_name, rpc_req, transport="stdio"
            )
            if intercepted is not None:
                out_bytes = (json.dumps(intercepted, ensure_ascii=False) + "\n").encode("utf-8")
                sys.stdout.buffer.write(out_bytes)
                sys.stdout.buffer.flush()
                continue

            if "id" in rpc_req:
                pending_requests[rpc_req["id"]] = (rpc_req, time.perf_counter())

            try:
                proc.stdin.write(line)
                await proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                break

    async def pump_server_to_client() -> None:
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            try:
                rpc_resp = json.loads(line.decode("utf-8"))
                resp_id = rpc_resp.get("id")
                if resp_id in pending_requests:
                    rpc_req, t0 = pending_requests.pop(resp_id)
                    dur_ms = (time.perf_counter() - t0) * 1000.0
                    session = replay_engine.sessions.resolve_session(session_id)
                    interceptor.record_live_mcp_call(
                        session, server_name, rpc_req, rpc_resp, dur_ms, transport="stdio"
                    )
            except ValueError:
                pass

            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()

    t_in = asyncio.create_task(pump_client_to_server())
    t_out = asyncio.create_task(pump_server_to_client())
    done, pending = await asyncio.wait({t_in, t_out}, return_when=asyncio.FIRST_COMPLETED)
    for p in pending:
        p.cancel()
    return await proc.wait()
