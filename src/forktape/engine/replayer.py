"""Dual-mode deterministic VCR replay engine and standalone counterfactual step forker."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from forktape.config import ForkTapeConfig, estimate_cost_usd
from forktape.engine.fingerprint import (
    analyze_prefix_cache_continuity,
    canonical_request_hash,
    compute_structural_similarity,
    estimate_step_token_breakdown,
    estimate_tokens,
    redact_payload,
)
from forktape.engine.patcher import apply_override_patch
from forktape.interceptor.sse_assembler import (
    extract_from_anthropic_json,
    extract_from_openai_json,
)
from forktape.models.db import TapeStore
from forktape.models.tape import (
    OverridePatch,
    TapeRecord,
    TapeStatus,
    TapeStep,
    new_id,
)


class TrajectoryDivergenceError(RuntimeError):
    """Raised in strict replay mode when an incoming request diverges from the recorded tape."""

    def __init__(self, step_index: int, similarity: float, expected_hash: str, actual_hash: str) -> None:
        self.step_index = step_index
        self.similarity = similarity
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash
        super().__init__(
            f"Trajectory diverged at step {step_index}: structural similarity={similarity:.3f} "
            f"(expected hash {expected_hash[:10]}..., got {actual_hash[:10]}...)"
        )


class StepExecutorProtocol(Protocol):
    async def execute_step(
        self,
        provider: str,
        request: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> tuple[dict[str, Any], float, float]:
        """Return (raw_response_dict, ttft_ms, total_duration_ms)."""
        ...


@dataclass
class SessionState:
    """Tracks an active Record, Replay, or Time-Travel Fork session."""

    session_id: str
    mode: str = "record"  # "record" | "replay" | "fork"
    tape_id: str = field(default_factory=lambda: new_id("tape"))
    source_tape_id: str | None = None
    fork_from_step: int | None = None
    override_patch: OverridePatch | None = None
    strict_matching: bool = False
    auto_fork_on_divergence: bool = True
    current_step_index: int = 0
    prev_raw_request: dict[str, Any] | None = None
    diverged_live: bool = False
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    consumed_mcp_spans: set[str] = field(default_factory=set, repr=False)

    def is_replaying_step(self, next_step_index: int) -> bool:
        if self.diverged_live:
            return False
        if self.mode == "replay":
            return True
        if self.mode == "fork" and self.fork_from_step is not None:
            return next_step_index < self.fork_from_step
        return False


class SessionManager:
    """Thread-safe & cross-process hydrated registry of active ForkTape proxy sessions."""

    def __init__(self, store: TapeStore) -> None:
        self.store = store
        self._lock = threading.RLock()
        self._sessions: dict[str, SessionState] = {}
        self._default_session_id: str | None = None

    def start_record_session(
        self,
        title: str = "Recorded Agent Run",
        agent_command: str = "",
        tape_id: str | None = None,
        tags: list[str] | None = None,
        set_default: bool = True,
    ) -> SessionState:
        tid = tape_id or new_id("tape")
        tape = TapeRecord(
            tape_id=tid,
            title=title,
            agent_command=agent_command,
            status=TapeStatus.RUNNING,
            tags=tags or [],
            metadata={"_session": {"mode": "record"}},
        )
        self.store.create_tape(tape)
        state = SessionState(session_id=tid, mode="record", tape_id=tid)
        with self._lock:
            self._sessions[tid] = state
            if set_default:
                self._default_session_id = tid
        return state

    def start_replay_or_fork_session(
        self,
        source_tape_id: str,
        fork_from_step: int | None = None,
        override_patch: OverridePatch | None = None,
        strict_matching: bool = False,
        auto_fork_on_divergence: bool = True,
        title: str | None = None,
        set_default: bool = True,
    ) -> SessionState:
        src = self.store.get_tape(source_tape_id)
        if src is None:
            raise KeyError(f"Source tape '{source_tape_id}' not found")

        is_fork = fork_from_step is not None
        new_tid = new_id("tape")
        mode = "fork" if is_fork else "replay"
        label = (override_patch.label if override_patch and override_patch.label else None) or (
            f"Fork @ Step {fork_from_step}" if is_fork else "Deterministic Replay"
        )
        child_meta = dict(src.metadata)
        child_meta["_session"] = {
            "mode": mode,
            "source_tape_id": source_tape_id,
            "fork_from_step": fork_from_step,
            "override_patch": override_patch.model_dump() if override_patch else None,
            "strict_matching": strict_matching,
            "auto_fork_on_divergence": auto_fork_on_divergence,
        }
        child_tape = TapeRecord(
            tape_id=new_tid,
            parent_tape_id=source_tape_id,
            fork_from_step=fork_from_step,
            fork_label=label,
            title=title or f"{src.title} ({label})",
            agent_command=src.agent_command,
            status=TapeStatus.RUNNING,
            tags=list(src.tags),
            metadata=child_meta,
        )
        self.store.create_tape(child_tape)

        state = SessionState(
            session_id=new_tid,
            mode=mode,
            tape_id=new_tid,
            source_tape_id=source_tape_id,
            fork_from_step=fork_from_step,
            override_patch=override_patch,
            strict_matching=strict_matching,
            auto_fork_on_divergence=auto_fork_on_divergence,
        )
        with self._lock:
            self._sessions[new_tid] = state
            if set_default:
                self._default_session_id = new_tid
        return state

    def resolve_session(self, session_id: str | None = None) -> SessionState:
        with self._lock:
            if session_id and session_id in self._sessions:
                return self._sessions[session_id]
            if session_id:
                existing_tape = self.store.get_tape(session_id)
                if existing_tape is not None:
                    sess_meta = existing_tape.metadata.get("_session") or {}
                    raw_patch = sess_meta.get("override_patch")
                    hydrated = SessionState(
                        session_id=session_id,
                        mode=str(sess_meta.get("mode") or ("fork" if existing_tape.fork_from_step else "record")),
                        tape_id=existing_tape.tape_id,
                        source_tape_id=sess_meta.get("source_tape_id") or existing_tape.parent_tape_id,
                        fork_from_step=sess_meta.get("fork_from_step", existing_tape.fork_from_step),
                        override_patch=OverridePatch.model_validate(raw_patch) if isinstance(raw_patch, dict) else None,
                        strict_matching=bool(sess_meta.get("strict_matching", False)),
                        auto_fork_on_divergence=bool(sess_meta.get("auto_fork_on_divergence", True)),
                        current_step_index=self.store.get_max_step_index(existing_tape.tape_id),
                    )
                    self._sessions[session_id] = hydrated
                    return hydrated
            if self._default_session_id and self._default_session_id in self._sessions:
                return self._sessions[self._default_session_id]
        return self.start_record_session(title="Auto-Captured Agent Session")

    def complete_session(
        self,
        session_id: str,
        status: TapeStatus = TapeStatus.SUCCESS,
        failure_reason: str | None = None,
    ) -> TapeRecord | None:
        with self._lock:
            state = self._sessions.get(session_id)
        target_tape_id = state.tape_id if state else session_id
        self.store.update_tape_status(target_tape_id, status, failure_reason)
        return self.store.refresh_tape_Rollups(target_tape_id)


def remap_response_tool_ids_for_client(recorded_response: dict[str, Any]) -> dict[str, Any]:
    """Ensure replayed responses remain valid clean copies."""
    return copy.deepcopy(recorded_response)


def _flatten_content_to_str(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(_flatten_content_to_str(item["content"]))
            else:
                parts.append(str(item))
        return " ".join(parts)
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False)


def _extract_all_tool_outputs_from_messages(messages: list[Any]) -> list[str]:
    outputs: list[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        if m.get("role") == "tool":
            outputs.append(_flatten_content_to_str(m.get("content")))
        elif isinstance(m.get("content"), list):
            for b in m["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    outputs.append(_flatten_content_to_str(b.get("content")))
    return outputs


class SmartOfflineStepExecutor:
    """Deterministic counterfactual evaluator when no live API key is configured.

    Models realistic LLM context-engineering mechanics (Tool Schema Attention Dilution,
    Context Middle Loss, Prefix Cache hits, and Hallucination Recovery) so `forktape demo`,
    offline `pytest` suites, and `ForkBench` run reproducibly with zero external API keys!
    """

    async def execute_step(
        self,
        provider: str,
        request: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> tuple[dict[str, Any], float, float]:
        t0 = time.perf_counter()
        model = str(request.get("model") or ("claude-3-7-sonnet" if provider == "anthropic" else "gpt-4o"))
        tools = request.get("tools") or []
        messages = request.get("messages") or []
        breakdown = estimate_step_token_breakdown(request)
        prompt_toks = breakdown.total

        sys_text = _flatten_content_to_str(request.get("system"))
        for m in messages:
            if isinstance(m, dict) and m.get("role") in ("system", "developer"):
                sys_text += " " + _flatten_content_to_str(m.get("content"))

        has_static_prefix = "current time:" not in sys_text.lower() and "timestamp:" not in sys_text.lower()
        cached_toks = int(prompt_toks * 0.88) if has_static_prefix and len(messages) > 2 else 0

        user_query = " ".join(
            _flatten_content_to_str(m.get("content"))
            for m in messages
            if isinstance(m, dict) and m.get("role") == "user"
        ).lower()
        tool_output_list = _extract_all_tool_outputs_from_messages(messages)
        all_tool_outputs = " ".join(tool_output_list)

        has_context_bloat = any(
            len(out) >= 5000 and "_forktape_compacted" not in out for out in tool_output_list
        )

        from forktape.doctor.heuristics import (
            _extract_preceding_tool_messages,
            _normalize_empty_check,
        )

        preceding_outs = _extract_preceding_tool_messages(request)
        last_tool_empty = bool(preceding_outs) and all(_normalize_empty_check(x) for x in preceding_outs)
        has_grounding_guard = (
            "never fabricate" in sys_text.lower()
            or "if empty" in sys_text.lower()
            or "do not hallucinate" in sys_text.lower()
            or "verify empty" in sys_text.lower()
        )

        chosen_tool: dict[str, Any] | None = None
        if tools:
            best_score = -1
            for t in tools:
                fn = t.get("function") or t
                t_name = str(fn.get("name", ""))
                t_desc = str(fn.get("description", "")).lower()
                words = set(t_name.lower().replace("_", " ").split())
                score = sum(3 for w in words if len(w) > 2 and w in user_query)
                score += sum(1 for w in t_desc.split() if len(w) > 4 and w in user_query)
                if ("deprecated" in t_desc or "internal_admin" in t_name) and len(tools) <= 12:
                    score -= 10
                if score > best_score:
                    best_score = score
                    chosen_tool = fn

        completion_toks = 65
        elapsed_ms = max(8.5, (time.perf_counter() - t0) * 1000.0 + 14.0)

        if last_tool_empty and has_grounding_guard:
            reply_text = (
                "No matching records were returned by the search tool. "
                "Escalating or requesting clarification instead of assuming missing values."
            )
            resp = _build_openai_or_anthropic_text(provider, model, reply_text, prompt_toks, cached_toks, completion_toks)
            return resp, 6.2, elapsed_ms

        if chosen_tool and not last_tool_empty and len(tools) <= 15 and not has_context_bloat:
            t_name = str(chosen_tool.get("name", "resolved_action"))
            params_schema = (chosen_tool.get("parameters") or chosen_tool.get("input_schema") or {}).get(
                "properties", {}
            )
            synth_args = _synthesize_grounded_args(params_schema, user_query, all_tool_outputs)
            resp = _build_openai_or_anthropic_tool(
                provider, model, t_name, synth_args, prompt_toks, cached_toks, completion_toks
            )
            return resp, 5.8, elapsed_ms

        if not has_context_bloat and ("_forktape_compacted" in all_tool_outputs or len(tools) == 0):
            reply_text = (
                "Verified constraints from compacted context and completed the requested operation accurately."
            )
            resp = _build_openai_or_anthropic_text(provider, model, reply_text, prompt_toks, cached_toks, completion_toks)
            return resp, 6.0, elapsed_ms

        fallback_name = str((chosen_tool or {}).get("name") or "ambiguous_fallback_tool")
        resp = _build_openai_or_anthropic_tool(
            provider,
            model,
            fallback_name,
            {"warning": "unpruned_or_uncompacted_context"},
            prompt_toks,
            cached_toks,
            completion_toks,
        )
        return resp, 9.5, elapsed_ms


def _synthesize_grounded_args(
    properties: dict[str, Any],
    user_query: str,
    tool_outputs: str,
) -> dict[str, Any]:
    import re as _re

    args: dict[str, Any] = {}
    order_match = _re.search(r"\b(ord[-_]\d+|inc[-_]\d+|usr[-_]\d+)\b", user_query + " " + tool_outputs, _re.I)
    amount_match = _re.search(r"\$?(\d+(?:\.\d{1,2})?)", user_query + " " + tool_outputs)

    for prop_name, spec in properties.items():
        ptype = spec.get("type", "string") if isinstance(spec, dict) else "string"
        pname_l = prop_name.lower()
        if "id" in pname_l and order_match:
            args[prop_name] = order_match.group(1).upper()
        elif ("amount" in pname_l or "price" in pname_l or "limit" in pname_l) and ptype in ("number", "integer"):
            val = float(amount_match.group(1)) if amount_match else 25.0
            args[prop_name] = int(val) if ptype == "integer" else val
        elif ptype == "boolean":
            args[prop_name] = True
        elif ptype in ("number", "integer"):
            args[prop_name] = 1
        else:
            enum_vals = spec.get("enum") if isinstance(spec, dict) else None
            args[prop_name] = enum_vals[0] if isinstance(enum_vals, list) and enum_vals else "verified"
    return args


def _build_openai_or_anthropic_text(
    provider: str,
    model: str,
    text: str,
    prompt_toks: int,
    cached_toks: int,
    completion_toks: int,
) -> dict[str, Any]:
    if provider == "anthropic":
        return {
            "id": f"msg_{uuid.uuid4().hex[:10]}",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": max(0, prompt_toks - cached_toks),
                "cache_read_input_tokens": cached_toks,
                "output_tokens": completion_toks,
                "prompt_tokens": prompt_toks,
            },
        }
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:10]}",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_toks,
            "completion_tokens": completion_toks,
            "total_tokens": prompt_toks + completion_toks,
            "prompt_tokens_details": {"cached_tokens": cached_toks},
        },
    }


def _build_openai_or_anthropic_tool(
    provider: str,
    model: str,
    tool_name: str,
    arguments: dict[str, Any],
    prompt_toks: int,
    cached_toks: int,
    completion_toks: int,
) -> dict[str, Any]:
    tc_id = f"call_{uuid.uuid4().hex[:10]}"
    if provider == "anthropic":
        return {
            "id": f"msg_{uuid.uuid4().hex[:10]}",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "tool_use", "id": tc_id, "name": tool_name, "input": arguments}],
            "stop_reason": "tool_use",
            "usage": {
                "input_tokens": max(0, prompt_toks - cached_toks),
                "cache_read_input_tokens": cached_toks,
                "output_tokens": completion_toks,
                "prompt_tokens": prompt_toks,
            },
        }
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:10]}",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tc_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(arguments, ensure_ascii=False),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_toks,
            "completion_tokens": completion_toks,
            "total_tokens": prompt_toks + completion_toks,
            "prompt_tokens_details": {"cached_tokens": cached_toks},
        },
    }


class LiveUpstreamStepExecutor:
    """Executes mutated step requests against live OpenAI-compatible or Anthropic endpoints."""

    def __init__(self, config: ForkTapeConfig) -> None:
        self.config = config

    async def execute_step(
        self,
        provider: str,
        request: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> tuple[dict[str, Any], float, float]:
        payload = copy.deepcopy(request)
        payload["stream"] = False
        payload.pop("stream_options", None)

        model_name = str(payload.get("model") or "").lower()
        if "reasoner" in model_name:
            payload.pop("temperature", None)
        if "reasoner" in model_name or model_name.startswith("deepseek"):
            msg_list = payload.get("messages") or []
            for m_idx, m in enumerate(msg_list):
                if not isinstance(m, dict) or m.get("role") != "assistant":
                    continue
                next_role = (
                    msg_list[m_idx + 1].get("role")
                    if m_idx + 1 < len(msg_list) and isinstance(msg_list[m_idx + 1], dict)
                    else None
                )
                if (m.get("tool_calls") or next_role != "user") and m.get("reasoning_content") is None:
                    m["reasoning_content"] = ""

        req_headers = {"Content-Type": "application/json"}
        if headers:
            req_headers.update(headers)

        if provider == "anthropic":
            base = self.config.anthropic_upstream.rstrip("/")
            if model_name.startswith("deepseek") and base == "https://api.anthropic.com":
                base = "https://api.deepseek.com/anthropic"
            url = f"{base}/messages" if base.endswith("/v1") else f"{base}/v1/messages"
            api_key = (
                os.environ.get("DEEPSEEK_API_KEY", "")
                if ("deepseek" in base or model_name.startswith("deepseek"))
                else ""
            ) or os.environ.get("ANTHROPIC_API_KEY", "")
            if api_key and "x-api-key" not in {k.lower() for k in req_headers}:
                req_headers["x-api-key"] = api_key
                req_headers.setdefault("anthropic-version", "2023-06-01")
        else:
            base = self.config.openai_upstream.rstrip("/")
            if model_name.startswith("deepseek") and base == "https://api.openai.com/v1":
                base = "https://api.deepseek.com/v1"
            url = f"{base}/chat/completions"
            api_key = (
                os.environ.get("DEEPSEEK_API_KEY", "")
                if "deepseek" in base or model_name.startswith("deepseek")
                else ""
            ) or os.environ.get("OPENAI_API_KEY", "")
            if api_key and "authorization" not in {k.lower() for k in req_headers}:
                req_headers["Authorization"] = f"Bearer {api_key}"

        t0 = time.perf_counter()
        is_local = "127.0.0.1" in url or "localhost" in url
        max_attempts = 1 if is_local else 3
        last_exc: Exception | None = None
        data: dict[str, Any] = {}

        for attempt in range(1, max_attempts + 1):
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    resp = await client.post(url, json=payload, headers=req_headers)
                    if not is_local and resp.status_code in (502, 503, 504) and attempt < max_attempts:
                        await asyncio.sleep(0.5 * attempt)
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    last_exc = None
                    break
            except httpx.TransportError as exc:
                last_exc = exc
                if attempt < max_attempts:
                    await asyncio.sleep(0.5 * attempt)
                else:
                    raise

        if last_exc is not None:
            raise last_exc
        dur_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return data, round(dur_ms * 0.35, 2), dur_ms


def has_live_api_credentials(provider: str, config: ForkTapeConfig) -> bool:
    if provider == "anthropic":
        return (
            bool(os.environ.get("ANTHROPIC_API_KEY"))
            or bool(os.environ.get("DEEPSEEK_API_KEY"))
            or "127.0.0.1" in config.anthropic_upstream
            or "localhost" in config.anthropic_upstream
        )
    return (
        bool(os.environ.get("OPENAI_API_KEY"))
        or bool(os.environ.get("DEEPSEEK_API_KEY"))
        or "127.0.0.1" in config.openai_upstream
        or "localhost" in config.openai_upstream
    )


class ReplayEngine:
    """Coordinates VCR fast-forward step lookup, divergence detection, and standalone step forks."""

    def __init__(
        self,
        store: TapeStore,
        config: ForkTapeConfig | None = None,
        offline_executor: StepExecutorProtocol | None = None,
    ) -> None:
        self.store = store
        self.config = config or ForkTapeConfig()
        self.sessions = SessionManager(store)
        self.offline_executor: StepExecutorProtocol = offline_executor or SmartOfflineStepExecutor()
        self.live_executor = LiveUpstreamStepExecutor(self.config)

    def lookup_vcr_step(
        self,
        session: SessionState,
        step_index: int,
        incoming_request: dict[str, Any],
    ) -> TapeStep | None:
        """Attempt <3ms deterministic VCR lookup for `step_index` from `session.source_tape_id`."""
        if not session.source_tape_id or not session.is_replaying_step(step_index):
            return None

        recorded_step = self.store.get_step(session.source_tape_id, step_index)
        if recorded_step is None:
            if session.strict_matching:
                raise TrajectoryDivergenceError(
                    step_index=step_index,
                    similarity=0.0,
                    expected_hash="<END_OF_TAPE>",
                    actual_hash=canonical_request_hash(incoming_request),
                )
            session.diverged_live = True
            return None

        inc_hash = canonical_request_hash(incoming_request)
        if inc_hash == recorded_step.canonical_hash:
            return recorded_step

        similarity = compute_structural_similarity(recorded_step.raw_request, incoming_request)
        if similarity >= self.config.similarity_threshold:
            return recorded_step

        if session.strict_matching:
            raise TrajectoryDivergenceError(
                step_index=step_index,
                similarity=similarity,
                expected_hash=recorded_step.canonical_hash,
                actual_hash=inc_hash,
            )

        if session.auto_fork_on_divergence:
            session.diverged_live = True
            return None

        return recorded_step

    def record_replayed_step_into_session(
        self,
        session: SessionState,
        source_step: TapeStep,
        replay_duration_ms: float = 1.1,
    ) -> TapeStep:
        """Persist a fast-forwarded VCR step onto the new replay/fork child tape."""
        cloned = TapeStep(
            tape_id=session.tape_id,
            step_index=source_step.step_index,
            provider=source_step.provider,
            model=source_step.model,
            canonical_hash=source_step.canonical_hash,
            token_breakdown=source_step.token_breakdown,
            prefix_cache_ratio=source_step.prefix_cache_ratio,
            prefix_break_offset=source_step.prefix_break_offset,
            prefix_break_snippet=source_step.prefix_break_snippet,
            raw_request=source_step.raw_request,
            raw_response=remap_response_tool_ids_for_client(source_step.raw_response),
            assembled_text=source_step.assembled_text,
            reasoning_text=source_step.reasoning_text,
            tool_calls=source_step.tool_calls,
            override_patch=None,
            replayed=True,
            ttft_ms=round(replay_duration_ms * 0.5, 2),
            duration_ms=round(replay_duration_ms, 2),
            prompt_tokens=source_step.prompt_tokens,
            cached_prompt_tokens=source_step.cached_prompt_tokens,
            completion_tokens=source_step.completion_tokens,
            cost_usd=0.0,
            anomaly_flags=[],
        )
        session.prev_raw_request = source_step.raw_request
        session.current_step_index = max(session.current_step_index, source_step.step_index)
        return self.store.append_step(cloned)

    def build_and_store_live_step(
        self,
        tape_id: str,
        step_index: int,
        provider: str,
        raw_request: dict[str, Any],
        raw_response: dict[str, Any],
        prev_raw_request: dict[str, Any] | None,
        ttft_ms: float,
        duration_ms: float,
        override_patch: OverridePatch | None = None,
    ) -> TapeStep:
        """Normalize, analyze Prefix KV Cache LCP, compute cost, redact secrets, and persist a live step."""
        clean_req = redact_payload(raw_request)
        clean_resp = redact_payload(raw_response)
        model = str(clean_resp.get("model") or clean_req.get("model") or "gpt-4o")

        if provider == "anthropic":
            text, reasoning, invocations, p_tok, c_tok, o_tok = extract_from_anthropic_json(clean_resp)
        else:
            text, reasoning, invocations, p_tok, c_tok, o_tok = extract_from_openai_json(clean_resp)

        breakdown = estimate_step_token_breakdown(clean_req)
        if p_tok <= 0:
            p_tok = breakdown.total
        if o_tok <= 0:
            o_tok = estimate_tokens(text + json.dumps([tc.model_dump() for tc in invocations]))

        cache_analysis = analyze_prefix_cache_continuity(prev_raw_request, clean_req)
        usage_obj = clean_resp.get("usage") if isinstance(clean_resp.get("usage"), dict) else {}
        has_explicit_cache_usage = (
            (
                isinstance(usage_obj.get("prompt_tokens_details"), dict)
                and "cached_tokens" in usage_obj["prompt_tokens_details"]
            )
            or ("prompt_cache_hit_tokens" in usage_obj)
            or ("cache_read_input_tokens" in usage_obj)
        )
        if (
            not has_explicit_cache_usage
            and c_tok <= 0
            and prev_raw_request is not None
            and cache_analysis.ratio >= 0.90
        ):
            c_tok = int(p_tok * cache_analysis.ratio * 0.85)

        cost = estimate_cost_usd(model, p_tok, c_tok, o_tok)
        step = TapeStep(
            tape_id=tape_id,
            step_index=step_index,
            provider=provider,
            model=model,
            canonical_hash=canonical_request_hash(clean_req),
            token_breakdown=breakdown,
            prefix_cache_ratio=cache_analysis.ratio,
            prefix_break_offset=cache_analysis.break_offset,
            prefix_break_snippet=cache_analysis.break_snippet,
            raw_request=clean_req,
            raw_response=clean_resp,
            assembled_text=text,
            reasoning_text=reasoning,
            tool_calls=invocations,
            override_patch=override_patch,
            replayed=False,
            ttft_ms=round(ttft_ms, 2),
            duration_ms=round(duration_ms, 2),
            prompt_tokens=p_tok,
            cached_prompt_tokens=c_tok,
            completion_tokens=o_tok,
            cost_usd=cost,
        )
        return self.store.append_step(step)

    async def fork_step_standalone(
        self,
        source_tape_id: str,
        step_index: int,
        patch: OverridePatch,
        prefer_live_upstream: bool = False,
        executor_override: StepExecutorProtocol | None = None,
        final_status: TapeStatus = TapeStatus.SUCCESS,
    ) -> tuple[TapeRecord, TapeStep]:
        """Instant UI / Doctor counterfactual fork at `step_index` without restarting client code!"""
        if step_index < 1:
            raise ValueError(f"step_index must be >= 1 (got {step_index})")
        src_tape = self.store.get_tape(source_tape_id)
        if src_tape is None:
            raise KeyError(f"Tape '{source_tape_id}' not found")
        src_steps = self.store.get_steps(source_tape_id)
        target_src_step = next((s for s in src_steps if s.step_index == step_index), None)
        if target_src_step is None:
            raise KeyError(f"Step {step_index} not found in tape '{source_tape_id}'")

        session = self.sessions.start_replay_or_fork_session(
            source_tape_id=source_tape_id,
            fork_from_step=step_index,
            override_patch=patch,
            set_default=False,
        )

        for s in src_steps:
            if s.step_index < step_index:
                self.record_replayed_step_into_session(session, s, replay_duration_ms=0.8)

        for span in self.store.get_mcp_spans(source_tape_id):
            if span.step_index < step_index:
                replayed_span = span.model_copy(
                    update={
                        "span_id": new_id("mcp"),
                        "tape_id": session.tape_id,
                        "replayed": True,
                        "duration_ms": 0.4,
                    }
                )
                self.store.append_mcp_span(replayed_span)

        prev_req = session.prev_raw_request
        if patch.stabilize_prefix_cache and prev_req is not None:
            prev_req = apply_override_patch(
                prev_req, OverridePatch(stabilize_prefix_cache=True)
            )

        mutated_req = apply_override_patch(target_src_step.raw_request, patch)
        provider = target_src_step.provider

        if executor_override is not None:
            executor = executor_override
        elif prefer_live_upstream and has_live_api_credentials(provider, self.config):
            executor = self.live_executor
        else:
            executor = self.offline_executor

        try:
            raw_resp, ttft_ms, dur_ms = await executor.execute_step(provider, mutated_req)
        except Exception:
            self.store.delete_tape(session.tape_id)
            raise
        forked_step = self.build_and_store_live_step(
            tape_id=session.tape_id,
            step_index=step_index,
            provider=provider,
            raw_request=mutated_req,
            raw_response=raw_resp,
            prev_raw_request=prev_req,
            ttft_ms=ttft_ms,
            duration_ms=dur_ms,
            override_patch=patch,
        )

        updated_tape = self.sessions.complete_session(session.session_id, final_status)
        return updated_tape or self.store.get_tape(session.tape_id), forked_step  # type: ignore[return-value]
