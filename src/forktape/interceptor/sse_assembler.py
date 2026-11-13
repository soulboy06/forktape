"""Lossless streaming SSE chunk assemblers and deterministic SSE replay synthesizers."""

from __future__ import annotations

import json
import uuid
from typing import Any

from forktape.models.tape import ToolCallInvocation


def _coerce_tool_args(raw: Any) -> dict[str, Any] | str:
    """Coerce arbitrary tool argument payloads safely into `dict[str, Any] | str`."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            loaded = json.loads(raw)
            if isinstance(loaded, dict):
                return loaded
        except ValueError:
            pass
        return raw
    if raw is None:
        return {}
    return json.dumps(raw, ensure_ascii=False)


class OpenAISSEAssembler:
    """Reconstructs a full ChatCompletion object from OpenAI-compatible SSE lines."""

    def __init__(self, fallback_model: str = "gpt-4o") -> None:
        self.response_id: str = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        self.model: str = fallback_model
        self.content_parts: list[str] = []
        self.reasoning_parts: list[str] = []
        self.tool_call_buffers: dict[int, dict[str, str]] = {}
        self.finish_reason: str = "stop"
        self.prompt_tokens: int = 0
        self.cached_prompt_tokens: int = 0
        self.completion_tokens: int = 0

    def feed_line(self, line: str) -> None:
        stripped = line.strip()
        if not stripped.startswith("data:"):
            return
        payload_str = stripped[5:].strip()
        if not payload_str or payload_str == "[DONE]":
            return
        try:
            chunk = json.loads(payload_str)
        except ValueError:
            return
        if not isinstance(chunk, dict):
            return

        if chunk.get("id"):
            self.response_id = str(chunk["id"])
        if chunk.get("model"):
            self.model = str(chunk["model"])

        usage = chunk.get("usage")
        if isinstance(usage, dict):
            self.prompt_tokens = int(usage.get("prompt_tokens", self.prompt_tokens) or 0)
            self.completion_tokens = int(usage.get("completion_tokens", self.completion_tokens) or 0)
            details = usage.get("prompt_tokens_details")
            if isinstance(details, dict) and "cached_tokens" in details:
                self.cached_prompt_tokens = int(details.get("cached_tokens") or 0)
            elif "prompt_cache_hit_tokens" in usage:
                self.cached_prompt_tokens = int(usage.get("prompt_cache_hit_tokens") or 0)

        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            return
        choice0 = choices[0]
        if not isinstance(choice0, dict):
            return
        if choice0.get("finish_reason"):
            self.finish_reason = str(choice0["finish_reason"])

        delta = choice0.get("delta")
        if not isinstance(delta, dict):
            return

        if delta.get("content") is not None and delta.get("content") != "":
            self.content_parts.append(str(delta["content"]))
        if delta.get("reasoning_content") is not None and delta.get("reasoning_content") != "":
            self.reasoning_parts.append(str(delta["reasoning_content"]))

        tcs = delta.get("tool_calls")
        if isinstance(tcs, list):
            for tc in tcs:
                if not isinstance(tc, dict):
                    continue
                idx = int(tc.get("index", 0))
                buf = self.tool_call_buffers.setdefault(
                    idx, {"id": "", "name": "", "arguments": ""}
                )
                if tc.get("id"):
                    buf["id"] = str(tc["id"])
                fn = tc.get("function")
                if isinstance(fn, dict):
                    if fn.get("name"):
                        buf["name"] += str(fn["name"])
                    if fn.get("arguments"):
                        buf["arguments"] += str(fn["arguments"])

    def build_response(self) -> tuple[dict[str, Any], str, str, list[ToolCallInvocation]]:
        assembled_text = "".join(self.content_parts)
        reasoning_text = "".join(self.reasoning_parts)
        openai_tool_calls: list[dict[str, Any]] = []
        invocations: list[ToolCallInvocation] = []

        for idx in sorted(self.tool_call_buffers.keys()):
            buf = self.tool_call_buffers[idx]
            tc_id = buf["id"] or f"call_{uuid.uuid4().hex[:10]}"
            raw_args = buf["arguments"] or "{}"
            parsed_args = _coerce_tool_args(raw_args)
            openai_tool_calls.append(
                {
                    "id": tc_id,
                    "type": "function",
                    "function": {"name": buf["name"], "arguments": raw_args},
                }
            )
            invocations.append(
                ToolCallInvocation(id=tc_id, name=buf["name"], arguments=parsed_args)
            )

        message: dict[str, Any] = {
            "role": "assistant",
            "content": assembled_text if assembled_text or not openai_tool_calls else None,
        }
        if reasoning_text:
            message["reasoning_content"] = reasoning_text
        if openai_tool_calls:
            message["tool_calls"] = openai_tool_calls

        response = {
            "id": self.response_id,
            "object": "chat.completion",
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if openai_tool_calls else self.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
                "prompt_tokens_details": {"cached_tokens": self.cached_prompt_tokens},
            },
        }
        return response, assembled_text, reasoning_text, invocations


def extract_from_openai_json(
    response: dict[str, Any],
) -> tuple[str, str, list[ToolCallInvocation], int, int, int]:
    """Extract text, reasoning, tool calls, and token usage from a non-streaming OpenAI JSON."""
    text = ""
    reasoning = ""
    invocations: list[ToolCallInvocation] = []

    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message") or {}
        if msg.get("content"):
            text = str(msg["content"])
        if msg.get("reasoning_content"):
            reasoning = str(msg["reasoning_content"])
        tcs = msg.get("tool_calls")
        if isinstance(tcs, list):
            for tc in tcs:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") or {}
                raw_args = fn.get("arguments", "{}")
                parsed_args = _coerce_tool_args(raw_args)
                invocations.append(
                    ToolCallInvocation(
                        id=str(tc.get("id") or f"call_{uuid.uuid4().hex[:8]}"),
                        name=str(fn.get("name") or ""),
                        arguments=parsed_args,
                    )
                )

    usage = response.get("usage") or {}
    p_tok = int(usage.get("prompt_tokens") or 0)
    c_tok = int(
        (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        or usage.get("prompt_cache_hit_tokens")
        or 0
    )
    o_tok = int(usage.get("completion_tokens") or 0)
    return text, reasoning, invocations, p_tok, c_tok, o_tok


def synthesize_openai_sse_chunks(response: dict[str, Any]) -> list[bytes]:
    """Deterministically convert a stored OpenAI response dict into valid SSE frames."""
    resp_id = response.get("id", f"chatcmpl-{uuid.uuid4().hex[:10]}")
    model = response.get("model", "gpt-4o")
    choices = response.get("choices") or [{}]
    msg = choices[0].get("message") or {}
    finish_reason = choices[0].get("finish_reason", "stop")

    frames: list[bytes] = []

    opener = {
        "id": resp_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    frames.append(f"data: {json.dumps(opener, ensure_ascii=False)}\n\n".encode())

    if msg.get("reasoning_content"):
        r_chunk = {
            "id": resp_id,
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"reasoning_content": msg["reasoning_content"]},
                    "finish_reason": None,
                }
            ],
        }
        frames.append(f"data: {json.dumps(r_chunk, ensure_ascii=False)}\n\n".encode())

    content = msg.get("content")
    if content:
        c_chunk = {
            "id": resp_id,
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
        }
        frames.append(f"data: {json.dumps(c_chunk, ensure_ascii=False)}\n\n".encode())

    tool_calls = msg.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        delta_tcs = []
        for idx, tc in enumerate(tool_calls):
            fn = tc.get("function") or {}
            args_val = fn.get("arguments", "{}")
            if not isinstance(args_val, str):
                args_val = json.dumps(args_val, ensure_ascii=False)
            delta_tcs.append(
                {
                    "index": idx,
                    "id": tc.get("id", f"call_{idx}"),
                    "type": "function",
                    "function": {"name": fn.get("name", ""), "arguments": args_val},
                }
            )
        tc_chunk = {
            "id": resp_id,
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": {"tool_calls": delta_tcs}, "finish_reason": None}],
        }
        frames.append(f"data: {json.dumps(tc_chunk, ensure_ascii=False)}\n\n".encode())

    final_chunk = {
        "id": resp_id,
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        "usage": response.get("usage", {}),
    }
    frames.append(f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n".encode())
    frames.append(b"data: [DONE]\n\n")
    return frames


class AnthropicSSEAssembler:
    """Reconstructs an Anthropic Messages API response from SSE events."""

    def __init__(self, fallback_model: str = "claude-3-7-sonnet") -> None:
        self.msg_id: str = f"msg_{uuid.uuid4().hex[:12]}"
        self.model: str = fallback_model
        self.stop_reason: str = "end_turn"
        self.blocks: dict[int, dict[str, Any]] = {}
        self.input_tokens: int = 0
        self.cached_tokens: int = 0
        self.output_tokens: int = 0

    def feed_line(self, line: str) -> None:
        stripped = line.strip()
        if not stripped.startswith("data:"):
            return
        payload_str = stripped[5:].strip()
        if not payload_str:
            return
        try:
            ev = json.loads(payload_str)
        except ValueError:
            return
        if not isinstance(ev, dict):
            return

        etype = ev.get("type")
        if etype == "message_start":
            msg = ev.get("message") or {}
            self.msg_id = str(msg.get("id") or self.msg_id)
            self.model = str(msg.get("model") or self.model)
            usage = msg.get("usage") or {}
            self.input_tokens = int(usage.get("input_tokens") or 0)
            self.cached_tokens = int(usage.get("cache_read_input_tokens") or 0)
        elif etype == "content_block_start":
            idx = int(ev.get("index", 0))
            cb = ev.get("content_block") or {}
            btype = cb.get("type", "text")
            if btype == "tool_use":
                initial_input = cb.get("input")
                self.blocks[idx] = {
                    "type": "tool_use",
                    "id": cb.get("id", f"toolu_{idx}"),
                    "name": cb.get("name", ""),
                    "_initial_input": initial_input if isinstance(initial_input, dict) and initial_input else None,
                    "_raw_json": "",
                }
            elif btype == "thinking":
                self.blocks[idx] = {
                    "type": "thinking",
                    "thinking": str(cb.get("thinking") or ""),
                    "signature": str(cb.get("signature") or ""),
                }
            else:
                self.blocks[idx] = {"type": "text", "text": str(cb.get("text") or "")}
        elif etype == "content_block_delta":
            idx = int(ev.get("index", 0))
            delta = ev.get("delta") or {}
            block = self.blocks.setdefault(idx, {"type": "text", "text": ""})
            dtype = delta.get("type")
            if dtype == "text_delta":
                block["text"] = block.get("text", "") + str(delta.get("text", ""))
            elif dtype == "thinking_delta":
                block["thinking"] = block.get("thinking", "") + str(delta.get("thinking", ""))
            elif dtype == "signature_delta":
                block["signature"] = block.get("signature", "") + str(delta.get("signature", ""))
            elif dtype == "input_json_delta":
                block["_raw_json"] = block.get("_raw_json", "") + str(delta.get("partial_json", ""))
        elif etype == "message_delta":
            delta = ev.get("delta") or {}
            if delta.get("stop_reason"):
                self.stop_reason = str(delta["stop_reason"])
            usage = ev.get("usage") or {}
            if "output_tokens" in usage:
                self.output_tokens = int(usage["output_tokens"] or 0)

    def build_response(self) -> tuple[dict[str, Any], str, str, list[ToolCallInvocation]]:
        content_list: list[dict[str, Any]] = []
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        invocations: list[ToolCallInvocation] = []

        for idx in sorted(self.blocks.keys()):
            b = self.blocks[idx]
            if b["type"] == "text":
                text_parts.append(b.get("text", ""))
                content_list.append({"type": "text", "text": b.get("text", "")})
            elif b["type"] == "thinking":
                thinking_parts.append(b.get("thinking", ""))
                t_block: dict[str, Any] = {"type": "thinking", "thinking": b.get("thinking", "")}
                if b.get("signature"):
                    t_block["signature"] = b["signature"]
                content_list.append(t_block)
            elif b["type"] == "tool_use":
                raw_json = b.get("_raw_json") or ""
                if raw_json:
                    coerced_args = _coerce_tool_args(raw_json)
                    block_input = coerced_args if isinstance(coerced_args, dict) else {"_raw_arguments": coerced_args}
                elif b.get("_initial_input") is not None:
                    coerced_args = b["_initial_input"]
                    block_input = b["_initial_input"]
                else:
                    coerced_args = {}
                    block_input = {}
                content_list.append(
                    {
                        "type": "tool_use",
                        "id": b["id"],
                        "name": b["name"],
                        "input": block_input,
                    }
                )
                invocations.append(
                    ToolCallInvocation(id=b["id"], name=b["name"], arguments=coerced_args)
                )

        total_prompt = self.input_tokens + self.cached_tokens
        response = {
            "id": self.msg_id,
            "type": "message",
            "role": "assistant",
            "model": self.model,
            "content": content_list,
            "stop_reason": self.stop_reason,
            "usage": {
                "input_tokens": self.input_tokens,
                "cache_read_input_tokens": self.cached_tokens,
                "output_tokens": self.output_tokens,
                "prompt_tokens": total_prompt,
            },
        }
        return response, "".join(text_parts), "".join(thinking_parts), invocations


def extract_from_anthropic_json(
    response: dict[str, Any],
) -> tuple[str, str, list[ToolCallInvocation], int, int, int]:
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    invocations: list[ToolCallInvocation] = []

    for block in response.get("content") or []:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text_parts.append(str(block.get("text", "")))
        elif btype == "thinking":
            thinking_parts.append(str(block.get("thinking", "")))
        elif btype == "tool_use":
            invocations.append(
                ToolCallInvocation(
                    id=str(block.get("id") or ""),
                    name=str(block.get("name") or ""),
                    arguments=_coerce_tool_args(block.get("input")),
                )
            )

    usage = response.get("usage") or {}
    cached = int(usage.get("cache_read_input_tokens") or 0)
    raw_in = int(usage.get("input_tokens") or 0)
    prompt_total = int(usage.get("prompt_tokens") or (raw_in + cached))
    out_tok = int(usage.get("output_tokens") or 0)
    return "".join(text_parts), "".join(thinking_parts), invocations, prompt_total, cached, out_tok


def synthesize_anthropic_sse_chunks(response: dict[str, Any]) -> list[bytes]:
    """Deterministically convert a stored Anthropic Messages dict into valid SSE frames."""
    frames: list[bytes] = []
    usage = response.get("usage") or {}
    start_ev = {
        "type": "message_start",
        "message": {
            "id": response.get("id", "msg_replay"),
            "type": "message",
            "role": "assistant",
            "model": response.get("model", "claude-3-7-sonnet"),
            "content": [],
            "stop_reason": None,
            "usage": {
                "input_tokens": usage.get("input_tokens", 0),
                "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
                "output_tokens": 0,
            },
        },
    }
    frames.append(f"event: message_start\ndata: {json.dumps(start_ev, ensure_ascii=False)}\n\n".encode())

    for idx, block in enumerate(response.get("content") or []):
        btype = block.get("type", "text")
        if btype == "tool_use":
            cb_start = {
                "type": "content_block_start",
                "index": idx,
                "content_block": {
                    "type": "tool_use",
                    "id": block.get("id", f"toolu_{idx}"),
                    "name": block.get("name", ""),
                    "input": {},
                },
            }
            frames.append(f"event: content_block_start\ndata: {json.dumps(cb_start, ensure_ascii=False)}\n\n".encode())
            cb_delta = {
                "type": "content_block_delta",
                "index": idx,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(block.get("input", {}), ensure_ascii=False),
                },
            }
            frames.append(f"event: content_block_delta\ndata: {json.dumps(cb_delta, ensure_ascii=False)}\n\n".encode())
        elif btype == "thinking":
            cb_start = {
                "type": "content_block_start",
                "index": idx,
                "content_block": {"type": "thinking", "thinking": ""},
            }
            frames.append(f"event: content_block_start\ndata: {json.dumps(cb_start, ensure_ascii=False)}\n\n".encode())
            cb_delta = {
                "type": "content_block_delta",
                "index": idx,
                "delta": {"type": "thinking_delta", "thinking": block.get("thinking", "")},
            }
            frames.append(f"event: content_block_delta\ndata: {json.dumps(cb_delta, ensure_ascii=False)}\n\n".encode())
            if block.get("signature"):
                sig_delta = {
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {"type": "signature_delta", "signature": block["signature"]},
                }
                frames.append(
                    f"event: content_block_delta\ndata: {json.dumps(sig_delta, ensure_ascii=False)}\n\n".encode()
                )
        else:
            cb_start = {
                "type": "content_block_start",
                "index": idx,
                "content_block": {"type": "text", "text": ""},
            }
            frames.append(f"event: content_block_start\ndata: {json.dumps(cb_start, ensure_ascii=False)}\n\n".encode())
            cb_delta = {
                "type": "content_block_delta",
                "index": idx,
                "delta": {"type": "text_delta", "text": block.get("text", "")},
            }
            frames.append(f"event: content_block_delta\ndata: {json.dumps(cb_delta, ensure_ascii=False)}\n\n".encode())

        cb_stop = {"type": "content_block_stop", "index": idx}
        frames.append(f"event: content_block_stop\ndata: {json.dumps(cb_stop, ensure_ascii=False)}\n\n".encode())

    msg_delta = {
        "type": "message_delta",
        "delta": {"stop_reason": response.get("stop_reason", "end_turn")},
        "usage": {"output_tokens": usage.get("output_tokens", 0)},
    }
    frames.append(f"event: message_delta\ndata: {json.dumps(msg_delta, ensure_ascii=False)}\n\n".encode())
    frames.append(b'event: message_stop\ndata: {"type":"message_stop"}\n\n')
    return frames
