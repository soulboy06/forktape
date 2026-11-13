"""3-Tier Canonical Fingerprinting, Secret Redaction, and Prefix KV Cache LCP Analyzer."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from forktape.models.tape import StepTokenBreakdown

# ---------------------------------------------------------------------------
# 1. Secret Redaction Patterns (prevents leaking credentials in git-committed tapes)
# ---------------------------------------------------------------------------
SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"Bearer\s+[A-Za-z0-9._\-]{16,}"), "Bearer <REDACTED_TOKEN>"),
    (re.compile(r"\bsk-(?:ant|proj|live|test)?-?[A-Za-z0-9_\-]{16,}\b"), "<REDACTED_API_KEY>"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"), "<REDACTED_GITHUB_TOKEN>"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "<REDACTED_AWS_KEY>"),
    (re.compile(r"(?i)(api[_-]?key|secret|password|authorization)\s*[:=]\s*['\"]?[A-Za-z0-9_\-./+=]{12,}['\"]?"), r"\1=<REDACTED_SECRET>"),
]

SENSITIVE_HEADER_KEYS = {
    "authorization",
    "x-api-key",
    "api-key",
    "openai-api-key",
    "anthropic-api-key",
    "cookie",
    "set-cookie",
}

# ---------------------------------------------------------------------------
# 2. Volatile Noise Scrubbing Patterns (stabilizes hashes across runs)
# ---------------------------------------------------------------------------
ISO_TIMESTAMP_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b"
)
UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b"
)
EPOCH_MS_KEY_NAMES = {
    "timestamp",
    "created",
    "created_at",
    "updated_at",
    "request_id",
    "trace_id",
    "span_id",
    "nonce",
    "x-request-id",
    "system_fingerprint",
}


def redact_secrets_in_text(text: str) -> str:
    """Scrub API keys, Bearer tokens, and credentials from strings before persistence."""
    out = text
    for pattern, replacement in SECRET_PATTERNS:
        out = pattern.sub(replacement, out)
    return out


def redact_payload(obj: Any) -> Any:
    """Recursively redact secrets from arbitrary JSON-serializable payloads."""
    if isinstance(obj, str):
        return redact_secrets_in_text(obj)
    if isinstance(obj, list):
        return [redact_payload(item) for item in obj]
    if isinstance(obj, dict):
        cleaned: dict[str, Any] = {}
        for k, v in obj.items():
            if str(k).lower() in SENSITIVE_HEADER_KEYS:
                cleaned[k] = "<REDACTED_HEADER>"
            else:
                cleaned[k] = redact_payload(v)
        return cleaned
    return obj


def scrub_volatile_text(text: str) -> str:
    """Normalize dynamic timestamps, UUIDs, and secrets for canonical hashing."""
    s = redact_secrets_in_text(text)
    s = ISO_TIMESTAMP_RE.sub("<ISO_TS>", s)
    s = UUID_RE.sub("<UUID>", s)
    return s


# ---------------------------------------------------------------------------
# 3. Topological tool_call_id Normalizer & Remapper
# ---------------------------------------------------------------------------
def normalize_request_for_fingerprint(request: dict[str, Any]) -> dict[str, Any]:
    """Produce a deterministic, canonicalized copy of an LLM or MCP request payload.

    - Strips transport-only flags (`stream`, `stream_options`, `user`, volatile metadata).
    - Topologically normalizes random `tool_call_id` / `tool_use.id` values (`<TC_0001>`).
    - Scrubs ISO timestamps and UUIDs inside text blocks.
    """
    payload = copy.deepcopy(request)

    # Remove non-semantic transport keys
    for ignore_key in ("stream", "stream_options", "user", "metadata", "extra_headers"):
        payload.pop(ignore_key, None)

    tool_id_map: dict[str, str] = {}

    def map_tool_id(raw_id: str | None) -> str:
        if not raw_id:
            return "<TC_NONE>"
        if raw_id not in tool_id_map:
            tool_id_map[raw_id] = f"<TC_{len(tool_id_map) + 1:04d}>"
        return tool_id_map[raw_id]

    # Normalize OpenAI-style & Anthropic-style messages
    messages = payload.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            # Assistant tool_calls (OpenAI)
            tcs = msg.get("tool_calls")
            if isinstance(tcs, list):
                for tc in tcs:
                    if isinstance(tc, dict):
                        if "id" in tc:
                            tc["id"] = map_tool_id(str(tc["id"]))
                        fn = tc.get("function")
                        if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
                            try:
                                parsed_args = json.loads(fn["arguments"])
                                fn["arguments"] = json.dumps(parsed_args, sort_keys=True, ensure_ascii=False)
                            except (ValueError, TypeError):
                                pass

            # Tool result reference (OpenAI)
            if msg.get("role") == "tool" and "tool_call_id" in msg:
                msg["tool_call_id"] = map_tool_id(str(msg["tool_call_id"]))

            # Anthropic content blocks (tool_use / tool_result)
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        btype = block.get("type")
                        if btype == "tool_use" and "id" in block:
                            block["id"] = map_tool_id(str(block["id"]))
                        elif btype == "tool_result" and "tool_use_id" in block:
                            block["tool_use_id"] = map_tool_id(str(block["tool_use_id"]))

    def scrub_tree(node: Any) -> Any:
        if isinstance(node, str):
            return scrub_volatile_text(node)
        if isinstance(node, list):
            return [scrub_tree(x) for x in node]
        if isinstance(node, dict):
            out: dict[str, Any] = {}
            for k in sorted(node.keys()):
                if str(k).lower() in EPOCH_MS_KEY_NAMES:
                    continue
                out[k] = scrub_tree(node[k])
            return out
        return node

    return scrub_tree(payload)


def canonical_request_hash(request: dict[str, Any]) -> str:
    """Compute SHA-256 canonical fingerprint of an LLM or MCP request."""
    normalized = normalize_request_for_fingerprint(request)
    serialized = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def compute_structural_similarity(req_a: dict[str, Any], req_b: dict[str, Any]) -> float:
    """Compute structural similarity [0.0, 1.0] between two requests for fuzzy VCR alignment."""
    norm_a = json.dumps(
        normalize_request_for_fingerprint(req_a),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    norm_b = json.dumps(
        normalize_request_for_fingerprint(req_b),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    if norm_a == norm_b:
        return 1.0
    # Guard SequenceMatcher runtime on huge payloads by sampling head+tail if >32k chars
    if len(norm_a) > 32_000 or len(norm_b) > 32_000:
        norm_a = norm_a[:16_000] + norm_a[-16_000:]
        norm_b = norm_b[:16_000] + norm_b[-16_000:]
    return round(SequenceMatcher(None, norm_a, norm_b).quick_ratio(), 4)


# ---------------------------------------------------------------------------
# 4. Token Budget Estimator & Prefix KV Cache Continuity Analyzer (LCP)
# ---------------------------------------------------------------------------
CJK_RE = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]")


def estimate_tokens(text: str) -> int:
    """Fast, deterministic token approximation supporting mixed English/JSON/CJK."""
    if not text:
        return 0
    cjk_count = len(CJK_RE.findall(text))
    ascii_chars = max(0, len(text) - cjk_count)
    return max(1, int(math.ceil(cjk_count * 0.75 + ascii_chars / 3.8)))


def estimate_step_token_breakdown(request: dict[str, Any]) -> StepTokenBreakdown:
    """X-ray an LLM request into System, Tools Schema, History, and Tool Output tokens."""
    sys_tok = 0
    tools_tok = 0
    hist_tok = 0
    tool_out_tok = 0

    # Anthropic top-level system field
    if "system" in request and request["system"]:
        sys_tok += estimate_tokens(json.dumps(request["system"], ensure_ascii=False))

    # Tools schema
    tools = request.get("tools")
    if tools:
        tools_tok += estimate_tokens(json.dumps(tools, ensure_ascii=False))

    # Messages breakdown
    messages = request.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            serialized_msg = json.dumps(msg, ensure_ascii=False)
            tok = estimate_tokens(serialized_msg)
            if role in ("system", "developer"):
                sys_tok += tok
            elif role == "tool":
                tool_out_tok += tok
            else:
                # Check Anthropic tool_result blocks inside user messages
                content = msg.get("content")
                if isinstance(content, list) and any(
                    isinstance(b, dict) and b.get("type") == "tool_result" for b in content
                ):
                    tool_out_tok += tok
                else:
                    hist_tok += tok

    return StepTokenBreakdown(
        system_tokens=sys_tok,
        tools_schema_tokens=tools_tok,
        history_tokens=hist_tok,
        tool_output_tokens=tool_out_tok,
    )


@dataclass(slots=True)
class PrefixCacheAnalysis:
    ratio: float
    break_offset: int | None = None
    break_snippet: str | None = None


def serialize_prompt_prefix_for_kv_cache(request: dict[str, Any], max_messages: int | None = None) -> str:
    """Serialize request prefix in exact order LLM inference engines hash KV cache blocks:
    [tools definitions] + [system prompt] + [messages up to max_messages].
    Note: Does NOT scrub timestamps or preserve-sort keys inside lists, because real LLM
    KV caches break when raw timestamps or tool orders mutate!
    """
    parts: list[str] = []
    if "tools" in request and request["tools"]:
        parts.append("TOOLS:" + json.dumps(request["tools"], ensure_ascii=False, separators=(",", ":")))
    if "system" in request and request["system"]:
        parts.append("SYSTEM:" + json.dumps(request["system"], ensure_ascii=False, separators=(",", ":")))
    msgs = request.get("messages")
    if isinstance(msgs, list):
        subset = msgs[:max_messages] if max_messages is not None else msgs
        for idx, m in enumerate(subset):
            parts.append(f"M{idx}:" + json.dumps(m, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(parts)


def analyze_prefix_cache_continuity(
    prev_request: dict[str, Any] | None,
    curr_request: dict[str, Any],
) -> PrefixCacheAnalysis:
    """Compare consecutive LLM steps to detect Prefix KV Cache invalidation (LCP check)."""
    if prev_request is None:
        return PrefixCacheAnalysis(ratio=1.0)

    prev_msgs = prev_request.get("messages")
    prev_msg_count = len(prev_msgs) if isinstance(prev_msgs, list) else 0
    prev_stream = serialize_prompt_prefix_for_kv_cache(prev_request, max_messages=prev_msg_count)
    curr_stream = serialize_prompt_prefix_for_kv_cache(curr_request, max_messages=prev_msg_count)

    if not prev_stream:
        return PrefixCacheAnalysis(ratio=1.0)

    limit = min(len(prev_stream), len(curr_stream))
    lcp = 0
    while lcp < limit and prev_stream[lcp] == curr_stream[lcp]:
        lcp += 1

    ratio = round(lcp / max(1, len(prev_stream)), 4)
    if ratio >= 0.90:
        return PrefixCacheAnalysis(ratio=ratio)

    start = max(0, lcp - 24)
    prev_frag = prev_stream[start : lcp + 48].replace("\n", " ")
    curr_frag = curr_stream[start : lcp + 48].replace("\n", " ")
    snippet = f"Prev@{lcp}: «{prev_frag}»  !=  Curr@{lcp}: «{curr_frag}»"
    return PrefixCacheAnalysis(
        ratio=ratio,
        break_offset=lcp,
        break_snippet=snippet,
    )
