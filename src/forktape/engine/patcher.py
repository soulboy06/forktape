"""Counterfactual request patcher: BM25 tool pruning, middle-context compaction, and KV cache stabilization."""

from __future__ import annotations

import copy
import json
import re
from typing import Any

from rank_bm25 import BM25Okapi

from forktape.engine.fingerprint import ISO_TIMESTAMP_RE, UUID_RE
from forktape.models.tape import OverridePatch

TOKENIZE_RE = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]")
STANDALONE_VOLATILE_LINE_RE = re.compile(
    r"(?im)^[ \t]*(?:current\s+time|timestamp|request[_\s-]?id|trace[_\s-]?id|session[_\s-]?nonce)\s*[:=]\s*"
    r"(?:\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?|[A-Za-z0-9._\-]+)[ \t]*$\n?"
)
VOLATILE_ID_KV_RE = re.compile(
    r"(?i)\b((?:request|trace|session|span)[_\s-]?(?:id|nonce))\s*([:=])\s*[A-Za-z0-9._\-]+"
)


def _is_anthropic_request(request: dict[str, Any]) -> bool:
    if "system" in request:
        return True
    model = str(request.get("model") or "").lower()
    if "claude" in model:
        return True
    for t in request.get("tools") or []:
        if isinstance(t, dict) and "input_schema" in t:
            return True
    for m in request.get("messages") or []:
        if isinstance(m, dict) and isinstance(m.get("content"), list):
            for b in m["content"]:
                if isinstance(b, dict) and b.get("type") in ("tool_use", "tool_result"):
                    return True
    return False


def _stabilize_volatile_str(text: str) -> str:
    out = STANDALONE_VOLATILE_LINE_RE.sub("", text)
    out = ISO_TIMESTAMP_RE.sub("<ISO_TS>", out)
    out = UUID_RE.sub("<UUID>", out)
    out = VOLATILE_ID_KV_RE.sub(r"\1\2 <STABLE_ID>", out)
    return out.strip()


def _strip_volatile_from_content(content: Any) -> Any:
    if isinstance(content, str):
        return _stabilize_volatile_str(content)
    if isinstance(content, list):
        updated = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                nb = dict(block)
                nb["text"] = _stabilize_volatile_str(nb["text"])
                updated.append(nb)
            else:
                updated.append(block)
        return updated
    return content


def _tokenize_for_bm25(text: str) -> list[str]:
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text).replace("_", " ").replace("-", " ")
    tokens = [t.lower() for t in TOKENIZE_RE.findall(expanded)]
    return tokens or ["empty"]


def _tool_to_document(tool: dict[str, Any]) -> tuple[str, list[str]]:
    if "function" in tool and isinstance(tool["function"], dict):
        fn = tool["function"]
        name = str(fn.get("name", ""))
        desc = str(fn.get("description", ""))
        params = json.dumps(fn.get("parameters", {}), ensure_ascii=False)
    else:
        name = str(tool.get("name", ""))
        desc = str(tool.get("description", ""))
        params = json.dumps(tool.get("input_schema", {}), ensure_ascii=False)
    doc_str = f"{name} {name} {name} {desc} {params}"
    return name, _tokenize_for_bm25(doc_str)


def _extract_query_from_messages(request: dict[str, Any]) -> str:
    msgs = request.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return ""
    snippets: list[str] = []
    user_msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
    if user_msgs:
        snippets.append(json.dumps(user_msgs[0].get("content", ""), ensure_ascii=False))
    for m in msgs[-3:]:
        if isinstance(m, dict) and m.get("role") != "system":
            content_str = json.dumps(m.get("content", ""), ensure_ascii=False)
            snippets.append(content_str[:1200])
    return " ".join(snippets)


def prune_tools_bm25_preserving_order(
    tools: list[dict[str, Any]],
    query_text: str,
    top_k: int,
    required_tool_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Select Top-K relevant tools via BM25 while preserving original list order (KV-Cache safe!)."""
    if len(tools) <= top_k or top_k <= 0:
        return tools

    required = required_tool_names or set()
    docs: list[list[str]] = []
    names: list[str] = []
    for t in tools:
        t_name, toks = _tool_to_document(t)
        names.append(t_name)
        docs.append(toks)

    bm25 = BM25Okapi(docs)
    q_tokens = _tokenize_for_bm25(query_text)
    scores = bm25.get_scores(q_tokens)

    ranked_indices = sorted(range(len(tools)), key=lambda i: float(scores[i]), reverse=True)
    top_bm25: set[int] = set(ranked_indices[:top_k])
    required_indices: set[int] = {i for i, n in enumerate(names) if n in required}
    selected = top_bm25 | required_indices

    return [tools[i] for i in sorted(selected)]


def compact_bloated_tool_content(raw_content: str, max_chars: int) -> str:
    """Deterministically compact oversized JSON/HTML/log tool payloads while keeping key fields."""
    if len(raw_content) <= max_chars:
        return raw_content

    try:
        parsed = json.loads(raw_content)
        if isinstance(parsed, dict):
            summary_dict: dict[str, Any] = {
                "_forktape_compacted": True,
                "original_chars": len(raw_content),
            }
            for k, v in parsed.items():
                if isinstance(v, list) and len(v) > 3:
                    summary_dict[k] = v[:2] + [f"... ({len(v) - 3} middle items compacted) ..."] + v[-1:]
                elif isinstance(v, str) and len(v) > max_chars // 2:
                    summary_dict[k] = v[: max_chars // 4] + " ...[compacted]... " + v[-max_chars // 6 :]
                else:
                    summary_dict[k] = v
            candidate = json.dumps(summary_dict, ensure_ascii=False)
            if len(candidate) <= max_chars * 1.4:
                return candidate
        elif isinstance(parsed, list) and len(parsed) > 4:
            trimmed = parsed[:2] + [{"_forktape_compacted_items": len(parsed) - 3}] + parsed[-1:]
            candidate = json.dumps(trimmed, ensure_ascii=False)
            if len(candidate) <= max_chars * 1.4:
                return candidate
    except (ValueError, TypeError):
        pass

    half = max(120, max_chars // 2 - 20)
    return (
        f"{raw_content[:half]}\n"
        f"...[ForkTape compacted {len(raw_content) - 2 * half} chars (_forktape_compacted)]...\n"
        f"{raw_content[-half:]}"
    )


def apply_override_patch(request: dict[str, Any], patch: OverridePatch | None) -> dict[str, Any]:
    """Return a mutated copy of an LLM request after applying counterfactual overrides."""
    if patch is None or patch.is_empty():
        return copy.deepcopy(request)

    mutated = copy.deepcopy(request)
    is_anth = _is_anthropic_request(mutated)

    if patch.swap_model:
        mutated["model"] = patch.swap_model

    if patch.override_temperature is not None:
        mutated["temperature"] = patch.override_temperature

    messages = mutated.get("messages")

    # 1. System prompt replacement or append (OpenAI messages + Anthropic top-level `system` str/list)
    if patch.system_prompt_replace is not None:
        if is_anth:
            existing_sys = mutated.get("system")
            if isinstance(existing_sys, list) and existing_sys and isinstance(existing_sys[0], dict):
                first_blk = dict(existing_sys[0])
                first_blk["type"] = "text"
                first_blk["text"] = patch.system_prompt_replace
                mutated["system"] = [first_blk]
            else:
                mutated["system"] = patch.system_prompt_replace
        elif isinstance(messages, list):
            replaced = False
            for m in messages:
                if isinstance(m, dict) and m.get("role") in ("system", "developer"):
                    m["content"] = patch.system_prompt_replace
                    replaced = True
                    break
            if not replaced:
                messages.insert(0, {"role": "system", "content": patch.system_prompt_replace})

    if patch.system_prompt_append:
        if is_anth:
            existing_sys = mutated.get("system")
            if isinstance(existing_sys, str):
                mutated["system"] = existing_sys.rstrip() + "\n\n" + patch.system_prompt_append
            elif isinstance(existing_sys, list):
                mutated["system"] = list(existing_sys) + [{"type": "text", "text": patch.system_prompt_append}]
            else:
                mutated["system"] = patch.system_prompt_append
        elif isinstance(messages, list):
            appended = False
            for m in messages:
                if isinstance(m, dict) and m.get("role") in ("system", "developer"):
                    if isinstance(m.get("content"), str):
                        m["content"] = m["content"].rstrip() + "\n\n" + patch.system_prompt_append
                        appended = True
                        break
                    if isinstance(m.get("content"), list):
                        m["content"] = list(m["content"]) + [{"type": "text", "text": patch.system_prompt_append}]
                        appended = True
                        break
            if not appended:
                messages.insert(0, {"role": "system", "content": patch.system_prompt_append})

    # 2. Prefix KV Cache Stabilization (strip dynamic timestamps in system prompt + canonical tool sort)
    if patch.stabilize_prefix_cache:
        if "system" in mutated:
            mutated["system"] = _strip_volatile_from_content(mutated["system"])
        if isinstance(messages, list):
            for m in messages:
                if isinstance(m, dict) and m.get("role") in ("system", "developer"):
                    m["content"] = _strip_volatile_from_content(m.get("content"))
        if isinstance(mutated.get("tools"), list):
            mutated["tools"] = sorted(
                mutated["tools"],
                key=lambda t: str((t.get("function") or t).get("name", "")),
            )

    # 3. BM25 Semantic Tool Pruning (keeps tools invoked in Anthropic history or recent OpenAI turns + Top-K)
    if patch.prune_tools_top_k is not None and isinstance(mutated.get("tools"), list):
        ordered_used: list[str] = []
        if isinstance(messages, list):
            for m in messages:
                if not isinstance(m, dict):
                    continue
                for tc in m.get("tool_calls") or []:
                    if isinstance(tc, dict):
                        fn_name = (tc.get("function") or {}).get("name")
                        if fn_name:
                            s_name = str(fn_name)
                            if s_name in ordered_used:
                                ordered_used.remove(s_name)
                            ordered_used.append(s_name)
                if isinstance(m.get("content"), list):
                    for block in m["content"]:
                        if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("name"):
                            s_name = str(block["name"])
                            if s_name in ordered_used:
                                ordered_used.remove(s_name)
                            ordered_used.append(s_name)
        # Anthropic validates all historical tool_use names; OpenAI/DeepSeek long threads only need recent context
        retained_used = set(ordered_used) if is_anth else set(ordered_used[-3:])
        query = _extract_query_from_messages(mutated)
        mutated["tools"] = prune_tools_bm25_preserving_order(
            mutated["tools"],
            query_text=query,
            top_k=patch.prune_tools_top_k,
            required_tool_names=retained_used,
        )

    # 4. Context Tool Output Compaction (compacts OpenAI role='tool' AND Anthropic tool_result str/list)
    if patch.compact_context_max_chars is not None and isinstance(messages, list):
        max_c = patch.compact_context_max_chars
        for m in messages:
            if not isinstance(m, dict):
                continue
            if m.get("role") == "tool":
                if isinstance(m.get("content"), str):
                    m["content"] = compact_bloated_tool_content(m["content"], max_c)
                elif isinstance(m.get("content"), list):
                    for sub in m["content"]:
                        if isinstance(sub, dict) and isinstance(sub.get("text"), str):
                            sub["text"] = compact_bloated_tool_content(sub["text"], max_c)
            elif isinstance(m.get("content"), list):
                for block in m["content"]:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        b_content = block.get("content")
                        if isinstance(b_content, str):
                            block["content"] = compact_bloated_tool_content(b_content, max_c)
                        elif isinstance(b_content, list):
                            for sub in b_content:
                                if isinstance(sub, dict) and isinstance(sub.get("text"), str):
                                    sub["text"] = compact_bloated_tool_content(sub["text"], max_c)

    return mutated
