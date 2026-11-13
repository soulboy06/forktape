"""Counterfactual request patcher: BM25 tool pruning, middle-context compaction, and KV cache stabilization."""

from __future__ import annotations

import copy
import json
import re
from typing import Any

from rank_bm25 import BM25Okapi

from forktape.models.tape import OverridePatch

TOKENIZE_RE = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]")
VOLATILE_HEADER_LINE_RE = re.compile(
    r"(?im)^.*(?:current\s+time|timestamp|request[_\s-]?id|trace[_\s-]?id|session[_\s-]?nonce)\s*[:=].*$\n?"
)


def _tokenize_for_bm25(text: str) -> list[str]:
    # Split camelCase and snake_case words for accurate tool retrieval
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
    # Repeat tool name to boost exact/partial name matches
    doc_str = f"{name} {name} {name} {desc} {params}"
    return name, _tokenize_for_bm25(doc_str)


def _extract_query_from_messages(request: dict[str, Any]) -> str:
    msgs = request.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return ""
    snippets: list[str] = []
    # Include first user task + last 3 non-system messages for multi-hop context
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
    selected: set[int] = {i for i, n in enumerate(names) if n in required}
    for idx in ranked_indices:
        if len(selected) >= top_k:
            break
        selected.add(idx)

    # Preserve original relative index ordering to maximize Prefix Cache reuse
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
    return f"{raw_content[:half]}\n...[ForkTape compacted {len(raw_content) - 2 * half} chars]...\n{raw_content[-half:]}"


def apply_override_patch(request: dict[str, Any], patch: OverridePatch | None) -> dict[str, Any]:
    """Return a mutated copy of an LLM request after applying counterfactual overrides."""
    if patch is None or patch.is_empty():
        return copy.deepcopy(request)

    mutated = copy.deepcopy(request)

    if patch.swap_model:
        mutated["model"] = patch.swap_model

    if patch.override_temperature is not None:
        mutated["temperature"] = patch.override_temperature

    messages = mutated.get("messages")

    # 1. System prompt replacement or append (OpenAI messages + Anthropic top-level `system`)
    if patch.system_prompt_replace is not None:
        if "system" in mutated:
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
        if "system" in mutated and isinstance(mutated["system"], str):
            mutated["system"] = mutated["system"].rstrip() + "\n\n" + patch.system_prompt_append
        elif isinstance(messages, list):
            appended = False
            for m in messages:
                if isinstance(m, dict) and m.get("role") in ("system", "developer") and isinstance(m.get("content"), str):
                    m["content"] = m["content"].rstrip() + "\n\n" + patch.system_prompt_append
                    appended = True
                    break
            if not appended:
                messages.insert(0, {"role": "system", "content": patch.system_prompt_append})

    # 2. Prefix KV Cache Stabilization (strip dynamic timestamps in system prompt + canonical tool sort)
    if patch.stabilize_prefix_cache:
        if "system" in mutated and isinstance(mutated["system"], str):
            mutated["system"] = VOLATILE_HEADER_LINE_RE.sub("", mutated["system"]).strip()
        if isinstance(messages, list):
            for m in messages:
                if isinstance(m, dict) and m.get("role") in ("system", "developer") and isinstance(m.get("content"), str):
                    m["content"] = VOLATILE_HEADER_LINE_RE.sub("", m["content"]).strip()
        if isinstance(mutated.get("tools"), list):
            mutated["tools"] = sorted(
                mutated["tools"],
                key=lambda t: str((t.get("function") or t).get("name", "")),
            )

    # 3. BM25 Semantic Tool Pruning (keeps tools already invoked in history + Top-K matches)
    if patch.prune_tools_top_k is not None and isinstance(mutated.get("tools"), list):
        historically_used: set[str] = set()
        if isinstance(messages, list):
            for m in messages:
                if isinstance(m, dict):
                    for tc in m.get("tool_calls") or []:
                        if isinstance(tc, dict):
                            fn_name = (tc.get("function") or {}).get("name")
                            if fn_name:
                                historically_used.add(str(fn_name))
        query = _extract_query_from_messages(mutated)
        mutated["tools"] = prune_tools_bm25_preserving_order(
            mutated["tools"],
            query_text=query,
            top_k=patch.prune_tools_top_k,
            required_tool_names=historically_used,
        )

    # 4. Context Tool Output Compaction (compacts any oversized tool payload > max_chars)
    if patch.compact_context_max_chars is not None and isinstance(messages, list):
        max_c = patch.compact_context_max_chars
        for m in messages:
            if not isinstance(m, dict):
                continue
            if m.get("role") == "tool" and isinstance(m.get("content"), str):
                m["content"] = compact_bloated_tool_content(m["content"], max_c)
            elif isinstance(m.get("content"), list):
                for block in m["content"]:
                    if isinstance(block, dict) and block.get("type") == "tool_result" and isinstance(block.get("content"), str):
                        block["content"] = compact_bloated_tool_content(block["content"], max_c)

    return mutated
