"""Deterministic multi-signal anomaly scanner and causal drift-step locator."""

from __future__ import annotations

import json
import re
from typing import Any

from forktape.models.tape import AnomalyFlag, MCPSpan, TapeRecord, TapeStatus, TapeStep

ENTITY_ID_RE = re.compile(r"\b(?:ORD|INC|TKT|USR|INV|TXN)[-_]\d{3,}\b")
EMPTY_PAYLOAD_MARKERS = {
    "",
    "[]",
    "{}",
    "null",
    "none",
    '{"results":[]}',
    '{"results": []}',
    '{"data":[]}',
    '{"data": []}',
    '{"items":[]}',
    '{"items": []}',
    "no results found",
    "0 rows returned",
}


def _safe_float(val: Any) -> float:
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    if isinstance(val, str):
        m = re.search(r"-?\d+(?:\.\d+)?", val.replace(",", ""))
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                return 0.0
    return 0.0


def _flatten_block_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(_flatten_block_content(item["content"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False)


def _normalize_empty_check(text: str) -> bool:
    s = text.strip().lower()
    if s in EMPTY_PAYLOAD_MARKERS:
        return True
    try:
        parsed = json.loads(text)
        if parsed in ([], {}, None):
            return True
        if isinstance(parsed, dict):
            for key in ("results", "data", "items", "rows", "records", "matches"):
                if key in parsed and parsed[key] in ([], None, ""):
                    return True
    except (ValueError, TypeError):
        pass
    return False


def _extract_preceding_tool_messages(request: dict[str, Any]) -> list[str]:
    msgs = request.get("messages")
    if not isinstance(msgs, list) or not msgs:
        return []
    collected: list[str] = []
    for m in reversed(msgs):
        if not isinstance(m, dict):
            break
        if m.get("role") == "tool":
            collected.append(_flatten_block_content(m.get("content")))
        elif isinstance(m.get("content"), list) and any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in m["content"]
        ):
            for b in m["content"]:
                if isinstance(b, dict) and b.get("type") == "tool_result":
                    collected.append(_flatten_block_content(b.get("content")))
        else:
            break
    return collected


def _collect_grounded_corpus_before_step(request: dict[str, Any]) -> str:
    msgs = request.get("messages")
    if not isinstance(msgs, list):
        return ""
    return " ".join(json.dumps(m, ensure_ascii=False) for m in msgs)


def _is_tool_output_error(out: str) -> bool:
    """Distinguish genuine tool/policy execution errors from healthy observability logs."""
    stripped = out.strip()
    if not stripped:
        return False
    error_tokens = (
        "error",
        "invalid",
        "denied",
        "exceed",
        "not_found",
        "not found",
        "deprecated",
        "failed",
        "rejected",
    )
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            if parsed.get("is_error") is True or parsed.get("ok") is False:
                return True
            if parsed.get("error") or parsed.get("exception"):
                return True
            status_val = str(parsed.get("status") or "").lower()
            if status_val:
                if status_val in ("ok", "healthy", "success", "completed", "ready", "green"):
                    return False
                if any(tok in status_val for tok in error_tokens):
                    return True
            if parsed.get("ok") is True:
                return False
            if len(stripped) < 600:
                out_l = stripped.lower()
                return any(tok in out_l for tok in error_tokens)
            return False
    except (ValueError, TypeError):
        pass

    if len(stripped) < 600:
        out_l = stripped.lower()
        return any(tok in out_l for tok in error_tokens)
    return False


def _did_step_trigger_downstream_tool_error(
    step_pos: int,
    steps: list[TapeStep],
    mcp_spans: list[MCPSpan] | None,
    tape_failed: bool = True,
) -> bool:
    """Forward Causal Consequence Check: inspect whether Step `k`'s tool call returned an error
    in Step `k+1`'s incoming tool messages or MCP spans (domain-agnostic!).
    """
    curr = steps[step_pos]
    if mcp_spans:
        for sp in mcp_spans:
            if sp.step_index == curr.step_index and sp.is_error:
                return True
    if step_pos + 1 < len(steps):
        next_msgs = steps[step_pos + 1].raw_request.get("messages") or []
        tool_outs = _extract_preceding_tool_messages({"messages": next_msgs})
        for out in tool_outs:
            if _is_tool_output_error(out):
                return True
    return tape_failed and step_pos == len(steps) - 1 and len(curr.tool_calls) > 0


def scan_tape_anomalies(
    tape: TapeRecord,
    steps: list[TapeStep],
    mcp_spans: list[MCPSpan] | None = None,
) -> list[AnomalyFlag]:
    """Scan an agent trajectory for 5 canonical engineering failure modes + forward causal errors."""
    findings: list[AnomalyFlag] = []
    seen_call_signatures: dict[str, list[int]] = {}
    tape_failed = tape.status != TapeStatus.SUCCESS

    for pos, step in enumerate(steps):
        idx = step.step_index
        req = step.raw_request
        tools_list = req.get("tools") if isinstance(req.get("tools"), list) else []
        msgs = req.get("messages") if isinstance(req.get("messages"), list) else []
        caused_downstream_error = _did_step_trigger_downstream_tool_error(
            pos, steps, mcp_spans, tape_failed=tape_failed
        )

        # 1. TOOL_CALL_LOOP: Repeated identical (tool_name, canonical_args) calls across distinct steps
        emitted_loop_sigs_in_step: set[str] = set()
        for tc in step.tool_calls:
            sig = f"{tc.name}:{json.dumps(tc.arguments, sort_keys=True, ensure_ascii=False)}"
            history = seen_call_signatures.setdefault(sig, [])
            if idx not in history:
                history.append(idx)
            if len(history) >= 2 and sig not in emitted_loop_sigs_in_step:
                emitted_loop_sigs_in_step.add(sig)
                findings.append(
                    AnomalyFlag(
                        code="TOOL_CALL_LOOP",
                        severity="critical" if len(history) >= 3 else "high",
                        step_index=idx,
                        title=f"Repeated Tool Call Loop (`{tc.name}`)",
                        detail=(
                            f"Tool `{tc.name}` was invoked with identical arguments across steps {history}, "
                            "indicating non-progressing retry/state loop."
                        ),
                        evidence={"tool": tc.name, "steps": list(history), "arguments": tc.arguments},
                    )
                )

        # 2. TOOL_SCHEMA_OVERLOAD: Massive MCP/function schema dilution (>= 20 tools or >= 2400 schema tokens)
        tool_count = len(tools_list)
        schema_tokens = step.token_breakdown.tools_schema_tokens
        called_distractor = any(
            "deprecated" in tc.name.lower()
            or "legacy" in tc.name.lower()
            or "internal_admin" in tc.name.lower()
            or "wrong_" in tc.name.lower()
            or (isinstance(tc.arguments, dict) and "warning" in tc.arguments)
            for tc in step.tool_calls
        )
        if (tool_count >= 20 or schema_tokens >= 2400) and step.tool_calls:
            triggered_fault = called_distractor or caused_downstream_error
            severity = "critical" if triggered_fault else "medium"
            if triggered_fault or tool_count >= 25:
                findings.append(
                    AnomalyFlag(
                        code="TOOL_SCHEMA_OVERLOAD",
                        severity=severity,
                        step_index=idx,
                        title=f"MCP Context Tax / Tool Overload ({tool_count} tools, {schema_tokens} schema tokens)",
                        detail=(
                            f"Step {idx} injected {tool_count} tool definitions ({schema_tokens} tokens), "
                            "diluting attention across overlapping parameter schemas."
                        ),
                        evidence={
                            "tool_count": tool_count,
                            "tools_schema_tokens": schema_tokens,
                            "called_tools": [tc.name for tc in step.tool_calls],
                            "called_distractor": triggered_fault,
                        },
                    )
                )

        # 3. CONTEXT_MIDDLE_LOSS: Oversized raw tool JSON/HTML in context window polluting attention
        bloated_turns: list[tuple[int, int]] = []
        for m_i, m in enumerate(msgs):
            if not isinstance(m, dict):
                continue
            if m.get("role") == "tool":
                c_text = _flatten_block_content(m.get("content"))
                if len(c_text) >= 5000 and "_forktape_compacted" not in c_text:
                    bloated_turns.append((m_i, len(c_text)))
            elif isinstance(m.get("content"), list):
                for b in m["content"]:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        c_text = _flatten_block_content(b.get("content"))
                        if len(c_text) >= 5000 and "_forktape_compacted" not in c_text:
                            bloated_turns.append((m_i, len(c_text)))
        if bloated_turns:
            max_chars = max(c for _, c in bloated_turns)
            violated_constraint = (
                any(
                    isinstance(tc.arguments, dict)
                    and (
                        tc.arguments.get("exceeds_policy") is True
                        or tc.arguments.get("override_cap") is True
                        or "uncompacted" in str(tc.arguments)
                        or _safe_float(tc.arguments.get("amount")) > 500
                    )
                    for tc in step.tool_calls
                )
                or caused_downstream_error
            )
            findings.append(
                AnomalyFlag(
                    code="CONTEXT_MIDDLE_LOSS",
                    severity="critical" if violated_constraint else "high",
                    step_index=idx,
                    title=f"Context Rot / Middle Loss ({max_chars:,} chars raw tool bloat)",
                    detail=(
                        f"Uncompacted tool payload ({max_chars:,} chars) in context window buried "
                        "earlier policy constraints during step reasoning."
                    ),
                    evidence={
                        "bloated_message_count": len(bloated_turns),
                        "max_payload_chars": max_chars,
                        "violated_constraint": violated_constraint,
                    },
                )
            )

        # 4. SILENT_EMPTY_RETRIEVAL: Empty tool response followed by ungrounded ID/value fabrication
        preceding_tool_outputs = _extract_preceding_tool_messages(req)
        if preceding_tool_outputs and all(_normalize_empty_check(out) for out in preceding_tool_outputs):
            grounded_corpus = _collect_grounded_corpus_before_step(req)
            output_surface = step.assembled_text + " " + json.dumps(
                [tc.model_dump() for tc in step.tool_calls], ensure_ascii=False
            )
            invented_ids = [
                eid for eid in ENTITY_ID_RE.findall(output_surface) if eid not in grounded_corpus
            ]
            mutating_after_empty = len(step.tool_calls) > 0 and not any(
                "search" in tc.name.lower() or "list" in tc.name.lower() or "ask" in tc.name.lower()
                for tc in step.tool_calls
            )
            if invented_ids or mutating_after_empty:
                findings.append(
                    AnomalyFlag(
                        code="SILENT_EMPTY_RETRIEVAL",
                        severity="critical",
                        step_index=idx,
                        title="Ungrounded Fabrication After Empty Tool Result",
                        detail=(
                            f"Preceding tool call returned an empty result set, yet Step {idx} "
                            f"fabricated ungrounded entities {invented_ids or '[unverified action]'} "
                            "instead of halting or requesting clarification."
                        ),
                        evidence={
                            "empty_outputs": preceding_tool_outputs[:2],
                            "fabricated_entities": invented_ids,
                            "called_tools": [tc.name for tc in step.tool_calls],
                        },
                    )
                )

        # 5. PREFIX_CACHE_THRASHING: Broken Prefix KV Cache continuity across turns
        if idx >= 2 and step.prefix_cache_ratio < 0.55 and step.prompt_tokens >= 600:
            findings.append(
                AnomalyFlag(
                    code="PREFIX_CACHE_THRASHING",
                    severity="high",
                    step_index=idx,
                    title=f"Prefix KV Cache Thrashing (LCP={step.prefix_cache_ratio * 100:.1f}%)",
                    detail=(
                        f"Only {step.prefix_cache_ratio * 100:.1f}% of the prompt prefix matched Step {idx - 1} "
                        f"(diverged at char offset {step.prefix_break_offset}), invalidating KV cache reuse."
                    ),
                    evidence={
                        "prefix_cache_ratio": step.prefix_cache_ratio,
                        "break_offset": step.prefix_break_offset,
                        "break_snippet": step.prefix_break_snippet,
                    },
                )
            )

    return findings


SEVERITY_WEIGHTS: dict[str, float] = {
    "critical": 10.0,
    "high": 5.0,
    "medium": 2.0,
    "low": 1.0,
}


def locate_suspected_drift_step(
    steps: list[TapeStep],
    findings: list[AnomalyFlag],
    tape_metadata: dict[str, Any] | None = None,
) -> tuple[int, str]:
    """Identify the primary causal inflection step (`drift_step`) and dominant anomaly code."""
    if not steps:
        return 1, "UNKNOWN"

    if not findings:
        return steps[-1].step_index, "PROMPT_AMBIGUITY"

    step_scores: dict[int, float] = {}
    step_top_code: dict[int, tuple[float, str]] = {}

    for f in findings:
        w = SEVERITY_WEIGHTS.get(f.severity, 1.0)
        if f.code == "SILENT_EMPTY_RETRIEVAL":
            w += 6.0
        elif f.code == "TOOL_SCHEMA_OVERLOAD" and f.evidence.get("called_distractor"):
            w += 6.5
        elif f.code == "CONTEXT_MIDDLE_LOSS" and f.evidence.get("violated_constraint"):
            w += 6.2
        elif f.code == "TOOL_CALL_LOOP":
            repeat_steps = f.evidence.get("steps") or [f.step_index]
            target_idx = repeat_steps[1] if len(repeat_steps) >= 2 else f.step_index
            step_scores[target_idx] = step_scores.get(target_idx, 0.0) + w + 4.0
            prev_best = step_top_code.get(target_idx, (0.0, ""))
            if w + 4.0 > prev_best[0]:
                step_top_code[target_idx] = (w + 4.0, f.code)
            continue

        step_scores[f.step_index] = step_scores.get(f.step_index, 0.0) + w
        prev_best = step_top_code.get(f.step_index, (0.0, ""))
        if w > prev_best[0]:
            step_top_code[f.step_index] = (w, f.code)

    best_step = sorted(step_scores.keys(), key=lambda s: (-step_scores[s], s))[0]
    best_code = step_top_code[best_step][1]
    return best_step, best_code
