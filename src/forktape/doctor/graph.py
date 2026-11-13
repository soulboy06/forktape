"""LangGraph StateGraph orchestrating Counterfactual Ablation Biopsy for broken AI Agents."""

from __future__ import annotations

import json
import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from forktape.doctor.heuristics import locate_suspected_drift_step, scan_tape_anomalies
from forktape.engine.replayer import ReplayEngine
from forktape.models.db import TapeStore
from forktape.models.tape import (
    AblationRunResult,
    AnomalyFlag,
    BiopsyReport,
    OverridePatch,
    TapeRecord,
    TapeStatus,
    TapeStep,
)


class PlannedHypothesis(TypedDict):
    name: str
    operator: str
    target_category: str
    step_index: int
    patch: OverridePatch


class BiopsyGraphState(TypedDict, total=False):
    tape_id: str
    tape: TapeRecord
    steps: list[TapeStep]
    findings: list[AnomalyFlag]
    drift_step: int
    initial_category: str
    hypotheses: list[PlannedHypothesis]
    ablation_results: list[AblationRunResult]
    final_report: BiopsyReport
    started_at_perf: float


def _summarize_step_outcome(step: TapeStep) -> str:
    if step.tool_calls:
        calls = ", ".join(
            f"{tc.name}({json.dumps(tc.arguments, ensure_ascii=False)[:90]})" for tc in step.tool_calls
        )
        return f"ToolCall: {calls}"
    return f"Text: {step.assembled_text[:110]}"


def _did_ablation_resolve_step(
    operator: str,
    orig_step: TapeStep,
    forked_step: TapeStep,
    findings_at_step: list[AnomalyFlag],
) -> tuple[bool, list[str]]:
    """Evaluate deterministically whether a counterfactual fork fixed the failure at `orig_step`."""
    notes: list[str] = []
    orig_tools = [tc.name for tc in orig_step.tool_calls]
    new_tools = [tc.name for tc in forked_step.tool_calls]
    orig_args_str = json.dumps([tc.arguments for tc in orig_step.tool_calls], ensure_ascii=False)
    new_args_str = json.dumps([tc.arguments for tc in forked_step.tool_calls], ensure_ascii=False)

    if operator == "prune_tools_bm25":
        before_cnt = len(orig_step.raw_request.get("tools") or [])
        after_cnt = len(forked_step.raw_request.get("tools") or [])
        notes.append(f"Pruned tool schemas from {before_cnt} down to {after_cnt} via BM25.")
        eliminated_distractor = any(
            "deprecated" in t.lower() or "legacy" in t.lower() or "internal_admin" in t.lower() or "wrong_" in t.lower()
            for t in orig_tools
        ) and not any(
            "deprecated" in t.lower() or "legacy" in t.lower() or "internal_admin" in t.lower() or "wrong_" in t.lower()
            for t in new_tools
        )
        cleaned_warning = "unpruned" not in new_args_str and (orig_tools != new_tools or orig_args_str != new_args_str)
        if eliminated_distractor or (before_cnt >= 20 and cleaned_warning):
            notes.append(f"Shifted tool call from `{orig_tools}` to `{new_tools}` with grounded arguments.")
            return True, notes
        return False, notes

    if operator == "compact_context":
        tok_saved = orig_step.prompt_tokens - forked_step.prompt_tokens
        notes.append(f"Compacted middle tool payloads, shrinking prompt by {tok_saved:,} tokens.")
        had_bloat_flag = any(f.code == "CONTEXT_MIDDLE_LOSS" for f in findings_at_step)
        mutated_req_str = json.dumps(forked_step.raw_request, ensure_ascii=False)
        compacted_applied = "_forktape_compacted" in mutated_req_str or "ForkTape compacted" in mutated_req_str
        fixed_violation = (
            had_bloat_flag
            and "exceeds_policy" not in new_args_str
            and "override_cap" not in new_args_str
            and "uncompacted" not in new_args_str
            and (
                orig_args_str != new_args_str
                or forked_step.assembled_text != orig_step.assembled_text
                or (compacted_applied and tok_saved >= 800)
            )
        )
        if fixed_violation:
            notes.append("Eliminated policy constraint amnesia caused by middle-context dilution.")
            return True, notes
        return False, notes

    if operator == "empty_retrieval_guard":
        had_empty_flag = any(f.code == "SILENT_EMPTY_RETRIEVAL" for f in findings_at_step)
        halted_fabrication = had_empty_flag and (
            len(forked_step.tool_calls) == 0
            or "clarification" in forked_step.assembled_text.lower()
            or "no matching" in forked_step.assembled_text.lower()
        )
        if halted_fabrication:
            notes.append("Prevented ungrounded ID fabrication; model safely escalated on empty tool output.")
            return True, notes
        return False, notes

    if operator == "stabilize_prefix_cache":
        notes.append(
            f"Prefix KV Cache LCP jumped from {orig_step.prefix_cache_ratio * 100:.1f}% "
            f"to {forked_step.prefix_cache_ratio * 100:.1f}%."
        )
        if orig_step.prefix_cache_ratio < 0.55 and forked_step.prefix_cache_ratio >= 0.90:
            notes.append("Restored prefix byte-stability across turns by stripping volatile header noise.")
            return True, notes
        return False, notes

    if operator == "loop_breaker_guard":
        had_loop = any(f.code == "TOOL_CALL_LOOP" for f in findings_at_step)
        broke_repeat = had_loop and (new_tools != orig_tools or len(new_tools) == 0)
        if broke_repeat:
            notes.append("Broke repeated identical tool-call cycle via explicit state-transition guard.")
            return True, notes
        return False, notes

    return False, notes


def _best_step_for_code(findings: list[AnomalyFlag], code: str, fallback: int) -> int:
    matching = [f for f in findings if f.code == code]
    if not matching:
        return fallback
    critical = [f for f in matching if f.severity == "critical"]
    chosen = critical[0] if critical else matching[0]
    if code == "TOOL_CALL_LOOP":
        rep = chosen.evidence.get("steps") or [chosen.step_index]
        return int(rep[1]) if len(rep) >= 2 else chosen.step_index
    return chosen.step_index


def build_biopsy_langgraph(store: TapeStore, engine: ReplayEngine) -> Any:
    """Compile the 5-node LangGraph Counterfactual Biopsy state machine."""

    def scan_heuristics_node(state: BiopsyGraphState) -> dict[str, Any]:
        tape = state["tape"]
        steps = state["steps"]
        mcp_spans = store.get_mcp_spans(tape.tape_id)
        findings = scan_tape_anomalies(tape, steps, mcp_spans)

        by_step: dict[int, list[AnomalyFlag]] = {}
        for f in findings:
            by_step.setdefault(f.step_index, []).append(f)
        for s in steps:
            if s.step_index in by_step:
                store.update_step_anomalies(tape.tape_id, s.step_index, by_step[s.step_index])

        return {"findings": findings}

    def locate_drift_step_node(state: BiopsyGraphState) -> dict[str, Any]:
        drift_step, category = locate_suspected_drift_step(
            state["steps"],
            state.get("findings", []),
            state["tape"].metadata,
        )
        return {"drift_step": drift_step, "initial_category": category}

    def plan_hypotheses_node(state: BiopsyGraphState) -> dict[str, Any]:
        drift_step = state["drift_step"]
        initial_category = state.get("initial_category", "")
        findings = state.get("findings", [])
        codes_present = {f.code for f in findings}
        planned: list[PlannedHypothesis] = []

        if "TOOL_SCHEMA_OVERLOAD" in codes_present:
            t_step = (
                drift_step
                if initial_category == "TOOL_SCHEMA_OVERLOAD"
                else _best_step_for_code(findings, "TOOL_SCHEMA_OVERLOAD", drift_step)
            )
            planned.append(
                {
                    "name": "BM25 Semantic Tool Pruning (Top-5)",
                    "operator": "prune_tools_bm25",
                    "target_category": "TOOL_SCHEMA_OVERLOAD",
                    "step_index": t_step,
                    "patch": OverridePatch(
                        label="Doctor: BM25 Prune Tools Top-5",
                        prune_tools_top_k=5,
                        override_temperature=0.0,
                    ),
                }
            )

        if "CONTEXT_MIDDLE_LOSS" in codes_present:
            c_step = (
                drift_step
                if initial_category == "CONTEXT_MIDDLE_LOSS"
                else _best_step_for_code(findings, "CONTEXT_MIDDLE_LOSS", drift_step)
            )
            planned.append(
                {
                    "name": "Structure-Aware Middle Context Compaction",
                    "operator": "compact_context",
                    "target_category": "CONTEXT_MIDDLE_LOSS",
                    "step_index": c_step,
                    "patch": OverridePatch(
                        label="Doctor: Compact Tool Bloat (<=900 chars)",
                        compact_context_max_chars=900,
                        override_temperature=0.0,
                    ),
                }
            )

        if "SILENT_EMPTY_RETRIEVAL" in codes_present:
            e_step = (
                drift_step
                if initial_category == "SILENT_EMPTY_RETRIEVAL"
                else _best_step_for_code(findings, "SILENT_EMPTY_RETRIEVAL", drift_step)
            )
            planned.append(
                {
                    "name": "Empty-Result Grounding Guardrail",
                    "operator": "empty_retrieval_guard",
                    "target_category": "SILENT_EMPTY_RETRIEVAL",
                    "step_index": e_step,
                    "patch": OverridePatch(
                        label="Doctor: Empty-Retrieval Anti-Hallucination Guard",
                        system_prompt_append=(
                            "CRITICAL GROUNDING RULE: If the preceding search/query tool returned empty results "
                            "([] or null), NEVER fabricate IDs or invoke mutation tools. Ask for clarification."
                        ),
                        override_temperature=0.0,
                    ),
                }
            )

        if "TOOL_CALL_LOOP" in codes_present:
            l_step = (
                drift_step
                if initial_category == "TOOL_CALL_LOOP"
                else _best_step_for_code(findings, "TOOL_CALL_LOOP", drift_step)
            )
            planned.append(
                {
                    "name": "Cyclic Tool-Call Loop Breaker",
                    "operator": "loop_breaker_guard",
                    "target_category": "TOOL_CALL_LOOP",
                    "step_index": l_step,
                    "patch": OverridePatch(
                        label="Doctor: Loop Breaker Guard",
                        system_prompt_append=(
                            "CRITICAL LOOP GUARD: Never invoke a tool with identical arguments more than once. "
                            "If a tool already returned empty or unchanged state, summarize and stop."
                        ),
                        prune_tools_top_k=5,
                        override_temperature=0.0,
                    ),
                }
            )

        if "PREFIX_CACHE_THRASHING" in codes_present:
            cache_flag = next(f for f in findings if f.code == "PREFIX_CACHE_THRASHING")
            planned.append(
                {
                    "name": "Prefix KV Cache Header & Schema Stabilization",
                    "operator": "stabilize_prefix_cache",
                    "target_category": "PREFIX_CACHE_THRASHING",
                    "step_index": cache_flag.step_index,
                    "patch": OverridePatch(
                        label="Doctor: Stabilize Prefix KV Cache",
                        stabilize_prefix_cache=True,
                    ),
                }
            )

        # Ensure primary initial_category hypothesis runs FIRST so it is never truncated by [:3]
        planned.sort(key=lambda h: 0 if h["target_category"] == initial_category else 1)

        if len(planned) < 2:
            if not any(h["operator"] == "prune_tools_bm25" for h in planned):
                planned.append(
                    {
                        "name": "BM25 Tool Scope Narrowing Probe",
                        "operator": "prune_tools_bm25",
                        "target_category": "TOOL_SCHEMA_OVERLOAD",
                        "step_index": drift_step,
                        "patch": OverridePatch(
                            label="Probe: Prune Tools Top-6",
                            prune_tools_top_k=6,
                            override_temperature=0.0,
                        ),
                    }
                )
            if not any(h["operator"] == "compact_context" for h in planned):
                planned.append(
                    {
                        "name": "Context Compaction & Zero-Temp Stability Probe",
                        "operator": "compact_context",
                        "target_category": "CONTEXT_MIDDLE_LOSS",
                        "step_index": drift_step,
                        "patch": OverridePatch(
                            label="Probe: Compact Context + Temp=0",
                            compact_context_max_chars=1000,
                            override_temperature=0.0,
                        ),
                    }
                )

        return {"hypotheses": planned[:3]}

    async def execute_ablations_node(state: BiopsyGraphState) -> dict[str, Any]:
        tape_id = state["tape_id"]
        steps_by_idx = {s.step_index: s for s in state["steps"]}
        findings = state.get("findings", [])
        results: list[AblationRunResult] = []

        for hyp in state.get("hypotheses", []):
            s_idx = hyp["step_index"]
            orig_step = steps_by_idx.get(s_idx) or state["steps"][-1]
            s_idx = orig_step.step_index

            forked_tape, forked_step = await engine.fork_step_standalone(
                source_tape_id=tape_id,
                step_index=s_idx,
                patch=hyp["patch"],
                prefer_live_upstream=False,
            )

            step_findings = [f for f in findings if f.step_index == s_idx] or findings
            resolved, notes = _did_ablation_resolve_step(
                hyp["operator"], orig_step, forked_step, step_findings
            )
            if not resolved:
                store.update_tape_status(
                    forked_tape.tape_id,
                    TapeStatus.DIVERGED,
                    failure_reason="Counterfactual control ablation did not resolve primary fault.",
                )

            tok_before = orig_step.prompt_tokens
            tok_after = forked_step.prompt_tokens
            red_pct = round(((tok_before - tok_after) / max(1, tok_before)) * 100.0, 2)

            results.append(
                AblationRunResult(
                    hypothesis_name=hyp["name"],
                    operator=hyp["operator"],
                    step_index=s_idx,
                    forked_tape_id=forked_tape.tape_id,
                    patch_applied=hyp["patch"],
                    before_summary=_summarize_step_outcome(orig_step),
                    after_summary=_summarize_step_outcome(forked_step),
                    resolved_failure=resolved,
                    prompt_tokens_before=tok_before,
                    prompt_tokens_after=tok_after,
                    token_reduction_pct=red_pct,
                    cache_ratio_before=orig_step.prefix_cache_ratio,
                    cache_ratio_after=forked_step.prefix_cache_ratio,
                    latency_ms=forked_step.duration_ms,
                    cost_usd=forked_step.cost_usd,
                    evidence_notes=notes,
                )
            )

        return {"ablation_results": results}

    def synthesize_report_node(state: BiopsyGraphState) -> dict[str, Any]:
        tape = state["tape"]
        findings = state.get("findings", [])
        ablations = state.get("ablation_results", [])
        drift_step = state["drift_step"]
        initial_category = state["initial_category"]
        winner_category = initial_category
        recommended_patch: OverridePatch | None = None

        resolved_runs = [r for r in ablations if r.resolved_failure]
        operator_to_category = {
            "prune_tools_bm25": "TOOL_SCHEMA_OVERLOAD",
            "compact_context": "CONTEXT_MIDDLE_LOSS",
            "empty_retrieval_guard": "SILENT_EMPTY_RETRIEVAL",
            "stabilize_prefix_cache": "PREFIX_CACHE_THRASHING",
            "loop_breaker_guard": "TOOL_CALL_LOOP",
        }
        behavioral_resolved = [r for r in resolved_runs if r.operator != "stabilize_prefix_cache"]

        best_run: AblationRunResult | None = None
        if behavioral_resolved:
            # Prefer a behavioral fix matching initial_category if multiple resolved
            matching_initial = [
                r for r in behavioral_resolved if operator_to_category.get(r.operator) == initial_category
            ]
            best_run = matching_initial[0] if matching_initial else behavioral_resolved[0]
        elif resolved_runs and initial_category == "PREFIX_CACHE_THRASHING":
            best_run = resolved_runs[0]

        if best_run is not None:
            winner_category = operator_to_category.get(best_run.operator, winner_category)
            drift_step = best_run.step_index
            recommended_patch = best_run.patch_applied
            confidence = 0.94
            exec_summary = (
                f"Verified `{winner_category}` at Step {drift_step} via {len(ablations)} counterfactual "
                f"ablation forks: applying `{best_run.hypothesis_name}` resolved the faulty step "
                f"(prompt tokens {best_run.prompt_tokens_before:,} -> {best_run.prompt_tokens_after:,}, "
                f"Prefix KV Cache LCP {best_run.cache_ratio_before * 100:.0f}% -> {best_run.cache_ratio_after * 100:.0f}%)."
            )
            fix_rec = _remediation_playbook(winner_category, best_run)
        else:
            confidence = 0.76 if findings else 0.55
            exec_summary = (
                f"Localized trajectory drift to Step {drift_step} (`{winner_category}`) across "
                f"{len(findings)} deterministic anomaly signals and {len(ablations)} counterfactual probes."
            )
            fix_rec = _remediation_playbook(winner_category, ablations[0] if ablations else None)
            if ablations:
                recommended_patch = ablations[0].patch_applied

        total_cost = round(sum(r.cost_usd for r in ablations), 6)
        total_ms = round((time.perf_counter() - state["started_at_perf"]) * 1000.0, 2)

        temp_report = BiopsyReport(
            tape_id=tape.tape_id,
            suspected_drift_step=drift_step,
            root_cause_category=winner_category,
            confidence=confidence,
            executive_summary=exec_summary,
            heuristic_findings=findings,
            ablation_runs=ablations,
            verified_fix_recommendation=fix_rec,
            recommended_patch=recommended_patch,
            total_verification_cost_usd=total_cost,
            total_verification_time_ms=total_ms,
        )
        from forktape.compiler.pytest_gen import compile_tape_to_pytest_code

        store.save_biopsy_report(temp_report)
        pytest_code = compile_tape_to_pytest_code(store, tape.tape_id)
        temp_report.generated_pytest_code = pytest_code
        store.save_biopsy_report(temp_report)
        return {"final_report": temp_report}

    graph = StateGraph(BiopsyGraphState)
    graph.add_node("scan_heuristics", scan_heuristics_node)
    graph.add_node("locate_drift_step", locate_drift_step_node)
    graph.add_node("plan_hypotheses", plan_hypotheses_node)
    graph.add_node("execute_ablations", execute_ablations_node)
    graph.add_node("synthesize_report", synthesize_report_node)

    graph.add_edge(START, "scan_heuristics")
    graph.add_edge("scan_heuristics", "locate_drift_step")
    graph.add_edge("locate_drift_step", "plan_hypotheses")
    graph.add_edge("plan_hypotheses", "execute_ablations")
    graph.add_edge("execute_ablations", "synthesize_report")
    graph.add_edge("synthesize_report", END)

    return graph.compile()


def _remediation_playbook(category: str, best_run: AblationRunResult | None) -> str:
    guidance = {
        "TOOL_SCHEMA_OVERLOAD": (
            "1. Replace eager injection of 30+ MCP tool schemas with BM25 / semantic lazy routing (Top-5 per turn) "
            "while preserving relative tool list order to protect Prefix KV Cache.\n"
            "2. Disambiguate overlapping parameter names and remove deprecated/internal tools from default agent scope."
        ),
        "CONTEXT_MIDDLE_LOSS": (
            "1. Add a deterministic structure-aware compactor on MCP/tool outputs exceeding 1,000 chars so raw JSON/HTML "
            "blobs do not bury system policy constraints in the middle of the context window.\n"
            "2. Re-anchor critical business invariants (e.g., refund caps, read-only rules) near the tail of the prompt."
        ),
        "SILENT_EMPTY_RETRIEVAL": (
            "1. Add an explicit post-tool state guard: when a retrieval/search tool returns `[]` or `null`, transition "
            "deterministically to a clarification/escalation node or append an explicit system reminder forbidding entity fabrication.\n"
            "2. Validate entity IDs in mutation tool calls against IDs observed in prior tool spans."
        ),
        "PREFIX_CACHE_THRASHING": (
            "1. Move volatile per-request metadata (`Current time`, `trace_id`, session nonces) OUT of the System Prompt "
            "header into the latest user/context turn so the static prefix remains byte-identical.\n"
            "2. Sort registered MCP tool schemas canonically instead of iterating over unordered sets/dicts."
        ),
        "TOOL_CALL_LOOP": (
            "1. Add a deterministic `(tool_name, canonical_args)` deduplication circuit breaker in your agent harness "
            "(max_repeat=2) before dispatching tool calls.\n"
            "2. Surface explicit state-transition feedback when a retry returns unchanged state."
        ),
    }
    base = guidance.get(
        category,
        "Tighten tool parameter JSON schemas, add deterministic post-step assertions, and pin regression tests via ForkTape.",
    )
    if best_run and best_run.forked_tape_id:
        return f"{base}\n3. Verified Proof Branch: `{best_run.forked_tape_id}` (Step {best_run.step_index})."
    return base


async def run_counterfactual_biopsy(
    store: TapeStore,
    engine: ReplayEngine,
    tape_id: str,
) -> BiopsyReport:
    """Execute the full LangGraph Counterfactual Biopsy pipeline on `tape_id`."""
    tape = store.get_tape(tape_id)
    if tape is None:
        raise KeyError(f"Tape '{tape_id}' not found")
    steps = store.get_steps(tape_id)
    if not steps:
        raise ValueError(f"Tape '{tape_id}' contains 0 steps")

    compiled = build_biopsy_langgraph(store, engine)
    final_state = await compiled.ainvoke(
        {
            "tape_id": tape_id,
            "tape": tape,
            "steps": steps,
            "started_at_perf": time.perf_counter(),
        }
    )
    return final_state["final_report"]
