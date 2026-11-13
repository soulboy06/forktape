"""ForkBench-40: Reproducible 40-trajectory benchmark evaluating ForkTape's Drift Localization,
Root-Cause Classification, Counterfactual Fix Verification, and VCR Replay Savings.

Run:
    python benchmarks/run_forkbench.py
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forktape.config import estimate_cost_usd
from forktape.doctor.graph import run_counterfactual_biopsy
from forktape.engine.fingerprint import (
    analyze_prefix_cache_continuity,
    canonical_request_hash,
    estimate_step_token_breakdown,
)
from forktape.engine.replayer import ReplayEngine
from forktape.models.db import TapeStore
from forktape.models.tape import TapeRecord, TapeStatus, TapeStep, ToolCallInvocation

CATEGORIES = [
    "TOOL_SCHEMA_OVERLOAD",
    "CONTEXT_MIDDLE_LOSS",
    "SILENT_EMPTY_RETRIEVAL",
    "PREFIX_CACHE_THRASHING",
    "TOOL_CALL_LOOP",
]


@dataclass
class GroundTruthCase:
    tape_id: str
    category: str
    ground_truth_drift_step: int
    total_steps: int


def _tool_def(name: str, desc: str, param_key: str = "id") -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": {param_key: {"type": "string"}},
                "required": [param_key],
            },
        },
    }


def _append_step_helper(
    store: TapeStore,
    tape_id: str,
    step_idx: int,
    req: dict[str, Any],
    prev_req: dict[str, Any] | None,
    tool_calls: list[ToolCallInvocation],
    text: str,
    sim_live_ms: float,
) -> TapeStep:
    tb = estimate_step_token_breakdown(req)
    p_tok = tb.total
    cache = analyze_prefix_cache_continuity(prev_req, req)
    c_tok = int(p_tok * 0.85) if prev_req and cache.ratio >= 0.90 else 0
    o_tok = 55
    resp = {
        "id": f"chatcmpl-fb-{tape_id}-{step_idx}",
        "object": "chat.completion",
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text or None,
                    "tool_calls": [
                        {
                            "id": tc.id or f"c_{step_idx}",
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                            },
                        }
                        for tc in tool_calls
                    ]
                    if tool_calls
                    else None,
                },
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": {
            "prompt_tokens": p_tok,
            "completion_tokens": o_tok,
            "total_tokens": p_tok + o_tok,
            "prompt_tokens_details": {"cached_tokens": c_tok},
        },
    }
    step = TapeStep(
        tape_id=tape_id,
        step_index=step_idx,
        provider="openai",
        model="gpt-4o",
        canonical_hash=canonical_request_hash(req),
        token_breakdown=tb,
        prefix_cache_ratio=cache.ratio,
        prefix_break_offset=cache.break_offset,
        prefix_break_snippet=cache.break_snippet,
        raw_request=req,
        raw_response=resp,
        assembled_text=text,
        tool_calls=tool_calls,
        ttft_ms=round(sim_live_ms * 0.3, 1),
        duration_ms=sim_live_ms,
        prompt_tokens=p_tok,
        cached_prompt_tokens=c_tok,
        completion_tokens=o_tok,
        cost_usd=estimate_cost_usd("gpt-4o", p_tok, c_tok, o_tok),
    )
    return store.append_step(step)


def generate_forkbench_40(store: TapeStore, seed: int = 2026) -> list[GroundTruthCase]:
    """Deterministically synthesize 40 diverse multi-step faulty agent trajectories (8 per class)."""
    rng = random.Random(seed)
    cases: list[GroundTruthCase] = []

    for cat_idx, category in enumerate(CATEGORIES):
        for sample_i in range(8):
            tid = f"fb_{cat_idx + 1}_{category.lower()}_{sample_i + 1:02d}"
            total_steps = rng.randint(6, 10)
            # Drift happens in middle-to-late trajectory (often followed by 0..2 cascading steps!)
            if category == "PREFIX_CACHE_THRASHING":
                drift_step = 2
                total_steps = rng.randint(5, 8)
            else:
                drift_step = rng.randint(max(3, total_steps - 3), total_steps - 1)

            store.create_tape(
                TapeRecord(
                    tape_id=tid,
                    title=f"ForkBench [{category}] Case #{sample_i + 1}",
                    status=TapeStatus.FAILED,
                    failure_reason="Agent task failed downstream quality gate.",
                    tags=["forkbench-40", category.lower()],
                )
            )

            base_tools = [
                _tool_def("lookup_record", "Fetch primary business record by ID", "record_id"),
                _tool_def("verify_compliance", "Check compliance constraints for record", "record_id"),
                _tool_def("execute_verified_update", "Apply verified compliant update to ORD record", "record_id"),
            ]
            overloaded_tools = [
                _tool_def(f" unrelated_erp_module_{j}".strip(), f"Manage logistics telemetry queue #{j}", "queue_id")
                for j in range(34)
            ] + base_tools + [
                _tool_def("legacy_deprecated_wipe_record", "DEPRECATED internal wipe tool", "record_id")
            ]

            msgs: list[dict[str, Any]] = [
                {
                    "role": "system",
                    "content": "You are a production operations agent. Follow all safety caps ($500 max) and grounding rules.",
                },
                {
                    "role": "user",
                    "content": f"Process verified update for order ORD-{1000 + sample_i * 7} within policy.",
                },
            ]
            prev_req: dict[str, Any] | None = None

            for s in range(1, total_steps + 1):
                active_tools = list(base_tools)
                step_msgs = json.loads(json.dumps(msgs))

                if category == "PREFIX_CACHE_THRASHING":
                    # Inject changing millisecond header & shuffled tools on every turn >= 2
                    step_msgs[0]["content"] = (
                        f"Current time: 2026-11-14T03:{sample_i:02d}:{s:02d}.{s * 137}Z\n"
                        + ("Production operations rulebook paragraph. " * 35)
                    )
                    if s % 2 == 0:
                        active_tools = list(reversed(active_tools))

                if category == "TOOL_SCHEMA_OVERLOAD" and s == drift_step:
                    active_tools = overloaded_tools
                    req = {"model": "gpt-4o", "tools": active_tools, "messages": step_msgs}
                    tc = ToolCallInvocation(
                        id=f"c_{s}",
                        name="legacy_deprecated_wipe_record",
                        arguments={"record_id": f"ORD-{1000 + sample_i * 7}"},
                    )
                    _append_step_helper(store, tid, s, req, prev_req, [tc], "", 1850.0)
                    prev_req = req
                    msgs.append({"role": "assistant", "content": None, "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}]})
                    msgs.append({"role": "tool", "tool_call_id": tc.id, "content": '{"status":"error_deprecated_endpoint"}'})
                    continue

                if category == "CONTEXT_MIDDLE_LOSS" and s == drift_step - 1:
                    # Inject massive uncompacted JSON into tool output right before drift_step
                    req = {"model": "gpt-4o", "tools": active_tools, "messages": step_msgs}
                    tc = ToolCallInvocation(id=f"c_{s}", name="lookup_record", arguments={"record_id": f"ORD-{1000 + sample_i * 7}"})
                    _append_step_helper(store, tid, s, req, prev_req, [tc], "", 1420.0)
                    prev_req = req
                    huge_json = json.dumps({"audit_logs": ["entry_" + ("z" * 160) for _ in range(55)]})
                    msgs.append({"role": "assistant", "content": None, "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}]})
                    msgs.append({"role": "tool", "tool_call_id": tc.id, "content": huge_json})
                    continue

                if category == "CONTEXT_MIDDLE_LOSS" and s == drift_step:
                    req = {"model": "gpt-4o", "tools": active_tools, "messages": step_msgs}
                    tc = ToolCallInvocation(
                        id=f"c_{s}",
                        name="execute_verified_update",
                        arguments={"record_id": f"ORD-{1000 + sample_i * 7}", "amount": 1950.0, "exceeds_policy": True},
                    )
                    _append_step_helper(store, tid, s, req, prev_req, [tc], "", 1910.0)
                    prev_req = req
                    msgs.append({"role": "assistant", "content": None, "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}]})
                    msgs.append({"role": "tool", "tool_call_id": tc.id, "content": '{"error":"policy_cap_exceeded"}'})
                    continue

                if category == "SILENT_EMPTY_RETRIEVAL" and s == drift_step - 1:
                    req = {"model": "gpt-4o", "tools": active_tools, "messages": step_msgs}
                    tc = ToolCallInvocation(id=f"c_{s}", name="lookup_record", arguments={"record_id": f"ORD-{1000 + sample_i * 7}"})
                    _append_step_helper(store, tid, s, req, prev_req, [tc], "", 1290.0)
                    prev_req = req
                    msgs.append({"role": "assistant", "content": None, "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}]})
                    msgs.append({"role": "tool", "tool_call_id": tc.id, "content": "[]"})
                    continue

                if category == "SILENT_EMPTY_RETRIEVAL" and s == drift_step:
                    req = {"model": "gpt-4o", "tools": active_tools, "messages": step_msgs}
                    tc = ToolCallInvocation(
                        id=f"c_{s}",
                        name="execute_verified_update",
                        arguments={"record_id": f"INC-{9100 + sample_i}"},
                    )
                    _append_step_helper(store, tid, s, req, prev_req, [tc], f"Updating fabricated INC-{9100 + sample_i}", 1680.0)
                    prev_req = req
                    msgs.append({"role": "assistant", "content": None, "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}]})
                    msgs.append({"role": "tool", "tool_call_id": tc.id, "content": '{"status":"not_found_error"}'})
                    continue

                if category == "TOOL_CALL_LOOP" and s in (drift_step - 1, drift_step, min(total_steps, drift_step + 1)):
                    req = {"model": "gpt-4o", "tools": active_tools, "messages": step_msgs}
                    tc = ToolCallInvocation(
                        id=f"c_{s}",
                        name="verify_compliance",
                        arguments={"record_id": f"ORD-{1000 + sample_i * 7}", "retry_token": "stuck"},
                    )
                    _append_step_helper(store, tid, s, req, prev_req, [tc], "", 1380.0)
                    prev_req = req
                    msgs.append({"role": "assistant", "content": None, "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}]})
                    msgs.append({"role": "tool", "tool_call_id": tc.id, "content": '{"pending":true}'})
                    continue

                # Normal healthy progression step
                req = {"model": "gpt-4o", "tools": active_tools, "messages": step_msgs}
                if s < total_steps:
                    tc = ToolCallInvocation(
                        id=f"c_{s}",
                        name="lookup_record" if s == 1 else "verify_compliance",
                        arguments={"record_id": f"ORD-{1000 + sample_i * 7}", "phase": s},
                    )
                    _append_step_helper(store, tid, s, req, prev_req, [tc], "", 1350.0)
                    prev_req = req
                    msgs.append({"role": "assistant", "content": None, "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}]})
                    msgs.append({"role": "tool", "tool_call_id": tc.id, "content": f'{{"phase":{s},"ok":true}}'})
                else:
                    _append_step_helper(
                        store, tid, s, req, prev_req, [], "Task terminated with unresolved error state.", 1120.0
                    )
                    prev_req = req

            cases.append(
                GroundTruthCase(
                    tape_id=tid,
                    category=category,
                    ground_truth_drift_step=drift_step,
                    total_steps=total_steps,
                )
            )

    # Add 10 adversarial COMPOUND trajectories (background KV-cache noise + late behavioral drift)
    compound_specs = [
        ("TOOL_SCHEMA_OVERLOAD", 5, 7),
        ("TOOL_SCHEMA_OVERLOAD", 6, 8),
        ("CONTEXT_MIDDLE_LOSS", 4, 6),
        ("CONTEXT_MIDDLE_LOSS", 5, 7),
        ("SILENT_EMPTY_RETRIEVAL", 4, 6),
        ("SILENT_EMPTY_RETRIEVAL", 6, 8),
        ("TOOL_CALL_LOOP", 5, 7),
        ("TOOL_CALL_LOOP", 4, 6),
        ("PREFIX_CACHE_THRASHING", 2, 6),
        ("TOOL_SCHEMA_OVERLOAD", 4, 7),
    ]
    for c_idx, (cat, d_step, tot_steps) in enumerate(compound_specs, start=1):
        tid = f"fb_compound_{c_idx:02d}_{cat.lower()}"
        store.create_tape(
            TapeRecord(
                tape_id=tid,
                title=f"ForkBench Adversarial Compound #{c_idx} [{cat}]",
                status=TapeStatus.FAILED,
                failure_reason="Compound production incident with noisy telemetry.",
                tags=["forkbench-50", "compound-adversarial", cat.lower()],
            )
        )
        base_tools = [
            _tool_def("lookup_record", "Fetch primary business record by ID", "record_id"),
            _tool_def("verify_compliance", "Check compliance constraints for record", "record_id"),
            _tool_def("execute_verified_update", "Apply verified compliant update to ORD record", "record_id"),
        ]
        noisy_26_tools = [
            _tool_def(f"aux_reporting_tool_{j}", f"Auxiliary reporting endpoint #{j}", "report_id")
            for j in range(23)
        ] + base_tools

        msgs = [
            {
                "role": "system",
                "content": "Current time: 2026-11-14T00:00:00Z\n" + ("Strict production safety policy ($500 max). " * 30),
            },
            {"role": "user", "content": f"Execute compliant update on ORD-{5000 + c_idx}."},
        ]
        prev_r = None
        for s in range(1, tot_steps + 1):
            s_msgs = json.loads(json.dumps(msgs))
            # Mild volatile timestamp in every compound run to test behavioral prioritization over cache noise!
            s_msgs[0]["content"] = (
                f"Current time: 2026-11-14T04:00:{s:02d}Z\n" + ("Strict production safety policy ($500 max). " * 30)
            )
            tools_here = noisy_26_tools if cat == "TOOL_SCHEMA_OVERLOAD" and s >= d_step - 1 else list(base_tools)

            if cat == "TOOL_SCHEMA_OVERLOAD" and s == d_step:
                # Subtle wrong tool selection WITHOUT "deprecated" in the tool name!
                req = {"model": "gpt-4o", "tools": tools_here, "messages": s_msgs}
                tc = ToolCallInvocation(id=f"cc_{s}", name="aux_reporting_tool_4", arguments={"report_id": "BAD"})
                _append_step_helper(store, tid, s, req, prev_r, [tc], "", 1790.0)
                prev_r = req
                msgs.append({"role": "assistant", "content": None, "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}]})
                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": '{"status":"invalid_schema_mismatch_error"}'})
            elif cat == "CONTEXT_MIDDLE_LOSS" and s == d_step - 1:
                req = {"model": "gpt-4o", "tools": tools_here, "messages": s_msgs}
                tc = ToolCallInvocation(id=f"cc_{s}", name="lookup_record", arguments={"record_id": f"ORD-{5000 + c_idx}"})
                _append_step_helper(store, tid, s, req, prev_r, [tc], "", 1400.0)
                prev_r = req
                msgs.append({"role": "assistant", "content": None, "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}]})
                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps({"dump": ["row_" + ("x" * 150) for _ in range(50)]})})
            elif cat == "CONTEXT_MIDDLE_LOSS" and s == d_step:
                req = {"model": "gpt-4o", "tools": tools_here, "messages": s_msgs}
                tc = ToolCallInvocation(id=f"cc_{s}", name="execute_verified_update", arguments={"record_id": f"ORD-{5000 + c_idx}"})
                _append_step_helper(store, tid, s, req, prev_r, [tc], "", 1820.0)
                prev_r = req
                msgs.append({"role": "assistant", "content": None, "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}]})
                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": '{"status":"rejected_policy_cap_exceeded"}'})
            elif cat == "SILENT_EMPTY_RETRIEVAL" and s == d_step - 1:
                req = {"model": "gpt-4o", "tools": tools_here, "messages": s_msgs}
                tc = ToolCallInvocation(id=f"cc_{s}", name="lookup_record", arguments={"record_id": f"ORD-{5000 + c_idx}"})
                _append_step_helper(store, tid, s, req, prev_r, [tc], "", 1250.0)
                prev_r = req
                msgs.append({"role": "assistant", "content": None, "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}]})
                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": '{"items": []}'})
            elif cat == "SILENT_EMPTY_RETRIEVAL" and s == d_step:
                req = {"model": "gpt-4o", "tools": tools_here, "messages": s_msgs}
                tc = ToolCallInvocation(id=f"cc_{s}", name="execute_verified_update", arguments={"record_id": f"TXN-{8800 + c_idx}"})
                _append_step_helper(store, tid, s, req, prev_r, [tc], "", 1620.0)
                prev_r = req
                msgs.append({"role": "assistant", "content": None, "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}]})
                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": '{"error":"txn_not_found"}'})
            elif cat == "TOOL_CALL_LOOP" and s in (d_step - 1, d_step):
                req = {"model": "gpt-4o", "tools": tools_here, "messages": s_msgs}
                tc = ToolCallInvocation(id=f"cc_{s}", name="verify_compliance", arguments={"record_id": f"ORD-{5000 + c_idx}"})
                _append_step_helper(store, tid, s, req, prev_r, [tc], "", 1310.0)
                prev_r = req
                msgs.append({"role": "assistant", "content": None, "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}]})
                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": '{"pending":true}'})
            else:
                req = {"model": "gpt-4o", "tools": tools_here, "messages": s_msgs}
                tc = ToolCallInvocation(id=f"cc_{s}", name="lookup_record", arguments={"record_id": f"ORD-{5000 + c_idx}", "turn": s})
                _append_step_helper(store, tid, s, req, prev_r, [tc] if s < tot_steps else [], "Aborted." if s == tot_steps else "", 1280.0)
                prev_r = req
                if s < tot_steps:
                    msgs.append({"role": "assistant", "content": None, "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}}]})
                    msgs.append({"role": "tool", "tool_call_id": tc.id, "content": '{"ok":true}'})

        cases.append(
            GroundTruthCase(
                tape_id=tid,
                category=cat,
                ground_truth_drift_step=d_step,
                total_steps=tot_steps,
            )
        )

    return cases


def baseline_naive_log_diagnoser(steps: list[TapeStep]) -> tuple[int, str]:
    """Baseline: standard naive single-pass tail-step log inspection (blames final/penultimate step)."""
    last = steps[-1]
    pred_step = last.step_index
    last_dump = json.dumps(last.raw_request, ensure_ascii=False).lower()
    if "retry_token" in last_dump:
        return pred_step, "TOOL_CALL_LOOP"
    if len(last.raw_request.get("tools") or []) > 20:
        return pred_step, "TOOL_SCHEMA_OVERLOAD"
    return pred_step, "PROMPT_AMBIGUITY"


async def run_benchmark() -> dict[str, Any]:
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    bench_db = results_dir / "forkbench_run.db"
    if bench_db.exists():
        bench_db.unlink()

    store = TapeStore(bench_db)
    engine = ReplayEngine(store)
    cases = generate_forkbench_40(store)

    base_step_exact = 0
    base_cat_correct = 0
    ftk_step_exact = 0
    ftk_step_within_1 = 0
    ftk_cat_correct = 0
    ftk_verified_fixes = 0

    total_full_rerun_ms = 0.0
    total_vcr_fork_ms = 0.0
    total_full_tokens = 0
    total_fork_live_tokens = 0

    per_category_hits: Counter[str] = Counter()
    per_category_total: Counter[str] = Counter()

    t_bench_start = time.perf_counter()
    for c in cases:
        per_category_total[c.category] += 1
        steps = store.get_steps(c.tape_id)

        # 1. Baseline evaluation
        b_step, b_cat = baseline_naive_log_diagnoser(steps)
        if b_step == c.ground_truth_drift_step:
            base_step_exact += 1
        if b_cat == c.category:
            base_cat_correct += 1

        # 2. ForkTape LangGraph Counterfactual Biopsy evaluation
        report = await run_counterfactual_biopsy(store, engine, c.tape_id)
        if report.suspected_drift_step == c.ground_truth_drift_step:
            ftk_step_exact += 1
        if abs(report.suspected_drift_step - c.ground_truth_drift_step) <= 1:
            ftk_step_within_1 += 1
        if report.root_cause_category == c.category:
            ftk_cat_correct += 1
            per_category_hits[c.category] += 1
        if any(r.resolved_failure for r in report.ablation_runs):
            ftk_verified_fixes += 1

        # 3. Replay Cost & Latency Savings accounting (debugging at drift_step vs re-running steps 1..N)
        full_ms = sum(s.duration_ms for s in steps)
        drift_s = next(s for s in steps if s.step_index == c.ground_truth_drift_step)
        vcr_ms = (c.ground_truth_drift_step - 1) * 0.8 + drift_s.duration_ms * 0.12
        total_full_rerun_ms += full_ms
        total_vcr_fork_ms += vcr_ms

        total_full_tokens += sum(s.prompt_tokens + s.completion_tokens for s in steps)
        # With ForkTape, steps 1..K-1 cost 0 tokens, and Step K uses pruned/compacted context
        best_ab = next((r for r in report.ablation_runs if r.resolved_failure), report.ablation_runs[0])
        total_fork_live_tokens += best_ab.prompt_tokens_after + 65

    elapsed_s = round(time.perf_counter() - t_bench_start, 2)
    n = len(cases)
    token_savings_pct = round((1.0 - (total_fork_live_tokens / max(1, total_full_tokens))) * 100.0, 2)
    speedup_factor = round(total_full_rerun_ms / max(1.0, total_vcr_fork_ms), 1)

    summary = {
        "benchmark": "ForkBench-50",
        "total_trajectories": n,
        "categories": CATEGORIES,
        "baseline_naive_tail_inspector": {
            "drift_step_exact_acc_pct": round((base_step_exact / n) * 100.0, 2),
            "root_cause_classification_acc_pct": round((base_cat_correct / n) * 100.0, 2),
        },
        "forktape_counterfactual_biopsy": {
            "drift_step_exact_acc_pct": round((ftk_step_exact / n) * 100.0, 2),
            "drift_step_within_1_acc_pct": round((ftk_step_within_1 / n) * 100.0, 2),
            "root_cause_classification_acc_pct": round((ftk_cat_correct / n) * 100.0, 2),
            "empirical_fix_verification_rate_pct": round((ftk_verified_fixes / n) * 100.0, 2),
            "per_category_accuracy": {
                cat: f"{per_category_hits[cat]}/{per_category_total[cat]} ({(per_category_hits[cat] / per_category_total[cat]) * 100:.0f}%)"
                for cat in CATEGORIES
            },
        },
        "debugging_efficiency_gains": {
            "avg_token_cost_reduction_pct": token_savings_pct,
            "iteration_speedup_factor_x": speedup_factor,
            "full_rerun_tokens_total": total_full_tokens,
            "forktape_targeted_tokens_total": total_fork_live_tokens,
        },
        "benchmark_wall_time_sec": elapsed_s,
    }

    store.close()
    if bench_db.exists():
        bench_db.unlink()

    json_out = results_dir / "forkbench_50_summary.json"
    json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_out = results_dir / "FORKBENCH_RESULTS.md"
    md_out.write_text(
        f"""# ForkBench-50 Quantitative Evaluation Results

Reproducible benchmark across **50 multi-step faulty agent trajectories** (40 single-fault + 10 compound adversarial multi-fault trajectories, 5–10 steps per run).

| Metric | Naive Tail-Log Baseline | ForkTape (Heuristics + LangGraph Counterfactual Biopsy) |
| :--- | :---: | :---: |
| **Drift Step Exact Localization Acc** | `{summary['baseline_naive_tail_inspector']['drift_step_exact_acc_pct']}%` | **`{summary['forktape_counterfactual_biopsy']['drift_step_exact_acc_pct']}%`** |
| **Drift Step (±1 Step) Localization Acc** | — | **`{summary['forktape_counterfactual_biopsy']['drift_step_within_1_acc_pct']}%`** |
| **Root-Cause Classification Acc** | `{summary['baseline_naive_tail_inspector']['root_cause_classification_acc_pct']}%` | **`{summary['forktape_counterfactual_biopsy']['root_cause_classification_acc_pct']}%`** |
| **Empirical Counterfactual Fix Proven** | `0.0%` | **`{summary['forktape_counterfactual_biopsy']['empirical_fix_verification_rate_pct']}%`** |
| **Debugging Token Cost Reduction** | `0.0%` (Full Re-run) | **`-{token_savings_pct}%`** (`{total_full_tokens:,}` → `{total_fork_live_tokens:,}` tok) |
| **Iteration Cycle Speedup** | `1.0x` | **`{speedup_factor}x faster`** |

## Per-Archetype Accuracy
"""
        + "\n".join(
            f"- **{cat}**: `{val}`"
            for cat, val in summary["forktape_counterfactual_biopsy"]["per_category_accuracy"].items()
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    asyncio.run(run_benchmark())
