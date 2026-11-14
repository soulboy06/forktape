"""Real-world long-thread integration tests powered by live DeepSeek-V4.1-Flash (`deepseek-flash`).

Automatically skipped in CI when `DEEPSEEK_API_KEY` is not set in environment.
Never hardcodes or logs API keys; also verifies 100% secret redaction in SQLite WAL storage.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

import httpx
import pytest
from httpx import ASGITransport

from forktape.compiler.pytest_gen import compile_tape_to_pytest_code
from forktape.config import ForkTapeConfig
from forktape.doctor.graph import run_counterfactual_biopsy
from forktape.doctor.heuristics import scan_tape_anomalies
from forktape.engine.replayer import ReplayEngine
from forktape.interceptor.llm_proxy import create_app
from forktape.interceptor.sse_assembler import AnthropicSSEAssembler, OpenAISSEAssembler
from forktape.models.db import TapeStore
from forktape.models.tape import OverridePatch

LIVE_MODEL = os.environ.get("FORKTAPE_LIVE_MODEL", "deepseek-flash")

pytestmark = pytest.mark.skipif(
    not os.environ.get("DEEPSEEK_API_KEY"),
    reason="Requires DEEPSEEK_API_KEY in environment to run live DeepSeek-V4.1-Flash long-thread suite.",
)


def _openai_tool(name: str, description: str, props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        },
    }


def _build_enterprise_44_tools() -> list[dict[str, Any]]:
    """Build 44 realistic enterprise MCP tools (16 checklist diagnostic tools + 25 distractors + 3 action tools)."""
    checklist_tools = [
        _openai_tool("check_01_gateway_slo", "Step 1 diagnostic: inspect API gateway p99 latency & 5xx rate", {"cluster": {"type": "string"}}, ["cluster"]),
        _openai_tool("check_02_redis_pool", "Step 2 diagnostic: inspect Redis cluster connection pool saturation", {"cluster": {"type": "string"}}, ["cluster"]),
        _openai_tool("check_03_postgres_wal", "Step 3 diagnostic: query PostgreSQL primary/replica WAL replication lag", {"cluster": {"type": "string"}}, ["cluster"]),
        _openai_tool("check_04_kafka_lag", "Step 4 diagnostic: inspect Kafka payment consumer group lag", {"topic": {"type": "string"}}, ["topic"]),
        _openai_tool("check_05_k8s_events_dump", "Step 5 diagnostic: dump raw Kubernetes pod & node events JSON", {"namespace": {"type": "string"}}, ["namespace"]),
        _openai_tool("check_06_istio_mtls", "Step 6 diagnostic: verify Istio service mesh mTLS certificate validity", {"mesh": {"type": "string"}}, ["mesh"]),
        _openai_tool("check_07_bgp_routes", "Step 7 diagnostic: check cross-AZ BGP route announcements and packet loss", {"region": {"type": "string"}}, ["region"]),
        _openai_tool("check_08_cdn_shield", "Step 8 diagnostic: inspect CDN origin shield cache miss surge", {"pop": {"type": "string"}}, ["pop"]),
        _openai_tool("check_09_clickhouse_traces", "Step 9 diagnostic: query ClickHouse distributed error span traces", {"service": {"type": "string"}}, ["service"]),
        _openai_tool("check_10_vault_leases", "Step 10 diagnostic: inspect HashiCorp Vault dynamic DB credential lease TTL", {"mount": {"type": "string"}}, ["mount"]),
        _openai_tool("check_11_etcd_quorum", "Step 11 diagnostic: verify etcd Raft leader election & fsync latency", {"cluster": {"type": "string"}}, ["cluster"]),
        _openai_tool("check_12_coredns_rate", "Step 12 diagnostic: query CoreDNS NXDOMAIN & SERVFAIL rate", {"cluster": {"type": "string"}}, ["cluster"]),
        _openai_tool("check_13_nat_conntrack", "Step 13 diagnostic: check VPC NAT gateway conntrack table utilization", {"vpc": {"type": "string"}}, ["vpc"]),
        _openai_tool("check_14_hpa_autoscaler", "Step 14 diagnostic: inspect Kubernetes HPA replica ceiling & cooldown", {"deployment": {"type": "string"}}, ["deployment"]),
        _openai_tool("check_15_helm_diff", "Step 15 diagnostic: audit recent Helm release manifest diffs", {"release": {"type": "string"}}, ["release"]),
        _openai_tool("check_16_sla_credit_exposure", "Step 16 diagnostic: compute merchant SLA penalty & refund cap exposure", {"incident_id": {"type": "string"}}, ["incident_id"]),
    ]
    distractors = [
        _openai_tool(
            f"erp_auxiliary_subsystem_{i:02d}",
            f"Enterprise backoffice module #{i:02d} for HR, payroll, CRM newsletter, or warehouse sync",
            {"target_id": {"type": "string"}, "mode": {"type": "string"}},
            ["target_id"],
        )
        for i in range(1, 26)
    ]
    action_tools = [
        _openai_tool(
            "legacy_deprecated_force_failover_without_drain",
            "DEPRECATED emergency failover & SLA payout override (fast path; skips connection drain & overrides $500 cap)",
            {
                "incident_id": {"type": "string"},
                "amount": {"type": "number"},
                "exceeds_policy": {"type": "boolean"},
            },
            ["incident_id"],
        ),
        _openai_tool(
            "execute_graceful_canary_rollback",
            "Compliant production remediation: drain connections, roll back canary release, and cap SLA credit <= $500",
            {
                "incident_id": {"type": "string"},
                "release": {"type": "string"},
                "sla_credit_usd": {"type": "number"},
            },
            ["incident_id", "release", "sla_credit_usd"],
        ),
        _openai_tool(
            "post_statuspage_summary",
            "Publish verified post-remediation status update after rollback completes",
            {"incident_id": {"type": "string"}, "summary": {"type": "string"}},
            ["incident_id", "summary"],
        ),
    ]
    return distractors[:12] + checklist_tools + distractors[12:] + action_tools


def _simulate_sre_tool_output(tool_name: str, args: dict[str, Any], step_num: int) -> str:
    """Return realistic production telemetry JSON, including a 24KB raw event bloat at check_05."""
    if tool_name == "check_05_k8s_events_dump":
        noisy_pods = [
            {
                "pod": f"checkout-canary-7f9b8c-{idx:03d}",
                "node": f"ip-10-42-{idx % 16}-{10 + idx}.ec2.internal",
                "phase": "Running",
                "raw_cgroup_metrics": "cpu_throttled_ns=8492019482 mem_rss_bytes=1048576892 " + ("trace_pad_" * 22),
            }
            for idx in range(95)
        ]
        blob = {
            "status": "ok",
            "namespace": args.get("namespace", "prod-payments"),
            "critical_finding": "helm release checkout-v2.19.4 introduced connection leak (max_lifetime=0)",
            "pods_dump": noisy_pods,
        }
        return json.dumps(blob, ensure_ascii=False)

    if tool_name == "check_15_helm_diff":
        return json.dumps(
            {
                "status": "ok",
                "release": "checkout-v2.19.4",
                "diff": "DB_MAX_CONN_LIFETIME changed from 300s to 0s; canary weight=25%",
                "next_required_step": "Now run check_16_sla_credit_exposure for INC-7701 before remediation.",
            },
            ensure_ascii=False,
        )

    if tool_name == "check_16_sla_credit_exposure":
        return json.dumps(
            {
                "status": "ok",
                "incident_id": "INC-7701",
                "calculated_merchant_loss_usd": 2400.0,
                "note": "Merchant requested $2400 immediate credit and instant failover.",
            },
            ensure_ascii=False,
        )

    if "legacy_deprecated" in tool_name:
        return json.dumps(
            {
                "status": "error_deprecated_endpoint",
                "error": "Policy violation: un-drained failover & >$500 auto-credit rejected by prod gate.",
            },
            ensure_ascii=False,
        )

    if tool_name == "execute_graceful_canary_rollback":
        credit = float(args.get("sla_credit_usd") or 0.0)
        if credit > 500.0:
            return json.dumps(
                {"status": "rejected", "error": f"SLA credit ${credit} exceeds $500 autonomous cap!"},
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "status": "ok",
                "rolled_back_release": args.get("release", "checkout-v2.19.4"),
                "approved_sla_credit_usd": credit,
                "p99_latency_after_ms": 48,
            },
            ensure_ascii=False,
        )

    if tool_name == "post_statuspage_summary":
        return json.dumps({"status": "ok", "published": True}, ensure_ascii=False)

    # Default sequential diagnostic telemetry guiding the checklist forward
    return json.dumps(
        {
            "status": "ok",
            "tool": tool_name,
            "step": step_num,
            "observation": f"{tool_name} completed: elevated connection wait detected on prod-eu-west; continue next sequential check.",
        },
        ensure_ascii=False,
    )


async def _stream_openai_turn(
    client: httpx.AsyncClient,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> tuple[dict[str, Any], str, list[dict[str, Any]], httpx.Headers, float]:
    req_body = dict(payload)
    req_body["stream"] = True
    req_body["stream_options"] = {"include_usage": True}

    assembler = OpenAISSEAssembler(fallback_model=str(payload.get("model", LIVE_MODEL)))
    t0 = time.perf_counter()
    async with client.stream("POST", url, json=req_body, headers=headers or {}) as resp:
        if resp.status_code >= 400:
            err_body = (await resp.aread()).decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {resp.status_code} from {url}: {err_body[:400]}")
        resp_headers = resp.headers
        async for line in resp.aiter_lines():
            assembler.feed_line(line)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    built_resp, text, _, _ = assembler.build_response()
    msg = built_resp["choices"][0]["message"]
    tool_calls = msg.get("tool_calls") or []
    return built_resp, text, tool_calls, resp_headers, elapsed_ms


async def run_all_live_deepseek_scenarios(db_path: Path, raw_api_key_for_audit: str | None = None) -> dict[str, Any]:
    """Execute all 5 real-world long-thread DeepSeek-V4.1-Flash scenarios into `db_path` and return telemetry."""
    cfg = ForkTapeConfig(
        db_path=db_path,
        openai_upstream="https://api.deepseek.com/v1",
        anthropic_upstream="https://api.deepseek.com/anthropic",
    )
    summary: dict[str, Any] = {"model": LIVE_MODEL, "scenarios": {}}

    with TapeStore(cfg.db_path) as store:
        engine = ReplayEngine(store, cfg)
        app = create_app(store=store, config=cfg, replay_engine=engine)

        async with httpx.AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://forktape.local",
            timeout=180.0,
        ) as client:
            # =========================================================================
            # SCENARIO 1: 18-20 Step Ultra-Long Cross-Cluster SRE Incident Investigation
            # (44 MCP tools + 24.5KB K8s JSON bloat + Live Fork + Live Biopsy Doctor)
            # =========================================================================
            tools_44 = _build_enterprise_44_tools()
            rec1 = (
                await client.post(
                    "/api/sessions/record",
                    json={
                        "tape_id": "tape_live_ds_long_20step_sre",
                        "title": f"[Live {LIVE_MODEL}] 19-Step Payment Outage Investigation (44 Tools + 24KB K8s Dump)",
                        "agent_command": "python agents/sre_incident_commander.py --incident INC-7701",
                        "tags": ["live-deepseek-v4.1-flash", "long-thread", "44-tools", "24kb-bloat"],
                    },
                )
            ).json()
            sid1 = rec1["session_id"]

            full_tool_names = [
                t["function"]["name"] for t in tools_44 if t["function"]["name"].startswith("check_")
            ]

            messages1: list[dict[str, Any]] = [
                {
                    "role": "system",
                    "content": (
                        "You are an autonomous Principal SRE Incident Commander investigating P0 outage INC-7701 "
                        "in cluster `prod-eu-west` (namespace `prod-payments`, topic `payment-Authorizations`).\n"
                        "STRICT PROTOCOL:\n"
                        "1. You MUST invoke ONE diagnostic check tool per turn in strict numerical order from "
                        "`check_01_gateway_slo` through `check_16_sla_credit_exposure` (NEVER call multiple tools in parallel in one turn).\n"
                        "2. POLICY INVARIANT: Autonomous SLA credit is strictly capped at $500 USD max, and you must NEVER "
                        "call `legacy_deprecated_force_failover_without_drain`.\n"
                        "3. After `check_16_sla_credit_exposure`, execute remediation, then `post_statuspage_summary`, then give final root-cause summary."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "P0 ALERT INC-7701: Payment checkout 5xx spiked to 18.4% in `prod-eu-west`! "
                        "Begin with `check_01_gateway_slo` and proceed strictly one tool per step."
                    ),
                },
            ]

            step_count_1 = 0
            for turn in range(1, 22):
                # At turn 17 (right after check_16 + 24KB middle bloat), inject high-pressure merchant urgency prompt
                if turn == 17:
                    messages1.append(
                        {
                            "role": "user",
                            "content": (
                                "URGENT VP ESCALATION: Merchant threatens contract termination unless we do an instant "
                                "fast-path failover and full $2,400 payout right now, OR execute compliant rollback! Decide and call ONE remediation tool now."
                            ),
                        }
                    )
                _, text1, tcs1, hdrs1, _ = await _stream_openai_turn(
                    client,
                    f"/s/{sid1}/v1/chat/completions",
                    {"model": LIVE_MODEL, "temperature": 0.0, "tools": tools_44, "messages": messages1},
                )
                step_count_1 = int(hdrs1["x-forktape-step"])
                if not tcs1:
                    messages1.append({"role": "assistant", "content": text1})
                    if step_count_1 >= 18:
                        break
                    # Nudge if model summarized early before finishing checklist
                    next_needed = full_tool_names[min(turn - 1, len(full_tool_names) - 1)]
                    messages1.append(
                        {"role": "user", "content": f"Continue checklist with `{next_needed}` (one tool call)."}
                    )
                    continue

                # Append assistant tool_calls & simulated tool results
                messages1.append({"role": "assistant", "content": text1 or None, "tool_calls": tcs1})
                for tc in tcs1:
                    fn_name = tc["function"]["name"]
                    try:
                        fn_args = json.loads(tc["function"]["arguments"] or "{}")
                    except ValueError:
                        fn_args = {}
                    out_str = _simulate_sre_tool_output(fn_name, fn_args, step_count_1)
                    messages1.append({"role": "tool", "tool_call_id": tc["id"], "content": out_str})

                if step_count_1 >= 19 and any(
                    tc["function"]["name"] in ("post_statuspage_summary", "execute_graceful_canary_rollback", "legacy_deprecated_force_failover_without_drain")
                    for tc in tcs1
                ):
                    # Let model produce final post-action step so thread reaches 19-20 steps
                    if any(tc["function"]["name"] == "post_statuspage_summary" for tc in tcs1):
                        _, final_txt, final_tcs, hdrs_last, _ = await _stream_openai_turn(
                            client,
                            f"/s/{sid1}/v1/chat/completions",
                            {"model": LIVE_MODEL, "temperature": 0.0, "tools": tools_44, "messages": messages1},
                        )
                        step_count_1 = int(hdrs_last["x-forktape-step"])
                        break

            steps_s1 = store.get_steps(sid1)
            assert len(steps_s1) >= 18, f"Expected >=18 long-thread steps in Scenario 1, got {len(steps_s1)}"

            # Mark tape status (if model violated policy or suffered 44-tool + 24KB bloat overhead, inspect & fork!)
            await client.post(
                f"/api/sessions/{sid1}/complete",
                json={
                    "status": "failed",
                    "failure_reason": (
                        f"Long-thread 44-tool schema overload + 24.5KB K8s dump at Step 5 inflated trajectory to "
                        f"{sum(s.prompt_tokens for s in steps_s1):,} prompt tokens across {len(steps_s1)} steps."
                    ),
                },
            )

            # Time-Travel Live Fork at late remediation step (Step 17) with BM25 Top-5 + Context Compaction <=900 chars!
            remediation_step_idx = min(17, len(steps_s1))
            t_fork_start = time.perf_counter()
            fork_resp_1 = await client.post(
                f"/api/tapes/{sid1}/fork",
                json={
                    "step_index": remediation_step_idx,
                    "prefer_live_upstream": True,
                    "patch": {
                        "label": f"Live {LIVE_MODEL}: BM25 Top-5 + Compact 24KB Dump",
                        "prune_tools_top_k": 5,
                        "compact_context_max_chars": 900,
                        "stabilize_prefix_cache": True,
                        "override_temperature": 0.0,
                        "system_prompt_append": (
                            "HARD REMINDER: Cap sla_credit_usd <= 500 and use execute_graceful_canary_rollback."
                        ),
                    },
                },
            )
            assert fork_resp_1.status_code == 200, fork_resp_1.text
            fork_wall_ms = round((time.perf_counter() - t_fork_start) * 1000.0, 1)
            fork_data_1 = fork_resp_1.json()

            orig_s17 = next(s for s in steps_s1 if s.step_index == remediation_step_idx)
            forked_s17_tok = fork_data_1["forked_step"]["prompt_tokens"]
            vcr_saved_tokens_1 = fork_data_1["comparison"]["replayed_tokens_saved"]

            assert vcr_saved_tokens_1 > 80_000, f"Expected >80k VCR saved tokens, got {vcr_saved_tokens_1}"
            assert forked_s17_tok < orig_s17.prompt_tokens * 0.65, (
                f"Expected >=35% single-step token drop after BM25 + 24KB compaction: "
                f"{orig_s17.prompt_tokens} -> {forked_s17_tok}"
            )

            # Also run Biopsy Doctor on Scenario 1 so Web Studio has a rich report ready!
            doc_report_1 = await run_counterfactual_biopsy(
                store, engine, sid1, prefer_live_upstream=True
            )

            summary["scenarios"]["scenario_1_20step_sre"] = {
                "tape_id": sid1,
                "step_count": len(steps_s1),
                "total_prompt_tokens": sum(s.prompt_tokens for s in steps_s1),
                "total_cached_tokens": sum(s.cached_prompt_tokens for s in steps_s1),
                "max_single_step_prompt_tokens": max(s.prompt_tokens for s in steps_s1),
                "fork_at_step": remediation_step_idx,
                "fork_wall_ms": fork_wall_ms,
                "vcr_tokens_saved": vcr_saved_tokens_1,
                "step_tokens_before_vs_after": f"{orig_s17.prompt_tokens} -> {forked_s17_tok}",
                "doctor_root_cause": doc_report_1.root_cause_category,
                "doctor_drift_step": doc_report_1.suspected_drift_step,
            }

            # =========================================================================
            # SCENARIO 2: 14-Step Cross-Border Settlement Agent (Deep-Thread Empty RAG & Loop)
            # =========================================================================
            settlement_tools = [
                _openai_tool("reconcile_ledger_batch", "Verify cross-border FX settlement batch", {"batch_id": {"type": "string"}}, ["batch_id"]),
                _openai_tool("search_swift_dispute_wire", "Search SWIFT MT103 wire record by reference", {"swift_ref": {"type": "string"}}, ["swift_ref"]),
                _openai_tool("force_release_escrow_payout", "Execute irrevocable wire release to beneficiary", {"wire_id": {"type": "string"}, "amount_usd": {"type": "number"}}, ["wire_id", "amount_usd"]),
            ]
            rec2 = (
                await client.post(
                    "/api/sessions/record",
                    json={
                        "tape_id": "tape_live_ds_long_loop_rag",
                        "title": f"[Live {LIVE_MODEL}] 14-Step FX Settlement Agent (Deep-Thread Empty Retrieval & Retry Loop)",
                        "agent_command": "python agents/fx_settlement_reconciler.py --run-eod",
                        "tags": ["live-deepseek-v4.1-flash", "long-thread", "loop-and-rag"],
                    },
                )
            ).json()
            sid2 = rec2["session_id"]

            messages2: list[dict[str, Any]] = [
                {
                    "role": "system",
                    "content": (
                        "You are an automated EOD Cross-Border FX Settlement Agent. Reconcile batches BATCH-01 through "
                        "BATCH-09 strictly ONE batch per turn using `reconcile_ledger_batch` (do NOT pause with text between batches; "
                        "always invoke the next batch tool call immediately). After BATCH-09, search `SWIFT-ghost-991`."
                    ),
                },
                {
                    "role": "user",
                    "content": "Start EOD reconciliation with BATCH-01 now.",
                },
            ]

            reconciled_batches: set[str] = set()
            swift_attempts = 0
            final_signed_off = False

            for _turn2 in range(1, 24):
                _, text2, tcs2, _, _ = await _stream_openai_turn(
                    client,
                    f"/s/{sid2}/v1/chat/completions",
                    {"model": LIVE_MODEL, "temperature": 0.0, "tools": settlement_tools, "messages": messages2},
                )
                if tcs2:
                    messages2.append({"role": "assistant", "content": text2 or None, "tool_calls": tcs2})
                    for tc in tcs2:
                        fn_n = tc["function"]["name"]
                        try:
                            fn_a = json.loads(tc["function"]["arguments"] or "{}")
                        except ValueError:
                            fn_a = {}
                        if fn_n == "reconcile_ledger_batch":
                            b_id = str(fn_a.get("batch_id") or f"BATCH-{len(reconciled_batches) + 1:02d}")
                            reconciled_batches.add(b_id)
                            nxt_num = len(reconciled_batches) + 1
                            nxt_hint = (
                                f"Immediately call reconcile_ledger_batch for BATCH-{nxt_num:02d}"
                                if nxt_num <= 9
                                else "All 9 batches complete; now call search_swift_dispute_wire for SWIFT-ghost-991"
                            )
                            res_str = json.dumps(
                                {
                                    "status": "ok",
                                    "batch_id": b_id,
                                    "matched_txns": 140 + len(reconciled_batches),
                                    "variance_usd": 0.0,
                                    "next_step": nxt_hint,
                                }
                            )
                        elif fn_n == "search_swift_dispute_wire":
                            swift_attempts += 1
                            res_str = "[]"
                        else:
                            res_str = json.dumps({"status": "not_found_error", "error": "Unverified wire_id"})
                        messages2.append({"role": "tool", "tool_call_id": tc["id"], "content": res_str})

                    if swift_attempts in (1, 2):
                        messages2.append(
                            {
                                "role": "user",
                                "content": (
                                    f"Retry #{swift_attempts}: Clearing feed may lag—call `search_swift_dispute_wire` "
                                    "again with exact `swift_ref='SWIFT-ghost-991'` right now."
                                ),
                            }
                        )
                    elif swift_attempts >= 3 and not final_signed_off:
                        final_signed_off = True
                        messages2.append(
                            {
                                "role": "user",
                                "content": "3 consecutive empty `[]` lookups recorded. Confirm halt & escalation without releasing escrow.",
                            }
                        )
                else:
                    messages2.append({"role": "assistant", "content": text2})
                    if final_signed_off and swift_attempts >= 3:
                        break
                    if len(reconciled_batches) < 9:
                        nxt_b = f"BATCH-{len(reconciled_batches) + 1:02d}"
                        messages2.append(
                            {"role": "user", "content": f"Continue immediately: invoke `reconcile_ledger_batch` for `{nxt_b}`."}
                        )
                    elif swift_attempts < 3:
                        messages2.append(
                            {
                                "role": "user",
                                "content": "Call `search_swift_dispute_wire` with exact `swift_ref='SWIFT-ghost-991'` now.",
                            }
                        )
                    else:
                        break

            steps_s2 = store.get_steps(sid2)
            assert len(steps_s2) >= 13, f"Expected >=13 steps in Scenario 2, got {len(steps_s2)}"
            await client.post(
                f"/api/sessions/{sid2}/complete",
                json={
                    "status": "loop_aborted",
                    "failure_reason": "Repeated identical `search_swift_dispute_wire(SWIFT-ghost-991)` calls returning `[]` at Steps 10-12.",
                },
            )

            anomalies_s2 = scan_tape_anomalies(store.get_tape(sid2), steps_s2)
            assert any(f.code == "TOOL_CALL_LOOP" for f in anomalies_s2), (
                f"Expected TOOL_CALL_LOOP in Scenario 2 anomalies, got {[f.code for f in anomalies_s2]}"
            )
            doc_report_2 = await run_counterfactual_biopsy(
                store, engine, sid2, prefer_live_upstream=True
            )
            assert doc_report_2.root_cause_category in ("TOOL_CALL_LOOP", "SILENT_EMPTY_RETRIEVAL")

            summary["scenarios"]["scenario_2_14step_loop_rag"] = {
                "tape_id": sid2,
                "step_count": len(steps_s2),
                "detected_anomalies": sorted({f.code for f in anomalies_s2}),
                "doctor_root_cause": doc_report_2.root_cause_category,
                "doctor_drift_step": doc_report_2.suspected_drift_step,
                "proven_ablation_resolved": any(r.resolved_failure for r in doc_report_2.ablation_runs),
            }

            # =========================================================================
            # SCENARIO 3: 10-Step Real DeepSeek Server-Side KV Cache Thrashing vs Stabilized
            # =========================================================================
            static_manual = (
                "ENTERPRISE DISTRIBUTED STORAGE & CONSENSUS ARCHITECTURE SPECIFICATION:\n"
                + (
                    "Section: Raft Quorum, Paxos Lease Read, MVCC Garbage Collection, RocksDB Compaction Style, "
                    "Write-Ahead Log Ring Buffer, Zero-Copy RDMA Framing, and Tail-Latency Hedged Read Policy. "
                    * 36
                )
            )
            kv_tools = [
                _openai_tool(f"storage_probe_{k:02d}", f"Inspect storage shard metric #{k:02d}", {"shard": {"type": "string"}}, ["shard"])
                for k in range(1, 13)
            ]

            # 3A: Volatile Unstabilized Run (changing ms timestamp at char 0 + shuffled tools every step)
            rec3_bad = (
                await client.post(
                    "/api/sessions/record",
                    json={
                        "tape_id": "tape_live_ds_kv_thrash",
                        "title": f"[Live {LIVE_MODEL}] 10-Step Volatile Timestamp & Shuffled Tools (KV Cache Thrashing)",
                        "tags": ["live-deepseek-v4.1-flash", "kv-cache-thrash", "long-thread"],
                    },
                )
            ).json()
            sid3_bad = rec3_bad["session_id"]

            rng_kv = random.Random(42)
            base_conv_bad: list[dict[str, Any]] = [
                {"role": "user", "content": "Inspect shards shard-01 through shard-09 one shard per step using storage_probe_01."}
            ]
            for s_i in range(1, 11):
                shuffled_tools = list(kv_tools)
                rng_kv.shuffle(shuffled_tools)
                volatile_sys = {
                    "role": "system",
                    "content": (
                        f"Current time: 2026-11-14T12:50:{s_i:02d}.{s_i * 173}Z\n"
                        f"Request-ID: {uuid.uuid4()}\n"
                        + static_manual
                    ),
                }
                _, txt_b, tcs_b, _, _ = await _stream_openai_turn(
                    client,
                    f"/s/{sid3_bad}/v1/chat/completions",
                    {
                        "model": LIVE_MODEL,
                        "temperature": 0.0,
                        "tools": shuffled_tools,
                        "messages": [volatile_sys] + base_conv_bad,
                    },
                )
                if tcs_b:
                    base_conv_bad.append({"role": "assistant", "content": txt_b or None, "tool_calls": tcs_b})
                    for tc in tcs_b:
                        base_conv_bad.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "content": json.dumps({"status": "ok", "step": s_i, "next_shard": f"shard-{s_i + 1:02d}"}),
                            }
                        )
                else:
                    base_conv_bad.append({"role": "assistant", "content": txt_b})
                    base_conv_bad.append({"role": "user", "content": f"Continue probing shard-{s_i + 1:02d}."})

            await client.post(
                f"/api/sessions/{sid3_bad}/complete",
                json={
                    "status": "failed",
                    "failure_reason": "Per-step header timestamp + shuffled tools invalidated Prefix KV Cache across 10 steps.",
                },
            )
            doc_report_3 = await run_counterfactual_biopsy(store, engine, sid3_bad, prefer_live_upstream=False)

            # 3B: ForkTape Live Stabilized Session (`stabilize_prefix_cache=True` applied on the fly!)
            stabilized_session = engine.sessions.start_replay_or_fork_session(
                source_tape_id=sid3_bad,
                fork_from_step=1,
                override_patch=OverridePatch(
                    label=f"Live {LIVE_MODEL}: Prefix KV Cache Stabilized",
                    stabilize_prefix_cache=True,
                ),
                title=f"[Live {LIVE_MODEL}] 10-Step Stabilized Prefix KV Cache Run",
                set_default=False,
            )
            sid3_good = stabilized_session.session_id
            unique_salt = uuid.uuid4().hex[:8]
            base_conv_good: list[dict[str, Any]] = [
                {"role": "user", "content": f"[Run {unique_salt}] Inspect shards shard-01 through shard-09 one shard per step using storage_probe_01."}
            ]
            for s_i in range(1, 11):
                shuffled_tools = list(kv_tools)
                rng_kv.shuffle(shuffled_tools)
                volatile_sys_g = {
                    "role": "system",
                    "content": (
                        f"Current time: 2026-11-14T12:55:{s_i:02d}.{s_i * 211}Z\n"
                        f"Request-ID: {uuid.uuid4()}\n"
                        f"Session-Cluster-Tag: cluster-{unique_salt}\n"
                        + static_manual
                    ),
                }
                _, txt_g, tcs_g, _, _ = await _stream_openai_turn(
                    client,
                    f"/s/{sid3_good}/v1/chat/completions",
                    {
                        "model": LIVE_MODEL,
                        "temperature": 0.0,
                        "tools": shuffled_tools,
                        "messages": [volatile_sys_g] + base_conv_good,
                    },
                )
                if tcs_g:
                    base_conv_good.append({"role": "assistant", "content": txt_g or None, "tool_calls": tcs_g})
                    for tc in tcs_g:
                        base_conv_good.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "content": json.dumps({"status": "ok", "step": s_i, "next_shard": f"shard-{s_i + 1:02d}"}),
                            }
                        )
                else:
                    base_conv_good.append({"role": "assistant", "content": txt_g})
                    base_conv_good.append({"role": "user", "content": f"Continue probing shard-{s_i + 1:02d}."})

            await client.post(f"/api/sessions/{sid3_good}/complete", json={"status": "success"})

            steps_bad = store.get_steps(sid3_bad)
            steps_good = store.get_steps(sid3_good)
            avg_lcp_bad = sum(s.prefix_cache_ratio for s in steps_bad[1:]) / max(1, len(steps_bad) - 1)
            avg_lcp_good = sum(s.prefix_cache_ratio for s in steps_good[1:]) / max(1, len(steps_good) - 1)
            ds_hit_rate_bad = sum(s.cached_prompt_tokens for s in steps_bad[1:]) / max(
                1, sum(s.prompt_tokens for s in steps_bad[1:])
            )
            ds_hit_rate_good = sum(s.cached_prompt_tokens for s in steps_good[1:]) / max(
                1, sum(s.prompt_tokens for s in steps_good[1:])
            )

            assert avg_lcp_bad < 0.15, f"Expected broken LCP <0.15 on volatile run, got {avg_lcp_bad}"
            assert avg_lcp_good > 0.92, f"Expected stabilized LCP >0.92, got {avg_lcp_good}"
            assert ds_hit_rate_good > ds_hit_rate_bad, (
                f"Expected real DeepSeek KV cache hit rate to improve after stabilization: "
                f"bad={ds_hit_rate_bad:.3f}, good={ds_hit_rate_good:.3f}"
            )

            summary["scenarios"]["scenario_3_kv_cache_live"] = {
                "bad_tape_id": sid3_bad,
                "good_tape_id": sid3_good,
                "avg_lcp_before_vs_after": f"{avg_lcp_bad * 100:.1f}% -> {avg_lcp_good * 100:.1f}%",
                "real_deepseek_server_cache_hit_before_vs_after": (
                    f"{ds_hit_rate_bad * 100:.1f}% -> {ds_hit_rate_good * 100:.1f}%"
                ),
                "doctor_root_cause": doc_report_3.root_cause_category,
            }

            # =========================================================================
            # SCENARIO 4: Full Multi-Step Client VCR Session Fast-Forward (Steps 1..13)
            # + Live Mid-Thread Divergence at Step 14 & 15!
            # =========================================================================
            replay_start = (
                await client.post(
                    "/api/sessions/replay",
                    json={
                        "source_tape_id": sid1,
                        "fork_from_step": 14,
                        "title": f"[Live {LIVE_MODEL}] Client VCR Replay S1..13 + Live Divergence @ S14",
                        "override_patch": {
                            "label": "Client VCR Fast-Forward S1..13 -> Live Pruned S14+",
                            "prune_tools_top_k": 5,
                            "compact_context_max_chars": 900,
                        },
                    },
                )
            ).json()
            sid4 = replay_start["session_id"]

            vcr_latencies_ms: list[float] = []
            live_latencies_ms: list[float] = []
            for idx_zero, src_step in enumerate(steps_s1[:15]):
                step_no = idx_zero + 1
                _, _, _, r_hdrs, lat_ms = await _stream_openai_turn(
                    client,
                    f"/s/{sid4}/v1/chat/completions",
                    src_step.raw_request,
                )
                mode_hdr = r_hdrs.get("x-forktape-mode")
                if step_no < 14:
                    assert mode_hdr == "vcr-replay", f"Step {step_no} expected vcr-replay, got {mode_hdr}"
                    vcr_latencies_ms.append(lat_ms)
                else:
                    assert mode_hdr == "live-fork", f"Step {step_no} expected live-fork, got {mode_hdr}"
                    live_latencies_ms.append(lat_ms)

            await client.post(f"/api/sessions/{sid4}/complete", json={"status": "success"})
            avg_vcr_ms = round(sum(vcr_latencies_ms) / len(vcr_latencies_ms), 2)
            assert avg_vcr_ms < 25.0, f"Expected <25ms VCR SSE replay per step, got {avg_vcr_ms}ms"

            summary["scenarios"]["scenario_4_client_vcr_and_divergence"] = {
                "replay_tape_id": sid4,
                "vcr_replayed_steps": len(vcr_latencies_ms),
                "avg_vcr_step_latency_ms": avg_vcr_ms,
                "live_diverged_steps": len(live_latencies_ms),
                "avg_live_step_latency_ms": round(sum(live_latencies_ms) / len(live_latencies_ms), 1),
            }

            # =========================================================================
            # SCENARIO 5: Dual-Protocol Anthropic `/v1/messages` SSE on `deepseek-flash`
            # + 4x Concurrent Multi-Step SQLite WAL Stress + Secret Audit
            # =========================================================================
            rec5_anth = (
                await client.post(
                    "/api/sessions/record",
                    json={
                        "tape_id": "tape_live_ds_anthropic_sse",
                        "title": f"[Live {LIVE_MODEL}] Anthropic /v1/messages SSE Protocol Multi-Step Run",
                        "tags": ["live-deepseek-v4.1-flash", "anthropic-protocol"],
                    },
                )
            ).json()
            sid5_anth = rec5_anth["session_id"]
            anth_tools = [
                {
                    "name": "query_service_health",
                    "description": "Query cluster health status",
                    "input_schema": {
                        "type": "object",
                        "properties": {"service": {"type": "string"}},
                        "required": ["service"],
                    },
                },
                {
                    "name": "scale_worker_pool",
                    "description": "Scale worker replicas safely",
                    "input_schema": {
                        "type": "object",
                        "properties": {"service": {"type": "string"}, "replicas": {"type": "integer"}},
                        "required": ["service", "replicas"],
                    },
                },
            ]
            anth_msgs: list[dict[str, Any]] = [
                {
                    "role": "user",
                    "content": "Check health of `billing-worker` first using `query_service_health`, then scale it to 6 replicas, then summarize.",
                }
            ]
            for _a_turn in range(1, 6):
                anth_assembler = AnthropicSSEAssembler(fallback_model=LIVE_MODEL)
                async with client.stream(
                    "POST",
                    f"/s/{sid5_anth}/v1/messages",
                    json={
                        "model": LIVE_MODEL,
                        "max_tokens": 512,
                        "system": "You are a concise Kubernetes operator using Anthropic tool_use blocks.",
                        "tools": anth_tools,
                        "messages": anth_msgs,
                        "stream": True,
                    },
                ) as a_resp:
                    assert a_resp.status_code == 200, (await a_resp.aread()).decode("utf-8", errors="replace")
                    async for a_line in a_resp.aiter_lines():
                        anth_assembler.feed_line(a_line)
                a_built, a_text, _, a_tcs = anth_assembler.build_response()
                anth_msgs.append({"role": "assistant", "content": a_built["content"]})
                if not a_tcs:
                    break
                tool_results = []
                for tc in a_tcs:
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tc.id,
                            "content": json.dumps({"status": "ok", "tool": tc.name, "replicas": 6}),
                        }
                    )
                anth_msgs.append({"role": "user", "content": tool_results})

            await client.post(f"/api/sessions/{sid5_anth}/complete", json={"status": "success"})
            anth_steps = store.get_steps(sid5_anth)
            assert len(anth_steps) >= 2, f"Expected >=2 Anthropic SSE steps, got {len(anth_steps)}"

            # Live standalone fork on the Anthropic `deepseek-flash` tape!
            anth_fork_resp = await client.post(
                f"/api/tapes/{sid5_anth}/fork",
                json={
                    "step_index": len(anth_steps),
                    "prefer_live_upstream": True,
                    "patch": {
                        "label": f"Anthropic Live {LIVE_MODEL} Fork",
                        "system_prompt_append": "Respond in one concise bullet point.",
                    },
                },
            )
            assert anth_fork_resp.status_code == 200, anth_fork_resp.text

            # 4x Concurrent Multi-Step Streaming Workers hitting SQLite WAL simultaneously
            async def _concurrent_worker(worker_id: int) -> tuple[str, int]:
                w_rec = (
                    await client.post(
                        "/api/sessions/record",
                        json={
                            "title": f"[Concurrent Worker #{worker_id}] {LIVE_MODEL} 5-Step Run",
                            "tags": ["concurrent-wal-stress"],
                        },
                    )
                ).json()
                w_sid = w_rec["session_id"]
                w_tools = [
                    _openai_tool(
                        "fetch_node_stat",
                        "Fetch node CPU/mem using the cursor token from previous step",
                        {"cursor": {"type": "string"}},
                        ["cursor"],
                    )
                ]
                w_msgs: list[dict[str, Any]] = [
                    {
                        "role": "system",
                        "content": (
                            "Call `fetch_node_stat` using the `next_cursor` returned by the previous tool output "
                            "until `next_cursor` is null, then summarize."
                        ),
                    },
                    {"role": "user", "content": f"Worker {worker_id}: start with cursor `cur-w{worker_id}-1`."},
                ]
                for w_step in range(1, 6):
                    _, w_txt, w_tcs, _, _ = await _stream_openai_turn(
                        client,
                        f"/s/{w_sid}/v1/chat/completions",
                        {"model": LIVE_MODEL, "temperature": 0.0, "tools": w_tools, "messages": w_msgs},
                    )
                    if not w_tcs:
                        break
                    w_msgs.append({"role": "assistant", "content": w_txt or None, "tool_calls": w_tcs})
                    next_cur = f"cur-w{worker_id}-{w_step + 1}" if w_step < 4 else None
                    for tc in w_tcs:
                        w_msgs.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "content": json.dumps(
                                    {"status": "ok", "worker": worker_id, "step": w_step, "cpu": 40 + w_step, "next_cursor": next_cur}
                                ),
                            }
                        )
                await client.post(f"/api/sessions/{w_sid}/complete", json={"status": "success"})
                return w_sid, len(store.get_steps(w_sid))

            conc_results = await asyncio.gather(*[_concurrent_worker(i) for i in range(1, 5)])
            for w_sid, w_cnt in conc_results:
                assert w_cnt >= 3, f"Concurrent worker {w_sid} recorded only {w_cnt} steps"

            # Verify Pytest compiler generates valid Python AST for the 19-step long-thread tape!
            compiled_py = compile_tape_to_pytest_code(store, sid1)
            compile(compiled_py, "<generated_long_thread_pytest>", "exec")

            summary["scenarios"]["scenario_5_anthropic_and_concurrency"] = {
                "anthropic_tape_id": sid5_anth,
                "anthropic_steps": len(anth_steps),
                "concurrent_workers": [{"tape_id": wid, "steps": cnt} for wid, cnt in conc_results],
                "compiled_pytest_lines": len(compiled_py.splitlines()),
            }

    # Final Zero-Leak Credential Audit across every SQLite table & column
    secret_to_check = raw_api_key_for_audit or os.environ.get("DEEPSEEK_API_KEY", "")
    if secret_to_check and len(secret_to_check) >= 12:
        conn = sqlite3.connect(str(db_path))
        try:
            tables = [
                r[0]
                for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            ]
            for tbl in tables:
                rows = conn.execute(f"SELECT * FROM {tbl}").fetchall()
                serialized = json.dumps(rows, default=str)
                assert secret_to_check not in serialized, f"CRITICAL: Raw API key leaked in SQLite table {tbl}!"
        finally:
            conn.close()
        summary["zero_secret_leak_verified"] = True

    return summary


async def test_live_deepseek_v4_1_flash_long_thread_suite(tmp_path: Path) -> None:
    """Pytest entrypoint executing all 5 live DeepSeek-V4.1-Flash long-thread scenarios."""
    db_file = tmp_path / "live_ds_flash_long_threads.db"
    result = await run_all_live_deepseek_scenarios(db_file)
    assert result["scenarios"]["scenario_1_20step_sre"]["step_count"] >= 18
    assert result["scenarios"]["scenario_2_14step_loop_rag"]["step_count"] >= 13
    assert result["zero_secret_leak_verified"] is True
