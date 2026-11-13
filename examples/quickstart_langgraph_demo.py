"""Minimal runnable example: Time-Travel Forking, LangGraph Biopsy, and Pytest Export in 30 lines."""

from __future__ import annotations

import asyncio
from pathlib import Path

from forktape.compiler.pytest_gen import export_tape_to_pytest_file
from forktape.demo.sample_tapes import seed_demo_showcase
from forktape.doctor.graph import run_counterfactual_biopsy
from forktape.engine.diff import compare_tapes
from forktape.engine.replayer import ReplayEngine
from forktape.models.db import TapeStore
from forktape.models.tape import OverridePatch


async def main() -> None:
    store = TapeStore(".forktape/quickstart.db")
    engine = ReplayEngine(store)

    # 1. Seed realistic failed trajectories
    await seed_demo_showcase(store)
    tape_id = "tape_demo_mcp_overload"

    # 2. Fork from Step 5 with BM25 Top-5 Tool Pruning (Steps 1..4 fast-forward at 0 Token cost!)
    child_tape, forked_step = await engine.fork_step_standalone(
        source_tape_id=tape_id,
        step_index=5,
        patch=OverridePatch(label="BM25 Top-5 Fix", prune_tools_top_k=5),
    )
    diff = compare_tapes(store, tape_id, child_tape.tape_id)
    print(f"[Fork] Saved {diff.replayed_tokens_saved:,} tokens across steps 1..4!")
    print(f"[Fork] New Step 5 tool call: {[tc.name for tc in forked_step.tool_calls]}")

    # 3. Run LangGraph Counterfactual Biopsy & export runnable pytest regression test
    report = await run_counterfactual_biopsy(store, engine, tape_id)
    print(f"[Biopsy] Verified Root Cause: {report.root_cause_category} @ Step {report.suspected_drift_step}")

    py_path, _ = export_tape_to_pytest_file(store, tape_id, Path(".forktape/generated/test_order_agent.py"))
    print(f"[Compiler] Exported standalone CI test -> {py_path}")
    store.close()


if __name__ == "__main__":
    asyncio.run(main())
