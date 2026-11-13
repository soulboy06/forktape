"""End-to-end tests for Phase 3: LangGraph Counterfactual Biopsy, Pytest Compiler, MCP Server, and CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from forktape.cli import app as cli_app
from forktape.compiler.pytest_gen import export_tape_to_pytest_file
from forktape.demo.sample_tapes import seed_demo_showcase
from forktape.mcp_server.server import build_mcp_server
from forktape.models.db import TapeStore


@pytest.mark.asyncio
async def test_langgraph_biopsy_verifies_all_four_failure_archetypes_and_compiles_runnable_pytest(
    tmp_path: Path,
) -> None:
    db_file = tmp_path / "showcase.db"
    store = TapeStore(db_file)
    seeded_ids = await seed_demo_showcase(store)
    assert len(seeded_ids) == 4

    expected = {
        "tape_demo_mcp_overload": ("TOOL_SCHEMA_OVERLOAD", 5),
        "tape_demo_context_rot": ("CONTEXT_MIDDLE_LOSS", 3),
        "tape_demo_empty_retrieval": ("SILENT_EMPTY_RETRIEVAL", 2),
        "tape_demo_kv_cache_thrash": ("PREFIX_CACHE_THRASHING", 2),
    }

    exported_test_dir = tmp_path / "generated_regression_suite"
    for tid, (exp_cat, exp_step) in expected.items():
        report = store.get_biopsy_report(tid)
        assert report is not None, f"Missing report for {tid}"
        assert report.root_cause_category == exp_cat, (
            f"Expected {exp_cat} for {tid}, got {report.root_cause_category}"
        )
        assert report.suspected_drift_step == exp_step, (
            f"Expected drift step {exp_step} for {tid}, got {report.suspected_drift_step}"
        )
        assert any(r.resolved_failure for r in report.ablation_runs), (
            f"Expected at least 1 proven counterfactual ablation for {tid}"
        )

        # Compile to disk as standalone pytest file + .ftk.json fixture
        py_path, bundle_path = export_tape_to_pytest_file(
            store, tid, exported_test_dir / f"test_{tid}.py"
        )
        assert py_path.exists()
        assert bundle_path.exists()

    store.close()

    # Actually execute the generated pytest regression suite in a real child process!
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(exported_test_dir),
            f"--basetemp={tmp_path / 'sub_pytest_tmp'}",
            "-v",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"Generated pytest suite failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert "4 passed" in proc.stdout


@pytest.mark.asyncio
async def test_mcp_server_exposes_all_six_self_debugging_tools(tmp_path: Path) -> None:
    store = TapeStore(tmp_path / "mcp_srv.db")
    await seed_demo_showcase(store)
    mcp_srv = build_mcp_server(store=store)

    tools = await mcp_srv.list_tools()
    tool_names = {t.name for t in tools}
    assert {
        "list_tapes",
        "inspect_tape_timeline",
        "inspect_step_detail",
        "fork_and_ablate_step",
        "run_biopsy_doctor",
        "export_pytest_regression",
    }.issubset(tool_names)

    list_res = await mcp_srv.call_tool("list_tapes", {"status": "failed", "limit": 10})
    assert "tape_demo_mcp_overload" in str(list_res)

    fork_res = await mcp_srv.call_tool(
        "fork_and_ablate_step",
        {"tape_id": "tape_demo_mcp_overload", "step_index": 5, "prune_tools_top_k": 5},
    )
    assert "forked_tape_id" in str(fork_res)
    store.close()


def test_cli_commands_demo_list_inspect_fork_and_doctor(tmp_path: Path) -> None:
    runner = CliRunner()
    db_str = str(tmp_path / "cli_test.db")

    r_demo = runner.invoke(cli_app, ["demo", "--seed-only", "--db", db_str])
    assert r_demo.exit_code == 0, r_demo.output

    r_list = runner.invoke(cli_app, ["list", "--db", db_str])
    assert r_list.exit_code == 0
    assert "tape_demo_mcp_overload" in r_list.output

    r_insp = runner.invoke(cli_app, ["inspect", "tape_demo_mcp_overload", "--db", db_str])
    assert r_insp.exit_code == 0
    assert "Context X-Ray" in r_insp.output

    r_fork = runner.invoke(
        cli_app,
        ["fork", "tape_demo_mcp_overload", "--step", "5", "--prune-tools", "5", "--db", db_str],
    )
    assert r_fork.exit_code == 0
    assert "Side-by-Side Trajectory Diff" in r_fork.output

    out_py = tmp_path / "cli_exported" / "test_cli_reg.py"
    r_doc = runner.invoke(
        cli_app,
        ["doctor", "tape_demo_empty_retrieval", "--export-pytest", str(out_py), "--db", db_str],
    )
    assert r_doc.exit_code == 0
    assert "SILENT_EMPTY_RETRIEVAL" in r_doc.output
    assert out_py.exists()
