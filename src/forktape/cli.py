"""Rich/Typer CLI entry point for ForkTape (`forktape` / `ftk`)."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Annotated

import typer
import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from forktape import __version__
from forktape.compiler.pytest_gen import compile_tape_to_pytest_code, export_tape_to_pytest_file
from forktape.config import ForkTapeConfig
from forktape.demo.sample_tapes import seed_demo_showcase
from forktape.doctor.graph import run_counterfactual_biopsy
from forktape.engine.diff import compare_tapes
from forktape.engine.replayer import ReplayEngine
from forktape.interceptor.llm_proxy import create_app
from forktape.interceptor.mcp_bridge import run_stdio_mcp_bridge
from forktape.mcp_server.server import build_mcp_server
from forktape.models.db import TapeStore
from forktape.models.tape import OverridePatch, TapeStatus

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

console = Console()
app = typer.Typer(
    name="forktape",
    help="Deterministic time-travel forking, dual-protocol (LLM SSE + MCP) VCR, and LangGraph counterfactual biopsy for AI Agents.",
    no_args_is_help=True,
    add_completion=False,
)


def _find_free_port(preferred: int = 18787) -> int:
    """Verify `preferred` can actually be bound (avoids Windows Hyper-V WinError 10013 excluded ranges)."""
    for candidate in (preferred, 18787, 28787, 0):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", candidate))
                return int(s.getsockname()[1])
        except OSError:
            continue
    return preferred


def _wait_for_server_ready(server: uvicorn.Server, thread: threading.Thread, timeout_s: float = 5.0) -> None:
    deadline = time.perf_counter() + timeout_s
    while not server.started and thread.is_alive() and time.perf_counter() < deadline:
        time.sleep(0.02)


def _resolve_parent_upstreams(chosen_port: int) -> tuple[str, str]:
    """Capture pre-existing OPENAI_BASE_URL / ANTHROPIC_BASE_URL before proxy env injection."""
    base_cfg = ForkTapeConfig()
    openai_up = base_cfg.openai_upstream
    anth_up = base_cfg.anthropic_upstream

    env_oai = os.environ.get("OPENAI_BASE_URL", "").strip().rstrip("/")
    if env_oai and f"127.0.0.1:{chosen_port}" not in env_oai and "FORKTAPE_OPENAI_UPSTREAM" not in os.environ:
        openai_up = env_oai

    env_anth = os.environ.get("ANTHROPIC_BASE_URL", "").strip().rstrip("/")
    if env_anth and f"127.0.0.1:{chosen_port}" not in env_anth and "FORKTAPE_ANTHROPIC_UPSTREAM" not in os.environ:
        anth_up = env_anth

    return openai_up, anth_up


def _status_badge(status: TapeStatus) -> str:
    palette = {
        TapeStatus.SUCCESS: "[bold green]SUCCESS[/bold green]",
        TapeStatus.FAILED: "[bold red]FAILED[/bold red]",
        TapeStatus.LOOP_ABORTED: "[bold magenta]LOOP_ABORTED[/bold magenta]",
        TapeStatus.RUNNING: "[bold yellow]RUNNING[/bold yellow]",
        TapeStatus.DIVERGED: "[bold cyan]DIVERGED[/bold cyan]",
    }
    return palette.get(status, status.value)


@app.command("version")
def cmd_version() -> None:
    """Print ForkTape version."""
    console.print(f"ForkTape v{__version__}")


@app.command("demo")
def cmd_demo(
    port: Annotated[int, typer.Option("--port", "-p", help="Studio HTTP port")] = 18787,
    db_path: Annotated[str, typer.Option("--db", help="SQLite tape database path")] = ".forktape/demo_tapes.db",
    no_browser: Annotated[bool, typer.Option("--no-browser", help="Do not auto-open browser")] = False,
    seed_only: Annotated[bool, typer.Option("--seed-only", help="Seed demo tapes and exit (for CI/testing)")] = False,
) -> None:
    """Launch ForkTape Time-Travel Studio preloaded with 4 realistic failure scenarios (0 API keys required)."""
    bound_port = port if seed_only else _find_free_port(port)
    cfg = ForkTapeConfig(db_path=Path(db_path), port=bound_port)
    with TapeStore(cfg.db_path) as store:
        seeded = asyncio.run(seed_demo_showcase(store))
        all_tapes = store.list_tapes(limit=100)

        console.print(
            Panel.fit(
                f"[bold cyan]ForkTape Time-Travel Studio v{__version__}[/bold cyan]\n"
                f"Database: [dim]{cfg.db_path}[/dim] ({len(all_tapes)} tapes, {len(seeded)} newly seeded)\n"
                f"Studio URL: [bold underline green]http://127.0.0.1:{bound_port}[/bold underline green]",
                title="Zero-Config Demo Ready",
                border_style="cyan",
            )
        )
        if seed_only:
            return

        fastapi_app = create_app(store=store, config=cfg)
        if not no_browser:
            threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{bound_port}")).start()
        uvicorn.run(fastapi_app, host="127.0.0.1", port=bound_port, log_level="warning")


@app.command("ui")
def cmd_ui(
    port: Annotated[int, typer.Option("--port", "-p", help="Studio HTTP port")] = 18787,
    host: Annotated[str, typer.Option("--host", help="Bind host")] = "127.0.0.1",
    db_path: Annotated[str, typer.Option("--db", help="SQLite tape database path")] = ".forktape/tapes.db",
    open_browser: Annotated[bool, typer.Option("--open/--no-open", help="Open Web Studio in browser")] = True,
) -> None:
    """Start the ForkTape Proxy & Visual Time-Travel Studio server."""
    bound_port = _find_free_port(port) if host == "127.0.0.1" else port
    cfg = ForkTapeConfig(db_path=Path(db_path), host=host, port=bound_port)
    with TapeStore(cfg.db_path) as store:
        fastapi_app = create_app(store=store, config=cfg)
        console.print(
            Panel.fit(
                f"[bold green]ForkTape Proxy & Studio Active[/bold green]\n"
                f"OpenAI Proxy Base : [cyan]http://{host}:{bound_port}/v1[/cyan]\n"
                f"Anthropic Base    : [cyan]http://{host}:{bound_port}[/cyan]\n"
                f"Visual Web Studio : [bold underline green]http://{host}:{bound_port}[/bold underline green]",
                border_style="green",
            )
        )
        if open_browser:
            threading.Timer(1.0, lambda: webbrowser.open(f"http://{host}:{bound_port}")).start()
        uvicorn.run(fastapi_app, host=host, port=bound_port, log_level="info")


@app.command("list")
def cmd_list(
    status: Annotated[str | None, typer.Option("--status", "-s", help="Filter by status")] = None,
    db_path: Annotated[str, typer.Option("--db", help="SQLite tape database path")] = ".forktape/tapes.db",
    limit: Annotated[int, typer.Option("--limit", "-n", help="Max rows")] = 50,
) -> None:
    """List recorded and forked agent tapes."""
    with TapeStore(db_path) as store:
        tapes = store.list_tapes(status=status, limit=limit)

    if not tapes:
        console.print(f"[yellow]No tapes found in {db_path}. Try `forktape demo --seed-only --db {db_path}`.[/yellow]")
        return

    table = Table(title=f"ForkTape Trajectory Vault ({db_path})")
    table.add_column("Tape ID", style="cyan", no_wrap=True)
    table.add_column("Parent / Fork", style="dim")
    table.add_column("Title", max_width=40)
    table.add_column("Status")
    table.add_column("Steps", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("KV Cache", justify="right")
    table.add_column("Cost ($)", justify="right")
    table.add_column("Latency", justify="right")

    for t in tapes:
        lineage = f"{t.parent_tape_id}@S{t.fork_from_step}" if t.parent_tape_id else "root"
        cache_pct = f"{t.cache_hit_rate * 100:.0f}%"
        table.add_row(
            t.tape_id,
            lineage,
            t.title,
            _status_badge(t.status),
            str(t.step_count),
            f"{t.total_tokens:,}",
            cache_pct,
            f"${t.total_cost_usd:.4f}",
            f"{t.total_latency_ms:.0f}ms",
        )
    console.print(table)


@app.command("inspect")
def cmd_inspect(
    tape_id: str = typer.Argument(..., help="Target tape ID"),
    db_path: Annotated[str, typer.Option("--db", help="SQLite tape database path")] = ".forktape/tapes.db",
) -> None:
    """Inspect the causal step timeline, token X-ray, and Prefix KV Cache health of a tape."""
    with TapeStore(db_path) as store:
        tape = store.get_tape(tape_id)
        if tape is None:
            console.print(f"[red]Tape '{tape_id}' not found in {db_path}[/red]")
            raise typer.Exit(code=1)
        steps = store.get_steps(tape_id)
        spans = store.get_mcp_spans(tape_id)

    root = Tree(
        f"[bold cyan]{tape.tape_id}[/bold cyan] — {tape.title} ({_status_badge(tape.status)}) "
        f"[dim]| {tape.total_tokens:,} tok | ${tape.total_cost_usd:.4f} | {tape.total_latency_ms:.0f}ms[/dim]"
    )
    if tape.failure_reason:
        root.add(f"[red]Failure Reason:[/red] {tape.failure_reason}")

    spans_by_step: dict[int, list] = {}
    for sp in spans:
        spans_by_step.setdefault(sp.step_index, []).append(sp)

    for s in steps:
        mode_tag = "[green][VCR REPLAY 0-TOK][/green]" if s.replayed else "[yellow][LIVE][/yellow]"
        cache_color = "green" if s.prefix_cache_ratio >= 0.85 else "red"
        tb = s.token_breakdown
        node = root.add(
            f"Step {s.step_index} {mode_tag} [bold]{s.model}[/bold] "
            f"({s.duration_ms:.1f}ms, {s.prompt_tokens:,} in / {s.completion_tokens} out, "
            f"[{cache_color}]KV LCP {s.prefix_cache_ratio * 100:.0f}%[/{cache_color}])"
        )
        node.add(
            f"[dim]Context X-Ray: sys={tb.system_tokens} | tools_schema={tb.tools_schema_tokens} "
            f"| history={tb.history_tokens} | tool_out={tb.tool_output_tokens}[/dim]"
        )
        for tc in s.tool_calls:
            node.add(f"[magenta]LLM ToolCall ->[/magenta] `{tc.name}({json.dumps(tc.arguments, ensure_ascii=False)[:100]})`")
        if s.assembled_text:
            node.add(f"[white]Assistant:[/white] {s.assembled_text[:140]}")
        for sp in spans_by_step.get(s.step_index, []):
            node.add(f"[blue]MCP [{sp.server_name}][/blue] `{sp.tool_name}` ({sp.duration_ms:.1f}ms)")
        for fl in s.anomaly_flags:
            node.add(f"[bold red]! {fl.code}:[/bold red] {fl.title}")

    console.print(root)


@app.command("fork")
def cmd_fork(
    tape_id: str = typer.Argument(..., help="Source tape ID to fork"),
    step: Annotated[int, typer.Option("--step", "-s", help="Step index K to fork from (steps 1..K-1 replay at 0 cost)")] = 1,
    prune_tools: Annotated[int | None, typer.Option("--prune-tools", help="BM25 semantic prune tools to Top-K")] = None,
    compact_context: Annotated[int | None, typer.Option("--compact-context", help="Compact middle tool outputs > N chars")] = None,
    stabilize_cache: Annotated[bool, typer.Option("--stabilize-cache", help="Strip volatile timestamps & sort schemas for KV Cache")] = False,
    append_system: Annotated[str | None, typer.Option("--append-system", help="Append rule to System Prompt")] = None,
    swap_model: Annotated[str | None, typer.Option("--swap-model", help="Override model name at fork point")] = None,
    live: Annotated[bool, typer.Option("--live", help="Call live upstream API if credentials exist")] = False,
    db_path: Annotated[str, typer.Option("--db", help="SQLite tape database path")] = ".forktape/tapes.db",
) -> None:
    """Execute an instant standalone Time-Travel Fork at Step K and display the side-by-side branch diff."""
    cfg = ForkTapeConfig(db_path=Path(db_path))
    with TapeStore(cfg.db_path) as store:
        engine = ReplayEngine(store, cfg)
        patch = OverridePatch(
            label=f"CLI Fork @ Step {step}",
            prune_tools_top_k=prune_tools,
            compact_context_max_chars=compact_context,
            stabilize_prefix_cache=stabilize_cache,
            system_prompt_append=append_system,
            swap_model=swap_model,
            override_temperature=0.0,
        )
        try:
            child_tape, _ = asyncio.run(
                engine.fork_step_standalone(
                    source_tape_id=tape_id,
                    step_index=step,
                    patch=patch,
                    prefer_live_upstream=live,
                )
            )
            diff = compare_tapes(store, tape_id, child_tape.tape_id)
        except (KeyError, ValueError) as exc:
            console.print(f"[red]Fork error:[/red] {exc}")
            raise typer.Exit(code=1) from exc
    _render_branch_comparison(diff)


@app.command("diff")
def cmd_diff(
    left_tape_id: str = typer.Argument(..., help="Baseline tape ID"),
    right_tape_id: str = typer.Argument(..., help="Forked/comparison tape ID"),
    db_path: Annotated[str, typer.Option("--db", help="SQLite tape database path")] = ".forktape/tapes.db",
) -> None:
    """Compare two trajectory branches side-by-side (steps, tools, tokens, KV cache, replay savings)."""
    with TapeStore(db_path) as store:
        try:
            diff = compare_tapes(store, left_tape_id, right_tape_id)
        except KeyError as exc:
            console.print(f"[red]Diff error:[/red] {exc}")
            raise typer.Exit(code=1) from exc
    _render_branch_comparison(diff)


def _render_branch_comparison(diff) -> None:
    console.print(
        Panel.fit(
            f"Left (Baseline) : [cyan]{diff.left_tape.tape_id}[/cyan] ({diff.left_tape.total_tokens:,} tok, KV Cache {diff.cache_hit_rate_left * 100:.0f}%)\n"
            f"Right (Forked)  : [green]{diff.right_tape.tape_id}[/green] ({diff.right_tape.total_tokens:,} tok, KV Cache {diff.cache_hit_rate_right * 100:.0f}%)\n"
            f"Divergence Step : [bold yellow]Step {diff.divergence_step}[/bold yellow] | "
            f"VCR Fast-Forward Saved: [bold green]{diff.replayed_tokens_saved:,} tokens & {diff.replayed_steps_saved_ms:.0f}ms[/bold green]",
            title="Side-by-Side Trajectory Diff",
            border_style="green",
        )
    )
    tbl = Table()
    tbl.add_column("Step", justify="right")
    tbl.add_column("Relation")
    tbl.add_column("Baseline Call / Output", max_width=42)
    tbl.add_column("Forked Call / Output", max_width=42)
    tbl.add_column("Prompt Tok (L -> R)", justify="right")
    tbl.add_column("KV LCP (L -> R)", justify="right")

    for row in diff.step_diffs:
        l_desc = ", ".join(row.left_tools_called) or row.left_text_preview or "-"
        r_desc = ", ".join(row.right_tools_called) or row.right_text_preview or "-"
        tbl.add_row(
            str(row.step_index),
            row.relation,
            l_desc,
            r_desc,
            f"{row.left_prompt_tokens:,} -> {row.right_prompt_tokens:,}",
            f"{row.left_cache_ratio * 100:.0f}% -> {row.right_cache_ratio * 100:.0f}%",
        )
    console.print(tbl)


@app.command("doctor")
def cmd_doctor(
    tape_id: str = typer.Argument(..., help="Failed tape ID to diagnose"),
    export_pytest: Annotated[str | None, typer.Option("--export-pytest", "-o", help="Export runnable pytest regression file")] = None,
    db_path: Annotated[str, typer.Option("--db", help="SQLite tape database path")] = ".forktape/tapes.db",
) -> None:
    """Run the LangGraph Counterfactual Biopsy Doctor to empirically verify root causes & generate pytest suites."""
    cfg = ForkTapeConfig(db_path=Path(db_path))
    with TapeStore(cfg.db_path) as store:
        engine = ReplayEngine(store, cfg)
        try:
            report = asyncio.run(run_counterfactual_biopsy(store, engine, tape_id))
        except (KeyError, ValueError) as exc:
            console.print(f"[red]Doctor error:[/red] {exc}")
            raise typer.Exit(code=1) from exc

        console.print(
            Panel(
                f"[bold red]Root Cause Verified:[/bold red] [bold yellow]{report.root_cause_category}[/bold yellow] "
                f"@ [bold cyan]Step {report.suspected_drift_step}[/bold cyan] (Confidence: {report.confidence * 100:.0f}%)\n\n"
                f"{report.executive_summary}\n\n"
                f"[bold green]Remediation Playbook:[/bold green]\n{report.verified_fix_recommendation}",
                title=f"Counterfactual Biopsy Report — {tape_id}",
                border_style="yellow",
            )
        )

        ab_table = Table(title="Controlled Counterfactual Ablation Experiments")
        ab_table.add_column("Hypothesis")
        ab_table.add_column("Step", justify="right")
        ab_table.add_column("Resolved?")
        ab_table.add_column("Tokens (Before -> After)", justify="right")
        ab_table.add_column("KV LCP", justify="right")
        ab_table.add_column("Fork Branch", style="cyan")

        for ab in report.ablation_runs:
            verdict = "[bold green]YES (PROVEN)[/bold green]" if ab.resolved_failure else "[dim]NO[/dim]"
            ab_table.add_row(
                ab.hypothesis_name,
                str(ab.step_index),
                verdict,
                f"{ab.prompt_tokens_before:,} -> {ab.prompt_tokens_after:,} ({-ab.token_reduction_pct:+.1f}%)",
                f"{ab.cache_ratio_before * 100:.0f}% -> {ab.cache_ratio_after * 100:.0f}%",
                ab.forked_tape_id or "-",
            )
        console.print(ab_table)

        if export_pytest:
            py_file, bundle_file = export_tape_to_pytest_file(store, tape_id, export_pytest)
            console.print(
                f"[bold green]Exported regression suite:[/bold green] {py_file}\n"
                f"[bold green]Companion VCR fixture   :[/bold green] {bundle_file}"
            )


@app.command("export-pytest")
def cmd_export_pytest(
    tape_id: str = typer.Argument(..., help="Target tape ID"),
    output: Annotated[str | None, typer.Option("--output", "-o", help="Write test_*.py and fixture to path")] = None,
    db_path: Annotated[str, typer.Option("--db", help="SQLite tape database path")] = ".forktape/tapes.db",
) -> None:
    """Compile a recorded tape into a standalone `pytest` CI regression test."""
    with TapeStore(db_path) as store:
        try:
            if output:
                py_file, bundle_file = export_tape_to_pytest_file(store, tape_id, output)
                console.print(
                    f"[green]Wrote pytest module:[/green] {py_file}\n[green]Wrote tape fixture :[/green] {bundle_file}"
                )
            else:
                console.print(compile_tape_to_pytest_code(store, tape_id))
        except (KeyError, ValueError) as exc:
            console.print(f"[red]Export error:[/red] {exc}")
            raise typer.Exit(code=1) from exc


@app.command("record")
def cmd_record(
    command: Annotated[list[str], typer.Argument(help="Agent command to execute (after `--`)")],
    title: Annotated[str, typer.Option("--title", "-t", help="Human-readable run title")] = "CLI Recorded Agent Run",
    db_path: Annotated[str, typer.Option("--db", help="SQLite tape database path")] = ".forktape/tapes.db",
    port: Annotated[int, typer.Option("--port", "-p", help="Local proxy port")] = 0,
) -> None:
    """Zero-intrusion wrapper: starts local SSE/MCP proxy, injects BASE_URL env vars, and records child agent run."""
    chosen_port = port or _find_free_port(18787)
    oai_up, anth_up = _resolve_parent_upstreams(chosen_port)
    cfg = ForkTapeConfig(
        db_path=Path(db_path),
        port=chosen_port,
        openai_upstream=oai_up,
        anthropic_upstream=anth_up,
    )
    with TapeStore(cfg.db_path) as store:
        engine = ReplayEngine(store, cfg)
        sess = engine.sessions.start_record_session(title=title, agent_command=" ".join(command))
        fastapi_app = create_app(store=store, config=cfg, replay_engine=engine)

        uv_cfg = uvicorn.Config(fastapi_app, host="127.0.0.1", port=chosen_port, log_level="error")
        server = uvicorn.Server(uv_cfg)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        _wait_for_server_ready(server, thread)

        child_env = os.environ.copy()
        child_env["FORKTAPE_OPENAI_UPSTREAM"] = oai_up
        child_env["FORKTAPE_ANTHROPIC_UPSTREAM"] = anth_up
        child_env["FORKTAPE_DB_PATH"] = str(cfg.db_path.resolve())
        child_env["FORKTAPE_SESSION_ID"] = sess.session_id
        child_env["OPENAI_BASE_URL"] = f"http://127.0.0.1:{chosen_port}/s/{sess.session_id}/v1"
        child_env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{chosen_port}/s/{sess.session_id}"

        console.print(f"[cyan]Recording session `{sess.tape_id}` on proxy 127.0.0.1:{chosen_port}...[/cyan]")
        proc = subprocess.run(command, env=child_env)
        status = TapeStatus.SUCCESS if proc.returncode == 0 else TapeStatus.FAILED
        final_tape = engine.sessions.complete_session(
            sess.session_id,
            status=status,
            failure_reason=f"Process exited with code {proc.returncode}" if proc.returncode != 0 else None,
        )
        server.should_exit = True
        thread.join(timeout=2.0)

    if final_tape:
        console.print(
            f"[bold green]Saved Tape `{final_tape.tape_id}`[/bold green]: "
            f"{final_tape.step_count} steps, {final_tape.total_tokens:,} tokens, status={final_tape.status.value}"
        )
    raise typer.Exit(code=proc.returncode)


@app.command("replay")
def cmd_replay(
    command: Annotated[list[str], typer.Argument(help="Agent command to execute (after `--`)")],
    tape: Annotated[str, typer.Option("--tape", help="Source tape ID to replay/fork")],
    fork_at: Annotated[int | None, typer.Option("--fork-at", help="Fast-forward steps 1..K-1, go live at step K")] = None,
    prune_tools: Annotated[int | None, typer.Option("--prune-tools", help="BM25 prune tools to Top-K at fork")] = None,
    compact_context: Annotated[int | None, typer.Option("--compact-context", help="Compact middle context > N chars")] = None,
    stabilize_cache: Annotated[bool, typer.Option("--stabilize-cache", help="Stabilize Prefix KV Cache")] = False,
    patch_system: Annotated[Path | None, typer.Option("--patch-system", help="Path to replacement system prompt file")] = None,
    swap_model: Annotated[str | None, typer.Option("--swap-model", help="Override model at fork")] = None,
    strict: Annotated[bool, typer.Option("--strict", help="Fail immediately if pre-fork steps diverge")] = False,
    db_path: Annotated[str, typer.Option("--db", help="SQLite tape database path")] = ".forktape/tapes.db",
    port: Annotated[int, typer.Option("--port", "-p", help="Local proxy port")] = 0,
) -> None:
    """Run an agent process with deterministic VCR fast-forward up to `--fork-at K` and live execution thereafter."""
    chosen_port = port or _find_free_port(18787)
    oai_up, anth_up = _resolve_parent_upstreams(chosen_port)
    cfg = ForkTapeConfig(
        db_path=Path(db_path),
        port=chosen_port,
        openai_upstream=oai_up,
        anthropic_upstream=anth_up,
    )
    with TapeStore(cfg.db_path) as store:
        engine = ReplayEngine(store, cfg)
        sys_replace = patch_system.read_text(encoding="utf-8") if patch_system else None
        patch = OverridePatch(
            label=f"Process Replay Fork @ {fork_at}" if fork_at else "Process VCR Replay",
            system_prompt_replace=sys_replace,
            prune_tools_top_k=prune_tools,
            compact_context_max_chars=compact_context,
            stabilize_prefix_cache=stabilize_cache,
            swap_model=swap_model,
        )
        try:
            sess = engine.sessions.start_replay_or_fork_session(
                source_tape_id=tape,
                fork_from_step=fork_at,
                override_patch=patch,
                strict_matching=strict,
            )
        except KeyError as exc:
            console.print(f"[red]Replay error:[/red] {exc}")
            raise typer.Exit(code=1) from exc

        fastapi_app = create_app(store=store, config=cfg, replay_engine=engine)
        uv_cfg = uvicorn.Config(fastapi_app, host="127.0.0.1", port=chosen_port, log_level="error")
        server = uvicorn.Server(uv_cfg)
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        _wait_for_server_ready(server, thread)

        child_env = os.environ.copy()
        child_env["FORKTAPE_OPENAI_UPSTREAM"] = oai_up
        child_env["FORKTAPE_ANTHROPIC_UPSTREAM"] = anth_up
        child_env["FORKTAPE_DB_PATH"] = str(cfg.db_path.resolve())
        child_env["FORKTAPE_SESSION_ID"] = sess.session_id
        child_env["OPENAI_BASE_URL"] = f"http://127.0.0.1:{chosen_port}/s/{sess.session_id}/v1"
        child_env["ANTHROPIC_BASE_URL"] = f"http://127.0.0.1:{chosen_port}/s/{sess.session_id}"

        proc = subprocess.run(command, env=child_env)
        status = TapeStatus.SUCCESS if proc.returncode == 0 else TapeStatus.FAILED
        engine.sessions.complete_session(sess.session_id, status=status)
        server.should_exit = True
        thread.join(timeout=2.0)

        diff = compare_tapes(store, tape, sess.tape_id)

    _render_branch_comparison(diff)
    raise typer.Exit(code=proc.returncode)


@app.command("mcp")
def cmd_mcp(
    transport: Annotated[str, typer.Option("--transport", help="MCP transport: stdio or streamable-http")] = "stdio",
    db_path: Annotated[str, typer.Option("--db", help="SQLite tape database path")] = ".forktape/tapes.db",
) -> None:
    """Start ForkTape as an MCP Server for Claude Code, Cursor, or Windsurf."""
    cfg = ForkTapeConfig(db_path=Path(db_path))
    with TapeStore(cfg.db_path) as store:
        mcp_srv = build_mcp_server(store=store, config=cfg)
        mcp_srv.run(transport=transport)  # type: ignore[arg-type]


@app.command("mcp-wrap")
def cmd_mcp_wrap(
    command: Annotated[list[str], typer.Argument(help="Downstream MCP stdio server command")],
    server_name: Annotated[str, typer.Option("--server-name", "-n", help="Logical MCP server name")] = "wrapped-mcp",
    session_id: Annotated[str | None, typer.Option("--session", help="Target ForkTape session ID")] = None,
    db_path: Annotated[str | None, typer.Option("--db", help="SQLite tape database path")] = None,
) -> None:
    """Wrap a downstream `stdio` MCP server for transparent span recording and zero-side-effect VCR replay."""
    resolved_db = Path(db_path or os.environ.get("FORKTAPE_DB_PATH", ".forktape/tapes.db"))
    cfg = ForkTapeConfig(db_path=resolved_db)
    with TapeStore(cfg.db_path) as store:
        engine = ReplayEngine(store, cfg)
        sid = session_id or os.environ.get("FORKTAPE_SESSION_ID")
        code = asyncio.run(run_stdio_mcp_bridge(engine, sid, server_name, command))
    raise typer.Exit(code=code)


if __name__ == "__main__":
    app()
