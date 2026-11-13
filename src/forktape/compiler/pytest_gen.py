"""Jinja2 Failure-to-Pytest regression suite generator."""

from __future__ import annotations

import re
from pathlib import Path

from jinja2 import Template

from forktape.doctor.heuristics import locate_suspected_drift_step, scan_tape_anomalies
from forktape.models.db import TapeStore
from forktape.models.tape import OverridePatch

PYTEST_TEMPLATE = Template(
    '''"""Auto-generated ForkTape regression test for tape `{{ tape_id }}` ({{ title }}).

Root-Cause Category: {{ category }}
Suspected Drift Step: {{ drift_step }}
Steps 1..{{ drift_step - 1 }} replay deterministically in <5ms with 0 API token cost.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from forktape.compiler.testing_sdk import (
    TapeReplayHarness,
    assert_no_tool_loop,
    assert_no_ungrounded_entities,
    assert_prefix_cache_healthy,
    assert_tool_not_called,
)
from forktape.models.tape import OverridePatch

BUNDLE_FILE = Path(__file__).parent / "{{ bundle_filename }}"


@pytest.mark.asyncio
async def test_regression_{{ safe_slug }}(tmp_path: Path) -> None:
    harness = TapeReplayHarness.from_bundle_file(BUNDLE_FILE, tmp_path / "ci_replay.db")
    try:
        patch = OverridePatch.model_validate_json(r\'\'\'{{ patch_json }}\'\'\')
        repaired = await harness.fork_at(step_index={{ drift_step }}, patch=patch)

        # 1. Verify prefix steps replayed at zero upstream cost
        for prefix_step in repaired.steps:
            if prefix_step.step_index < {{ drift_step }}:
                assert prefix_step.replayed is True
                assert prefix_step.cost_usd == 0.0

        # 2. Verify behavioral & context-engineering regression assertions
        assert_no_tool_loop(repaired, max_repeat=2)
{%- if forbidden_tool %}
        assert_tool_not_called(repaired, "{{ forbidden_tool }}", step_index={{ drift_step }})
{%- endif %}
{%- if check_grounding %}
        assert_no_ungrounded_entities(repaired, step_index={{ drift_step }})
{%- endif %}
{%- if check_prefix_cache %}
        assert_prefix_cache_healthy(repaired, min_ratio=0.85, step_index={{ drift_step }})
{%- endif %}
    finally:
        harness.close()
'''
)


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return slug[:42] or "agent_tape"


def _default_patch_for_category(category: str) -> OverridePatch:
    if category == "TOOL_SCHEMA_OVERLOAD":
        return OverridePatch(
            label="CI Regression Fix: BM25 Top-5 Tools",
            prune_tools_top_k=5,
            override_temperature=0.0,
        )
    if category == "CONTEXT_MIDDLE_LOSS":
        return OverridePatch(
            label="CI Regression Fix: Compact Context",
            compact_context_max_chars=900,
            override_temperature=0.0,
        )
    if category == "SILENT_EMPTY_RETRIEVAL":
        return OverridePatch(
            label="CI Regression Fix: Empty-Result Grounding Guard",
            system_prompt_append=(
                "CRITICAL GROUNDING RULE: If the preceding search/query tool returned empty results "
                "([] or null), NEVER fabricate IDs or invoke mutation tools. Ask for clarification."
            ),
            override_temperature=0.0,
        )
    if category == "PREFIX_CACHE_THRASHING":
        return OverridePatch(
            label="CI Regression Fix: Stabilize Prefix KV Cache",
            stabilize_prefix_cache=True,
        )
    return OverridePatch(
        label="CI Regression Fix: Loop & Scope Guard",
        prune_tools_top_k=5,
        override_temperature=0.0,
    )


def compile_tape_to_pytest_code(
    store: TapeStore,
    tape_id: str,
    bundle_filename: str | None = None,
) -> str:
    """Compile `tape_id` into a standalone `pytest` Python module string."""
    tape = store.get_tape(tape_id)
    if tape is None:
        raise KeyError(f"Tape '{tape_id}' not found")
    steps = store.get_steps(tape_id)
    mcp_spans = store.get_mcp_spans(tape_id)
    report = store.get_biopsy_report(tape_id)

    if report is not None:
        drift_step = report.suspected_drift_step
        category = report.root_cause_category
        patch = report.recommended_patch or _default_patch_for_category(category)
    else:
        findings = scan_tape_anomalies(tape, steps, mcp_spans)
        drift_step, category = locate_suspected_drift_step(steps, findings, tape.metadata)
        patch = _default_patch_for_category(category)

    orig_drift = next((s for s in steps if s.step_index == drift_step), None)
    forbidden_tool: str | None = None
    if category == "TOOL_SCHEMA_OVERLOAD" and orig_drift and orig_drift.tool_calls:
        cand = orig_drift.tool_calls[0].name
        if any(tag in cand.lower() for tag in ("deprecated", "legacy", "internal_admin", "wrong_")):
            forbidden_tool = cand

    fname = bundle_filename or f"{tape_id}.ftk.json"
    code = PYTEST_TEMPLATE.render(
        tape_id=tape_id,
        title=tape.title,
        category=category,
        drift_step=drift_step,
        bundle_filename=fname,
        safe_slug=_slugify(f"{tape_id}_{category}"),
        patch_json=patch.model_dump_json(),
        forbidden_tool=forbidden_tool,
        check_grounding=(category == "SILENT_EMPTY_RETRIEVAL"),
        check_prefix_cache=(category == "PREFIX_CACHE_THRASHING" and drift_step >= 2),
    )
    return code.strip() + "\n"


def export_tape_to_pytest_file(
    store: TapeStore,
    tape_id: str,
    output_py_path: str | Path,
) -> tuple[Path, Path]:
    """Write both `test_*.py` AND its companion `<tape_id>.ftk.json` fixture to disk."""
    py_path = Path(output_py_path)
    py_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_name = f"{tape_id}.ftk.json"
    bundle_path = py_path.parent / bundle_name

    store.export_bundle_file(tape_id, bundle_path)
    code = compile_tape_to_pytest_code(store, tape_id, bundle_filename=bundle_name)
    py_path.write_text(code, encoding="utf-8")
    return py_path, bundle_path
