"""Public deterministic assertion SDK for ForkTape-compiled `pytest` regression suites."""

from __future__ import annotations

import json
import re
from pathlib import Path

from forktape.engine.replayer import ReplayEngine
from forktape.models.db import TapeStore
from forktape.models.tape import OverridePatch, TapeBundle, TapeRecord, TapeStep

ENTITY_ID_RE = re.compile(r"\b(?:ORD|INC|TKT|USR|INV|TXN)[-_]\d{3,}\b")


class TapeReplayHarness:
    """Self-contained CI harness for replaying & counterfactually verifying `.ftk.json` tapes."""

    def __init__(self, store: TapeStore, tape_id: str) -> None:
        self.store = store
        self.engine = ReplayEngine(store)
        self.tape_id = tape_id

    @classmethod
    def from_bundle_file(cls, bundle_path: str | Path, tmp_db_path: str | Path) -> TapeReplayHarness:
        store = TapeStore(tmp_db_path)
        imported = store.import_bundle_file(bundle_path)
        return cls(store=store, tape_id=imported.tape_id)

    @classmethod
    def from_bundle_json(cls, raw_json: str, tmp_db_path: str | Path) -> TapeReplayHarness:
        store = TapeStore(tmp_db_path)
        bundle = TapeBundle.model_validate_json(raw_json)
        imported = store.import_bundle(bundle)
        return cls(store=store, tape_id=imported.tape_id)

    @property
    def tape(self) -> TapeRecord:
        rec = self.store.get_tape(self.tape_id)
        assert rec is not None, f"Tape '{self.tape_id}' missing from harness store"
        return rec

    @property
    def steps(self) -> list[TapeStep]:
        return self.store.get_steps(self.tape_id)

    async def fork_at(
        self,
        step_index: int,
        patch: OverridePatch,
        prefer_live_upstream: bool = False,
    ) -> TapeReplayHarness:
        child_tape, _ = await self.engine.fork_step_standalone(
            source_tape_id=self.tape_id,
            step_index=step_index,
            patch=patch,
            prefer_live_upstream=prefer_live_upstream,
        )
        return TapeReplayHarness(store=self.store, tape_id=child_tape.tape_id)

    def close(self) -> None:
        self.store.close()


def assert_no_tool_loop(harness: TapeReplayHarness, max_repeat: int = 1) -> None:
    """Assert no identical `(tool_name, canonical_arguments)` call repeats more than `max_repeat` times."""
    counts: dict[str, int] = {}
    for step in harness.steps:
        for tc in step.tool_calls:
            sig = f"{tc.name}({json.dumps(tc.arguments, sort_keys=True, ensure_ascii=False)})"
            counts[sig] = counts.get(sig, 0) + 1
            if counts[sig] > max_repeat:
                raise AssertionError(
                    f"Tool call loop detected on `{sig}`: called {counts[sig]} times (max_repeat={max_repeat})."
                )


def assert_tool_called(
    harness: TapeReplayHarness,
    name: str,
    step_index: int | None = None,
) -> None:
    """Assert that tool `name` was called (optionally at exact `step_index`)."""
    candidates = (
        [s for s in harness.steps if s.step_index == step_index]
        if step_index is not None
        else harness.steps
    )
    called = [tc.name for s in candidates for tc in s.tool_calls]
    if name not in called:
        scope = f"at step {step_index}" if step_index is not None else "in trajectory"
        raise AssertionError(f"Expected tool `{name}` {scope}, but observed calls: {called}")


def assert_tool_not_called(
    harness: TapeReplayHarness,
    forbidden_name: str,
    step_index: int | None = None,
) -> None:
    """Assert that a deprecated, destructive, or distractor tool was NOT invoked."""
    candidates = (
        [s for s in harness.steps if s.step_index == step_index]
        if step_index is not None
        else harness.steps
    )
    for s in candidates:
        for tc in s.tool_calls:
            if tc.name == forbidden_name:
                raise AssertionError(
                    f"Forbidden tool `{forbidden_name}` was invoked at step {s.step_index} with args={tc.arguments}"
                )


def assert_prefix_cache_healthy(
    harness: TapeReplayHarness,
    min_ratio: float = 0.85,
    step_index: int | None = None,
) -> None:
    """Assert Prefix KV Cache continuity ratio stays >= `min_ratio` across multi-turn steps."""
    candidates = [
        s
        for s in harness.steps
        if s.step_index >= 2 and (step_index is None or s.step_index == step_index)
    ]
    for s in candidates:
        if s.prefix_cache_ratio < min_ratio:
            raise AssertionError(
                f"Prefix KV Cache thrashing at step {s.step_index}: LCP ratio={s.prefix_cache_ratio:.2%} "
                f"< {min_ratio:.2%} (break snippet: {s.prefix_break_snippet})"
            )


def assert_no_ungrounded_entities(harness: TapeReplayHarness, step_index: int) -> None:
    """Assert that all structured entity IDs emitted at `step_index` appeared in prior prompt context."""
    step = next((s for s in harness.steps if s.step_index == step_index), None)
    assert step is not None, f"Step {step_index} not found"
    corpus = json.dumps(step.raw_request.get("messages") or [], ensure_ascii=False)
    emitted = step.assembled_text + " " + json.dumps(
        [tc.model_dump() for tc in step.tool_calls], ensure_ascii=False
    )
    ungrounded = [eid for eid in ENTITY_ID_RE.findall(emitted) if eid not in corpus]
    if ungrounded:
        raise AssertionError(
            f"Step {step_index} fabricated ungrounded entity IDs not present in context: {ungrounded}"
        )
