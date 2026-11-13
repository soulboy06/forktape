"""Side-by-side trajectory comparison and resource savings calculator."""

from __future__ import annotations

import json

from forktape.models.db import TapeStore
from forktape.models.tape import BranchComparison, StepDiffEntry


def _preview(text: str, max_len: int = 140) -> str:
    cleaned = text.replace("\n", " ").strip()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1] + "…"


def compare_tapes(store: TapeStore, left_tape_id: str, right_tape_id: str) -> BranchComparison:
    """Compute step-level structural diff, KV cache delta, and replay savings between two tapes."""
    left = store.get_tape(left_tape_id)
    right = store.get_tape(right_tape_id)
    if left is None:
        raise KeyError(f"Left tape '{left_tape_id}' not found")
    if right is None:
        raise KeyError(f"Right tape '{right_tape_id}' not found")

    left_steps = {s.step_index: s for s in store.get_steps(left_tape_id)}
    right_steps = {s.step_index: s for s in store.get_steps(right_tape_id)}
    all_indices = sorted(set(left_steps.keys()) | set(right_steps.keys()))

    divergence_step: int | None = None
    replayed_saved_ms = 0.0
    replayed_saved_tokens = 0
    entries: list[StepDiffEntry] = []

    for idx in all_indices:
        ls = left_steps.get(idx)
        rs = right_steps.get(idx)

        if ls and rs:
            l_tools = [tc.name for tc in ls.tool_calls]
            r_tools = [tc.name for tc in rs.tool_calls]
            l_args = json.dumps([tc.arguments for tc in ls.tool_calls], ensure_ascii=False) if ls.tool_calls else ""
            r_args = json.dumps([tc.arguments for tc in rs.tool_calls], ensure_ascii=False) if rs.tool_calls else ""

            if rs.replayed and ls.canonical_hash == rs.canonical_hash:
                relation = "identical_replay"
                replayed_saved_ms += max(0.0, ls.duration_ms - rs.duration_ms)
                replayed_saved_tokens += ls.prompt_tokens + ls.completion_tokens
            elif right.fork_from_step == idx or rs.override_patch is not None:
                relation = "fork_point_modified"
                if divergence_step is None:
                    divergence_step = idx
            elif l_tools != r_tools or l_args != r_args or ls.assembled_text != rs.assembled_text:
                relation = "diverged_continuation"
                if divergence_step is None:
                    divergence_step = idx
            else:
                relation = "identical_replay"

            entries.append(
                StepDiffEntry(
                    step_index=idx,
                    relation=relation,
                    left_model=ls.model,
                    right_model=rs.model,
                    left_tools_called=l_tools,
                    right_tools_called=r_tools,
                    left_args_preview=_preview(l_args),
                    right_args_preview=_preview(r_args),
                    left_text_preview=_preview(ls.assembled_text),
                    right_text_preview=_preview(rs.assembled_text),
                    left_prompt_tokens=ls.prompt_tokens,
                    right_prompt_tokens=rs.prompt_tokens,
                    left_cache_ratio=ls.prefix_cache_ratio,
                    right_cache_ratio=rs.prefix_cache_ratio,
                    left_duration_ms=ls.duration_ms,
                    right_duration_ms=rs.duration_ms,
                )
            )
        elif ls and not rs:
            if divergence_step is None:
                divergence_step = idx
            entries.append(
                StepDiffEntry(
                    step_index=idx,
                    relation="only_left",
                    left_model=ls.model,
                    left_tools_called=[tc.name for tc in ls.tool_calls],
                    left_args_preview=_preview(
                        json.dumps([tc.arguments for tc in ls.tool_calls], ensure_ascii=False)
                    ),
                    left_text_preview=_preview(ls.assembled_text),
                    left_prompt_tokens=ls.prompt_tokens,
                    left_cache_ratio=ls.prefix_cache_ratio,
                    left_duration_ms=ls.duration_ms,
                )
            )
        elif rs and not ls:
            if divergence_step is None:
                divergence_step = idx
            entries.append(
                StepDiffEntry(
                    step_index=idx,
                    relation="only_right",
                    right_model=rs.model,
                    right_tools_called=[tc.name for tc in rs.tool_calls],
                    right_args_preview=_preview(
                        json.dumps([tc.arguments for tc in rs.tool_calls], ensure_ascii=False)
                    ),
                    right_text_preview=_preview(rs.assembled_text),
                    right_prompt_tokens=rs.prompt_tokens,
                    right_cache_ratio=rs.prefix_cache_ratio,
                    right_duration_ms=rs.duration_ms,
                )
            )

    tok_delta = right.total_tokens - left.total_tokens
    tok_pct = round((tok_delta / max(1, left.total_tokens)) * 100.0, 2)

    return BranchComparison(
        left_tape=left,
        right_tape=right,
        divergence_step=divergence_step,
        token_delta=tok_delta,
        token_delta_pct=tok_pct,
        cost_delta_usd=round(right.total_cost_usd - left.total_cost_usd, 6),
        latency_delta_ms=round(right.total_latency_ms - left.total_latency_ms, 2),
        replayed_steps_saved_ms=round(replayed_saved_ms, 2),
        replayed_tokens_saved=replayed_saved_tokens,
        cache_hit_rate_left=left.cache_hit_rate,
        cache_hit_rate_right=right.cache_hit_rate,
        step_diffs=entries,
    )
