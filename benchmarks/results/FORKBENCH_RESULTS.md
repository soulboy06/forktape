# ForkBench-50 Quantitative Evaluation Results

Reproducible benchmark across **50 multi-step faulty agent trajectories** (40 single-fault + 10 compound adversarial multi-fault trajectories, 5–10 steps per run).

| Metric | Naive Tail-Log Baseline | ForkTape (Heuristics + LangGraph Counterfactual Biopsy) |
| :--- | :---: | :---: |
| **Drift Step Exact Localization Acc** | `0.0%` | **`96.0%`** |
| **Drift Step (±1 Step) Localization Acc** | — | **`96.0%`** |
| **Root-Cause Classification Acc** | `22.0%` | **`96.0%`** |
| **Empirical Counterfactual Fix Proven** | `0.0%` | **`100.0%`** |
| **Debugging Token Cost Reduction** | `0.0%` (Full Re-run) | **`-89.41%`** (`374,706` → `39,687` tok) |
| **Iteration Cycle Speedup** | `1.0x` | **`53.7x faster`** |

## Per-Archetype Accuracy
- **TOOL_SCHEMA_OVERLOAD**: `11/11 (100%)`
- **CONTEXT_MIDDLE_LOSS**: `8/10 (80%)`
- **SILENT_EMPTY_RETRIEVAL**: `10/10 (100%)`
- **PREFIX_CACHE_THRASHING**: `9/9 (100%)`
- **TOOL_CALL_LOOP**: `10/10 (100%)`
