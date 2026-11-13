export interface AnomalyFlag {
  code: string;
  severity: "critical" | "high" | "medium" | "low";
  step_index: number;
  title: string;
  detail: string;
  evidence: Record<string, unknown>;
}

export interface StepTokenBreakdown {
  system_tokens: number;
  tools_schema_tokens: number;
  history_tokens: number;
  tool_output_tokens: number;
}

export interface ToolCallInvocation {
  id: string;
  name: string;
  arguments: Record<string, unknown> | string;
}

export interface OverridePatch {
  label?: string | null;
  system_prompt_replace?: string | null;
  system_prompt_append?: string | null;
  swap_model?: string | null;
  prune_tools_top_k?: number | null;
  compact_context_max_chars?: number | null;
  stabilize_prefix_cache?: boolean;
  override_temperature?: number | null;
  dry_run_write_tools?: boolean;
}

export interface TapeStep {
  step_id: string;
  tape_id: string;
  step_index: number;
  provider: string;
  model: string;
  canonical_hash: string;
  token_breakdown: StepTokenBreakdown;
  prefix_cache_ratio: number;
  prefix_break_offset: number | null;
  prefix_break_snippet: string | null;
  raw_request: Record<string, unknown>;
  raw_response: Record<string, unknown>;
  assembled_text: string;
  reasoning_text: string;
  tool_calls: ToolCallInvocation[];
  override_patch: OverridePatch | null;
  replayed: boolean;
  ttft_ms: number;
  duration_ms: number;
  prompt_tokens: number;
  cached_prompt_tokens: number;
  completion_tokens: number;
  cost_usd: number;
  anomaly_flags: AnomalyFlag[];
  created_at: string;
}

export interface MCPSpan {
  span_id: string;
  tape_id: string;
  step_index: number;
  server_name: string;
  transport: string;
  method: string;
  tool_name: string;
  arguments: Record<string, unknown>;
  result: unknown;
  is_error: boolean;
  replayed: boolean;
  duration_ms: number;
}

export interface TapeRecord {
  tape_id: string;
  parent_tape_id: string | null;
  fork_from_step: number | null;
  fork_label: string | null;
  title: string;
  agent_command: string;
  status: "running" | "success" | "failed" | "loop_aborted" | "diverged";
  failure_reason: string | null;
  step_count: number;
  total_prompt_tokens: number;
  total_cached_tokens: number;
  total_completion_tokens: number;
  total_tokens?: number;
  cache_hit_rate?: number;
  total_cost_usd: number;
  total_latency_ms: number;
  tags: string[];
  created_at: string;
}

export interface AblationRunResult {
  experiment_id: string;
  hypothesis_name: string;
  operator: string;
  step_index: number;
  forked_tape_id: string | null;
  patch_applied: OverridePatch;
  before_summary: string;
  after_summary: string;
  resolved_failure: boolean;
  prompt_tokens_before: number;
  prompt_tokens_after: number;
  token_reduction_pct: number;
  cache_ratio_before: number;
  cache_ratio_after: number;
  latency_ms: number;
  cost_usd: number;
  evidence_notes: string[];
}

export interface BiopsyReport {
  report_id: string;
  tape_id: string;
  suspected_drift_step: number;
  root_cause_category: string;
  confidence: number;
  executive_summary: string;
  heuristic_findings: AnomalyFlag[];
  ablation_runs: AblationRunResult[];
  verified_fix_recommendation: string;
  recommended_patch: OverridePatch | null;
  generated_pytest_code: string;
  total_verification_cost_usd: number;
  total_verification_time_ms: number;
}

export interface TapeDetailResponse {
  tape: TapeRecord;
  steps: TapeStep[];
  mcp_spans: MCPSpan[];
  children: TapeRecord[];
  heuristic_findings: AnomalyFlag[];
  biopsy_report: BiopsyReport | null;
}

export interface StepDiffEntry {
  step_index: number;
  relation: string;
  left_model: string | null;
  right_model: string | null;
  left_tools_called: string[];
  right_tools_called: string[];
  left_args_preview: string;
  right_args_preview: string;
  left_text_preview: string;
  right_text_preview: string;
  left_prompt_tokens: number;
  right_prompt_tokens: number;
  left_cache_ratio: number;
  right_cache_ratio: number;
  left_duration_ms: number;
  right_duration_ms: number;
}

export interface BranchComparison {
  left_tape: TapeRecord;
  right_tape: TapeRecord;
  divergence_step: number | null;
  token_delta: number;
  token_delta_pct: number;
  cost_delta_usd: number;
  latency_delta_ms: number;
  replayed_steps_saved_ms: number;
  replayed_tokens_saved: number;
  cache_hit_rate_left: number;
  cache_hit_rate_right: number;
  step_diffs: StepDiffEntry[];
}
