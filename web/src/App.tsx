import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Code2,
  Copy,
  Cpu,
  Database,
  Download,
  FastForward,
  FileCode2,
  GitBranch,
  GitCompare,
  Layers,
  Play,
  RefreshCw,
  Search,
  ShieldAlert,
  Stethoscope,
  Terminal,
  Zap,
} from "lucide-react";
import type {
  BranchComparison,
  OverridePatch,
  TapeDetailResponse,
  TapeRecord,
  TapeStep,
} from "./types";

type WorkbenchTab = "biopsy" | "fork" | "diff" | "xray" | "pytest";

function statusBadge(status: TapeRecord["status"]) {
  const map: Record<TapeRecord["status"], string> = {
    success: "bg-emerald-500/15 text-emerald-400 border-emerald-500/30",
    failed: "bg-rose-500/15 text-rose-400 border-rose-500/30",
    loop_aborted: "bg-amber-500/15 text-amber-400 border-amber-500/30",
    running: "bg-sky-500/15 text-sky-400 border-sky-500/30",
    diverged: "bg-cyan-500/15 text-cyan-400 border-cyan-500/30",
  };
  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded text-[11px] font-mono font-semibold uppercase border ${map[status] || map.running}`}
    >
      {status}
    </span>
  );
}

export default function App() {
  const [tapes, setTapes] = useState<TapeRecord[]>([]);
  const [filterMode, setFilterMode] = useState<"all" | "roots" | "failed">("roots");
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedTapeId, setSelectedTapeId] = useState<string | null>(null);
  const [detail, setDetail] = useState<TapeDetailResponse | null>(null);
  const [selectedStepIdx, setSelectedStepIdx] = useState<number>(1);
  const [activeTab, setActiveTab] = useState<WorkbenchTab>("biopsy");
  const [loading, setLoading] = useState<boolean>(false);
  const [runningAction, setRunningAction] = useState<string | null>(null);
  const [errorBanner, setErrorBanner] = useState<string | null>(null);

  // Fork Studio controls
  const [forkLabel, setForkLabel] = useState("Studio Counterfactual Fork");
  const [enablePruneTools, setEnablePruneTools] = useState(true);
  const [pruneTopK, setPruneTopK] = useState(5);
  const [enableCompactContext, setEnableCompactContext] = useState(false);
  const [compactMaxChars, setCompactMaxChars] = useState(900);
  const [stabilizeCache, setStabilizeCache] = useState(true);
  const [dryRunWrites, setDryRunWrites] = useState(true);
  const [appendSystemRule, setAppendSystemRule] = useState("");
  const [swapModel, setSwapModel] = useState("");

  // Branch Diff state
  const [compareTargetId, setCompareTargetId] = useState<string>("");
  const [comparison, setComparison] = useState<BranchComparison | null>(null);

  // Pytest state
  const [pytestCode, setPytestCode] = useState<string>("");
  const [copiedPytest, setCopiedPytest] = useState<boolean>(false);

  const loadTapes = async (preserveSelection = true) => {
    try {
      const resp = await fetch("/api/tapes?limit=200");
      const data = await resp.json();
      const list: TapeRecord[] = data.tapes || [];
      setTapes(list);
      if (!preserveSelection || !selectedTapeId) {
        const firstRoot = list.find((t) => !t.parent_tape_id) || list[0];
        if (firstRoot) {
          setSelectedTapeId(firstRoot.tape_id);
        }
      }
    } catch (err) {
      setErrorBanner(`Failed to connect to ForkTape API: ${String(err)}`);
    }
  };

  const loadTapeDetail = async (tapeId: string) => {
    setLoading(true);
    setErrorBanner(null);
    try {
      const resp = await fetch(`/api/tapes/${tapeId}`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const d: TapeDetailResponse = await resp.json();
      setDetail(d);

      const driftStep =
        d.biopsy_report?.suspected_drift_step ||
        d.steps[d.steps.length - 1]?.step_index ||
        1;
      setSelectedStepIdx(driftStep);

      if (d.biopsy_report?.generated_pytest_code) {
        setPytestCode(d.biopsy_report.generated_pytest_code);
      } else {
        const pyResp = await fetch(`/api/tapes/${tapeId}/export-pytest`);
        if (pyResp.ok) {
          const pyData = await pyResp.json();
          setPytestCode(pyData.pytest_code || "");
        }
      }

      // Auto-load branch comparison if children exist or if viewing a child fork
      if (d.tape.parent_tape_id) {
        setCompareTargetId(d.tape.parent_tape_id);
        await fetchDiff(d.tape.parent_tape_id, d.tape.tape_id);
      } else if (d.children.length > 0) {
        const provenBranchId =
          d.biopsy_report?.ablation_runs.find((r) => r.resolved_failure)?.forked_tape_id ||
          d.children[0].tape_id;
        setCompareTargetId(provenBranchId);
        await fetchDiff(d.tape.tape_id, provenBranchId);
      } else {
        setComparison(null);
        setCompareTargetId("");
      }
    } catch (err) {
      setErrorBanner(`Failed loading tape ${tapeId}: ${String(err)}`);
    } finally {
      setLoading(false);
    }
  };

  const fetchDiff = async (leftId: string, rightId: string) => {
    try {
      const resp = await fetch(`/api/tapes/${leftId}/diff/${rightId}`);
      if (resp.ok) {
        setComparison(await resp.json());
      }
    } catch {
      // Ignore transient diff errors
    }
  };

  useEffect(() => {
    void loadTapes(false);
  }, []);

  useEffect(() => {
    if (selectedTapeId) {
      void loadTapeDetail(selectedTapeId);
    }
  }, [selectedTapeId]);

  const selectedStep: TapeStep | undefined = useMemo(() => {
    if (!detail) return undefined;
    return detail.steps.find((s) => s.step_index === selectedStepIdx) || detail.steps[0];
  }, [detail, selectedStepIdx]);

  const childrenMap = useMemo(() => {
    const map: Record<string, TapeRecord[]> = {};
    for (const t of tapes) {
      if (t.parent_tape_id) {
        (map[t.parent_tape_id] ||= []).push(t);
      }
    }
    return map;
  }, [tapes]);

  const visibleRootTapes = useMemo(() => {
    return tapes.filter((t) => {
      if (filterMode === "roots" && t.parent_tape_id) return false;
      if (filterMode === "failed" && t.status !== "failed" && t.status !== "loop_aborted")
        return false;
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        return (
          t.title.toLowerCase().includes(q) ||
          t.tape_id.toLowerCase().includes(q) ||
          t.tags.some((tag) => tag.toLowerCase().includes(q))
        );
      }
      return true;
    });
  }, [tapes, filterMode, searchQuery]);

  const handleRunDoctor = async () => {
    if (!detail) return;
    setRunningAction("doctor");
    try {
      const resp = await fetch(`/api/tapes/${detail.tape.tape_id}/doctor`, {
        method: "POST",
      });
      if (!resp.ok) throw new Error(`Doctor returned ${resp.status}`);
      await loadTapes(true);
      await loadTapeDetail(detail.tape.tape_id);
      setActiveTab("biopsy");
    } catch (err) {
      setErrorBanner(`Biopsy Doctor error: ${String(err)}`);
    } finally {
      setRunningAction(null);
    }
  };

  const handleExecuteFork = async () => {
    if (!detail || !selectedStep) return;
    setRunningAction("fork");
    try {
      const patch: OverridePatch = {
        label: forkLabel || `Fork @ Step ${selectedStep.step_index}`,
        prune_tools_top_k: enablePruneTools ? pruneTopK : null,
        compact_context_max_chars: enableCompactContext ? compactMaxChars : null,
        stabilize_prefix_cache: stabilizeCache,
        dry_run_write_tools: dryRunWrites,
        system_prompt_append: appendSystemRule.trim() || null,
        swap_model: swapModel.trim() || null,
        override_temperature: 0.0,
      };
      const resp = await fetch(`/api/tapes/${detail.tape.tape_id}/fork`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          step_index: selectedStep.step_index,
          patch,
          prefer_live_upstream: false,
        }),
      });
      if (!resp.ok) throw new Error(`Fork failed: HTTP ${resp.status}`);
      const resJson = await resp.json();
      setComparison(resJson.comparison);
      setCompareTargetId(resJson.forked_tape.tape_id);
      await loadTapes(true);
      await loadTapeDetail(detail.tape.tape_id);
      setActiveTab("diff");
    } catch (err) {
      setErrorBanner(`Counterfactual fork failed: ${String(err)}`);
    } finally {
      setRunningAction(null);
    }
  };

  const handleCopyPytest = async () => {
    if (!pytestCode) return;
    await navigator.clipboard.writeText(pytestCode);
    setCopiedPytest(true);
    setTimeout(() => setCopiedPytest(false), 1800);
  };

  const maxStepTokens = useMemo(() => {
    if (!detail || detail.steps.length === 0) return 1;
    return Math.max(1, ...detail.steps.map((s) => s.prompt_tokens));
  }, [detail]);

  return (
    <div className="h-screen w-screen flex flex-col bg-[#0b0e14] text-slate-100 overflow-hidden">
      {/* Top Global Navigation Bar */}
      <header className="h-12 shrink-0 border-b border-slate-800/90 bg-[#0e131d] px-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="h-7 w-7 rounded bg-cyan-500/15 border border-cyan-400/40 flex items-center justify-center text-cyan-400">
            <GitBranch className="w-4 h-4" />
          </div>
          <div className="flex items-baseline gap-2">
            <span className="font-bold tracking-tight text-slate-100">ForkTape Studio</span>
            <span className="text-[11px] font-mono text-cyan-400 bg-cyan-950/60 border border-cyan-800/50 px-1.5 py-0.5 rounded">
              v0.1.0 · VCR + LangGraph Biopsy
            </span>
          </div>
        </div>

        <div className="flex items-center gap-3 text-xs text-slate-400 font-mono">
          <span className="hidden md:inline-flex items-center gap-1.5 bg-slate-900 border border-slate-800 px-2.5 py-1 rounded">
            <Terminal className="w-3.5 h-3.5 text-emerald-400" />
            OPENAI_BASE_URL=http://127.0.0.1:8787/v1
          </span>
          <button
            onClick={() => void loadTapes(true)}
            className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded bg-slate-800/80 hover:bg-slate-700 text-slate-200 border border-slate-700 transition cursor-pointer"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            Sync Vault
          </button>
        </div>
      </header>

      {errorBanner && (
        <div className="bg-rose-950/80 border-b border-rose-700/60 px-4 py-2 text-xs text-rose-200 flex items-center justify-between">
          <span>{errorBanner}</span>
          <button
            onClick={() => setErrorBanner(null)}
            className="text-rose-300 underline cursor-pointer"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Main 3-Pane Engineering Workspace */}
      <div className="flex-1 flex min-h-0">
        {/* LEFT PANE: Trajectory Vault & Branch Tree */}
        <aside className="w-80 shrink-0 border-r border-slate-800/90 bg-[#0d111a] flex flex-col min-h-0">
          <div className="p-3 border-b border-slate-800/80 space-y-2.5">
            <div className="relative">
              <Search className="w-3.5 h-3.5 text-slate-500 absolute left-2.5 top-2.5" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Filter tapes, tags, IDs..."
                className="w-full bg-slate-900/90 border border-slate-800 rounded pl-8 pr-2.5 py-1.5 text-xs text-slate-200 placeholder:text-slate-500 focus:outline-none focus:border-cyan-500/60"
              />
            </div>
            <div className="grid grid-cols-3 gap-1 bg-slate-900/80 p-1 rounded border border-slate-800/80 text-[11px] font-medium">
              {(
                [
                  ["roots", "Root Runs"],
                  ["failed", "Failed Only"],
                  ["all", "All Branches"],
                ] as const
              ).map(([key, label]) => (
                <button
                  key={key}
                  onClick={() => setFilterMode(key)}
                  className={`py-1 rounded transition cursor-pointer ${
                    filterMode === key
                      ? "bg-slate-800 text-cyan-300 shadow-sm"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          <div className="flex-1 overflow-y-auto divide-y divide-slate-800/50">
            {visibleRootTapes.map((tape) => {
              const isSelected = tape.tape_id === selectedTapeId;
              const kids = filterMode === "roots" ? childrenMap[tape.tape_id] || [] : [];
              return (
                <div key={tape.tape_id}>
                  <button
                    onClick={() => setSelectedTapeId(tape.tape_id)}
                    className={`w-full text-left p-3 transition cursor-pointer ${
                      isSelected
                        ? "bg-cyan-950/30 border-l-2 border-l-cyan-400"
                        : "hover:bg-slate-900/60"
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2 mb-1">
                      <span className="font-mono text-[11px] text-slate-400 truncate">
                        {tape.tape_id}
                      </span>
                      {statusBadge(tape.status)}
                    </div>
                    <div className="text-xs font-semibold text-slate-100 line-clamp-2 leading-snug mb-1.5">
                      {tape.title}
                    </div>
                    <div className="flex items-center justify-between text-[11px] font-mono text-slate-400">
                      <span>{tape.step_count} steps</span>
                      <span>{(tape.total_tokens ?? tape.total_prompt_tokens).toLocaleString()} tok</span>
                      <span>${tape.total_cost_usd.toFixed(4)}</span>
                    </div>
                  </button>

                  {kids.length > 0 && (
                    <div className="bg-[#090c12] border-t border-slate-800/40 pl-4 pr-2 py-1.5 space-y-1">
                      {kids.slice(0, 4).map((child) => (
                        <button
                          key={child.tape_id}
                          onClick={() => setSelectedTapeId(child.tape_id)}
                          className={`w-full text-left px-2 py-1.5 rounded text-[11px] flex items-center justify-between gap-1.5 transition cursor-pointer ${
                            selectedTapeId === child.tape_id
                              ? "bg-cyan-950/50 text-cyan-300 border border-cyan-800/50"
                              : "text-slate-400 hover:bg-slate-900 hover:text-slate-200"
                          }`}
                        >
                          <span className="truncate flex items-center gap-1">
                            <GitBranch className="w-3 h-3 text-emerald-400 shrink-0" />
                            {child.fork_label || `Fork @ S${child.fork_from_step}`}
                          </span>
                          <span className="font-mono text-[10px] text-emerald-400 shrink-0">
                            S{child.fork_from_step}
                          </span>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </aside>

        {/* CENTER + RIGHT PANES */}
        {detail ? (
          <main className="flex-1 flex flex-col min-w-0 min-h-0">
            {/* Tape KPI & Action Strip */}
            <section className="shrink-0 border-b border-slate-800/90 bg-[#0f1420] px-5 py-3.5 space-y-3">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2.5 flex-wrap">
                    {statusBadge(detail.tape.status)}
                    <h1 className="text-sm md:text-base font-bold text-slate-100 truncate">
                      {detail.tape.title}
                    </h1>
                    {detail.tape.parent_tape_id && (
                      <button
                        onClick={() => setSelectedTapeId(detail.tape.parent_tape_id)}
                        className="text-[11px] font-mono text-cyan-400 hover:underline cursor-pointer"
                      >
                        ← Parent: {detail.tape.parent_tape_id}
                      </button>
                    )}
                  </div>
                  {detail.tape.failure_reason && (
                    <p className="mt-1.5 text-xs text-rose-300/95 bg-rose-950/35 border border-rose-800/40 rounded px-2.5 py-1 flex items-center gap-2">
                      <AlertTriangle className="w-3.5 h-3.5 text-rose-400 shrink-0" />
                      <span className="truncate">{detail.tape.failure_reason}</span>
                    </p>
                  )}
                </div>

                {/* Action Bar */}
                <div className="flex items-center gap-2 shrink-0">
                  <button
                    onClick={() => void handleRunDoctor()}
                    disabled={runningAction === "doctor"}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-semibold bg-amber-500/15 hover:bg-amber-500/25 text-amber-300 border border-amber-500/40 transition cursor-pointer disabled:opacity-50"
                  >
                    <Stethoscope className="w-3.5 h-3.5" />
                    {runningAction === "doctor" ? "Running Biopsy..." : "Run Biopsy Doctor"}
                  </button>
                  <button
                    onClick={() => setActiveTab("fork")}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-semibold bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-300 border border-cyan-500/40 transition cursor-pointer"
                  >
                    <Zap className="w-3.5 h-3.5" />
                    Fork @ Step {selectedStepIdx}
                  </button>
                  <button
                    onClick={() => setActiveTab("pytest")}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-semibold bg-emerald-500/15 hover:bg-emerald-500/25 text-emerald-300 border border-emerald-500/40 transition cursor-pointer"
                  >
                    <FileCode2 className="w-3.5 h-3.5" />
                    Export Pytest
                  </button>
                  <a
                    href={`/api/tapes/${detail.tape.tape_id}/bundle`}
                    className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition"
                    title="Download .ftk.json VCR bundle"
                  >
                    <Download className="w-3.5 h-3.5" />
                  </a>
                </div>
              </div>

              {/* 4 Telemetry Metrics */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2.5">
                <div className="bg-[#0b0e14] border border-slate-800/90 rounded px-3 py-2">
                  <div className="text-[10px] font-mono uppercase text-slate-400 flex items-center gap-1">
                    <Layers className="w-3 h-3 text-cyan-400" /> Trajectory Steps & Time
                  </div>
                  <div className="text-sm font-mono font-bold text-slate-100 mt-0.5">
                    {detail.tape.step_count} steps · {(detail.tape.total_latency_ms / 1000).toFixed(2)}s
                  </div>
                </div>
                <div className="bg-[#0b0e14] border border-slate-800/90 rounded px-3 py-2">
                  <div className="text-[10px] font-mono uppercase text-slate-400 flex items-center gap-1">
                    <Database className="w-3 h-3 text-sky-400" /> Total Context Tokens
                  </div>
                  <div className="text-sm font-mono font-bold text-slate-100 mt-0.5">
                    {(
                      detail.tape.total_tokens ??
                      detail.tape.total_prompt_tokens + detail.tape.total_completion_tokens
                    ).toLocaleString()}{" "}
                    tok
                  </div>
                </div>
                <div className="bg-[#0b0e14] border border-slate-800/90 rounded px-3 py-2">
                  <div className="text-[10px] font-mono uppercase text-slate-400 flex items-center gap-1">
                    <Cpu className="w-3 h-3 text-emerald-400" /> Prefix KV Cache Hit Rate
                  </div>
                  <div
                    className={`text-sm font-mono font-bold mt-0.5 ${
                      (detail.tape.cache_hit_rate ?? 0) >= 0.5
                        ? "text-emerald-400"
                        : "text-amber-400"
                    }`}
                  >
                    {((detail.tape.cache_hit_rate ?? 0) * 100).toFixed(1)}% cached
                  </div>
                </div>
                <div className="bg-[#0b0e14] border border-slate-800/90 rounded px-3 py-2">
                  <div className="text-[10px] font-mono uppercase text-slate-400 flex items-center gap-1">
                    <Activity className="w-3 h-3 text-amber-400" /> Estimated Run Cost
                  </div>
                  <div className="text-sm font-mono font-bold text-slate-100 mt-0.5">
                    ${detail.tape.total_cost_usd.toFixed(4)} USD
                  </div>
                </div>
              </div>
            </section>

            {/* Split Timeline (Left-Center) + Multi-Tab Workbench (Right) */}
            <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 min-h-0">
              {/* CENTER: Step Flamegraph & Context Token X-Ray */}
              <div className="lg:col-span-5 border-r border-slate-800/90 flex flex-col min-h-0 bg-[#0b0e14]">
                <div className="px-4 py-2.5 border-b border-slate-800/80 flex items-center justify-between text-xs">
                  <span className="font-semibold text-slate-200 flex items-center gap-1.5">
                    <FastForward className="w-3.5 h-3.5 text-cyan-400" />
                    Causal Step Flamegraph & Context X-Ray
                  </span>
                  <div className="flex items-center gap-2 text-[10px] font-mono text-slate-400">
                    <span className="inline-flex items-center gap-1">
                      <span className="w-2 h-2 rounded-sm bg-sky-500" /> Sys
                    </span>
                    <span className="inline-flex items-center gap-1">
                      <span className="w-2 h-2 rounded-sm bg-amber-500" /> Tools
                    </span>
                    <span className="inline-flex items-center gap-1">
                      <span className="w-2 h-2 rounded-sm bg-indigo-400" /> Hist
                    </span>
                    <span className="inline-flex items-center gap-1">
                      <span className="w-2 h-2 rounded-sm bg-purple-500" /> ToolOut
                    </span>
                  </div>
                </div>

                <div className="flex-1 overflow-y-auto p-3 space-y-2">
                  {detail.steps.map((step) => {
                    const isSelected = step.step_index === selectedStepIdx;
                    const isDrift =
                      detail.biopsy_report?.suspected_drift_step === step.step_index;
                    const hasAnomalies = step.anomaly_flags.length > 0;
                    const tb = step.token_breakdown;
                    const tbSum = Math.max(
                      1,
                      tb.system_tokens +
                        tb.tools_schema_tokens +
                        tb.history_tokens +
                        tb.tool_output_tokens
                    );
                    const widthScale = Math.max(18, (step.prompt_tokens / maxStepTokens) * 100);

                    return (
                      <div
                        key={step.step_id}
                        onClick={() => setSelectedStepIdx(step.step_index)}
                        className={`rounded-md border p-3 transition cursor-pointer ${
                          isSelected
                            ? "bg-slate-900/95 border-cyan-400/70 shadow-md shadow-cyan-950/30"
                            : isDrift
                              ? "bg-rose-950/20 border-rose-700/60 hover:border-rose-500"
                              : "bg-[#101521] border-slate-800/80 hover:border-slate-700"
                        }`}
                      >
                        <div className="flex items-center justify-between gap-2 mb-1.5">
                          <div className="flex items-center gap-2">
                            <span className="font-mono text-xs font-bold text-slate-200">
                              Step {step.step_index}
                            </span>
                            {step.replayed ? (
                              <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">
                                VCR 0-TOK ({step.duration_ms.toFixed(1)}ms)
                              </span>
                            ) : (
                              <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-slate-800 text-slate-300 border border-slate-700">
                                {step.model} · {step.duration_ms.toFixed(0)}ms
                              </span>
                            )}
                            {isDrift && (
                              <span className="px-1.5 py-0.5 rounded text-[10px] font-mono font-bold bg-rose-500/20 text-rose-300 border border-rose-500/40">
                                ★ DRIFT INFLECTION
                              </span>
                            )}
                          </div>

                          <span
                            className={`text-[11px] font-mono ${
                              step.prefix_cache_ratio >= 0.85
                                ? "text-emerald-400"
                                : "text-rose-400 font-semibold"
                            }`}
                          >
                            KV LCP {(step.prefix_cache_ratio * 100).toFixed(0)}%
                          </span>
                        </div>

                        {/* Tool Call or Text Summary */}
                        <div className="text-xs font-mono text-slate-300 truncate mb-2">
                          {step.tool_calls.length > 0 ? (
                            <span className="text-cyan-300">
                              → {step.tool_calls.map((tc) => tc.name).join(", ")}
                              <span className="text-slate-400 ml-1">
                                ({JSON.stringify(step.tool_calls[0].arguments)})
                              </span>
                            </span>
                          ) : (
                            <span className="text-slate-300">{step.assembled_text}</span>
                          )}
                        </div>

                        {/* Proportional Context Token X-Ray Stacked Bar */}
                        <div className="space-y-1">
                          <div
                            className="h-2 rounded overflow-hidden flex bg-slate-950 border border-slate-800/80"
                            style={{ width: `${widthScale}%` }}
                          >
                            <div
                              style={{ width: `${(tb.system_tokens / tbSum) * 100}%` }}
                              className="bg-sky-500"
                              title={`System: ${tb.system_tokens} tok`}
                            />
                            <div
                              style={{ width: `${(tb.tools_schema_tokens / tbSum) * 100}%` }}
                              className="bg-amber-500"
                              title={`Tool Schemas: ${tb.tools_schema_tokens} tok`}
                            />
                            <div
                              style={{ width: `${(tb.history_tokens / tbSum) * 100}%` }}
                              className="bg-indigo-400"
                              title={`History: ${tb.history_tokens} tok`}
                            />
                            <div
                              style={{ width: `${(tb.tool_output_tokens / tbSum) * 100}%` }}
                              className="bg-purple-500"
                              title={`Tool Outputs: ${tb.tool_output_tokens} tok`}
                            />
                          </div>
                          <div className="flex items-center justify-between text-[10px] font-mono text-slate-400">
                            <span>
                              {step.prompt_tokens.toLocaleString()} prompt tok (tools:{" "}
                              {tb.tools_schema_tokens.toLocaleString()}, out:{" "}
                              {tb.tool_output_tokens.toLocaleString()})
                            </span>
                            {hasAnomalies && (
                              <span className="text-rose-400">
                                {step.anomaly_flags.map((a) => a.code).join(", ")}
                              </span>
                            )}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* RIGHT: Engineering Workbench */}
              <div className="lg:col-span-7 flex flex-col min-h-0 bg-[#0d111a]">
                {/* Workbench Tabs */}
                <div className="px-4 border-b border-slate-800/90 flex items-center gap-1 bg-[#0f1420] overflow-x-auto">
                  {(
                    [
                      ["biopsy", "Biopsy Doctor", Stethoscope],
                      ["fork", `Time-Travel Fork (S${selectedStepIdx})`, Zap],
                      ["diff", "Side-by-Side Diff", GitCompare],
                      ["xray", `Step ${selectedStepIdx} Payload`, Code2],
                      ["pytest", "Pytest Regression", FileCode2],
                    ] as const
                  ).map(([tabKey, label, Icon]) => (
                    <button
                      key={tabKey}
                      onClick={() => setActiveTab(tabKey)}
                      className={`px-3 py-2.5 text-xs font-semibold border-b-2 inline-flex items-center gap-1.5 whitespace-nowrap transition cursor-pointer ${
                        activeTab === tabKey
                          ? "border-cyan-400 text-cyan-300 bg-cyan-950/20"
                          : "border-transparent text-slate-400 hover:text-slate-200"
                      }`}
                    >
                      <Icon className="w-3.5 h-3.5" />
                      {label}
                    </button>
                  ))}
                </div>

                {/* Tab Body */}
                <div className="flex-1 overflow-y-auto p-4 space-y-4">
                  {loading && (
                    <div className="text-xs font-mono text-slate-400 animate-pulse">
                      Loading trajectory telemetry...
                    </div>
                  )}

                  {/* TAB 1: COUNTERFACTUAL BIOPSY DOCTOR */}
                  {activeTab === "biopsy" && (
                    <div className="space-y-4">
                      {detail.biopsy_report ? (
                        <>
                          <div className="rounded-lg border border-amber-500/40 bg-amber-950/15 p-4 space-y-2">
                            <div className="flex items-center justify-between flex-wrap gap-2">
                              <div className="flex items-center gap-2">
                                <ShieldAlert className="w-4 h-4 text-amber-400" />
                                <span className="text-xs font-mono uppercase text-amber-300 font-bold">
                                  Verified Root Cause: {detail.biopsy_report.root_cause_category}
                                </span>
                                <span className="text-xs font-mono px-2 py-0.5 rounded bg-rose-500/20 text-rose-300 border border-rose-500/30">
                                  Drift @ Step {detail.biopsy_report.suspected_drift_step}
                                </span>
                              </div>
                              <span className="text-xs font-mono text-emerald-400">
                                Confidence {(detail.biopsy_report.confidence * 100).toFixed(0)}%
                              </span>
                            </div>
                            <p className="text-xs text-slate-200 leading-relaxed">
                              {detail.biopsy_report.executive_summary}
                            </p>
                          </div>

                          {/* Controlled Counterfactual Ablations Table */}
                          <div className="space-y-2">
                            <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 font-mono">
                              Controlled Counterfactual Ablation Experiments (LangGraph StateGraph)
                            </h3>
                            <div className="space-y-2">
                              {detail.biopsy_report.ablation_runs.map((ab) => (
                                <div
                                  key={ab.experiment_id}
                                  className={`rounded border p-3 space-y-2 ${
                                    ab.resolved_failure
                                      ? "bg-emerald-950/15 border-emerald-600/40"
                                      : "bg-slate-900/60 border-slate-800"
                                  }`}
                                >
                                  <div className="flex items-center justify-between gap-2">
                                    <div className="flex items-center gap-2">
                                      {ab.resolved_failure ? (
                                        <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                                      ) : (
                                        <Activity className="w-4 h-4 text-slate-500 shrink-0" />
                                      )}
                                      <span className="text-xs font-semibold text-slate-100">
                                        {ab.hypothesis_name} (Step {ab.step_index})
                                      </span>
                                    </div>
                                    <div className="flex items-center gap-2">
                                      {ab.resolved_failure && (
                                        <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40">
                                          PROVEN FIX
                                        </span>
                                      )}
                                      {ab.forked_tape_id && (
                                        <button
                                          onClick={async () => {
                                            setCompareTargetId(ab.forked_tape_id!);
                                            await fetchDiff(
                                              detail.tape.tape_id,
                                              ab.forked_tape_id!
                                            );
                                            setActiveTab("diff");
                                          }}
                                          className="text-[11px] font-mono text-cyan-400 hover:underline cursor-pointer"
                                        >
                                          Inspect Diff →
                                        </button>
                                      )}
                                    </div>
                                  </div>

                                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-[11px] font-mono">
                                    <div className="bg-slate-950/80 border border-slate-800/80 rounded p-2 text-rose-300 truncate">
                                      Before: {ab.before_summary}
                                    </div>
                                    <div className="bg-slate-950/80 border border-slate-800/80 rounded p-2 text-emerald-300 truncate">
                                      After: {ab.after_summary}
                                    </div>
                                  </div>

                                  <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] font-mono text-slate-400">
                                    <span>
                                      Prompt Tokens: {ab.prompt_tokens_before.toLocaleString()} →{" "}
                                      <strong className="text-slate-200">
                                        {ab.prompt_tokens_after.toLocaleString()}
                                      </strong>{" "}
                                      ({ab.token_reduction_pct >= 0 ? "-" : "+"}
                                      {Math.abs(ab.token_reduction_pct).toFixed(1)}%)
                                    </span>
                                    <span>
                                      Prefix KV LCP: {(ab.cache_ratio_before * 100).toFixed(0)}% →{" "}
                                      <strong className="text-emerald-400">
                                        {(ab.cache_ratio_after * 100).toFixed(0)}%
                                      </strong>
                                    </span>
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>

                          {/* Remediation Playbook */}
                          <div className="rounded border border-slate-800 bg-slate-900/50 p-3.5 space-y-1.5">
                            <h4 className="text-xs font-bold text-cyan-300 font-mono uppercase">
                              Engineering Remediation Playbook
                            </h4>
                            <pre className="text-xs text-slate-300 whitespace-pre-wrap font-sans leading-relaxed">
                              {detail.biopsy_report.verified_fix_recommendation}
                            </pre>
                          </div>
                        </>
                      ) : (
                        <div className="rounded border border-slate-800 bg-slate-900/40 p-6 text-center space-y-3">
                          <Stethoscope className="w-8 h-8 text-amber-400 mx-auto" />
                          <div className="text-sm font-semibold">
                            No Counterfactual Biopsy Report Yet
                          </div>
                          <p className="text-xs text-slate-400 max-w-md mx-auto">
                            Run the LangGraph Biopsy Doctor to scan deterministic anomalies, isolate
                            the causal drift step, and execute controlled counterfactual forks.
                          </p>
                          <button
                            onClick={() => void handleRunDoctor()}
                            className="inline-flex items-center gap-1.5 px-4 py-2 rounded text-xs font-semibold bg-amber-500/20 text-amber-300 border border-amber-500/40 cursor-pointer"
                          >
                            <Play className="w-3.5 h-3.5" /> Launch Counterfactual Biopsy
                          </button>
                        </div>
                      )}
                    </div>
                  )}

                  {/* TAB 2: INTERACTIVE TIME-TRAVEL FORK WORKBENCH */}
                  {activeTab === "fork" && selectedStep && (
                    <div className="space-y-4">
                      <div className="rounded border border-cyan-500/30 bg-cyan-950/15 p-3.5 text-xs space-y-1">
                        <div className="font-bold text-cyan-300 flex items-center gap-1.5">
                          <Zap className="w-4 h-4" />
                          Time-Travel Fork Target: Step {selectedStep.step_index} of{" "}
                          {detail.tape.step_count}
                        </div>
                        <p className="text-slate-300">
                          Steps <strong>1..{Math.max(0, selectedStep.step_index - 1)}</strong> will
                          fast-forward deterministically from SQLite in{" "}
                          <strong>&lt;3ms with 0 Token cost &amp; 0 MCP side effects</strong>. Step{" "}
                          <strong>{selectedStep.step_index}</strong> will re-execute with your
                          patches below.
                        </p>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        <div>
                          <label className="block text-[11px] font-mono text-slate-400 mb-1">
                            Branch Label
                          </label>
                          <input
                            type="text"
                            value={forkLabel}
                            onChange={(e) => setForkLabel(e.target.value)}
                            className="w-full bg-slate-900 border border-slate-700 rounded px-2.5 py-1.5 text-xs text-slate-100"
                          />
                        </div>
                        <div>
                          <label className="block text-[11px] font-mono text-slate-400 mb-1">
                            Swap Model (Optional)
                          </label>
                          <select
                            value={swapModel}
                            onChange={(e) => setSwapModel(e.target.value)}
                            className="w-full bg-slate-900 border border-slate-700 rounded px-2.5 py-1.5 text-xs text-slate-100"
                          >
                            <option value="">Keep Recorded ({selectedStep.model})</option>
                            <option value="gpt-4o">gpt-4o</option>
                            <option value="gpt-4.1-mini">gpt-4.1-mini</option>
                            <option value="claude-3-7-sonnet">claude-3-7-sonnet</option>
                            <option value="deepseek-chat">deepseek-chat</option>
                            <option value="qwen-max">qwen-max</option>
                          </select>
                        </div>
                      </div>

                      {/* Context Engineering Switches */}
                      <div className="space-y-2.5 rounded border border-slate-800 bg-slate-900/50 p-3.5">
                        <div className="text-xs font-mono uppercase text-slate-400 font-bold">
                          Context &amp; Protocol Interventions
                        </div>

                        <label className="flex items-center justify-between gap-3 text-xs cursor-pointer">
                          <span>
                            <strong>BM25 Semantic Tool Pruning</strong> — collapse overloaded MCP
                            tool schemas while preserving order
                          </span>
                          <div className="flex items-center gap-2 shrink-0">
                            <input
                              type="checkbox"
                              checked={enablePruneTools}
                              onChange={(e) => setEnablePruneTools(e.target.checked)}
                            />
                            <input
                              type="number"
                              min={1}
                              max={30}
                              value={pruneTopK}
                              onChange={(e) => setPruneTopK(Number(e.target.value))}
                              disabled={!enablePruneTools}
                              className="w-14 bg-slate-950 border border-slate-700 rounded px-1.5 py-0.5 text-xs font-mono text-center"
                            />
                          </div>
                        </label>

                        <label className="flex items-center justify-between gap-3 text-xs cursor-pointer">
                          <span>
                            <strong>Structure-Aware Context Compactor</strong> — prune oversized
                            middle JSON/HTML tool blobs
                          </span>
                          <div className="flex items-center gap-2 shrink-0">
                            <input
                              type="checkbox"
                              checked={enableCompactContext}
                              onChange={(e) => setEnableCompactContext(e.target.checked)}
                            />
                            <input
                              type="number"
                              min={300}
                              max={5000}
                              step={100}
                              value={compactMaxChars}
                              onChange={(e) => setCompactMaxChars(Number(e.target.value))}
                              disabled={!enableCompactContext}
                              className="w-20 bg-slate-950 border border-slate-700 rounded px-1.5 py-0.5 text-xs font-mono text-center"
                            />
                          </div>
                        </label>

                        <label className="flex items-center justify-between gap-3 text-xs cursor-pointer">
                          <span>
                            <strong>Stabilize Prefix KV Cache</strong> — strip dynamic header
                            timestamps &amp; sort tool schemas
                          </span>
                          <input
                            type="checkbox"
                            checked={stabilizeCache}
                            onChange={(e) => setStabilizeCache(e.target.checked)}
                          />
                        </label>

                        <label className="flex items-center justify-between gap-3 text-xs cursor-pointer">
                          <span>
                            <strong>MCP Dry-Run Write Sandbox</strong> — block destructive mutations
                            (`delete_*`, `send_*`)
                          </span>
                          <input
                            type="checkbox"
                            checked={dryRunWrites}
                            onChange={(e) => setDryRunWrites(e.target.checked)}
                          />
                        </label>
                      </div>

                      <div>
                        <div className="flex items-center justify-between mb-1">
                          <label className="text-[11px] font-mono text-slate-400">
                            Append System Guardrail Rule (Optional)
                          </label>
                          <button
                            type="button"
                            onClick={() =>
                              setAppendSystemRule(
                                "CRITICAL GROUNDING RULE: If the preceding search/query tool returned empty results ([] or null), NEVER fabricate IDs or invoke mutation tools. Ask for clarification."
                              )
                            }
                            className="text-[11px] text-cyan-400 hover:underline cursor-pointer"
                          >
                            + Insert Empty-Retrieval Guard Preset
                          </button>
                        </div>
                        <textarea
                          rows={3}
                          value={appendSystemRule}
                          onChange={(e) => setAppendSystemRule(e.target.value)}
                          placeholder="e.g. Never invoke admin wipe tools; cap autonomous refunds at $500..."
                          className="w-full bg-slate-900 border border-slate-700 rounded p-2.5 text-xs font-mono text-slate-200"
                        />
                      </div>

                      <button
                        onClick={() => void handleExecuteFork()}
                        disabled={runningAction === "fork"}
                        className="w-full py-2.5 rounded font-semibold text-xs bg-cyan-500 hover:bg-cyan-400 text-slate-950 transition flex items-center justify-center gap-2 cursor-pointer disabled:opacity-50"
                      >
                        <Zap className="w-4 h-4" />
                        {runningAction === "fork"
                          ? "Executing Counterfactual Fork..."
                          : `Execute Time-Travel Fork @ Step ${selectedStep.step_index}`}
                      </button>
                    </div>
                  )}

                  {/* TAB 3: SIDE-BY-SIDE BRANCH DIFF */}
                  {activeTab === "diff" && (
                    <div className="space-y-4">
                      {detail.children.length > 0 && (
                        <div className="flex items-center gap-2 text-xs">
                          <span className="font-mono text-slate-400">Compare against fork:</span>
                          <select
                            value={compareTargetId}
                            onChange={async (e) => {
                              const cid = e.target.value;
                              setCompareTargetId(cid);
                              await fetchDiff(detail.tape.tape_id, cid);
                            }}
                            className="bg-slate-900 border border-slate-700 rounded px-2.5 py-1 text-xs font-mono text-slate-100"
                          >
                            {detail.children.map((ch) => (
                              <option key={ch.tape_id} value={ch.tape_id}>
                                {ch.tape_id} — {ch.fork_label || `Fork @ S${ch.fork_from_step}`}
                              </option>
                            ))}
                          </select>
                        </div>
                      )}

                      {comparison ? (
                        <>
                          <div className="rounded border border-emerald-500/35 bg-emerald-950/15 p-3.5 grid grid-cols-3 gap-2 text-xs font-mono">
                            <div>
                              <div className="text-[10px] text-slate-400 uppercase">
                                Divergence Point
                              </div>
                              <div className="text-emerald-300 font-bold mt-0.5">
                                Step {comparison.divergence_step ?? "-"}
                              </div>
                            </div>
                            <div>
                              <div className="text-[10px] text-slate-400 uppercase">
                                VCR Tokens Saved
                              </div>
                              <div className="text-emerald-300 font-bold mt-0.5">
                                {comparison.replayed_tokens_saved.toLocaleString()} tok
                              </div>
                            </div>
                            <div>
                              <div className="text-[10px] text-slate-400 uppercase">
                                VCR Latency Saved
                              </div>
                              <div className="text-emerald-300 font-bold mt-0.5">
                                {(comparison.replayed_steps_saved_ms / 1000).toFixed(2)}s
                              </div>
                            </div>
                          </div>

                          <div className="space-y-2">
                            {comparison.step_diffs.map((row) => (
                              <div
                                key={row.step_index}
                                className={`rounded border p-3 text-xs space-y-2 ${
                                  row.relation === "fork_point_modified" ||
                                  row.relation === "diverged_continuation"
                                    ? "bg-cyan-950/15 border-cyan-600/50"
                                    : "bg-slate-900/50 border-slate-800/80"
                                }`}
                              >
                                <div className="flex items-center justify-between font-mono text-[11px]">
                                  <span className="font-bold text-slate-200">
                                    Step {row.step_index} ·{" "}
                                    <span className="text-cyan-400">{row.relation}</span>
                                  </span>
                                  <span className="text-slate-400">
                                    Tokens: {row.left_prompt_tokens.toLocaleString()} →{" "}
                                    <strong className="text-slate-200">
                                      {row.right_prompt_tokens.toLocaleString()}
                                    </strong>{" "}
                                    | KV LCP: {(row.left_cache_ratio * 100).toFixed(0)}% →{" "}
                                    <strong className="text-emerald-400">
                                      {(row.right_cache_ratio * 100).toFixed(0)}%
                                    </strong>
                                  </span>
                                </div>

                                <div className="grid grid-cols-1 md:grid-cols-2 gap-2 font-mono text-[11px]">
                                  <div className="bg-slate-950/90 border border-slate-800 rounded p-2 text-rose-300 break-all">
                                    <div className="text-[10px] text-slate-500 mb-0.5">
                                      BASELINE ({comparison.left_tape.tape_id})
                                    </div>
                                    {row.left_tools_called.length > 0
                                      ? `${row.left_tools_called.join(", ")}(${row.left_args_preview})`
                                      : row.left_text_preview || "—"}
                                  </div>
                                  <div className="bg-slate-950/90 border border-slate-800 rounded p-2 text-emerald-300 break-all">
                                    <div className="text-[10px] text-slate-500 mb-0.5">
                                      FORKED ({comparison.right_tape.tape_id})
                                    </div>
                                    {row.right_tools_called.length > 0
                                      ? `${row.right_tools_called.join(", ")}(${row.right_args_preview})`
                                      : row.right_text_preview || "—"}
                                  </div>
                                </div>
                              </div>
                            ))}
                          </div>
                        </>
                      ) : (
                        <div className="rounded border border-slate-800 bg-slate-900/40 p-6 text-center text-xs text-slate-400">
                          No forked branch selected yet. Use the <strong>Time-Travel Fork</strong>{" "}
                          or <strong>Biopsy Doctor</strong> tab to create a counterfactual branch!
                        </div>
                      )}
                    </div>
                  )}

                  {/* TAB 4: STEP PAYLOAD & PREFIX KV CACHE X-RAY */}
                  {activeTab === "xray" && selectedStep && (
                    <div className="space-y-3 text-xs">
                      {selectedStep.prefix_break_snippet && (
                        <div className="rounded border border-rose-600/40 bg-rose-950/20 p-3 space-y-1">
                          <div className="font-mono font-bold text-rose-300">
                            Prefix KV Cache Divergence Detected @ Char Offset{" "}
                            {selectedStep.prefix_break_offset}
                          </div>
                          <div className="font-mono text-[11px] text-rose-200 break-all">
                            {selectedStep.prefix_break_snippet}
                          </div>
                        </div>
                      )}

                      <div className="rounded border border-slate-800 bg-slate-950 p-3 space-y-1.5">
                        <div className="font-mono text-[11px] text-slate-400 uppercase font-bold">
                          Canonical Fingerprint SHA-256
                        </div>
                        <div className="font-mono text-xs text-cyan-300 break-all">
                          {selectedStep.canonical_hash}
                        </div>
                      </div>

                      <div className="rounded border border-slate-800 bg-slate-950 p-3 space-y-2">
                        <div className="font-mono text-[11px] text-slate-400 uppercase font-bold">
                          Raw LLM Request Payload (Secrets Auto-Redacted)
                        </div>
                        <pre className="text-[11px] font-mono text-slate-300 overflow-x-auto max-h-96">
                          {JSON.stringify(selectedStep.raw_request, null, 2)}
                        </pre>
                      </div>
                    </div>
                  )}

                  {/* TAB 5: GENERATED PYTEST REGRESSION SUITE */}
                  {activeTab === "pytest" && (
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <div className="text-xs text-slate-300">
                          Deterministic CI regression test compiled from{" "}
                          <code className="text-cyan-300 font-mono">{detail.tape.tape_id}</code>
                        </div>
                        <button
                          onClick={() => void handleCopyPytest()}
                          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 cursor-pointer"
                        >
                          <Copy className="w-3.5 h-3.5" />
                          {copiedPytest ? "Copied!" : "Copy Pytest File"}
                        </button>
                      </div>
                      <pre className="rounded border border-slate-800 bg-[#080b10] p-4 text-[11px] font-mono text-emerald-300 overflow-x-auto leading-relaxed">
                        {pytestCode || "# Run Biopsy Doctor or click Export Pytest to generate."}
                      </pre>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </main>
        ) : (
          <div className="flex-1 flex items-center justify-center text-sm text-slate-500 font-mono">
            Select a trajectory tape from the left vault to begin time-travel inspection.
          </div>
        )}
      </div>
    </div>
  );
}
