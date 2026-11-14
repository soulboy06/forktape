import { useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertCircle,
  Check,
  CheckCircle2,
  ChevronRight,
  Code2,
  Copy,
  Download,
  FileCode2,
  GitBranch,
  GitCompare,
  Menu,
  Play,
  RefreshCw,
  Search,
  Sparkles,
  Stethoscope,
  Terminal,
  X,
  Zap,
} from "lucide-react";
import {
  type Lang,
  UI_TEXT,
  localizeExecutiveSummary,
  localizeFailureReason,
  localizeForkLabel,
  localizeHypothesis,
  localizePlaybook,
  localizeRelation,
  localizeRootCause,
  localizeStatus,
  localizeTapeTitle,
} from "./i18n";
import type {
  BranchComparison,
  OverridePatch,
  TapeDetailResponse,
  TapeRecord,
  TapeStep,
} from "./types";

type WorkbenchTab = "biopsy" | "fork" | "diff" | "xray" | "pytest";
type MobilePane = "timeline" | "workbench";

function StatusPill({ status, lang }: { status: TapeRecord["status"]; lang: Lang }) {
  const styles: Record<TapeRecord["status"], { pill: string; dot: string }> = {
    success: {
      pill: "bg-emerald-500/10 text-emerald-300 border-emerald-500/20",
      dot: "bg-emerald-400",
    },
    failed: {
      pill: "bg-rose-500/10 text-rose-300 border-rose-500/20",
      dot: "bg-rose-400",
    },
    loop_aborted: {
      pill: "bg-amber-500/10 text-amber-300 border-amber-500/20",
      dot: "bg-amber-400",
    },
    running: {
      pill: "bg-sky-500/10 text-sky-300 border-sky-500/20",
      dot: "bg-sky-400",
    },
    diverged: {
      pill: "bg-zinc-500/15 text-zinc-300 border-zinc-500/25",
      dot: "bg-zinc-400",
    },
  };
  const st = styles[status] || styles.running;
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[11px] font-medium tracking-wide border shrink-0 ${st.pill}`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${st.dot}`} />
      {localizeStatus(status, lang)}
    </span>
  );
}

function ToggleSwitch({
  checked,
  onChange,
}: {
  checked: boolean;
  onChange: (next: boolean) => void;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className={`relative inline-flex h-5 w-9 shrink-0 items-center rounded-full transition-colors duration-200 cursor-pointer ${
        checked ? "bg-zinc-100" : "bg-zinc-800 border border-white/10"
      }`}
    >
      <span
        className={`inline-block h-3.5 w-3.5 transform rounded-full transition-transform duration-200 ${
          checked ? "translate-x-4.5 bg-zinc-950" : "translate-x-0.5 bg-zinc-400"
        }`}
      />
    </button>
  );
}

export default function App() {
  const [lang, setLang] = useState<Lang>(() => {
    try {
      const saved = localStorage.getItem("forktape_lang");
      if (saved === "zh" || saved === "en") return saved;
    } catch {
      // ignore storage errors
    }
    return "zh";
  });

  const t = UI_TEXT[lang];

  const [mobileVaultOpen, setMobileVaultOpen] = useState(false);
  const [mobilePane, setMobilePane] = useState<MobilePane>("workbench");

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
  const [preferLiveUpstream, setPreferLiveUpstream] = useState(false);
  const [appendSystemRule, setAppendSystemRule] = useState("");
  const [swapModel, setSwapModel] = useState("");

  // Branch Diff state
  const [compareTargetId, setCompareTargetId] = useState<string>("");
  const [comparison, setComparison] = useState<BranchComparison | null>(null);

  // Pytest state
  const [pytestCode, setPytestCode] = useState<string>("");
  const [copiedPytest, setCopiedPytest] = useState<boolean>(false);

  const switchLang = (next: Lang) => {
    setLang(next);
    try {
      localStorage.setItem("forktape_lang", next);
    } catch {
      // ignore
    }
  };

  const openWorkbenchTab = (tab: WorkbenchTab) => {
    setActiveTab(tab);
    setMobilePane("workbench");
  };

  const loadTapes = async (preserveSelection = true) => {
    try {
      const resp = await fetch("/api/tapes?limit=200");
      const data = await resp.json();
      const list: TapeRecord[] = data.tapes || [];
      setTapes(list);
      if (!preserveSelection || !selectedTapeId) {
        const firstRoot = list.find((item) => !item.parent_tape_id) || list[0];
        if (firstRoot) {
          setSelectedTapeId(firstRoot.tape_id);
        }
      }
    } catch (err) {
      setErrorBanner(`API Error: ${String(err)}`);
    }
  };

  const fetchDiff = async (leftId: string, rightId: string) => {
    try {
      const resp = await fetch(`/api/tapes/${leftId}/diff/${rightId}`);
      if (resp.ok) {
        setComparison(await resp.json());
      }
    } catch {
      // ignore transient diff errors
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

      const recPatch = d.biopsy_report?.recommended_patch;
      if (recPatch) {
        setEnablePruneTools(recPatch.prune_tools_top_k != null);
        if (recPatch.prune_tools_top_k != null) setPruneTopK(recPatch.prune_tools_top_k);
        setEnableCompactContext(recPatch.compact_context_max_chars != null);
        if (recPatch.compact_context_max_chars != null) {
          setCompactMaxChars(recPatch.compact_context_max_chars);
        }
        setStabilizeCache(Boolean(recPatch.stabilize_prefix_cache));
        setAppendSystemRule(recPatch.system_prompt_append || "");
      }

      if (d.biopsy_report?.generated_pytest_code) {
        setPytestCode(d.biopsy_report.generated_pytest_code);
      } else {
        const pyResp = await fetch(`/api/tapes/${tapeId}/export-pytest`);
        if (pyResp.ok) {
          const pyData = await pyResp.json();
          setPytestCode(pyData.pytest_code || "");
        }
      }

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
      setErrorBanner(`${tapeId}: ${String(err)}`);
    } finally {
      setLoading(false);
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
    for (const item of tapes) {
      if (item.parent_tape_id) {
        (map[item.parent_tape_id] ||= []).push(item);
      }
    }
    return map;
  }, [tapes]);

  const visibleRootTapes = useMemo(() => {
    return tapes.filter((item) => {
      if (filterMode === "roots" && item.parent_tape_id) return false;
      if (filterMode === "failed" && item.status !== "failed" && item.status !== "loop_aborted")
        return false;
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        const zhTitle = localizeTapeTitle(item, "zh").toLowerCase();
        return (
          item.title.toLowerCase().includes(q) ||
          zhTitle.includes(q) ||
          item.tape_id.toLowerCase().includes(q) ||
          item.tags.some((tag) => tag.toLowerCase().includes(q))
        );
      }
      return true;
    });
  }, [tapes, filterMode, searchQuery]);

  const handleSelectTape = (id: string) => {
    setSelectedTapeId(id);
    setMobileVaultOpen(false);
  };

  const handleRunDoctor = async () => {
    if (!detail) return;
    setRunningAction("doctor");
    try {
      const resp = await fetch(`/api/tapes/${detail.tape.tape_id}/doctor`, {
        method: "POST",
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      await loadTapes(true);
      await loadTapeDetail(detail.tape.tape_id);
      openWorkbenchTab("biopsy");
    } catch (err) {
      setErrorBanner(String(err));
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
          prefer_live_upstream: preferLiveUpstream,
        }),
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const resJson = await resp.json();
      setComparison(resJson.comparison);
      setCompareTargetId(resJson.forked_tape.tape_id);
      await loadTapes(true);
      await loadTapeDetail(detail.tape.tape_id);
      openWorkbenchTab("diff");
    } catch (err) {
      setErrorBanner(String(err));
    } finally {
      setRunningAction(null);
    }
  };

  const handleCopyPytest = async () => {
    if (!pytestCode) return;
    try {
      await navigator.clipboard.writeText(pytestCode);
    } catch {
      const ta = document.createElement("textarea");
      ta.value = pytestCode;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
    }
    setCopiedPytest(true);
    setTimeout(() => setCopiedPytest(false), 1800);
  };

  const maxStepTokens = useMemo(() => {
    if (!detail || detail.steps.length === 0) return 1;
    return Math.max(1, ...detail.steps.map((s) => s.prompt_tokens));
  }, [detail]);

  const rootCount = useMemo(() => tapes.filter((item) => !item.parent_tape_id).length, [tapes]);

  const renderVaultContent = () => (
    <>
      <div className="p-3.5 border-b border-white/[0.06] space-y-2.5">
        <div className="relative">
          <Search className="w-3.5 h-3.5 text-zinc-500 absolute left-3 top-1/2 -translate-y-1/2" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder={t.searchPlaceholder}
            className="w-full bg-zinc-900/80 border border-white/[0.07] rounded-lg pl-8.5 pr-3 py-2 text-xs text-zinc-100 placeholder:text-zinc-500 focus:outline-none focus:border-zinc-500 transition"
          />
        </div>
        <div className="grid grid-cols-3 gap-1 bg-zinc-900/90 p-1 rounded-lg border border-white/[0.06] text-[11px] font-medium">
          {(
            [
              ["roots", t.filterRoots],
              ["failed", t.filterFailed],
              ["all", t.filterAll],
            ] as const
          ).map(([key, label]) => (
            <button
              key={key}
              onClick={() => setFilterMode(key)}
              className={`py-1 rounded-md transition cursor-pointer ${
                filterMode === key
                  ? "bg-zinc-800 text-zinc-100 shadow-xs"
                  : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto divide-y divide-white/[0.04]">
        {visibleRootTapes.map((tape) => {
          const isSelected = tape.tape_id === selectedTapeId;
          const kids = filterMode === "roots" ? childrenMap[tape.tape_id] || [] : [];
          return (
            <div key={tape.tape_id}>
              <button
                onClick={() => handleSelectTape(tape.tape_id)}
                className={`w-full text-left px-3.5 py-3 transition cursor-pointer relative ${
                  isSelected ? "bg-white/[0.05]" : "hover:bg-white/[0.025]"
                }`}
              >
                {isSelected && (
                  <span className="absolute left-0 top-2.5 bottom-2.5 w-0.5 rounded-r bg-zinc-100" />
                )}
                <div className="flex items-center justify-between gap-2 mb-1.5">
                  <span className="font-mono text-[11px] text-zinc-500 truncate">
                    {tape.tape_id}
                  </span>
                  <StatusPill status={tape.status} lang={lang} />
                </div>
                <div className="text-[13px] font-medium text-zinc-100 line-clamp-2 leading-snug mb-2">
                  {localizeTapeTitle(tape, lang)}
                </div>
                <div className="flex items-center justify-between text-[11px] font-mono tabular-nums text-zinc-500">
                  <span>
                    {tape.step_count} {t.stepsUnit}
                  </span>
                  <span>
                    {(tape.total_tokens ?? tape.total_prompt_tokens).toLocaleString()} tok
                  </span>
                  <span>${tape.total_cost_usd.toFixed(4)}</span>
                </div>
              </button>

              {kids.length > 0 && (
                <div className="bg-black/30 border-t border-white/[0.04] pl-4 pr-2.5 py-1.5 space-y-0.5">
                  {[...kids].reverse().slice(0, 8).map((child) => {
                    const childActive = selectedTapeId === child.tape_id;
                    const isSuccess = child.status === "success";
                    return (
                      <button
                        key={child.tape_id}
                        onClick={() => handleSelectTape(child.tape_id)}
                        className={`w-full text-left px-2.5 py-1.5 rounded-md text-[11px] flex items-center justify-between gap-2 transition cursor-pointer ${
                          childActive
                            ? "bg-white/[0.08] text-zinc-100"
                            : "text-zinc-400 hover:bg-white/[0.04] hover:text-zinc-200"
                        }`}
                      >
                        <span className="truncate flex items-center gap-1.5">
                          <GitBranch
                            className={`w-3 h-3 shrink-0 ${
                              isSuccess ? "text-emerald-400" : "text-zinc-500"
                            }`}
                          />
                          <span className="truncate">
                            {localizeForkLabel(child.fork_label, child.fork_from_step, lang)}
                          </span>
                        </span>
                        <span
                          className={`font-mono text-[10px] shrink-0 px-1.5 py-0.5 rounded ${
                            isSuccess
                              ? "text-emerald-400 bg-emerald-500/10"
                              : "text-zinc-500 bg-white/[0.04]"
                          }`}
                        >
                          S{child.fork_from_step}
                        </span>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </>
  );

  const failureReasonText = detail ? localizeFailureReason(detail.tape, lang) : null;

  return (
    <div className="min-h-[100dvh] lg:h-screen w-full flex flex-col bg-[#09090b] text-zinc-100 lg:overflow-hidden">
      {/* Top Minimalist Header */}
      <header className="h-13 shrink-0 border-b border-white/[0.07] bg-[#09090b]/90 backdrop-blur-md sticky top-0 z-30 px-3.5 lg:px-5 flex items-center justify-between gap-2">
        <div className="flex items-center gap-2.5 min-w-0">
          {/* Mobile Vault Drawer Button */}
          <button
            onClick={() => setMobileVaultOpen(true)}
            className="lg:hidden inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-white/[0.05] hover:bg-white/[0.09] border border-white/[0.08] text-xs font-medium text-zinc-200 transition cursor-pointer shrink-0"
          >
            <Menu className="w-3.5 h-3.5 text-zinc-400" />
            <span>{t.vaultBtn}</span>
            <span className="px-1.5 py-0.2 rounded-full bg-white/[0.08] text-[10px] font-mono text-zinc-300">
              {rootCount}
            </span>
          </button>

          <div className="flex items-center gap-2.5 min-w-0">
            <div className="h-7 w-7 rounded-lg bg-zinc-100 text-zinc-950 flex items-center justify-center shadow-xs shrink-0">
              <GitBranch className="w-4 h-4 stroke-[2.2]" />
            </div>
            <div className="flex items-center gap-2 min-w-0">
              <span className="font-semibold text-sm tracking-tight text-zinc-100 truncate">
                ForkTape
              </span>
              <span className="hidden sm:inline-block text-xs text-zinc-500 truncate">
                · {t.subtitle}
              </span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <span className="hidden xl:inline-flex items-center gap-1.5 bg-zinc-900/90 border border-white/[0.07] px-2.5 py-1 rounded-md text-[11px] font-mono text-zinc-400">
            <Terminal className="w-3 h-3 text-emerald-400" />
            OPENAI_BASE_URL=
            {typeof window !== "undefined" ? window.location.origin : "http://127.0.0.1:18787"}/v1
          </span>

          {/* Bilingual Pill Switcher */}
          <div className="inline-flex items-center bg-zinc-900 border border-white/[0.08] rounded-lg p-0.5 text-[11px] font-medium">
            <button
              onClick={() => switchLang("zh")}
              className={`px-2 py-1 rounded-md transition cursor-pointer ${
                lang === "zh" ? "bg-zinc-100 text-zinc-950 font-semibold" : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              中文
            </button>
            <button
              onClick={() => switchLang("en")}
              className={`px-2 py-1 rounded-md transition cursor-pointer ${
                lang === "en" ? "bg-zinc-100 text-zinc-950 font-semibold" : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              EN
            </button>
          </div>

          <button
            onClick={() => void loadTapes(true)}
            title={t.sync}
            className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg bg-white/[0.04] hover:bg-white/[0.08] text-xs font-medium text-zinc-300 border border-white/[0.08] transition cursor-pointer"
          >
            <RefreshCw className="w-3.5 h-3.5 text-zinc-400" />
            <span className="hidden sm:inline">{t.sync}</span>
          </button>
        </div>
      </header>

      {/* Mobile Slide-Over Vault Drawer */}
      {mobileVaultOpen && (
        <div className="fixed inset-0 z-50 lg:hidden flex">
          <div
            className="fixed inset-0 bg-black/70 backdrop-blur-xs"
            onClick={() => setMobileVaultOpen(false)}
          />
          <aside className="relative z-10 w-[86vw] max-w-[340px] h-full bg-[#0c0c0f] border-r border-white/[0.08] flex flex-col shadow-2xl">
            <div className="h-12 px-3.5 border-b border-white/[0.07] flex items-center justify-between">
              <span className="text-xs font-semibold text-zinc-200 flex items-center gap-2">
                <GitBranch className="w-3.5 h-3.5 text-zinc-400" />
                {t.vaultBtn}
              </span>
              <button
                onClick={() => setMobileVaultOpen(false)}
                className="p-1.5 rounded-lg text-zinc-400 hover:text-zinc-100 hover:bg-white/[0.06] cursor-pointer"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
            {renderVaultContent()}
          </aside>
        </div>
      )}

      {errorBanner && (
        <div className="bg-rose-500/10 border-b border-rose-500/25 px-4 py-2 text-xs text-rose-200 flex items-center justify-between gap-3">
          <span className="truncate">{errorBanner}</span>
          <button
            onClick={() => setErrorBanner(null)}
            className="text-rose-300 hover:text-white shrink-0 underline cursor-pointer"
          >
            {t.dismiss}
          </button>
        </div>
      )}

      {/* Main Responsive Workspace */}
      <div className="flex-1 flex min-h-0">
        {/* Desktop Sidebar */}
        <aside className="hidden lg:flex w-76 xl:w-82 shrink-0 border-r border-white/[0.07] bg-[#0c0c0f] flex-col min-h-0">
          {renderVaultContent()}
        </aside>

        {/* Center + Right Workspace */}
        {detail ? (
          <main className="flex-1 flex flex-col min-w-0 lg:min-h-0">
            {/* Clean Hero Summary & Primary Actions */}
            <section className="shrink-0 border-b border-white/[0.07] bg-[#0c0c0f]/60 px-4 lg:px-6 py-4 space-y-3.5">
              <div className="flex flex-col md:flex-row md:items-start justify-between gap-3">
                <div className="min-w-0 flex-1 space-y-1.5">
                  <div className="flex items-center gap-2 flex-wrap">
                    <StatusPill status={detail.tape.status} lang={lang} />
                    {detail.tape.parent_tape_id && (
                      <button
                        onClick={() => handleSelectTape(detail.tape.parent_tape_id!)}
                        className="inline-flex items-center gap-1 text-[11px] font-mono text-zinc-400 hover:text-zinc-100 bg-white/[0.04] border border-white/[0.07] px-2 py-0.5 rounded-full transition cursor-pointer"
                      >
                        ← {t.parentPrefix}: {detail.tape.parent_tape_id}
                      </button>
                    )}
                    <span className="text-[11px] font-mono text-zinc-500 hidden sm:inline">
                      {detail.tape.tape_id}
                    </span>
                  </div>

                  <h1 className="text-[15px] sm:text-base font-semibold tracking-tight text-zinc-100 leading-snug">
                    {localizeTapeTitle(detail.tape, lang)}
                  </h1>

                  {failureReasonText && (
                    <div className="text-xs text-rose-200/90 bg-rose-500/[0.07] border border-rose-500/20 rounded-lg px-3 py-2 flex items-start gap-2 leading-relaxed">
                      <AlertCircle className="w-3.5 h-3.5 text-rose-400 shrink-0 mt-0.5" />
                      <span>{failureReasonText}</span>
                    </div>
                  )}
                </div>

                {/* Crisp Action Buttons (Mobile Responsive Wrap) */}
                <div className="flex items-center gap-2 flex-wrap shrink-0">
                  <button
                    onClick={() => void handleRunDoctor()}
                    disabled={runningAction === "doctor"}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-white/[0.05] hover:bg-white/[0.1] text-zinc-100 border border-white/[0.1] transition cursor-pointer disabled:opacity-50"
                  >
                    <Sparkles className="w-3.5 h-3.5 text-amber-300" />
                    {runningAction === "doctor" ? t.runningDoctor : t.runDoctor}
                  </button>

                  <button
                    onClick={() => openWorkbenchTab("fork")}
                    className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-zinc-100 hover:bg-white text-zinc-950 shadow-xs transition cursor-pointer"
                  >
                    <Zap className="w-3.5 h-3.5 fill-zinc-950" />
                    {t.forkStepBtn(selectedStepIdx)}
                  </button>

                  <button
                    onClick={() => openWorkbenchTab("pytest")}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-white/[0.04] hover:bg-white/[0.08] text-zinc-300 border border-white/[0.08] transition cursor-pointer"
                  >
                    <FileCode2 className="w-3.5 h-3.5 text-zinc-400" />
                    {t.exportPytest}
                  </button>

                  <a
                    href={`/api/tapes/${detail.tape.tape_id}/bundle`}
                    className="inline-flex items-center justify-center p-1.5 rounded-lg bg-white/[0.04] hover:bg-white/[0.08] text-zinc-400 hover:text-zinc-100 border border-white/[0.08] transition"
                    title={t.downloadBundle}
                  >
                    <Download className="w-3.5 h-3.5" />
                  </a>
                </div>
              </div>

              {/* Minimalist Hairline Telemetry Strip */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 sm:gap-3 pt-1">
                <div className="rounded-xl bg-zinc-900/50 border border-white/[0.06] px-3 py-2">
                  <div className="text-[11px] text-zinc-500">{t.kpiSteps}</div>
                  <div className="text-sm font-mono font-semibold tabular-nums text-zinc-100 mt-0.5">
                    {detail.tape.step_count} {t.stepsUnit} · {(detail.tape.total_latency_ms / 1000).toFixed(2)}s
                  </div>
                </div>

                <div className="rounded-xl bg-zinc-900/50 border border-white/[0.06] px-3 py-2">
                  <div className="text-[11px] text-zinc-500">{t.kpiTokens}</div>
                  <div className="text-sm font-mono font-semibold tabular-nums text-zinc-100 mt-0.5">
                    {(
                      detail.tape.total_tokens ??
                      detail.tape.total_prompt_tokens + detail.tape.total_completion_tokens
                    ).toLocaleString()}
                  </div>
                </div>

                <div className="rounded-xl bg-zinc-900/50 border border-white/[0.06] px-3 py-2">
                  <div className="text-[11px] text-zinc-500">{t.kpiCache}</div>
                  <div
                    className={`text-sm font-mono font-semibold tabular-nums mt-0.5 ${
                      (detail.tape.cache_hit_rate ?? 0) >= 0.5
                        ? "text-emerald-400"
                        : "text-amber-300"
                    }`}
                  >
                    {((detail.tape.cache_hit_rate ?? 0) * 100).toFixed(1)}% {t.cachedSuffix}
                  </div>
                </div>

                <div className="rounded-xl bg-zinc-900/50 border border-white/[0.06] px-3 py-2">
                  <div className="text-[11px] text-zinc-500">{t.kpiCost}</div>
                  <div className="text-sm font-mono font-semibold tabular-nums text-zinc-100 mt-0.5">
                    ${detail.tape.total_cost_usd.toFixed(4)}
                  </div>
                </div>
              </div>
            </section>

            {/* Mobile Sticky Pane Switcher (< 1024px) */}
            <div className="lg:hidden sticky top-13 z-20 bg-[#09090b]/95 backdrop-blur-md border-b border-white/[0.07] px-4 py-2">
              <div className="grid grid-cols-2 gap-1 bg-zinc-900 p-1 rounded-xl border border-white/[0.06] text-xs font-medium">
                <button
                  onClick={() => setMobilePane("timeline")}
                  className={`py-1.5 rounded-lg transition cursor-pointer ${
                    mobilePane === "timeline"
                      ? "bg-zinc-800 text-zinc-100 shadow-xs"
                      : "text-zinc-400"
                  }`}
                >
                  {t.mobileSegTimeline(detail.steps.length)}
                </button>
                <button
                  onClick={() => setMobilePane("workbench")}
                  className={`py-1.5 rounded-lg transition cursor-pointer ${
                    mobilePane === "workbench"
                      ? "bg-zinc-800 text-zinc-100 shadow-xs"
                      : "text-zinc-400"
                  }`}
                >
                  {t.mobileSegWorkbench}
                </button>
              </div>
            </div>

            {/* Split Timeline + Multi-Tab Engineering Workbench */}
            <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 lg:min-h-0">
              {/* LEFT-CENTER: Step Timeline & Token X-Ray */}
              <div
                className={`${
                  mobilePane === "timeline" ? "flex" : "hidden"
                } lg:flex lg:col-span-5 lg:border-r border-white/[0.07] flex-col lg:min-h-0 bg-[#09090b]`}
              >
                <div className="px-4 py-2.5 border-b border-white/[0.06] flex items-center justify-between gap-2 text-xs">
                  <span className="font-medium text-zinc-300">{t.timelineTitle}</span>
                  <div className="flex items-center gap-2.5 text-[10px] text-zinc-500">
                    <span className="inline-flex items-center gap-1">
                      <span className="w-2 h-2 rounded-full bg-indigo-400" /> {t.legendSys}
                    </span>
                    <span className="inline-flex items-center gap-1">
                      <span className="w-2 h-2 rounded-full bg-amber-400" /> {t.legendTools}
                    </span>
                    <span className="inline-flex items-center gap-1">
                      <span className="w-2 h-2 rounded-full bg-sky-400" /> {t.legendHist}
                    </span>
                    <span className="inline-flex items-center gap-1">
                      <span className="w-2 h-2 rounded-full bg-violet-400" /> {t.legendOut}
                    </span>
                  </div>
                </div>

                <div className="flex-1 lg:overflow-y-auto p-3.5 space-y-2.5">
                  {detail.steps.map((step) => {
                    const isSelected = step.step_index === selectedStepIdx;
                    const isDrift =
                      detail.biopsy_report?.suspected_drift_step === step.step_index;
                    const tb = step.token_breakdown;
                    const tbSum = Math.max(
                      1,
                      tb.system_tokens +
                        tb.tools_schema_tokens +
                        tb.history_tokens +
                        tb.tool_output_tokens
                    );
                    const widthScale = Math.max(22, (step.prompt_tokens / maxStepTokens) * 100);

                    return (
                      <div
                        key={step.step_id}
                        onClick={() => setSelectedStepIdx(step.step_index)}
                        className={`group rounded-xl border p-3.5 transition cursor-pointer ${
                          isSelected
                            ? "bg-zinc-900/90 border-zinc-400/60 shadow-sm"
                            : isDrift
                              ? "bg-rose-500/[0.04] border-rose-500/30 hover:border-rose-400/50"
                              : "bg-zinc-900/35 border-white/[0.06] hover:border-white/[0.14]"
                        }`}
                      >
                        <div className="flex items-center justify-between gap-2 mb-2">
                          <div className="flex items-center gap-2 flex-wrap">
                            <span className="font-mono text-xs font-semibold text-zinc-100">
                              {t.stepPrefix(step.step_index)}
                            </span>
                            {step.replayed ? (
                              <span className="px-2 py-0.5 rounded-full text-[10px] font-mono bg-emerald-500/10 text-emerald-300 border border-emerald-500/20">
                                {t.vcrBadge(step.duration_ms.toFixed(1))}
                              </span>
                            ) : (
                              <span className="px-2 py-0.5 rounded-full text-[10px] font-mono bg-white/[0.05] text-zinc-400 border border-white/[0.06]">
                                {step.model} · {step.duration_ms.toFixed(0)}ms
                              </span>
                            )}
                            {isDrift && (
                              <span className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-rose-500/15 text-rose-300 border border-rose-500/30">
                                {t.driftBadge}
                              </span>
                            )}
                          </div>

                          <div className="flex items-center gap-2 shrink-0">
                            <span
                              className={`text-[11px] font-mono tabular-nums ${
                                step.prefix_cache_ratio >= 0.85
                                  ? "text-emerald-400"
                                  : "text-amber-300"
                              }`}
                            >
                              LCP {(step.prefix_cache_ratio * 100).toFixed(0)}%
                            </span>
                            <button
                              type="button"
                              onClick={(e) => {
                                e.stopPropagation();
                                setSelectedStepIdx(step.step_index);
                                openWorkbenchTab("fork");
                              }}
                              className="lg:hidden text-[11px] font-medium text-zinc-300 bg-white/[0.06] px-2 py-0.5 rounded-md"
                            >
                              {t.jumpFork}
                            </button>
                          </div>
                        </div>

                        {/* Tool Call or Response Preview */}
                        <div className="text-xs font-mono text-zinc-300 truncate mb-2.5">
                          {step.tool_calls.length > 0 ? (
                            <span>
                              <span className="text-zinc-100 font-medium">
                                {step.tool_calls.map((tc) => tc.name).join(", ")}
                              </span>
                              <span className="text-zinc-500 ml-1.5">
                                {JSON.stringify(step.tool_calls[0].arguments)}
                              </span>
                            </span>
                          ) : (
                            <span className="text-zinc-400">{step.assembled_text}</span>
                          )}
                        </div>

                        {/* Sleek Proportional Token X-Ray Bar */}
                        <div className="space-y-1.5">
                          <div
                            className="h-1.5 rounded-full overflow-hidden flex bg-zinc-950"
                            style={{ width: `${widthScale}%` }}
                          >
                            <div
                              style={{ width: `${(tb.system_tokens / tbSum) * 100}%` }}
                              className="bg-indigo-400"
                            />
                            <div
                              style={{ width: `${(tb.tools_schema_tokens / tbSum) * 100}%` }}
                              className="bg-amber-400"
                            />
                            <div
                              style={{ width: `${(tb.history_tokens / tbSum) * 100}%` }}
                              className="bg-sky-400"
                            />
                            <div
                              style={{ width: `${(tb.tool_output_tokens / tbSum) * 100}%` }}
                              className="bg-violet-400"
                            />
                          </div>

                          <div className="flex items-center justify-between gap-2 text-[11px] font-mono tabular-nums text-zinc-500">
                            <span className="truncate">
                              {step.prompt_tokens.toLocaleString()} {t.promptTokLabel} ({t.toolsTokShort}{" "}
                              {tb.tools_schema_tokens.toLocaleString()} · {t.outTokShort}{" "}
                              {tb.tool_output_tokens.toLocaleString()})
                            </span>
                            {step.anomaly_flags.length > 0 && (
                              <span className="text-rose-400 shrink-0">
                                {localizeRootCause(step.anomaly_flags[0].code, lang)}
                              </span>
                            )}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* RIGHT: Multi-Tab Engineering Workbench */}
              <div
                className={`${
                  mobilePane === "workbench" ? "flex" : "hidden"
                } lg:flex lg:col-span-7 flex-col lg:min-h-0 bg-[#0c0c0f]/40`}
              >
                {/* Clean Hairline Tabs */}
                <div className="px-3 sm:px-5 border-b border-white/[0.07] flex items-center gap-1 bg-[#09090b] overflow-x-auto no-scrollbar">
                  {(
                    [
                      ["biopsy", t.tabBiopsy, Stethoscope],
                      ["fork", t.tabFork(selectedStepIdx), Zap],
                      ["diff", t.tabDiff, GitCompare],
                      ["xray", t.tabXray(selectedStepIdx), Code2],
                      ["pytest", t.tabPytest, FileCode2],
                    ] as const
                  ).map(([tabKey, label, Icon]) => {
                    const active = activeTab === tabKey;
                    return (
                      <button
                        key={tabKey}
                        onClick={() => setActiveTab(tabKey)}
                        className={`px-3 py-3 text-xs font-medium border-b-2 inline-flex items-center gap-1.5 whitespace-nowrap transition cursor-pointer ${
                          active
                            ? "border-zinc-100 text-zinc-100"
                            : "border-transparent text-zinc-500 hover:text-zinc-300"
                        }`}
                      >
                        <Icon className="w-3.5 h-3.5" />
                        {label}
                      </button>
                    );
                  })}
                </div>

                {/* Tab Body */}
                <div className="flex-1 lg:overflow-y-auto p-4 sm:p-5 space-y-4">
                  {loading && (
                    <div className="text-xs text-zinc-500 animate-pulse">{t.loadingTelemetry}</div>
                  )}

                  {/* TAB 1: COUNTERFACTUAL BIOPSY DOCTOR */}
                  {activeTab === "biopsy" && (
                    <div className="space-y-5">
                      {detail.biopsy_report ? (
                        <>
                          {/* Executive Root-Cause Card */}
                          <div className="rounded-2xl border border-white/[0.09] bg-zinc-900/60 p-4 sm:p-5 space-y-3">
                            <div className="flex items-center justify-between flex-wrap gap-2">
                              <div className="flex items-center gap-2 flex-wrap">
                                <span className="text-xs font-semibold text-zinc-100 flex items-center gap-1.5">
                                  <Sparkles className="w-3.5 h-3.5 text-amber-300" />
                                  {t.verifiedRootCause}：
                                  {localizeRootCause(
                                    detail.biopsy_report.root_cause_category,
                                    lang
                                  )}
                                </span>
                                <span className="text-[11px] font-mono px-2 py-0.5 rounded-full bg-rose-500/15 text-rose-300 border border-rose-500/25">
                                  {t.driftAtStep(detail.biopsy_report.suspected_drift_step)}
                                </span>
                              </div>
                              <span className="text-[11px] font-mono text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 rounded-full">
                                {t.confidence((detail.biopsy_report.confidence * 100).toFixed(0))}
                              </span>
                            </div>

                            <p className="text-xs sm:text-[13px] text-zinc-300 leading-relaxed">
                              {localizeExecutiveSummary(
                                detail.tape.tape_id,
                                detail.biopsy_report.executive_summary,
                                lang
                              )}
                            </p>
                          </div>

                          {/* Ablation Runs */}
                          <div className="space-y-2.5">
                            <h3 className="text-[11px] font-medium uppercase tracking-wider text-zinc-500">
                              {t.ablationHeader}
                            </h3>
                            <div className="space-y-2.5">
                              {detail.biopsy_report.ablation_runs.map((ab) => (
                                <div
                                  key={ab.experiment_id}
                                  className={`rounded-xl border p-3.5 sm:p-4 space-y-3 transition ${
                                    ab.resolved_failure
                                      ? "bg-emerald-500/[0.04] border-emerald-500/30"
                                      : "bg-zinc-900/40 border-white/[0.06]"
                                  }`}
                                >
                                  <div className="flex items-center justify-between gap-2 flex-wrap">
                                    <div className="flex items-center gap-2 min-w-0">
                                      {ab.resolved_failure ? (
                                        <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                                      ) : (
                                        <Activity className="w-4 h-4 text-zinc-500 shrink-0" />
                                      )}
                                      <span className="text-xs sm:text-[13px] font-medium text-zinc-100">
                                        {localizeHypothesis(ab.hypothesis_name, lang)}{" "}
                                        <span className="text-zinc-500 font-mono text-xs">
                                          (S{ab.step_index})
                                        </span>
                                      </span>
                                    </div>

                                    <div className="flex items-center gap-2 shrink-0">
                                      {ab.resolved_failure && (
                                        <span className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
                                          {t.provenFix}
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
                                          className="inline-flex items-center gap-0.5 text-xs text-zinc-300 hover:text-white cursor-pointer"
                                        >
                                          {t.inspectDiff}
                                        </button>
                                      )}
                                    </div>
                                  </div>

                                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-[11px] font-mono">
                                    <div className="bg-black/40 border border-white/[0.06] rounded-lg p-2.5 text-rose-300/95 break-all">
                                      <span className="text-zinc-500 block text-[10px] mb-0.5 font-sans">
                                        {t.beforeLabel}
                                      </span>
                                      {ab.before_summary}
                                    </div>
                                    <div className="bg-black/40 border border-white/[0.06] rounded-lg p-2.5 text-emerald-300/95 break-all">
                                      <span className="text-zinc-500 block text-[10px] mb-0.5 font-sans">
                                        {t.afterLabel}
                                      </span>
                                      {ab.after_summary}
                                    </div>
                                  </div>

                                  <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] font-mono tabular-nums text-zinc-500">
                                    <span>
                                      Tokens: {ab.prompt_tokens_before.toLocaleString()} →{" "}
                                      <strong className="text-zinc-200">
                                        {ab.prompt_tokens_after.toLocaleString()}
                                      </strong>{" "}
                                      ({ab.token_reduction_pct >= 0 ? "-" : "+"}
                                      {Math.abs(ab.token_reduction_pct).toFixed(1)}%)
                                    </span>
                                    <span>
                                      KV LCP: {(ab.cache_ratio_before * 100).toFixed(0)}% →{" "}
                                      <strong className="text-emerald-400">
                                        {(ab.cache_ratio_after * 100).toFixed(0)}%
                                      </strong>
                                    </span>
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>

                          {/* Engineering Playbook */}
                          <div className="rounded-xl border border-white/[0.07] bg-zinc-900/40 p-4 space-y-2">
                            <h4 className="text-xs font-semibold text-zinc-200">
                              {t.remediationHeader}
                            </h4>
                            <pre className="text-xs text-zinc-400 whitespace-pre-wrap font-sans leading-relaxed">
                              {localizePlaybook(
                                detail.tape.tape_id,
                                detail.biopsy_report.verified_fix_recommendation,
                                lang
                              )}
                            </pre>
                          </div>
                        </>
                      ) : (
                        <div className="rounded-2xl border border-white/[0.07] bg-zinc-900/30 p-8 text-center space-y-3">
                          <div className="w-10 h-10 rounded-xl bg-white/[0.05] border border-white/[0.08] flex items-center justify-center mx-auto">
                            <Stethoscope className="w-5 h-5 text-zinc-300" />
                          </div>
                          <div className="text-sm font-semibold text-zinc-100">
                            {t.noBiopsyTitle}
                          </div>
                          <p className="text-xs text-zinc-400 max-w-md mx-auto leading-relaxed">
                            {t.noBiopsyDesc}
                          </p>
                          <button
                            onClick={() => void handleRunDoctor()}
                            className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg text-xs font-semibold bg-zinc-100 hover:bg-white text-zinc-950 transition cursor-pointer"
                          >
                            <Play className="w-3.5 h-3.5 fill-zinc-950" /> {t.launchBiopsy}
                          </button>
                        </div>
                      )}
                    </div>
                  )}

                  {/* TAB 2: INTERACTIVE TIME-TRAVEL FORK WORKBENCH */}
                  {activeTab === "fork" && selectedStep && (
                    <div className="space-y-4">
                      <div className="rounded-xl border border-white/[0.09] bg-zinc-900/60 p-4 text-xs space-y-1.5">
                        <div className="font-semibold text-zinc-100 flex items-center gap-1.5">
                          <Zap className="w-3.5 h-3.5 text-amber-300" />
                          {t.forkTargetTitle(selectedStep.step_index, detail.tape.step_count)}
                        </div>
                        <p className="text-zinc-400 leading-relaxed">
                          {selectedStep.step_index > 1
                            ? t.forkTargetDescPrefix(
                                selectedStep.step_index - 1,
                                selectedStep.step_index
                              )
                            : t.forkTargetDescStep1}
                        </p>
                      </div>

                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                        <div>
                          <label className="block text-[11px] text-zinc-400 mb-1.5">
                            {t.branchLabel}
                          </label>
                          <input
                            type="text"
                            value={forkLabel}
                            onChange={(e) => setForkLabel(e.target.value)}
                            className="w-full bg-zinc-900 border border-white/[0.09] rounded-lg px-3 py-2 text-xs text-zinc-100 focus:outline-none focus:border-zinc-400"
                          />
                        </div>
                        <div>
                          <label className="block text-[11px] text-zinc-400 mb-1.5">
                            {t.swapModel}
                          </label>
                          <select
                            value={swapModel}
                            onChange={(e) => setSwapModel(e.target.value)}
                            className="w-full bg-zinc-900 border border-white/[0.09] rounded-lg px-3 py-2 text-xs text-zinc-100 focus:outline-none focus:border-zinc-400"
                          >
                            <option value="">{t.keepModel(selectedStep.model)}</option>
                            <option value="gpt-4o">gpt-4o</option>
                            <option value="gpt-4.1-mini">gpt-4.1-mini</option>
                            <option value="claude-3-7-sonnet">claude-3-7-sonnet</option>
                            <option value="deepseek-chat">deepseek-chat</option>
                            <option value="qwen-max">qwen-max</option>
                          </select>
                        </div>
                      </div>

                      {/* Clean Minimalist Switch List */}
                      <div className="rounded-xl border border-white/[0.07] bg-zinc-900/40 divide-y divide-white/[0.05]">
                        <div className="px-4 py-2.5 text-[11px] font-medium text-zinc-500">
                          {t.interventionsHeader}
                        </div>

                        <div className="px-4 py-3 flex items-center justify-between gap-3">
                          <div className="min-w-0">
                            <div className="text-xs font-medium text-zinc-100">{t.pruneTitle}</div>
                            <div className="text-[11px] text-zinc-500 mt-0.5">{t.pruneDesc}</div>
                          </div>
                          <div className="flex items-center gap-2.5 shrink-0">
                            <input
                              type="number"
                              min={1}
                              max={30}
                              value={pruneTopK}
                              onChange={(e) => setPruneTopK(Number(e.target.value))}
                              disabled={!enablePruneTools}
                              className="w-14 bg-zinc-950 border border-white/10 rounded-md px-2 py-1 text-xs font-mono text-center disabled:opacity-40"
                            />
                            <ToggleSwitch
                              checked={enablePruneTools}
                              onChange={setEnablePruneTools}
                            />
                          </div>
                        </div>

                        <div className="px-4 py-3 flex items-center justify-between gap-3">
                          <div className="min-w-0">
                            <div className="text-xs font-medium text-zinc-100">
                              {t.compactTitle}
                            </div>
                            <div className="text-[11px] text-zinc-500 mt-0.5">{t.compactDesc}</div>
                          </div>
                          <div className="flex items-center gap-2.5 shrink-0">
                            <input
                              type="number"
                              min={300}
                              max={5000}
                              step={100}
                              value={compactMaxChars}
                              onChange={(e) => setCompactMaxChars(Number(e.target.value))}
                              disabled={!enableCompactContext}
                              className="w-18 bg-zinc-950 border border-white/10 rounded-md px-2 py-1 text-xs font-mono text-center disabled:opacity-40"
                            />
                            <ToggleSwitch
                              checked={enableCompactContext}
                              onChange={setEnableCompactContext}
                            />
                          </div>
                        </div>

                        <div className="px-4 py-3 flex items-center justify-between gap-3">
                          <div className="min-w-0">
                            <div className="text-xs font-medium text-zinc-100">
                              {t.stabilizeTitle}
                            </div>
                            <div className="text-[11px] text-zinc-500 mt-0.5">
                              {t.stabilizeDesc}
                            </div>
                          </div>
                          <ToggleSwitch checked={stabilizeCache} onChange={setStabilizeCache} />
                        </div>

                        <div className="px-4 py-3 flex items-center justify-between gap-3">
                          <div className="min-w-0">
                            <div className="text-xs font-medium text-zinc-100">
                              {t.sandboxTitle}
                            </div>
                            <div className="text-[11px] text-zinc-500 mt-0.5">{t.sandboxDesc}</div>
                          </div>
                          <ToggleSwitch checked={dryRunWrites} onChange={setDryRunWrites} />
                        </div>

                        <div className="px-4 py-3 flex items-center justify-between gap-3">
                          <div className="min-w-0">
                            <div className="text-xs font-medium text-zinc-100">
                              {t.liveApiTitle}
                            </div>
                            <div className="text-[11px] text-zinc-500 mt-0.5">{t.liveApiDesc}</div>
                          </div>
                          <ToggleSwitch
                            checked={preferLiveUpstream}
                            onChange={setPreferLiveUpstream}
                          />
                        </div>
                      </div>

                      <div>
                        <div className="flex items-center justify-between gap-2 mb-1.5 flex-wrap">
                          <label className="text-[11px] text-zinc-400">{t.appendGuardLabel}</label>
                          <button
                            type="button"
                            onClick={() =>
                              setAppendSystemRule(
                                "CRITICAL GROUNDING RULE: If the preceding search/query tool returned empty results ([] or null), NEVER fabricate IDs or invoke mutation tools. Ask for clarification."
                              )
                            }
                            className="text-[11px] text-zinc-300 hover:text-white underline cursor-pointer"
                          >
                            {t.insertPreset}
                          </button>
                        </div>
                        <textarea
                          rows={3}
                          value={appendSystemRule}
                          onChange={(e) => setAppendSystemRule(e.target.value)}
                          placeholder={t.appendPlaceholder}
                          className="w-full bg-zinc-900 border border-white/[0.09] rounded-xl p-3 text-xs font-mono text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:border-zinc-400"
                        />
                      </div>

                      <button
                        onClick={() => void handleExecuteFork()}
                        disabled={runningAction === "fork"}
                        className="w-full py-3 rounded-xl font-semibold text-xs bg-zinc-100 hover:bg-white text-zinc-950 shadow-sm transition flex items-center justify-center gap-2 cursor-pointer disabled:opacity-50"
                      >
                        <Zap className="w-4 h-4 fill-zinc-950" />
                        {runningAction === "fork"
                          ? t.executingFork
                          : t.executeForkBtn(selectedStep.step_index)}
                      </button>
                    </div>
                  )}

                  {/* TAB 3: SIDE-BY-SIDE BRANCH DIFF */}
                  {activeTab === "diff" && (
                    <div className="space-y-4">
                      {detail.children.length > 0 && (
                        <div className="flex flex-col sm:flex-row sm:items-center gap-2 text-xs">
                          <span className="text-zinc-400 shrink-0">{t.compareSelectLabel}：</span>
                          <select
                            value={compareTargetId}
                            onChange={async (e) => {
                              const cid = e.target.value;
                              setCompareTargetId(cid);
                              await fetchDiff(detail.tape.tape_id, cid);
                            }}
                            className="w-full sm:w-auto flex-1 bg-zinc-900 border border-white/[0.09] rounded-lg px-3 py-2 text-xs font-mono text-zinc-100 focus:outline-none"
                          >
                            {detail.children.map((ch) => (
                              <option key={ch.tape_id} value={ch.tape_id}>
                                {ch.tape_id} —{" "}
                                {localizeForkLabel(ch.fork_label, ch.fork_from_step, lang)}
                              </option>
                            ))}
                          </select>
                        </div>
                      )}

                      {comparison ? (
                        <>
                          <div className="rounded-xl border border-emerald-500/25 bg-emerald-500/[0.05] p-3.5 grid grid-cols-3 gap-2 text-xs">
                            <div>
                              <div className="text-[11px] text-zinc-400">{t.diffPoint}</div>
                              <div className="text-emerald-300 font-mono font-semibold mt-0.5">
                                {comparison.divergence_step
                                  ? t.stepPrefix(comparison.divergence_step)
                                  : "—"}
                              </div>
                            </div>
                            <div>
                              <div className="text-[11px] text-zinc-400">{t.diffSavedTok}</div>
                              <div className="text-emerald-300 font-mono font-semibold tabular-nums mt-0.5">
                                {comparison.replayed_tokens_saved.toLocaleString()}
                              </div>
                            </div>
                            <div>
                              <div className="text-[11px] text-zinc-400">{t.diffSavedTime}</div>
                              <div className="text-emerald-300 font-mono font-semibold tabular-nums mt-0.5">
                                {(comparison.replayed_steps_saved_ms / 1000).toFixed(2)}s
                              </div>
                            </div>
                          </div>

                          <div className="space-y-2.5">
                            {comparison.step_diffs.map((row) => {
                              const isChanged =
                                row.relation === "fork_point_modified" ||
                                row.relation === "diverged_continuation";
                              return (
                                <div
                                  key={row.step_index}
                                  className={`rounded-xl border p-3.5 text-xs space-y-2.5 ${
                                    isChanged
                                      ? "bg-zinc-900/80 border-zinc-500/40"
                                      : "bg-zinc-900/30 border-white/[0.06]"
                                  }`}
                                >
                                  <div className="flex flex-wrap items-center justify-between gap-2 text-[11px]">
                                    <span className="font-semibold text-zinc-100 flex items-center gap-2">
                                      <span>{t.stepPrefix(row.step_index)}</span>
                                      <span
                                        className={`px-2 py-0.5 rounded-full text-[10px] ${
                                          isChanged
                                            ? "bg-zinc-100 text-zinc-950 font-semibold"
                                            : "bg-white/[0.06] text-zinc-400"
                                        }`}
                                      >
                                        {localizeRelation(row.relation, lang)}
                                      </span>
                                    </span>

                                    <span className="font-mono tabular-nums text-zinc-400">
                                      Tok {row.left_prompt_tokens.toLocaleString()} →{" "}
                                      <strong className="text-zinc-100">
                                        {row.right_prompt_tokens.toLocaleString()}
                                      </strong>{" "}
                                      · LCP {(row.left_cache_ratio * 100).toFixed(0)}% →{" "}
                                      <strong className="text-emerald-400">
                                        {(row.right_cache_ratio * 100).toFixed(0)}%
                                      </strong>
                                    </span>
                                  </div>

                                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 font-mono text-[11px]">
                                    <div className="bg-black/45 border border-white/[0.06] rounded-lg p-2.5 text-rose-300/95 break-all">
                                      <div className="text-[10px] font-sans text-zinc-500 mb-1">
                                        {t.baselineTag} ({comparison.left_tape.tape_id})
                                      </div>
                                      {row.left_tools_called.length > 0
                                        ? `${row.left_tools_called.join(", ")}(${row.left_args_preview})`
                                        : row.left_text_preview || "—"}
                                    </div>
                                    <div className="bg-black/45 border border-white/[0.06] rounded-lg p-2.5 text-emerald-300/95 break-all">
                                      <div className="text-[10px] font-sans text-zinc-500 mb-1">
                                        {t.forkedTag} ({comparison.right_tape.tape_id})
                                      </div>
                                      {row.right_tools_called.length > 0
                                        ? `${row.right_tools_called.join(", ")}(${row.right_args_preview})`
                                        : row.right_text_preview || "—"}
                                    </div>
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </>
                      ) : (
                        <div className="rounded-2xl border border-white/[0.07] bg-zinc-900/30 p-8 text-center text-xs text-zinc-400">
                          {t.noDiffHint}
                        </div>
                      )}
                    </div>
                  )}

                  {/* TAB 4: STEP PAYLOAD & PREFIX KV CACHE X-RAY */}
                  {activeTab === "xray" && selectedStep && (
                    <div className="space-y-3.5 text-xs">
                      {selectedStep.prefix_break_snippet && (
                        <div className="rounded-xl border border-rose-500/30 bg-rose-500/[0.06] p-3.5 space-y-1.5">
                          <div className="font-semibold text-rose-300">
                            {t.cacheBreakTitle(selectedStep.prefix_break_offset)}
                          </div>
                          <div className="font-mono text-[11px] text-rose-200/90 break-all">
                            {selectedStep.prefix_break_snippet}
                          </div>
                        </div>
                      )}

                      <div className="rounded-xl border border-white/[0.07] bg-zinc-900/40 p-3.5 space-y-1">
                        <div className="text-[11px] text-zinc-500">{t.fingerprintTitle}</div>
                        <div className="font-mono text-xs text-zinc-200 break-all">
                          {selectedStep.canonical_hash}
                        </div>
                      </div>

                      <div className="rounded-xl border border-white/[0.07] bg-zinc-900/40 p-3.5 space-y-2">
                        <div className="text-[11px] text-zinc-500">{t.rawReqTitle}</div>
                        <pre className="text-[11px] font-mono text-zinc-300 overflow-x-auto max-h-96 leading-relaxed">
                          {JSON.stringify(selectedStep.raw_request, null, 2)}
                        </pre>
                      </div>

                      {selectedStep.reasoning_text && (
                        <div className="rounded-xl border border-violet-500/25 bg-violet-500/[0.05] p-3.5 space-y-1.5">
                          <div className="text-[11px] text-violet-300 font-medium">
                            {t.reasoningTitle}
                          </div>
                          <pre className="text-[11px] font-mono text-violet-100 whitespace-pre-wrap leading-relaxed">
                            {selectedStep.reasoning_text}
                          </pre>
                        </div>
                      )}

                      {detail.mcp_spans.filter((sp) => sp.step_index === selectedStep.step_index)
                        .length > 0 && (
                        <div className="rounded-xl border border-emerald-500/25 bg-emerald-500/[0.04] p-3.5 space-y-2">
                          <div className="text-[11px] text-emerald-300 font-medium">
                            {t.mcpSpansTitle(selectedStep.step_index)}
                          </div>
                          <pre className="text-[11px] font-mono text-emerald-200/90 overflow-x-auto max-h-64 leading-relaxed">
                            {JSON.stringify(
                              detail.mcp_spans.filter(
                                (sp) => sp.step_index === selectedStep.step_index
                              ),
                              null,
                              2
                            )}
                          </pre>
                        </div>
                      )}

                      <div className="rounded-xl border border-white/[0.07] bg-zinc-900/40 p-3.5 space-y-2">
                        <div className="text-[11px] text-zinc-500">{t.rawRespTitle}</div>
                        <pre className="text-[11px] font-mono text-zinc-300 overflow-x-auto max-h-96 leading-relaxed">
                          {JSON.stringify(selectedStep.raw_response, null, 2)}
                        </pre>
                      </div>
                    </div>
                  )}

                  {/* TAB 5: GENERATED PYTEST REGRESSION SUITE */}
                  {activeTab === "pytest" && (
                    <div className="space-y-3">
                      <div className="flex items-center justify-between gap-2 flex-wrap">
                        <div className="text-xs text-zinc-400">
                          {t.pytestDesc} (
                          <code className="text-zinc-200 font-mono">{detail.tape.tape_id}</code>)
                        </div>
                        <button
                          onClick={() => void handleCopyPytest()}
                          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-zinc-100 hover:bg-white text-zinc-950 transition cursor-pointer"
                        >
                          {copiedPytest ? (
                            <Check className="w-3.5 h-3.5" />
                          ) : (
                            <Copy className="w-3.5 h-3.5" />
                          )}
                          {copiedPytest ? t.copied : t.copyPytest}
                        </button>
                      </div>
                      <pre className="rounded-xl border border-white/[0.07] bg-[#060608] p-4 text-[11px] font-mono text-zinc-200 overflow-x-auto leading-relaxed">
                        {pytestCode || "# Click Run Biopsy Doctor to synthesize regression suite."}
                      </pre>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </main>
        ) : (
          <div className="flex-1 flex flex-col items-center justify-center p-8 text-center gap-3 text-sm text-zinc-500">
            <span>{t.emptyVaultHint}</span>
            <button
              onClick={() => setMobileVaultOpen(true)}
              className="lg:hidden inline-flex items-center gap-1 px-3 py-1.5 rounded-lg bg-zinc-100 text-zinc-950 text-xs font-semibold"
            >
              {t.vaultBtn} <ChevronRight className="w-3.5 h-3.5" />
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
