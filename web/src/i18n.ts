import type { TapeRecord } from "./types";

export type Lang = "zh" | "en";

const DEMO_TAPE_ZH: Record<
  string,
  { title: string; failureReason?: string; executiveSummary?: string; playbook?: string }
> = {
  tape_demo_mcp_overload: {
    title: "电商售后客服 Agent — 42 个 MCP 工具 Schema 过载调错退款接口",
    failureReason:
      "第 5 步因挂载 42 个冗余工具导致注意力稀释，误调 `legacy_internal_admin_wipe_order` 清空整单而非 `issue_partial_refund` 部分退款。",
    executiveSummary:
      "在 Step 5 定位到典型『MCP 工具 Schema 过载（Tool Schema Overload）』：42 个工具定义占据超 80% 输入上下文，导致模型混淆废弃管理接口与部分退款接口。经反事实分叉实验证实，启用保序 BM25 Top-5 动态工具裁剪可在保持前缀顺序的前提下精准恢复 `issue_partial_refund` 调用。",
    playbook:
      "1. 接入动态工具路由层（BM25 / 向量 Top-K），单轮暴露工具数控制在 5–8 个以内，并保留原始相对顺序以保护 Prefix KV Cache。\n2. 将 `wipe_*` / `delete_*` 等高危内部管理接口从客服 Agent 默认工具集中物理隔离，或在网关层开启二次确认护栏。",
  },
  tape_demo_context_rot: {
    title: "FinOps 云账单审批 Agent — 18KB 脏 JSON 引发上下文遗忘超额打款",
    failureReason:
      "第 2 步塞入 18,400 字符未压缩监控 JSON，触发 Lost-in-the-Middle 上下文腐烂，导致第 3 步遗忘 $500 自动审批上限并违规批出 $2,400。",
    executiveSummary:
      "上游因果拐点位于工具输出膨胀段：超长原始监控 JSON 淹没了 System Prompt 中的 $500 审批红线约束。经结构感知上下文压缩（<=900 字符）分叉验证，输入 Token 大幅下降且模型重新严格遵守审批上限策略。",
    playbook:
      "1. 对超过 1,000 字符的工具返回 JSON/日志启用结构感知摘要压缩（保留首尾关键字段与聚合指标）。\n2. 对金额审批等硬性安全红线，除 System Prompt 声明外应在工具 Schema 或执行层增加确定性校验。",
  },
  tape_demo_empty_retrieval: {
    title: "SRE 值班排障 Agent — 空日志检索触发幻觉捏造事故单并误回滚",
    failureReason:
      "第 1 步 `query_datadog_incidents` 返回空数组 `[]` 后，模型在第 2 步凭空捏造事故单号 `INC-9042` 并触发生产服务回滚。",
    executiveSummary:
      "检测到『静默空检索幻觉（Silent Empty Retrieval）』：前置查询工具返回空结果 `[]`，但 Prompt 缺少空结果停机约束，诱发模型编造 `INC-9042` 参数调用高危回滚工具。注入空检索接地护栏后，模型安全终止写操作并请求人工确认。",
    playbook:
      "1. 在 System Prompt 中显式声明空检索接地规则（当查询返回 `[]` 或 `null` 时严禁猜测实体 ID 或调用变更类工具）。\n2. 在回滚/删除等变更工具入口校验前置上下文中实体 ID 的真实来源（Grounding Check）。",
  },
  tape_demo_kv_cache_thrash: {
    title: "代码重构 Agent — 秒级动态时间戳与工具乱序击穿 Prefix KV Cache",
    failureReason:
      "System Prompt 顶部注入每秒变化的当前时间戳且每轮工具列表顺序翻转，导致 Prefix KV Cache 命中率跌至 3%，推理成本暴涨 4.2 倍。",
    executiveSummary:
      "长公共前缀（LCP）X 光检测发现请求头部在字符偏移早期即发生断裂：动态秒级时间戳与非确定性工具数组顺序使跨轮次 KV Cache 完全失效。启用前缀稳定化算子后，跨轮次前缀重合度恢复至 95%+。",
    playbook:
      "1. 严禁在 System Prompt 顶部放置秒级时间戳、随机 Trace ID 或会话 Nonce（如需时间信息请置于末尾用户消息或按小时/天粒度对齐）。\n2. 对 `tools` JSON Schema 数组按工具名称做确定性固定排序后再发送给推理引擎。",
  },
  tape_live_ds_long_20step_sre: {
    title: "[真实 DeepSeek-V4.1-Flash · 19步超长线程] 跨集群支付事故排障（44 工具 + 24KB 脏日志膨胀）",
    failureReason:
      "连续 19 步真实流式调用累计消耗 304,673 输入 Token；第 5 步注入 24.5KB 原始 K8s 转储后引发后续 14 步严重上下文膨胀（单步达 21,616 Token）。",
  },
  tape_live_ds_long_loop_rag: {
    title: "[真实 DeepSeek-V4.1-Flash · 13步长线程] 跨境清结算核账 Agent（深层空检索与死循环重试）",
    failureReason:
      "完成 9 批对账后，在第 10–12 步连续 3 次以相同参数调用 `search_swift_dispute_wire(SWIFT-ghost-991)` 返回空数组 `[]`，触发循环熔断。",
  },
  tape_live_ds_kv_thrash: {
    title: "[真实 DeepSeek-V4.1-Flash · 10步] 动态毫秒时间戳与工具乱序击穿真实 KV Cache 实测",
    failureReason:
      "每轮变更 System Prompt 顶部毫秒时间戳并打乱工具列表顺序，导致 DeepSeek 服务端真实 KV Cache 命中率跌至 0.0%（开启稳定化后实测恢复至 94.1%）。",
  },
  tape_live_ds_anthropic_sse: {
    title: "[真实 DeepSeek-V4.1-Flash · 双协议] Anthropic /v1/messages 流式多步工具调用实测",
  },
};

const ROOT_CAUSE_LABELS: Record<string, { zh: string; en: string }> = {
  TOOL_SCHEMA_OVERLOAD: { zh: "工具 Schema 过载", en: "Tool Schema Overload" },
  CONTEXT_MIDDLE_LOSS: { zh: "中间上下文腐烂遗忘", en: "Context Middle Loss" },
  SILENT_EMPTY_RETRIEVAL: { zh: "空检索实体幻觉", en: "Silent Empty Retrieval" },
  PREFIX_CACHE_THRASHING: { zh: "Prefix KV 缓存击穿", en: "Prefix KV Cache Thrashing" },
  TOOL_CALL_LOOP: { zh: "死循环重复调用", en: "Cyclic Tool-Call Loop" },
  PROMPT_AMBIGUITY: { zh: "指令边界模糊", en: "Prompt Ambiguity" },
};

const HYPOTHESIS_ZH: Record<string, string> = {
  "BM25 Semantic Tool Pruning (Top-5)": "保序 BM25 语义工具裁剪 (Top-5)",
  "BM25 Tool Scope Narrowing Probe": "BM25 工具范围收敛探针 (Top-6)",
  "Structure-Aware Middle Context Compaction": "结构感知中间上下文压缩 (≤900 字)",
  "Context Compaction & Zero-Temp Stability Probe": "上下文压缩与零温稳定性探针",
  "Empty-Result Grounding Guardrail": "空检索防幻觉强约束护栏注入",
  "Cyclic Tool-Call Loop Breaker": "重复工具调用熔断护栏",
  "Prefix KV Cache Header & Schema Stabilization": "Prefix KV 缓存头部去噪与工具保序",
};

const RELATION_LABELS: Record<string, { zh: string; en: string }> = {
  identical_replay: { zh: "VCR 快进重放", en: "VCR Replay" },
  fork_point_modified: { zh: "分叉干预拐点", en: "Fork Point" },
  diverged_continuation: { zh: "分叉后延续", en: "Diverged" },
  only_left: { zh: "仅原分支", en: "Baseline Only" },
  only_right: { zh: "仅新分支", en: "Fork Only" },
};

const STATUS_LABELS: Record<TapeRecord["status"], { zh: string; en: string }> = {
  success: { zh: "成功", en: "SUCCESS" },
  failed: { zh: "失败", en: "FAILED" },
  loop_aborted: { zh: "循环熔断", en: "LOOP" },
  running: { zh: "运行中", en: "RUNNING" },
  diverged: { zh: "对照偏离", en: "CONTROL" },
};

export function localizeStatus(status: TapeRecord["status"], lang: Lang): string {
  return STATUS_LABELS[status]?.[lang] ?? status.toUpperCase();
}

export function localizeRootCause(code: string, lang: Lang): string {
  return ROOT_CAUSE_LABELS[code]?.[lang] ?? code;
}

export function localizeHypothesis(name: string, lang: Lang): string {
  if (lang === "en") return name;
  return HYPOTHESIS_ZH[name] ?? name;
}

export function localizeRelation(rel: string, lang: Lang): string {
  return RELATION_LABELS[rel]?.[lang] ?? rel;
}

export function localizeForkLabel(label: string | null | undefined, step: number | null | undefined, lang: Lang): string {
  if (!label) return lang === "zh" ? `分叉 @ 第 ${step ?? "?"} 步` : `Fork @ S${step ?? "?"}`;
  if (lang === "en") return label;
  return label
    .replace("Doctor: BM25 Prune Tools Top-5", "诊断实验：BM25 裁剪 Top-5")
    .replace("Doctor: Compact Tool Bloat (<=900 chars)", "诊断实验：上下文压缩 (≤900)")
    .replace("Doctor: Empty-Retrieval Anti-Hallucination Guard", "诊断实验：空检索防幻觉护栏")
    .replace("Doctor: Loop Breaker Guard", "诊断实验：循环调用熔断")
    .replace("Doctor: Stabilize Prefix KV Cache", "诊断实验：稳定 KV 缓存前缀")
    .replace("Studio Counterfactual Fork", "手动时间旅行分叉");
}

export function localizeTapeTitle(tape: TapeRecord, lang: Lang): string {
  if (lang === "en") return tape.title;
  if (DEMO_TAPE_ZH[tape.tape_id]) return DEMO_TAPE_ZH[tape.tape_id].title;
  if (tape.parent_tape_id && DEMO_TAPE_ZH[tape.parent_tape_id]) {
    const parentZh = DEMO_TAPE_ZH[tape.parent_tape_id].title.split(" — ")[0];
    const forkPart = localizeForkLabel(tape.fork_label, tape.fork_from_step, "zh");
    return `${parentZh} · ${forkPart}`;
  }
  return tape.title;
}

export function localizeFailureReason(tape: TapeRecord, lang: Lang): string | null {
  if (!tape.failure_reason) return null;
  if (lang === "en") return tape.failure_reason;
  if (DEMO_TAPE_ZH[tape.tape_id]?.failureReason) {
    return DEMO_TAPE_ZH[tape.tape_id].failureReason!;
  }
  if (tape.failure_reason === "Counterfactual control ablation did not resolve primary fault.") {
    return "该组对照消融实验未消除主故障（用于排除非主因干扰项）。";
  }
  return tape.failure_reason;
}

export function localizeExecutiveSummary(tapeId: string, raw: string, lang: Lang): string {
  if (lang === "en") return raw;
  return DEMO_TAPE_ZH[tapeId]?.executiveSummary ?? raw;
}

export function localizePlaybook(tapeId: string, raw: string, lang: Lang): string {
  if (lang === "en") return raw;
  return DEMO_TAPE_ZH[tapeId]?.playbook ?? raw;
}

export const UI_TEXT = {
  zh: {
    subtitle: "时间旅行调试与反事实诊断",
    vaultBtn: "轨迹库",
    sync: "刷新",
    searchPlaceholder: "搜索轨迹标题、标签或 ID...",
    filterRoots: "主轨迹",
    filterFailed: "仅看故障",
    filterAll: "全部分支",
    stepsUnit: "步",
    parentPrefix: "源轨迹",
    runDoctor: "AI 根因诊断",
    runningDoctor: "正在消融诊断...",
    forkStepBtn: (s: number) => `从第 ${s} 步分叉`,
    exportPytest: "生成单测",
    downloadBundle: "下载 .ftk.json 录制包",
    kpiSteps: "轨迹步数 / 耗时",
    kpiTokens: "上下文总 Token",
    kpiCache: "KV 缓存命中率",
    kpiCost: "预估调用成本",
    cachedSuffix: "命中",
    mobileSegTimeline: (n: number) => `执行时间轴 (${n})`,
    mobileSegWorkbench: "诊断与分叉台",
    timelineTitle: "因果步骤火焰图 & 上下文透视",
    legendSys: "系统",
    legendTools: "工具定义",
    legendHist: "对话",
    legendOut: "工具输出",
    stepPrefix: (n: number) => `第 ${n} 步`,
    vcrBadge: (ms: string) => `VCR 零成本快进 · ${ms}ms`,
    driftBadge: "根因拐点",
    jumpXray: "报文",
    jumpFork: "去分叉 →",
    promptTokLabel: "输入 Token",
    toolsTokShort: "工具",
    outTokShort: "返回",
    tabBiopsy: "根因诊断",
    tabFork: (s: number) => `时间旅行分叉 (S${s})`,
    tabDiff: "分支对比",
    tabXray: (s: number) => `报文透视 (S${s})`,
    tabPytest: "Pytest 单测",
    loadingTelemetry: "正在加载轨迹遥测数据...",
    loadingRawPayload: "正在按需加载该步原始请求/响应报文...",
    verifiedRootCause: "实证根因",
    driftAtStep: (s: number) => `拐点 @ 第 ${s} 步`,
    confidence: (pct: string) => `置信度 ${pct}%`,
    ablationHeader: "LangGraph 受控反事实消融实验",
    provenFix: "实证修复",
    inspectDiff: "查看对比 →",
    beforeLabel: "干预前",
    afterLabel: "干预后",
    remediationHeader: "工程修复建议清单",
    noBiopsyTitle: "尚未生成反事实活检报告",
    noBiopsyDesc: "运行 5 节点 LangGraph 诊断医生，自动扫描异常指标、定位上游因果拐点并执行对照分叉验证。",
    launchBiopsy: "立即启动根因诊断",
    forkTargetTitle: (s: number, total: number) => `时间旅行目标：第 ${s} / ${total} 步`,
    forkTargetDescPrefix: (prevEnd: number, cur: number) =>
      `第 1..${prevEnd} 步将从本地 SQLite 以 <3ms、0 Token、0 外部写副作用确定性快进跳过；仅对第 ${cur} 步应用下方干预参数重新执行。`,
    forkTargetDescStep1: "第 1 步无前序步骤，将直接应用下方反事实干预参数执行。",
    branchLabel: "新分支备注名",
    swapModel: "切换模型（可选）",
    keepModel: (m: string) => `保持原模型 (${m})`,
    interventionsHeader: "上下文工程与协议沙箱开关",
    pruneTitle: "BM25 语义工具裁剪",
    pruneDesc: "按相关性只保留 Top-K 工具 Schema，保持原始顺序不破坏缓存",
    compactTitle: "结构感知上下文压缩",
    compactDesc: "自动折叠历史步骤中超长的 JSON / 日志工具返回内容",
    stabilizeTitle: "Prefix KV 缓存前缀稳定化",
    stabilizeDesc: "剥离动态秒级时间戳并固定工具列表顺序，消除缓存击穿",
    sandboxTitle: "MCP 破坏性写操作沙箱",
    sandboxDesc: "自动拦截 delete_* / wipe_* / send_* 等危险外部变更调用",
    liveApiTitle: "优先调用真实上游大模型",
    liveApiDesc: "若已配置 DEEPSEEK_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY 则发起真实推理",
    appendGuardLabel: "追加 System 强约束护栏（可选）",
    insertPreset: "+ 填入防空检索幻觉模板",
    appendPlaceholder: "例如：当查询返回空结果时严禁猜测单号；单笔自动退款上限 500 元...",
    executeForkBtn: (s: number) => `执行时间旅行分叉 @ 第 ${s} 步`,
    executingFork: "正在执行反事实分叉...",
    compareSelectLabel: "选择对比分支",
    diffPoint: "分叉拐点",
    diffSavedTok: "VCR 节省 Token",
    diffSavedTime: "VCR 节省耗时",
    baselineTag: "原故障分支",
    forkedTag: "分叉修复分支",
    noDiffHint: "当前暂无对比分支。请先通过「根因诊断」或「时间旅行分叉」创建实验分支。",
    cacheBreakTitle: (offset: number | null | undefined) =>
      `检测到 Prefix KV 缓存前缀断裂（字符偏移 #${offset ?? 0}）`,
    fingerprintTitle: "规范化语义指纹 (SHA-256)",
    rawReqTitle: "原始大模型请求报文（敏感凭证已自动脱敏）",
    reasoningTitle: "模型深度思考链路 (Reasoning / Thinking)",
    mcpSpansTitle: (s: number) => `第 ${s} 步录制的 MCP JSON-RPC 调用记录`,
    rawRespTitle: "原始大模型响应报文",
    pytestDesc: "由当前故障磁带自动编译生成的零 API 成本 CI 回归测试",
    copyPytest: "复制测试代码",
    copied: "已复制",
    emptyVaultHint: "请从轨迹库选择一条运行记录开始时间旅行调试。",
    dismiss: "关闭",
  },
  en: {
    subtitle: "Time-Travel & Counterfactual Biopsy",
    vaultBtn: "Tapes",
    sync: "Sync",
    searchPlaceholder: "Search tapes, tags, IDs...",
    filterRoots: "Roots",
    filterFailed: "Failed",
    filterAll: "All",
    stepsUnit: "steps",
    parentPrefix: "Parent",
    runDoctor: "Run Biopsy Doctor",
    runningDoctor: "Running Biopsy...",
    forkStepBtn: (s: number) => `Fork @ Step ${s}`,
    exportPytest: "Pytest",
    downloadBundle: "Download .ftk.json bundle",
    kpiSteps: "Steps & Latency",
    kpiTokens: "Context Tokens",
    kpiCache: "Prefix KV Cache",
    kpiCost: "Est. Run Cost",
    cachedSuffix: "hit",
    mobileSegTimeline: (n: number) => `Timeline (${n})`,
    mobileSegWorkbench: "Workbench",
    timelineTitle: "Step Timeline & Token X-Ray",
    legendSys: "Sys",
    legendTools: "Tools",
    legendHist: "Hist",
    legendOut: "Output",
    stepPrefix: (n: number) => `Step ${n}`,
    vcrBadge: (ms: string) => `VCR 0-TOK · ${ms}ms`,
    driftBadge: "DRIFT",
    jumpXray: "Payload",
    jumpFork: "Fork →",
    promptTokLabel: "prompt tok",
    toolsTokShort: "tools",
    outTokShort: "out",
    tabBiopsy: "Biopsy Doctor",
    tabFork: (s: number) => `Time-Travel Fork (S${s})`,
    tabDiff: "Branch Diff",
    tabXray: (s: number) => `Payload (S${s})`,
    tabPytest: "Pytest",
    loadingTelemetry: "Loading trajectory telemetry...",
    loadingRawPayload: "Loading single-step raw payload...",
    verifiedRootCause: "Verified Root Cause",
    driftAtStep: (s: number) => `Drift @ Step ${s}`,
    confidence: (pct: string) => `${pct}% confidence`,
    ablationHeader: "Controlled Counterfactual Ablations (LangGraph)",
    provenFix: "PROVEN FIX",
    inspectDiff: "View Diff →",
    beforeLabel: "Before",
    afterLabel: "After",
    remediationHeader: "Engineering Remediation Playbook",
    noBiopsyTitle: "No Counterfactual Biopsy Report Yet",
    noBiopsyDesc:
      "Run the 5-node LangGraph Biopsy Doctor to scan anomalies, isolate the upstream drift step, and verify fixes via controlled forks.",
    launchBiopsy: "Launch Counterfactual Biopsy",
    forkTargetTitle: (s: number, total: number) => `Time-Travel Target: Step ${s} of ${total}`,
    forkTargetDescPrefix: (prevEnd: number, cur: number) =>
      `Steps 1..${prevEnd} fast-forward from local SQLite in <3ms with 0 tokens & 0 MCP side effects. Only Step ${cur} re-executes with your overrides.`,
    forkTargetDescStep1:
      "Step 1 executes directly with your counterfactual overrides (no prior prefix steps).",
    branchLabel: "Branch Label",
    swapModel: "Swap Model (Optional)",
    keepModel: (m: string) => `Keep Recorded (${m})`,
    interventionsHeader: "Context Engineering & Sandbox Controls",
    pruneTitle: "BM25 Semantic Tool Pruning",
    pruneDesc: "Keep Top-K relevant MCP tools while preserving schema order",
    compactTitle: "Structure-Aware Context Compactor",
    compactDesc: "Compact oversized historical JSON / log tool outputs",
    stabilizeTitle: "Stabilize Prefix KV Cache",
    stabilizeDesc: "Strip volatile header timestamps & sort tool schemas",
    sandboxTitle: "MCP Dry-Run Write Sandbox",
    sandboxDesc: "Intercept destructive calls (delete_*, wipe_*, send_*)",
    liveApiTitle: "Prefer Live Upstream LLM API",
    liveApiDesc: "Call live model when DEEPSEEK_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY is set",
    appendGuardLabel: "Append System Guardrail (Optional)",
    insertPreset: "+ Empty-Retrieval Guard Preset",
    appendPlaceholder: "e.g. Never invoke admin wipe tools; cap autonomous refunds at $500...",
    executeForkBtn: (s: number) => `Execute Time-Travel Fork @ Step ${s}`,
    executingFork: "Executing Counterfactual Fork...",
    compareSelectLabel: "Compare branch",
    diffPoint: "Divergence",
    diffSavedTok: "VCR Tokens Saved",
    diffSavedTime: "VCR Time Saved",
    baselineTag: "Baseline",
    forkedTag: "Forked Branch",
    noDiffHint: "No forked branch selected yet. Run Biopsy Doctor or Time-Travel Fork first.",
    cacheBreakTitle: (offset: number | null | undefined) =>
      `Prefix KV Cache Break Detected @ Char Offset #${offset ?? 0}`,
    fingerprintTitle: "Canonical Fingerprint (SHA-256)",
    rawReqTitle: "Raw LLM Request Payload (Secrets Auto-Redacted)",
    reasoningTitle: "Extended Thinking / Reasoning Trace",
    mcpSpansTitle: (s: number) => `Recorded MCP JSON-RPC Spans @ Step ${s}`,
    rawRespTitle: "Raw LLM Response Payload",
    pytestDesc: "Zero-API-cost CI regression test compiled from this trajectory",
    copyPytest: "Copy Code",
    copied: "Copied",
    emptyVaultHint: "Select a trajectory from the vault to begin time-travel inspection.",
    dismiss: "Dismiss",
  },
} as const;
