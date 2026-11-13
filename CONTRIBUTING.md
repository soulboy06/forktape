# Contributing to ForkTape

Thank you for your interest in contributing to **ForkTape** (`ftk`)! Our goal is to make deterministic time-travel debugging, dual-protocol flight recording (`LLM SSE` + `MCP JSON-RPC`), and counterfactual root-cause biopsy accessible to every AI Agent engineer.

## Development Setup

### Prerequisites
- **Python 3.11+** and [`uv`](https://docs.astral.sh/uv/)
- **Node.js 20+** and `npm` (only required if modifying the React Web Studio in `web/`)

### 1. Clone & Install Python Environment

```bash
git clone https://github.com/soulboy06/forktape.git
cd forktape
uv sync --extra dev
```

### 2. Run Linters, Tests, and ForkBench-50

```bash
# Lint codebase
uv run ruff check src tests benchmarks examples

# Run unit & integration suite
uv run pytest -v

# Run the 50-trajectory counterfactual biopsy benchmark
uv run python benchmarks/run_forkbench.py
```

### 3. Rebuild Web Studio (Optional)

The pre-compiled Studio SPA lives in `src/forktape/static_ui/` so `pip install forktape` / `uvx forktape demo` works out-of-the-box without requiring Node.js at runtime. If you edit files under `web/`:

```bash
cd web
npm ci
npm run build
```

Vite outputs the production bundle directly into `../src/forktape/static_ui/`.

## High-Impact Areas for Contribution

1. **New Failure Heuristics & Counterfactual Operators** (`src/forktape/doctor/heuristics.py` & `src/forktape/engine/patcher.py`):
   - Add detectors for multi-agent handoff loops, JSON schema truncation, or tool permission escalation.
2. **Additional Provider Wire Formats** (`src/forktape/interceptor/sse_assembler.py`):
   - Expand beyond OpenAI Chat Completions and Anthropic Messages to Google Gemini Live/Interactions SSE, AWS Bedrock ConverseStream, or OpenAI Responses API.
3. **ForkBench Trajectory Contributions** (`benchmarks/run_forkbench.py`):
   - Contribute anonymized failure `.ftk.json` bundles from real-world agent outages to expand `ForkBench-50` toward `ForkBench-100`.
4. **IDE & CI Integrations** (`src/forktape/mcp_server/` & `src/forktape/compiler/`):
   - GitHub Actions PR comment bots for `.ftk.json` regression diffs.

## Pull Request Checklist

- [ ] `uv run ruff check src tests benchmarks examples` passes with zero warnings.
- [ ] `uv run pytest -v` passes and includes tests covering your change.
- [ ] Secrets/API keys are never committed (use `redact_secrets` fixtures for test payloads).
- [ ] If `web/` changed, `src/forktape/static_ui/` has been rebuilt cleanly via `npm run build`.

## License

By contributing to ForkTape, you agree that your contributions will be licensed under the [MIT License](LICENSE).
