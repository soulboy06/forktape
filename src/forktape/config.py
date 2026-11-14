"""Central configuration and model pricing estimator for ForkTape."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Approximate per-1M token pricing (input_usd, cached_input_usd, output_usd)
MODEL_PRICING_PER_1M: dict[str, tuple[float, float, float]] = {
    "gpt-4o": (2.50, 1.25, 10.00),
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "gpt-4.1": (2.00, 0.50, 8.00),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "o3-mini": (1.10, 0.55, 4.40),
    "claude-3-7-sonnet": (3.00, 0.30, 15.00),
    "claude-sonnet-4": (3.00, 0.30, 15.00),
    "claude-3-5-haiku": (0.80, 0.08, 4.00),
    "deepseek-flash": (0.15, 0.03, 0.60),
    "deepseek-v4-pro": (0.55, 0.14, 2.19),
    "deepseek-chat": (0.27, 0.07, 1.10),
    "deepseek-reasoner": (0.55, 0.14, 2.19),
    "qwen-max": (1.60, 0.32, 6.40),
    "qwen-plus": (0.40, 0.08, 1.20),
}

DEFAULT_PRICING_PER_1M: tuple[float, float, float] = (2.00, 0.40, 8.00)


def estimate_cost_usd(
    model: str,
    prompt_tokens: int,
    cached_prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> float:
    """Estimate USD cost of a single LLM call accounting for Prefix KV Cache discounts."""
    model_lower = (model or "").lower()
    pricing = DEFAULT_PRICING_PER_1M
    for prefix, rates in MODEL_PRICING_PER_1M.items():
        if prefix in model_lower:
            pricing = rates
            break

    in_rate, cached_rate, out_rate = pricing
    uncached_prompt = max(0, prompt_tokens - cached_prompt_tokens)
    cached_prompt = max(0, min(prompt_tokens, cached_prompt_tokens))
    cost = (
        (uncached_prompt / 1_000_000.0) * in_rate
        + (cached_prompt / 1_000_000.0) * cached_rate
        + (completion_tokens / 1_000_000.0) * out_rate
    )
    return round(cost, 6)


def _default_openai_upstream() -> str:
    explicit = os.environ.get("FORKTAPE_OPENAI_UPSTREAM") or os.environ.get("OPENAI_UPSTREAM_URL")
    if explicit:
        return explicit.rstrip("/")
    if os.environ.get("DEEPSEEK_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        return "https://api.deepseek.com/v1"
    return "https://api.openai.com/v1"


@dataclass(slots=True)
class ForkTapeConfig:
    """Runtime settings loaded from environment variables or CLI flags."""

    db_path: Path = field(
        default_factory=lambda: Path(os.environ.get("FORKTAPE_DB_PATH", ".forktape/tapes.db"))
    )
    host: str = field(default_factory=lambda: os.environ.get("FORKTAPE_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(os.environ.get("FORKTAPE_PORT", "8787")))
    openai_upstream: str = field(default_factory=_default_openai_upstream)
    anthropic_upstream: str = field(
        default_factory=lambda: os.environ.get(
            "FORKTAPE_ANTHROPIC_UPSTREAM",
            os.environ.get("ANTHROPIC_UPSTREAM_URL", "https://api.anthropic.com"),
        ).rstrip("/")
    )
    similarity_threshold: float = field(
        default_factory=lambda: float(os.environ.get("FORKTAPE_SIMILARITY_THRESHOLD", "0.92"))
    )

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"
