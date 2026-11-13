"""Core replay, fingerprinting, and trajectory diffing engines."""

from forktape.engine.fingerprint import (
    PrefixCacheAnalysis,
    analyze_prefix_cache_continuity,
    canonical_request_hash,
    compute_structural_similarity,
    estimate_step_token_breakdown,
    estimate_tokens,
    normalize_request_for_fingerprint,
    redact_payload,
    redact_secrets_in_text,
)

__all__ = [
    "PrefixCacheAnalysis",
    "analyze_prefix_cache_continuity",
    "canonical_request_hash",
    "compute_structural_similarity",
    "estimate_step_token_breakdown",
    "estimate_tokens",
    "normalize_request_for_fingerprint",
    "redact_payload",
    "redact_secrets_in_text",
]
