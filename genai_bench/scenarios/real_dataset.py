"""Real dataset scenarios for benchmarking with actual conversation data."""

import re
from typing import Optional, Tuple

from genai_bench.scenarios.base import Scenario, SpecialScenario


class RealDatasetScenario(Scenario):
    """
    Real dataset scenario — uncached mode.
    Prepends a unique nonce to the first user message to bust ALL prefix caching.

    Format: RD or RD(output_tokens)
    Examples:
        RD          — natural generation (no forced output)
        RD(200)     — forced 200 output tokens
    """

    scenario_type = SpecialScenario.REAL_DATASET
    validation_pattern = r"^RD(?:\(\d+\))?$"

    def __init__(self, max_output_tokens: Optional[int] = None):
        self.max_output_tokens = max_output_tokens

    def sample(self) -> Tuple[Optional[int], str]:
        """Returns (max_output_tokens, cache_mode)."""
        return self.max_output_tokens, "uncached"

    def to_string(self) -> str:
        if self.max_output_tokens is not None:
            return f"RD({self.max_output_tokens})"
        return "RD"

    @classmethod
    def parse(cls, params_str: str) -> "RealDatasetScenario":
        if not params_str:
            return cls(max_output_tokens=None)
        match = re.match(r"^\((\d+)\)$", params_str)
        if not match:
            raise ValueError(
                f"Invalid RD format: '{params_str}'. Expected '' or '(N)'."
            )
        return cls(max_output_tokens=int(match.group(1)))


class CachedRealDatasetScenario(Scenario):
    """
    Real dataset scenario — cached/natural mode.
    No nonce injection. Non-repeating samples across concurrency levels.
    Simulates production with KV-cache-aware routing on a single replica.

    Format: RDC or RDC(output_tokens)
    Examples:
        RDC         — natural generation (no forced output)
        RDC(200)    — forced 200 output tokens
    """

    scenario_type = SpecialScenario.REAL_DATASET_CACHED
    validation_pattern = r"^RDC(?:\(\d+\))?$"

    def __init__(self, max_output_tokens: Optional[int] = None):
        self.max_output_tokens = max_output_tokens

    def sample(self) -> Tuple[Optional[int], str]:
        """Returns (max_output_tokens, cache_mode)."""
        return self.max_output_tokens, "natural"

    def to_string(self) -> str:
        if self.max_output_tokens is not None:
            return f"RDC({self.max_output_tokens})"
        return "RDC"

    @classmethod
    def parse(cls, params_str: str) -> "CachedRealDatasetScenario":
        if not params_str:
            return cls(max_output_tokens=None)
        match = re.match(r"^\((\d+)\)$", params_str)
        if not match:
            raise ValueError(
                f"Invalid RDC format: '{params_str}'. Expected '' or '(N)'."
            )
        return cls(max_output_tokens=int(match.group(1)))
