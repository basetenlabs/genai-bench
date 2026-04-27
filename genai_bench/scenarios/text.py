import re
from typing import List, Optional, Tuple

import numpy as np

from genai_bench.scenarios.base import (
    EmbeddingDistribution,
    ReRankDistribution,
    Scenario,
    TextDistribution,
    parse_params_str,
)


class NormalDistribution(Scenario):
    """
    Normal distribution
    e.g.
    N(300,150)/(480,240)
    """

    scenario_type = TextDistribution.NORMAL
    validation_pattern = r"^N\(\d+,\d+\)\/\(\d+,\d+\)$"

    def __init__(
        self,
        mean_input_tokens: int,
        stddev_input_tokens: int,
        mean_output_tokens: int,
        stddev_output_tokens: int,
    ):
        self.mean_input_tokens = mean_input_tokens
        self.stddev_input_tokens = stddev_input_tokens
        self.mean_output_tokens = mean_output_tokens
        self.stddev_output_tokens = stddev_output_tokens

    def sample(self) -> Tuple[int, int]:
        num_input_tokens = max(
            1,
            int(np.random.normal(self.mean_input_tokens, self.stddev_input_tokens)),
        )
        num_output_tokens = max(
            2,
            int(np.random.normal(self.mean_output_tokens, self.stddev_output_tokens)),
        )
        return num_input_tokens, num_output_tokens

    def to_string(self) -> str:
        return (
            f"N({self.mean_input_tokens},{self.stddev_input_tokens})/"
            f"({self.mean_output_tokens},{self.stddev_output_tokens})"
        )

    @classmethod
    def parse(cls, params_str: str) -> "NormalDistribution":
        (mean_input, stddev_input), (mean_output, stddev_output) = parse_params_str(
            params_str
        )
        return cls(
            mean_input_tokens=mean_input,
            stddev_input_tokens=stddev_input,
            mean_output_tokens=mean_output,
            stddev_output_tokens=stddev_output,
        )


class UniformDistribution(Scenario):
    """
    Uniform distribution
    e.g.
    U(100,200)/(200,300),
    U(1000,100)
    """

    scenario_type = TextDistribution.UNIFORM
    validation_pattern = r"^U\(\d+,\d+\)(?:\/\(\d+,\d+\))?$"

    def __init__(
        self,
        max_input_tokens: int,
        max_output_tokens: int,
        min_input_tokens: Optional[int] = None,
        min_output_tokens: Optional[int] = None,
    ):
        self.min_input_tokens = min_input_tokens
        self.max_input_tokens = max_input_tokens
        self.min_output_tokens = min_output_tokens
        self.max_output_tokens = max_output_tokens

    def sample(self) -> Tuple[int, int]:
        num_input_tokens = max(
            1,
            int(np.random.uniform(self.min_input_tokens or 1, self.max_input_tokens)),
        )
        num_output_tokens = max(
            2,
            int(np.random.uniform(self.min_output_tokens or 1, self.max_output_tokens)),
        )
        return num_input_tokens, num_output_tokens

    def to_string(self) -> str:
        if self.min_input_tokens and self.min_output_tokens:
            return (
                f"U({self.min_input_tokens},{self.max_input_tokens})/"
                f"({self.min_output_tokens},{self.max_output_tokens})"
            )
        else:
            return f"U({self.max_input_tokens},{self.max_output_tokens})"

    @classmethod
    def parse(cls, params_str: str) -> "UniformDistribution":
        parsed_parts = parse_params_str(params_str)
        if len(parsed_parts) == 2:
            (min_input, max_input), (min_output, max_output) = parsed_parts
            return cls(
                min_input_tokens=min_input,
                max_input_tokens=max_input,
                min_output_tokens=min_output,
                max_output_tokens=max_output,
            )
        else:
            (
                max_input,
                max_output,
            ) = parsed_parts[0]
            return cls(
                max_input_tokens=max_input,
                max_output_tokens=max_output,
            )


class DeterministicDistribution(Scenario):
    """
    Deterministic Distribution, aka constant
    e.g.
    D(100,1000)
    """

    scenario_type = TextDistribution.DETERMINISTIC
    validation_pattern = r"^D\(\d+,\d+\)$"

    def __init__(self, num_input_tokens: int, num_output_tokens: int):
        self.num_input_tokens = num_input_tokens
        self.num_output_tokens = num_output_tokens

    def sample(self) -> Tuple[int, int]:
        return self.num_input_tokens, self.num_output_tokens

    def to_string(self) -> str:
        return f"D({self.num_input_tokens},{self.num_output_tokens})"

    @classmethod
    def parse(cls, params_str: str) -> "DeterministicDistribution":
        num_input, num_output = parse_params_str(params_str)[0]
        return cls(
            num_input_tokens=num_input,
            num_output_tokens=num_output,
        )


class EmbeddingScenario(Scenario):
    """
    A class to represent an embedding scenario
    e.g. E(tokens_per_document)
    """

    scenario_type = EmbeddingDistribution.EMBEDDING
    validation_pattern = r"^E\(\d+\)$"

    def __init__(self, tokens_per_document: int):
        self.tokens_per_document = tokens_per_document

    def sample(self) -> int:
        """Returns tokens per document"""
        return self.tokens_per_document

    def to_string(self) -> str:
        """
        Returns the embedding scenario object back in its string representation.
        For example E(1024).
        """
        return f"E({self.tokens_per_document})"

    @classmethod
    def parse(cls, params_str: str) -> "EmbeddingScenario":
        """
        Parse the embedding scenario from a string, e.g. E(1024)
        """
        tokens_per_document = int(params_str[1:-1])
        return cls(tokens_per_document=tokens_per_document)


class ReRankScenario(Scenario):
    """
    A class to represent re-rank scenario
    e.g. R(tokens_per_document,tokens_per_query)
    """

    scenario_type = ReRankDistribution.RE_RANK
    validation_pattern = r"^R\(\d+,\d+\)$"

    def __init__(self, tokens_per_document: int, tokens_per_query: int):
        self.tokens_per_document = tokens_per_document
        self.tokens_per_query = tokens_per_query

    def sample(self) -> Tuple[int, int]:
        """Returns tokens per document"""
        return self.tokens_per_document, self.tokens_per_query

    def to_string(self) -> str:
        """
        Returns the re-rank scenario object back in its string representation.
        For example R(1024,100).
        """
        return f"R({self.tokens_per_document},{self.tokens_per_query})"

    @classmethod
    def parse(cls, params_str: str) -> "ReRankScenario":
        """
        Parse the re-rank scenario from a string, e.g. R(1024,100)
        """
        tokens_per_document, tokens_per_query = parse_params_str(params_str)[0]
        return cls(
            tokens_per_document=tokens_per_document, tokens_per_query=tokens_per_query
        )


class PrefixRepetitionScenario(Scenario):
    """
    Prefix repetition scenario for KV cache benchmarking.

    All concurrent requests share the same prefix but have unique suffixes.
    This enables benchmarking of KV cache performance, chunked prefill efficiency,
    and automatic prefix caching (APC) features in LLM serving engines.

    Format: P(prefix_len,suffix_len)/output_len
    Example: P(2000,500)/200

    In this example:
    - All requests share a 2000-token prefix (cached after first request)
    - Each request has a unique 500-token suffix
    - Expected output is 200 tokens

    This scenario is particularly useful for:
    - Testing KV cache hit rates and speedups
    - Benchmarking prefill performance with cached prefixes
    - Measuring Time To First Token (TTFT) improvements
    - Evaluating automatic prefix caching implementations
    """

    scenario_type = TextDistribution.PREFIX_REPETITION
    validation_pattern = r"^P\(\d+,\d+\)/\d+$"

    def __init__(self, prefix_len: int, suffix_len: int, output_len: int):
        self.prefix_len = prefix_len
        self.suffix_len = suffix_len
        self.output_len = output_len

    def sample(self) -> Tuple[int, int, int]:
        """Returns (prefix_len, suffix_len, output_len)"""
        return self.prefix_len, self.suffix_len, self.output_len

    def to_string(self) -> str:
        """
        Returns the prefix repetition scenario back in its string representation.
        For example P(2000,500)/200.
        """
        return f"P({self.prefix_len},{self.suffix_len})/{self.output_len}"

    @classmethod
    def parse(cls, params_str: str) -> "PrefixRepetitionScenario":
        """
        Parse the prefix repetition scenario from a string.

        Example: "(2000,500)/200" -> PrefixRepetitionScenario(2000, 500, 200)
        """
        # Parse P(prefix_len,suffix_len)/output_len
        # params_str will be "(2000,500)/200"
        match = re.match(r"\((\d+),(\d+)\)/(\d+)", params_str)
        if not match:
            raise ValueError(
                f"Invalid prefix repetition format: {params_str}. "
                f"Expected format: (prefix_len,suffix_len)/output_len"
            )
        prefix_len = int(match.group(1))
        suffix_len = int(match.group(2))
        output_len = int(match.group(3))
        return cls(prefix_len, suffix_len, output_len)


def _split_top_level_commas(body: str) -> List[str]:
    """Split body on commas that are at paren depth 0."""
    parts = []
    depth = 0
    start = 0
    for i, ch in enumerate(body):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                raise ValueError(f"Unbalanced parentheses in: {body}")
        elif ch == "," and depth == 0:
            parts.append(body[start:i].strip())
            start = i + 1
    if depth != 0:
        raise ValueError(f"Unbalanced parentheses in: {body}")
    parts.append(body[start:].strip())
    return [p for p in parts if p]


class MixedScenario(Scenario):
    """
    A weighted mixture of sub-scenarios that dispatches per request.

    Format: M(w1:SUBSCEN,w2:SUBSCEN,...)
    Example: M(0.4:P(7840,160)/200,0.4:P(31360,640)/500,0.2:P(78400,1600)/1000)

    Each sub-scenario is any other scenario string (P/D/N/U). Weights are
    normalized to sum to 1. On each sample() call, a sub-scenario is drawn by
    weighted random choice; the returned Scenario is dispatched by the sampler
    to its normal generation path.
    """

    scenario_type = TextDistribution.MIXED
    validation_pattern = r"^M\(.+\)$"

    def __init__(
        self, weights: List[float], sub_scenarios: List[Scenario]
    ):
        if len(weights) != len(sub_scenarios) or not sub_scenarios:
            raise ValueError(
                "MixedScenario requires matching non-empty weights and "
                "sub_scenarios"
            )
        total = float(sum(weights))
        if total <= 0:
            raise ValueError("MixedScenario weights must sum to a positive number")
        self.weights = [w / total for w in weights]
        self.sub_scenarios = sub_scenarios
        self._rng = np.random.default_rng()

    def sample(self) -> Scenario:
        """Return one of the sub-scenarios by weighted random choice.

        The return type is a Scenario (not a tuple), since MixedScenario is a
        dispatcher: the sampler inspects the returned Scenario and delegates
        to the appropriate generation path (e.g. prefix-repetition).
        """
        idx = int(self._rng.choice(len(self.sub_scenarios), p=self.weights))
        return self.sub_scenarios[idx]

    def to_string(self) -> str:
        parts = [
            f"{w}:{s.to_string()}" for w, s in zip(self.weights, self.sub_scenarios)
        ]
        return f"M({','.join(parts)})"

    @classmethod
    def parse(cls, params_str: str) -> "MixedScenario":
        """
        Parse a Mixed scenario string.

        Example: "(0.4:P(7840,160)/200,0.4:P(31360,640)/500,0.2:P(78400,1600)/1000)"
        """
        if not (params_str.startswith("(") and params_str.endswith(")")):
            raise ValueError(
                f"Invalid mixed format: {params_str}. Expected M(w:SUB,w:SUB,...)"
            )
        body = params_str[1:-1]
        entries = _split_top_level_commas(body)
        if not entries:
            raise ValueError(f"MixedScenario requires at least one sub-scenario: {params_str}")

        weights: List[float] = []
        subs: List[Scenario] = []
        for entry in entries:
            if ":" not in entry:
                raise ValueError(
                    f"Mixed sub-scenario '{entry}' must be WEIGHT:SUBSCEN"
                )
            weight_str, sub_str = entry.split(":", 1)
            weight_str = weight_str.strip()
            sub_str = sub_str.strip()
            try:
                weight = float(weight_str)
            except ValueError as e:
                raise ValueError(
                    f"Invalid weight '{weight_str}' in mixed scenario"
                ) from e
            if weight <= 0:
                raise ValueError(
                    f"Mixed weight must be positive, got {weight} for '{entry}'"
                )
            sub = Scenario.from_string(sub_str)
            if isinstance(sub, MixedScenario):
                raise ValueError("MixedScenario cannot be nested")
            weights.append(weight)
            subs.append(sub)
        return cls(weights, subs)
