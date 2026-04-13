"""Sampler for real-dataset benchmarking with pre-built conversation data."""

import copy
import uuid
from typing import Any, Dict, List, Optional

from genai_bench.data.config import DatasetConfig
from genai_bench.logging import init_logger
from genai_bench.protocol import UserConversationRequest
from genai_bench.scenarios.base import Scenario
from genai_bench.scenarios.real_dataset import (
    CachedRealDatasetScenario,
    RealDatasetScenario,
)

logger = init_logger(__name__)


class ConversationSampler:
    """Sampler for pre-built conversation datasets (RD/RDC scenarios).

    Uses real dataset messages as-is. Supports two cache modes:
    - RD (uncached): Prepends unique nonce to first user message, busting prefix cache.
    - RDC (natural): No modification. Non-repeating sampling simulates production.

    The global cursor persists across concurrency levels, ensuring each level
    gets different samples (non-repeating until dataset is exhausted).
    """

    def __init__(
        self,
        tokenizer: Any,
        model: str,
        data: List[Dict],
        additional_request_params: Optional[Dict[str, Any]] = None,
        dataset_config: Optional[DatasetConfig] = None,
        no_min_tokens: bool = False,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.model = model
        self.data = data  # Pre-processed conversation samples from loader
        self.additional_request_params = additional_request_params or {}
        self.dataset_config = dataset_config
        self.no_min_tokens = no_min_tokens

        # Global cursor for non-repeating sampling
        self._cursor: int = 0
        self._wrap_count: int = 0

        logger.info(
            f"ConversationSampler initialized with {len(self.data)} samples."
        )

    def sample(self, scenario: Optional[Scenario]) -> UserConversationRequest:
        """Return next conversation sample as a request."""
        if not isinstance(scenario, (RealDatasetScenario, CachedRealDatasetScenario)):
            raise ValueError(
                f"ConversationSampler requires RealDatasetScenario or "
                f"CachedRealDatasetScenario, got {type(scenario)}"
            )

        max_output_tokens, cache_mode = scenario.sample()

        # Get next sample (non-repeating with wrap-around)
        sample = self._next_sample()

        # Deep copy messages to avoid mutation across requests
        messages = copy.deepcopy(sample["messages"])

        # Apply cache mode
        if cache_mode == "uncached":
            self._inject_nonce(messages)

        # Build additional_request_params for this request
        params = dict(self.additional_request_params)
        if max_output_tokens is not None:
            params["ignore_eos"] = True
            params["max_tokens"] = max_output_tokens
            if not self.no_min_tokens:
                params["min_tokens"] = max_output_tokens

        return UserConversationRequest(
            model=self.model,
            messages=messages,
            num_prefill_tokens=None,  # Can't easily count for multi-turn + images
            max_tokens=max_output_tokens,
            num_images=sample.get("num_images", 0),
            additional_request_params=params,
        )

    def _next_sample(self) -> Dict:
        """Return next sample using global cursor. Wraps with warning."""
        if self._cursor >= len(self.data):
            self._wrap_count += 1
            if self._wrap_count == 1:
                logger.warning(
                    f"Dataset exhausted ({len(self.data)} samples). "
                    f"Wrapping around — some samples will repeat."
                )
            self._cursor = 0
        sample = self.data[self._cursor]
        self._cursor += 1
        return sample

    def _inject_nonce(self, messages: List[Dict]) -> None:
        """Prepend unique nonce to first user message to bust prefix cache.

        Uses a random UUID to ensure nonces are unique across sequential
        benchmark runs on the same pod (an incrementing counter would produce
        identical nonces across runs, allowing prefix cache hits).
        """
        nonce = f"[NONCE-{uuid.uuid4().hex[:12]}] "

        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    msg["content"] = nonce + content
                elif isinstance(content, list):
                    # Find first text block and prepend
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            part["text"] = nonce + part["text"]
                            break
                break  # Only inject into first user message
