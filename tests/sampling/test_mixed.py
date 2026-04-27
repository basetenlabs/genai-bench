"""Tests for mixed scenario dispatch and nested mega-prefix sharing in the sampler."""

import unittest
from unittest.mock import MagicMock

from genai_bench.sampling.text import TextSampler
from genai_bench.scenarios.text import (
    MixedScenario,
    PrefixRepetitionScenario,
)


class _FakeTokenizer:
    """Minimal tokenizer: each whitespace-separated token -> one id.

    Decode just joins ids back with spaces. Deterministic so tests can verify
    slice-based prefix sharing without numeric drift.
    """

    def __init__(self):
        self._next_id = 100_000
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}

    def _id_for(self, token: str) -> int:
        if token not in self._token_to_id:
            self._token_to_id[token] = self._next_id
            self._id_to_token[self._next_id] = token
            self._next_id += 1
        return self._token_to_id[token]

    def encode(self, text, add_special_tokens=False):  # noqa: D401
        return [self._id_for(t) for t in text.split(" ") if t]

    def decode(self, ids, skip_special_tokens=True):  # noqa: D401
        return " ".join(self._id_to_token[i] for i in ids)


class TestMixedSamplerNestedPrefix(unittest.TestCase):
    def setUp(self):
        self.tokenizer = _FakeTokenizer()
        # Each "line" is a wide unique-token chunk so _sample_text produces
        # a long deterministic stream.
        self.data = [" ".join(f"w{i:05d}" for i in range(k * 200, (k + 1) * 200)) for k in range(100)]
        self.sampler = TextSampler(
            tokenizer=self.tokenizer,
            model="mock_model",
            output_modality="text",
            data=self.data,
        )

    def test_ensure_mega_prefix_generates_once_sized_to_max(self):
        scen = MixedScenario(
            weights=[0.5, 0.5],
            sub_scenarios=[
                PrefixRepetitionScenario(50, 10, 5),
                PrefixRepetitionScenario(200, 10, 5),
            ],
        )
        self.sampler._ensure_mega_prefix_for_mixed(scen)
        assert self.sampler._mega_prefix_tokens is not None
        assert len(self.sampler._mega_prefix_tokens) >= 200
        assert self.sampler._mega_prefix_pinned is True

    def test_mega_prefix_slices_produce_nested_prefixes(self):
        short = PrefixRepetitionScenario(50, 10, 5)
        medium = PrefixRepetitionScenario(120, 10, 5)
        long_ = PrefixRepetitionScenario(300, 10, 5)
        scen = MixedScenario(
            weights=[0.4, 0.4, 0.2],
            sub_scenarios=[short, medium, long_],
        )

        # Trigger a dispatch via sample() which internally ensures mega-prefix.
        self.sampler._ensure_mega_prefix_for_mixed(scen)

        short_req = self.sampler._sample_prefix_repetition_request(short)
        medium_req = self.sampler._sample_prefix_repetition_request(medium)
        long_req = self.sampler._sample_prefix_repetition_request(long_)

        short_prefix = short_req.prompt.split("\n\n--- Request #")[0]
        medium_prefix = medium_req.prompt.split("\n\n--- Request #")[0]
        long_prefix = long_req.prompt.split("\n\n--- Request #")[0]

        # Nested: shorter prefix is a prefix of longer prefix.
        assert medium_prefix.startswith(short_prefix)
        assert long_prefix.startswith(medium_prefix)
        assert long_prefix.startswith(short_prefix)

    def test_reset_prefix_cache_noop_when_pinned(self):
        scen = MixedScenario(
            weights=[1.0],
            sub_scenarios=[PrefixRepetitionScenario(50, 10, 5)],
        )
        self.sampler._ensure_mega_prefix_for_mixed(scen)
        before_tokens = self.sampler._mega_prefix_tokens
        self.sampler.reset_prefix_cache()
        # Pinned state preserved; mega-prefix untouched
        assert self.sampler._mega_prefix_pinned is True
        assert self.sampler._mega_prefix_tokens is before_tokens

    def test_sample_dispatches_mixed_via_sample_method(self):
        short = PrefixRepetitionScenario(50, 10, 5)
        scen = MixedScenario(weights=[1.0], sub_scenarios=[short])
        req = self.sampler.sample(scen)
        # Request generated through mixed path should have a prompt with the
        # '--- Request #' marker emitted by prefix-repetition sampling.
        assert "Request #" in req.prompt

    def test_additional_request_params_are_per_request(self):
        """Regression: each request gets its own params dict so concurrent
        requests with different output_len don't clobber each other."""
        short = PrefixRepetitionScenario(50, 10, 7)
        long_ = PrefixRepetitionScenario(300, 10, 42)
        scen = MixedScenario(weights=[0.5, 0.5], sub_scenarios=[short, long_])
        self.sampler._ensure_mega_prefix_for_mixed(scen)

        short_req = self.sampler._sample_prefix_repetition_request(short)
        long_req = self.sampler._sample_prefix_repetition_request(long_)
        assert short_req.additional_request_params["max_tokens"] == 7
        assert long_req.additional_request_params["max_tokens"] == 42
        # After generating long_req, short_req's params dict must not be
        # mutated.
        assert short_req.additional_request_params["max_tokens"] == 7
        assert short_req.additional_request_params is not long_req.additional_request_params


if __name__ == "__main__":
    unittest.main()
