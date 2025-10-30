import time
from unittest.mock import MagicMock

from genai_bench.user.openai_user import OpenAIUser


class _MockResponse:
    def __init__(self, chunks):
        self._chunks = chunks

    def iter_lines(self, chunk_size=None):  # noqa: ARG002 - signature compatibility
        for c in self._chunks:
            if callable(c):
                yield c()
            else:
                yield c

    def close(self):
        pass


def test_ttft_captured_on_first_chunk(monkeypatch):
    """Test that TTFT is captured on the first valid streamed chunk with choices."""
    OpenAIUser.host = "http://example.com"
    user = OpenAIUser(environment=MagicMock())
    user.disable_streaming = False

    start = time.monotonic()

    # First chunk: empty choices (should be ignored - no TTFT)
    empty_choices = b'data: {"id":"chat-1","object":"chat.completion.chunk","choices":[]}'
    
    # Second chunk: valid choices with role (should set TTFT here)
    first_valid = (
        b'data: {"id":"chat-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"},"logprobs":null,"finish_reason":null}]}'
    )

    # Delay to distinguish first-valid-chunk vs first-content timing
    def delayed_content_chunk():
        time.sleep(0.1)
        return (
            b'data: {"id":"chat-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"H"},"logprobs":null,"finish_reason":null}]}'
        )

    # Finish chunk includes finish_reason inside choices[0]
    finish = (
        b'data: {"id":"chat-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":null},"logprobs":null,"finish_reason":"length"}]}'
    )
    # Usage-only final chunk
    usage = (
        b'data: {"id":"chat-1","object":"chat.completion.chunk","choices":[],"usage":{"prompt_tokens":5,"total_tokens":7,"completion_tokens":2}}'
    )

    done = b"data: [DONE]"

    chunks = [empty_choices, first_valid, delayed_content_chunk, finish, usage, done]

    resp = _MockResponse(chunks)

    parsed = user.parse_chat_response(
        response=resp,
        start_time=start,
        num_prefill_tokens=5,
        _=start,  # placeholder per signature
    )

    assert parsed.status_code == 200
    assert parsed.tokens_received == 2
    assert parsed.generated_text == "H"

    # TTFT should be captured at first valid chunk (before 100ms sleep)
    assert parsed.time_at_first_token is not None
    ttft_s = parsed.time_at_first_token - start
    assert ttft_s < 0.08, f"TTFT too large; expected first-valid-chunk capture, got {ttft_s:.4f}s"
    
    # Verify TTFT was captured on the second chunk (first_valid), not the first (empty_choices)
    # This ensures we're testing the correct behavior: first chunk WITH non-empty choices


def test_ttft_not_captured_on_empty_choices():
    """Test that TTFT is NOT captured on chunks with empty choices array."""
    OpenAIUser.host = "http://example.com"
    user = OpenAIUser(environment=MagicMock())
    user.disable_streaming = False

    start = time.monotonic()

    # Only empty choices chunks - should NOT set TTFT
    empty_choices1 = b'data: {"id":"chat-1","object":"chat.completion.chunk","choices":[]}'
    empty_choices2 = b'data: {"id":"chat-1","object":"chat.completion.chunk","choices":[]}'
    
    # Valid choices with content (should set TTFT here)
    first_valid = (
        b'data: {"id":"chat-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"H"},"logprobs":null,"finish_reason":null}]}'
    )

    # Finish chunk
    finish = (
        b'data: {"id":"chat-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":null},"logprobs":null,"finish_reason":"length"}]}'
    )
    usage = (
        b'data: {"id":"chat-1","object":"chat.completion.chunk","choices":[],"usage":{"prompt_tokens":5,"total_tokens":7,"completion_tokens":2}}'
    )
    done = b"data: [DONE]"

    chunks = [empty_choices1, empty_choices2, first_valid, finish, usage, done]
    resp = _MockResponse(chunks)

    parsed = user.parse_chat_response(
        response=resp,
        start_time=start,
        num_prefill_tokens=5,
        _=start,
    )

    assert parsed.status_code == 200
    assert parsed.tokens_received == 2
    assert parsed.generated_text == "H"

    # TTFT should be captured only on first_valid chunk, not on empty_choices
    assert parsed.time_at_first_token is not None
    ttft_s = parsed.time_at_first_token - start
    # Should be small since first_valid comes after empty_choices
    assert ttft_s < 0.05, f"TTFT should be small since it was captured on first valid chunk, got {ttft_s:.4f}s"


