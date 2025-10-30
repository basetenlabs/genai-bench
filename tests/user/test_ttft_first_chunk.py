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
    OpenAIUser.host = "http://example.com"
    user = OpenAIUser(environment=MagicMock())
    user.disable_streaming = False

    start = time.monotonic()

    # First chunk has choices but no content -> should set TTFT here
    first = (
        b'data: {"id":"chat-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"},"logprobs":null,"finish_reason":null}]}'
    )

    # Delay to distinguish first-chunk vs first-content timing
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

    chunks = [first, delayed_content_chunk, finish, usage, done]

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

    # TTFT should be captured at first chunk (before 100ms sleep)
    assert parsed.time_at_first_token is not None
    ttft_s = parsed.time_at_first_token - start
    assert ttft_s < 0.08, f"TTFT too large; expected first-chunk capture, got {ttft_s:.4f}s"


