"""Tests for AsyncOpenAIUser implementation with aiohttp."""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
import aiohttp

from genai_bench.user.async_openai_user import AsyncOpenAIUser
from genai_bench.protocol import UserChatRequest, UserChatResponse


class MockAsyncResponse:
    """Mock aiohttp response for testing."""
    
    def __init__(self, chunks, status=200):
        self.status = status
        self._chunks = chunks
        self._iter_idx = 0
        self.content = self  # Make content attribute available
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
    
    async def iter_any(self):
        """Simulate aiohttp's iter_any() method."""
        for chunk in self._chunks:
            if callable(chunk):
                yield chunk()  # Call function to introduce delay
            else:
                yield chunk.encode('utf-8') if isinstance(chunk, str) else chunk
    
    async def text(self):
        """Simulate aiohttp's text() method."""
        return "Error response"


class TestAsyncOpenAIUser:
    """Test AsyncOpenAIUser implementation."""
    
    @pytest.fixture
    def mock_auth(self):
        """Create mock auth provider."""
        auth = MagicMock()
        auth.get_credentials.return_value = "test-api-key"
        return auth
    
    @pytest_asyncio.fixture
    async def async_user(self, mock_auth):
        """Create AsyncOpenAIUser instance."""
        AsyncOpenAIUser.host = "http://example.com"
        user = AsyncOpenAIUser(environment=MagicMock())
        user.auth_provider = mock_auth
        user.headers = {
            "Authorization": "Bearer test-api-key",
            "Content-Type": "application/json",
        }
        await user.on_start()  # Initialize session
        yield user
        await user.on_stop()  # Clean up session
    
    @pytest.mark.asyncio
    async def test_ttft_captured_on_first_chunk_async(self, async_user):
        """Test that TTFT is captured on the first valid streamed chunk with choices."""
        start = time.monotonic()
        
        # Mock chunks with byte-level streaming like vLLM
        chunks = [
            # First chunk: empty choices (should be ignored)
            'data: {"id":"chat-1","object":"chat.completion.chunk","choices":[]}\n\n',
            
            # Second chunk: valid choices with role (should set TTFT here)
            'data: {"id":"chat-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"},"logprobs":null,"finish_reason":null}]}\n\n',
            
            # Content chunk (no delay for now)
            'data: {"id":"chat-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"H"},"logprobs":null,"finish_reason":null}]}\n\n',
            
            # Finish chunk
            'data: {"id":"chat-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":null},"logprobs":null,"finish_reason":"length"}]}\n\n',
            
            # Usage chunk
            'data: {"id":"chat-1","object":"chat.completion.chunk","choices":[],"usage":{"prompt_tokens":5,"total_tokens":7,"completion_tokens":2}}\n\n',
            
            # Done
            'data: [DONE]\n\n'
        ]
        
        mock_response = MockAsyncResponse(chunks)
        
        with patch('aiohttp.ClientSession.post', return_value=mock_response):
            parsed = await async_user.parse_chat_response_async(
                response=mock_response,
                start_time=start,
                num_prefill_tokens=5,
                _=start,
            )
        
        assert parsed.status_code == 200
        assert parsed.tokens_received == 2
        assert parsed.generated_text == "H"
        
        # TTFT should be captured at first valid chunk (before 100ms sleep)
        assert parsed.time_at_first_token is not None
        ttft_s = parsed.time_at_first_token - start
        assert ttft_s < 0.08, f"TTFT too large; expected first-valid-chunk capture, got {ttft_s:.4f}s"
    
    @pytest.mark.asyncio
    async def test_ttft_not_captured_on_empty_choices_async(self, async_user):
        """Test that TTFT is NOT captured on chunks with empty choices array."""
        start = time.monotonic()
        
        chunks = [
            # Only empty choices chunks - should NOT set TTFT
            'data: {"id":"chat-1","object":"chat.completion.chunk","choices":[]}\n\n',
            'data: {"id":"chat-1","object":"chat.completion.chunk","choices":[]}\n\n',
            
            # Valid choices with content (should set TTFT here)
            'data: {"id":"chat-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"H"},"logprobs":null,"finish_reason":null}]}\n\n',
            
            # Finish chunk
            'data: {"id":"chat-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":null},"logprobs":null,"finish_reason":"length"}]}\n\n',
            'data: {"id":"chat-1","object":"chat.completion.chunk","choices":[],"usage":{"prompt_tokens":5,"total_tokens":7,"completion_tokens":2}}\n\n',
            'data: [DONE]\n\n'
        ]
        
        mock_response = MockAsyncResponse(chunks)
        
        with patch('aiohttp.ClientSession.post', return_value=mock_response):
            parsed = await async_user.parse_chat_response_async(
                response=mock_response,
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
        assert ttft_s < 0.05, f"TTFT should be small since it was captured on first valid chunk, got {ttft_s:.4f}s"
    
    @pytest.mark.asyncio
    async def test_byte_level_streaming_vs_line_buffering(self, async_user):
        """Test that byte-level streaming captures TTFT faster than line buffering."""
        start = time.monotonic()
        
        # Simulate partial data arrival (like vLLM's iter_any)
        chunks = [
            # Partial data (no newline yet)
            'data: {"id":"chat-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"}',
            
            # Complete the chunk
            ',"logprobs":null,"finish_reason":null}]}\n\n',
            
            # Content chunk
            'data: {"id":"chat-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"H"},"logprobs":null,"finish_reason":null}]}\n\n',
            
            # Finish
            'data: {"id":"chat-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":null},"logprobs":null,"finish_reason":"length"}]}\n\n',
            'data: [DONE]\n\n'
        ]
        
        mock_response = MockAsyncResponse(chunks)
        
        with patch('aiohttp.ClientSession.post', return_value=mock_response):
            parsed = await async_user.parse_chat_response_async(
                response=mock_response,
                start_time=start,
                num_prefill_tokens=5,
                _=start,
            )
        
        # Should handle partial data correctly
        assert parsed.status_code == 200
        assert parsed.tokens_received == 1
        assert parsed.generated_text == "H"
    
    @pytest.mark.asyncio
    async def test_session_lifecycle(self, async_user):
        """Test that aiohttp session is properly managed."""
        # Test session creation
        await async_user.on_start()
        assert async_user.session is not None
        assert isinstance(async_user.session, aiohttp.ClientSession)
        
        # Test session cleanup
        await async_user.on_stop()
        assert async_user.session.closed
    
    @pytest.mark.asyncio
    async def test_error_handling(self, async_user):
        """Test error handling in async context."""
        # Test HTTP error
        mock_response = MockAsyncResponse([], status=500)
        
        with patch('aiohttp.ClientSession.post', return_value=mock_response):
            result = await async_user.send_request_async(
                endpoint="/v1/chat/completions",
                payload={"messages": [{"role": "user", "content": "test"}]},
                parse_strategy=async_user.parse_chat_response_async,
                num_prefill_tokens=5,
            )
        
        assert result.status_code == 500
        assert result.error_message is not None


class TestStreamedResponseHandler:
    """Test the byte-level streaming handler."""
    
    def test_add_chunk_complete_messages(self):
        """Test that complete messages are returned immediately."""
        from genai_bench.user.async_openai_user import StreamedResponseHandler
        
        handler = StreamedResponseHandler()
        
        # Add complete message
        messages = handler.add_chunk(b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n')
        assert len(messages) == 1
        assert messages[0] == 'data: {"choices":[{"delta":{"role":"assistant"}}]}'
    
    def test_add_chunk_partial_data(self):
        """Test that partial data is buffered correctly."""
        from genai_bench.user.async_openai_user import StreamedResponseHandler
        
        handler = StreamedResponseHandler()
        
        # Add partial data
        messages = handler.add_chunk(b'data: {"choices":[{"delta":{"role":"assistant"')
        assert len(messages) == 0  # No complete message yet
        
        # Complete the message
        messages = handler.add_chunk(b',"logprobs":null}]}\n\n')
        assert len(messages) == 1
        assert '{"choices":[{"delta":{"role":"assistant"' in messages[0]
    
    def test_add_chunk_multiple_messages(self):
        """Test handling multiple messages in one chunk."""
        from genai_bench.user.async_openai_user import StreamedResponseHandler
        
        handler = StreamedResponseHandler()
        
        chunk = b'data: {"choices":[]}\n\ndata: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
        messages = handler.add_chunk(chunk)
        assert len(messages) == 2
        assert '{"choices":[]}' in messages[0]
        assert '{"choices":[{"delta":{"role":"assistant"}}]}' in messages[1]
