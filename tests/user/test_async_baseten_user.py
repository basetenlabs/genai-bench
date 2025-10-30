"""Tests for AsyncBasetenUser implementation with aiohttp."""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
import aiohttp

from genai_bench.user.async_baseten_user import AsyncBasetenUser
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


class TestAsyncBasetenUser:
    """Test AsyncBasetenUser implementation."""
    
    @pytest.fixture
    def mock_auth(self):
        """Create mock auth provider."""
        auth = MagicMock()
        auth.get_credentials.return_value = "test-api-key"
        return auth
    
    @pytest_asyncio.fixture
    async def async_user(self, mock_auth):
        """Create AsyncBasetenUser instance."""
        AsyncBasetenUser.host = "http://example.com"
        user = AsyncBasetenUser(environment=MagicMock())
        user.auth_provider = mock_auth
        user.headers = {
            "Authorization": "Bearer test-api-key",
            "Content-Type": "application/json",
        }
        await user.on_start()  # Initialize session
        yield user
        await user.on_stop()  # Clean up session
    
    @pytest.mark.asyncio
    async def test_openai_format_streaming(self, async_user):
        """Test OpenAI-compatible format streaming with byte-level processing."""
        start = time.monotonic()
        
        # Mock OpenAI-compatible streaming chunks
        chunks = [
            'data: {"id":"chat-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant"},"logprobs":null,"finish_reason":null}]}\n\n',
            'data: {"id":"chat-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"},"logprobs":null,"finish_reason":null}]}\n\n',
            'data: {"id":"chat-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" world"},"logprobs":null,"finish_reason":null}]}\n\n',
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
        
        assert parsed.status_code == 200
        assert parsed.generated_text == "Hello world"
        assert parsed.tokens_received == 2
        assert parsed.time_at_first_token is not None
    
    @pytest.mark.asyncio
    async def test_plain_text_streaming(self, async_user):
        """Test plain text format streaming with byte-level processing."""
        start = time.monotonic()
        
        # Mock plain text streaming chunks
        chunks = [
            "Hello",
            " world",
            " from",
            " Baseten!"
        ]
        
        mock_response = MockAsyncResponse(chunks)
        
        with patch('aiohttp.ClientSession.post', return_value=mock_response):
            parsed = await async_user._parse_plain_text_streaming_response_async(
                response=mock_response,
                start_time=start,
                num_prefill_tokens=5,
                _=start,
            )
        
        assert parsed.status_code == 200
        assert parsed.generated_text == "Hello world from Baseten!"
        assert parsed.time_at_first_token is not None
        assert parsed.time_at_first_token > start
    
    @pytest.mark.asyncio
    async def test_plain_text_non_streaming(self, async_user):
        """Test plain text non-streaming response."""
        start = time.monotonic()
        
        mock_response = MockAsyncResponse([])
        mock_response.text = AsyncMock(return_value="Hello from Baseten!")
        
        with patch('aiohttp.ClientSession.post', return_value=mock_response):
            parsed = await async_user._parse_plain_text_response_async(
                response=mock_response,
                start_time=start,
                num_prefill_tokens=5,
                _=start,
            )
        
        assert parsed.status_code == 200
        assert parsed.generated_text == "Hello from Baseten!"
        assert parsed.time_at_first_token == start + 0.001  # 1ms offset for non-streaming
    
    @pytest.mark.asyncio
    async def test_json_plain_text_response(self, async_user):
        """Test plain text response that's actually JSON."""
        start = time.monotonic()
        
        json_response = {"text": "Hello from JSON Baseten!"}
        mock_response = MockAsyncResponse([])
        mock_response.text = AsyncMock(return_value=json.dumps(json_response))
        
        with patch('aiohttp.ClientSession.post', return_value=mock_response):
            parsed = await async_user._parse_plain_text_response_async(
                response=mock_response,
                start_time=start,
                num_prefill_tokens=5,
                _=start,
            )
        
        assert parsed.status_code == 200
        assert parsed.generated_text == "Hello from JSON Baseten!"
    
    @pytest.mark.asyncio
    async def test_chat_with_prompt_format(self, async_user):
        """Test chat method with prompt format."""
        # Mock the sample method to return a request with use_prompt_format=True
        user_request = UserChatRequest(
            model="test-model",
            prompt="Hello",
            num_prefill_tokens=5,
            additional_request_params={"use_prompt_format": True},
            max_tokens=10,
        )
        async_user.sample = lambda: user_request
        
        # Mock the response
        mock_response = MockAsyncResponse(["Hello", " from", " prompt", " format!"])
        
        with patch('aiohttp.ClientSession.post', return_value=mock_response):
            await async_user.chat()
        
        # The test passes if no exception is raised
    
    @pytest.mark.asyncio
    async def test_chat_with_openai_format(self, async_user):
        """Test chat method with OpenAI format (default)."""
        # Mock the sample method to return a standard request
        user_request = UserChatRequest(
            model="test-model",
            prompt="Hello",
            num_prefill_tokens=5,
            additional_request_params={},
            max_tokens=10,
        )
        async_user.sample = lambda: user_request
        
        # Mock OpenAI-compatible response
        chunks = [
            'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
            'data: {"choices":[{"delta":{"content":"Hello from OpenAI format"}}]}\n\n',
            'data: [DONE]\n\n'
        ]
        mock_response = MockAsyncResponse(chunks)
        
        with patch('aiohttp.ClientSession.post', return_value=mock_response):
            await async_user.chat()
        
        # The test passes if no exception is raised
    
    @pytest.mark.asyncio
    async def test_send_request_async_baseten_url(self, async_user):
        """Test that send_request_async uses host directly (not host + endpoint)."""
        mock_response = MockAsyncResponse([])
        
        with patch('aiohttp.ClientSession.post', return_value=mock_response) as mock_post:
            await async_user.send_request_async(
                endpoint="test-endpoint",  # This should be ignored
                payload={"test": "data"},
                parse_strategy=async_user.parse_chat_response_async,
                num_prefill_tokens=5,
            )
        
        # Verify that the URL used was just the host, not host + endpoint
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]['url'] == "http://example.com"  # Just the host
        assert 'test-endpoint' not in call_args[1]['url']
    
    @pytest.mark.asyncio
    async def test_error_handling(self, async_user):
        """Test error handling in async context."""
        # Test HTTP error
        mock_response = MockAsyncResponse([], status=500)
        
        with patch('aiohttp.ClientSession.post', return_value=mock_response):
            result = await async_user.send_request_async(
                endpoint="test",
                payload={"test": "data"},
                parse_strategy=async_user.parse_chat_response_async,
                num_prefill_tokens=5,
            )
        
        assert result.status_code == 500
        assert result.error_message is not None
