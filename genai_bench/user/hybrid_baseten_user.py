"""Hybrid BasetenUser that uses aiohttp for better performance but works with Locust's distributed runner."""

import asyncio
import json
import time
from typing import Any, Callable, Dict, Optional

import aiohttp
from locust import task
from requests import Response

from genai_bench.logging import init_logger
from genai_bench.protocol import UserChatRequest, UserImageChatRequest, UserResponse, UserChatResponse
from genai_bench.user.baseten_user import BasetenUser

logger = init_logger(__name__)


class HybridBasetenUser(BasetenUser):
    """
    Hybrid BasetenUser that uses aiohttp for better performance but works with Locust.
    
    This user inherits from BasetenUser but overrides the HTTP request methods
    to use aiohttp for better streaming performance while maintaining compatibility
    with Locust's distributed runner system.
    """
    
    BACKEND_NAME = "hybrid-baseten"
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._session = None
        self._loop = None
    
    def _get_session(self):
        """Get or create aiohttp session for this user."""
        if self._session is None or self._session.closed:
            # Create new event loop for this thread if needed
            try:
                self._loop = asyncio.get_event_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
            
            self._session = aiohttp.ClientSession()
        return self._session
    
    def _close_session(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            if self._loop and self._loop.is_running():
                # Schedule the close for later
                self._loop.create_task(self._session.close())
            else:
                # Run in the event loop
                self._loop.run_until_complete(self._session.close())
            self._session = None
    
    def send_request(
        self,
        stream: bool,
        endpoint: str,
        payload: Dict[str, Any],
        parse_strategy: Callable[..., UserResponse],
        num_prefill_tokens: Optional[int] = None,
    ) -> UserResponse:
        """
        Override send_request to use aiohttp for better performance.
        """
        try:
            # Use aiohttp for the request
            return self._run_async_request(stream, endpoint, payload, parse_strategy, num_prefill_tokens)
        except Exception as e:
            logger.error(f"Async request failed, falling back to sync: {e}")
            # Fallback to parent's synchronous implementation
            return super().send_request(stream, endpoint, payload, parse_strategy, num_prefill_tokens)
    
    def _run_async_request(
        self,
        stream: bool,
        endpoint: str,
        payload: Dict[str, Any],
        parse_strategy: Callable[..., UserResponse],
        num_prefill_tokens: Optional[int] = None,
    ) -> UserResponse:
        """Run an async request using aiohttp."""
        session = self._get_session()
        
        async def _async_request():
            start_time = time.monotonic()
            try:
                # For Baseten, use the host directly as the URL
                async with session.post(
                    url=self.host,  # Use host directly instead of host + endpoint
                    json=payload,
                    headers=self.headers,
                ) as response:
                    non_stream_post_end_time = time.monotonic()
                    
                    if response.status == 200:
                        # Parse the response using the strategy
                        if stream:
                            return await self._parse_streaming_response_async(
                                response, start_time, num_prefill_tokens, non_stream_post_end_time
                            )
                        else:
                            return await self._parse_non_streaming_response_async(
                                response, start_time, num_prefill_tokens, non_stream_post_end_time
                            )
                    else:
                        error_text = await response.text()
                        return UserResponse(
                            status_code=response.status,
                            error_message=error_text,
                        )
            except Exception as e:
                logger.error(f"Async request error: {e}")
                return UserResponse(
                    status_code=-1,
                    error_message=str(e),
                )
        
        # Run the async request in the event loop
        try:
            return self._loop.run_until_complete(_async_request())
        except Exception as e:
            logger.warning(f"Async request failed, falling back to sync: {e}")
            return super().send_request(stream, endpoint, payload, parse_strategy, num_prefill_tokens)
    
    async def _parse_streaming_response_async(
        self,
        response: aiohttp.ClientResponse,
        start_time: float,
        num_prefill_tokens: Optional[int],
        non_stream_post_end_time: Optional[float],
    ) -> UserResponse:
        """Parse streaming response using aiohttp with byte-level processing."""
        generated_text = ""
        tokens_received = 0
        time_at_first_token = None
        finish_reason = None
        num_prompt_tokens = None
        
        try:
            # Use byte-level streaming like vLLM
            from genai_bench.user.async_base_user import StreamedResponseHandler
            handler = StreamedResponseHandler()
            
            async for chunk_bytes in response.content.iter_any():
                messages = handler.add_chunk(chunk_bytes)
                
                for message_bytes in messages:
                    if not message_bytes.startswith(b"data: "):
                        continue  # Skip non-data lines
                    
                    message_data = message_bytes[6:].strip()  # Remove "data: " prefix
                    if message_data == b"[DONE]":
                        break
                    
                    try:
                        data = json.loads(message_data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON decode error: {e}")
                        continue
                    
                    if data.get("error") is not None:
                        return UserResponse(
                            status_code=data["error"].get("code", -1),
                            error_message=data["error"].get("message", "Unknown error"),
                        )
                    
                    # Skip chunks with empty choices
                    if not data.get("choices"):
                        continue
                    
                    delta = data["choices"][0]["delta"]
                    content = delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning")
                    usage = delta.get("usage")
                    
                    if usage:
                        tokens_received = usage["completion_tokens"]
                    
                    if time_at_first_token is None:
                        if tokens_received > 1:
                            logger.warning(
                                f"🚨🚨🚨 The first chunk the server returned "
                                f"has >1 tokens: {tokens_received}. It will "
                                f"affect the accuracy of time_at_first_token!"
                            )
                        time_at_first_token = time.monotonic()
                    
                    if content:
                        generated_text += content
                    
                    finish_reason = data["choices"][0].get("finish_reason", None)
                    
                    if finish_reason and "usage" in data and data["usage"]:
                        num_prefill_tokens, num_prompt_tokens, tokens_received = (
                            self._get_usage_info(data, num_prefill_tokens)
                        )
                        break
                        
        except Exception as e:
            logger.error(f"Error parsing streaming response: {e}")
            return UserResponse(
                status_code=500,
                error_message=f"Error parsing streaming response: {e}",
            )
        
        end_time = time.monotonic()
        
        return UserResponse(
            status_code=response.status,
            time_at_first_token=time_at_first_token,
            start_time=start_time,
            end_time=end_time,
            tokens_received=tokens_received,
            generated_text=generated_text,
            num_prefill_tokens=num_prefill_tokens,
            num_prompt_tokens=num_prompt_tokens,
        )
    
    async def _parse_non_streaming_response_async(
        self,
        response: aiohttp.ClientResponse,
        start_time: float,
        num_prefill_tokens: Optional[int],
        non_stream_post_end_time: Optional[float],
    ) -> UserResponse:
        """Parse non-streaming response using aiohttp."""
        try:
            response_text = await response.text()
            end_time = time.monotonic()
            
            # Parse JSON response
            data = json.loads(response_text)
            
            if "error" in data:
                return UserResponse(
                    status_code=data["error"].get("code", -1),
                    error_message=data["error"].get("message", "Unknown error"),
                )
            
            # Extract response data
            choices = data.get("choices", [])
            if not choices:
                return UserResponse(
                    status_code=500,
                    error_message="No choices in response",
                )
            
            choice = choices[0]
            message = choice.get("message", {})
            generated_text = message.get("content", "")
            
            # Get usage information
            usage = data.get("usage", {})
            tokens_received = usage.get("completion_tokens", 0)
            num_prompt_tokens = usage.get("prompt_tokens", 0)
            
            if num_prefill_tokens is None:
                num_prefill_tokens = num_prompt_tokens
            
            return UserResponse(
                status_code=response.status,
                time_at_first_token=start_time + 0.001,  # Small offset for non-streaming
                start_time=start_time,
                end_time=end_time,
                tokens_received=tokens_received,
                generated_text=generated_text,
                num_prefill_tokens=num_prefill_tokens,
                num_prompt_tokens=num_prompt_tokens,
            )
            
        except Exception as e:
            logger.error(f"Error parsing non-streaming response: {e}")
            return UserResponse(
                status_code=500,
                error_message=f"Error parsing non-streaming response: {e}",
            )
    
    def on_stop(self):
        """Clean up aiohttp session when user stops."""
        self._close_session()
        super().on_stop()
