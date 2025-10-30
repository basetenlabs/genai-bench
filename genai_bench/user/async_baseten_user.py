"""Async Baseten user implementation with aiohttp and byte-level streaming."""

import asyncio
import json
import time
from typing import Any, Callable, Dict, Optional

import aiohttp

from genai_bench.logging import init_logger
from genai_bench.protocol import UserChatRequest, UserImageChatRequest, UserResponse, UserChatResponse
from genai_bench.user.async_openai_user import AsyncOpenAIUser

logger = init_logger(__name__)


class AsyncBasetenUser(AsyncOpenAIUser):
    """Async Baseten user with aiohttp and byte-level streaming for both OpenAI and prompt formats.
    
    Supports both OpenAI-compatible chat format and simple prompt format for non-instruct models.
    Format is controlled via use_prompt_format in additional_request_params.
    Streaming is controlled via the global --disable-streaming flag (consistent with other backends).
    """
    
    BACKEND_NAME = "async-baseten"
    disable_streaming: bool = False

    async def chat(self):
        """Override chat method to support both OpenAI-compatible and prompt formats."""
        user_request = self.sample()

        if not isinstance(user_request, UserChatRequest):
            raise AttributeError(
                f"user_request should be of type "
                f"UserChatRequest for AsyncBasetenUser.chat, got "
                f"{type(user_request)}"
            )

        # Check if we should use prompt format
        use_prompt_format = user_request.additional_request_params.get("use_prompt_format", False)
        
        if use_prompt_format:
            # Use simple prompt format for non-instruct models
            payload = self._prepare_prompt_request(user_request)
            endpoint = "prompt"  # Use different endpoint name for metrics
        else:
            # Use OpenAI-compatible chat format (default)
            payload = self._prepare_chat_request(user_request)
            endpoint = "/v1/chat/completions"

        # Use global disable_streaming setting (consistent with other backends)
        use_streaming = not self.disable_streaming

        # Send request using the overridden send_request_async method
        if use_streaming:
            if use_prompt_format:
                response = await self.send_request_async(
                    endpoint=endpoint,
                    payload=payload,
                    parse_strategy=self._parse_plain_text_streaming_response_async,
                    num_prefill_tokens=user_request.num_prefill_tokens,
                )
            else:
                response = await self.send_request_async(
                    endpoint=endpoint,
                    payload=payload,
                    parse_strategy=self.parse_chat_response_async,
                    num_prefill_tokens=user_request.num_prefill_tokens,
                )
        else:
            if use_prompt_format:
                response = await self.send_request_async(
                    endpoint=endpoint,
                    payload=payload,
                    parse_strategy=self._parse_plain_text_response_async,
                    num_prefill_tokens=user_request.num_prefill_tokens,
                )
            else:
                response = await self.send_request_async(
                    endpoint=endpoint,
                    payload=payload,
                    parse_strategy=self.parse_non_streaming_chat_response_async,
                    num_prefill_tokens=user_request.num_prefill_tokens,
                )
        
        self.collect_metrics(response, endpoint)

    def _prepare_chat_request(self, user_request: UserChatRequest) -> Dict[str, Any]:
        """Prepare OpenAI-compatible chat request."""
        if isinstance(user_request, UserImageChatRequest):
            text_content = [{"type": "text", "text": user_request.prompt}]
            image_content = [
                {
                    "type": "image_url",
                    "image_url": {"url": image},  # image already contains the full data URL
                }
                for image in user_request.image_content
            ]
            content = text_content + image_content
        else:
            content = user_request.prompt

        # Use global disable_streaming setting (consistent with other backends)
        use_streaming = not self.disable_streaming

        payload = {
            "model": user_request.model,
            "messages": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            "max_tokens": user_request.max_tokens,
            "temperature": user_request.additional_request_params.get(
                "temperature", 0.0
            ),
            "ignore_eos": user_request.additional_request_params.get(
                "ignore_eos",
                bool(user_request.max_tokens),
            ),
            "stream": use_streaming,
        }
        
        # Add additional params except use_prompt_format and stream
        for key, value in user_request.additional_request_params.items():
            if key not in ["use_prompt_format", "stream"]:
                payload[key] = value
        
        # Only add stream_options if streaming is enabled
        if use_streaming:
            payload["stream_options"] = {
                "include_usage": True,
            }
            
        return payload

    def _prepare_prompt_request(self, user_request: UserChatRequest) -> Dict[str, Any]:
        """Prepare simple prompt request for non-instruct models."""
        # Use global disable_streaming setting (consistent with other backends)
        use_streaming = not self.disable_streaming

        payload = {
            "prompt": user_request.prompt,
            "max_tokens": user_request.max_tokens,
            "temperature": user_request.additional_request_params.get(
                "temperature", 0.0
            ),
            "stream": use_streaming,
        }
        
        # Add additional params except use_prompt_format and stream
        for key, value in user_request.additional_request_params.items():
            if key not in ["use_prompt_format", "stream"]:
                payload[key] = value
                
        return payload

    async def parse_chat_response_async(
        self,
        response: aiohttp.ClientResponse,
        start_time: float,
        num_prefill_tokens: Optional[int] = None,
        non_stream_post_end_time: Optional[float] = None,
        _: Optional[float] = None,  # Compatibility with sync version
    ) -> UserResponse:
        """
        Override parse_chat_response_async to handle both OpenAI format and plain text responses.
        """
        # This method is now only used for OpenAI format
        # Prompt format uses _parse_plain_text_streaming_response_async directly
        return await super().parse_chat_response_async(
            response, start_time, num_prefill_tokens, non_stream_post_end_time, _
        )

    async def parse_non_streaming_chat_response_async(
        self,
        response: aiohttp.ClientResponse,
        start_time: float,
        num_prefill_tokens: Optional[int] = None,
        non_stream_post_end_time: Optional[float] = None,
        _: Optional[float] = None,  # Compatibility with sync version
    ) -> UserResponse:
        """
        Override parse_non_streaming_chat_response_async to handle both OpenAI format and plain text responses.
        """
        # First, try to determine if this is JSON or plain text
        response_text = (await response.text()).strip()
        
        try:
            # Try to parse as JSON
            data = json.loads(response_text)
            # If successful, try to parse as OpenAI format
            return await super().parse_non_streaming_chat_response_async(
                response, start_time, num_prefill_tokens, non_stream_post_end_time, _
            )
        except (json.JSONDecodeError, AttributeError) as e:
            # If JSON parsing fails, treat as plain text
            logger.debug(f"Response is not JSON, treating as plain text: {e}")
            return await self._parse_plain_text_response_async(
                response, start_time, num_prefill_tokens, non_stream_post_end_time, _
            )

    async def _parse_plain_text_streaming_response_async(
        self,
        response: aiohttp.ClientResponse,
        start_time: float,
        num_prefill_tokens: Optional[int] = None,
        non_stream_post_end_time: Optional[float] = None,
        _: Optional[float] = None,
    ) -> UserResponse:
        """
        Parse plain text streaming response for prompt format with byte-level processing.
        """
        generated_text = ""
        time_at_first_token = None
        end_time = None

        try:
            # For plain text, process bytes directly without SSE message parsing
            async for chunk_bytes in response.content.iter_any():
                # Convert bytes to string
                chunk_text = chunk_bytes.decode('utf-8')
                
                # Set first token time on first non-empty chunk
                if not time_at_first_token and chunk_text.strip():
                    time_at_first_token = time.monotonic()
                
                # Add to generated text
                generated_text += chunk_text
                end_time = time.monotonic()
                
        except Exception as e:
            logger.error(f"Error parsing plain text streaming response: {e}")
            return UserResponse(
                status_code=500,
                error_message=f"Failed to parse plain text streaming response: {e}",
            )

        if not end_time:
            end_time = time.monotonic()

        # Estimate tokens received
        tokens_received = self.environment.sampler.get_token_length(
            generated_text, add_special_tokens=False
        )

        return UserChatResponse(
            status_code=200,
            generated_text=generated_text,
            tokens_received=tokens_received,
            time_at_first_token=time_at_first_token or start_time,
            num_prefill_tokens=num_prefill_tokens,
            start_time=start_time,
            end_time=end_time,
        )

    async def _parse_plain_text_response_async(
        self,
        response: aiohttp.ClientResponse,
        start_time: float,
        num_prefill_tokens: Optional[int] = None,
        non_stream_post_end_time: Optional[float] = None,
        _: Optional[float] = None,
    ) -> UserResponse:
        """
        Parse plain text non-streaming response for prompt format.
        """
        try:
            # Try to get the response as text
            response_text = (await response.text()).strip()
            end_time = time.monotonic()
            
            # Try to parse as JSON first (in case it's actually JSON)
            try:
                data = json.loads(response_text)
                # If it's JSON, try to extract text from common fields
                if isinstance(data, dict):
                    generated_text = (
                        data.get("text") or 
                        data.get("output") or 
                        data.get("response") or 
                        data.get("generated_text") or
                        response_text  # fallback to full response
                    )
                else:
                    generated_text = response_text
            except json.JSONDecodeError:
                # Not JSON, treat as plain text
                generated_text = response_text
            
            # Estimate tokens received
            tokens_received = self.environment.sampler.get_token_length(
                generated_text, add_special_tokens=False
            )
            
            # For non-streaming, we can't measure TTFT, so we use a small offset
            time_at_first_token = start_time + 0.001  # 1ms offset
            
            logger.debug(
                f"Plain text response: {generated_text}\n"
                f"Estimated tokens: {tokens_received}\n"
                f"Start time: {start_time}\n"
                f"End time: {end_time}"
            )
            
            return UserChatResponse(
                status_code=200,
                generated_text=generated_text,
                tokens_received=tokens_received,
                time_at_first_token=time_at_first_token,
                num_prefill_tokens=num_prefill_tokens,
                start_time=start_time,
                end_time=end_time,
            )
            
        except Exception as e:
            logger.error(f"Error parsing plain text response: {e}")
            return UserResponse(
                status_code=500,
                error_message=f"Failed to parse plain text response: {e}",
            )

    async def send_request_async(
        self,
        endpoint: str,
        payload: dict,
        parse_strategy,
        num_prefill_tokens: Optional[int] = None,
    ) -> UserResponse:
        """Send an async HTTP request and parse the response for Baseten."""
        if not self.session:
            raise RuntimeError("aiohttp session not initialized. Call on_start() first.")
        
        start_time = time.monotonic()
        
        try:
            # For Baseten, use the host directly as the URL (don't append endpoint)
            async with self.session.post(
                url=self.host,  # Use host directly instead of host + endpoint
                json=payload,
                headers=self.headers,
            ) as response:
                non_stream_post_end_time = time.monotonic()
                
                if response.status == 200:
                    metrics_response = await parse_strategy(
                        response,
                        start_time,
                        num_prefill_tokens,
                        non_stream_post_end_time,
                    )
                else:
                    error_text = await response.text()
                    metrics_response = UserResponse(
                        status_code=response.status,
                        error_message=error_text,
                    )
                
                return metrics_response
                
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return UserResponse(
                status_code=-1,
                error_message=str(e),
            )
