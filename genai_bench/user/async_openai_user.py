"""Async OpenAI user implementation with aiohttp and byte-level streaming."""

import asyncio
import json
import time
from typing import Any, Callable, Dict, Optional

import aiohttp

from genai_bench.auth.model_auth_provider import ModelAuthProvider
from genai_bench.logging import init_logger
from genai_bench.protocol import (
    UserChatRequest,
    UserChatResponse,
    UserEmbeddingRequest,
    UserImageChatRequest,
    UserResponse,
)
from genai_bench.user.async_base_user import AsyncBaseUser, StreamedResponseHandler

logger = init_logger(__name__)


class AsyncOpenAIUser(AsyncBaseUser):
    """Async OpenAI user with aiohttp and byte-level streaming (vLLM-compatible)."""
    
    BACKEND_NAME = "async-openai"
    supported_tasks = {
        "text-to-text": "chat",
        "image-text-to-text": "chat",
        "text-to-embeddings": "embeddings",
    }

    host: Optional[str] = None
    auth_provider: Optional[ModelAuthProvider] = None
    headers = None
    disable_streaming: bool = False

    async def on_start(self):
        """Initialize headers and session."""
        if not self.host or not self.auth_provider:
            raise ValueError("API key and base must be set for AsyncOpenAIUser.")
        self.headers = {
            "Authorization": f"Bearer {self.auth_provider.get_credentials()}",
            "Content-Type": "application/json",
        }
        await super().on_start()

    async def chat(self):
        """Async chat task."""
        endpoint = "/v1/chat/completions"
        request = self.sample()
        
        if isinstance(request, UserChatRequest):
            payload = self._build_chat_payload(request)
            response = await self.send_request_async(
                endpoint=endpoint,
                payload=payload,
                parse_strategy=self.parse_chat_response_async,
                num_prefill_tokens=request.num_prefill_tokens,
            )
            self.collect_metrics(response, endpoint)
        else:
            logger.error(f"Unsupported request type: {type(request)}")

    async def embeddings(self):
        """Async embeddings task."""
        endpoint = "/v1/embeddings"
        request = self.sample()
        
        if isinstance(request, UserEmbeddingRequest):
            payload = self._build_embeddings_payload(request)
            response = await self.send_request_async(
                endpoint=endpoint,
                payload=payload,
                parse_strategy=self.parse_embeddings_response_async,
            )
            self.collect_metrics(response, endpoint)
        else:
            logger.error(f"Unsupported request type: {type(request)}")

    def _build_chat_payload(self, request: UserChatRequest) -> dict:
        """Build chat completion payload."""
        payload = {
            "model": request.model,
            "messages": request.messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": not self.disable_streaming,
        }
        
        if request.stream_options:
            payload["stream_options"] = request.stream_options
        
        return payload

    def _build_embeddings_payload(self, request: UserEmbeddingRequest) -> dict:
        """Build embeddings payload."""
        return {
            "model": request.model,
            "input": request.input,
            "encoding_format": request.encoding_format,
        }

    async def parse_chat_response_async(
        self,
        response: aiohttp.ClientResponse,
        start_time: float,
        num_prefill_tokens: Optional[int] = None,
        non_stream_post_end_time: Optional[float] = None,
        _: Optional[float] = None,  # Compatibility with sync version
    ) -> UserResponse:
        """Parse async streaming chat response with byte-level processing (vLLM-compatible)."""
        if not self.disable_streaming:
            return await self._parse_streaming_response_async(
                response, start_time, num_prefill_tokens, non_stream_post_end_time
            )
        else:
            return await self._parse_non_streaming_response_async(
                response, start_time, num_prefill_tokens, non_stream_post_end_time
            )

    async def _parse_streaming_response_async(
        self,
        response: aiohttp.ClientResponse,
        start_time: float,
        num_prefill_tokens: Optional[int] = None,
        non_stream_post_end_time: Optional[float] = None,
    ) -> UserResponse:
        """Parse streaming response with byte-level processing like vLLM."""
        generated_text = ""
        tokens_received = 0
        time_at_first_token = None
        finish_reason = None
        previous_data = None
        num_prompt_tokens = None
        
        # Use byte-level streaming like vLLM for immediate processing
        handler = StreamedResponseHandler()
        
        try:
            async for chunk_bytes in response.content.iter_any():
                messages = handler.add_chunk(chunk_bytes)
                
                for message in messages:
                    # Skip SSE comments (like vLLM)
                    if message.startswith(":"):
                        continue
                    
                    chunk = message.removeprefix("data: ")
                    if chunk == "[DONE]":
                        break
                    
                    try:
                        data = json.loads(chunk)
                        
                        # Capture TTFT at first valid JSON with choices (vLLM-compatible)
                        if time_at_first_token is None and ("choices" in data):
                            time_at_first_token = time.monotonic()
                        
                        # Handle streaming error response
                        if data.get("error") is not None:
                            return UserResponse(
                                status_code=data["error"].get("code", -1),
                                error_message=data["error"].get(
                                    "message", "Unknown error, please check server logs"
                                ),
                            )
                        
                        # Process choices
                        if "choices" in data and data["choices"]:
                            choice = data["choices"][0]
                            if "delta" in choice:
                                delta = choice["delta"]
                                if "content" in delta and delta["content"]:
                                    generated_text += delta["content"]
                                    tokens_received += 1
                                
                                if "finish_reason" in choice and choice["finish_reason"]:
                                    finish_reason = choice["finish_reason"]
                        
                        # Handle usage information
                        if "usage" in data and data["usage"]:
                            usage = data["usage"]
                            if "prompt_tokens" in usage:
                                num_prompt_tokens = usage["prompt_tokens"]
                            if "completion_tokens" in usage:
                                tokens_received = usage["completion_tokens"]
                        
                        previous_data = data
                        
                    except json.JSONDecodeError:
                        # Skip malformed JSON chunks
                        continue
            
            # Handle case where no streaming data was received
            if not time_at_first_token and num_prompt_tokens is None:
                return UserResponse(
                    status_code=500,
                    error_message="No valid streaming data received",
                )
            
            # Create response
            end_time = time.monotonic()
            return UserChatResponse(
                status_code=200,
                generated_text=generated_text,
                tokens_received=tokens_received,
                num_prefill_tokens=num_prefill_tokens or num_prompt_tokens,
                finish_reason=finish_reason,
                start_time=start_time,
                end_time=end_time,
                time_at_first_token=time_at_first_token,
            )
            
        except Exception as e:
            logger.error(f"Error parsing streaming response: {e}")
            return UserResponse(
                status_code=500,
                error_message=f"Error parsing streaming response: {str(e)}",
            )

    async def _parse_non_streaming_response_async(
        self,
        response: aiohttp.ClientResponse,
        start_time: float,
        num_prefill_tokens: Optional[int] = None,
        non_stream_post_end_time: Optional[float] = None,
    ) -> UserResponse:
        """Parse non-streaming response."""
        try:
            data = await response.json()
            
            if "error" in data:
                return UserResponse(
                    status_code=data["error"].get("code", -1),
                    error_message=data["error"].get("message", "Unknown error"),
                )
            
            # Extract response data
            choices = data.get("choices", [])
            generated_text = ""
            finish_reason = None
            
            if choices:
                choice = choices[0]
                generated_text = choice.get("message", {}).get("content", "")
                finish_reason = choice.get("finish_reason")
            
            # Extract usage information
            usage = data.get("usage", {})
            tokens_received = usage.get("completion_tokens", 0)
            num_prompt_tokens = usage.get("prompt_tokens", num_prefill_tokens)
            
            end_time = time.monotonic()
            time_at_first_token = start_time + 0.001  # 1ms offset for non-streaming
            
            return UserChatResponse(
                status_code=200,
                generated_text=generated_text,
                tokens_received=tokens_received,
                num_prefill_tokens=num_prompt_tokens,
                finish_reason=finish_reason,
                start_time=start_time,
                end_time=end_time,
                time_at_first_token=time_at_first_token,
            )
            
        except Exception as e:
            logger.error(f"Error parsing non-streaming response: {e}")
            return UserResponse(
                status_code=500,
                error_message=f"Error parsing response: {str(e)}",
            )

    async def parse_embeddings_response_async(
        self,
        response: aiohttp.ClientResponse,
        start_time: float,
        num_prefill_tokens: Optional[int] = None,
        non_stream_post_end_time: Optional[float] = None,
    ) -> UserResponse:
        """Parse embeddings response."""
        try:
            data = await response.json()
            
            if "error" in data:
                return UserResponse(
                    status_code=data["error"].get("code", -1),
                    error_message=data["error"].get("message", "Unknown error"),
                )
            
            # Extract embeddings data
            embeddings = data.get("data", [])
            embeddings_list = [item.get("embedding", []) for item in embeddings]
            
            end_time = time.monotonic()
            
            return UserResponse(
                status_code=200,
                start_time=start_time,
                end_time=end_time,
                time_at_first_token=start_time + 0.001,
                embeddings=embeddings_list,
            )
            
        except Exception as e:
            logger.error(f"Error parsing embeddings response: {e}")
            return UserResponse(
                status_code=500,
                error_message=f"Error parsing embeddings response: {str(e)}",
            )
