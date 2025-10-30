"""Async base user class with aiohttp support."""

import asyncio
import time
from typing import Dict, Optional

import aiohttp
from locust import User

from genai_bench.logging import init_logger
from genai_bench.metrics.request_metrics_collector import RequestMetricsCollector
from genai_bench.protocol import UserRequest, UserResponse

logger = init_logger(__name__)


class StreamedResponseHandler:
    """Handles streaming HTTP responses by accumulating chunks until complete
    messages are available (like vLLM's implementation)."""
    
    def __init__(self):
        self.buffer = ""
    
    def add_chunk(self, chunk_bytes: bytes) -> list[str]:
        """Add a chunk of bytes to the buffer and return any complete messages."""
        chunk_str = chunk_bytes.decode("utf-8")
        self.buffer += chunk_str
        
        messages = []
        
        # Split by double newlines (SSE message separator)
        while "\n\n" in self.buffer:
            message, self.buffer = self.buffer.split("\n\n", 1)
            message = message.strip()
            if message:
                messages.append(message)
        
        # if self.buffer is not empty, check if it is a complete message
        # by removing data: prefix and check if it is a valid JSON
        if self.buffer.startswith("data: "):
            message_content = self.buffer.removeprefix("data: ").strip()
            if message_content == "[DONE]":
                messages.append(self.buffer.strip())
                self.buffer = ""
            elif message_content:
                try:
                    # Try to parse as JSON to see if it's complete
                    import json
                    json.loads(message_content)
                    messages.append(self.buffer.strip())
                    self.buffer = ""
                except json.JSONDecodeError:
                    # Not complete yet, keep buffering
                    pass
        
        return messages


class AsyncBaseUser(User):
    """Async base user class with aiohttp support for byte-level streaming."""
    
    supported_tasks: Dict[str, str] = {}
    
    def __new__(cls, *args, **kwargs):
        if cls is AsyncBaseUser:
            raise TypeError("AsyncBaseUser is not meant to be instantiated directly.")
        return super().__new__(cls)
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.session: Optional[aiohttp.ClientSession] = None
        self.metrics_collector = RequestMetricsCollector()
    
    async def on_start(self):
        """Initialize aiohttp session on start."""
        self.session = aiohttp.ClientSession()
        super().on_start()
    
    async def on_stop(self):
        """Clean up aiohttp session on stop."""
        if self.session and not self.session.closed:
            await self.session.close()
        super().on_stop()
    
    @classmethod
    def is_task_supported(cls, task: str) -> bool:
        return task in cls.supported_tasks
    
    def sample(self) -> UserRequest:
        if not (
            hasattr(self.environment, "scenario")
            and self.environment.scenario is not None
        ):
            raise AttributeError(
                f"Environment {self.environment} has no attribute "
                f"'scenario' or it is empty."
            )
        if not (
            hasattr(self.environment, "sampler")
            and self.environment.sampler is not None
        ):
            raise AttributeError(
                f"Environment {self.environment} has no attribute "
                f"'sampler' or it is empty."
            )
        return self.environment.sampler.sample(self.environment.scenario)
    
    def collect_metrics(
        self,
        user_response: UserResponse,
        endpoint: str,
    ):
        """Collect metrics for the user response."""
        self.metrics_collector.calculate_metrics(user_response)
    
    async def send_request_async(
        self,
        endpoint: str,
        payload: dict,
        parse_strategy,
        num_prefill_tokens: Optional[int] = None,
    ) -> UserResponse:
        """Send an async HTTP request and parse the response."""
        if not self.session:
            raise RuntimeError("aiohttp session not initialized. Call on_start() first.")
        
        start_time = time.monotonic()
        
        try:
            async with self.session.post(
                url=f"{self.host}{endpoint}",
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
