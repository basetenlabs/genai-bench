"""Loader for conversation datasets (JSONL/HuggingFace with messages arrays)."""

import base64
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from genai_bench.data.loaders.base import DatasetFormat, DatasetLoader
from genai_bench.logging import init_logger

logger = init_logger(__name__)


class ConversationDatasetLoader(DatasetLoader):
    """Load pre-built conversation datasets for real-dataset benchmarking.

    Handles JSONL files and HuggingFace datasets where each row contains
    a conversation (messages array). Converts images from local file paths
    or URLs to base64 data URLs at load time.

    Expected row format:
        {"conversations": [{"role": "...", "content": ...}, ...]}
    or  {"messages": [{"role": "...", "content": ...}, ...]}

    The messages_column config field controls which field to read.
    """

    supported_formats: Set[DatasetFormat] = {
        DatasetFormat.JSONL,
        DatasetFormat.JSON,
        DatasetFormat.HUGGINGFACE_HUB,
    }
    media_type = "Conversation"

    def __init__(self, dataset_config, dataset_dir: Optional[Path] = None):
        """Initialize with dataset config and optional base directory for images.

        Args:
            dataset_config: DatasetConfig with source and messages_column.
            dataset_dir: Base directory for resolving relative image paths.
                If None, uses the directory of the dataset source file.
        """
        super().__init__(dataset_config)
        if dataset_dir is not None:
            self._dataset_dir = dataset_dir
        elif dataset_config.source.path:
            self._dataset_dir = Path(dataset_config.source.path).parent
        else:
            self._dataset_dir = Path(".")

    def _process_loaded_data(self, data: Any) -> List[Dict]:
        """Convert raw data rows to pre-processed conversation samples."""
        messages_column = self.dataset_config.messages_column or "messages"

        # Handle HuggingFace Dataset objects
        if hasattr(data, "__len__") and hasattr(data, "__getitem__"):
            rows = [data[i] for i in range(len(data))]
        elif isinstance(data, list):
            rows = data
        else:
            raise ValueError(
                f"Unsupported data type for conversation loading: {type(data)}"
            )

        samples = []
        truncated_count = 0
        for row in rows:
            if messages_column not in row:
                raise ValueError(
                    f"Row missing '{messages_column}' column. "
                    f"Available keys: {list(row.keys()) if isinstance(row, dict) else 'N/A'}. "
                    f"Set messages_column in your dataset config."
                )

            raw_messages = row[messages_column]

            # Truncate last assistant message (the response to predict)
            if raw_messages and raw_messages[-1].get("role") == "assistant":
                raw_messages = raw_messages[:-1]
                truncated_count += 1

            messages = self._convert_messages(raw_messages)
            num_images = self._count_images(messages)
            samples.append(
                {
                    "id": row.get("id"),
                    "messages": messages,
                    "num_images": num_images,
                }
            )

        # Pre-shuffle for non-repeating sampling
        random.shuffle(samples)

        logger.info(
            f"Loaded {len(samples)} conversation samples "
            f"({truncated_count} with last assistant message truncated, "
            f"total images: {sum(s['num_images'] for s in samples)})"
        )
        return samples

    def _convert_messages(self, raw_messages: List[Dict]) -> List[Dict]:
        """Convert conversation messages to OpenAI format with base64 images."""
        messages = []
        for turn in raw_messages:
            role = turn.get("role", "user")
            content = turn.get("content", "")

            if isinstance(content, str):
                messages.append({"role": role, "content": content})
            elif isinstance(content, list):
                oai_parts = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    part_type = part.get("type", "")
                    if part_type == "text":
                        oai_parts.append({"type": "text", "text": part.get("text", "")})
                    elif part_type == "image":
                        image_url = self._load_image(part.get("image", ""))
                        oai_parts.append(
                            {"type": "image_url", "image_url": {"url": image_url}}
                        )
                    elif part_type == "image_url":
                        # Already in OpenAI format
                        oai_parts.append(part)
                    else:
                        # Unknown part type — pass through as text
                        oai_parts.append(
                            {"type": "text", "text": str(part.get("text", ""))}
                        )
                messages.append({"role": role, "content": oai_parts})
            else:
                messages.append({"role": role, "content": str(content)})

        return messages

    def _load_image(self, image_ref: str) -> str:
        """Convert image reference to base64 data URL or pass through URL."""
        if not image_ref:
            return ""

        # HTTP(S) URLs — pass through
        if image_ref.startswith(("http://", "https://")):
            return image_ref

        # Data URLs — already base64
        if image_ref.startswith("data:"):
            return image_ref

        # Local file path — load and convert to base64
        path = Path(image_ref)
        if not path.is_absolute():
            path = self._dataset_dir / path

        if not path.exists():
            logger.warning(f"Image file not found: {path}")
            return ""

        with open(path, "rb") as f:
            image_bytes = f.read()

        b64 = base64.b64encode(image_bytes).decode("utf-8")

        # Detect MIME type from magic bytes
        if image_bytes[:4] == b"\x89PNG":
            mime = "image/png"
        elif image_bytes[:2] == b"\xff\xd8":
            mime = "image/jpeg"
        elif image_bytes[:4] == b"GIF8":
            mime = "image/gif"
        elif image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
            mime = "image/webp"
        else:
            mime = "image/png"  # default

        return f"data:{mime};base64,{b64}"

    @staticmethod
    def _count_images(messages: List[Dict]) -> int:
        """Count total images across all messages."""
        count = 0
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        count += 1
        return count
