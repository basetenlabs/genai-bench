import base64
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import PIL
from PIL.Image import Image
from six import BytesIO

from genai_bench.data.config import DatasetConfig
from genai_bench.logging import init_logger
from genai_bench.protocol import (
    UserImageChatRequest,
    UserImageEmbeddingRequest,
    UserRequest,
)
from genai_bench.sampling.base import Sampler
from genai_bench.scenarios.base import MultiModality, Scenario
from genai_bench.scenarios.multimodal import (
    DeterministicImageScenario,
    PrefixImageScenario,
)
from genai_bench.utils import safe_eval_prompt

logger = init_logger(__name__)


class ImageSampler(Sampler):
    """
    A sampler for image-based tasks, supporting multiple output modalities:
    - `image-text-to-text`: Generates `UserImageChatRequest` for vision-based chat
      tasks.
    - `image-to-embeddings`: Generates `UserImageEmbeddingRequest` for image
      embedding tasks.
    """

    input_modality = "image"
    supported_tasks = {"image-text-to-text", "image-to-embeddings"}

    def __init__(
        self,
        tokenizer,
        model: str,
        output_modality: str,
        data: Any,
        dataset_config: Optional[DatasetConfig] = None,
        additional_request_params: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(
            tokenizer,
            model,
            output_modality,
            additional_request_params,
            dataset_config=dataset_config,
            **kwargs,
        )
        self.data = data

        # Text corpus for synthetic text generation (ID/IP scenarios)
        self._text_data = self._load_text_corpus()

        # Shuffle-without-replacement state for image sampling
        self._image_indices: List[int] = []
        self._image_index_pos: int = 0

        # Prefix cache for IP scenarios (mirrors TextSampler pattern)
        self._shared_prefix_cache: Dict[str, str] = {}
        self._suffix_counter = 0

    def _load_text_corpus(self) -> List[str]:
        """Load sonnet.txt lines for synthetic text generation."""
        sonnet_path = Path(__file__).parent.parent / "data" / "sonnet.txt"
        with open(sonnet_path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    def sample(self, scenario: Optional[Scenario]) -> UserRequest:
        """
        Samples a request based on the scenario or dataset configuration.

        Args:
            scenario (Scenario, optional): The scenario to use for sampling.
                If None, uses dataset configuration directly.

        Returns:
            UserRequest: A request object for the task.
        """
        # Dataset mode when scenario is dataset or None
        if self._is_dataset_mode(scenario):
            return self._sample_legacy_request(None, None, None)

        self._validate_scenario(scenario)
        sampled = scenario.sample()

        # Branch on scenario type by tuple length
        if isinstance(scenario, PrefixImageScenario):
            # IP scenario: 5-tuple ((w,h), num_images, prefix, suffix, output)
            image_dimension, num_images, prefix_tokens, suffix_tokens, output_tokens = (
                sampled
            )
            return self._sample_prefix_image_request(
                image_dimension, num_images, prefix_tokens, suffix_tokens, output_tokens
            )
        elif isinstance(scenario, DeterministicImageScenario):
            # ID scenario: 4-tuple ((w,h), num_images, input_tokens, output_tokens)
            image_dimension, num_images, num_input_tokens, num_output_tokens = sampled
            return self._sample_deterministic_image_request(
                image_dimension, num_images, num_input_tokens, num_output_tokens
            )
        else:
            # Legacy I scenario: 3-tuple ((w,h), num_images, max_output_token)
            image_dimension, num_images, num_output_tokens = sampled
            return self._sample_legacy_request(
                image_dimension, num_images, num_output_tokens
            )

    def _sample_deterministic_image_request(
        self,
        image_dimension: Tuple[int, int],
        num_images: int,
        num_input_tokens: int,
        num_output_tokens: int,
    ) -> UserRequest:
        """Handle ID() scenarios: unique synthetic text + non-repeating images."""
        image_content = self._sample_images(image_dimension, num_images)
        prompt = self._sample_text(num_input_tokens)
        num_prefill_tokens = self.get_token_length(prompt)

        self._check_discrepancy(num_input_tokens, num_prefill_tokens, threshold=0.1)

        # Set output token control (matching TextSampler pattern)
        self.additional_request_params["ignore_eos"] = True
        if not self.no_min_tokens:
            self.additional_request_params["min_tokens"] = num_output_tokens
        self.additional_request_params["max_tokens"] = num_output_tokens

        if self.output_modality == "text":
            return UserImageChatRequest(
                model=self.model,
                prompt=prompt,
                image_content=image_content,
                num_images=num_images,
                max_tokens=num_output_tokens,
                num_prefill_tokens=num_prefill_tokens,
                additional_request_params=self.additional_request_params,
            )
        elif self.output_modality == "embeddings":
            return self._generate_image_embedding_request(image_content, num_images)
        else:
            raise ValueError(f"Unsupported output modality: {self.output_modality}")

    def _sample_prefix_image_request(
        self,
        image_dimension: Tuple[int, int],
        num_images: int,
        prefix_tokens: int,
        suffix_tokens: int,
        output_tokens: int,
    ) -> UserRequest:
        """Handle IP() scenarios: shared prefix + unique suffix + non-repeating images."""
        image_content = self._sample_images(image_dimension, num_images)

        # Get or create shared prefix
        cache_key = f"prefix_{prefix_tokens}"
        if cache_key not in self._shared_prefix_cache:
            prefix = self._sample_text(prefix_tokens)
            self._shared_prefix_cache[cache_key] = prefix
            logger.info(
                f"Generated shared prefix ({prefix_tokens} tokens) for VLM KV cache "
                f"benchmarking. All subsequent requests will reuse this prefix."
            )
        else:
            prefix = self._shared_prefix_cache[cache_key]

        # Generate unique suffix with separator
        self._suffix_counter += 1
        separator = f"\n\n--- Request #{self._suffix_counter} ---\n\n"
        separator_tokens = self.get_token_length(separator)
        adjusted_suffix_len = max(1, suffix_tokens - separator_tokens)
        suffix = self._sample_text(adjusted_suffix_len)

        prompt = f"{prefix}{separator}{suffix}"
        num_prefill_tokens = self.get_token_length(prompt)

        expected_tokens = prefix_tokens + suffix_tokens
        self._check_discrepancy(expected_tokens, num_prefill_tokens, threshold=0.05)

        # Set output token control
        self.additional_request_params["ignore_eos"] = True
        if not self.no_min_tokens:
            self.additional_request_params["min_tokens"] = output_tokens
        self.additional_request_params["max_tokens"] = output_tokens

        if self.output_modality == "text":
            return UserImageChatRequest(
                model=self.model,
                prompt=prompt,
                image_content=image_content,
                num_images=num_images,
                max_tokens=output_tokens,
                num_prefill_tokens=num_prefill_tokens,
                additional_request_params=self.additional_request_params,
            )
        elif self.output_modality == "embeddings":
            return self._generate_image_embedding_request(image_content, num_images)
        else:
            raise ValueError(f"Unsupported output modality: {self.output_modality}")

    def _sample_legacy_request(
        self,
        image_dimension: Optional[Tuple[int, int]],
        num_images: Optional[int],
        num_output_tokens: Optional[int],
    ) -> UserRequest:
        """Handle legacy I() scenarios and dataset mode (backward compatible)."""
        if num_images is None:
            num_images = 1
        prompt, image_content = self._sample_image_and_text(image_dimension, num_images)

        if self.output_modality == "text":
            return self._generate_image_chat_request(
                prompt, image_content, num_images, num_output_tokens
            )
        elif self.output_modality == "embeddings":
            return self._generate_image_embedding_request(image_content, num_images)
        else:
            raise ValueError(f"Unsupported output modality: {self.output_modality}")

    def _generate_image_chat_request(
        self,
        prompt: str,
        image_content: List[str],
        num_images: int,
        num_output_tokens: int | None,
    ) -> UserImageChatRequest:
        """
        Generates a `UserImageChatRequest` for legacy I() / dataset mode.
        """
        return UserImageChatRequest(
            model=self.model,
            prompt=prompt,
            image_content=image_content,
            num_images=num_images,
            max_tokens=num_output_tokens,
            num_prefill_tokens=None,
            additional_request_params=self.additional_request_params,
        )

    def _generate_image_embedding_request(
        self, image_content: List[str], num_images: int
    ) -> UserImageEmbeddingRequest:
        """
        Generates a `UserImageEmbeddingRequest` for image-to-embedding tasks.
        """
        return UserImageEmbeddingRequest(
            model=self.model,
            documents=[],
            image_content=image_content,
            num_images=num_images,
            num_prefill_tokens=None,
            additional_request_params=self.additional_request_params,
        )

    def _validate_scenario(self, scenario: Scenario) -> None:
        """Validates that a scenario has the correct type."""
        if not isinstance(scenario.scenario_type, MultiModality):
            raise ValueError(
                f"Expected MultiModality for image tasks, got "
                f"{type(scenario.scenario_type)}"
            )

    # --- Text generation (for ID/IP scenarios) ---

    def _sample_text(self, num_input_tokens: int) -> str:
        """Generate text with exact token count using sonnet.txt corpus.

        Mirrors TextSampler._sample_text() logic.
        """
        data_copy = self._text_data.copy()
        prompt = ""
        left_tokens_to_sample = num_input_tokens

        while left_tokens_to_sample > 0:
            random.shuffle(data_copy)
            for line in data_copy:
                line_with_space = (" " if prompt else "") + line
                line_tokens = self.tokenizer.encode(
                    line_with_space, add_special_tokens=False
                )
                num_line_tokens = len(line_tokens)

                if num_line_tokens > left_tokens_to_sample:
                    truncated_text = self.tokenizer.decode(
                        line_tokens[:left_tokens_to_sample], skip_special_tokens=True
                    )
                    prompt += (" " if prompt else "") + truncated_text
                    actual_tokens = len(
                        self.tokenizer.encode(prompt, add_special_tokens=False)
                    )
                    if actual_tokens != num_input_tokens:
                        prompt_tokens = self.tokenizer.encode(
                            prompt, add_special_tokens=False
                        )
                        prompt = self.tokenizer.decode(
                            prompt_tokens[:num_input_tokens], skip_special_tokens=True
                        )
                    return prompt

                prompt += (" " if prompt else "") + line
                left_tokens_to_sample -= num_line_tokens
        return prompt

    def _check_discrepancy(
        self, num_input_tokens: int, num_prefill_tokens: int, threshold: float = 0.1
    ) -> None:
        """Log warning if actual token count diverges from expected."""
        discrepancy = abs(num_input_tokens - num_prefill_tokens)
        if discrepancy > threshold * num_input_tokens:
            logger.warning(
                f"Sampling discrepancy detected: "
                f"num_input_tokens={num_input_tokens}, "
                f"num_prefill_tokens={num_prefill_tokens}, "
                f"discrepancy={discrepancy}"
            )

    # --- Image sampling ---

    def _next_image_index(self) -> int:
        """Return next image index using shuffle-without-replacement.

        All images are used before any repeats. Reshuffles when exhausted.
        """
        if self._image_index_pos >= len(self._image_indices):
            self._image_indices = list(range(len(self.data)))
            random.shuffle(self._image_indices)
            self._image_index_pos = 0
        idx = self._image_indices[self._image_index_pos]
        self._image_index_pos += 1
        return idx

    def _sample_images(
        self,
        image_dimension: Optional[Tuple[int, int]],
        num_images: int,
    ) -> List[str]:
        """Sample images using shuffle-without-replacement (for ID/IP scenarios)."""
        images: List[str] = []
        for _ in range(num_images):
            idx = self._next_image_index()
            item = self.data[idx]

            raw_image: Any = None
            if isinstance(item, tuple) and len(item) == 2:
                _, raw_image = item
            elif isinstance(item, dict) and self.dataset_config is not None:
                cfg = self.dataset_config
                if cfg.image_column:
                    raw_image = item.get(cfg.image_column)
            else:
                continue

            if raw_image is None:
                continue
            processed_image = ImageSampler.process_image(
                raw_image, resize=image_dimension
            )
            images.append(processed_image)
        return images

    # --- Legacy sampling (for I() and dataset mode) ---

    def _sample_image_and_text(
        self, image_dimension: Optional[Tuple[int, int]] = None, num_images: int = 1
    ) -> Tuple[str, List[str]]:
        """
        Legacy: sample images AND text from the dataset together.
        Used by I() scenarios and dataset mode.
        """
        images: List[str] = []
        texts: List[str] = []

        chosen = random.choices(self.data, k=num_images)
        for item in chosen:
            prompt: str = ""
            raw_image: Any = None
            # Backward-compatible format
            if isinstance(item, tuple) and len(item) == 2:
                prompt, raw_image = item
            # Dict row format
            elif isinstance(item, dict) and self.dataset_config is not None:
                cfg = self.dataset_config
                if cfg.image_column:
                    raw_image = item.get(cfg.image_column)
                if cfg.prompt_lambda:
                    prompt = safe_eval_prompt(cfg.prompt_lambda, item)
                elif cfg.prompt_column:
                    prompt = str(item.get(cfg.prompt_column, ""))
            else:
                continue

            if raw_image is None:
                continue
            processed_image = ImageSampler.process_image(
                raw_image, resize=image_dimension
            )
            images.append(processed_image)
            texts.append(prompt or "")

        return " ".join(texts), images

    # --- Prefix cache management ---

    def reset_prefix_cache(self):
        """Clear the prefix cache and reset counter between scenario runs."""
        if self._suffix_counter > 0:
            logger.info(
                f"Resetting prefix cache. Previous scenario generated "
                f"{self._suffix_counter} requests with "
                f"{len(self._shared_prefix_cache)} cached prefix(es)."
            )
        self._shared_prefix_cache.clear()
        self._suffix_counter = 0

    @staticmethod
    def process_image(image: Any, resize: Optional[Tuple[int, int]] = None) -> str:
        """
        Process a single image input and return a data URL or HTTP(S) URL.

        Supports three input types:
        1. Dictionary with raw image bytes
        2. PIL.Image.Image input
        3. String input (URL or file path)
        """
        if isinstance(image, dict) and "bytes" in image:
            image = PIL.Image.open(BytesIO(image["bytes"]))

        if isinstance(image, Image):
            if image.mode != "RGB":
                image = image.convert("RGB")
            if resize:
                image = image.resize(resize, PIL.Image.Resampling.LANCZOS)
            with BytesIO() as image_data:
                image.save(image_data, format="JPEG")
                image_base64 = base64.b64encode(image_data.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{image_base64}"

        if isinstance(image, str) and image.startswith(("http://", "https://")):
            return image

        raise ValueError(
            f"Invalid image input {image}. Must be a PIL.Image.Image"
            " or str or dictionary with raw image bytes."
        )
