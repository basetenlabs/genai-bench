"""Dataset configuration models for flexible dataset loading."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, field_validator


class DatasetSourceConfig(BaseModel):
    """Configuration for a dataset source.

    Supports multiple dataset source types:
    - file: Local files (txt, csv, json)
    - huggingface: HuggingFace Hub datasets
    - custom: Custom dataset loaders
    """

    type: str = Field(
        ..., description="Dataset source type: 'file', 'huggingface', or 'custom'"
    )
    path: Optional[str] = Field(
        None, description="Path to dataset (file path or HuggingFace ID)"
    )

    # For file sources
    file_format: Optional[str] = Field(
        None, description="File format: 'csv', 'txt', 'json'"
    )

    # For HuggingFace sources - accepts ANY parameter that load_dataset supports
    huggingface_kwargs: Optional[Dict[str, Any]] = Field(
        None,
        description="Keyword arguments passed directly to HuggingFace load_dataset",
    )

    # For custom sources
    loader_class: Optional[str] = Field(
        None, description="Python import path for custom dataset loader"
    )
    loader_kwargs: Optional[Dict[str, Any]] = Field(
        None, description="Keyword arguments for custom loader"
    )

    @field_validator("type")
    def validate_type(cls, v):
        valid_types = {"file", "huggingface", "custom"}
        if v not in valid_types:
            raise ValueError(f"Dataset source type must be one of {valid_types}")
        return v


class DatasetConfig(BaseModel):
    """Complete dataset configuration."""

    source: DatasetSourceConfig
    prompt_column: Optional[str] = Field(
        None, description="Column name containing prompts"
    )
    image_column: Optional[str] = Field(
        None, description="Column name containing images"
    )
    prompt_lambda: Optional[str] = Field(
        None,
        description="Lambda expression string, "
        'e.g. \'lambda item: f"Question: {item["question"]}"\'',
    )
    messages_column: Optional[str] = Field(
        None,
        description="Column name containing conversation messages array "
        "(for RD/RDC scenarios). Default: 'messages'",
    )
    unsafe_allow_large_images: bool = Field(
        False,
        description="Overrides pillows internal DDOS protection",
    )

    @classmethod
    def from_file(cls, config_path: str) -> "DatasetConfig":
        """Load configuration from a JSON file."""
        with open(config_path, "r") as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_cli_args(
        cls,
        dataset_path: Optional[str] = None,
        prompt_column: Optional[str] = None,
        image_column: Optional[str] = None,
        **kwargs,
    ) -> "DatasetConfig":
        """Create configuration from CLI arguments for backward compatibility."""
        if dataset_path is None:
            # Default to built-in sonnet.txt
            dataset_path = str(Path(__file__).parent / "sonnet.txt")
            source_type = "file"
            file_format = "txt"
        else:
            # Determine source type from path
            path = Path(dataset_path)
            # If path has a file extension, treat it as a file (even if it doesn't exist yet)
            # This prevents local files from being incorrectly treated as HuggingFace datasets
            supported_extensions = {".csv", ".txt", ".json", ".jsonl"}
            if path.suffix.lower() in supported_extensions:
                source_type = "file"
                file_format = path.suffix.lower().lstrip(".")
            elif path.exists():
                # Path exists but doesn't have a recognized extension
                raise ValueError(
                    f"Unsupported file format: {path.suffix}. "
                    f"Supported formats: {', '.join(supported_extensions)}"
                )
            else:
                # No file extension and doesn't exist - assume it's a HuggingFace ID
                source_type = "huggingface"
                file_format = None

        source_config = DatasetSourceConfig(
            type=source_type,
            path=dataset_path,
            file_format=file_format,
            huggingface_kwargs=None,
            loader_class=None,
            loader_kwargs=None,
        )

        return cls(
            source=source_config,
            prompt_column=prompt_column,
            image_column=image_column,
            prompt_lambda=None,
            messages_column=None,
            unsafe_allow_large_images=False,
        )

    @classmethod
    def default_image_config(cls) -> "DatasetConfig":
        """Return config for built-in COCO image dataset (~5K diverse images).

        Uses sayakpaul/coco-30-val-2014 in parquet format, which supports
        split slicing so only ~800MB is downloaded (not the full 20GB COCO).

        Used as the default image source for VLM benchmarks when no
        --dataset-config or --dataset-path is provided.
        """
        return cls(
            source=DatasetSourceConfig(
                type="huggingface",
                path="sayakpaul/coco-30-val-2014",
                file_format=None,
                huggingface_kwargs={"split": "train[:5000]"},
                loader_class=None,
                loader_kwargs=None,
            ),
            prompt_column=None,
            image_column="image",
            prompt_lambda=None,
            messages_column=None,
            unsafe_allow_large_images=False,
        )
