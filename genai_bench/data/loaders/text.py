from typing import Any, Dict, List, Set, Union

from genai_bench.data.loaders.base import DatasetFormat, DatasetLoader
from genai_bench.logging import init_logger

logger = init_logger(__name__)


class TextDatasetLoader(DatasetLoader):
    """
    This datasetLoader is responsible for loading prompts from a data source.

    Supports plain text datasets (List[str]) and JSONL messages datasets
    (List[List[Dict]]) where each item is a pre-formatted messages array
    in OpenAI chat format: [{"role": "system", "content": "..."}, ...].

    TODO: Add support for prompt lambdas similar to ImageDatasetLoader.
    """

    supported_formats: Set[DatasetFormat] = {
        DatasetFormat.TEXT,
        DatasetFormat.CSV,
        DatasetFormat.JSON,
        DatasetFormat.JSONL,
        DatasetFormat.HUGGINGFACE_HUB,
    }
    media_type = "Text"

    def _process_loaded_data(self, data: Any) -> Union[List[str], List[List[Dict]]]:
        """Process data loaded from dataset source.

        Returns either List[str] for plain-text datasets or List[List[Dict]]
        for JSONL messages datasets.
        """
        # Handle data from dataset sources
        if isinstance(data, list):
            if not data:
                return data
            # Detect JSONL messages format: each item is a dict with a "messages" key
            # whose value is a list of role/content dicts.
            first = data[0]
            if (
                isinstance(first, dict)
                and "messages" in first
                and isinstance(first["messages"], list)
            ):
                messages_list = [item["messages"] for item in data]
                logger.info(
                    f"Detected JSONL messages dataset with {len(messages_list)} entries"
                )
                return messages_list
            return data

        # Handle dictionary data (from CSV files) or HuggingFace datasets
        prompt_column = self.dataset_config.prompt_column
        try:
            column_data = data[prompt_column]
            # Ensure we return a list of strings
            if isinstance(column_data, list):
                return [str(item) for item in column_data]
            else:
                # For HuggingFace datasets, convert to list
                return list(column_data)
        except (ValueError, KeyError) as e:
            # Provide helpful error message with available columns
            if isinstance(data, dict):
                available_columns = list(data.keys())
                raise ValueError(
                    f"Column '{prompt_column}' not found in CSV file. "
                    f"Available columns: {available_columns}"
                ) from e
            else:
                raise ValueError(
                    f"Cannot extract prompts from data: {type(data)}, error: {str(e)}"
                ) from e
