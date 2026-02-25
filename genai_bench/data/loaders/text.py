from typing import Any, List, Set, Tuple, Union

from genai_bench.data.loaders.base import DatasetFormat, DatasetLoader
from genai_bench.logging import init_logger

logger = init_logger(__name__)


class TextDatasetLoader(DatasetLoader):
    """
    This datasetLoader is responsible for loading prompts from a data source.

    TODO: Add support for prompt lambdas similar to ImageDatasetLoader.
    """

    supported_formats: Set[DatasetFormat] = {
        DatasetFormat.TEXT,
        DatasetFormat.CSV,
        DatasetFormat.JSON,
        DatasetFormat.HUGGINGFACE_HUB,
    }
    media_type = "Text"

    def _process_loaded_data(
        self, data: Any
    ) -> Union[List[str], List[Tuple[str, Any]], List[List[dict]]]:
        """Process data loaded from dataset source."""
        # Handle data from dataset sources
        if isinstance(data, list):
            # Check if this is a list of message lists
            if (
                data
                and isinstance(data[0], list)
                and self.dataset_config.message_format == "openai"
            ):
                # Return raw message lists for TextSampler
                return data
            return data  # Return as-is for string lists

        # Handle dictionary data (from CSV files) or HuggingFace datasets
        prompt_column = self.dataset_config.prompt_column
        message_format = self.dataset_config.message_format

        try:
            column_data = data[prompt_column]

            # Check if we have message lists and message_format is specified
            if (
                message_format == "openai"
                and isinstance(column_data, list)
                and column_data
                and isinstance(column_data[0], list)
            ):
                # We have message lists
                logger.info(f"Detected message lists in column '{prompt_column}'")
                validated = self._validate_message_lists(column_data)
                # Return raw message lists for TextSampler
                return validated

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

    def _validate_message_lists(
        self, message_lists: List[List[dict]]
    ) -> List[List[dict]]:
        """Validate and clean message lists in OpenAI format.

        Strips None-valued fields and validates required fields.
        Supports tool/function calling messages.
        """
        if not message_lists:
            raise ValueError(
                "Message lists column is empty. Please check your dataset."
            )

        VALID_ROLES = {"system", "user", "assistant", "tool", "function"}

        validated = []
        for i, messages in enumerate(message_lists):
            if not isinstance(messages, list):
                raise ValueError(
                    f"Message list at index {i} is not a list (found {type(messages).__name__}). "
                    f"Expected a list of message dictionaries."
                )

            if not messages:
                raise ValueError(
                    f"Message list at index {i} is empty. Each message list must contain at least one message."
                )

            cleaned_messages = []
            for j, message in enumerate(messages):
                if not isinstance(message, dict):
                    raise ValueError(
                        f"Message at index {i}[{j}] is not a dictionary (found {type(message).__name__}). "
                        f"Expected a dict with 'role' and 'content' fields."
                    )

                if "role" not in message:
                    raise ValueError(
                        f"Message at index {i}[{j}] is missing required 'role' field. "
                        f"Available fields: {list(message.keys())}"
                    )

                role = message["role"]
                if role not in VALID_ROLES:
                    raise ValueError(
                        f"Invalid message role '{role}' at index {i}[{j}]. "
                        f"Supported roles: {VALID_ROLES}"
                    )

                # Strip None-valued fields to avoid API validation errors
                cleaned = {k: v for k, v in message.items() if v is not None}
                cleaned_messages.append(cleaned)

            validated.append(cleaned_messages)

        return validated
