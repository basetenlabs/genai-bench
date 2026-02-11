"""Tests for message list support in text sampling."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from genai_bench.data.config import DatasetConfig
from genai_bench.data.loaders.text import TextDatasetLoader
from genai_bench.protocol import UserChatMessagesRequest, UserChatRequest
from genai_bench.sampling.text import TextSampler


# -- Fixtures --


@pytest.fixture
def sample_message_lists():
    """Reusable message list data."""
    return [
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ],
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "How are you?"},
        ],
    ]


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer that returns predictable token counts."""
    tokenizer = Mock()
    tokenizer.encode.return_value = [1, 2, 3, 4, 5]
    return tokenizer


@pytest.fixture
def message_sampler(sample_message_lists, mock_tokenizer):
    """TextSampler configured for message lists."""
    return TextSampler(
        tokenizer=mock_tokenizer,
        model="test-model",
        output_modality="text",
        data=sample_message_lists,
        dataset_config=DatasetConfig.from_cli_args(message_format="openai"),
    )


# -- DatasetConfig Tests --


class TestDatasetConfigMessageFormat:
    """Test DatasetConfig with message_format."""

    def test_config_with_message_format(self):
        config = DatasetConfig.from_cli_args(
            dataset_path="test.json", message_format="openai"
        )
        assert config.message_format == "openai"
        assert config.source.type == "file"
        assert config.source.file_format == "json"

    def test_config_without_message_format(self):
        config = DatasetConfig.from_cli_args(dataset_path="test.json")
        assert config.message_format is None

    def test_config_huggingface_with_slash(self):
        """HuggingFace IDs with slashes should be detected correctly."""
        config = DatasetConfig.from_cli_args(
            dataset_path="baseten/gamma_paste_text_benchmarking",
            message_format="openai",
            prompt_column="messages",
        )
        assert config.source.type == "huggingface"
        assert config.message_format == "openai"
        assert config.prompt_column == "messages"
        assert config.source.huggingface_kwargs == {"split": "train"}


# -- TextDatasetLoader Tests --


class TestTextDatasetLoaderMessages:
    """Test TextDatasetLoader with message list data."""

    def test_load_message_list_from_json(self, sample_message_lists):
        """Test loading message lists from a JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_message_lists, f)
            temp_path = f.name

        try:
            config = DatasetConfig.from_cli_args(
                dataset_path=temp_path, message_format="openai"
            )
            loader = TextDatasetLoader(config)
            data = loader.load_request()

            assert len(data) == 2
            assert isinstance(data[0], list)
            assert data[0][0]["role"] == "system"
        finally:
            Path(temp_path).unlink()

    def test_process_dict_data_with_message_lists(self, sample_message_lists):
        """Test _process_loaded_data with dict-shaped data (like HuggingFace)."""
        config = DatasetConfig.from_cli_args(
            message_format="openai", prompt_column="messages"
        )
        loader = TextDatasetLoader(config)
        dict_data = {"messages": sample_message_lists}

        result = loader._process_loaded_data(dict_data)
        assert len(result) == 2
        assert isinstance(result[0], list)

    def test_process_list_of_lists_with_message_format(self, sample_message_lists):
        """Test _process_loaded_data with a flat list of message lists."""
        config = DatasetConfig.from_cli_args(message_format="openai")
        loader = TextDatasetLoader(config)

        result = loader._process_loaded_data(sample_message_lists)
        assert len(result) == 2
        assert isinstance(result[0], list)

    def test_process_string_data_unchanged(self):
        """Test that string data still works with message_format set."""
        config = DatasetConfig.from_cli_args(message_format="openai")
        loader = TextDatasetLoader(config)
        string_data = ["Hello", "World"]

        result = loader._process_loaded_data(string_data)
        assert result == ["Hello", "World"]

    def test_validate_message_lists_valid(self):
        """Test validation passes for well-formed message lists."""
        loader = TextDatasetLoader(DatasetConfig.from_cli_args(message_format="openai"))
        messages = [
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello!"},
            ]
        ]
        result = loader._validate_message_lists(messages)
        assert len(result) == 1

    def test_validate_message_lists_missing_role(self):
        """Test validation fails when role is missing."""
        loader = TextDatasetLoader(DatasetConfig.from_cli_args(message_format="openai"))
        with pytest.raises(ValueError, match="missing required 'role' field"):
            loader._validate_message_lists([[{"content": "No role"}]])

    def test_validate_message_lists_missing_content(self):
        """Test validation fails when content is missing."""
        loader = TextDatasetLoader(DatasetConfig.from_cli_args(message_format="openai"))
        with pytest.raises(ValueError, match="missing required 'content' field"):
            loader._validate_message_lists([[{"role": "user"}]])

    def test_validate_message_lists_invalid_role(self):
        """Test validation fails for unsupported roles."""
        loader = TextDatasetLoader(DatasetConfig.from_cli_args(message_format="openai"))
        with pytest.raises(ValueError, match="Invalid message role 'admin'"):
            loader._validate_message_lists([[{"role": "admin", "content": "Bad role"}]])

    def test_validate_message_lists_empty_content(self):
        """Test validation fails for empty/whitespace content."""
        loader = TextDatasetLoader(DatasetConfig.from_cli_args(message_format="openai"))
        with pytest.raises(ValueError, match="cannot be empty or whitespace only"):
            loader._validate_message_lists([[{"role": "user", "content": "   "}]])

    def test_validate_message_lists_empty_list(self):
        """Test validation fails for empty message list."""
        loader = TextDatasetLoader(DatasetConfig.from_cli_args(message_format="openai"))
        with pytest.raises(ValueError, match="column is empty"):
            loader._validate_message_lists([])

    def test_validate_message_lists_empty_inner_list(self):
        """Test validation fails for empty inner message list."""
        loader = TextDatasetLoader(DatasetConfig.from_cli_args(message_format="openai"))
        with pytest.raises(ValueError, match="is empty"):
            loader._validate_message_lists([[]])

    def test_validate_message_lists_non_dict_message(self):
        """Test validation fails when message is not a dict."""
        loader = TextDatasetLoader(DatasetConfig.from_cli_args(message_format="openai"))
        with pytest.raises(ValueError, match="is not a dictionary"):
            loader._validate_message_lists([["not a dict"]])

    def test_validate_message_lists_non_list_inner(self):
        """Test validation fails when inner item is not a list."""
        loader = TextDatasetLoader(DatasetConfig.from_cli_args(message_format="openai"))
        with pytest.raises(ValueError, match="is not a list"):
            loader._validate_message_lists(["not a list"])

    def test_validate_message_lists_non_string_content(self):
        """Test validation fails when content is not a string."""
        loader = TextDatasetLoader(DatasetConfig.from_cli_args(message_format="openai"))
        with pytest.raises(ValueError, match="must be a string"):
            loader._validate_message_lists([[{"role": "user", "content": 123}]])

    def test_validate_message_lists_all_roles(self):
        """Test validation accepts all valid roles."""
        loader = TextDatasetLoader(DatasetConfig.from_cli_args(message_format="openai"))
        messages = [
            [
                {"role": "system", "content": "System prompt."},
                {"role": "user", "content": "User message."},
                {"role": "assistant", "content": "Assistant response."},
            ]
        ]
        result = loader._validate_message_lists(messages)
        assert len(result) == 1
        assert len(result[0]) == 3


# -- TextSampler Tests --


class TestTextSamplerMessageLists:
    """Test TextSampler with message list data."""

    def test_has_message_lists_true(self, message_sampler):
        assert message_sampler._has_message_lists()

    def test_has_message_lists_false_with_strings(self, mock_tokenizer):
        sampler = TextSampler(
            tokenizer=mock_tokenizer,
            model="test-model",
            output_modality="text",
            data=["Hello", "World"],
            dataset_config=DatasetConfig.from_cli_args(),
        )
        assert not sampler._has_message_lists()

    def test_has_message_lists_false_with_empty(self, mock_tokenizer):
        sampler = TextSampler(
            tokenizer=mock_tokenizer,
            model="test-model",
            output_modality="text",
            data=[],
            dataset_config=DatasetConfig.from_cli_args(),
        )
        assert not sampler._has_message_lists()

    def test_sample_returns_message_request(self, message_sampler):
        request = message_sampler.sample(scenario=None)
        assert isinstance(request, UserChatMessagesRequest)

    def test_sample_message_request_fields(self, message_sampler):
        request = message_sampler.sample(scenario=None)
        assert request.model == "test-model"
        assert request.messages is not None
        assert len(request.messages) >= 1
        assert all("role" in msg for msg in request.messages)
        assert all("content" in msg for msg in request.messages)

    def test_sample_message_request_num_prefill_tokens(self, message_sampler):
        request = message_sampler.sample(scenario=None)
        assert request.num_prefill_tokens is not None
        assert request.num_prefill_tokens > 0

    def test_sample_message_request_ignore_eos_default(self, message_sampler):
        """In dataset mode (no scenario), ignore_eos should be False."""
        request = message_sampler.sample(scenario=None)
        assert request.additional_request_params.get("ignore_eos") is False

    def test_count_message_tokens_with_tokenizer(self, mock_tokenizer):
        sampler = TextSampler(
            tokenizer=mock_tokenizer,
            model="test-model",
            output_modality="text",
            data=[],
            dataset_config=DatasetConfig.from_cli_args(message_format="openai"),
        )

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]

        count = sampler._count_message_tokens(messages)
        assert count > 0
        assert isinstance(count, int)

    def test_count_message_tokens_without_tokenizer(self):
        """Test fallback char-based estimation when no tokenizer."""
        sampler = TextSampler(
            tokenizer=None,
            model="test-model",
            output_modality="text",
            data=[],
            dataset_config=DatasetConfig.from_cli_args(message_format="openai"),
        )

        messages = [
            {"role": "user", "content": "Hello world, this is a test."},
        ]

        count = sampler._count_message_tokens(messages)
        assert count > 0

    def test_count_message_tokens_adds_assistant_starter(self, mock_tokenizer):
        """Token count should include assistant turn starter when last msg is user."""
        sampler = TextSampler(
            tokenizer=mock_tokenizer,
            model="test-model",
            output_modality="text",
            data=[],
            dataset_config=DatasetConfig.from_cli_args(message_format="openai"),
        )

        # Last message is user - should add assistant starter tokens
        messages_user = [{"role": "user", "content": "Hello!"}]
        count_user = sampler._count_message_tokens(messages_user)

        # Last message is assistant - should NOT add assistant starter tokens
        messages_asst = [{"role": "assistant", "content": "Hi there!"}]
        count_asst = sampler._count_message_tokens(messages_asst)

        # Both should be positive
        assert count_user > 0
        assert count_asst > 0

    def test_backward_compatibility_string_data(self, mock_tokenizer):
        """String-based prompts should still produce UserChatRequest."""
        mock_tokenizer.__len__ = Mock(return_value=3)
        sampler = TextSampler(
            tokenizer=mock_tokenizer,
            model="test-model",
            output_modality="text",
            data=["Hello world", "How are you?"],
            dataset_config=DatasetConfig.from_cli_args(),
        )

        request = sampler.sample(scenario=None)
        assert isinstance(request, UserChatRequest)
        assert not isinstance(request, UserChatMessagesRequest)
        assert isinstance(request.prompt, str)


# -- UserChatMessagesRequest Protocol Tests --


class TestUserChatMessagesRequest:
    """Test the UserChatMessagesRequest protocol model."""

    def test_create_valid_request(self):
        request = UserChatMessagesRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            num_prefill_tokens=5,
            max_tokens=10,
        )
        assert request.model == "test-model"
        assert request.messages == [{"role": "user", "content": "Hello"}]
        assert request.prompt is None  # Default for message requests
        assert request.num_prefill_tokens == 5

    def test_inherits_from_user_chat_request(self):
        request = UserChatMessagesRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            num_prefill_tokens=5,
            max_tokens=10,
        )
        assert isinstance(request, UserChatRequest)

    def test_prompt_is_optional(self):
        request = UserChatMessagesRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            num_prefill_tokens=5,
            max_tokens=10,
        )
        assert request.prompt is None

    def test_multi_turn_messages(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
        ]
        request = UserChatMessagesRequest(
            model="test-model",
            messages=messages,
            num_prefill_tokens=20,
            max_tokens=10,
        )
        assert len(request.messages) == 4


# -- OpenAI User Tests --


class TestOpenAIUserMessageLists:
    """Test OpenAI user with message list requests."""

    @patch("genai_bench.user.openai_user.requests.post")
    def test_chat_with_message_list_request(self, mock_post):
        """Test that OpenAIUser.chat sends messages correctly."""
        from genai_bench.user.openai_user import OpenAIUser

        mock_auth = MagicMock()
        mock_auth.get_credentials.return_value = "fake_api_key"
        mock_auth.get_headers.return_value = {"Authorization": "Bearer fake_api_key"}
        mock_auth.get_config.return_value = {
            "api_base": "http://example.com",
            "api_key": "fake_api_key",
        }
        OpenAIUser.auth_provider = mock_auth
        OpenAIUser.host = "http://example.com"

        user = OpenAIUser(environment=MagicMock())
        user.on_start()

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]
        user.sample = lambda: UserChatMessagesRequest(
            model="gpt-4",
            messages=messages,
            num_prefill_tokens=10,
            max_tokens=20,
        )

        # Mock response
        response_mock = MagicMock()
        response_mock.status_code = 200
        response_mock.iter_lines = MagicMock(
            return_value=[
                b'data: {"id":"chat-1","object":"chat.completion.chunk","created":1,"model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}',
                b'data: {"id":"chat-1","object":"chat.completion.chunk","created":1,"model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":1,"total_tokens":11}}',
                b"data: [DONE]",
            ]
        )
        mock_post.return_value = response_mock

        user.chat()

        # Verify the messages were passed directly (not wrapped in a user message)
        call_args = mock_post.call_args
        payload = call_args.kwargs["json"]
        assert payload["messages"] == messages
        assert payload["model"] == "gpt-4"

    @patch("genai_bench.user.openai_user.requests.post")
    def test_chat_accepts_messages_request_type(self, mock_post):
        """Test that chat() accepts UserChatMessagesRequest without error."""
        from genai_bench.user.openai_user import OpenAIUser

        mock_auth = MagicMock()
        mock_auth.get_credentials.return_value = "fake_key"
        mock_auth.get_headers.return_value = {"Authorization": "Bearer fake_key"}
        mock_auth.get_config.return_value = {
            "api_base": "http://example.com",
            "api_key": "fake_key",
        }
        OpenAIUser.auth_provider = mock_auth
        OpenAIUser.host = "http://example.com"

        user = OpenAIUser(environment=MagicMock())
        user.on_start()

        user.sample = lambda: UserChatMessagesRequest(
            model="gpt-4",
            messages=[{"role": "user", "content": "Test"}],
            num_prefill_tokens=5,
            max_tokens=10,
        )

        response_mock = MagicMock()
        response_mock.status_code = 200
        response_mock.iter_lines = MagicMock(
            return_value=[
                b'data: {"id":"chat-1","choices":[{"delta":{"content":"OK"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}',
                b"data: [DONE]",
            ]
        )
        mock_post.return_value = response_mock

        # Should not raise
        user.chat()
        mock_post.assert_called_once()


# -- Async Runner Tests --


class TestAsyncRunnerMessageLists:
    """Test BaseAsyncRunner with message list requests."""

    def test_prepare_request_dataset_scenario(self):
        """Test that 'dataset' scenario string passes None to sampler."""
        from genai_bench.async_runner.base import BaseAsyncRunner

        mock_sampler = MagicMock()
        mock_sampler.sample = MagicMock(
            return_value=UserChatMessagesRequest(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
                num_prefill_tokens=10,
                max_tokens=20,
            )
        )

        class TestRunner(BaseAsyncRunner):
            async def run(self, *, duration_s: int, scenario: str, **kwargs):
                return 1.0

        mock_auth = MagicMock()
        mock_auth.get_headers = MagicMock(
            return_value={
                "Authorization": "Bearer test",
                "Content-Type": "application/json",
            }
        )

        runner = TestRunner(
            sampler=mock_sampler,
            api_backend="openai",
            api_base="https://api.openai.com",
            api_model_name="gpt-4",
            auth_provider=mock_auth,
            aggregated_metrics_collector=MagicMock(),
            dashboard=None,
        )

        req = runner._prepare_request("dataset")

        # Sampler should be called with None (dataset mode)
        mock_sampler.sample.assert_called_once_with(None)
        assert isinstance(req, UserChatMessagesRequest)

    def test_prepare_request_validates_messages(self):
        """Test that prepare_request validates messages field."""
        from genai_bench.async_runner.base import BaseAsyncRunner

        mock_sampler = MagicMock()
        mock_sampler.sample = MagicMock(
            return_value=UserChatMessagesRequest(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hello"}],
                num_prefill_tokens=10,
                max_tokens=20,
            )
        )

        class TestRunner(BaseAsyncRunner):
            async def run(self, *, duration_s: int, scenario: str, **kwargs):
                return 1.0

        mock_auth = MagicMock()
        mock_auth.get_headers = MagicMock(
            return_value={
                "Authorization": "Bearer test",
                "Content-Type": "application/json",
            }
        )

        runner = TestRunner(
            sampler=mock_sampler,
            api_backend="openai",
            api_base="https://api.openai.com",
            api_model_name="gpt-4",
            auth_provider=mock_auth,
            aggregated_metrics_collector=MagicMock(),
            dashboard=None,
        )

        # Valid request - should not raise
        req = runner._prepare_request("dataset")
        assert isinstance(req, UserChatMessagesRequest)

    def test_prepare_request_rejects_empty_messages(self):
        """Test that prepare_request rejects requests with empty messages."""
        from genai_bench.async_runner.base import BaseAsyncRunner

        mock_sampler = MagicMock()
        mock_sampler.sample = MagicMock(
            return_value=UserChatMessagesRequest(
                model="gpt-4",
                messages=[],
                num_prefill_tokens=10,
                max_tokens=20,
            )
        )

        class TestRunner(BaseAsyncRunner):
            async def run(self, *, duration_s: int, scenario: str, **kwargs):
                return 1.0

        mock_auth = MagicMock()
        mock_auth.get_headers = MagicMock(
            return_value={
                "Authorization": "Bearer test",
                "Content-Type": "application/json",
            }
        )

        runner = TestRunner(
            sampler=mock_sampler,
            api_backend="openai",
            api_base="https://api.openai.com",
            api_model_name="gpt-4",
            auth_provider=mock_auth,
            aggregated_metrics_collector=MagicMock(),
            dashboard=None,
        )

        with pytest.raises(ValueError, match="missing required 'messages' field"):
            runner._prepare_request("dataset")
