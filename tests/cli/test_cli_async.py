"""
Tests for async execution engine CLI parameters.

These tests verify that:
- --execution-engine async works correctly
- --qps-level accepts single and multiple values
- QPS values are stored correctly as decimals (not scaled)
"""

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from genai_bench.cli.cli import benchmark


@pytest.fixture
def cli_runner():
    return CliRunner()


@pytest.fixture
def async_options():
    """Base options for async runner tests"""
    return [
        "--api-backend",
        "openai",
        "--api-base",
        "https://api.openai.com",
        "--api-key",
        "test_key",
        "--task",
        "text-to-text",
        "--api-model-name",
        "gpt-3.5-turbo",
        "--model-tokenizer",
        "gpt2",
        "--max-time-per-run",
        "1",
        "--max-requests-per-run",
        "5",
        "--execution-engine",
        "async",
    ]


@pytest.fixture
def mock_env_variables():
    with patch.dict("os.environ", {"HF_TOKEN": "dummy_key"}):
        yield


@pytest.fixture
def mock_dashboard():
    with patch("genai_bench.cli.cli.create_dashboard") as mock_dashboard_patch:
        yield mock_dashboard_patch


@pytest.fixture
def mock_validate_tokenizer():
    mock_tokenizer = MagicMock()
    with patch("genai_bench.cli.cli.validate_tokenizer", return_value=mock_tokenizer):
        yield mock_tokenizer


@pytest.fixture
def mock_time_sleep():
    with patch("time.sleep", return_value=None):
        yield


@pytest.fixture
def mock_file_system():
    with patch("genai_bench.cli.cli.Path.write_text") as mock_write_text:
        yield mock_write_text


@pytest.fixture
def mock_report_and_plot():
    mock_experiment_metadata = MagicMock()
    mock_experiment_metadata.server_gpu_count = 4
    with (
        patch(
            "genai_bench.cli.cli.load_one_experiment",
            return_value=(mock_experiment_metadata, MagicMock()),
        ) as mock_load_experiment,
        patch("genai_bench.cli.cli.create_workbook") as mock_create_workbook,
        patch("genai_bench.cli.cli.plot_experiment_data_flexible") as mock_plot,
        patch("genai_bench.cli.cli.plot_single_scenario_inference_speed_vs_throughput"),
    ):
        yield {
            "load_experiment": mock_load_experiment,
            "create_workbook": mock_create_workbook,
            "plot_experiment_data_flexible": mock_plot,
            "experiment_metadata": mock_experiment_metadata,
        }


@pytest.fixture
def mock_experiment_path():
    with patch("genai_bench.cli.cli.get_experiment_path") as mock_path:
        mock_path.return_value = MagicMock()
        mock_path.return_value.__truediv__ = lambda self, other: MagicMock()
        mock_path.return_value.__str__ = lambda: "/mock/path"
        yield mock_path


@pytest.mark.usefixtures(
    "mock_env_variables",
    "mock_dashboard",
    "mock_validate_tokenizer",
    "mock_time_sleep",
    "mock_file_system",
    "mock_report_and_plot",
    "mock_experiment_path",
)
def test_disable_streaming_with_multiple_async_params(
    cli_runner, async_options, caplog
):
    """Test warning includes all async-specific parameters."""
    import logging

    with caplog.at_level(logging.WARNING):
        with (
            patch(
                "genai_bench.async_runner.factory.create_runner"
            ) as mock_create_runner,
            patch(
                "genai_bench.metrics.aggregated_metrics_collector.AggregatedMetricsCollector"
            ) as mock_metrics_class,
            patch("genai_bench.cli.cli.Sampler.create") as mock_sampler_create,
            patch("genai_bench.cli.cli.time.monotonic", side_effect=lambda: 1.0),
        ):
            # Mock sampler
            mock_sampler = MagicMock()
            mock_sampler_create.return_value = mock_sampler

            # Mock metrics collector
            mock_metrics_collector = MagicMock()
            mock_metrics_collector.aggregated_metrics = MagicMock()
            mock_metrics_collector.aggregated_metrics.num_concurrency = 1.0
            mock_metrics_collector.aggregated_metrics.total_arrivals = 1
            mock_metrics_collector.clear = MagicMock()
            mock_metrics_collector.save = MagicMock()
            mock_metrics_collector.set_run_metadata = MagicMock()
            mock_metrics_collector.aggregate_metrics_data = MagicMock()
            mock_metrics_collector.get_ui_scatter_plot_metrics = MagicMock(
                return_value=[]
            )
            mock_metrics_class.return_value = mock_metrics_collector

            # Mock runner
            mock_runner = MagicMock()
            mock_runner.run = MagicMock(return_value=1.0)
            mock_create_runner.return_value = mock_runner

            result = cli_runner.invoke(
                benchmark,
                async_options
                + [
                    "--disable-streaming",
                    "--qps-level",
                    "1.0",
                    "--distribution",
                    "uniform",
                    "--track-network-timing",
                    "--traffic-scenario",
                    "D(100,100)",
                ],
                catch_exceptions=False,
            )

            assert result.exit_code == 0
            assert (
                "--qps-level, --distribution, --track-network-timing are only supported"
                in caplog.text
            )
            assert (
                "Running with --execution-engine=async with streaming enabled"
                in caplog.text
            )

            # Verify async runner WAS called
            assert mock_create_runner.called


@pytest.mark.usefixtures(
    "mock_env_variables",
    "mock_dashboard",
    "mock_validate_tokenizer",
    "mock_time_sleep",
    "mock_file_system",
    "mock_report_and_plot",
    "mock_experiment_path",
)
def test_async_with_multiple_qps_levels(cli_runner, async_options):
    """Test that --execution-engine async with multiple --qps-level values works."""
    with (
        patch("genai_bench.async_runner.factory.create_runner") as mock_create_runner,
        patch(
            "genai_bench.metrics.aggregated_metrics_collector.AggregatedMetricsCollector"
        ) as mock_metrics_class,
        patch("genai_bench.cli.cli.Sampler.create") as mock_sampler_create,
        patch(
            "genai_bench.cli.cli.time.monotonic",
            side_effect=[0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        ),  # Multiple runs
    ):
        # Mock runner - mock run() to return immediately
        mock_runner = MagicMock()
        mock_runner.run = MagicMock(
            return_value=1.0
        )  # Return immediately, no async execution
        mock_create_runner.return_value = mock_runner

        # Mock sampler
        mock_sampler = MagicMock()
        mock_sampler_create.return_value = mock_sampler

        # Mock metrics collector
        mock_metrics_collector = MagicMock()
        mock_metrics_collector.aggregated_metrics = MagicMock()
        mock_metrics_collector.clear = MagicMock()
        mock_metrics_collector.save = MagicMock()

        # Track calls to set_run_metadata to verify QPS values
        stored_values = []

        def capture_set_run_metadata(iteration, scenario_str, iteration_type):
            stored_values.append((iteration, scenario_str, iteration_type))
            mock_metrics_collector.aggregated_metrics.num_concurrency = iteration

        mock_metrics_collector.set_run_metadata.side_effect = capture_set_run_metadata
        mock_metrics_collector.aggregate_metrics_data = MagicMock()
        mock_metrics_collector.get_ui_scatter_plot_metrics = MagicMock(return_value=[])
        mock_metrics_class.return_value = mock_metrics_collector

        cli_runner.invoke(
            benchmark,
            [
                *async_options,
                "--qps-level",
                "0.5",
                "--qps-level",
                "1.0",
                "--qps-level",
                "2.0",
                "--traffic-scenario",
                "D(100,100)",
            ],
            catch_exceptions=False,
        )

        # Verify that set_run_metadata was called with decimal QPS values (not scaled)
        assert len(stored_values) >= 3, (
            "Should have called set_run_metadata for each QPS level"
        )

        # Check that QPS values are stored as decimals (not scaled to integers)
        qps_values = [val[0] for val in stored_values]
        assert 0.5 in qps_values, "QPS 0.5 should be stored as 0.5, not 50"
        assert 1.0 in qps_values, "QPS 1.0 should be stored as 1.0"
        assert 2.0 in qps_values, "QPS 2.0 should be stored as 2.0"

        # Verify no scaled values (50, 100, 200) are present
        assert 50 not in qps_values, "QPS 0.5 should not be scaled to 50"
        assert 100 not in qps_values, "QPS 1.0 should not be scaled to 100"
        assert 200 not in qps_values, "QPS 2.0 should not be scaled to 200"

        # Verify iteration_type is "num_concurrency" for compatibility
        for _, _, iteration_type in stored_values:
            assert iteration_type == "num_concurrency"


@pytest.mark.usefixtures(
    "mock_env_variables",
    "mock_dashboard",
    "mock_validate_tokenizer",
    "mock_time_sleep",
    "mock_file_system",
    "mock_report_and_plot",
    "mock_experiment_path",
)
def test_async_qps_values_stored_as_decimals(cli_runner, async_options):
    """Test that QPS values are stored as decimals in metrics, not scaled integers."""
    with (
        patch("genai_bench.async_runner.factory.create_runner") as mock_create_runner,
        patch(
            "genai_bench.metrics.aggregated_metrics_collector.AggregatedMetricsCollector"
        ) as mock_metrics_class,
        patch("genai_bench.cli.cli.Sampler.create") as mock_sampler_create,
        patch(
            "genai_bench.cli.cli.time.monotonic", side_effect=lambda: 1.0
        ),  # Return fixed time
    ):
        # Mock runner
        mock_runner = MagicMock()
        mock_runner.run = MagicMock(return_value=1.0)  # Return immediately
        mock_create_runner.return_value = mock_runner

        # Mock sampler
        mock_sampler = MagicMock()
        mock_sampler_create.return_value = mock_sampler

        # Mock metrics collector
        mock_metrics_collector = MagicMock()
        mock_metrics_collector.aggregated_metrics = MagicMock()
        mock_metrics_collector.clear = MagicMock()
        mock_metrics_collector.save = MagicMock()
        mock_metrics_collector.aggregate_metrics_data = MagicMock()
        mock_metrics_collector.get_ui_scatter_plot_metrics = MagicMock(return_value=[])

        # Track what gets set in num_concurrency
        stored_iterations = []

        def track_set_run_metadata(iteration, scenario_str, iteration_type):
            stored_iterations.append(iteration)
            mock_metrics_collector.aggregated_metrics.num_concurrency = iteration

        mock_metrics_collector.set_run_metadata.side_effect = track_set_run_metadata
        mock_metrics_class.return_value = mock_metrics_collector

        cli_runner.invoke(
            benchmark,
            [
                *async_options,
                "--qps-level",
                "0.5",
                "--traffic-scenario",
                "D(100,100)",
            ],
            catch_exceptions=False,
        )

        # Verify that num_concurrency was set to the decimal value (0.5), not scaled (50)
        assert mock_metrics_collector.set_run_metadata.called
        # Check stored iterations
        assert 0.5 in stored_iterations, "QPS 0.5 should be stored as 0.5"
        assert 50 not in stored_iterations, "QPS should not be scaled to 50"


@pytest.mark.usefixtures(
    "mock_env_variables",
    "mock_dashboard",
    "mock_validate_tokenizer",
    "mock_time_sleep",
    "mock_file_system",
    "mock_report_and_plot",
    "mock_experiment_path",
)
def test_async_requires_qps_or_concurrency(cli_runner, async_options):
    """Test that async runner requires either --qps-level or --num-concurrency."""
    with (
        patch("genai_bench.async_runner.factory.create_runner") as mock_create_runner,
        patch(
            "genai_bench.metrics.aggregated_metrics_collector.AggregatedMetricsCollector"
        ) as mock_metrics_class,
        patch("genai_bench.cli.cli.Sampler.create") as mock_sampler_create,
        patch(
            "genai_bench.cli.cli.time.monotonic", side_effect=lambda: 1.0
        ),  # Return fixed time
    ):
        # Mock all the components
        mock_sampler = MagicMock()
        mock_sampler_create.return_value = mock_sampler

        mock_metrics_collector = MagicMock()
        mock_metrics_collector.aggregated_metrics = MagicMock()
        mock_metrics_collector.clear = MagicMock()
        mock_metrics_collector.save = MagicMock()
        mock_metrics_collector.set_run_metadata = MagicMock()
        mock_metrics_collector.aggregate_metrics_data = MagicMock()
        mock_metrics_collector.get_ui_scatter_plot_metrics = MagicMock(return_value=[])
        mock_metrics_class.return_value = mock_metrics_collector

        # Mock runner - mock run() to return immediately
        mock_runner = MagicMock()
        mock_runner.run = MagicMock(return_value=1.0)
        mock_create_runner.return_value = mock_runner

        # Note: num_concurrency has defaults, so this will actually proceed with default values
        # The test verifies that the CLI can run with default num_concurrency (closed-loop mode)
        cli_runner.invoke(
            benchmark,
            [
                *async_options,
                "--traffic-scenario",
                "D(100,100)",
            ],
            catch_exceptions=False,
        )

        # With default num_concurrency, the CLI should proceed in closed-loop mode
        # So we verify that create_runner was called (not an error)
        assert mock_create_runner.called
        # Verify it was called with target_concurrency (closed-loop mode)
        call_kwargs = mock_create_runner.call_args[1]
        assert call_kwargs["target_concurrency"] is not None
        assert call_kwargs["qps_level"] is None


@pytest.mark.usefixtures(
    "mock_env_variables",
    "mock_dashboard",
    "mock_validate_tokenizer",
    "mock_time_sleep",
    "mock_file_system",
    "mock_report_and_plot",
    "mock_experiment_path",
)
def test_async_closed_loop_mode(cli_runner, async_options):
    """Test that async runner can work in closed-loop mode with --num-concurrency."""
    with (
        patch("genai_bench.async_runner.factory.create_runner") as mock_create_runner,
        patch(
            "genai_bench.metrics.aggregated_metrics_collector.AggregatedMetricsCollector"
        ) as mock_metrics_class,
        patch("genai_bench.cli.cli.Sampler.create") as mock_sampler_create,
        patch(
            "genai_bench.cli.cli.time.monotonic", side_effect=lambda: 1.0
        ),  # Return fixed time
    ):
        # Mock runner
        mock_runner = MagicMock()
        mock_runner.run = MagicMock(return_value=1.0)  # Return immediately
        mock_create_runner.return_value = mock_runner

        # Mock sampler
        mock_sampler = MagicMock()
        mock_sampler_create.return_value = mock_sampler

        # Mock metrics collector
        mock_metrics_collector = MagicMock()
        mock_metrics_collector.aggregated_metrics = MagicMock()
        mock_metrics_collector.clear = MagicMock()
        mock_metrics_collector.save = MagicMock()
        mock_metrics_collector.set_run_metadata = MagicMock()
        mock_metrics_collector.aggregate_metrics_data = MagicMock()
        mock_metrics_collector.get_ui_scatter_plot_metrics = MagicMock(return_value=[])
        mock_metrics_class.return_value = mock_metrics_collector

        cli_runner.invoke(
            benchmark,
            [
                *async_options,
                "--num-concurrency",
                "10",
                "--traffic-scenario",
                "D(100,100)",
            ],
            catch_exceptions=False,
        )

        # Verify runner was created with target_concurrency (closed-loop mode)
        assert mock_create_runner.called
        call_kwargs = mock_create_runner.call_args[1]
        assert call_kwargs["target_concurrency"] is not None
        assert call_kwargs["qps_level"] is None


@pytest.mark.usefixtures(
    "mock_env_variables",
    "mock_dashboard",
    "mock_validate_tokenizer",
    "mock_time_sleep",
    "mock_file_system",
    "mock_report_and_plot",
    "mock_experiment_path",
)
def test_disable_streaming_with_async_fallback_to_locust(
    cli_runner, async_options, caplog
):
    """Test that --disable-streaming with --execution-engine=async (no async params) falls back to Locust."""
    import logging

    with caplog.at_level(logging.WARNING):
        with (
            patch(
                "genai_bench.async_runner.factory.create_runner"
            ) as mock_create_runner,
            patch(
                "genai_bench.metrics.aggregated_metrics_collector.AggregatedMetricsCollector"
            ) as mock_metrics_class,
            patch("genai_bench.cli.cli.Sampler.create") as mock_sampler_create,
            patch("genai_bench.cli.cli.time.monotonic", side_effect=lambda: 1.0),
            patch(
                "genai_bench.distributed.runner.DistributedRunner"
            ) as mock_distributed_runner,
            patch("locust.env.Environment") as mock_environment,
        ):
            # Mock sampler
            mock_sampler = MagicMock()
            mock_sampler_create.return_value = mock_sampler

            # Mock metrics collector
            mock_metrics_collector = MagicMock()
            mock_metrics_collector.aggregated_metrics = MagicMock()
            mock_metrics_collector.clear = MagicMock()
            mock_metrics_collector.save = MagicMock()
            mock_metrics_collector.set_run_metadata = MagicMock()
            mock_metrics_collector.aggregate_metrics_data = MagicMock()
            mock_metrics_collector.get_ui_scatter_plot_metrics = MagicMock(
                return_value=[]
            )
            mock_metrics_class.return_value = mock_metrics_collector

            # Mock distributed runner
            mock_runner_instance = MagicMock()
            mock_runner_instance.metrics_collector = mock_metrics_collector
            mock_distributed_runner.return_value = mock_runner_instance

            # Mock environment
            mock_env_instance = MagicMock()
            mock_env_instance.runner = MagicMock()
            mock_environment.return_value = mock_env_instance

            result = cli_runner.invoke(
                benchmark,
                async_options
                + [
                    "--disable-streaming",
                    "--num-concurrency",
                    "2",  # Required for Locust
                    "--traffic-scenario",
                    "D(100,100)",
                ],
                catch_exceptions=False,
            )

            assert result.exit_code == 0
            assert (
                "--execution-engine=async is not supported with --disable-streaming"
                in caplog.text
            )
            assert "Automatically switching to --execution-engine=locust" in caplog.text

            # Verify async runner was NOT called (we fell back to Locust)
            assert not mock_create_runner.called


@pytest.mark.usefixtures(
    "mock_env_variables",
    "mock_dashboard",
    "mock_validate_tokenizer",
    "mock_time_sleep",
    "mock_file_system",
    "mock_report_and_plot",
    "mock_experiment_path",
)
def test_disable_streaming_with_async_params_uses_async_streaming(
    cli_runner, async_options, caplog
):
    """Test that --disable-streaming with --execution-engine=async and --qps-level uses async with streaming."""
    import logging

    with caplog.at_level(logging.WARNING):
        with (
            patch(
                "genai_bench.async_runner.factory.create_runner"
            ) as mock_create_runner,
            patch(
                "genai_bench.metrics.aggregated_metrics_collector.AggregatedMetricsCollector"
            ) as mock_metrics_class,
            patch("genai_bench.cli.cli.Sampler.create") as mock_sampler_create,
            patch("genai_bench.cli.cli.time.monotonic", side_effect=lambda: 1.0),
        ):
            # Mock sampler
            mock_sampler = MagicMock()
            mock_sampler_create.return_value = mock_sampler

            # Mock metrics collector
            mock_metrics_collector = MagicMock()
            mock_metrics_collector.aggregated_metrics = MagicMock()
            mock_metrics_collector.aggregated_metrics.num_concurrency = 1.0
            mock_metrics_collector.aggregated_metrics.total_arrivals = 1
            mock_metrics_collector.clear = MagicMock()
            mock_metrics_collector.save = MagicMock()
            mock_metrics_collector.set_run_metadata = MagicMock()
            mock_metrics_collector.aggregate_metrics_data = MagicMock()
            mock_metrics_collector.get_ui_scatter_plot_metrics = MagicMock(
                return_value=[]
            )
            mock_metrics_class.return_value = mock_metrics_collector

            # Mock runner
            mock_runner = MagicMock()
            mock_runner.run = MagicMock(return_value=1.0)
            mock_create_runner.return_value = mock_runner

            result = cli_runner.invoke(
                benchmark,
                async_options
                + [
                    "--disable-streaming",
                    "--qps-level",
                    "1.0",  # Async-specific param
                    "--traffic-scenario",
                    "D(100,100)",
                ],
                catch_exceptions=False,
            )

            assert result.exit_code == 0
            assert (
                "--qps-level is only supported with the async execution engine"
                in caplog.text
            )
            assert (
                "Running with --execution-engine=async with streaming enabled"
                in caplog.text
            )

            # Verify async runner WAS called (we overrode disable_streaming)
            assert mock_create_runner.called
