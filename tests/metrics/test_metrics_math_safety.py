"""
Tests for metrics calculation math safety issues.

These tests validate that the fixes for audit findings in docs/genai-bench/audit.md
work correctly. The fixes guard against division-by-zero, negative values, and edge cases.

Each test verifies that the fixes handle edge cases gracefully without errors.
"""

import math
from unittest.mock import MagicMock

import pytest

from genai_bench.metrics.aggregated_metrics_collector import AggregatedMetricsCollector
from genai_bench.metrics.metrics import RequestLevelMetrics
from genai_bench.metrics.request_metrics_collector import RequestMetricsCollector
from genai_bench.protocol import UserChatResponse


class TestRequestMetricsCollectorDivisionByZeroFixes:
    """Tests verifying division-by-zero fixes in RequestMetricsCollector."""

    def test_tpot_zero_handled_gracefully(self):
        """
        FIX VERIFIED: Line 75 of request_metrics_collector.py now guards against
        division by zero when tpot would be 0.

        When output_latency is 0 (time_at_first_token == end_time), the fix
        sets tpot, output_inference_speed, and output_throughput to 0.
        """
        mock_response = MagicMock(spec=UserChatResponse)
        mock_response.status_code = 200
        mock_response.tokens_received = 10  # Multiple tokens
        mock_response.num_prefill_tokens = 5

        # Timestamps where first token and end time are the same
        # This makes output_latency = 0
        mock_response.start_time = 1000.0
        mock_response.time_at_first_token = 1001.0  # ttft = 1.0
        mock_response.end_time = 1001.0  # output_latency = 0

        collector = RequestMetricsCollector()
        # Should NOT raise ZeroDivisionError anymore
        collector.calculate_metrics(mock_response)

        # Metrics should be set to 0 instead of causing division error
        assert collector.metrics.tpot == 0.0
        assert collector.metrics.output_inference_speed == 0.0
        assert collector.metrics.output_throughput == 0.0

    def test_ttft_zero_handled_correctly(self):
        """
        Verify that ttft=0 is handled correctly.
        Line 47-51 guards against this with conditional.
        """
        mock_response = MagicMock(spec=UserChatResponse)
        mock_response.status_code = 200
        mock_response.tokens_received = 10
        mock_response.num_prefill_tokens = 5

        # Timestamps where start_time == time_at_first_token (ttft=0)
        mock_response.start_time = 1000.0
        mock_response.time_at_first_token = 1000.0  # ttft = 0
        mock_response.end_time = 1001.0

        collector = RequestMetricsCollector()
        collector.calculate_metrics(mock_response)

        # Should be guarded - input_throughput should be 0 when ttft=0
        assert collector.metrics.ttft == 0
        assert collector.metrics.input_throughput == 0


class TestRequestMetricsCollectorNegativeValueFixes:
    """Tests verifying negative time value fixes."""

    def test_negative_ttft_clamped_to_zero(self):
        """
        FIX VERIFIED: Negative ttft from clock skew is now clamped to 0.
        """
        mock_response = MagicMock(spec=UserChatResponse)
        mock_response.status_code = 200
        mock_response.tokens_received = 10
        mock_response.num_prefill_tokens = 5

        # Out-of-order timestamps (simulating clock skew)
        mock_response.start_time = 1001.0
        mock_response.time_at_first_token = 1000.0  # Before start!
        mock_response.end_time = 1002.0

        collector = RequestMetricsCollector()
        collector.calculate_metrics(mock_response)

        # ttft should be clamped to 0, not -1
        assert collector.metrics.ttft == 0.0
        # input_throughput should also be 0 (division guarded)
        assert collector.metrics.input_throughput == 0

    def test_negative_e2e_latency_clamped_to_zero(self):
        """
        FIX VERIFIED: Negative e2e_latency from clock skew is now clamped to 0.
        """
        mock_response = MagicMock(spec=UserChatResponse)
        mock_response.status_code = 200
        mock_response.tokens_received = 10
        mock_response.num_prefill_tokens = 5

        # End time before start time (clock skew)
        mock_response.start_time = 1001.0
        mock_response.time_at_first_token = 1000.5
        mock_response.end_time = 1000.0  # Before start!

        collector = RequestMetricsCollector()
        collector.calculate_metrics(mock_response)

        # e2e_latency should be clamped to 0, not -1
        assert collector.metrics.e2e_latency == 0.0

    def test_negative_output_latency_clamped_to_zero(self):
        """
        FIX VERIFIED: Negative output_latency is now clamped to 0.
        """
        mock_response = MagicMock(spec=UserChatResponse)
        mock_response.status_code = 200
        mock_response.tokens_received = 10
        mock_response.num_prefill_tokens = 5

        # Scenario where first token arrives after end time
        # After clamping ttft/e2e_latency, output_latency = e2e - ttft
        # With the clamping, this shouldn't produce negative values
        mock_response.start_time = 1000.0
        mock_response.time_at_first_token = 1003.0  # ttft = 3
        mock_response.end_time = 1002.0  # e2e_latency = 2

        collector = RequestMetricsCollector()
        collector.calculate_metrics(mock_response)

        # output_latency should be clamped to 0
        assert collector.metrics.output_latency == 0.0


class TestAggregatedMetricsCollectorDivisionByZeroFixes:
    """Tests verifying division-by-zero fixes in AggregatedMetricsCollector."""

    def test_zero_run_duration_handled_gracefully(self):
        """
        FIX VERIFIED: Zero run_duration no longer causes ZeroDivisionError.
        Throughput metrics are set to 0 instead.
        """
        collector = AggregatedMetricsCollector()

        # Add a valid metric
        metrics = RequestLevelMetrics(
            ttft=0.1,
            tpot=0.2,
            e2e_latency=1.0,
            output_latency=0.9,
            input_throughput=20.0,
            output_throughput=11.111,
            num_input_tokens=2,
            num_output_tokens=10,
            output_inference_speed=5,
            total_tokens=12,
        )
        collector.add_single_request_metrics(metrics)

        # Run with zero duration (start_time == end_time)
        # Should NOT raise ZeroDivisionError anymore
        collector.aggregate_metrics_data(
            start_time=1000.0,
            end_time=1000.0,  # Same as start!
            dataset_character_to_token_ratio=4.0,
            warmup_ratio=0.0,
            cooldown_ratio=0.0,
        )

        # Throughput metrics should be 0
        assert collector.aggregated_metrics.mean_output_throughput_tokens_per_s == 0.0
        assert collector.aggregated_metrics.mean_input_throughput_tokens_per_s == 0.0
        assert (
            collector.aggregated_metrics.mean_total_tokens_throughput_tokens_per_s
            == 0.0
        )
        assert collector.aggregated_metrics.mean_total_chars_per_hour == 0.0

    def test_very_small_run_duration_still_calculates(self):
        """
        Very small (but positive) run_duration should still calculate throughput.
        The values may be extreme but that's mathematically correct.
        """
        collector = AggregatedMetricsCollector()

        metrics = RequestLevelMetrics(
            ttft=0.1,
            tpot=0.2,
            e2e_latency=1.0,
            output_latency=0.9,
            input_throughput=20.0,
            output_throughput=11.111,
            num_input_tokens=100,
            num_output_tokens=100,
            output_inference_speed=5,
            total_tokens=200,
        )
        collector.add_single_request_metrics(metrics)

        # Very small duration (1 microsecond)
        collector.aggregate_metrics_data(
            start_time=1000.0,
            end_time=1000.000001,  # 1 microsecond later
            dataset_character_to_token_ratio=4.0,
            warmup_ratio=0.0,
            cooldown_ratio=0.0,
        )

        # Throughput = 200 tokens / 0.000001s = 200,000,000 tokens/s
        # This is unrealistic but mathematically correct
        assert (
            collector.aggregated_metrics.mean_total_tokens_throughput_tokens_per_s > 1e8
        )


class TestAggregatedMetricsCollectorEdgeCases:
    """Tests for edge cases in AggregatedMetricsCollector."""

    def test_single_output_token_skips_tpot_calculation(self):
        """
        When num_output_tokens <= 1, tpot calculation is skipped.
        This test verifies the guard at line 71.
        """
        mock_response = MagicMock(spec=UserChatResponse)
        mock_response.status_code = 200
        mock_response.tokens_received = 1  # Only 1 token
        mock_response.num_prefill_tokens = 5
        mock_response.start_time = 1000.0
        mock_response.time_at_first_token = 1001.0
        mock_response.end_time = 1002.0

        collector = RequestMetricsCollector()
        collector.calculate_metrics(mock_response)

        # With 1 output token, tpot should remain None (not calculated)
        assert collector.metrics.tpot is None
        assert collector.metrics.output_inference_speed is None

    def test_inf_values_propagate_through_aggregation(self):
        """
        Test that inf values from division issues propagate through aggregation.
        """
        collector = AggregatedMetricsCollector()

        # Create metric with inf value (simulating a near-zero division)
        metrics = RequestLevelMetrics(
            ttft=0.1,
            tpot=0.2,
            e2e_latency=1.0,
            output_latency=0.9,
            input_throughput=float("inf"),  # Problematic value
            output_throughput=11.111,
            num_input_tokens=2,
            num_output_tokens=10,
            output_inference_speed=5,
            total_tokens=12,
        )
        collector.add_single_request_metrics(metrics)

        collector.aggregate_metrics_data(
            start_time=0,
            end_time=1.0,
            dataset_character_to_token_ratio=4.0,
            warmup_ratio=0.0,
            cooldown_ratio=0.0,
        )

        # inf propagates to mean
        assert math.isinf(collector.aggregated_metrics.stats.input_throughput.mean)

    def test_nan_values_propagate_through_aggregation(self):
        """
        Test that NaN values from invalid operations propagate through aggregation.
        """
        collector = AggregatedMetricsCollector()

        metrics = RequestLevelMetrics(
            ttft=float("nan"),  # NaN value
            tpot=0.2,
            e2e_latency=1.0,
            output_latency=0.9,
            input_throughput=20.0,
            output_throughput=11.111,
            num_input_tokens=2,
            num_output_tokens=10,
            output_inference_speed=5,
            total_tokens=12,
        )
        collector.add_single_request_metrics(metrics)

        collector.aggregate_metrics_data(
            start_time=0,
            end_time=1.0,
            dataset_character_to_token_ratio=4.0,
            warmup_ratio=0.0,
            cooldown_ratio=0.0,
        )

        # NaN propagates to mean
        assert math.isnan(collector.aggregated_metrics.stats.ttft.mean)


class TestMetricsFiltering:
    """
    Tests related to the disabled filtering in AggregatedMetricsCollector.

    Metrics with output_latency >= 0.001s are kept even with extreme speeds.
    Metrics with output_latency < 0.001s have tpot/inference_speed nulled out.
    Non-streaming tasks (tpot=0) are never filtered.
    """

    def test_extreme_inference_speed_not_filtered_when_latency_normal(self):
        """
        With output_latency=0.9 (normal), extreme inference speed is kept
        because the latency is well above the 0.001s threshold.
        """
        collector = AggregatedMetricsCollector()

        extreme_metrics = RequestLevelMetrics(
            ttft=0.1,
            tpot=0.0000002,  # 200 nanoseconds per token
            e2e_latency=1.0,
            output_latency=0.9,  # Normal latency - above threshold
            input_throughput=20.0,
            output_throughput=11.111,
            num_input_tokens=2,
            num_output_tokens=10,
            output_inference_speed=1 / 0.0000002,  # 5,000,000 tokens/sec!
            total_tokens=12,
        )

        collector.add_single_request_metrics(extreme_metrics)

        # Metrics are kept - output_latency is normal
        assert extreme_metrics in collector.all_request_metrics
        assert extreme_metrics.tpot == 0.0000002
        assert extreme_metrics.output_inference_speed is not None

    def test_short_output_latency_nulls_tpot(self):
        """
        With output_latency < 0.001s, tpot and output_inference_speed
        are set to None to avoid misleading aggregation.
        """
        collector = AggregatedMetricsCollector()

        jittery_metrics = RequestLevelMetrics(
            ttft=0.1,
            tpot=0.00001,
            e2e_latency=0.1001,
            output_latency=0.0001,  # Very short - below threshold
            input_throughput=20.0,
            output_throughput=11.111,
            num_input_tokens=2,
            num_output_tokens=1,
            output_inference_speed=100000.0,
            total_tokens=3,
        )

        collector.add_single_request_metrics(jittery_metrics)

        # Metrics are still collected (not dropped)
        assert jittery_metrics in collector.all_request_metrics
        # But tpot and output_inference_speed are nulled out
        assert jittery_metrics.tpot is None
        assert jittery_metrics.output_inference_speed is None

    def test_non_streaming_tpot_zero_not_filtered(self):
        """
        Non-streaming tasks (embeddings) where tpot=0 is intentional
        should NOT have their metrics nulled out.
        """
        collector = AggregatedMetricsCollector()

        embedding_metrics = RequestLevelMetrics(
            ttft=0.1,
            tpot=0,  # Intentionally zero for non-streaming
            e2e_latency=0.1,
            output_latency=0.0,  # Zero is expected
            input_throughput=20.0,
            output_throughput=0.0,
            num_input_tokens=2,
            num_output_tokens=0,
            output_inference_speed=0.0,
            total_tokens=2,
        )

        collector.add_single_request_metrics(embedding_metrics)

        assert embedding_metrics in collector.all_request_metrics
        # tpot stays at 0 (not nulled) because tpot=0 is intentional
        assert embedding_metrics.tpot == 0


class TestRobustnessWithValidData:
    """Sanity tests with valid data to ensure basic functionality works."""

    def test_normal_chat_response_metrics(self):
        """Test normal metrics calculation with realistic values."""
        mock_response = MagicMock(spec=UserChatResponse)
        mock_response.status_code = 200
        mock_response.tokens_received = 100
        mock_response.num_prefill_tokens = 50
        mock_response.start_time = 1000.0
        mock_response.time_at_first_token = 1000.5  # 500ms TTFT
        mock_response.end_time = 1002.0  # 2 sec total

        collector = RequestMetricsCollector()
        collector.calculate_metrics(mock_response)

        assert collector.metrics.ttft == 0.5
        assert collector.metrics.e2e_latency == 2.0
        assert collector.metrics.output_latency == 1.5
        assert collector.metrics.num_input_tokens == 50
        assert collector.metrics.num_output_tokens == 100
        # tpot = 1.5 / 99 = ~0.0152
        assert collector.metrics.tpot == pytest.approx(1.5 / 99, rel=0.01)
        # output_inference_speed = 1 / tpot = ~66 tokens/sec
        assert collector.metrics.output_inference_speed == pytest.approx(
            99 / 1.5, rel=0.01
        )

    def test_normal_aggregation_with_multiple_requests(self):
        """Test aggregation with multiple valid requests."""
        collector = AggregatedMetricsCollector()

        for i in range(10):
            metrics = RequestLevelMetrics(
                ttft=0.1 + i * 0.01,
                tpot=0.02,
                e2e_latency=1.0 + i * 0.1,
                output_latency=0.9 + i * 0.1,
                input_throughput=100.0,
                output_throughput=50.0,
                num_input_tokens=50,
                num_output_tokens=100,
                output_inference_speed=50.0,
                total_tokens=150,
            )
            collector.add_single_request_metrics(metrics)

        collector.aggregate_metrics_data(
            start_time=0,
            end_time=10.0,
            dataset_character_to_token_ratio=4.0,
            warmup_ratio=0.0,
            cooldown_ratio=0.0,
        )

        assert collector.aggregated_metrics.num_completed_requests == 10
        assert collector.aggregated_metrics.run_duration == 10.0
        # Total tokens = 150 * 10 = 1500, throughput = 150 tokens/sec
        assert (
            collector.aggregated_metrics.mean_total_tokens_throughput_tokens_per_s
            == 150.0
        )
