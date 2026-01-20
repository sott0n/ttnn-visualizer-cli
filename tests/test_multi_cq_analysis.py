"""Tests for multi-CQ analysis module."""

import pytest

from ttnn_vis_cli.data.multi_cq_analysis import (
    IO_BOTTLENECK_THRESHOLD,
    MULTI_CQ_RECOMMENDED_THRESHOLD,
    MultiCQAnalyzer,
    MultiCQSummary,
    OperationIOAnalysis,
)
from ttnn_vis_cli.data.models import OperationPerf


def create_operation_perf(
    op_code: str = "ttnn::add",
    op_name: str = "add",
    execution_time_ns: float = 1000,
    dispatch_cq_cmd_time_ns: float = 100,
    dispatch_wait_time_ns: float = 50,
    erisc_kernel_duration_ns: float = 50,
    core_count: int = 64,
) -> OperationPerf:
    """Create a test OperationPerf object."""
    return OperationPerf(
        op_code=op_code,
        op_name=op_name,
        device_id=0,
        core_count=core_count,
        parallelization_strategy="",
        execution_time_ns=execution_time_ns,
        host_time_ns=0,
        math_utilization=0,
        dram_read_bw=0,
        dram_write_bw=0,
        l1_read_bw=0,
        l1_write_bw=0,
        dispatch_cq_cmd_time_ns=dispatch_cq_cmd_time_ns,
        dispatch_wait_time_ns=dispatch_wait_time_ns,
        erisc_kernel_duration_ns=erisc_kernel_duration_ns,
    )


class TestMultiCQAnalyzer:
    """Tests for MultiCQAnalyzer class."""

    def test_empty_operations(self):
        """Test with no operations."""
        analyzer = MultiCQAnalyzer([])
        summary = analyzer.get_summary()

        assert summary.total_operations == 0
        assert summary.total_device_time_ns == 0
        assert summary.total_io_time_ns == 0
        assert summary.is_io_bound is False
        assert summary.multi_cq_recommended is False

    def test_compute_bound_model(self):
        """Test model that is compute-bound (low I/O overhead)."""
        operations = [
            create_operation_perf(
                execution_time_ns=10000,
                dispatch_cq_cmd_time_ns=50,
                dispatch_wait_time_ns=25,
                erisc_kernel_duration_ns=25,
            ),
            create_operation_perf(
                execution_time_ns=10000,
                dispatch_cq_cmd_time_ns=50,
                dispatch_wait_time_ns=25,
                erisc_kernel_duration_ns=25,
            ),
        ]
        analyzer = MultiCQAnalyzer(operations)
        summary = analyzer.get_summary()

        # Total device: 20000, I/O: 200 (dispatch: 100, wait: 50, erisc: 50)
        assert summary.total_device_time_ns == 20000
        assert summary.total_io_time_ns == 200
        assert summary.io_overhead_percent < 5
        assert summary.is_io_bound is False
        assert summary.multi_cq_recommended is False

    def test_io_bound_model(self):
        """Test model that is I/O-bound (high I/O overhead)."""
        operations = [
            create_operation_perf(
                execution_time_ns=1000,
                dispatch_cq_cmd_time_ns=1000,
                dispatch_wait_time_ns=500,
                erisc_kernel_duration_ns=500,
            ),
            create_operation_perf(
                execution_time_ns=1000,
                dispatch_cq_cmd_time_ns=1000,
                dispatch_wait_time_ns=500,
                erisc_kernel_duration_ns=500,
            ),
        ]
        analyzer = MultiCQAnalyzer(operations)
        summary = analyzer.get_summary()

        # High I/O overhead
        assert summary.total_device_time_ns == 2000
        assert summary.total_io_time_ns == 4000  # dispatch + wait + erisc
        assert summary.io_overhead_percent > IO_BOTTLENECK_THRESHOLD
        assert summary.is_io_bound is True
        assert summary.multi_cq_recommended is True

    def test_multi_cq_threshold(self):
        """Test 2CQ recommendation threshold."""
        # Create operations with ~25% I/O overhead (above recommendation threshold)
        operations = [
            create_operation_perf(
                execution_time_ns=3000,
                dispatch_cq_cmd_time_ns=500,
                dispatch_wait_time_ns=200,
                erisc_kernel_duration_ns=300,
            ),
        ]
        analyzer = MultiCQAnalyzer(operations)
        summary = analyzer.get_summary()

        # I/O time = 1000, total = 3700, overhead ~27%
        assert summary.io_overhead_percent > MULTI_CQ_RECOMMENDED_THRESHOLD
        assert summary.multi_cq_recommended is True

    def test_get_io_bound_operations(self):
        """Test getting operations sorted by I/O overhead."""
        operations = [
            create_operation_perf(
                op_code="op1",
                execution_time_ns=10000,
                dispatch_cq_cmd_time_ns=100,
                dispatch_wait_time_ns=50,
                erisc_kernel_duration_ns=50,
            ),
            create_operation_perf(
                op_code="op2",
                execution_time_ns=1000,
                dispatch_cq_cmd_time_ns=500,
                dispatch_wait_time_ns=300,
                erisc_kernel_duration_ns=200,
            ),
            create_operation_perf(
                op_code="op3",
                execution_time_ns=5000,
                dispatch_cq_cmd_time_ns=300,
                dispatch_wait_time_ns=100,
                erisc_kernel_duration_ns=100,
            ),
        ]
        analyzer = MultiCQAnalyzer(operations)
        io_ops = analyzer.get_io_bound_operations(limit=2)

        assert len(io_ops) == 2
        assert io_ops[0].op_code == "op2"  # Highest I/O overhead
        assert io_ops[1].op_code == "op3"  # Second highest

    def test_get_io_distribution(self):
        """Test I/O overhead distribution calculation."""
        operations = [
            # Low I/O overhead (~2%)
            create_operation_perf(
                execution_time_ns=10000,
                dispatch_cq_cmd_time_ns=100,
                dispatch_wait_time_ns=50,
                erisc_kernel_duration_ns=50,
            ),
            # Medium I/O overhead (~36%)
            create_operation_perf(
                execution_time_ns=5000,
                dispatch_cq_cmd_time_ns=1500,
                dispatch_wait_time_ns=500,
                erisc_kernel_duration_ns=500,
            ),
            # High I/O overhead (~55%)
            create_operation_perf(
                execution_time_ns=1000,
                dispatch_cq_cmd_time_ns=1000,
                dispatch_wait_time_ns=500,
                erisc_kernel_duration_ns=500,
            ),
        ]
        analyzer = MultiCQAnalyzer(operations)
        distribution = analyzer.get_io_distribution()

        assert distribution["0-10%"] == 1
        assert distribution["30-50%"] == 1
        assert distribution["50%+"] == 1

    def test_summary_statistics(self):
        """Test summary statistics calculations."""
        operations = [
            create_operation_perf(
                execution_time_ns=1000,
                dispatch_cq_cmd_time_ns=100,
                dispatch_wait_time_ns=50,
                erisc_kernel_duration_ns=50,
            ),
            create_operation_perf(
                execution_time_ns=2000,
                dispatch_cq_cmd_time_ns=200,
                dispatch_wait_time_ns=100,
                erisc_kernel_duration_ns=100,
            ),
        ]
        analyzer = MultiCQAnalyzer(operations)
        summary = analyzer.get_summary()

        assert summary.total_operations == 2
        assert summary.total_device_time_ns == 3000
        assert summary.total_dispatch_cq_time_ns == 300
        assert summary.total_wait_time_ns == 150
        assert summary.total_erisc_time_ns == 150
        assert summary.total_io_time_ns == 600

    def test_zero_time_operations(self):
        """Test with operations that have all zero timing values."""
        operations = [
            create_operation_perf(
                execution_time_ns=0,
                dispatch_cq_cmd_time_ns=0,
                dispatch_wait_time_ns=0,
                erisc_kernel_duration_ns=0,
            ),
        ]
        analyzer = MultiCQAnalyzer(operations)
        summary = analyzer.get_summary()

        # Should not raise, should return sensible defaults
        assert summary.total_operations == 1
        assert summary.io_overhead_percent == 0
        assert summary.is_io_bound is False

    def test_zero_time_in_distribution(self):
        """Test distribution skips operations with zero total time."""
        operations = [
            create_operation_perf(
                execution_time_ns=0,
                dispatch_cq_cmd_time_ns=0,
                dispatch_wait_time_ns=0,
                erisc_kernel_duration_ns=0,
            ),
            create_operation_perf(
                execution_time_ns=10000,
                dispatch_cq_cmd_time_ns=100,
                dispatch_wait_time_ns=50,
                erisc_kernel_duration_ns=50,
            ),
        ]
        analyzer = MultiCQAnalyzer(operations)
        distribution = analyzer.get_io_distribution()

        # Zero-time op should be skipped, only 1 op counted
        total = sum(distribution.values())
        assert total == 1

    def test_get_io_bound_operations_limit_greater_than_count(self):
        """Test limit parameter greater than operation count."""
        operations = [
            create_operation_perf(op_code="op1"),
            create_operation_perf(op_code="op2"),
        ]
        analyzer = MultiCQAnalyzer(operations)
        io_ops = analyzer.get_io_bound_operations(limit=100)

        # Should return all operations, not error
        assert len(io_ops) == 2

    def test_get_io_bound_operations_limit_zero(self):
        """Test limit=0 returns empty list."""
        operations = [
            create_operation_perf(op_code="op1"),
            create_operation_perf(op_code="op2"),
        ]
        analyzer = MultiCQAnalyzer(operations)
        io_ops = analyzer.get_io_bound_operations(limit=0)

        assert len(io_ops) == 0


class TestMultiCQSummary:
    """Tests for MultiCQSummary dataclass."""

    def test_to_dict(self):
        """Test JSON serialization."""
        summary = MultiCQSummary(
            total_operations=10,
            total_device_time_ns=1000000,
            total_io_time_ns=200000,
            total_dispatch_cq_time_ns=100000,
            total_wait_time_ns=50000,
            total_erisc_time_ns=50000,
            total_compute_time_ns=950000,
            io_overhead_percent=16.67,
            is_io_bound=False,
            multi_cq_recommended=False,
            io_bound_operations=2,
            recommendations=["Test recommendation"],
        )

        result = summary.to_dict()

        assert result["total_operations"] == 10
        assert result["total_device_time_ns"] == 1000000
        assert result["total_device_time_ms"] == 1.0
        assert result["total_io_time_ms"] == 0.2
        assert result["io_overhead_percent"] == 16.67
        assert result["is_io_bound"] is False
        assert result["multi_cq_recommended"] is False
        assert result["recommendations"] == ["Test recommendation"]


class TestOperationIOAnalysis:
    """Tests for OperationIOAnalysis dataclass."""

    def test_to_dict(self):
        """Test JSON serialization."""
        analysis = OperationIOAnalysis(
            op_code="ttnn::matmul",
            op_name="matmul",
            device_time_ns=5000,
            dispatch_time_ns=500,
            wait_time_ns=250,
            erisc_time_ns=250,
            io_overhead_percent=16.67,
            is_io_bound=False,
        )

        result = analysis.to_dict()

        assert result["op_code"] == "ttnn::matmul"
        assert result["device_time_ns"] == 5000
        assert result["device_time_us"] == 5.0
        assert result["total_io_time_ns"] == 1000  # 500 + 250 + 250
        assert result["total_io_time_us"] == 1.0
        assert result["dispatch_time_us"] == 0.5
        assert result["wait_time_us"] == 0.25
        assert result["erisc_time_us"] == 0.25
        assert result["io_overhead_percent"] == 16.67
        assert result["is_io_bound"] is False

    def test_total_io_time_property(self):
        """Test total_io_time_ns property calculation."""
        analysis = OperationIOAnalysis(
            op_code="ttnn::add",
            op_name="add",
            device_time_ns=1000,
            dispatch_time_ns=100,
            wait_time_ns=50,
            erisc_time_ns=50,
            io_overhead_percent=10.0,
            is_io_bound=False,
        )

        assert analysis.total_io_time_ns == 200  # 100 + 50 + 50


class TestRecommendations:
    """Tests for recommendation generation."""

    def test_io_bound_recommendation(self):
        """Test recommendation for I/O-bound model."""
        operations = [
            create_operation_perf(
                execution_time_ns=1000,
                dispatch_cq_cmd_time_ns=2000,
                dispatch_wait_time_ns=1000,
                erisc_kernel_duration_ns=1000,
            ),
        ]
        analyzer = MultiCQAnalyzer(operations)
        summary = analyzer.get_summary()

        assert any("I/O-BOUND" in rec for rec in summary.recommendations)
        assert any("2CQ" in rec for rec in summary.recommendations)

    def test_compute_bound_recommendation(self):
        """Test recommendation for compute-bound model."""
        operations = [
            create_operation_perf(
                execution_time_ns=10000,
                dispatch_cq_cmd_time_ns=50,
                dispatch_wait_time_ns=25,
                erisc_kernel_duration_ns=25,
            ),
        ]
        analyzer = MultiCQAnalyzer(operations)
        summary = analyzer.get_summary()

        assert any("COMPUTE-BOUND" in rec for rec in summary.recommendations)

    def test_dispatch_dominance_recommendation(self):
        """Test recommendation when dispatch CQ time dominates."""
        operations = [
            create_operation_perf(
                execution_time_ns=1000,
                dispatch_cq_cmd_time_ns=2000,  # Dominates I/O
                dispatch_wait_time_ns=100,
                erisc_kernel_duration_ns=100,
            ),
        ]
        analyzer = MultiCQAnalyzer(operations)
        summary = analyzer.get_summary()

        assert any("Dispatch CQ time dominates" in rec for rec in summary.recommendations)

    def test_erisc_dominance_recommendation(self):
        """Test recommendation when ERISC time dominates."""
        operations = [
            create_operation_perf(
                execution_time_ns=1000,
                dispatch_cq_cmd_time_ns=100,
                dispatch_wait_time_ns=100,
                erisc_kernel_duration_ns=2000,  # Dominates I/O
            ),
        ]
        analyzer = MultiCQAnalyzer(operations)
        summary = analyzer.get_summary()

        assert any("ERISC" in rec for rec in summary.recommendations)
