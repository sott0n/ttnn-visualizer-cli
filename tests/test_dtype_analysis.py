"""Tests for data format analysis module."""

import pytest

from ttnn_vis_cli.data.dtype_analysis import (
    BFLOAT8_B_USAGE_LOW_THRESHOLD,
    LOFI_RECOMMENDED_THRESHOLD,
    TILE_LAYOUT_LOW_THRESHOLD,
    DataFormatAnalyzer,
    DataFormatSummary,
    DTypeDistribution,
    LayoutDistribution,
    MathFidelityAnalyzer,
    MathFidelityDistribution,
    MathFidelitySummary,
    normalize_dtype,
    normalize_layout,
    normalize_math_fidelity,
)
from ttnn_vis_cli.data.models import OperationPerf, Tensor


def create_tensor(
    dtype: str = "BFLOAT16",
    layout: str = "TILE",
    tensor_id: int = 1,
) -> Tensor:
    """Create a test Tensor object."""
    return Tensor(
        id=tensor_id,
        shape="[1, 1, 32, 32]",
        dtype=dtype,
        layout=layout,
        memory_config="",
        device_id=0,
        address=0,
        buffer_type="",
    )


def create_operation_perf(
    op_code: str = "ttnn::matmul",
    op_name: str = "matmul",
    math_fidelity: str = "LoFi",
) -> OperationPerf:
    """Create a test OperationPerf object."""
    return OperationPerf(
        op_code=op_code,
        op_name=op_name,
        device_id=0,
        core_count=64,
        parallelization_strategy="",
        execution_time_ns=1000,
        host_time_ns=0,
        math_utilization=0,
        dram_read_bw=0,
        dram_write_bw=0,
        l1_read_bw=0,
        l1_write_bw=0,
        op_to_op_gap_ns=100,
        math_fidelity=math_fidelity,
    )


class TestNormalizeFunctions:
    """Tests for normalization functions."""

    def test_normalize_dtype_bfloat16(self):
        """Test normalizing bfloat16 variants."""
        assert normalize_dtype("BFLOAT16") == "BFLOAT16"
        assert normalize_dtype("DataType.BFLOAT16") == "BFLOAT16"
        assert normalize_dtype("torch.bfloat16") == "BFLOAT16"
        assert normalize_dtype("bfloat16") == "BFLOAT16"

    def test_normalize_dtype_bfloat8_b(self):
        """Test normalizing bfloat8_b variants."""
        assert normalize_dtype("BFLOAT8_B") == "BFLOAT8_B"
        assert normalize_dtype("DataType.BFLOAT8_B") == "BFLOAT8_B"

    def test_normalize_dtype_float32(self):
        """Test normalizing float32 variants."""
        assert normalize_dtype("FLOAT32") == "FLOAT32"
        assert normalize_dtype("float32") == "FLOAT32"

    def test_normalize_dtype_empty(self):
        """Test normalizing empty dtype."""
        assert normalize_dtype("") == "UNKNOWN"
        assert normalize_dtype(None) == "UNKNOWN"

    def test_normalize_layout_tile(self):
        """Test normalizing TILE layout."""
        assert normalize_layout("TILE") == "TILE"
        assert normalize_layout("Layout.TILE") == "TILE"
        assert normalize_layout("tile") == "TILE"

    def test_normalize_layout_row_major(self):
        """Test normalizing ROW_MAJOR layout."""
        assert normalize_layout("ROW_MAJOR") == "ROW_MAJOR"
        assert normalize_layout("STRIDED") == "ROW_MAJOR"

    def test_normalize_layout_empty(self):
        """Test normalizing empty layout."""
        assert normalize_layout("") == "UNKNOWN"
        assert normalize_layout(None) == "UNKNOWN"

    def test_normalize_math_fidelity(self):
        """Test normalizing math fidelity."""
        assert normalize_math_fidelity("LoFi") == "LoFi"
        assert normalize_math_fidelity("LOFI") == "LoFi"
        assert normalize_math_fidelity("HiFi2") == "HiFi2"
        assert normalize_math_fidelity("HIFI2") == "HiFi2"
        assert normalize_math_fidelity("HiFi3") == "HiFi3"
        assert normalize_math_fidelity("HiFi4") == "HiFi4"
        assert normalize_math_fidelity("") == "UNKNOWN"


class TestDataFormatAnalyzer:
    """Tests for DataFormatAnalyzer class."""

    def test_empty_tensors(self):
        """Test with no tensors."""
        analyzer = DataFormatAnalyzer([])
        summary = analyzer.get_summary()

        assert summary.total_tensors == 0
        assert summary.dtype_distribution == []
        assert summary.layout_distribution == []
        assert "No tensors found" in summary.recommendations

    def test_dtype_distribution(self):
        """Test dtype distribution calculation."""
        tensors = [
            create_tensor(dtype="BFLOAT16"),
            create_tensor(dtype="BFLOAT16"),
            create_tensor(dtype="BFLOAT8_B"),
            create_tensor(dtype="FLOAT32"),
        ]
        analyzer = DataFormatAnalyzer(tensors)
        distribution = analyzer.get_dtype_distribution()

        assert len(distribution) == 3
        # BFLOAT16 should be first (highest count)
        assert distribution[0].dtype == "BFLOAT16"
        assert distribution[0].count == 2
        assert distribution[0].percent == 50.0

    def test_layout_distribution(self):
        """Test layout distribution calculation."""
        tensors = [
            create_tensor(layout="TILE"),
            create_tensor(layout="TILE"),
            create_tensor(layout="TILE"),
            create_tensor(layout="ROW_MAJOR"),
        ]
        analyzer = DataFormatAnalyzer(tensors)
        distribution = analyzer.get_layout_distribution()

        assert len(distribution) == 2
        assert distribution[0].layout == "TILE"
        assert distribution[0].count == 3
        assert distribution[0].percent == 75.0

    def test_summary_with_low_bfloat8_b(self):
        """Test recommendations for low bfloat8_b usage."""
        tensors = [
            create_tensor(dtype="BFLOAT16"),
            create_tensor(dtype="BFLOAT16"),
            create_tensor(dtype="BFLOAT16"),
            create_tensor(dtype="BFLOAT16"),
            create_tensor(dtype="BFLOAT16"),
        ]
        analyzer = DataFormatAnalyzer(tensors)
        summary = analyzer.get_summary()

        assert summary.bfloat8_b_usage_percent == 0
        assert any("bfloat8_b" in rec.lower() for rec in summary.recommendations)

    def test_summary_with_low_tile_layout(self):
        """Test recommendations for low TILE layout usage."""
        tensors = [
            create_tensor(layout="ROW_MAJOR"),
            create_tensor(layout="ROW_MAJOR"),
            create_tensor(layout="ROW_MAJOR"),
            create_tensor(layout="TILE"),
        ]
        analyzer = DataFormatAnalyzer(tensors)
        summary = analyzer.get_summary()

        assert summary.tile_layout_percent == 25.0
        assert any("TILE" in rec for rec in summary.recommendations)

    def test_summary_with_float32(self):
        """Test recommendations for float32 usage."""
        tensors = [
            create_tensor(dtype="BFLOAT16"),
            create_tensor(dtype="FLOAT32"),
            create_tensor(dtype="FLOAT32"),
        ]
        analyzer = DataFormatAnalyzer(tensors)
        summary = analyzer.get_summary()

        assert summary.float32_count == 2
        assert any("FLOAT32" in rec for rec in summary.recommendations)

    def test_summary_good_configuration(self):
        """Test when configuration looks good."""
        # High bfloat8_b usage, high TILE usage, no float32
        tensors = [
            create_tensor(dtype="BFLOAT8_B", layout="TILE"),
            create_tensor(dtype="BFLOAT8_B", layout="TILE"),
            create_tensor(dtype="BFLOAT8_B", layout="TILE"),
            create_tensor(dtype="BFLOAT8_B", layout="TILE"),
            create_tensor(dtype="BFLOAT16", layout="TILE"),
        ]
        analyzer = DataFormatAnalyzer(tensors)
        summary = analyzer.get_summary()

        assert summary.bfloat8_b_usage_percent >= BFLOAT8_B_USAGE_LOW_THRESHOLD
        assert summary.tile_layout_percent >= TILE_LAYOUT_LOW_THRESHOLD
        assert "looks good" in summary.recommendations[0].lower()


class TestMathFidelityAnalyzer:
    """Tests for MathFidelityAnalyzer class."""

    def test_empty_operations(self):
        """Test with no operations."""
        analyzer = MathFidelityAnalyzer([])
        summary = analyzer.get_summary()

        assert summary.total_operations == 0
        assert summary.fidelity_distribution == []

    def test_operations_without_fidelity(self):
        """Test operations without math fidelity data."""
        operations = [
            create_operation_perf(math_fidelity=""),
            create_operation_perf(math_fidelity=""),
        ]
        analyzer = MathFidelityAnalyzer(operations)
        summary = analyzer.get_summary()

        assert summary.fidelity_distribution == []
        assert "No math fidelity data" in summary.recommendations[0]

    def test_fidelity_distribution(self):
        """Test fidelity distribution calculation."""
        operations = [
            create_operation_perf(math_fidelity="LoFi"),
            create_operation_perf(math_fidelity="LoFi"),
            create_operation_perf(math_fidelity="HiFi2"),
            create_operation_perf(math_fidelity="HiFi4"),
        ]
        analyzer = MathFidelityAnalyzer(operations)
        summary = analyzer.get_summary()

        assert summary.total_operations == 4
        assert summary.lofi_count == 2
        assert summary.hifi2_count == 1
        assert summary.hifi4_count == 1
        assert summary.lofi_percent == 50.0

    def test_low_lofi_recommendation(self):
        """Test recommendation for low LoFi usage."""
        operations = [
            create_operation_perf(math_fidelity="HiFi4"),
            create_operation_perf(math_fidelity="HiFi4"),
            create_operation_perf(math_fidelity="HiFi4"),
            create_operation_perf(math_fidelity="LoFi"),
        ]
        analyzer = MathFidelityAnalyzer(operations)
        summary = analyzer.get_summary()

        assert summary.lofi_percent == 25.0
        assert any("LoFi" in rec for rec in summary.recommendations)

    def test_hifi4_recommendation(self):
        """Test recommendation for HiFi4 usage."""
        operations = [
            create_operation_perf(math_fidelity="HiFi4"),
            create_operation_perf(math_fidelity="HiFi4"),
        ]
        analyzer = MathFidelityAnalyzer(operations)
        summary = analyzer.get_summary()

        assert any("HiFi4" in rec for rec in summary.recommendations)

    def test_good_fidelity_configuration(self):
        """Test when fidelity configuration looks reasonable."""
        # High LoFi usage, no HiFi4
        operations = [
            create_operation_perf(math_fidelity="LoFi"),
            create_operation_perf(math_fidelity="LoFi"),
            create_operation_perf(math_fidelity="LoFi"),
            create_operation_perf(math_fidelity="HiFi2"),
        ]
        analyzer = MathFidelityAnalyzer(operations)
        summary = analyzer.get_summary()

        assert summary.lofi_percent >= LOFI_RECOMMENDED_THRESHOLD
        assert summary.hifi4_count == 0
        assert "reasonable" in summary.recommendations[0].lower()


class TestDataFormatSummary:
    """Tests for DataFormatSummary dataclass."""

    def test_to_dict(self):
        """Test JSON serialization."""
        summary = DataFormatSummary(
            total_tensors=100,
            dtype_distribution=[
                DTypeDistribution(dtype="BFLOAT16", count=60, percent=60.0),
                DTypeDistribution(dtype="BFLOAT8_B", count=40, percent=40.0),
            ],
            layout_distribution=[
                LayoutDistribution(layout="TILE", count=80, percent=80.0),
            ],
            bfloat16_count=60,
            bfloat8_b_count=40,
            float32_count=0,
            tile_layout_count=80,
            row_major_count=20,
            bfloat8_b_usage_percent=40.0,
            tile_layout_percent=80.0,
            recommendations=["Test recommendation"],
        )

        result = summary.to_dict()

        assert result["total_tensors"] == 100
        assert len(result["dtype_distribution"]) == 2
        assert result["bfloat8_b_usage_percent"] == 40.0
        assert result["recommendations"] == ["Test recommendation"]


class TestMathFidelitySummary:
    """Tests for MathFidelitySummary dataclass."""

    def test_to_dict(self):
        """Test JSON serialization."""
        summary = MathFidelitySummary(
            total_operations=50,
            fidelity_distribution=[
                MathFidelityDistribution(fidelity="LoFi", count=30, percent=60.0),
                MathFidelityDistribution(fidelity="HiFi2", count=20, percent=40.0),
            ],
            lofi_count=30,
            hifi2_count=20,
            hifi3_count=0,
            hifi4_count=0,
            lofi_percent=60.0,
            recommendations=["Test fidelity recommendation"],
        )

        result = summary.to_dict()

        assert result["total_operations"] == 50
        assert len(result["fidelity_distribution"]) == 2
        assert result["lofi_percent"] == 60.0
        assert result["recommendations"] == ["Test fidelity recommendation"]
