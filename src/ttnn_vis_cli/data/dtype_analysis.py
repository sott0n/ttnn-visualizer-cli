"""Data format analysis logic for TTNN profiling data."""

from collections import defaultdict
from dataclasses import dataclass

from .models import OperationPerf, Tensor

# Thresholds for recommendations
BFLOAT8_B_USAGE_LOW_THRESHOLD = 20  # Recommend bfloat8_b if usage < 20%
TILE_LAYOUT_LOW_THRESHOLD = 50  # Recommend TILE layout if usage < 50%
LOFI_RECOMMENDED_THRESHOLD = 50  # Recommend LoFi if usage < 50%


@dataclass
class DTypeDistribution:
    """Data type distribution entry."""

    dtype: str
    count: int
    percent: float

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "dtype": self.dtype,
            "count": self.count,
            "percent": round(self.percent, 2),
        }


@dataclass
class LayoutDistribution:
    """Layout distribution entry."""

    layout: str
    count: int
    percent: float

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "layout": self.layout,
            "count": self.count,
            "percent": round(self.percent, 2),
        }


@dataclass
class MathFidelityDistribution:
    """Math fidelity distribution entry."""

    fidelity: str
    count: int
    percent: float

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "fidelity": self.fidelity,
            "count": self.count,
            "percent": round(self.percent, 2),
        }


@dataclass
class DataFormatSummary:
    """Summary of data format analysis."""

    total_tensors: int
    dtype_distribution: list[DTypeDistribution]
    layout_distribution: list[LayoutDistribution]
    bfloat16_count: int
    bfloat8_b_count: int
    float32_count: int
    tile_layout_count: int
    row_major_count: int
    bfloat8_b_usage_percent: float
    tile_layout_percent: float
    recommendations: list[str]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "total_tensors": self.total_tensors,
            "dtype_distribution": [d.to_dict() for d in self.dtype_distribution],
            "layout_distribution": [d.to_dict() for d in self.layout_distribution],
            "bfloat16_count": self.bfloat16_count,
            "bfloat8_b_count": self.bfloat8_b_count,
            "float32_count": self.float32_count,
            "tile_layout_count": self.tile_layout_count,
            "row_major_count": self.row_major_count,
            "bfloat8_b_usage_percent": round(self.bfloat8_b_usage_percent, 2),
            "tile_layout_percent": round(self.tile_layout_percent, 2),
            "recommendations": self.recommendations,
        }


@dataclass
class MathFidelitySummary:
    """Summary of math fidelity analysis from performance data."""

    total_operations: int
    fidelity_distribution: list[MathFidelityDistribution]
    lofi_count: int
    hifi2_count: int
    hifi3_count: int
    hifi4_count: int
    lofi_percent: float
    recommendations: list[str]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "total_operations": self.total_operations,
            "fidelity_distribution": [d.to_dict() for d in self.fidelity_distribution],
            "lofi_count": self.lofi_count,
            "hifi2_count": self.hifi2_count,
            "hifi3_count": self.hifi3_count,
            "hifi4_count": self.hifi4_count,
            "lofi_percent": round(self.lofi_percent, 2),
            "recommendations": self.recommendations,
        }


def normalize_dtype(dtype_str: str) -> str:
    """Normalize dtype string to standard format.

    Args:
        dtype_str: Raw dtype string from tensor.

    Returns:
        Normalized dtype string.
    """
    if not dtype_str:
        return "UNKNOWN"

    dtype_upper = dtype_str.upper()

    # Handle various formats: DataType.BFLOAT16, torch.bfloat16, BFLOAT16
    if "BFLOAT8_B" in dtype_upper:
        return "BFLOAT8_B"
    elif "BFLOAT16" in dtype_upper:
        return "BFLOAT16"
    elif "BFLOAT4_B" in dtype_upper:
        return "BFLOAT4_B"
    elif "FLOAT32" in dtype_upper:
        return "FLOAT32"
    elif "FLOAT16" in dtype_upper:
        return "FLOAT16"
    elif "UINT32" in dtype_upper:
        return "UINT32"
    elif "UINT16" in dtype_upper:
        return "UINT16"
    elif "UINT8" in dtype_upper:
        return "UINT8"
    elif "INT32" in dtype_upper:
        return "INT32"

    return dtype_str


def normalize_layout(layout_str: str) -> str:
    """Normalize layout string to standard format.

    Args:
        layout_str: Raw layout string from tensor.

    Returns:
        Normalized layout string.
    """
    if not layout_str:
        return "UNKNOWN"

    layout_upper = layout_str.upper()

    if "TILE" in layout_upper:
        return "TILE"
    elif "ROW_MAJOR" in layout_upper or "STRIDED" in layout_upper:
        return "ROW_MAJOR"

    return layout_str


def normalize_math_fidelity(fidelity_str: str) -> str:
    """Normalize math fidelity string.

    Args:
        fidelity_str: Raw math fidelity string.

    Returns:
        Normalized math fidelity string.
    """
    if not fidelity_str:
        return "UNKNOWN"

    fidelity_upper = fidelity_str.upper()

    if "LOFI" in fidelity_upper:
        return "LoFi"
    elif "HIFI4" in fidelity_upper:
        return "HiFi4"
    elif "HIFI3" in fidelity_upper:
        return "HiFi3"
    elif "HIFI2" in fidelity_upper:
        return "HiFi2"

    return fidelity_str


class DataFormatAnalyzer:
    """Analyzer for tensor data formats."""

    def __init__(self, tensors: list[Tensor]):
        """Initialize with tensor data.

        Args:
            tensors: List of Tensor objects to analyze.
        """
        self.tensors = tensors

    def get_summary(self) -> DataFormatSummary:
        """Get data format summary with recommendations.

        Returns:
            DataFormatSummary with analysis results.
        """
        if not self.tensors:
            return DataFormatSummary(
                total_tensors=0,
                dtype_distribution=[],
                layout_distribution=[],
                bfloat16_count=0,
                bfloat8_b_count=0,
                float32_count=0,
                tile_layout_count=0,
                row_major_count=0,
                bfloat8_b_usage_percent=0,
                tile_layout_percent=0,
                recommendations=["No tensors found"],
            )

        # Count dtypes
        dtype_counts: dict[str, int] = defaultdict(int)
        layout_counts: dict[str, int] = defaultdict(int)

        for tensor in self.tensors:
            dtype = normalize_dtype(tensor.dtype)
            layout = normalize_layout(tensor.layout)
            dtype_counts[dtype] += 1
            layout_counts[layout] += 1

        total = len(self.tensors)

        # Create distributions
        dtype_dist = [
            DTypeDistribution(
                dtype=dtype,
                count=count,
                percent=(count / total * 100) if total > 0 else 0,
            )
            for dtype, count in sorted(dtype_counts.items(), key=lambda x: -x[1])
        ]

        layout_dist = [
            LayoutDistribution(
                layout=layout,
                count=count,
                percent=(count / total * 100) if total > 0 else 0,
            )
            for layout, count in sorted(layout_counts.items(), key=lambda x: -x[1])
        ]

        # Calculate specific counts
        bfloat16_count = dtype_counts.get("BFLOAT16", 0)
        bfloat8_b_count = dtype_counts.get("BFLOAT8_B", 0)
        float32_count = dtype_counts.get("FLOAT32", 0)
        tile_count = layout_counts.get("TILE", 0)
        row_major_count = layout_counts.get("ROW_MAJOR", 0)

        bfloat8_b_percent = (bfloat8_b_count / total * 100) if total > 0 else 0
        tile_percent = (tile_count / total * 100) if total > 0 else 0

        # Generate recommendations
        recommendations = self._generate_recommendations(
            bfloat8_b_percent=bfloat8_b_percent,
            tile_percent=tile_percent,
            bfloat16_count=bfloat16_count,
            float32_count=float32_count,
        )

        return DataFormatSummary(
            total_tensors=total,
            dtype_distribution=dtype_dist,
            layout_distribution=layout_dist,
            bfloat16_count=bfloat16_count,
            bfloat8_b_count=bfloat8_b_count,
            float32_count=float32_count,
            tile_layout_count=tile_count,
            row_major_count=row_major_count,
            bfloat8_b_usage_percent=bfloat8_b_percent,
            tile_layout_percent=tile_percent,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        bfloat8_b_percent: float,
        tile_percent: float,
        bfloat16_count: int,
        float32_count: int,
    ) -> list[str]:
        """Generate optimization recommendations.

        Args:
            bfloat8_b_percent: Percentage of tensors using bfloat8_b.
            tile_percent: Percentage of tensors using TILE layout.
            bfloat16_count: Number of bfloat16 tensors.
            float32_count: Number of float32 tensors.

        Returns:
            List of recommendation strings.
        """
        recommendations = []

        # Check bfloat8_b usage for weights
        if bfloat8_b_percent < BFLOAT8_B_USAGE_LOW_THRESHOLD and bfloat16_count > 0:
            recommendations.append(
                f"Low bfloat8_b usage ({bfloat8_b_percent:.1f}%): "
                "Consider using bfloat8_b for weights (2x memory reduction)"
            )

        # Check TILE layout usage
        if tile_percent < TILE_LAYOUT_LOW_THRESHOLD:
            recommendations.append(
                f"Low TILE layout usage ({tile_percent:.1f}%): "
                "TILE layout is required for most compute operations"
            )

        # Check float32 usage
        if float32_count > 0:
            recommendations.append(
                f"{float32_count} tensors use FLOAT32: "
                "Consider BFLOAT16 for activations to reduce memory"
            )

        if not recommendations:
            recommendations.append("Data format configuration looks good")

        return recommendations

    def get_dtype_distribution(self) -> list[DTypeDistribution]:
        """Get dtype distribution.

        Returns:
            List of DTypeDistribution sorted by count descending.
        """
        if not self.tensors:
            return []

        dtype_counts: dict[str, int] = defaultdict(int)
        for tensor in self.tensors:
            dtype = normalize_dtype(tensor.dtype)
            dtype_counts[dtype] += 1

        total = len(self.tensors)
        return [
            DTypeDistribution(
                dtype=dtype,
                count=count,
                percent=(count / total * 100) if total > 0 else 0,
            )
            for dtype, count in sorted(dtype_counts.items(), key=lambda x: -x[1])
        ]

    def get_layout_distribution(self) -> list[LayoutDistribution]:
        """Get layout distribution.

        Returns:
            List of LayoutDistribution sorted by count descending.
        """
        if not self.tensors:
            return []

        layout_counts: dict[str, int] = defaultdict(int)
        for tensor in self.tensors:
            layout = normalize_layout(tensor.layout)
            layout_counts[layout] += 1

        total = len(self.tensors)
        return [
            LayoutDistribution(
                layout=layout,
                count=count,
                percent=(count / total * 100) if total > 0 else 0,
            )
            for layout, count in sorted(layout_counts.items(), key=lambda x: -x[1])
        ]


class MathFidelityAnalyzer:
    """Analyzer for math fidelity settings from performance data."""

    def __init__(self, operations: list[OperationPerf]):
        """Initialize with operation performance data.

        Args:
            operations: List of OperationPerf objects.
        """
        self.operations = operations

    def get_summary(self) -> MathFidelitySummary:
        """Get math fidelity summary.

        Returns:
            MathFidelitySummary with analysis results.
        """
        if not self.operations:
            return MathFidelitySummary(
                total_operations=0,
                fidelity_distribution=[],
                lofi_count=0,
                hifi2_count=0,
                hifi3_count=0,
                hifi4_count=0,
                lofi_percent=0,
                recommendations=["No operations with math fidelity data"],
            )

        # Count fidelities
        fidelity_counts: dict[str, int] = defaultdict(int)
        ops_with_fidelity = 0

        for op in self.operations:
            if op.math_fidelity:
                fidelity = normalize_math_fidelity(op.math_fidelity)
                fidelity_counts[fidelity] += 1
                ops_with_fidelity += 1

        if ops_with_fidelity == 0:
            return MathFidelitySummary(
                total_operations=len(self.operations),
                fidelity_distribution=[],
                lofi_count=0,
                hifi2_count=0,
                hifi3_count=0,
                hifi4_count=0,
                lofi_percent=0,
                recommendations=["No math fidelity data in performance report"],
            )

        # Create distribution
        fidelity_dist = [
            MathFidelityDistribution(
                fidelity=fidelity,
                count=count,
                percent=(count / ops_with_fidelity * 100),
            )
            for fidelity, count in sorted(fidelity_counts.items(), key=lambda x: -x[1])
        ]

        lofi_count = fidelity_counts.get("LoFi", 0)
        hifi2_count = fidelity_counts.get("HiFi2", 0)
        hifi3_count = fidelity_counts.get("HiFi3", 0)
        hifi4_count = fidelity_counts.get("HiFi4", 0)
        lofi_percent = (lofi_count / ops_with_fidelity * 100) if ops_with_fidelity > 0 else 0

        # Generate recommendations
        recommendations = self._generate_recommendations(
            lofi_percent=lofi_percent,
            hifi4_count=hifi4_count,
            ops_with_fidelity=ops_with_fidelity,
        )

        return MathFidelitySummary(
            total_operations=ops_with_fidelity,
            fidelity_distribution=fidelity_dist,
            lofi_count=lofi_count,
            hifi2_count=hifi2_count,
            hifi3_count=hifi3_count,
            hifi4_count=hifi4_count,
            lofi_percent=lofi_percent,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        lofi_percent: float,
        hifi4_count: int,
        ops_with_fidelity: int,
    ) -> list[str]:
        """Generate math fidelity recommendations.

        Args:
            lofi_percent: Percentage using LoFi.
            hifi4_count: Count of HiFi4 operations.
            ops_with_fidelity: Total operations with fidelity data.

        Returns:
            List of recommendation strings.
        """
        recommendations = []

        if lofi_percent < LOFI_RECOMMENDED_THRESHOLD and ops_with_fidelity > 0:
            recommendations.append(
                f"LoFi usage is {lofi_percent:.1f}%: "
                "Consider starting with LoFi for better performance, increase only if PCC is insufficient"
            )

        if hifi4_count > 0:
            recommendations.append(
                f"{hifi4_count} operations use HiFi4: "
                "HiFi4 has lowest throughput, consider HiFi2/HiFi3 if precision allows"
            )

        if not recommendations:
            recommendations.append("Math fidelity configuration looks reasonable")

        return recommendations

    def get_distribution(self) -> list[MathFidelityDistribution]:
        """Get math fidelity distribution.

        Returns:
            List of MathFidelityDistribution.
        """
        if not self.operations:
            return []

        fidelity_counts: dict[str, int] = defaultdict(int)
        for op in self.operations:
            if op.math_fidelity:
                fidelity = normalize_math_fidelity(op.math_fidelity)
                fidelity_counts[fidelity] += 1

        total = sum(fidelity_counts.values())
        if total == 0:
            return []

        return [
            MathFidelityDistribution(
                fidelity=fidelity,
                count=count,
                percent=(count / total * 100),
            )
            for fidelity, count in sorted(fidelity_counts.items(), key=lambda x: -x[1])
        ]
