"""Sharding analysis logic for TTNN profiling data."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from .models import Tensor

# Thresholds for recommendations
INTERLEAVED_WARNING_THRESHOLD = 50  # Warn if INTERLEAVED usage exceeds this %
RESHARD_WARNING_THRESHOLD = 10  # Warn if reshard count exceeds this


@dataclass
class ShardingDistribution:
    """Sharding strategy distribution data."""

    strategy: str
    count: int
    percent: float
    l1_count: int
    dram_count: int

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "strategy": self.strategy,
            "count": self.count,
            "percent": self.percent,
            "l1_count": self.l1_count,
            "dram_count": self.dram_count,
        }


@dataclass
class OperationSharding:
    """Sharding information for an operation."""

    operation_id: int
    operation_name: str
    input_shardings: list[str]
    output_shardings: list[str]
    has_reshard: bool
    reshard_detail: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "operation_id": self.operation_id,
            "operation_name": self.operation_name,
            "input_shardings": self.input_shardings,
            "output_shardings": self.output_shardings,
            "has_reshard": self.has_reshard,
            "reshard_detail": self.reshard_detail,
        }


@dataclass
class TensorShardingInfo:
    """Detailed sharding information for a tensor."""

    tensor_id: int
    shape: str
    dtype: str
    layout: str
    buffer_type: str
    sharding_strategy: str
    memory_config: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "tensor_id": self.tensor_id,
            "shape": self.shape,
            "dtype": self.dtype,
            "layout": self.layout,
            "buffer_type": self.buffer_type,
            "sharding_strategy": self.sharding_strategy,
            "memory_config": self.memory_config,
        }


@dataclass
class ShardingSummary:
    """Overall sharding analysis summary."""

    total_tensors: int
    height_sharded_count: int
    width_sharded_count: int
    block_sharded_count: int
    interleaved_count: int
    single_bank_count: int
    unknown_count: int
    sharded_percent: float
    interleaved_percent: float
    reshard_count: int
    recommendations: list[str]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON output."""
        return {
            "total_tensors": self.total_tensors,
            "height_sharded_count": self.height_sharded_count,
            "width_sharded_count": self.width_sharded_count,
            "block_sharded_count": self.block_sharded_count,
            "interleaved_count": self.interleaved_count,
            "single_bank_count": self.single_bank_count,
            "unknown_count": self.unknown_count,
            "sharded_percent": self.sharded_percent,
            "interleaved_percent": self.interleaved_percent,
            "reshard_count": self.reshard_count,
            "recommendations": self.recommendations,
        }


def parse_sharding_strategy(memory_config: Optional[str]) -> str:
    """Parse sharding strategy from memory_config string.

    Args:
        memory_config: Memory configuration string from tensor.

    Returns:
        Sharding strategy name.
    """
    if not memory_config:
        return "UNKNOWN"

    memory_config_upper = memory_config.upper()

    if "HEIGHT_SHARDED" in memory_config_upper:
        return "HEIGHT_SHARDED"
    elif "WIDTH_SHARDED" in memory_config_upper:
        return "WIDTH_SHARDED"
    elif "BLOCK_SHARDED" in memory_config_upper:
        return "BLOCK_SHARDED"
    elif "INTERLEAVED" in memory_config_upper:
        return "INTERLEAVED"
    elif "SINGLE_BANK" in memory_config_upper:
        return "SINGLE_BANK"

    return "UNKNOWN"


def parse_buffer_type(memory_config: Optional[str], buffer_type: Optional[str]) -> str:
    """Parse buffer type from memory_config or buffer_type field.

    Args:
        memory_config: Memory configuration string.
        buffer_type: Buffer type field value.

    Returns:
        Buffer type string (L1 or DRAM).
    """
    if buffer_type:
        buffer_type_upper = str(buffer_type).upper()
        if "DRAM" in buffer_type_upper:
            return "DRAM"
        elif "L1" in buffer_type_upper:
            return "L1"

    if memory_config:
        memory_config_upper = memory_config.upper()
        if "DRAM" in memory_config_upper:
            return "DRAM"
        elif "L1" in memory_config_upper:
            return "L1"

    return "UNKNOWN"


class ShardingAnalyzer:
    """Analyzer for tensor sharding strategies."""

    def __init__(self, tensors: list[Tensor]):
        """Initialize with tensor data.

        Args:
            tensors: List of Tensor objects to analyze.
        """
        self.tensors = tensors
        self._tensor_shardings: dict[int, TensorShardingInfo] = {}
        self._parse_all_tensors()

    def _parse_all_tensors(self) -> None:
        """Parse sharding information for all tensors."""
        for tensor in self.tensors:
            strategy = parse_sharding_strategy(tensor.memory_config)
            buffer_type = parse_buffer_type(tensor.memory_config, tensor.buffer_type)

            self._tensor_shardings[tensor.id] = TensorShardingInfo(
                tensor_id=tensor.id,
                shape=tensor.shape,
                dtype=tensor.dtype,
                layout=tensor.layout,
                buffer_type=buffer_type,
                sharding_strategy=strategy,
                memory_config=tensor.memory_config or "",
            )

    def get_distribution(self) -> list[ShardingDistribution]:
        """Get sharding strategy distribution.

        Returns:
            List of ShardingDistribution sorted by count descending.
        """
        if not self._tensor_shardings:
            return []

        # Count by strategy
        strategy_counts: dict[str, dict] = defaultdict(
            lambda: {"count": 0, "l1": 0, "dram": 0}
        )

        for info in self._tensor_shardings.values():
            strategy_counts[info.sharding_strategy]["count"] += 1
            if info.buffer_type == "L1":
                strategy_counts[info.sharding_strategy]["l1"] += 1
            elif info.buffer_type == "DRAM":
                strategy_counts[info.sharding_strategy]["dram"] += 1

        total = len(self._tensor_shardings)
        distributions = []

        for strategy, counts in strategy_counts.items():
            distributions.append(
                ShardingDistribution(
                    strategy=strategy,
                    count=counts["count"],
                    percent=(counts["count"] / total * 100) if total > 0 else 0,
                    l1_count=counts["l1"],
                    dram_count=counts["dram"],
                )
            )

        # Sort by count descending
        distributions.sort(key=lambda x: x.count, reverse=True)
        return distributions

    def get_tensor_sharding(self, tensor_id: int) -> Optional[TensorShardingInfo]:
        """Get sharding information for a specific tensor.

        Args:
            tensor_id: Tensor ID.

        Returns:
            TensorShardingInfo or None if not found.
        """
        return self._tensor_shardings.get(tensor_id)

    def get_all_tensor_shardings(
        self,
        strategy_filter: Optional[str] = None,
        buffer_filter: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[TensorShardingInfo]:
        """Get sharding information for all tensors with optional filtering.

        Args:
            strategy_filter: Filter by sharding strategy.
            buffer_filter: Filter by buffer type (L1/DRAM).
            limit: Maximum number of results.

        Returns:
            List of TensorShardingInfo.
        """
        results = list(self._tensor_shardings.values())

        if strategy_filter:
            results = [r for r in results if r.sharding_strategy == strategy_filter.upper()]

        if buffer_filter:
            results = [r for r in results if r.buffer_type == buffer_filter.upper()]

        if limit:
            results = results[:limit]

        return results

    def get_summary(self, reshard_count: int = 0) -> ShardingSummary:
        """Get overall sharding summary with recommendations.

        Args:
            reshard_count: Number of reshard operations detected.

        Returns:
            ShardingSummary with analysis and recommendations.
        """
        if not self._tensor_shardings:
            return ShardingSummary(
                total_tensors=0,
                height_sharded_count=0,
                width_sharded_count=0,
                block_sharded_count=0,
                interleaved_count=0,
                single_bank_count=0,
                unknown_count=0,
                sharded_percent=0,
                interleaved_percent=0,
                reshard_count=0,
                recommendations=[],
            )

        # Count by strategy
        counts = defaultdict(int)
        for info in self._tensor_shardings.values():
            counts[info.sharding_strategy] += 1

        total = len(self._tensor_shardings)
        sharded_count = (
            counts["HEIGHT_SHARDED"]
            + counts["WIDTH_SHARDED"]
            + counts["BLOCK_SHARDED"]
        )

        # Generate recommendations
        recommendations = []

        # Check if many tensors are not sharded
        interleaved_percent = (counts["INTERLEAVED"] / total * 100) if total > 0 else 0
        if interleaved_percent > INTERLEAVED_WARNING_THRESHOLD:
            recommendations.append(
                f"High INTERLEAVED usage ({interleaved_percent:.1f}%): "
                "Consider sharding for better L1 utilization"
            )

        # Check for reshard operations
        if reshard_count > RESHARD_WARNING_THRESHOLD:
            recommendations.append(
                f"High reshard count ({reshard_count}): "
                "Consider consistent sharding strategy across operations"
            )

        # Check for HEIGHT_SHARDED dominance (recommended for most cases)
        if sharded_count > 0:
            height_ratio = counts["HEIGHT_SHARDED"] / sharded_count
            if height_ratio < 0.5 and counts["HEIGHT_SHARDED"] < counts["WIDTH_SHARDED"]:
                recommendations.append(
                    "Consider HEIGHT_SHARDED for most operations "
                    "(recommended for spatial operations)"
                )

        # Check for unknown sharding
        if counts["UNKNOWN"] > 0:
            recommendations.append(
                f"{counts['UNKNOWN']} tensors have unknown sharding strategy"
            )

        if not recommendations:
            recommendations.append("Sharding configuration looks reasonable")

        return ShardingSummary(
            total_tensors=total,
            height_sharded_count=counts["HEIGHT_SHARDED"],
            width_sharded_count=counts["WIDTH_SHARDED"],
            block_sharded_count=counts["BLOCK_SHARDED"],
            interleaved_count=counts["INTERLEAVED"],
            single_bank_count=counts["SINGLE_BANK"],
            unknown_count=counts["UNKNOWN"],
            sharded_percent=(sharded_count / total * 100) if total > 0 else 0,
            interleaved_percent=interleaved_percent,
            reshard_count=reshard_count,
            recommendations=recommendations,
        )


def detect_reshards(
    operations_with_tensors: list[tuple[int, str, list[Tensor], list[Tensor]]]
) -> list[OperationSharding]:
    """Detect reshard operations between consecutive operations.

    Args:
        operations_with_tensors: List of (op_id, op_name, input_tensors, output_tensors).

    Returns:
        List of OperationSharding with reshard detection.
    """
    results = []
    prev_output_shardings: list[str] = []

    for op_id, op_name, inputs, outputs in operations_with_tensors:
        input_shardings = [
            parse_sharding_strategy(t.memory_config) for t in inputs
        ]
        output_shardings = [
            parse_sharding_strategy(t.memory_config) for t in outputs
        ]

        # Detect reshard: input sharding differs from previous output
        has_reshard = False
        reshard_detail = None

        if prev_output_shardings and input_shardings:
            # Check if any input tensor has different sharding than previous outputs
            for in_shard in input_shardings:
                if in_shard != "UNKNOWN":
                    for out_shard in prev_output_shardings:
                        if out_shard != "UNKNOWN" and in_shard != out_shard:
                            has_reshard = True
                            reshard_detail = f"{out_shard} -> {in_shard}"
                            break
                    if has_reshard:
                        break

        results.append(
            OperationSharding(
                operation_id=op_id,
                operation_name=op_name,
                input_shardings=input_shardings,
                output_shardings=output_shardings,
                has_reshard=has_reshard,
                reshard_detail=reshard_detail,
            )
        )

        # Update previous output shardings for next iteration
        if output_shardings:
            prev_output_shardings = output_shardings

    return results
