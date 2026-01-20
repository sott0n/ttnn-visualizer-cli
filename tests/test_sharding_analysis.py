"""Tests for sharding analysis module."""

import pytest

from ttnn_vis_cli.data.models import Tensor
from ttnn_vis_cli.data.sharding_analysis import (
    ShardingAnalyzer,
    detect_reshards,
    parse_buffer_type,
    parse_sharding_strategy,
)


class TestParseShardingStrategy:
    """Tests for parse_sharding_strategy function."""

    def test_height_sharded(self):
        config = "MemoryConfig(TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1)"
        assert parse_sharding_strategy(config) == "HEIGHT_SHARDED"

    def test_width_sharded(self):
        config = "MemoryConfig(TensorMemoryLayout::WIDTH_SHARDED, BufferType::L1)"
        assert parse_sharding_strategy(config) == "WIDTH_SHARDED"

    def test_block_sharded(self):
        config = "MemoryConfig(TensorMemoryLayout::BLOCK_SHARDED, BufferType::L1)"
        assert parse_sharding_strategy(config) == "BLOCK_SHARDED"

    def test_interleaved(self):
        config = "MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM)"
        assert parse_sharding_strategy(config) == "INTERLEAVED"

    def test_single_bank(self):
        config = "MemoryConfig(TensorMemoryLayout::SINGLE_BANK, BufferType::L1)"
        assert parse_sharding_strategy(config) == "SINGLE_BANK"

    def test_unknown(self):
        config = "SomeOtherConfig()"
        assert parse_sharding_strategy(config) == "UNKNOWN"

    def test_none(self):
        assert parse_sharding_strategy(None) == "UNKNOWN"

    def test_empty_string(self):
        assert parse_sharding_strategy("") == "UNKNOWN"


class TestParseBufferType:
    """Tests for parse_buffer_type function."""

    def test_l1_from_buffer_type(self):
        assert parse_buffer_type(None, "L1") == "L1"

    def test_dram_from_buffer_type(self):
        assert parse_buffer_type(None, "DRAM") == "DRAM"

    def test_l1_from_memory_config(self):
        config = "MemoryConfig(TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1)"
        assert parse_buffer_type(config, None) == "L1"

    def test_dram_from_memory_config(self):
        config = "MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM)"
        assert parse_buffer_type(config, None) == "DRAM"

    def test_unknown(self):
        assert parse_buffer_type(None, None) == "UNKNOWN"


class TestShardingAnalyzer:
    """Tests for ShardingAnalyzer class."""

    @pytest.fixture
    def sample_tensors(self):
        """Create sample tensors for testing."""
        return [
            Tensor(
                id=1,
                shape="[1, 64, 224, 224]",
                dtype="bfloat16",
                layout="TILE",
                memory_config="MemoryConfig(TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1)",
                buffer_type="L1",
            ),
            Tensor(
                id=2,
                shape="[1, 128, 112, 112]",
                dtype="bfloat16",
                layout="TILE",
                memory_config="MemoryConfig(TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1)",
                buffer_type="L1",
            ),
            Tensor(
                id=3,
                shape="[64, 64, 3, 3]",
                dtype="bfloat8_b",
                layout="TILE",
                memory_config="MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM)",
                buffer_type="DRAM",
            ),
            Tensor(
                id=4,
                shape="[1, 256, 56, 56]",
                dtype="bfloat16",
                layout="TILE",
                memory_config="MemoryConfig(TensorMemoryLayout::BLOCK_SHARDED, BufferType::L1)",
                buffer_type="L1",
            ),
        ]

    def test_distribution(self, sample_tensors):
        analyzer = ShardingAnalyzer(sample_tensors)
        distribution = analyzer.get_distribution()

        assert len(distribution) == 3  # HEIGHT_SHARDED, INTERLEAVED, BLOCK_SHARDED
        # Should be sorted by count descending
        assert distribution[0].strategy == "HEIGHT_SHARDED"
        assert distribution[0].count == 2

    def test_summary(self, sample_tensors):
        analyzer = ShardingAnalyzer(sample_tensors)
        summary = analyzer.get_summary()

        assert summary.total_tensors == 4
        assert summary.height_sharded_count == 2
        assert summary.interleaved_count == 1
        assert summary.block_sharded_count == 1
        assert summary.sharded_percent == 75.0  # 3 out of 4

    def test_get_all_tensor_shardings_with_filter(self, sample_tensors):
        analyzer = ShardingAnalyzer(sample_tensors)

        # Filter by strategy
        height_sharded = analyzer.get_all_tensor_shardings(strategy_filter="HEIGHT_SHARDED")
        assert len(height_sharded) == 2

        # Filter by buffer
        l1_tensors = analyzer.get_all_tensor_shardings(buffer_filter="L1")
        assert len(l1_tensors) == 3

    def test_get_tensor_sharding(self, sample_tensors):
        analyzer = ShardingAnalyzer(sample_tensors)
        info = analyzer.get_tensor_sharding(1)

        assert info is not None
        assert info.tensor_id == 1
        assert info.sharding_strategy == "HEIGHT_SHARDED"


class TestDetectReshards:
    """Tests for detect_reshards function."""

    @pytest.fixture
    def operations_with_tensors(self):
        """Create sample operations with tensors."""
        return [
            (
                1,
                "ttnn.conv2d",
                [
                    Tensor(
                        id=1,
                        shape="[1, 64, 224, 224]",
                        dtype="bfloat16",
                        layout="TILE",
                        memory_config="MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM)",
                    )
                ],
                [
                    Tensor(
                        id=2,
                        shape="[1, 64, 224, 224]",
                        dtype="bfloat16",
                        layout="TILE",
                        memory_config="MemoryConfig(TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1)",
                    )
                ],
            ),
            (
                2,
                "ttnn.relu",
                [
                    Tensor(
                        id=2,
                        shape="[1, 64, 224, 224]",
                        dtype="bfloat16",
                        layout="TILE",
                        memory_config="MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM)",
                    )
                ],
                [
                    Tensor(
                        id=3,
                        shape="[1, 64, 224, 224]",
                        dtype="bfloat16",
                        layout="TILE",
                        memory_config="MemoryConfig(TensorMemoryLayout::INTERLEAVED, BufferType::DRAM)",
                    )
                ],
            ),
        ]

    def test_detects_reshard(self, operations_with_tensors):
        reshards = detect_reshards(operations_with_tensors)

        assert len(reshards) == 2
        # Second operation has reshard: previous output was HEIGHT_SHARDED, input is INTERLEAVED
        assert reshards[1].has_reshard is True
        assert reshards[1].reshard_detail == "HEIGHT_SHARDED -> INTERLEAVED"

    def test_no_reshard_when_consistent(self):
        """Test no reshard detected when sharding is consistent."""
        consistent_ops = [
            (
                1,
                "ttnn.conv2d",
                [],
                [
                    Tensor(
                        id=1,
                        shape="[1, 64, 224, 224]",
                        dtype="bfloat16",
                        layout="TILE",
                        memory_config="MemoryConfig(TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1)",
                    )
                ],
            ),
            (
                2,
                "ttnn.relu",
                [
                    Tensor(
                        id=1,
                        shape="[1, 64, 224, 224]",
                        dtype="bfloat16",
                        layout="TILE",
                        memory_config="MemoryConfig(TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1)",
                    )
                ],
                [
                    Tensor(
                        id=2,
                        shape="[1, 64, 224, 224]",
                        dtype="bfloat16",
                        layout="TILE",
                        memory_config="MemoryConfig(TensorMemoryLayout::HEIGHT_SHARDED, BufferType::L1)",
                    )
                ],
            ),
        ]

        reshards = detect_reshards(consistent_ops)

        assert reshards[0].has_reshard is False
        assert reshards[1].has_reshard is False
