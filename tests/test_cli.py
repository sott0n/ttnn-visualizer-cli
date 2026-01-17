"""Tests for CLI commands."""

import json

from click.testing import CliRunner

from ttnn_vis_cli.cli import cli


def test_cli_version():
    """Test CLI version command."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_cli_help():
    """Test CLI help command."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "TTNN Visualizer CLI" in result.output


def test_info_command(temp_db):
    """Test info command."""
    runner = CliRunner()
    result = runner.invoke(cli, ["info", "--profiler", temp_db])
    assert result.exit_code == 0
    assert "Operations:" in result.output
    assert "Tensors:" in result.output


def test_info_command_json(temp_db):
    """Test info command with JSON output."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--format", "json", "info", "--profiler", temp_db])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "data" in data
    assert "profiler" in data["data"]


def test_devices_command(temp_db):
    """Test devices command."""
    runner = CliRunner()
    result = runner.invoke(cli, ["devices", "--profiler", temp_db])
    assert result.exit_code == 0
    assert "Device 0" in result.output


def test_devices_command_json(temp_db):
    """Test devices command with JSON output."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--format", "json", "devices", "--profiler", temp_db])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "data" in data
    assert len(data["data"]) == 1


def test_operations_command(temp_db):
    """Test operations command."""
    runner = CliRunner()
    result = runner.invoke(cli, ["operations", "--profiler", temp_db])
    assert result.exit_code == 0
    assert "ttnn.matmul" in result.output


def test_operations_command_top(temp_db):
    """Test operations command with --top."""
    runner = CliRunner()
    result = runner.invoke(cli, ["operations", "--profiler", temp_db, "--top", "2"])
    assert result.exit_code == 0
    assert "ttnn.matmul" in result.output


def test_operations_command_json(temp_db):
    """Test operations command with JSON output."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--format", "json", "operations", "--profiler", temp_db])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "data" in data
    assert len(data["data"]) == 3


def test_operation_command(temp_db):
    """Test operation detail command."""
    runner = CliRunner()
    result = runner.invoke(cli, ["operation", "1", "--profiler", temp_db])
    assert result.exit_code == 0
    assert "ttnn.matmul" in result.output
    assert "transpose_a" in result.output


def test_operation_command_not_found(temp_db):
    """Test operation command with non-existent ID."""
    runner = CliRunner()
    result = runner.invoke(cli, ["operation", "999", "--profiler", temp_db])
    assert result.exit_code != 0
    assert "not found" in result.output


def test_tensors_command(temp_db):
    """Test tensors command."""
    runner = CliRunner()
    result = runner.invoke(cli, ["tensors", "--profiler", temp_db])
    assert result.exit_code == 0
    assert "bfloat16" in result.output


def test_tensors_command_json(temp_db):
    """Test tensors command with JSON output."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--format", "json", "tensors", "--profiler", temp_db])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "data" in data
    assert len(data["data"]) == 3


def test_tensor_command(temp_db):
    """Test tensor detail command."""
    runner = CliRunner()
    result = runner.invoke(cli, ["tensor", "1", "--profiler", temp_db])
    assert result.exit_code == 0
    assert "[1, 32, 128, 128]" in result.output


def test_memory_command(temp_db):
    """Test memory command."""
    runner = CliRunner()
    result = runner.invoke(cli, ["memory", "--profiler", temp_db])
    assert result.exit_code == 0
    assert "L1 Memory:" in result.output
    assert "DRAM Memory:" in result.output


def test_memory_command_json(temp_db):
    """Test memory command with JSON output."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--format", "json", "memory", "--profiler", temp_db])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "data" in data
    assert "l1_used" in data["data"]


def test_buffers_command(temp_db):
    """Test buffers command."""
    runner = CliRunner()
    result = runner.invoke(cli, ["buffers", "--profiler", temp_db])
    assert result.exit_code == 0
    assert "L1" in result.output or "DRAM" in result.output


def test_buffers_command_filter_type(temp_db):
    """Test buffers command with type filter."""
    runner = CliRunner()
    result = runner.invoke(cli, ["buffers", "--profiler", temp_db, "--type", "L1"])
    assert result.exit_code == 0


def test_buffers_command_filter_operation(temp_db):
    """Test buffers command with operation filter."""
    runner = CliRunner()
    result = runner.invoke(cli, ["buffers", "--profiler", temp_db, "--operation", "1"])
    assert result.exit_code == 0


def test_perf_command(temp_perf_csv):
    """Test perf command."""
    runner = CliRunner()
    result = runner.invoke(cli, ["perf", "--performance", temp_perf_csv])
    assert result.exit_code == 0
    assert "ttnn.matmul" in result.output


def test_perf_command_top(temp_perf_csv):
    """Test perf command with --top."""
    runner = CliRunner()
    result = runner.invoke(cli, ["perf", "--performance", temp_perf_csv, "--top", "2"])
    assert result.exit_code == 0
    assert "ttnn.matmul" in result.output


def test_perf_command_json(temp_perf_csv):
    """Test perf command with JSON output."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--format", "json", "perf", "--performance", temp_perf_csv])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "data" in data
    assert len(data["data"]) == 3


def test_perf_report_command(temp_perf_csv):
    """Test perf report command."""
    runner = CliRunner()
    result = runner.invoke(cli, ["perf", "--performance", temp_perf_csv, "report"])
    assert result.exit_code == 0
    assert "Performance Report" in result.output
    assert "Summary:" in result.output


def test_perf_summary_command(temp_perf_csv):
    """Test perf summary command."""
    runner = CliRunner()
    result = runner.invoke(cli, ["perf", "--performance", temp_perf_csv, "summary"])
    assert result.exit_code == 0
    assert "Performance Summary" in result.output
