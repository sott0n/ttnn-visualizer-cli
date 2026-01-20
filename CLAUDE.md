# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CLI tool for analyzing TTNN (Tenstorrent Neural Network) profiling data. Parses SQLite profiler databases and CSV performance reports to provide text-based analysis of operations, tensors, memory usage, and performance metrics.

## Build & Development Commands

```bash
# Install dependencies
uv sync

# Run CLI
uv run ttnn-vis-cli <command>

# Run tests
uv run pytest

# Run single test file
uv run pytest tests/test_profiler_db.py

# Run specific test
uv run pytest tests/test_profiler_db.py::test_get_operations -v

# Run tests with coverage
uv run pytest --cov=ttnn_vis_cli
```

## Architecture

### Data Layer (`src/ttnn_vis_cli/data/`)
- `models.py` - Dataclasses for all domain objects (Device, Operation, Tensor, Buffer, OperationPerf, etc.)
- `profiler_db.py` - `ProfilerDB` class for SQLite database access (reads TTNN profiler db.sqlite files)
- `perf_csv.py` - CSV parser for performance report files (ops_perf_results_*.csv)
- `*_analysis.py` - Analysis modules (sharding, host_overhead, dtype, multi_cq, perf)

### Command Layer (`src/ttnn_vis_cli/commands/`)
Each file corresponds to a CLI command group. Commands use Click decorators and call into the data layer.

### Output Layer (`src/ttnn_vis_cli/output/`)
- `formatter.py` - `OutputFormatter` class handles table/json/csv output formats
- `memory_map.py` - ASCII memory visualization for L1 reports

### TUI (`src/ttnn_vis_cli/tui/`)
Optional Textual-based interactive interface. Requires `[tui]` extra.

### Entry Point
`cli.py` - Main Click group that registers all commands. Entry point is `ttnn_vis_cli.cli:main`.

## Data Sources

The CLI works with two types of data:
1. **Profiler database** (`--profiler`/`-p`): SQLite file (db.sqlite) containing operations, tensors, buffers, devices
2. **Performance report** (`--performance`/`-P`): Directory or CSV file with execution timing and utilization metrics

## Testing

Tests use pytest fixtures from `tests/conftest.py`:
- `temp_db` - Creates in-memory SQLite database with test schema and sample data
- `temp_perf_csv` - Creates temporary CSV with performance test data
