# TTNN Visualizer CLI

CLI tool for TTNN profiling visualization and analysis.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and set up the project
uv sync
```

## Usage

All commands can be run with `uv run ttnn-vis-cli` or by activating the virtual environment first.

### Report Overview

```bash
uv run ttnn-vis-cli info --profiler /path/to/db.sqlite
uv run ttnn-vis-cli info --profiler /path/to/db.sqlite --performance /path/to/perf-report
```

### Device Information

```bash
uv run ttnn-vis-cli devices --profiler /path/to/db.sqlite
```

### Operations

```bash
# List all operations
uv run ttnn-vis-cli operations --profiler /path/to/db.sqlite

# Top 10 operations by duration
uv run ttnn-vis-cli operations --profiler /path/to/db.sqlite --top 10

# Operation details
uv run ttnn-vis-cli operation 1 --profiler /path/to/db.sqlite
```

### Tensors

```bash
# List all tensors
uv run ttnn-vis-cli tensors --profiler /path/to/db.sqlite

# Tensor details
uv run ttnn-vis-cli tensor 1 --profiler /path/to/db.sqlite
```

### Memory

```bash
# Memory usage summary
uv run ttnn-vis-cli memory --profiler /path/to/db.sqlite

# List buffers
uv run ttnn-vis-cli buffers --profiler /path/to/db.sqlite
uv run ttnn-vis-cli buffers --profiler /path/to/db.sqlite --type L1
uv run ttnn-vis-cli buffers --profiler /path/to/db.sqlite --operation 1
```

### Performance

```bash
# Performance data
uv run ttnn-vis-cli perf --performance /path/to/perf-report
uv run ttnn-vis-cli perf --performance /path/to/perf-report --top 10

# Detailed performance report
uv run ttnn-vis-cli perf --performance /path/to/perf-report report

# Performance summary
uv run ttnn-vis-cli perf --performance /path/to/perf-report summary
```

## Output Formats

All commands support multiple output formats:

- `--format table` (default): Human-readable table format
- `--format json`: JSON format for programmatic use
- `--format csv`: CSV format for spreadsheet import

Example:
```bash
uv run ttnn-vis-cli operations --profiler /path/to/db.sqlite --format json
```

## Development

```bash
# Install with dev dependencies
uv sync

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=ttnn_vis_cli
```

## License

MIT
