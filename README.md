# TTNN Visualizer CLI

A Claude Code-friendly CLI tool for TTNN profiling visualization and analysis.

This project is inspired by [ttnn-visualizer](https://github.com/tenstorrent/ttnn-visualizer). While ttnn-visualizer provides a rich web-based GUI, this CLI tool outputs text-based analysis that works seamlessly with Claude Code and other AI coding assistants.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and set up the project
uv sync
```

## Quick Start

All commands can be run with `uv run ttnn-vis-cli` or by activating the virtual environment first.

### Report Overview

```bash
uv run ttnn-vis-cli info --profiler /path/to/db.sqlite
```

### Operations & Tensors

```bash
# List operations
uv run ttnn-vis-cli operations --profiler /path/to/db.sqlite --top 10

# View tensor details
uv run ttnn-vis-cli tensor 1 --profiler /path/to/db.sqlite
```

### Memory Analysis

```bash
# Memory usage summary
uv run ttnn-vis-cli memory --profiler /path/to/db.sqlite

# L1 memory visualization for an operation
uv run ttnn-vis-cli l1-report 5 --profiler /path/to/db.sqlite
```

### Performance Analysis

```bash
# Performance overview
uv run ttnn-vis-cli perf --performance /path/to/perf-report --top 10

# Detailed performance report with filtering
uv run ttnn-vis-cli perf --performance /path/to/perf-report perf-report --op-code Matmul

# Performance analysis summary
uv run ttnn-vis-cli perf --performance /path/to/perf-report analysis summary

# Identify bottlenecks
uv run ttnn-vis-cli perf --performance /path/to/perf-report analysis bottlenecks
```

### Interactive TUI (Experimental)

> **Note**: The TUI feature is experimental and may have limitations or bugs. Feedback is welcome.

Launch an interactive terminal UI for browsing and exploring profiling data:

```bash
# Launch TUI with profiler data
uv run ttnn-vis-cli tui --profiler /path/to/db.sqlite

# Launch with both profiler and performance data
uv run ttnn-vis-cli tui --profiler /path/to/db.sqlite --performance /path/to/perf-report
```

The TUI provides four tabs:
- **Dashboard**: Overview of profiler report and performance summary
- **Operations**: Browse operations with details panel
- **Tensors**: Browse tensors with memory configuration details
- **Performance**: Performance analysis with metrics and bound information

Keyboard shortcuts: `d` Dashboard, `o` Operations, `t` Tensors, `p` Performance, `q` Quit, `?` Help

## Output Formats

All commands support `--format table|json|csv`. The option must be specified before the command:

```bash
uv run ttnn-vis-cli --format json operations --profiler /path/to/db.sqlite --top 3
```

## Documentation

For detailed command documentation and examples, see:

- [Command Reference](docs/commands.md) - Complete documentation for all commands

## Available Commands

| Command | Description |
|---------|-------------|
| `info` | Report overview and summary |
| `devices` | Device information |
| `operations` | List and search operations |
| `operation <id>` | Operation details |
| `tensors` | List tensors |
| `tensor <id>` | Tensor details |
| `memory` | Memory usage summary |
| `buffers` | List memory buffers |
| `l1-report <op_id>` | L1 memory visualization |
| `perf` | Performance data |
| `perf perf-report` | Detailed performance table |
| `perf summary` | Performance summary |
| `perf analysis op-distribution` | Operation type distribution |
| `perf analysis core-efficiency` | Core efficiency analysis |
| `perf analysis matmul` | Matmul operations analysis |
| `perf analysis conv` | Conv operations analysis |
| `perf analysis bottlenecks` | Bottleneck detection |
| `perf analysis summary` | Overall analysis summary |
| `tui` | Interactive TUI for data browsing |

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
