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

Output:
```
Profiler Report Summary
========================
Path:           /path/to/db.sqlite
Operations:     76
Tensors:        80
Buffers:        2,226
Devices:        1
Total Duration: 15 ns

Devices:
  Device 0:
    Compute Cores: 64 (8x8)
    L1 Memory:     91.5 MB
    L1 for Tensors:0.0 B
```

### Device Information

```bash
uv run ttnn-vis-cli devices --profiler /path/to/db.sqlite
```

Output:
```
Devices
=======

Device 0
--------------------
  Architecture:       N/A
  Chip ID:            0

  Core Configuration:
    Total Cores:      64 (8x8)
    Compute Cores:    64 (8x8)
    Num Compute:      64
    Num Storage:      0

  L1 Memory:
    Worker L1 Size:   1.4 MB
    Total L1 Memory:  91.5 MB
    L1 for Tensors:   0.0 B
    L1 Num Banks:     64
    L1 Bank Size:     1.3 MB
    CB Limit:         1.3 MB
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

Output (`operations`):
```
Operations
==========

  ID  Name                              Duration    Device
----  --------------------------------  ----------  --------
   1  ttnn.from_torch                   0 ns        -
   2  ttnn.from_torch                   0 ns        -
  ...
  33  ttnn.linear                       1 ns        -
  54  ttnn.matmul                       1 ns        -
```

Output (`operations --top 5`):
```
Top 5 Operations by Duration
============================

  ID  Name                                                        Duration    Device
----  ----------------------------------------------------------  ----------  --------
  48  ttnn.transformer.paged_scaled_dot_product_attention_decode  1 ns        -
  31  ttnn.all_gather                                             1 ns        -
  56  ttnn.reduce_scatter                                         1 ns        -
```

### Tensors

```bash
# List all tensors
uv run ttnn-vis-cli tensors --profiler /path/to/db.sqlite

# Tensor details
uv run ttnn-vis-cli tensor 1 --profiler /path/to/db.sqlite
```

Output (`tensor 10`):
```
Tensor 10
=========

Shape:         torch.Size([1, 1, 32, 32])
Data Type:     torch.float32
Layout:        torch.strided
Memory Config: N/A
Device:        N/A
Address:       N/A
Buffer Type:   N/A
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

Output (`memory`):
```
Memory Usage Summary
====================

L1 Memory:
  Used:         1.8 MB
  Total:        0.0 B
  Usage:        0.0%
  Buffer Count: 330

DRAM Memory:
  Used:         489.8 MB
  Total:        0.0 B
  Usage:        0.0%
  Buffer Count: 1,896
```

Output (`buffers --limit 5`):
```
Buffers
=======

  ID  Type    Address    Size      Device    Operation
----  ------  ---------  ------  --------  -----------
   0  DRAM    0x30d4020  6.0 KB         0            1
   1  DRAM    0x30d5820  6.0 KB         0            2
   2  DRAM    0x30d4020  6.0 KB         0            2
   3  L1      0x16d800   2.0 KB         0            3
   4  DRAM    0x30d5820  6.0 KB         0            3
```

### L1 Memory Report

```bash
# L1 memory report for an operation (shows previous and current)
uv run ttnn-vis-cli l1-report 5 --profiler /path/to/db.sqlite

# Without previous operation comparison
uv run ttnn-vis-cli l1-report 5 --profiler /path/to/db.sqlite --no-previous

# Filter by device
uv run ttnn-vis-cli l1-report 5 --profiler /path/to/db.sqlite --device 0

# Show addresses in decimal
uv run ttnn-vis-cli l1-report 5 --profiler /path/to/db.sqlite --no-hex
```

Output:
```
L1 Memory Report - Operation 5: ttnn.matmul
============================================================
Device: 0 | Total L1 for Tensors: 1.4 MB

Previous L1 Report (Operation 4: ttnn.add):
--------------------------------------------------
Memory Map:
|██████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░| 28% used
|   -T12-   -T15-                                  |
0x0                                          0x160000

Address         Size          Shape                 Dtype         Layout            Tensor
--------------------------------------------------------------------------------------------------------
0x00100000      64.0 KB       [1, 32, 128]          BFLOAT16      INTERLEAVED       Tensor 12
0x00110000      32.0 KB       [1, 128, 64]          BFLOAT16      HEIGHT_SHARDED    Tensor 15

Current L1 Report (Operation 5: ttnn.matmul):
--------------------------------------------------
Memory Map:
|█████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░| 42% used
|   -T12-   -T18-  -T15-                           |
0x0                                          0x160000

Address         Size          Shape                 Dtype         Layout            Tensor
--------------------------------------------------------------------------------------------------------
0x00100000      64.0 KB       [1, 32, 128]          BFLOAT16      INTERLEAVED       Tensor 12
0x00110000      48.0 KB       [1, 32, 64]           BFLOAT16      BLOCK_SHARDED     Tensor 18 (new)
0x0011C000      32.0 KB       [1, 128, 64]          BFLOAT16      HEIGHT_SHARDED    Tensor 15
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

Output (`perf --top 5`):
```
Top 5 Operations by Execution Time
==================================

Op Name    Op Code    Exec Time      Cores    Math Util %    DRAM R BW %
---------  ---------  -----------  -------  -------------  -------------
           Matmul     64.531 µs         64              0              0
           Matmul     64.374 µs         64              0              0
           Matmul     64.330 µs         64              0              0
           Matmul     63.739 µs         64              0              0
           Matmul     60.819 µs         64              0              0
```

Output (`perf summary`):
```
Performance Summary
===================

CSV File:           ops_perf_results_2025_07_16_18_41_46.csv
Total Operations:   68
Total Exec Time:    1.103 ms
Avg Exec Time:      16.214 µs
Max Exec Time:      64.531 µs
Min Exec Time:      601 ns
Avg Math Util:      0.0%
```

## Output Formats

All commands support multiple output formats. The `--format` option is a global option and must be specified before the command:

- `--format table` (default): Human-readable table format
- `--format json`: JSON format for programmatic use
- `--format csv`: CSV format for spreadsheet import

Example:
```bash
uv run ttnn-vis-cli --format json operations --profiler /path/to/db.sqlite --top 3
```

Output (JSON):
```json
{
  "data": [
    {
      "id": 48,
      "name": "ttnn.transformer.paged_scaled_dot_product_attention_decode",
      "duration": 1.385,
      "device_id": null
    }
  ],
  "title": "Top 3 Operations by Duration"
}
```

Output (CSV):
```csv
id,name,duration,device_id,stack_trace_id,captured_graph_id
48,ttnn.transformer.paged_scaled_dot_product_attention_decode,1.385,,,
31,ttnn.all_gather,1.254,,,
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
