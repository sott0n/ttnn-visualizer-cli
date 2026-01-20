# Command Reference

Detailed documentation for all TTNN Visualizer CLI commands.

## Table of Contents

- [Interactive TUI](#interactive-tui)
- [Report Overview](#report-overview)
- [Device Information](#device-information)
- [Operations](#operations)
- [Tensors](#tensors)
- [Memory](#memory)
- [L1 Memory Report](#l1-memory-report)
- [Performance](#performance)
- [Performance Analysis](#performance-analysis)
- [Output Formats](#output-formats)

## Interactive TUI

Launch an interactive terminal user interface for browsing and exploring profiling data.

```bash
# Launch TUI with profiler data only
uv run ttnn-vis-cli tui --profiler /path/to/db.sqlite

# Launch with both profiler and performance data
uv run ttnn-vis-cli tui --profiler /path/to/db.sqlite --performance /path/to/perf-report

# Short options
uv run ttnn-vis-cli tui -p /path/to/db.sqlite -P /path/to/perf-report
```

### TUI Tabs

The TUI provides four main tabs for exploring different aspects of profiling data:

**Dashboard Tab**
- Profiler report summary (operations, tensors, buffers, devices)
- Performance overview (total ops, total time, avg FPU utilization)
- Device information (cores, L1 memory)
- Memory usage (L1 and DRAM)

**Operations Tab**
- Scrollable table of all operations
- Columns: ID, Name, Duration, Device
- Detail panel showing:
  - Operation name and duration
  - Input and output tensors
  - Operation arguments

**Tensors Tab**
- Scrollable table of all tensors
- Columns: ID, Shape, Dtype, Layout, Memory
- Detail panel showing:
  - Full shape and data type
  - Memory configuration (layout type, address)
  - Device information

**Performance Tab**
- Summary panel with:
  - Operation counts by bound type (Compute/Memory/Balanced)
  - Device time and gap time totals
  - Average FPU and DRAM utilization
  - Top operation codes
- Scrollable table of operations
- Columns: ID, Op Code, Time, FPU%, Bound, Cores
- Detail panel showing:
  - Timing details (device time, host time, op-to-op gap)
  - Performance metrics (cores, bound type, utilization)
  - Performance model data (ideal/compute/bandwidth times, efficiency)
  - Memory and shape information

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `d` | Switch to Dashboard tab |
| `o` | Switch to Operations tab |
| `t` | Switch to Tensors tab |
| `p` | Switch to Performance tab |
| `↑/↓` | Navigate table rows |
| `Enter` | Select row to view details |
| `Tab` | Switch between tabs |
| `?` | Show help |
| `q` | Quit |

## Report Overview

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

## Device Information

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

## Operations

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

## Tensors

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

## Memory

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

## L1 Memory Report

Visualize L1 memory allocation for operations with memory map and tensor details.

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

## Performance

### Basic Performance Commands

```bash
# Performance data
uv run ttnn-vis-cli perf --performance /path/to/perf-report
uv run ttnn-vis-cli perf --performance /path/to/perf-report --top 10

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

### Detailed Performance Report

```bash
# Detailed performance report table
uv run ttnn-vis-cli perf --performance /path/to/perf-report perf-report
uv run ttnn-vis-cli perf --performance /path/to/perf-report perf-report -l 20

# Sort by device time (descending by default)
uv run ttnn-vis-cli perf --performance /path/to/perf-report perf-report --sort-by device_time
uv run ttnn-vis-cli perf --performance /path/to/perf-report perf-report --sort-by device_time --asc

# Filter by op code
uv run ttnn-vis-cli perf --performance /path/to/perf-report perf-report --op-code Matmul

# Filter by bound type (Compute, Memory, Balanced)
uv run ttnn-vis-cli perf --performance /path/to/perf-report perf-report --bound Compute

# Filter by buffer type (L1, DRAM, System)
uv run ttnn-vis-cli perf --performance /path/to/perf-report perf-report --buffer-type L1

# Filter by device time range (in microseconds)
uv run ttnn-vis-cli perf --performance /path/to/perf-report perf-report --min-time 10 --max-time 100

# Combine sort and filter
uv run ttnn-vis-cli perf --performance /path/to/perf-report perf-report --op-code Conv2d --sort-by device_time -l 10
```

Output (`perf perf-report -l 10`):
```
Performance Report
==================

ID        Total     Bound     Op Code               Device ID    Buffer Type    Layout     Device Time    Op-to-Op Gap    Cores     DRAM      DRAM %    FLOPs     FLOPs %    Math Fidelity
--------  --------  --------  --------------------  -----------  -------------  ---------  -------------  --------------  --------  --------  --------  --------  ---------  ---------------
1024      1         Compute   Pad                   0            L1             ROW_MAJOR  14.325 µs      -               128       -         -         12926     90.2       -
2048      2         Compute   Transpose             0            L1             ROW_MAJOR  9.747 µs       659.144 ms      128       -         -         2543      26.1       HiFi4
3072      3         Compute   Pad                   0            L1             ROW_MAJOR  4.564 µs       661.436 ms      128       -         -         2895      63.4       -
4096      4         Compute   Transpose             0            L1             ROW_MAJOR  16.000 µs      692.271 ms      128       -         -         2895      18.1       -
5120      5         Compute   Transpose             0            L1             ROW_MAJOR  104.497 µs     677.559 ms      128       -         -         2895      2.8        HiFi4
6144      6         Compute   Transpose             0            L1             ROW_MAJOR  43.113 µs      751.440 ms      128       -         -         2895      6.7        -
7168      7         Compute   SliceDeviceOperation  0            L1             ROW_MAJOR  46.750 µs      754.352 ms      128       -         -         2366      5.1        -
8192      8         Balanced  HaloDeviceOperation   0            L1             ROW_MAJOR  17.116 µs      764.166 ms      128       -         -         1         0.0        -
9216      9         Compute   Conv2d                0            L1             ROW_MAJOR  85.299 µs      915.767 ms      130       -         -         20586     24.1       LoFi
────────  ────────  ────────  ────────              ────────     ────────       ────────   ────────       ────────        ────────  ────────  ────────  ────────  ────────   ────────
Total     9                   5 types                                                      341.411 µs     5.876 s
```

Output (`perf perf-report --op-code Conv2d --sort-by device_time -l 5`):
```
Performance Report
==================

ID        Total     Bound     Op Code    Device ID    Buffer Type    Layout     Device Time    Op-to-Op Gap    Cores     DRAM      DRAM %    FLOPs     FLOPs %    Math Fidelity
--------  --------  --------  ---------  -----------  -------------  ---------  -------------  --------------  --------  --------  --------  --------  ---------  ---------------
9216      1         Compute   Conv2d     0            L1             ROW_MAJOR  85.299 µs      915.767 ms      130       -         -         20586     24.1       LoFi
30720     2         Compute   Conv2d     0            L1             ROW_MAJOR  45.348 µs      818.453 ms      130       -         -         11580     25.5       LoFi
25600     3         Compute   Conv2d     0            L1             ROW_MAJOR  45.146 µs      818.749 ms      130       -         -         11580     25.6       LoFi
16384     4         Compute   Conv2d     0            L1             ROW_MAJOR  41.591 µs      926.234 ms      130       -         -         11580     27.8       LoFi
35840     5         Compute   Conv2d     0            L1             ROW_MAJOR  30.504 µs      835.500 ms      130       -         -         11580     38.0       LoFi
────────  ────────  ────────  ────────   ────────     ────────       ────────   ────────       ────────        ────────  ────────  ────────  ────────  ────────   ────────
Total     5                   1 types                                           247.888 µs     4.315 s
```

## Performance Analysis

The `perf analysis` subcommand group provides detailed performance insights.

### Operation Distribution

```bash
uv run ttnn-vis-cli perf --performance /path/to/perf-report analysis op-distribution
```

Output:
```
Operation Type Distribution
===========================

Op Code                     Count   Total Time     Avg Time   % Time  % Count
--------------------------------------------------------------------------------
Conv2d                         20   739.577 µs    36.979 µs    34.1%    17.2%
Matmul                         34   582.365 µs    17.128 µs    26.9%    29.3%
HaloDeviceOperation            22   278.217 µs    12.646 µs    12.8%    19.0%
...

Summary:
  Total: 116 operations, 2.167 ms
  Top 3 by time: Conv2d (34.1%), Matmul (26.9%), HaloDeviceOperation (12.8%)
```

### Core Efficiency

```bash
uv run ttnn-vis-cli perf --performance /path/to/perf-report analysis core-efficiency
```

Output:
```
Core Efficiency Analysis
========================

 Cores  Op Count   Total Time     Avg Time  Avg FPU% Bound Distribution
------------------------------------------------------------------------------------------
    64         3    47.310 µs    15.770 µs     58.4% Compute: 2, Memory: 0, Balanced: 1
   128        10   468.002 µs    46.800 µs     22.1% Compute: 8, Memory: 0, Balanced: 2
...

Insights:
  - 64-core operations show highest FPU utilization (58.4%)
  - Most operations are compute-bound (84/116 = 72%)
```

### Matmul Analysis

```bash
uv run ttnn-vis-cli perf --performance /path/to/perf-report analysis matmul
```

Output:
```
Matmul Operations Analysis
===========================

      ID  Cores  Device Time   Ideal Time  Efficiency    FPU% Bound
---------------------------------------------------------------------------
   95232     80    22.849 µs    10.294 µs       45.1%    45.1 Compute
   57344    112    21.194 µs    10.294 µs       48.6%    48.6 Compute
...

Summary:
  Total: 34 Matmul operations
  Total Time: 582.365 µs (26.9% of all ops)
  Avg Efficiency: 42.8% (Ideal/Device ratio)
  Avg FPU Utilization: 33.3%

Efficiency Distribution:
  High (>80%):      1 ops (2.9%)
  Medium (50-80%):   8 ops (23.5%)
  Low (<50%):      25 ops (73.5%)

Math Fidelity:
  LoFi: 34 ops
```

### Conv Analysis

```bash
uv run ttnn-vis-cli perf --performance /path/to/perf-report analysis conv
```

Output format is similar to Matmul analysis.

### Bottleneck Detection

```bash
uv run ttnn-vis-cli perf --performance /path/to/perf-report analysis bottlenecks

# Custom thresholds
uv run ttnn-vis-cli perf --performance /path/to/perf-report analysis bottlenecks --efficiency-threshold 30 --gap-threshold 50
```

Output:
```
Performance Bottlenecks
=======================

Low Efficiency Operations (<50%):
        ID  Op Code               Device Time   Efficiency  Issue
  ------------------------------------------------------------------------------------------
     11264  Pool2D                 170.214 µs         8.2%  Low FPU utilization (8.2%)
      5120  Transpose              104.497 µs         2.8%  Low FPU utilization (2.8%)
...

High Op-to-Op Gap (>100ms):
        ID  Op Code                       Gap  Possible Cause
  --------------------------------------------------------------------------------
    103424  Matmul                 981.528 ms  Host overhead / data transfer
...

Summary:
  Low efficiency operations: 88
  High op-to-op gap operations: 60
  Memory-bound low utilization: 0
```

### Analysis Summary

```bash
uv run ttnn-vis-cli perf --performance /path/to/perf-report analysis summary
```

Output:
```
Performance Analysis Summary
============================

Overview:
  Total Operations: 116
  Total Device Time: 2.167 ms
  Total Op-to-Op Gap: 47.907 s

Operation Distribution:
  Compute-bound: 84 ops (72.4%)
  Memory-bound: 10 ops (8.6%)
  Balanced: 22 ops (19.0%)

Top Op Codes by Time:
  1. Conv2d          (20 ops):   739.577 µs (34.1%)
  2. Matmul          (34 ops):   582.365 µs (26.9%)
  3. HaloDeviceOperation (22 ops):   278.217 µs (12.8%)

Utilization:
  Avg FPU Utilization: 49.4%
  Avg DRAM Utilization: 0.0%

Potential Issues:
  - 88 operations with <50% efficiency
  - 60 operations with op-to-op gap >100ms
```

## Output Formats

All commands support multiple output formats using the global `--format` option:

- `--format table` (default): Human-readable table format
- `--format json`: JSON format for programmatic use
- `--format csv`: CSV format for spreadsheet import

**Important:** The `--format` option must be specified before the command.

### JSON Output Example

```bash
uv run ttnn-vis-cli --format json operations --profiler /path/to/db.sqlite --top 3
```

Output:
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

### CSV Output Example

```bash
uv run ttnn-vis-cli --format csv operations --profiler /path/to/db.sqlite --top 3
```

Output:
```csv
id,name,duration,device_id,stack_trace_id,captured_graph_id
48,ttnn.transformer.paged_scaled_dot_product_attention_decode,1.385,,,
31,ttnn.all_gather,1.254,,,
```
