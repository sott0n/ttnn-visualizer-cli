# Interactive TUI

Launch an interactive terminal user interface for browsing and exploring profiling data.

```bash
# Launch TUI with profiler data only
uv run ttnn-vis-cli tui --profiler /path/to/db.sqlite

# Launch with both profiler and performance data
uv run ttnn-vis-cli tui --profiler /path/to/db.sqlite --performance /path/to/perf-report

# Short options
uv run ttnn-vis-cli tui -p /path/to/db.sqlite -P /path/to/perf-report
```

## TUI Tabs

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

## Keyboard Shortcuts

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
