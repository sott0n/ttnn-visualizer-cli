"""ASCII memory map visualization for L1 memory layout."""

from typing import Optional

from ..data.models import L1MemoryEntry


def render_memory_map(
    entries: list[L1MemoryEntry],
    total_memory: int,
    width: int = 50,
) -> list[str]:
    """Render L1 memory as ASCII bar chart.

    Args:
        entries: List of L1MemoryEntry objects sorted by address.
        total_memory: Total L1 memory size in bytes.
        width: Width of the memory bar in characters.

    Returns:
        Lines of ASCII art showing memory usage.
    """
    lines = []

    if total_memory <= 0:
        lines.append("Total memory not available")
        return lines

    # Calculate total used memory
    used_memory = sum(e.size for e in entries)
    usage_percent = (used_memory / total_memory) * 100 if total_memory > 0 else 0

    # Create the memory bar
    filled_width = int((used_memory / total_memory) * width)
    empty_width = width - filled_width

    bar = "█" * filled_width + "░" * empty_width
    lines.append(f"|{bar}| {usage_percent:.0f}% used")

    # Create tensor labels below the bar if there are entries
    if entries:
        label_line = _create_tensor_label_line(entries, total_memory, width)
        if label_line:
            lines.append(label_line)

        # Create address markers
        address_line = _create_address_line(entries, total_memory, width)
        if address_line:
            lines.append(address_line)

    return lines


def _create_tensor_label_line(
    entries: list[L1MemoryEntry],
    total_memory: int,
    width: int,
) -> str:
    """Create a line showing tensor labels below the memory bar.

    Args:
        entries: List of L1MemoryEntry objects.
        total_memory: Total L1 memory size.
        width: Width of the memory bar.

    Returns:
        String with tensor labels positioned below their memory regions.
    """
    if not entries:
        return ""

    # Build a character buffer for labels
    label_chars = [" "] * (width + 2)  # +2 for the | delimiters

    for entry in entries:
        if entry.size <= 0:
            continue

        # Calculate position and width for this entry
        start_pos = int((entry.address / total_memory) * width) + 1
        end_pos = int(((entry.address + entry.size) / total_memory) * width) + 1

        # Clamp to valid range
        start_pos = max(1, min(start_pos, width))
        end_pos = max(start_pos + 1, min(end_pos, width + 1))

        # Create label (tensor ID or abbreviated name)
        if entry.tensor_id is not None:
            label = f"T{entry.tensor_id}"
        else:
            label = "?"

        # Add marker
        available_width = end_pos - start_pos
        if available_width >= len(label) + 2:
            # Can fit label with dashes
            label_str = f"-{label}-"
            mid_pos = start_pos + (available_width - len(label_str)) // 2
            for i, char in enumerate(label_str):
                if mid_pos + i < width + 1:
                    label_chars[mid_pos + i] = char
        elif available_width >= 2:
            # Just show abbreviated label
            short_label = label[:available_width]
            for i, char in enumerate(short_label):
                if start_pos + i < width + 1:
                    label_chars[start_pos + i] = char

    return "|" + "".join(label_chars[1:-1]) + "|"


def _create_address_line(
    entries: list[L1MemoryEntry],
    total_memory: int,
    width: int,
) -> str:
    """Create a line showing address markers.

    Args:
        entries: List of L1MemoryEntry objects.
        total_memory: Total L1 memory size.
        width: Width of the memory bar.

    Returns:
        String with address markers.
    """
    # Show start address (0x0) and end address
    start_addr = "0x0"
    end_addr = f"0x{total_memory:x}"

    # Calculate spacing
    middle_space = width - len(start_addr) - len(end_addr) + 2
    if middle_space < 1:
        middle_space = 1

    return f"{start_addr}{' ' * middle_space}{end_addr}"


def format_l1_entry_table(
    entries: list[L1MemoryEntry],
    show_hex: bool = True,
    mark_new: bool = False,
    previous_addresses: Optional[set[int]] = None,
) -> list[str]:
    """Format L1 memory entries as a table.

    Args:
        entries: List of L1MemoryEntry objects.
        show_hex: Whether to show addresses in hexadecimal.
        mark_new: Whether to mark new entries.
        previous_addresses: Set of addresses from previous operation for comparison.

    Returns:
        Lines of formatted table output.
    """
    from .formatter import format_bytes

    lines = []

    if not entries:
        lines.append("No L1 buffers allocated")
        return lines

    # Table headers
    headers = ["Address", "Size", "Shape", "Dtype", "Layout", "Tensor"]
    col_widths = [14, 12, 20, 12, 16, 20]

    # Header line
    header_line = "  ".join(
        h.ljust(w) for h, w in zip(headers, col_widths)
    )
    lines.append(header_line)
    lines.append("-" * len(header_line))

    # Data rows
    for entry in entries:
        # Check if this is a new entry
        is_new = False
        if mark_new and previous_addresses is not None:
            is_new = entry.address not in previous_addresses

        # Format address
        if show_hex:
            addr_str = f"0x{entry.address:08x}"
        else:
            addr_str = str(entry.address)

        # Format size
        size_str = format_bytes(entry.size)

        # Format shape (truncate if too long)
        shape_str = entry.shape[:18] if entry.shape else "-"

        # Format dtype
        dtype_str = entry.dtype[:10] if entry.dtype else "-"

        # Format layout
        layout_str = entry.memory_layout[:14] if entry.memory_layout else "-"

        # Format tensor name
        tensor_str = entry.tensor_name[:18] if entry.tensor_name else "-"
        if is_new:
            tensor_str += " (new)"

        row = [
            addr_str.ljust(col_widths[0]),
            size_str.ljust(col_widths[1]),
            shape_str.ljust(col_widths[2]),
            dtype_str.ljust(col_widths[3]),
            layout_str.ljust(col_widths[4]),
            tensor_str.ljust(col_widths[5]),
        ]
        lines.append("  ".join(row))

    return lines
