"""SQLite database access for TTNN profiling data."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from .models import (
    Buffer,
    BufferPage,
    BufferType,
    Device,
    L1MemoryEntry,
    MemoryLayout,
    MemorySummary,
    Operation,
    OperationArgument,
    ReportInfo,
    Tensor,
)


class ProfilerDB:
    """Access layer for TTNN profiler SQLite database."""

    def __init__(self, db_path: str | Path):
        """Initialize with path to db.sqlite file.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Create a database connection context."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def get_devices(self) -> list[Device]:
        """Get all devices from the database."""
        with self._connection() as conn:
            cursor = conn.execute("SELECT * FROM devices")
            rows = cursor.fetchall()
            return [self._row_to_device(row) for row in rows]

    def get_device(self, device_id: int) -> Optional[Device]:
        """Get a specific device by ID."""
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM devices WHERE device_id = ?", (device_id,)
            )
            row = cursor.fetchone()
            return self._row_to_device(row) if row else None

    def _row_to_device(self, row: sqlite3.Row) -> Device:
        """Convert database row to Device object."""
        keys = row.keys()
        return Device(
            id=row["device_id"],
            num_y_cores=row["num_y_cores"],
            num_x_cores=row["num_x_cores"],
            num_y_compute_cores=row["num_y_compute_cores"],
            num_x_compute_cores=row["num_x_compute_cores"],
            worker_l1_size=row["worker_l1_size"],
            l1_num_banks=row["l1_num_banks"],
            l1_bank_size=row["l1_bank_size"],
            address_at_first_l1_bank=row["address_at_first_l1_bank"],
            address_at_first_l1_cb_buffer=row["address_at_first_l1_cb_buffer"],
            num_banks_per_storage_core=row["num_banks_per_storage_core"],
            num_compute_cores=row["num_compute_cores"],
            num_storage_cores=row["num_storage_cores"],
            total_l1_memory=row["total_l1_memory"],
            total_l1_for_tensors=row["total_l1_for_tensors"],
            cb_limit=row["cb_limit"],
            arch=row["arch"] if "arch" in keys else "",
            chip_id=row["chip_id"] if "chip_id" in keys else 0,
        )

    def get_operations(
        self,
        limit: Optional[int] = None,
        order_by_duration: bool = False,
    ) -> list[Operation]:
        """Get operations from the database.

        Args:
            limit: Maximum number of operations to return.
            order_by_duration: If True, order by duration descending.

        Returns:
            List of Operation objects.
        """
        query = "SELECT * FROM operations"
        if order_by_duration:
            query += " ORDER BY duration DESC"
        if limit:
            query += f" LIMIT {limit}"

        with self._connection() as conn:
            cursor = conn.execute(query)
            rows = cursor.fetchall()
            return [self._row_to_operation(row) for row in rows]

    def get_operation(self, operation_id: int) -> Optional[Operation]:
        """Get a specific operation by ID."""
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM operations WHERE operation_id = ?", (operation_id,)
            )
            row = cursor.fetchone()
            return self._row_to_operation(row) if row else None

    def _row_to_operation(self, row: sqlite3.Row) -> Operation:
        """Convert database row to Operation object."""
        keys = row.keys()
        return Operation(
            id=row["operation_id"],
            name=row["name"],
            duration=row["duration"] if "duration" in keys else None,
            device_id=row["device_id"] if "device_id" in keys else None,
            stack_trace_id=row["stack_trace_id"] if "stack_trace_id" in keys else None,
            captured_graph_id=row["captured_graph_id"] if "captured_graph_id" in keys else None,
        )

    def get_operation_arguments(self, operation_id: int) -> list[OperationArgument]:
        """Get arguments for a specific operation."""
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM operation_arguments WHERE operation_id = ?",
                (operation_id,),
            )
            rows = cursor.fetchall()
            return [
                OperationArgument(
                    operation_id=row["operation_id"],
                    name=row["name"],
                    value=row["value"],
                )
                for row in rows
            ]

    def get_tensors(self, limit: Optional[int] = None) -> list[Tensor]:
        """Get all tensors from the database."""
        query = "SELECT * FROM tensors"
        if limit:
            query += f" LIMIT {limit}"

        with self._connection() as conn:
            cursor = conn.execute(query)
            rows = cursor.fetchall()
            return [self._row_to_tensor(row) for row in rows]

    def get_tensor(self, tensor_id: int) -> Optional[Tensor]:
        """Get a specific tensor by ID."""
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM tensors WHERE tensor_id = ?", (tensor_id,)
            )
            row = cursor.fetchone()
            return self._row_to_tensor(row) if row else None

    def _row_to_tensor(self, row: sqlite3.Row) -> Tensor:
        """Convert database row to Tensor object."""
        keys = row.keys()
        return Tensor(
            id=row["tensor_id"],
            shape=row["shape"],
            dtype=row["dtype"],
            layout=row["layout"],
            memory_config=row["memory_config"] if "memory_config" in keys else None,
            device_id=row["device_id"] if "device_id" in keys else None,
            address=row["address"] if "address" in keys else None,
            buffer_type=row["buffer_type"] if "buffer_type" in keys else None,
        )

    def get_input_tensors(self, operation_id: int) -> list[Tensor]:
        """Get input tensors for a specific operation."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                SELECT t.* FROM tensors t
                JOIN input_tensors it ON t.tensor_id = it.tensor_id
                WHERE it.operation_id = ?
                """,
                (operation_id,),
            )
            rows = cursor.fetchall()
            return [self._row_to_tensor(row) for row in rows]

    def get_output_tensors(self, operation_id: int) -> list[Tensor]:
        """Get output tensors for a specific operation."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                SELECT t.* FROM tensors t
                JOIN output_tensors ot ON t.tensor_id = ot.tensor_id
                WHERE ot.operation_id = ?
                """,
                (operation_id,),
            )
            rows = cursor.fetchall()
            return [self._row_to_tensor(row) for row in rows]

    def get_buffers(
        self,
        buffer_type: Optional[BufferType] = None,
        operation_id: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> list[Buffer]:
        """Get buffers from the database.

        Args:
            buffer_type: Filter by buffer type (L1, DRAM).
            operation_id: Filter by operation ID.
            limit: Maximum number of buffers to return.

        Returns:
            List of Buffer objects.
        """
        query = "SELECT * FROM buffers WHERE 1=1"
        params: list = []

        if buffer_type:
            type_value = {
                BufferType.DRAM: 0,
                BufferType.L1: 1,
                BufferType.SYSTEM_MEMORY: 2,
                BufferType.L1_SMALL: 3,
                BufferType.TRACE: 4,
            }.get(buffer_type, 1)
            query += " AND buffer_type = ?"
            params.append(type_value)

        if operation_id is not None:
            query += " AND operation_id = ?"
            params.append(operation_id)

        if limit:
            query += f" LIMIT {limit}"

        with self._connection() as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            return [self._row_to_buffer(row, idx) for idx, row in enumerate(rows)]

    def get_buffer(self, buffer_id: int) -> Optional[Buffer]:
        """Get a specific buffer by ID."""
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM buffers WHERE buffer_id = ?", (buffer_id,)
            )
            row = cursor.fetchone()
            return self._row_to_buffer(row) if row else None

    def _row_to_buffer(self, row: sqlite3.Row, index: int = 0) -> Buffer:
        """Convert database row to Buffer object.

        Args:
            row: Database row.
            index: Row index to use as ID if buffer_id column doesn't exist.
        """
        keys = row.keys()

        # Handle different schema versions
        if "buffer_id" in keys:
            buffer_id = row["buffer_id"]
        else:
            buffer_id = index

        if "max_size" in keys:
            max_size = row["max_size"]
        elif "max_size_per_bank" in keys:
            max_size = row["max_size_per_bank"]
        else:
            max_size = 0

        return Buffer(
            id=buffer_id,
            address=row["address"],
            max_size=max_size,
            buffer_type=BufferType.from_int(row["buffer_type"]),
            device_id=row["device_id"],
            operation_id=row["operation_id"] if "operation_id" in keys else None,
        )

    def get_buffer_pages(self, buffer_id: int) -> list[BufferPage]:
        """Get buffer pages for a specific buffer."""
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM buffer_pages WHERE buffer_id = ?", (buffer_id,)
            )
            rows = cursor.fetchall()
            return [
                BufferPage(
                    buffer_id=row["buffer_id"],
                    core_x=row["core_x"],
                    core_y=row["core_y"],
                    page_index=row["page_index"],
                    page_address=row["page_address"],
                    page_size=row["page_size"],
                    device_id=row["device_id"],
                )
                for row in rows
            ]

    def _get_buffer_size_column(self, conn: sqlite3.Connection) -> str:
        """Get the correct column name for buffer size.

        Different database versions use different column names:
        - max_size: older schema
        - max_size_per_bank: newer schema
        """
        cursor = conn.execute("PRAGMA table_info(buffers)")
        columns = [row[1] for row in cursor.fetchall()]
        if "max_size" in columns:
            return "max_size"
        elif "max_size_per_bank" in columns:
            return "max_size_per_bank"
        return "max_size"  # fallback

    def get_memory_summary(self) -> MemorySummary:
        """Get memory usage summary."""
        summary = MemorySummary()

        with self._connection() as conn:
            size_col = self._get_buffer_size_column(conn)

            # Get buffer counts and sizes by type
            cursor = conn.execute(
                f"""
                SELECT buffer_type, COUNT(*) as count, SUM({size_col}) as total_size
                FROM buffers
                GROUP BY buffer_type
                """
            )
            rows = cursor.fetchall()

            for row in rows:
                buffer_type = BufferType.from_int(row["buffer_type"])
                if buffer_type == BufferType.L1 or buffer_type == BufferType.L1_SMALL:
                    summary.l1_used += row["total_size"] or 0
                    summary.l1_buffer_count += row["count"]
                elif buffer_type == BufferType.DRAM:
                    summary.dram_used += row["total_size"] or 0
                    summary.dram_buffer_count += row["count"]

            # Get total L1 from devices
            cursor = conn.execute("SELECT SUM(total_l1_for_tensors) as total FROM devices")
            row = cursor.fetchone()
            if row and row["total"]:
                summary.l1_total = row["total"]

        return summary

    def get_stack_trace(self, stack_trace_id: int) -> Optional[str]:
        """Get stack trace by ID."""
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT stack_trace FROM stack_traces WHERE stack_trace_id = ?",
                (stack_trace_id,),
            )
            row = cursor.fetchone()
            return row["stack_trace"] if row else None

    def get_report_info(self) -> ReportInfo:
        """Get summary information about the profiling report."""
        info = ReportInfo(profiler_path=str(self.db_path))

        with self._connection() as conn:
            # Count operations
            cursor = conn.execute("SELECT COUNT(*) as count FROM operations")
            row = cursor.fetchone()
            info.operation_count = row["count"] if row else 0

            # Count tensors
            cursor = conn.execute("SELECT COUNT(*) as count FROM tensors")
            row = cursor.fetchone()
            info.tensor_count = row["count"] if row else 0

            # Count buffers
            cursor = conn.execute("SELECT COUNT(*) as count FROM buffers")
            row = cursor.fetchone()
            info.buffer_count = row["count"] if row else 0

            # Count devices
            cursor = conn.execute("SELECT COUNT(*) as count FROM devices")
            row = cursor.fetchone()
            info.device_count = row["count"] if row else 0

            # Total duration
            cursor = conn.execute("SELECT SUM(duration) as total FROM operations")
            row = cursor.fetchone()
            info.total_duration_ns = row["total"] if row and row["total"] else 0.0

        # Get device details
        info.devices = self.get_devices()

        return info

    def get_table_names(self) -> list[str]:
        """Get list of table names in the database."""
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            return [row["name"] for row in cursor.fetchall()]

    def get_l1_report(
        self,
        operation_id: int,
        device_id: Optional[int] = None,
    ) -> list[L1MemoryEntry]:
        """Get L1 memory report for a specific operation.

        Args:
            operation_id: Operation ID to get L1 report for.
            device_id: Optional device ID filter.

        Returns:
            List of L1MemoryEntry objects sorted by address.
        """
        with self._connection() as conn:
            size_col = self._get_buffer_size_column(conn)

            # Query L1 buffers for this operation
            # buffer_type = 1 is L1, buffer_type = 3 is L1_SMALL
            query = f"""
                SELECT b.address, b.{size_col} as size, b.buffer_type, b.device_id,
                       b.operation_id
                FROM buffers b
                WHERE b.operation_id = ?
                AND b.buffer_type IN (1, 3)
            """
            params: list = [operation_id]

            if device_id is not None:
                query += " AND b.device_id = ?"
                params.append(device_id)

            query += " ORDER BY b.address"

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            entries = []
            for row in rows:
                # Try to find matching tensor by address
                tensor_info = self._get_tensor_by_address(
                    conn, row["address"], operation_id
                )

                buffer_type_val = row["buffer_type"]
                buffer_type_str = "L1" if buffer_type_val == 1 else "L1_SMALL"

                entry = L1MemoryEntry(
                    address=row["address"],
                    size=row["size"] or 0,
                    tensor_id=tensor_info.get("tensor_id"),
                    tensor_name=tensor_info.get("tensor_name", ""),
                    shape=tensor_info.get("shape", ""),
                    dtype=tensor_info.get("dtype", ""),
                    memory_layout=tensor_info.get("memory_layout", ""),
                    buffer_type=buffer_type_str,
                    operation_id=row["operation_id"],
                )
                entries.append(entry)

            return entries

    def get_previous_l1_report(
        self,
        operation_id: int,
        device_id: Optional[int] = None,
    ) -> list[L1MemoryEntry]:
        """Get L1 memory state from previous operation.

        Args:
            operation_id: Current operation ID (will get report for operation_id - 1).
            device_id: Optional device ID filter.

        Returns:
            List of L1MemoryEntry objects for the previous operation.
        """
        if operation_id <= 0:
            return []

        # Find the previous operation ID
        with self._connection() as conn:
            cursor = conn.execute(
                """
                SELECT operation_id FROM operations
                WHERE operation_id < ?
                ORDER BY operation_id DESC
                LIMIT 1
                """,
                (operation_id,),
            )
            row = cursor.fetchone()
            if not row:
                return []

            prev_operation_id = row["operation_id"]

        return self.get_l1_report(prev_operation_id, device_id)

    def _get_tensor_by_address(
        self,
        conn: sqlite3.Connection,
        address: int,
        operation_id: int,
    ) -> dict:
        """Find tensor information by address for an operation.

        Args:
            conn: Database connection.
            address: Memory address to look up.
            operation_id: Operation ID for context.

        Returns:
            Dictionary with tensor info or empty dict if not found.
        """
        # First try to find tensor in input/output tensors for this operation
        cursor = conn.execute(
            """
            SELECT t.tensor_id, t.shape, t.dtype, t.layout, t.memory_config, t.address
            FROM tensors t
            JOIN (
                SELECT tensor_id FROM input_tensors WHERE operation_id = ?
                UNION
                SELECT tensor_id FROM output_tensors WHERE operation_id = ?
            ) op_tensors ON t.tensor_id = op_tensors.tensor_id
            WHERE t.address = ?
            """,
            (operation_id, operation_id, address),
        )
        row = cursor.fetchone()

        if row:
            return self._extract_tensor_info(row)

        # If not found, try to find any tensor with this address
        cursor = conn.execute(
            """
            SELECT tensor_id, shape, dtype, layout, memory_config, address
            FROM tensors
            WHERE address = ?
            LIMIT 1
            """,
            (address,),
        )
        row = cursor.fetchone()

        if row:
            return self._extract_tensor_info(row)

        return {}

    def _extract_tensor_info(self, row: sqlite3.Row) -> dict:
        """Extract tensor information from database row.

        Args:
            row: Database row with tensor data.

        Returns:
            Dictionary with tensor information.
        """
        keys = row.keys()
        memory_layout = ""

        # Try to parse memory_config for layout information
        if "memory_config" in keys and row["memory_config"]:
            memory_config = row["memory_config"]
            # memory_config is typically a string like "MemoryConfig(...)"
            # Try to extract the layout type
            if "INTERLEAVED" in memory_config:
                memory_layout = "INTERLEAVED"
            elif "HEIGHT_SHARDED" in memory_config:
                memory_layout = "HEIGHT_SHARDED"
            elif "WIDTH_SHARDED" in memory_config:
                memory_layout = "WIDTH_SHARDED"
            elif "BLOCK_SHARDED" in memory_config:
                memory_layout = "BLOCK_SHARDED"
            elif "SINGLE_BANK" in memory_config:
                memory_layout = "SINGLE_BANK"

        return {
            "tensor_id": row["tensor_id"],
            "tensor_name": f"Tensor {row['tensor_id']}",
            "shape": row["shape"] if "shape" in keys else "",
            "dtype": row["dtype"] if "dtype" in keys else "",
            "memory_layout": memory_layout,
        }
