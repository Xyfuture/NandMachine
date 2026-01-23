"""Runtime resource entry classes for tracking allocated resources."""

from typing import Optional, Dict, Any
from .tables import DeviceType, Permission


class RuntimeResourceEntryBase:
    """
    Base class for runtime resource entries.

    Manages common attributes and operations for all resource types including
    memory mappings, allocations, and prefetch buffers.

    Attributes:
        start_logical_addr: Starting logical address of the allocation
        allocated_logical_pages: Set of allocated logical page numbers
        size: Size of the allocation in bytes
        page_size: Size of each page in bytes
        valid: Whether this entry is valid
    """

    def __init__(self, start_logical_addr: int, size: int, page_size: int = 4096):
        """
        Initialize a runtime resource entry.

        Args:
            start_logical_addr: Starting logical address
            size: Size of the allocation in bytes
            page_size: Size of each page in bytes (default: 4096)
        """
        self.start_logical_addr = start_logical_addr
        self.size = size
        self.page_size = page_size

        # Calculate and allocate logical pages
        num_pages = (size + page_size - 1) // page_size  # Ceiling division
        start_page = start_logical_addr // page_size
        self.allocated_logical_pages: set[int] = set(range(start_page, start_page + num_pages))

        # State management
        self.valid = True

    def is_valid(self) -> bool:
        """
        Check if this entry is valid.

        Returns:
            True if entry is valid, False otherwise
        """
        return self.valid

    def get_size(self) -> int:
        """
        Get the size of the allocation.

        Returns:
            Size in bytes
        """
        return self.size

    def get_page_count(self) -> int:
        """
        Get the number of allocated pages.

        Returns:
            Number of pages
        """
        return len(self.allocated_logical_pages)

    def get_logical_pages(self) -> set[int]:
        """
        Get all allocated logical page numbers.

        Returns:
            Set of logical page numbers
        """
        return self.allocated_logical_pages.copy()


class NandMmapEntry(RuntimeResourceEntryBase):
    """
    Entry for NAND memory-mapped files.

    Manages memory-mapped regions from NAND flash storage, tracking
    file associations and access permissions.

    Attributes:
        file_id: Original file identifier in NAND storage
        permission: Access permissions (READ/WRITE flags)
    """

    def __init__(self, start_logical_addr: int, size: int, file_id: int,
                 permission: int = Permission.READ | Permission.WRITE,
                 page_size: int = 4096):
        """
        Initialize a NAND mmap entry.

        Args:
            start_logical_addr: Starting logical address
            size: Size of the mapping in bytes
            file_id: Original file identifier
            permission: Access permissions (default: READ | WRITE)
            page_size: Size of each page in bytes (default: 4096)
        """
        super().__init__(start_logical_addr, size, page_size)
        self.file_id = file_id
        self.permission = permission

    def get_file_id(self) -> int:
        """
        Get the original file identifier.

        Returns:
            File ID
        """
        return self.file_id

    def has_read_permission(self) -> bool:
        """
        Check if this entry has read permission.

        Returns:
            True if read permission is granted, False otherwise
        """
        return Permission.has_read(self.permission)

    def has_write_permission(self) -> bool:
        """
        Check if this entry has write permission.

        Returns:
            True if write permission is granted, False otherwise
        """
        return Permission.has_write(self.permission)



class MallocEntry(RuntimeResourceEntryBase):
    """
    Entry for dynamically allocated memory regions.

    Manages malloc-style allocations on fast memory devices (DRAM/SRAM),
    tracking device placement and allocation metadata.

    Attributes:
        device_type: Physical device type (DRAM or SRAM)
    """

    def __init__(self, start_logical_addr: int, size: int, device_type: DeviceType,
                 page_size: int = 4096):
        """
        Initialize a malloc entry.

        Args:
            start_logical_addr: Starting logical address
            size: Size of the allocation in bytes
            device_type: Physical device type (must be DRAM or SRAM)
            page_size: Size of each page in bytes (default: 4096)

        Raises:
            ValueError: If device_type is not DRAM or SRAM
        """
        if device_type not in (DeviceType.DRAM, DeviceType.SRAM):
            raise ValueError(f"MallocEntry only supports DRAM or SRAM, got {device_type}")

        super().__init__(start_logical_addr, size, page_size)
        self.device_type = device_type

    def get_device_type(self) -> DeviceType:
        """
        Get the physical device type.

        Returns:
            Device type (DRAM or SRAM)
        """
        return self.device_type



class PrefetchEntry(RuntimeResourceEntryBase):
    """
    Entry for prefetched memory regions.

    Manages prefetch buffers that cache data from slower storage,
    maintaining mappings between prefetch addresses and original source addresses.

    Attributes:
        source_logical_pages: Mapping from prefetch logical page to source logical page
    """

    def __init__(self, start_logical_addr: int, size: int,
                 source_mapping: dict[int, int], page_size: int = 4096):
        """
        Initialize a prefetch entry.

        Args:
            start_logical_addr: Starting logical address of prefetch buffer
            size: Size of the prefetch buffer in bytes
            source_mapping: Mapping from prefetch logical page to source logical page
            page_size: Size of each page in bytes (default: 4096)
        """
        super().__init__(start_logical_addr, size, page_size)
        self.source_logical_pages = source_mapping.copy()

    def get_source_page(self, prefetch_page: int) -> Optional[int]:
        """
        Get the source logical page for a prefetch page.

        Args:
            prefetch_page: Prefetch logical page number

        Returns:
            Source logical page number if mapping exists, None otherwise
        """
        if not self.is_valid():
            return None

        return self.source_logical_pages.get(prefetch_page)

    def get_all_source_pages(self) -> dict[int, int]:
        """
        Get all source page mappings.

        Returns:
            Dictionary mapping prefetch pages to source pages
        """
        return self.source_logical_pages.copy()

    def has_source_mapping(self, prefetch_page: int) -> bool:
        """
        Check if a prefetch page has a source mapping.

        Args:
            prefetch_page: Prefetch logical page number

        Returns:
            True if mapping exists, False otherwise
        """
        return prefetch_page in self.source_logical_pages

    def get_source_page_count(self) -> int:
        """
        Get the number of source pages mapped.

        Returns:
            Number of source page mappings
        """
        return len(self.source_logical_pages)


class RuntimeResourceTable:
    """
    Resource management table for tracking RuntimeResourceEntryBase instances.

    Provides a mapping from start_logical_addr to entry instances, supporting
    add, remove, and lookup operations.

    Attributes:
        _entries: Dictionary mapping start_logical_addr to entry instances
    """

    def __init__(self):
        """Initialize an empty resource table."""
        self._entries: Dict[int, RuntimeResourceEntryBase] = {}

    def add_entry(self, entry: RuntimeResourceEntryBase) -> bool:
        """
        Add an entry to the resource table.

        Args:
            entry: The entry to add

        Returns:
            True if entry was added successfully, False if address conflicts
        """
        if not entry.is_valid():
            return False

        addr = entry.start_logical_addr
        if addr in self._entries:
            return False

        self._entries[addr] = entry
        return True

    def remove_entry(self, start_logical_addr: int) -> bool:
        """
        Remove an entry from the resource table.

        Args:
            start_logical_addr: The start logical address of the entry to remove

        Returns:
            True if entry was removed, False if not found
        """
        if start_logical_addr in self._entries:
            del self._entries[start_logical_addr]
            return True
        return False

    def get_entry(self, start_logical_addr: int) -> Optional[RuntimeResourceEntryBase]:
        """
        Get an entry by its start logical address.

        Args:
            start_logical_addr: The start logical address to look up

        Returns:
            The entry if found, None otherwise
        """
        return self._entries.get(start_logical_addr)

    def has_entry(self, start_logical_addr: int) -> bool:
        """
        Check if an entry exists at the given start logical address.

        Args:
            start_logical_addr: The start logical address to check

        Returns:
            True if entry exists, False otherwise
        """
        return start_logical_addr in self._entries

    def get_all_entries(self) -> Dict[int, RuntimeResourceEntryBase]:
        """
        Get all entries in the resource table.

        Returns:
            Dictionary mapping start_logical_addr to entry instances
        """
        return self._entries.copy()

    def get_entry_count(self) -> int:
        """
        Get the number of entries in the resource table.

        Returns:
            Number of entries
        """
        return len(self._entries)

    def clear(self) -> None:
        """Remove all entries from the resource table."""
        self._entries.clear()



__all__ = [
    "RuntimeResourceEntryBase",
    "NandMmapEntry",
    "MallocEntry",
    "PrefetchEntry",
    "RuntimeResourceTable",
]
