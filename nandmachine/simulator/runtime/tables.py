
from enum import Enum


class DeviceType(Enum):
    """Hardware device types for physical memory."""
    NAND = "nand"
    DRAM = "dram"
    SRAM = "sram"


class Permission:
    """Permission flags for page table entries."""
    READ = 0x1   # 0b01
    WRITE = 0x2  # 0b10

    @staticmethod
    def has_read(perm: int) -> bool:
        """Check if permission includes read access."""
        return (perm & Permission.READ) != 0

    @staticmethod
    def has_write(perm: int) -> bool:
        """Check if permission includes write access."""
        return (perm & Permission.WRITE) != 0


class BaseFreeTable:
    pass 


class NandFreeTable(BaseFreeTable):

    # 管理 free 的 nand page，

    pass 




class NandFileTable:
    
    # 记录一下 file id 到 nand physical id 的转换
    # 支持 各个层级地址的维护 --> 需要一个地址转换器
    # 支持 按照某些要求，写入到指定的 plane 中 --> 需要 NandFreeTable 的支持
    
    # 加入一堆 checker 函数， 检索一下 weight 和 kv cache 有没有按照想要的方式分布 
    pass 




class RAMFreeTable(BaseFreeTable):
    """
    Free table for managing free pages in RAM.
    Uses a set-based approach for efficient space management.
    """

    def __init__(self, total_pages: int):
        """
        Initialize RAM free table.

        Args:
            total_pages: Total number of pages in RAM
        """
        self.total_pages = total_pages
        self.free_pages = set(range(total_pages))  # Set of free page indices
        self.allocated_pages = {}  # Maps page_id -> allocation_info

    def allocate_page(self) -> int:
        """
        Allocate a free page from RAM.

        Returns:
            Page index if successful, -1 if no free pages available
        """
        if not self.free_pages:
            return -1

        page_id = self.free_pages.pop()
        self.allocated_pages[page_id] = {'allocated': True}
        return page_id

    def free_page(self, page_id: int) -> bool:
        """
        Free an allocated page back to the pool.

        Args:
            page_id: Page index to free

        Returns:
            True if successful, False if page was not allocated
        """
        if page_id not in self.allocated_pages:
            return False

        del self.allocated_pages[page_id]
        self.free_pages.add(page_id)
        return True

    def get_free_count(self) -> int:
        """
        Get the number of free pages.

        Returns:
            Number of free pages available
        """
        return len(self.free_pages)

    def get_allocated_count(self) -> int:
        """
        Get the number of allocated pages.

        Returns:
            Number of allocated pages
        """
        return len(self.allocated_pages)

    def is_page_free(self, page_id: int) -> bool:
        """
        Check if a specific page is free.

        Args:
            page_id: Page index to check

        Returns:
            True if page is free, False otherwise
        """
        return page_id in self.free_pages


class DRAMFreeTable(RAMFreeTable):
    def __init__(self, config):
        super().__init__(0)




class SRAMFreeTable(RAMFreeTable):
    def __init__(self, config):
        super().__init__(0)




class PageTableEntry:
    """
    Page table entry that maps a logical page to a physical page.

    Attributes:
        device_type: Type of hardware device (NAND/DRAM/SRAM)
        physical_page: Physical page number on the device
        valid: Whether this entry is valid
        permission: Access permissions (READ/WRITE)
    """

    def __init__(self, device_type: DeviceType, physical_page: int,
                 permission: int = Permission.READ | Permission.WRITE):
        """
        Initialize a page table entry.

        Args:
            device_type: Type of hardware device
            physical_page: Physical page number on the device
            permission: Access permissions (default: READ | WRITE)
        """
        self.device_type = device_type
        self.physical_page = physical_page
        self.valid = True
        self.permission = permission


class PageTable:
    """
    Page table for mapping logical addresses to physical device addresses.
    Supports mapping to NAND, DRAM, and SRAM devices.
    """

    def __init__(self, page_size: int = 4096):
        """
        Initialize page table.

        Args:
            page_size: Size of each page in bytes (configurable)
        """
        self.page_size = page_size
        self.entries: dict[int, PageTableEntry] = {}  # logical_page -> PTE

    def map_page(self, logical_page: int, device_type: DeviceType,
                 physical_page: int, permission: int = Permission.READ | Permission.WRITE) -> bool:
        """
        Map a logical page to a physical page on a device.

        Args:
            logical_page: Logical page number
            device_type: Type of hardware device
            physical_page: Physical page number on the device
            permission: Access permissions (default: READ | WRITE)

        Returns:
            True if mapping was successful, False if page already mapped
        """
        if logical_page in self.entries:
            return False

        self.entries[logical_page] = PageTableEntry(device_type, physical_page, permission)
        return True

    def unmap_page(self, logical_page: int) -> bool:
        """
        Remove mapping for a logical page.

        Args:
            logical_page: Logical page number to unmap

        Returns:
            True if unmapping was successful, False if page was not mapped
        """
        if logical_page not in self.entries:
            return False

        del self.entries[logical_page]
        return True

    def translate(self, logical_page: int) -> tuple[DeviceType, int] | None:
        """
        Translate logical page to (device_type, physical_page).

        Args:
            logical_page: Logical page number to translate

        Returns:
            Tuple of (device_type, physical_page) if valid mapping exists, None otherwise
        """
        if logical_page not in self.entries:
            return None

        entry = self.entries[logical_page]
        if not entry.valid:
            return None

        return (entry.device_type, entry.physical_page)

    def check_permission(self, logical_page: int, required_perm: int) -> bool:
        """
        Check if the page has required permissions.

        Args:
            logical_page: Logical page number to check
            required_perm: Required permission flags (READ/WRITE)

        Returns:
            True if page has required permissions, False otherwise
        """
        if logical_page not in self.entries:
            return False

        entry = self.entries[logical_page]
        if not entry.valid:
            return False

        return (entry.permission & required_perm) == required_perm

    def is_valid(self, logical_page: int) -> bool:
        """
        Check if a logical page has a valid mapping.

        Args:
            logical_page: Logical page number to check

        Returns:
            True if page has a valid mapping, False otherwise
        """
        if logical_page not in self.entries:
            return False

        return self.entries[logical_page].valid

    def invalidate(self, logical_page: int) -> bool:
        """
        Invalidate a page table entry without removing it.

        Args:
            logical_page: Logical page number to invalidate

        Returns:
            True if invalidation was successful, False if page was not mapped
        """
        if logical_page not in self.entries:
            return False

        self.entries[logical_page].valid = False
        return True

    def get_mapped_count(self) -> int:
        """
        Get number of mapped pages.

        Returns:
            Number of pages with mappings (including invalid entries)
        """
        return len(self.entries)

    def get_device_pages(self, device_type: DeviceType) -> list[int]:
        """
        Get all logical pages mapped to a specific device.

        Args:
            device_type: Type of hardware device to filter by

        Returns:
            List of logical page numbers mapped to the specified device
        """
        return [
            logical_page
            for logical_page, entry in self.entries.items()
            if entry.device_type == device_type and entry.valid
        ]







__all__ = [
    # "BaseFreeTable",
    "NandFreeTable",
    "NandFileTable",
    # "RAMFreeTable",
    "DRAMFreeTable",
    "SRAMFreeTable",
    "PageTable",
    "PageTableEntry",
    "DeviceType",
    "Permission"
]