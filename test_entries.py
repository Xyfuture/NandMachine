"""Test script for runtime resource entries."""

from nandmachine.simulator.runtime.entries import (
    RuntimeResourceEntryBase,
    NandMmapEntry,
    MallocEntry,
    PrefetchEntry
)
from nandmachine.simulator.runtime.tables import DeviceType, Permission


def test_nand_mmap_entry():
    """Test NandMmapEntry functionality."""
    print("Testing NandMmapEntry...")

    # Create entry
    entry = NandMmapEntry(0x1000, 8192, file_id=42)

    # Test basic properties
    assert entry.is_valid(), "Entry should be valid"
    assert entry.get_size() == 8192, "Size should be 8192"
    assert entry.get_page_count() == 2, "Should have 2 pages"
    assert entry.get_file_id() == 42, "File ID should be 42"

    # Test permissions
    assert entry.has_read_permission(), "Should have read permission"
    assert entry.has_write_permission(), "Should have write permission"

    # Test permission modification
    entry.remove_permission(Permission.WRITE)
    assert entry.has_read_permission(), "Should still have read permission"
    assert not entry.has_write_permission(), "Should not have write permission"

    entry.add_permission(Permission.WRITE)
    assert entry.has_write_permission(), "Should have write permission again"

    # Test invalidation
    entry.invalidate()
    assert not entry.is_valid(), "Entry should be invalid"

    print("[PASS] NandMmapEntry tests passed")


def test_malloc_entry():
    """Test MallocEntry functionality."""
    print("Testing MallocEntry...")

    # Create entry on DRAM
    entry = MallocEntry(0x2000, 4096, DeviceType.DRAM)

    # Test basic properties
    assert entry.is_valid(), "Entry should be valid"
    assert entry.get_size() == 4096, "Size should be 4096"
    assert entry.get_page_count() == 1, "Should have 1 page"

    # Test device type
    assert entry.is_on_dram(), "Should be on DRAM"
    assert not entry.is_on_sram(), "Should not be on SRAM"
    assert entry.get_device_type() == DeviceType.DRAM, "Device type should be DRAM"
    assert entry.get_device_name() == "dram", "Device name should be 'dram'"

    # Test SRAM entry
    sram_entry = MallocEntry(0x3000, 4096, DeviceType.SRAM)
    assert sram_entry.is_on_sram(), "Should be on SRAM"
    assert not sram_entry.is_on_dram(), "Should not be on DRAM"

    # Test invalid device type
    try:
        invalid_entry = MallocEntry(0x4000, 4096, DeviceType.NAND)
        assert False, "Should raise ValueError for NAND device type"
    except ValueError as e:
        assert "only supports DRAM or SRAM" in str(e)

    print("[PASS] MallocEntry tests passed")


def test_prefetch_entry():
    """Test PrefetchEntry functionality."""
    print("Testing PrefetchEntry...")

    # Create source mapping
    source_mapping = {
        0x10: 0x100,
        0x11: 0x101,
        0x12: 0x102,
    }

    # Create entry
    entry = PrefetchEntry(0x10000, 12288, source_mapping)

    # Test basic properties
    assert entry.is_valid(), "Entry should be valid"
    assert entry.get_size() == 12288, "Size should be 12288"
    assert entry.get_page_count() == 3, "Should have 3 pages"

    # Test source mapping
    assert entry.get_source_page(0x10) == 0x100, "Should map to 0x100"
    assert entry.get_source_page(0x11) == 0x101, "Should map to 0x101"
    assert entry.get_source_page(0x12) == 0x102, "Should map to 0x102"
    assert entry.get_source_page(0x99) is None, "Should return None for unmapped page"

    # Test mapping queries
    assert entry.has_source_mapping(0x10), "Should have mapping for 0x10"
    assert not entry.has_source_mapping(0x99), "Should not have mapping for 0x99"
    assert entry.get_source_page_count() == 3, "Should have 3 mappings"

    # Test get all mappings
    all_mappings = entry.get_all_source_pages()
    assert len(all_mappings) == 3, "Should return 3 mappings"
    assert all_mappings[0x10] == 0x100, "Mapping should be correct"

    # Test that returned mapping is a copy
    all_mappings[0x99] = 0x999
    assert not entry.has_source_mapping(0x99), "Original should not be modified"

    print("[PASS] PrefetchEntry tests passed")


def test_runtime_manager():
    """Test RuntimeManager integration."""
    print("Testing RuntimeManager integration...")

    from nandmachine.simulator.runtime.manager import RuntimeManager

    manager = RuntimeManager()

    # Create and register entries
    mmap_entry = NandMmapEntry(0x1000, 4096, file_id=1)
    malloc_entry = MallocEntry(0x2000, 4096, DeviceType.DRAM)
    prefetch_entry = PrefetchEntry(0x3000, 4096, {0x3: 0x300})

    # Register entries
    assert manager.register_mmap(mmap_entry), "Should register mmap entry"
    assert manager.register_malloc(malloc_entry), "Should register malloc entry"
    assert manager.register_prefetch(prefetch_entry), "Should register prefetch entry"

    # Test duplicate registration
    assert not manager.register_mmap(mmap_entry), "Should not register duplicate"

    # Test find by page
    found = manager.find_entry_by_page(0x1)  # Page 0x1 is in mmap_entry (addr 0x1000)
    assert found is not None, "Should find entry by page"
    assert found.start_logical_addr == 0x1000, "Should find correct entry"

    # Test find by address
    found = manager.find_entry_by_addr(0x2000)
    assert found is not None, "Should find entry by address"
    assert isinstance(found, MallocEntry), "Should be MallocEntry"

    # Test cleanup
    mmap_entry.invalidate()
    cleaned = manager.cleanup_invalid_entries()
    assert cleaned == 1, "Should clean up 1 entry"
    assert manager.find_entry_by_addr(0x1000) is None, "Entry should be removed"

    print("[PASS] RuntimeManager tests passed")


if __name__ == "__main__":
    test_nand_mmap_entry()
    test_malloc_entry()
    test_prefetch_entry()
    test_runtime_manager()
    print("\n[SUCCESS] All tests passed!")
