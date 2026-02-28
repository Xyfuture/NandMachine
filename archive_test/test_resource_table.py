"""Test script for RuntimeResourceTable."""

from nandmachine.simulator.runtime.entries import (
    RuntimeResourceEntryBase,
    NandMmapEntry,
    MallocEntry,
    PrefetchEntry,
    RuntimeResourceTable
)
from nandmachine.simulator.runtime.tables import DeviceType, Permission


def test_resource_table_basic():
    """Test basic RuntimeResourceTable functionality."""
    print("Testing RuntimeResourceTable basic operations...")

    table = RuntimeResourceTable()

    # Create entries
    mmap_entry = NandMmapEntry(0x1000, 8192, file_id=1)
    malloc_entry = MallocEntry(0x2000, 4096, DeviceType.DRAM)
    prefetch_entry = PrefetchEntry(0x3000, 4096, {0x3: 0x300})

    # Test add entries
    assert table.add_entry(mmap_entry), "Should add mmap entry"
    assert table.add_entry(malloc_entry), "Should add malloc entry"
    assert table.add_entry(prefetch_entry), "Should add prefetch entry"
    assert table.get_entry_count() == 3, "Should have 3 entries"

    # Test duplicate add
    duplicate_entry = NandMmapEntry(0x1000, 4096, file_id=2)
    assert not table.add_entry(duplicate_entry), "Should not add duplicate address"

    # Test get entry
    found = table.get_entry(0x1000)
    assert found is not None, "Should find entry"
    assert found.start_logical_addr == 0x1000, "Should find correct entry"
    assert found.get_file_id() == 1, "Should have correct file ID"

    # Test has entry
    assert table.has_entry(0x1000), "Should have entry at 0x1000"
    assert not table.has_entry(0x9999), "Should not have entry at 0x9999"

    print("[PASS] Basic operations test passed")


def test_resource_table_lookup():
    """Test RuntimeResourceTable lookup operations."""
    print("Testing RuntimeResourceTable lookup operations...")

    table = RuntimeResourceTable()

    # Create entry spanning multiple pages
    mmap_entry = NandMmapEntry(0x1000, 8192, file_id=1)  # 2 pages
    table.add_entry(mmap_entry)

    # Test find by page
    found = table.find_entry_by_page(0x1)  # Page 0x1 is in mmap_entry
    assert found is not None, "Should find entry by page"
    assert found.start_logical_addr == 0x1000, "Should find correct entry"

    found = table.find_entry_by_page(0x99)
    assert found is None, "Should not find entry for non-existent page"

    # Test find by address
    found = table.find_entry_by_addr(0x1500)  # Address in page 0x1
    assert found is not None, "Should find entry by address"
    assert found.start_logical_addr == 0x1000, "Should find correct entry"

    found = table.find_entry_by_addr(0x9999)
    assert found is None, "Should not find entry for non-existent address"

    print("[PASS] Lookup operations test passed")


def test_resource_table_remove():
    """Test RuntimeResourceTable remove operations."""
    print("Testing RuntimeResourceTable remove operations...")

    table = RuntimeResourceTable()

    # Create and add entries
    mmap_entry = NandMmapEntry(0x1000, 8192, file_id=1)
    malloc_entry = MallocEntry(0x2000, 4096, DeviceType.DRAM)
    table.add_entry(mmap_entry)
    table.add_entry(malloc_entry)

    assert table.get_entry_count() == 2, "Should have 2 entries"

    # Test remove entry
    assert table.remove_entry(0x1000), "Should remove entry"
    assert table.get_entry_count() == 1, "Should have 1 entry left"
    assert not table.has_entry(0x1000), "Entry should be removed"

    # Test remove non-existent entry
    assert not table.remove_entry(0x9999), "Should not remove non-existent entry"

    print("[PASS] Remove operations test passed")


def test_resource_table_invalid_entries():
    """Test RuntimeResourceTable invalid entry handling."""
    print("Testing RuntimeResourceTable invalid entry handling...")

    table = RuntimeResourceTable()

    # Create entries
    valid_entry = NandMmapEntry(0x1000, 4096, file_id=1)
    invalid_entry = NandMmapEntry(0x2000, 4096, file_id=2)

    # Add entries
    table.add_entry(valid_entry)
    table.add_entry(invalid_entry)
    assert table.get_entry_count() == 2, "Should have 2 entries"

    # Invalidate one entry
    invalid_entry.invalidate()
    assert not invalid_entry.is_valid(), "Entry should be invalid"

    # Remove invalid entries
    removed = table.remove_invalid_entries()
    assert removed == 1, "Should remove 1 invalid entry"
    assert table.get_entry_count() == 1, "Should have 1 entry left"
    assert table.has_entry(0x1000), "Valid entry should remain"
    assert not table.has_entry(0x2000), "Invalid entry should be removed"

    print("[PASS] Invalid entry handling test passed")


def test_resource_table_clear():
    """Test RuntimeResourceTable clear operation."""
    print("Testing RuntimeResourceTable clear operation...")

    table = RuntimeResourceTable()

    # Add entries
    table.add_entry(NandMmapEntry(0x1000, 4096, file_id=1))
    table.add_entry(MallocEntry(0x2000, 4096, DeviceType.DRAM))
    assert table.get_entry_count() == 2, "Should have 2 entries"

    # Clear table
    table.clear()
    assert table.get_entry_count() == 0, "Should have 0 entries after clear"
    assert not table.has_entry(0x1000), "Should not have any entries"

    print("[PASS] Clear operation test passed")


def test_resource_table_get_all():
    """Test RuntimeResourceTable get all entries."""
    print("Testing RuntimeResourceTable get all entries...")

    table = RuntimeResourceTable()

    # Add entries
    mmap_entry = NandMmapEntry(0x1000, 4096, file_id=1)
    malloc_entry = MallocEntry(0x2000, 4096, DeviceType.DRAM)
    table.add_entry(mmap_entry)
    table.add_entry(malloc_entry)

    # Get all entries
    all_entries = table.get_all_entries()
    assert len(all_entries) == 2, "Should return 2 entries"
    assert 0x1000 in all_entries, "Should contain mmap entry"
    assert 0x2000 in all_entries, "Should contain malloc entry"

    # Test that returned dict is a copy
    all_entries[0x9999] = mmap_entry
    assert table.get_entry_count() == 2, "Original table should not be modified"

    print("[PASS] Get all entries test passed")


if __name__ == "__main__":
    test_resource_table_basic()
    test_resource_table_lookup()
    test_resource_table_remove()
    test_resource_table_invalid_entries()
    test_resource_table_clear()
    test_resource_table_get_all()
    print("\n[SUCCESS] All RuntimeResourceTable tests passed!")
