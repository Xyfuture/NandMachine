"""
Test script for PageTable implementation.
Tests basic mapping, permissions, invalidation, and multi-device mapping.
"""

from nandmachine.simulator.runtime.tables import PageTable, DeviceType, Permission


def test_basic_mapping():
    """Test basic page mapping and translation."""
    print("Testing basic mapping...")
    pt = PageTable(page_size=4096)

    # Test mapping to different devices
    assert pt.map_page(0, DeviceType.DRAM, 100) == True
    assert pt.map_page(1, DeviceType.SRAM, 50) == True
    assert pt.map_page(2, DeviceType.NAND, 200) == True

    # Test address translation
    assert pt.translate(0) == (DeviceType.DRAM, 100)
    assert pt.translate(1) == (DeviceType.SRAM, 50)
    assert pt.translate(2) == (DeviceType.NAND, 200)

    # Test invalid page
    assert pt.translate(999) == None

    print("[PASS] Basic mapping tests passed")


def test_duplicate_mapping():
    """Test that duplicate mappings are rejected."""
    print("Testing duplicate mapping...")
    pt = PageTable()

    assert pt.map_page(0, DeviceType.DRAM, 100) == True
    assert pt.map_page(0, DeviceType.SRAM, 50) == False  # Should fail

    print("[PASS] Duplicate mapping tests passed")


def test_permissions():
    """Test permission checking."""
    print("Testing permissions...")
    pt = PageTable()

    # Map with different permissions
    pt.map_page(0, DeviceType.DRAM, 100, Permission.READ | Permission.WRITE)
    pt.map_page(1, DeviceType.SRAM, 50, Permission.READ)
    pt.map_page(2, DeviceType.NAND, 200, Permission.WRITE)

    # Test read permissions
    assert pt.check_permission(0, Permission.READ) == True
    assert pt.check_permission(1, Permission.READ) == True
    assert pt.check_permission(2, Permission.READ) == False

    # Test write permissions
    assert pt.check_permission(0, Permission.WRITE) == True
    assert pt.check_permission(1, Permission.WRITE) == False
    assert pt.check_permission(2, Permission.WRITE) == True

    # Test combined permissions
    assert pt.check_permission(0, Permission.READ | Permission.WRITE) == True
    assert pt.check_permission(1, Permission.READ | Permission.WRITE) == False

    print("[PASS] Permission tests passed")


def test_invalidation():
    """Test page invalidation."""
    print("Testing invalidation...")
    pt = PageTable()

    # Map a page
    pt.map_page(0, DeviceType.DRAM, 100)
    assert pt.is_valid(0) == True
    assert pt.translate(0) == (DeviceType.DRAM, 100)

    # Invalidate the page
    assert pt.invalidate(0) == True
    assert pt.is_valid(0) == False
    assert pt.translate(0) == None  # Should return None for invalid page

    # Try to invalidate non-existent page
    assert pt.invalidate(999) == False

    print("[PASS] Invalidation tests passed")


def test_unmapping():
    """Test page unmapping."""
    print("Testing unmapping...")
    pt = PageTable()

    # Map and unmap
    pt.map_page(0, DeviceType.DRAM, 100)
    assert pt.is_valid(0) == True

    assert pt.unmap_page(0) == True
    assert pt.is_valid(0) == False
    assert pt.translate(0) == None

    # Try to unmap non-existent page
    assert pt.unmap_page(999) == False

    print("[PASS] Unmapping tests passed")


def test_statistics():
    """Test statistics methods."""
    print("Testing statistics...")
    pt = PageTable()

    # Initially empty
    assert pt.get_mapped_count() == 0

    # Add some mappings
    pt.map_page(0, DeviceType.DRAM, 100)
    pt.map_page(1, DeviceType.DRAM, 101)
    pt.map_page(2, DeviceType.SRAM, 50)
    pt.map_page(3, DeviceType.NAND, 200)

    assert pt.get_mapped_count() == 4

    # Test device-specific queries
    dram_pages = pt.get_device_pages(DeviceType.DRAM)
    assert len(dram_pages) == 2
    assert 0 in dram_pages
    assert 1 in dram_pages

    sram_pages = pt.get_device_pages(DeviceType.SRAM)
    assert len(sram_pages) == 1
    assert 2 in sram_pages

    nand_pages = pt.get_device_pages(DeviceType.NAND)
    assert len(nand_pages) == 1
    assert 3 in nand_pages

    # Invalidate a page - should not appear in device pages
    pt.invalidate(0)
    dram_pages = pt.get_device_pages(DeviceType.DRAM)
    assert len(dram_pages) == 1
    assert 0 not in dram_pages

    # But still counted in total
    assert pt.get_mapped_count() == 4

    print("[PASS] Statistics tests passed")


def test_multi_device_mapping():
    """Test mapping to multiple devices."""
    print("Testing multi-device mapping...")
    pt = PageTable()

    # Map pages to different devices
    pt.map_page(0, DeviceType.DRAM, 100, Permission.READ | Permission.WRITE)
    pt.map_page(1, DeviceType.SRAM, 50, Permission.READ)
    pt.map_page(2, DeviceType.NAND, 200, Permission.READ | Permission.WRITE)
    pt.map_page(3, DeviceType.DRAM, 101, Permission.READ)

    # Verify all mappings
    assert pt.get_mapped_count() == 4
    assert len(pt.get_device_pages(DeviceType.DRAM)) == 2
    assert len(pt.get_device_pages(DeviceType.SRAM)) == 1
    assert len(pt.get_device_pages(DeviceType.NAND)) == 1

    print("[PASS] Multi-device mapping tests passed")


def run_all_tests():
    """Run all test cases."""
    print("=" * 50)
    print("Running PageTable Tests")
    print("=" * 50)

    test_basic_mapping()
    test_duplicate_mapping()
    test_permissions()
    test_invalidation()
    test_unmapping()
    test_statistics()
    test_multi_device_mapping()

    print("=" * 50)
    print("All tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    run_all_tests()
