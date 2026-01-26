"""
Test script for NandBlockAddress class.
Tests block-level address encoding/decoding without page component.
"""

from nandmachine.simulator.runtime.addr import NandBlockAddress, NandAddress
from nandmachine.config.config import NandConfig


def test_block_address_creation():
    """Test creating block addresses"""
    print("\nTesting block address creation...")
    config = NandConfig(num_channels=4, num_plane=2, num_block=1024, num_pages=2048)

    # Test address 0
    addr = NandBlockAddress(0, config)
    assert addr.channel == 0
    assert addr.plane == 0
    assert addr.block == 0
    print(f"  Address 0: {addr}")

    # Test address 1000 (channel=0, plane=0, block=1000)
    addr = NandBlockAddress(1000, config)
    assert addr.channel == 0
    assert addr.plane == 0
    assert addr.block == 1000
    print(f"  Address 1000: {addr}")

    print("[PASS] Block address creation works")


def test_block_address_from_components():
    """Test creating block address from components"""
    print("\nTesting from_components...")
    config = NandConfig(num_channels=4, num_plane=2, num_block=1024, num_pages=2048)

    addr = NandBlockAddress.from_components(channel=1, plane=1, block=500, config=config)
    assert addr.channel == 1
    assert addr.plane == 1
    assert addr.block == 500
    print(f"  Created: {addr}")
    print("[PASS] from_components works")


def test_block_address_setters():
    """Test block address component setters"""
    print("\nTesting block address setters...")
    config = NandConfig(num_channels=4, num_plane=2, num_block=1024, num_pages=2048)
    addr = NandBlockAddress(0, config)

    addr.channel = 2
    addr.plane = 1
    addr.block = 500

    assert addr.channel == 2
    assert addr.plane == 1
    assert addr.block == 500
    print("[PASS] Block address setters work")


def test_block_address_increment():
    """Test block address increment operation"""
    print("\nTesting block address increment...")
    config = NandConfig(num_channels=4, num_plane=2, num_block=1024, num_pages=2048)
    addr = NandBlockAddress(0, config)

    new_addr = addr + 1
    assert new_addr.block == 1

    new_addr = addr + 10
    assert new_addr.block == 10
    print("[PASS] Block address increment works")


def test_block_address_carry():
    """Test carry logic across block address levels"""
    print("\nTesting block address carry logic...")
    config = NandConfig(num_channels=4, num_plane=2, num_block=1024, num_pages=2048)

    # Test block -> plane carry
    addr = NandBlockAddress(0, config)
    addr.block = 1023
    new_addr = addr + 1
    assert new_addr.block == 0
    assert new_addr.plane == 1

    # Test plane -> channel carry
    addr = NandBlockAddress(0, config)
    addr.plane = 1
    addr.block = 1023
    new_addr = addr + 1
    assert new_addr.block == 0
    assert new_addr.plane == 0
    assert new_addr.channel == 1

    print("[PASS] Block address carry logic works")


def test_block_address_overflow():
    """Test overflow detection"""
    print("\nTesting block address overflow...")
    config = NandConfig(num_channels=4, num_plane=2, num_block=1024, num_pages=2048)
    addr = NandBlockAddress(0, config)
    addr.channel = 3
    addr.plane = 1
    addr.block = 1023

    try:
        new_addr = addr + 1
        assert False, "Should have raised OverflowError"
    except OverflowError:
        pass
    print("[PASS] Block address overflow detection works")


def test_nand_address_to_block_address():
    """Test converting NandAddress to NandBlockAddress"""
    print("\nTesting NandAddress to NandBlockAddress conversion...")
    config = NandConfig(num_channels=4, num_plane=2, num_block=1024, num_pages=2048)

    # Create a full NandAddress with all components
    nand_addr = NandAddress(0, config)
    nand_addr.channel = 2
    nand_addr.plane = 1
    nand_addr.block = 500
    nand_addr.page = 1234

    print(f"  Original NandAddress: {nand_addr}")

    # Convert to block address
    block_addr = nand_addr.to_block_address()

    print(f"  Converted NandBlockAddress: {block_addr}")

    # Verify components are preserved (except page)
    assert block_addr.channel == 2
    assert block_addr.plane == 1
    assert block_addr.block == 500

    # Verify it's a NandBlockAddress instance
    assert isinstance(block_addr, NandBlockAddress)

    print("[PASS] NandAddress to NandBlockAddress conversion works")


def run_all_tests():
    """Run all test cases."""
    print("=" * 60)
    print("Running NandBlockAddress Tests")
    print("=" * 60)

    test_block_address_creation()
    test_block_address_from_components()
    test_block_address_setters()
    test_block_address_increment()
    test_block_address_carry()
    test_block_address_overflow()
    test_nand_address_to_block_address()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
