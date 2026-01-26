"""
Test script for address translation module.
Tests basic instantiation, inheritance, and translation functionality.
"""

from nandmachine.simulator.runtime.addr import (
    AddressTranslatorBase,
    AddressBase,
    NandAddressTranslator,
    NandAddress,
    DramAddressTranslator,
    DramAddress,
    SramAddressTranslator,
    SramAddress,
)
from nandmachine.config.config import NandConfig, DramConfig, SramConfig


def test_imports():
    """Test that all classes can be imported."""
    print("Testing imports...")
    print("[PASS] All classes imported successfully")


def test_nand_translator():
    """Test NAND address translator."""
    print("\nTesting NAND translator...")

    # Test with a known address
    config = NandConfig(
        num_channels=4,
        num_plane=2,
        num_block=1024,
        num_pages=2048
    )
    translator = NandAddressTranslator(config)

    # Test address 0
    addr = translator.translate(0)
    assert addr.channel == 0
    assert addr.plane == 0
    assert addr.page == 0
    assert addr.block == 0

    # Test address 1000001000 (channel=1, plane=0, page=0, block=1000)
    addr = translator.translate(1000001000)
    assert addr.channel == 1
    assert addr.plane == 0
    assert addr.page == 0
    assert addr.block == 1000

    print(f"  Created: {addr}")
    print("[PASS] NAND translator tests passed")


def test_dram_translator():
    """Test DRAM address translator."""
    print("\nTesting DRAM translator...")

    config = DramConfig()
    translator = DramAddressTranslator(config)

    # Test translation
    addr = translator.translate(67890)

    assert isinstance(addr, DramAddress)
    assert isinstance(addr, AddressBase)
    assert addr.addr == 67890
    assert addr.device_type == "dram"
    assert addr.config is config

    print(f"  Created: {addr}")
    print("[PASS] DRAM translator tests passed")


def test_sram_translator():
    """Test SRAM address translator."""
    print("\nTesting SRAM translator...")

    config = SramConfig()
    translator = SramAddressTranslator(config)

    # Test translation
    addr = translator.translate(11111)

    assert isinstance(addr, SramAddress)
    assert isinstance(addr, AddressBase)
    assert addr.addr == 11111
    assert addr.device_type == "sram"
    assert addr.config is config

    print(f"  Created: {addr}")
    print("[PASS] SRAM translator tests passed")


def test_inheritance():
    """Test inheritance relationships."""
    print("\nTesting inheritance...")

    nand_config = NandConfig(num_channels=4, num_plane=2, num_block=1024, num_pages=2048)
    dram_config = DramConfig()
    sram_config = SramConfig()

    nand_translator = NandAddressTranslator(nand_config)
    dram_translator = DramAddressTranslator(dram_config)
    sram_translator = SramAddressTranslator(sram_config)

    # Test translator inheritance
    assert isinstance(nand_translator, AddressTranslatorBase)
    assert isinstance(dram_translator, AddressTranslatorBase)
    assert isinstance(sram_translator, AddressTranslatorBase)

    # Test address inheritance
    nand_addr = nand_translator.translate(100)
    dram_addr = dram_translator.translate(200)
    sram_addr = sram_translator.translate(300)

    assert isinstance(nand_addr, AddressBase)
    assert isinstance(dram_addr, AddressBase)
    assert isinstance(sram_addr, AddressBase)

    print("[PASS] Inheritance tests passed")


def test_base_class_abstract():
    """Test that base class raises NotImplementedError."""
    print("\nTesting abstract base class...")

    base = AddressTranslatorBase()

    try:
        base.translate(123)
        assert False, "Should have raised NotImplementedError"
    except NotImplementedError as e:
        assert "Subclasses must implement translate() method" in str(e)
        print(f"  Correctly raised: {e}")

    print("[PASS] Abstract base class tests passed")


def test_nand_address_setters():
    """Test address component setters"""
    print("\nTesting address setters...")
    config = NandConfig(num_channels=4, num_plane=2, num_block=1024, num_pages=2048)
    addr = NandAddress(0, config)

    addr.channel = 2
    addr.plane = 1
    addr.page = 100
    addr.block = 500

    assert addr.channel == 2
    assert addr.plane == 1
    assert addr.page == 100
    assert addr.block == 500
    print("[PASS] Setters work correctly")


def test_nand_address_increment():
    """Test address increment operation"""
    print("\nTesting address increment...")
    config = NandConfig(num_channels=4, num_plane=2, num_block=1024, num_pages=2048)
    addr = NandAddress(0, config)

    new_addr = addr + 1
    assert new_addr.block == 1

    new_addr = addr + 10
    assert new_addr.block == 10
    print("[PASS] Increment operations work")


def test_nand_address_carry():
    """Test carry logic across address levels"""
    print("\nTesting carry logic...")
    config = NandConfig(num_channels=4, num_plane=2, num_block=1024, num_pages=2048)

    # Test block -> page carry
    addr = NandAddress(0, config)
    addr.block = 1023
    new_addr = addr + 1
    assert new_addr.block == 0
    assert new_addr.page == 1

    # Test page -> plane carry
    addr = NandAddress(0, config)
    addr.page = 2047
    addr.block = 1023
    new_addr = addr + 1
    assert new_addr.block == 0
    assert new_addr.page == 0
    assert new_addr.plane == 1

    print("[PASS] Carry logic works correctly")


def test_nand_address_overflow():
    """Test overflow detection"""
    print("\nTesting overflow detection...")
    config = NandConfig(num_channels=4, num_plane=2, num_block=1024, num_pages=2048)
    addr = NandAddress(0, config)
    addr.channel = 3
    addr.plane = 1
    addr.page = 2047
    addr.block = 1023

    try:
        new_addr = addr + 1
        assert False, "Should have raised OverflowError"
    except OverflowError:
        pass
    print("[PASS] Overflow detection works")


def run_all_tests():
    """Run all test cases."""
    print("=" * 60)
    print("Running Address Translation Module Tests")
    print("=" * 60)

    test_imports()
    test_nand_translator()
    test_dram_translator()
    test_sram_translator()
    test_inheritance()
    test_base_class_abstract()
    test_nand_address_setters()
    test_nand_address_increment()
    test_nand_address_carry()
    test_nand_address_overflow()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
