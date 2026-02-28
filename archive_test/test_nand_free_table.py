"""
Test suite for NandFreeTable implementation.
Tests allocation, freeing, and page checking functionality.
"""

from nandmachine.config.config import NandConfig
from nandmachine.simulator.runtime.addr import NandBlockAddress, NandAddress
from nandmachine.simulator.runtime.tables import NandFreeTable


def test_basic_allocation():
    """Test basic page allocation in a single block."""
    print("Test 1: Basic allocation")

    config = NandConfig(
        num_channels=2,
        num_plane=2,
        num_block=10,
        num_pages=256
    )

    free_table = NandFreeTable(config)
    block_addr = NandBlockAddress.from_components(
        channel=0, plane=0, block=5, config=config
    )

    # Allocate first few pages
    addr1 = free_table.allocate(block_addr)
    addr2 = free_table.allocate(block_addr)
    addr3 = free_table.allocate(block_addr)

    # Verify addresses are not None
    assert addr1 is not None, "First allocation should not be None"
    assert addr2 is not None, "Second allocation should not be None"
    assert addr3 is not None, "Third allocation should not be None"

    # Verify addresses
    nand_addr1 = NandAddress(addr1, config)
    nand_addr2 = NandAddress(addr2, config)
    nand_addr3 = NandAddress(addr3, config)

    assert nand_addr1.page == 0, f"Expected page 0, got {nand_addr1.page}"
    assert nand_addr2.page == 1, f"Expected page 1, got {nand_addr2.page}"
    assert nand_addr3.page == 2, f"Expected page 2, got {nand_addr3.page}"

    # Verify channel, plane, block are preserved
    assert nand_addr1.channel == 0
    assert nand_addr1.plane == 0
    assert nand_addr1.block == 5

    print("  ✓ Pages allocated sequentially: 0, 1, 2")
    print("  ✓ Block components preserved correctly")
    print()


def test_block_full():
    """Test allocation when block is full."""
    print("Test 2: Block full scenario")

    config = NandConfig(
        num_channels=1,
        num_plane=1,
        num_block=5,
        num_pages=4  # Small number for easy testing
    )

    free_table = NandFreeTable(config)
    block_addr = NandBlockAddress.from_components(
        channel=0, plane=0, block=2, config=config
    )

    # Allocate all pages
    addrs = []
    for i in range(4):
        addr = free_table.allocate(block_addr)
        addrs.append(addr)
        assert addr is not None, f"Allocation {i} failed unexpectedly"

    # Try to allocate one more (should fail)
    addr_overflow = free_table.allocate(block_addr)
    assert addr_overflow is None, "Expected None when block is full"

    print(f"  ✓ Successfully allocated {len(addrs)} pages")
    print("  ✓ Returned None when block full")
    print()


def test_free_block():
    """Test freeing a block and reallocating."""
    print("Test 3: Free block and reallocate")

    config = NandConfig(
        num_channels=1,
        num_plane=1,
        num_block=5,
        num_pages=10
    )

    free_table = NandFreeTable(config)
    block_addr = NandBlockAddress.from_components(
        channel=0, plane=0, block=3, config=config
    )

    # Allocate a few pages
    addr1 = free_table.allocate(block_addr)
    addr2 = free_table.allocate(block_addr)
    addr3 = free_table.allocate(block_addr)

    assert addr3 is not None, "Third allocation should not be None"
    nand_addr3 = NandAddress(addr3, config)
    assert nand_addr3.page == 2, f"Expected page 2, got {nand_addr3.page}"

    # Free the block
    free_table.free(block_addr)

    # Allocate again - should start from page 0
    addr4 = free_table.allocate(block_addr)
    assert addr4 is not None, "Allocation after free should not be None"
    nand_addr4 = NandAddress(addr4, config)
    assert nand_addr4.page == 0, f"Expected page 0 after free, got {nand_addr4.page}"

    print("  ✓ Allocated pages 0, 1, 2")
    print("  ✓ After free, allocation restarted from page 0")
    print()


def test_check_free_page():
    """Test check_free_page functionality."""
    print("Test 4: Check free page")

    config = NandConfig(
        num_channels=1,
        num_plane=1,
        num_block=5,
        num_pages=10
    )

    free_table = NandFreeTable(config)
    block_addr = NandBlockAddress.from_components(
        channel=0, plane=0, block=1, config=config
    )

    # Allocate up to page 5
    for i in range(5):
        free_table.allocate(block_addr)

    # Create addresses for different pages
    addr_page4 = NandAddress(0, config)
    addr_page4.channel = 0
    addr_page4.plane = 0
    addr_page4.block = 1
    addr_page4.page = 4

    addr_page5 = NandAddress(0, config)
    addr_page5.channel = 0
    addr_page5.plane = 0
    addr_page5.block = 1
    addr_page5.page = 5

    addr_page6 = NandAddress(0, config)
    addr_page6.channel = 0
    addr_page6.plane = 0
    addr_page6.block = 1
    addr_page6.page = 6

    # Check: page 4 is already allocated (not free)
    assert not free_table.check_free_page(addr_page4.addr), "Page 4 should not be free"

    # Check: page 5 is the next writable page (free)
    assert free_table.check_free_page(addr_page5.addr), "Page 5 should be free"

    # Check: page 6 is not yet the next writable page
    assert not free_table.check_free_page(addr_page6.addr), "Page 6 should not be free yet"

    print("  ✓ Page 4 (already allocated) is not free")
    print("  ✓ Page 5 (next writable) is free")
    print("  ✓ Page 6 (future page) is not free yet")
    print()


def test_multiple_blocks():
    """Test allocation across multiple blocks independently."""
    print("Test 5: Multiple blocks independence")

    config = NandConfig(
        num_channels=2,
        num_plane=2,
        num_block=10,
        num_pages=256
    )

    free_table = NandFreeTable(config)

    # Create different block addresses
    block1 = NandBlockAddress.from_components(
        channel=0, plane=0, block=1, config=config
    )
    block2 = NandBlockAddress.from_components(
        channel=1, plane=1, block=5, config=config
    )
    block3 = NandBlockAddress.from_components(
        channel=0, plane=1, block=3, config=config
    )

    # Allocate different numbers of pages in each block
    addr1_1 = free_table.allocate(block1)
    addr1_2 = free_table.allocate(block1)

    addr2_1 = free_table.allocate(block2)

    addr3_1 = free_table.allocate(block3)
    addr3_2 = free_table.allocate(block3)
    addr3_3 = free_table.allocate(block3)

    # Verify allocations succeeded
    assert addr1_2 is not None, "Block1 allocation should not be None"
    assert addr2_1 is not None, "Block2 allocation should not be None"
    assert addr3_3 is not None, "Block3 allocation should not be None"

    # Verify each block maintains its own page counter
    nand1_2 = NandAddress(addr1_2, config)
    nand2_1 = NandAddress(addr2_1, config)
    nand3_3 = NandAddress(addr3_3, config)

    assert nand1_2.page == 1, f"Block1 should be at page 1, got {nand1_2.page}"
    assert nand2_1.page == 0, f"Block2 should be at page 0, got {nand2_1.page}"
    assert nand3_3.page == 2, f"Block3 should be at page 2, got {nand3_3.page}"

    print("  ✓ Block (0,0,1) allocated 2 pages: 0, 1")
    print("  ✓ Block (1,1,5) allocated 1 page: 0")
    print("  ✓ Block (0,1,3) allocated 3 pages: 0, 1, 2")
    print("  ✓ Each block maintains independent page counter")
    print()


def test_edge_cases():
    """Test edge cases like empty block and boundary conditions."""
    print("Test 6: Edge cases")

    config = NandConfig(
        num_channels=1,
        num_plane=1,
        num_block=3,
        num_pages=5
    )

    free_table = NandFreeTable(config)
    block_addr = NandBlockAddress.from_components(
        channel=0, plane=0, block=0, config=config
    )

    # Test: check_free_page on a never-allocated block (should be page 0)
    addr_page0 = NandAddress(0, config)
    addr_page0.channel = 0
    addr_page0.plane = 0
    addr_page0.block = 0
    addr_page0.page = 0

    assert free_table.check_free_page(addr_page0.addr), "Page 0 should be free for new block"

    # Test: free a block that was never allocated (should not crash)
    block_never_used = NandBlockAddress.from_components(
        channel=0, plane=0, block=2, config=config
    )
    free_table.free(block_never_used)  # Should not raise error

    print("  ✓ Page 0 is free for never-allocated block")
    print("  ✓ Freeing never-allocated block does not crash")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("NandFreeTable Test Suite")
    print("=" * 60)
    print()

    try:
        test_basic_allocation()
        test_block_full()
        test_free_block()
        test_check_free_page()
        test_multiple_blocks()
        test_edge_cases()

        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise

