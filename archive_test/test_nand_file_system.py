import pytest

from nandmachine.config.config import NandConfig
from nandmachine.simulator.runtime.addr import NandAddress
from nandmachine.simulator.runtime.manager import NandFileSystem
from nandmachine.simulator.runtime.tables import NandFileMeta, Permission


def make_config(
    num_channels: int = 2,
    num_plane: int = 2,
    num_block: int = 8,
    num_pages: int = 8,
) -> NandConfig:
    return NandConfig(
        num_channels=num_channels,
        num_plane=num_plane,
        num_block=num_block,
        num_pages=num_pages,
        tRead=10.0,
        tWrite=100.0,
        tErase=1000.0,
        page_size=4,
    )


def addr_to_slot(addr: int, config: NandConfig) -> int:
    nand_addr = NandAddress(addr, config)
    return nand_addr.channel * config.num_plane + nand_addr.plane


def test_create_static_file_balanced_and_contiguous_in_one_file():
    NandFileMeta.reset_id_counter()
    config = make_config(num_channels=2, num_plane=2, num_block=4, num_pages=4)
    fs = NandFileSystem(config)

    meta = NandFileMeta("w0.bin", num_pages=10, file_size=10 * 4096, permission=Permission.READ)
    file_id = fs.create_static_file(meta)
    entry = fs.nand_file_table.get_file_by_id(file_id)
    assert entry is not None

    slots = [addr_to_slot(addr, config) for addr in entry.nand_pages]
    assert slots == [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]

    counts = [slots.count(i) for i in range(config.num_channels * config.num_plane)]
    assert max(counts) - min(counts) <= 1


def test_create_static_file_is_contiguous_across_files():
    NandFileMeta.reset_id_counter()
    config = make_config(num_channels=2, num_plane=2, num_block=4, num_pages=4)
    fs = NandFileSystem(config)

    meta0 = NandFileMeta("w0.bin", num_pages=3, file_size=3 * 4096)
    id0 = fs.create_static_file(meta0)
    entry0 = fs.nand_file_table.get_file_by_id(id0)
    assert entry0 is not None

    meta1 = NandFileMeta("w1.bin", num_pages=5, file_size=5 * 4096)
    id1 = fs.create_static_file(meta1)
    entry1 = fs.nand_file_table.get_file_by_id(id1)
    assert entry1 is not None

    slots0 = [addr_to_slot(addr, config) for addr in entry0.nand_pages]
    slots1 = [addr_to_slot(addr, config) for addr in entry1.nand_pages]
    assert slots0 == [0, 1, 2]
    assert slots1 == [3, 0, 1, 2, 3]


def test_create_static_file_rolls_block_per_slot_when_full():
    NandFileMeta.reset_id_counter()
    config = make_config(num_channels=1, num_plane=2, num_block=3, num_pages=2)
    fs = NandFileSystem(config)

    meta = NandFileMeta("w0.bin", num_pages=6, file_size=6 * 4096)
    file_id = fs.create_static_file(meta)
    entry = fs.nand_file_table.get_file_by_id(file_id)
    assert entry is not None

    slot0_addrs = []
    slot1_addrs = []
    for addr in entry.nand_pages:
        nand_addr = NandAddress(addr, config)
        if nand_addr.plane == 0:
            slot0_addrs.append(nand_addr)
        else:
            slot1_addrs.append(nand_addr)

    assert [a.block for a in slot0_addrs] == [0, 0, 1]
    assert [a.page for a in slot0_addrs] == [0, 1, 0]
    assert [a.block for a in slot1_addrs] == [0, 0, 1]
    assert [a.page for a in slot1_addrs] == [0, 1, 0]


def test_create_static_file_asserts_when_out_of_space():
    NandFileMeta.reset_id_counter()
    config = make_config(num_channels=1, num_plane=1, num_block=1, num_pages=2)
    fs = NandFileSystem(config)

    meta = NandFileMeta("w0.bin", num_pages=3, file_size=3 * 4096)
    with pytest.raises(AssertionError):
        fs.create_static_file(meta)


def test_create_static_file_updates_tables():
    NandFileMeta.reset_id_counter()
    config = make_config(num_channels=1, num_plane=1, num_block=2, num_pages=4)
    fs = NandFileSystem(config)

    meta = NandFileMeta("w0.bin", num_pages=3, file_size=3 * 4096)
    file_id = fs.create_static_file(meta)
    entry = fs.nand_file_table.get_file_by_id(file_id)
    assert entry is not None
    assert entry.file_id == file_id
    assert entry.num_nand_pages == 3

    assert len(fs.nand_free_table.next_page) == 1
    next_page = list(fs.nand_free_table.next_page.values())[0]
    assert next_page == 3
