"""Tests for NandFileMeta/NandFileEntry/NandFileTable."""

from nandmachine.simulator.runtime.tables import (
    NandFileEntry,
    NandFileMeta,
    NandFileTable,
    Permission,
)


def test_meta_auto_increment_id():
    NandFileMeta.reset_id_counter()

    meta0 = NandFileMeta("a.bin", 1, 4096)
    meta1 = NandFileMeta("b.bin", 2, 8192)
    meta2 = NandFileMeta("c.bin", 3, 12288)

    assert meta0.file_id == 0
    assert meta1.file_id == 1
    assert meta2.file_id == 2
    assert NandFileMeta.peek_next_file_id() == 3


def test_table_behavior_and_no_id_reuse():
    NandFileMeta.reset_id_counter()
    table = NandFileTable()

    meta0 = NandFileMeta("w0.bin", 2, 8192, permission=Permission.READ, type="weight", source="unit")
    entry0 = NandFileEntry(meta0, [10, 11])
    table.add_entry(entry0)
    assert table.get_file_by_id(meta0.file_id) is entry0
    assert entry0.file_id == meta0.file_id

    meta1 = NandFileMeta("w1.bin", 1, 4096)
    entry1 = NandFileEntry(meta1, [12])
    table.add_entry(entry1)

    table.remove_entry(meta0.file_id)
    assert table.get_file_by_id(meta0.file_id) is None

    meta2 = NandFileMeta("w2.bin", 1, 4096)
    assert meta2.file_id == 2


if __name__ == "__main__":
    test_meta_auto_increment_id()
    test_table_behavior_and_no_id_reuse()
    print("[SUCCESS] Nand file table tests passed")
