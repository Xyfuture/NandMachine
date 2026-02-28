from nandmachine.commands.micro import DataForward, MemoryOperation, NandPageRead, NandRequest
from nandmachine.config.config import NandConfig
from nandmachine.simulator.hardware.nand import NandSimCore
from nandmachine.simulator.runtime.addr import NandAddress


def make_config() -> NandConfig:
    return NandConfig(
        num_channels=2,
        num_plane=2,
        num_block=16,
        num_pages=256,
        tRead=10.0,
        tWrite=100.0,
        tErase=1000.0,
    )


def make_nand_addr(config: NandConfig, channel: int, plane: int, block: int, page: int) -> int:
    addr = NandAddress(0, config)
    addr.channel = channel
    addr.plane = plane
    addr.block = block
    addr.page = page
    return addr.addr


def make_core(config: NandConfig) -> NandSimCore:
    return NandSimCore(
        nand_config=config,
        t_nand_to_base=3.0,
        t_base_to_xpu=2.0,
        t_sram_read=1.5,
    )


def test_single_nand_read_latency():
    config = make_config()
    core = make_core(config)

    addr = make_nand_addr(config, channel=0, plane=0, block=0, page=0)
    req = NandRequest(MemoryOperation(NandPageRead(addr)))

    end_time = core.handle_request(req, arrive_time=5.0)
    assert end_time == 15.0


def test_same_plane_reads_are_serialized_across_requests():
    config = make_config()
    core = make_core(config)

    addr0 = make_nand_addr(config, channel=0, plane=0, block=0, page=0)
    addr1 = make_nand_addr(config, channel=0, plane=0, block=0, page=1)

    req0 = NandRequest(MemoryOperation(NandPageRead(addr0)))
    req1 = NandRequest(MemoryOperation(NandPageRead(addr1)))

    first_end = core.handle_request(req0, arrive_time=0.0)
    second_end = core.handle_request(req1, arrive_time=0.0)

    assert first_end == 10.0
    assert second_end == 20.0


def test_different_planes_can_overlap():
    config = make_config()
    core = make_core(config)

    addr_plane0 = make_nand_addr(config, channel=0, plane=0, block=0, page=0)
    addr_plane1 = make_nand_addr(config, channel=0, plane=1, block=0, page=0)

    req_plane0 = NandRequest(MemoryOperation(NandPageRead(addr_plane0)))
    req_plane1 = NandRequest(MemoryOperation(NandPageRead(addr_plane1)))

    end0 = core.handle_request(req_plane0, arrive_time=0.0)
    end1 = core.handle_request(req_plane1, arrive_time=0.0)

    assert end0 == 10.0
    assert end1 == 10.0


def test_full_pipeline_read_and_forward_path():
    config = make_config()
    core = make_core(config)

    addr = make_nand_addr(config, channel=1, plane=0, block=0, page=0)
    op = MemoryOperation(
        NandPageRead(addr),
        DataForward(src_type="nand", dst_type="base", src=addr, dst=1),
        DataForward(src_type="base", dst_type="xpu", src=1, dst=1),
    )
    req = NandRequest(op)

    end_time = core.handle_request(req, arrive_time=0.0)
    assert end_time == 15.0


def test_memory_operations_in_one_request_run_in_parallel():
    config = make_config()
    core = make_core(config)

    addr0 = make_nand_addr(config, channel=0, plane=0, block=0, page=0)
    addr1 = make_nand_addr(config, channel=0, plane=1, block=0, page=0)

    op0 = MemoryOperation(NandPageRead(addr0))
    op1 = MemoryOperation(NandPageRead(addr1))
    req = NandRequest(op0, op1)

    end_time = core.handle_request(req, arrive_time=0.0)

    # 两个 operation 并行发起，且落在不同 plane，可同时在 t=10 完成
    assert end_time == 10.0


def test_forward_channel_mismatch_raises_value_error():
    config = make_config()
    core = make_core(config)

    addr = make_nand_addr(config, channel=0, plane=0, block=0, page=0)
    req = NandRequest(
        MemoryOperation(
            NandPageRead(addr),
            DataForward(src_type="nand", dst_type="base", src=addr, dst=1),
        )
    )

    try:
        core.handle_request(req, arrive_time=0.0)
        assert False, "Expected ValueError for channel mismatch"
    except ValueError:
        pass


def test_forward_without_endpoint_uses_last_channel():
    config = make_config()
    core = make_core(config)

    addr = make_nand_addr(config, channel=1, plane=1, block=0, page=0)
    req = NandRequest(
        MemoryOperation(
            NandPageRead(addr),
            DataForward(src_type="nand", dst_type="base"),
        )
    )

    end_time = core.handle_request(req, arrive_time=0.0)
    assert end_time == 13.0
