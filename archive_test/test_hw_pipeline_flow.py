import math

from Desim import SimSession

from nandmachine.commands.macro import MatMul, NandMmap, SramPrefetch
from nandmachine.commands.micro import DataForward, MemoryOperation, NandPageRead, NandRequest
from nandmachine.config.config import NandConfig
from nandmachine.simulator.hardware.nand import NandSimCore
from nandmachine.simulator.hardware.xpu import xPU
from nandmachine.simulator.runtime.addr import NandAddress
from nandmachine.simulator.runtime.manager import NandFileSystem
from nandmachine.simulator.runtime.tables import DeviceType, NandFileMeta, Permission


def make_config() -> NandConfig:
    return NandConfig(
        num_channels=2,
        num_plane=2,
        num_block=32,
        num_pages=32,
        tRead=100.0,
        tWrite=1000.0,
        tErase=10000.0,
        page_size=1,
    )


def test_hw_pipeline_flow_runs_and_maps_sram_destination():
    config = make_config()
    weight_m = 64
    weight_n = 64
    weight_bits = 16
    batch = 8

    page_bytes = config.page_size_bytes
    weight_bytes = weight_m * weight_n * (weight_bits // 8)
    weight_pages = math.ceil(weight_bytes / page_bytes)

    nfs = NandFileSystem(config)
    weight_meta = NandFileMeta(
        file_name="w_64x64_fp16.bin",
        num_pages=weight_pages,
        file_size=weight_bytes,
        permission=Permission.READ,
        type="weight",
    )
    file_id = nfs.create_static_file(weight_meta)

    weight_logic_base = 1 << 12
    sram_logic_base = weight_logic_base + weight_pages + 32

    commands = [
        NandMmap(file_id=file_id, pre_alloc_logic_addr=weight_logic_base),
        SramPrefetch(
            prefetch_addr=weight_logic_base,
            num_pages=weight_pages,
            pre_alloc_logic_addr=sram_logic_base,
        ),
        MatMul(dim=(batch, weight_m, weight_n), addr=sram_logic_base, weight_bits=weight_bits),
    ]

    SimSession.reset()
    SimSession.init()

    xpu = xPU(config)
    xpu.runtime_manager.load_nand_file_system(nfs.nand_file_table, nfs.nand_free_table)
    xpu.load_command(commands)

    SimSession.scheduler.run()

    src_map = xpu.runtime_manager.page_table.translate(weight_logic_base)
    dst_map = xpu.runtime_manager.page_table.translate(sram_logic_base)

    assert SimSession.sim_time.cycle > 0
    assert src_map is not None and src_map[0] == DeviceType.NAND
    assert dst_map is not None and dst_map[0] == DeviceType.SRAM


def test_data_forward_accepts_device_type_enum():
    config = make_config()
    addr = NandAddress(0, config)
    addr.channel = 0
    addr.plane = 0
    addr.block = 0
    addr.page = 0

    request = NandRequest(
        MemoryOperation(
            NandPageRead(addr.addr),
            DataForward(DeviceType.NAND, "base", addr.addr, 0),
        )
    )
    core = NandSimCore(config)

    # DeviceType enum should be normalized and handled by the core.
    finish_time = core.handle_request(request, 0.0)
    assert finish_time > 0


def test_nand_request_rejects_non_memory_operation_items():
    try:
        NandRequest([NandPageRead(0)])
        assert False, "Expected TypeError for invalid request operation payload"
    except TypeError:
        pass
