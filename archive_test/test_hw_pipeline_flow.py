from Desim import SimSession

from nandmachine.commands.macro import (
    All2AllOp,
    FlashAttnOp,
    MatMulOp,
    SramPrefetch,
    SramPrefetchRelease,
    VectorOp,
)
from nandmachine.config.config import NandConfig
from nandmachine.simulator.entry_point import run_macro_ops
from nandmachine.simulator.hardware.xpu import xPU


def make_config() -> NandConfig:
    return NandConfig(
        num_channels=1,
        num_plane=2,
        num_block=8,
        num_pages=32,
        tRead=4.0,
        tWrite=8.0,
        tErase=16.0,
        page_size=16,
        sram_threshold=64,
    )


def test_xpu_instances_keep_their_own_compute_settings():
    config = make_config()

    SimSession.reset()
    SimSession.init()

    fast_xpu = xPU(config, compile_mode="heuristic-GPU")
    throughput_xpu = xPU(config, compile_mode="heuristic-our-throughput")

    assert fast_xpu.compute_engine.compile_mode == "heuristic-GPU"
    assert throughput_xpu.compute_engine.compile_mode == "heuristic-our-throughput"
    assert fast_xpu.compute_engine.device_name == "A100_80GB"
    assert throughput_xpu.compute_engine.device_name == "A100_80GB"

    SimSession.reset()


def test_xpu_routes_transfer_ops_to_transfer_engine():
    config = make_config()

    prefetch = SramPrefetch(num_prefetch_pages=2)
    transfer = All2AllOp(num_gpus=4, data_size=128, weight_bits=16).with_inputs(prefetch)
    matmul = MatMulOp(dim=(2, 16, 8), weight_bits=16).with_inputs(transfer)

    SimSession.reset()
    SimSession.init()

    sim_xpu = xPU(config)
    sim_xpu.load_command([prefetch, transfer, matmul])

    assert [slot.payload for slot in sim_xpu.prefetch_engine.prefetch_command_queue] == [
        prefetch
    ]
    assert [slot.payload for slot in sim_xpu.transfer_engine.transfer_command_queue] == [
        transfer
    ]
    assert [slot.payload for slot in sim_xpu.compute_engine.command_queue] == [matmul]
    assert sim_xpu.transfer_engine.transfer_command_queue[0].input_slots == [
        sim_xpu.prefetch_engine.prefetch_command_queue[0]
    ]
    assert sim_xpu.compute_engine.command_queue[0].input_slots == [
        sim_xpu.transfer_engine.transfer_command_queue[0]
    ]

    SimSession.reset()


def test_hw_pipeline_flow_runs_with_current_macro_ops():
    config = make_config()

    vector_norm = VectorOp(vector_op_type="rms_norm", vector_shape=[2, 16], weight_bits=16)
    prefetch_linear = SramPrefetch(num_prefetch_pages=4).with_inputs(vector_norm)
    matmul = MatMulOp(dim=(2, 16, 8), weight_bits=16).with_inputs(prefetch_linear)
    vector_act = VectorOp(vector_op_type="silu_mul", vector_shape=[2, 8], weight_bits=16).with_inputs(
        matmul
    )
    prefetch_attn = SramPrefetch(num_prefetch_pages=2).with_inputs(vector_act)
    flash_attn = FlashAttnOp(
        qk_bmm_shape=(4, 2, 4, 2),
        sv_bmm_shape=(4, 2, 2, 4),
        softmax_shape=(2, 2),
        weight_bits=16,
    ).with_inputs(prefetch_attn)
    release = SramPrefetchRelease().with_inputs(flash_attn)

    final_cycle = run_macro_ops(
        config,
        [
            vector_norm,
            prefetch_linear,
            matmul,
            vector_act,
            prefetch_attn,
            flash_attn,
            release,
        ],
    )

    assert final_cycle > 0


def test_hw_pipeline_flow_runs_without_prefetch_or_release():
    config = make_config()

    vector_norm = VectorOp(vector_op_type="rms_norm", vector_shape=[2, 16], weight_bits=16)
    matmul = MatMulOp(dim=(2, 16, 8), weight_bits=16).with_inputs(vector_norm)
    vector_act = VectorOp(vector_op_type="silu_mul", vector_shape=[2, 8], weight_bits=16).with_inputs(
        matmul
    )
    flash_attn = FlashAttnOp(
        qk_bmm_shape=(4, 2, 4, 2),
        sv_bmm_shape=(4, 2, 2, 4),
        softmax_shape=(8, 2),
        weight_bits=16,
    ).with_inputs(vector_act)

    final_cycle = run_macro_ops(
        config,
        [
            vector_norm,
            matmul,
            vector_act,
            flash_attn,
        ],
    )

    assert final_cycle > 0
