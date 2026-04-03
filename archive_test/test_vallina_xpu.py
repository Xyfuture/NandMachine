from Desim import SimSession

import nandmachine.simulator.hardware.xpu as xpu_module
import nandmachine.simulator.hardware.vallina_xpu as vallina_xpu_module

from nandmachine.commands.macro import (
    FlashAttnOp,
    MatMulOp,
    SramPrefetch,
    SramPrefetchRelease,
    VectorOp,
)
from nandmachine.config.config import NandConfig


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


def test_vallina_compute_engine_adds_tread_only_to_matmul_and_flashattn(monkeypatch):
    SimSession.reset()
    SimSession.init()

    engine = vallina_xpu_module.VallinaComputeEngine(make_config())

    def fake_execute_macro_op(self, macro_op):
        if isinstance(macro_op, MatMulOp):
            return 10.0
        if isinstance(macro_op, FlashAttnOp):
            return 20.0
        if isinstance(macro_op, VectorOp):
            return 30.0
        raise TypeError(f"Unsupported macro op type: {type(macro_op).__name__}")

    monkeypatch.setattr(
        xpu_module.ComputeEngine,
        "execute_macro_op",
        fake_execute_macro_op,
    )

    matmul_time_ns = engine.execute_macro_op(MatMulOp(dim=(2, 4, 8), weight_bits=16))
    flash_time_ns = engine.execute_macro_op(
        FlashAttnOp(
            qk_bmm_shape=(2, 4, 8, 6),
            sv_bmm_shape=(2, 4, 6, 3),
            softmax_shape=(4, 6),
            weight_bits=16,
        )
    )
    vector_time_ns = engine.execute_macro_op(
        VectorOp(vector_op_type="rms_norm", vector_shape=[2, 16], weight_bits=16)
    )

    assert matmul_time_ns == 14
    assert flash_time_ns == 28
    assert vector_time_ns == 30

    SimSession.reset()


def test_vallina_xpu_load_command_routes_prefetch_to_dedicated_engine():
    SimSession.reset()
    SimSession.init()

    vector_norm = VectorOp(vector_op_type="rms_norm", vector_shape=[2, 16], weight_bits=16)
    prefetch = SramPrefetch(num_prefetch_pages=2).with_inputs(vector_norm)
    matmul = MatMulOp(dim=(2, 16, 8), weight_bits=16).with_inputs(prefetch)
    release = SramPrefetchRelease().with_inputs(matmul)
    vector_act = VectorOp(vector_op_type="silu_mul", vector_shape=[2, 8], weight_bits=16).with_inputs(matmul)

    sim_xpu = vallina_xpu_module.VallinaXPU(make_config())
    sim_xpu.load_command([vector_norm, prefetch, matmul, release, vector_act])

    assert [slot.payload for slot in sim_xpu.prefetch_engine.prefetch_command_queue] == [
        prefetch
    ]
    assert [slot.payload for slot in sim_xpu.compute_engine.command_queue] == [
        vector_norm,
        matmul,
        vector_act,
    ]
    assert sim_xpu.compute_engine.command_queue[1].input_slots == [
        sim_xpu.prefetch_engine.prefetch_command_queue[0]
    ]
    assert sim_xpu.compute_engine.command_queue[2].input_slots == [
        sim_xpu.compute_engine.command_queue[1]
    ]
    assert sim_xpu.prefetch_engine.prefetch_command_queue[0].input_slots == [
        sim_xpu.compute_engine.command_queue[0]
    ]

    SimSession.reset()


def test_vallina_xpu_prefetch_engine_waits_one_ns(monkeypatch):
    SimSession.reset()
    SimSession.init()

    vector_norm = VectorOp(vector_op_type="rms_norm", vector_shape=[2, 16], weight_bits=16)
    prefetch = SramPrefetch(num_prefetch_pages=2).with_inputs(vector_norm)
    matmul = MatMulOp(dim=(2, 16, 8), weight_bits=16).with_inputs(prefetch)
    release = SramPrefetchRelease().with_inputs(matmul)
    vector_act = VectorOp(vector_op_type="silu_mul", vector_shape=[2, 8], weight_bits=16).with_inputs(matmul)

    sim_xpu = vallina_xpu_module.VallinaXPU(make_config())
    sim_xpu.load_command([vector_norm, prefetch, matmul, release, vector_act])

    def fake_execute_macro_op(macro_op):
        if isinstance(macro_op, MatMulOp):
            return 7.0
        if isinstance(macro_op, VectorOp):
            if macro_op.vector_op_type == "rms_norm":
                return 5.0
            if macro_op.vector_op_type == "silu_mul":
                return 3.0
        raise TypeError(f"Unsupported macro op type: {type(macro_op).__name__}")

    monkeypatch.setattr(
        sim_xpu.compute_engine,
        "execute_macro_op",
        fake_execute_macro_op,
    )

    SimSession.scheduler.run()

    assert int(SimSession.sim_time.cycle) == 20

    SimSession.reset()
