from Desim import SimModule, SimSession, SimTime

import nandmachine.simulator.hardware.xpu as xpu_module

from nandmachine.commands.macro import (
    All2AllOp,
    AllGatherOp,
    AllReduceOp,
    FlashAttnOp,
    MatMulOp,
    ReduceScatterOp,
)
from nandmachine.config.config import NandConfig
from nandmachine.config.hardware_config import A100_80GB_FP16
from nandmachine.simulator.hardware.utils import DepSlot
from nandmachine.simulator.software.flash_attention import (
    FlashAttn_BatchedMatMul_Simulation,
)
from nandmachine.simulator.software.matmul import MatMul_Simulation


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


def test_matmul_get_instance_reuses_same_object():
    MatMul_Simulation.clear_caches()

    first = MatMul_Simulation.get_instance(dim=(3, 5, 7), weight_bits=16)
    second = MatMul_Simulation.get_instance(dim=(3, 5, 7), weight_bits=16)
    third = MatMul_Simulation.get_instance(dim=(3, 5, 9), weight_bits=16)

    assert first is second
    assert first is not third


def test_flashattn_get_instance_reuses_same_object():
    FlashAttn_BatchedMatMul_Simulation.clear_caches()

    first = FlashAttn_BatchedMatMul_Simulation.get_instance(
        dim=(2, 4, 8, 6),
        weight_bits=16,
        matmul_type="QK",
    )
    second = FlashAttn_BatchedMatMul_Simulation.get_instance(
        dim=(2, 4, 8, 6),
        weight_bits=16,
        matmul_type="QK",
    )
    third = FlashAttn_BatchedMatMul_Simulation.get_instance(
        dim=(2, 4, 8, 6),
        weight_bits=16,
        matmul_type="SV",
    )

    assert first is second
    assert first is not third


def test_matmul_compile_and_simulate_uses_cached_result():
    MatMul_Simulation.clear_caches()
    instance = MatMul_Simulation.get_instance(dim=(9, 1, 11), weight_bits=16)

    first_cycles = instance.compile_and_simulate(
        pcb_module=A100_80GB_FP16,
        compile_mode="heuristic-GPU",
    )
    second_cycles = instance.compile_and_simulate(
        pcb_module=A100_80GB_FP16,
        compile_mode="heuristic-GPU",
    )
    cache_info = MatMul_Simulation.compile_and_simulate.cache_info()

    assert first_cycles == second_cycles
    assert instance.best_cycle_count == first_cycles
    assert cache_info.misses == 1
    assert cache_info.hits == 1


def test_flashattn_compile_and_simulate_uses_cached_result():
    FlashAttn_BatchedMatMul_Simulation.clear_caches()
    instance = FlashAttn_BatchedMatMul_Simulation.get_instance(
        dim=(2, 1, 8, 4),
        weight_bits=16,
        matmul_type="QK",
    )

    first_cycles = instance.compile_and_simulate(
        pcb_module=A100_80GB_FP16,
        compile_mode="heuristic-GPU",
    )
    second_cycles = instance.compile_and_simulate(
        pcb_module=A100_80GB_FP16,
        compile_mode="heuristic-GPU",
    )
    cache_info = FlashAttn_BatchedMatMul_Simulation.compile_and_simulate.cache_info()

    assert first_cycles == second_cycles
    assert instance.best_cycle_count == first_cycles
    assert cache_info.misses == 1
    assert cache_info.hits == 1


def test_compute_engine_uses_cached_simulator_entrypoints(monkeypatch):
    SimSession.reset()
    SimSession.init()
    engine = xpu_module.ComputeEngine(make_config())
    matmul_calls: list[tuple[tuple[int, int, int], int]] = []
    flash_calls: list[tuple[tuple[int, int, int, int], int, str]] = []
    softmax_calls: list[tuple[tuple[int, int], int]] = []

    class FakeMatMulSimulation:
        def compile_and_simulate(self, pcb_module, compile_mode):
            return 321

    class FakeFlashSimulation:
        def __init__(self, cycles: int):
            self.cycles = cycles

        def compile_and_simulate(self, pcb_module, compile_mode):
            return self.cycles

    class FakeSoftmaxSimulation:
        def __init__(self, dim, weight_bits):
            softmax_calls.append((dim, weight_bits))

        def compile_and_simulate(self, pcb_module, compile_mode):
            return 30

    def fake_matmul_get_instance(dim, weight_bits=16):
        matmul_calls.append((dim, weight_bits))
        return FakeMatMulSimulation()

    def fake_flash_get_instance(dim, weight_bits=16, matmul_type="QK"):
        flash_calls.append((dim, weight_bits, matmul_type))
        if matmul_type == "QK":
            return FakeFlashSimulation(50)
        if matmul_type == "SV":
            return FakeFlashSimulation(70)
        raise ValueError(f"Unsupported matmul_type: {matmul_type}")

    monkeypatch.setattr(
        xpu_module.MatMul_Simulation,
        "get_instance",
        staticmethod(fake_matmul_get_instance),
    )
    monkeypatch.setattr(
        xpu_module.FlashAttn_BatchedMatMul_Simulation,
        "get_instance",
        staticmethod(fake_flash_get_instance),
    )
    monkeypatch.setattr(
        xpu_module,
        "Softmax_Simulation",
        FakeSoftmaxSimulation,
    )

    matmul_cycles = engine.execute_macro_op(MatMulOp(dim=(9, 1, 11), weight_bits=16))
    flash_cycles = engine.execute_macro_op(
        FlashAttnOp(
            qk_bmm_shape=(2, 4, 8, 6),
            sv_bmm_shape=(2, 4, 6, 3),
            softmax_shape=(4, 6),
            weight_bits=16,
        )
    )

    assert matmul_cycles == 321
    assert flash_cycles == 150
    assert matmul_calls == [((9, 1, 11), 16)]
    assert flash_calls == [
        ((2, 4, 8, 6), 16, "QK"),
        ((2, 4, 6, 3), 16, "SV"),
    ]
    assert softmax_calls == [((4, 6), 16)]

    SimSession.reset()


def test_compute_engine_global_weight_bits_override_all_latency_paths(monkeypatch):
    SimSession.reset()
    SimSession.init()

    engine = xpu_module.ComputeEngine(make_config(), weight_bits=8)

    matmul_calls: list[tuple[tuple[int, int, int], int]] = []
    flash_calls: list[tuple[tuple[int, int, int, int], int, str]] = []
    softmax_calls: list[tuple[tuple[int, int], int]] = []

    class FakeMatMulSimulation:
        def compile_and_simulate(self, pcb_module, compile_mode):
            return 101

    class FakeFlashSimulation:
        def __init__(self, cycles: int):
            self.cycles = cycles

        def compile_and_simulate(self, pcb_module, compile_mode):
            return self.cycles

    class FakeSoftmaxSimulation:
        def __init__(self, dim, weight_bits):
            softmax_calls.append((dim, weight_bits))

        def compile_and_simulate(self, pcb_module, compile_mode):
            return 31

    def fake_matmul_get_instance(dim, weight_bits=16):
        matmul_calls.append((dim, weight_bits))
        return FakeMatMulSimulation()

    def fake_flash_get_instance(dim, weight_bits=16, matmul_type="QK"):
        flash_calls.append((dim, weight_bits, matmul_type))
        if matmul_type == "QK":
            return FakeFlashSimulation(50)
        if matmul_type == "SV":
            return FakeFlashSimulation(70)
        raise ValueError(f"Unsupported matmul_type: {matmul_type}")

    monkeypatch.setattr(
        xpu_module.MatMul_Simulation,
        "get_instance",
        staticmethod(fake_matmul_get_instance),
    )
    monkeypatch.setattr(
        xpu_module.FlashAttn_BatchedMatMul_Simulation,
        "get_instance",
        staticmethod(fake_flash_get_instance),
    )
    monkeypatch.setattr(
        xpu_module,
        "Softmax_Simulation",
        FakeSoftmaxSimulation,
    )

    matmul_cycles = engine.execute_macro_op(MatMulOp(dim=(9, 1, 11), weight_bits=16))
    flash_cycles = engine.execute_macro_op(
        FlashAttnOp(
            qk_bmm_shape=(2, 4, 8, 6),
            sv_bmm_shape=(2, 4, 6, 3),
            softmax_shape=(4, 6),
            weight_bits=16,
        )
    )

    assert matmul_cycles == 101
    assert flash_cycles == 151
    assert matmul_calls == [((9, 1, 11), 8)]
    assert flash_calls == [
        ((2, 4, 8, 6), 8, "QK"),
        ((2, 4, 6, 3), 8, "SV"),
    ]
    assert softmax_calls == [((4, 6), 8)]

    SimSession.reset()


def test_transfer_engine_accepts_supported_transfer_ops():
    SimSession.reset()
    SimSession.init()

    engine = xpu_module.TransferEngine()

    assert engine.execute_macro_op(AllGatherOp(num_ranks=4, data_size=128)) == 1.0
    assert engine.execute_macro_op(ReduceScatterOp(num_ranks=4, data_size=128)) == 1.0

    SimSession.reset()


def test_transfer_engine_allreduce_uses_estimate_path(monkeypatch):
    SimSession.reset()
    SimSession.init()

    engine = xpu_module.TransferEngine(weight_bits=8)
    allreduce_calls: list[tuple[int, int, int]] = []

    class FakeAllReduceSimulation:
        def __init__(self, num_gpus, data_size, weight_bits):
            allreduce_calls.append((num_gpus, data_size, weight_bits))

        def compile_and_simulate(self, pcb_module, interconnect_module, compile_mode):
            return 66

    monkeypatch.setattr(
        xpu_module,
        "AllReduceSimulation",
        FakeAllReduceSimulation,
    )
    monkeypatch.setattr(
        xpu_module,
        "get_interconnect_for_device_or_raise",
        lambda device_name, device_count: object(),
    )

    allreduce_cycles = engine.execute_macro_op(
        AllReduceOp(num_ranks=4, data_size=128, weight_bits=16)
    )

    assert allreduce_cycles == 66
    assert allreduce_calls == [(4, 64, 8)]

    SimSession.reset()


def test_transfer_engine_all2all_uses_estimate_path(monkeypatch):
    SimSession.reset()
    SimSession.init()

    engine = xpu_module.TransferEngine(weight_bits=8)
    all2all_calls: list[tuple[int, int, int]] = []

    class FakeAllToAllSimulation:
        def __init__(self, num_gpus, data_size, weight_bits):
            all2all_calls.append((num_gpus, data_size, weight_bits))

        def compile_and_simulate(self, pcb_module, interconnect_module, compile_mode):
            return 77

    monkeypatch.setattr(
        xpu_module,
        "AllToAllPrimitive_Simulation",
        FakeAllToAllSimulation,
    )
    monkeypatch.setattr(
        xpu_module,
        "get_interconnect_for_device_or_raise",
        lambda device_name, device_count: object(),
    )

    all2all_cycles = engine.execute_macro_op(
        All2AllOp(num_gpus=4, data_size=128, weight_bits=16)
    )

    assert all2all_cycles == 77
    assert all2all_calls == [(4, 64, 8)]

    SimSession.reset()


def test_transfer_engine_rejects_compute_ops():
    SimSession.reset()
    SimSession.init()

    engine = xpu_module.TransferEngine()

    try:
        engine.execute_macro_op(MatMulOp(dim=(2, 4, 8), weight_bits=16))
        assert False, "Expected TypeError for compute op in transfer engine"
    except TypeError:
        pass

    SimSession.reset()


def test_transfer_engine_waits_for_dependency_before_running(monkeypatch):
    class DelayedFinish(SimModule):
        def __init__(self, slot: DepSlot):
            super().__init__()
            self.slot = slot
            self.register_coroutine(self.process)

        def process(self):
            SimModule.wait_time(SimTime(5))
            self.slot.is_finished = True
            self.slot.finish_event.notify(SimTime(1))

    SimSession.reset()
    SimSession.init()

    dependency_slot = DepSlot(AllReduceOp(num_ranks=4, data_size=64, weight_bits=16))
    transfer_slot = DepSlot(All2AllOp(num_gpus=4, data_size=128, weight_bits=16))
    transfer_slot.input_slots = [dependency_slot]

    engine = xpu_module.TransferEngine()
    engine.load_command_queue([transfer_slot])

    execute_start_cycles: list[int] = []

    def fake_execute_macro_op(macro_op):
        execute_start_cycles.append(int(SimSession.sim_time.cycle))
        return 3.0

    monkeypatch.setattr(engine, "execute_macro_op", fake_execute_macro_op)

    DelayedFinish(dependency_slot)
    SimSession.scheduler.run()

    assert execute_start_cycles == [6]
    assert transfer_slot.is_finished is True

    SimSession.reset()
