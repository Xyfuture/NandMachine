import math

from Desim import SimModule, SimSession, SimTime
import pytest

import nandmachine.simulator.software.flash_attention as flash_attention_module
import nandmachine.simulator.hardware.xpu as xpu_module
import nandmachine.simulator.software.matmul as matmul_module

from nandmachine.commands.macro import (
    All2AllOp,
    AllGatherOp,
    AllReduceOp,
    FlashAttnOp,
    MatMulOp,
    ReduceScatterOp,
    VectorOp,
)
from nandmachine.config.config import NandConfig
from nandmachine.config.hbm_hbf_architecture import (
    build_device_for_hbm_hbf_architecture_or_raise,
)
from nandmachine.config.hardware_config import A100_80GB_FP16
from nandmachine.config.hardware_config import Device, IOModule
from nandmachine.simulator.hardware.utils import DepSlot
from nandmachine.simulator.software.flash_attention import (
    FlashAttn_BatchedMatMul_Simulation,
    MatMul_Simulation as FlashAttentionMatMul_Simulation,
    Softmax_Simulation,
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


def hbm_only_architecture() -> dict[str, object]:
    return {"mode": "hbm_only"}


def h100_cli_architecture() -> dict[str, object]:
    return {"mode": "cli", "hbm_stacks": 2, "hbf_stacks": 3}


def h100_cli_architecture_more_hbm() -> dict[str, object]:
    return {"mode": "cli", "hbm_stacks": 3, "hbf_stacks": 2}


def clone_device_with_io(
    device: Device,
    *,
    total_bandwidth: float,
    hbm_bandwidth: float,
    hbf_bandwidth: float,
) -> Device:
    return Device(
        compute_module=device.compute_module,
        io_module=IOModule(
            bandwidth=total_bandwidth,
            hbm_bandwidth=hbm_bandwidth,
            hbf_bandwidth=hbf_bandwidth,
        ),
        memory_capacity_bytes=device.total_memory_capacity_bytes,
        hbm_memory_capacity_bytes=device.hbm_memory_capacity_bytes,
        hbf_memory_capacity_bytes=device.hbf_memory_capacity_bytes,
        memory_architecture_mode=device.memory_architecture_mode,
        hbm_stack_count=device.hbm_stack_count,
        hbf_stack_count=device.hbf_stack_count,
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


def test_matmul_compile_and_simulate_supports_time_ns_and_rehydrates_fields():
    MatMul_Simulation.clear_caches()
    instance = MatMul_Simulation.get_instance(dim=(9, 1, 11), weight_bits=16)

    cycles = instance.compile_and_simulate(
        pcb_module=A100_80GB_FP16,
        compile_mode="heuristic-GPU",
    )
    time_ns = instance.compile_and_simulate(
        pcb_module=A100_80GB_FP16,
        compile_mode="heuristic-GPU",
        return_unit="time_ns",
    )
    expected_time_ns = math.ceil(
        cycles * 1e9 / A100_80GB_FP16.compute_module.clock_freq
    )
    cache_info = MatMul_Simulation.compile_and_simulate.cache_info()

    assert time_ns == expected_time_ns
    assert instance.best_cycle_count == cycles
    assert instance.best_time_ns == expected_time_ns
    assert cache_info.misses == 1
    assert cache_info.hits == 1


def test_flashattn_compile_and_simulate_supports_time_ns_and_rehydrates_fields():
    FlashAttn_BatchedMatMul_Simulation.clear_caches()
    instance = FlashAttn_BatchedMatMul_Simulation.get_instance(
        dim=(2, 1, 8, 4),
        weight_bits=16,
        matmul_type="QK",
    )

    cycles = instance.compile_and_simulate(
        pcb_module=A100_80GB_FP16,
        compile_mode="heuristic-GPU",
    )
    time_ns = instance.compile_and_simulate(
        pcb_module=A100_80GB_FP16,
        compile_mode="heuristic-GPU",
        return_unit="time_ns",
    )
    expected_time_ns = math.ceil(
        cycles * 1e9 / A100_80GB_FP16.compute_module.clock_freq
    )
    cache_info = FlashAttn_BatchedMatMul_Simulation.compile_and_simulate.cache_info()

    assert time_ns == expected_time_ns
    assert instance.best_cycle_count == cycles
    assert instance.best_time_ns == expected_time_ns
    assert cache_info.misses == 1
    assert cache_info.hits == 1


def test_flashattn_batched_sv_merged_batch_extra_write_uses_cli_buffer_bandwidth(
    monkeypatch,
):
    FlashAttn_BatchedMatMul_Simulation.clear_caches()

    batch_size = 4
    m_dim = 8
    k_dim = 2
    n_dim = 16
    merged_batch_cycle_count = 10
    per_batch_cycle_count = 100
    extra_write_bytes = (batch_size - 1) * m_dim * n_dim * 2
    cli_device = build_device_for_hbm_hbf_architecture_or_raise(
        "H100_SXM",
        h100_cli_architecture(),
    )
    total_bandwidth_per_cycle = (
        cli_device.io_module.total_bandwidth / cli_device.compute_module.clock_freq
    )
    hbm_bandwidth_per_cycle = (
        cli_device.io_module.hbm_bandwidth / cli_device.compute_module.clock_freq
    )

    class FakeMatMulSimulation:
        def __init__(self, M, K, N, weight_bits=16, matmul_type="QK"):
            assert (M, N, weight_bits, matmul_type) == (m_dim, n_dim, 16, "SV")
            self.K = K

        def compile_and_simulate(self, pcb_module, compile_mode, return_unit="cycle"):
            assert pcb_module is cli_device
            assert compile_mode == "heuristic-GPU"
            assert return_unit == "cycle"
            if self.K == k_dim:
                return per_batch_cycle_count
            if self.K == k_dim * batch_size:
                return merged_batch_cycle_count
            raise AssertionError(f"Unexpected K dimension: {self.K}")

    monkeypatch.setattr(
        flash_attention_module,
        "MatMul_Simulation",
        FakeMatMulSimulation,
    )
    monkeypatch.setattr(
        flash_attention_module,
        "ENABLE_FLASHATTN_CLI_HBF_SRAM_BUFFER",
        True,
    )
    with_buffer = FlashAttn_BatchedMatMul_Simulation(
        dim=(batch_size, m_dim, k_dim, n_dim),
        weight_bits=16,
        matmul_type="SV",
    )
    with_buffer_cycle_count = with_buffer.compile_and_simulate(
        pcb_module=cli_device,
        compile_mode="heuristic-GPU",
        return_unit="cycle",
    )

    assert with_buffer.best_mapping == "merged_batch"
    assert with_buffer_cycle_count == merged_batch_cycle_count + math.ceil(
        extra_write_bytes / total_bandwidth_per_cycle
    )

    FlashAttn_BatchedMatMul_Simulation.clear_caches()
    monkeypatch.setattr(
        flash_attention_module,
        "ENABLE_FLASHATTN_CLI_HBF_SRAM_BUFFER",
        False,
    )
    without_buffer = FlashAttn_BatchedMatMul_Simulation(
        dim=(batch_size, m_dim, k_dim, n_dim),
        weight_bits=16,
        matmul_type="SV",
    )
    without_buffer_cycle_count = without_buffer.compile_and_simulate(
        pcb_module=cli_device,
        compile_mode="heuristic-GPU",
        return_unit="cycle",
    )

    assert without_buffer.best_mapping == "merged_batch"
    assert without_buffer_cycle_count == merged_batch_cycle_count + math.ceil(
        extra_write_bytes / hbm_bandwidth_per_cycle
    )
    assert with_buffer_cycle_count <= without_buffer_cycle_count


def test_flashattn_batched_qk_merged_batch_adds_no_extra_write_under_cli_buffer_toggle(
    monkeypatch,
):
    FlashAttn_BatchedMatMul_Simulation.clear_caches()

    batch_size = 4
    m_dim = 8
    k_dim = 2
    n_dim = 16
    merged_batch_cycle_count = 10
    per_batch_cycle_count = 100
    cli_device = build_device_for_hbm_hbf_architecture_or_raise(
        "H100_SXM",
        h100_cli_architecture(),
    )

    class FakeMatMulSimulation:
        def __init__(self, M, K, N, weight_bits=16, matmul_type="QK"):
            assert (M, N, weight_bits, matmul_type) == (m_dim, n_dim, 16, "QK")
            self.K = K

        def compile_and_simulate(self, pcb_module, compile_mode, return_unit="cycle"):
            assert pcb_module is cli_device
            assert compile_mode == "heuristic-GPU"
            assert return_unit == "cycle"
            if self.K == k_dim:
                return per_batch_cycle_count
            if self.K == k_dim * batch_size:
                return merged_batch_cycle_count
            raise AssertionError(f"Unexpected K dimension: {self.K}")

    monkeypatch.setattr(
        flash_attention_module,
        "MatMul_Simulation",
        FakeMatMulSimulation,
    )
    monkeypatch.setattr(
        flash_attention_module,
        "ENABLE_FLASHATTN_CLI_HBF_SRAM_BUFFER",
        True,
    )
    with_buffer = FlashAttn_BatchedMatMul_Simulation(
        dim=(batch_size, m_dim, k_dim, n_dim),
        weight_bits=16,
        matmul_type="QK",
    )
    with_buffer_cycle_count = with_buffer.compile_and_simulate(
        pcb_module=cli_device,
        compile_mode="heuristic-GPU",
        return_unit="cycle",
    )

    FlashAttn_BatchedMatMul_Simulation.clear_caches()
    monkeypatch.setattr(
        flash_attention_module,
        "ENABLE_FLASHATTN_CLI_HBF_SRAM_BUFFER",
        False,
    )
    without_buffer = FlashAttn_BatchedMatMul_Simulation(
        dim=(batch_size, m_dim, k_dim, n_dim),
        weight_bits=16,
        matmul_type="QK",
    )
    without_buffer_cycle_count = without_buffer.compile_and_simulate(
        pcb_module=cli_device,
        compile_mode="heuristic-GPU",
        return_unit="cycle",
    )

    assert with_buffer.best_mapping == "merged_batch"
    assert without_buffer.best_mapping == "merged_batch"
    assert with_buffer_cycle_count == merged_batch_cycle_count
    assert without_buffer_cycle_count == merged_batch_cycle_count


def test_flashattn_batched_sv_merged_batch_matches_between_hbm_only_and_csi(
    monkeypatch,
):
    FlashAttn_BatchedMatMul_Simulation.clear_caches()

    batch_size = 4
    m_dim = 8
    k_dim = 2
    n_dim = 16
    merged_batch_cycle_count = 10
    per_batch_cycle_count = 100
    hbm_only_device = build_device_for_hbm_hbf_architecture_or_raise(
        "H100_SXM",
        hbm_only_architecture(),
    )
    csi_device = build_device_for_hbm_hbf_architecture_or_raise(
        "H100_SXM",
        {"mode": "csi"},
    )
    total_bandwidth_per_cycle = (
        hbm_only_device.io_module.total_bandwidth
        / hbm_only_device.compute_module.clock_freq
    )
    extra_write_bytes = (batch_size - 1) * m_dim * n_dim * 2

    class FakeMatMulSimulation:
        def __init__(self, M, K, N, weight_bits=16, matmul_type="QK"):
            assert (M, N, weight_bits, matmul_type) == (m_dim, n_dim, 16, "SV")
            self.K = K

        def compile_and_simulate(self, pcb_module, compile_mode, return_unit="cycle"):
            assert compile_mode == "heuristic-GPU"
            assert return_unit == "cycle"
            if self.K == k_dim:
                return per_batch_cycle_count
            if self.K == k_dim * batch_size:
                return merged_batch_cycle_count
            raise AssertionError(f"Unexpected K dimension: {self.K}")

    monkeypatch.setattr(
        flash_attention_module,
        "MatMul_Simulation",
        FakeMatMulSimulation,
    )
    monkeypatch.setattr(
        flash_attention_module,
        "ENABLE_FLASHATTN_CLI_HBF_SRAM_BUFFER",
        False,
    )

    hbm_only_instance = FlashAttn_BatchedMatMul_Simulation(
        dim=(batch_size, m_dim, k_dim, n_dim),
        weight_bits=16,
        matmul_type="SV",
    )
    hbm_only_cycle_count = hbm_only_instance.compile_and_simulate(
        pcb_module=hbm_only_device,
        compile_mode="heuristic-GPU",
        return_unit="cycle",
    )

    FlashAttn_BatchedMatMul_Simulation.clear_caches()
    csi_instance = FlashAttn_BatchedMatMul_Simulation(
        dim=(batch_size, m_dim, k_dim, n_dim),
        weight_bits=16,
        matmul_type="SV",
    )
    csi_cycle_count = csi_instance.compile_and_simulate(
        pcb_module=csi_device,
        compile_mode="heuristic-GPU",
        return_unit="cycle",
    )

    expected_cycle_count = merged_batch_cycle_count + math.ceil(
        extra_write_bytes / total_bandwidth_per_cycle
    )
    assert hbm_only_instance.best_mapping == "merged_batch"
    assert csi_instance.best_mapping == "merged_batch"
    assert hbm_only_cycle_count == expected_cycle_count
    assert csi_cycle_count == expected_cycle_count


def test_softmax_compile_and_simulate_supports_time_ns():
    instance = Softmax_Simulation(dim=(4, 6), weight_bits=16)

    cycles = instance.compile_and_simulate(
        pcb_module=A100_80GB_FP16,
        compile_mode=None,
    )
    time_ns = instance.compile_and_simulate(
        pcb_module=A100_80GB_FP16,
        compile_mode=None,
        return_unit="time_ns",
    )
    expected_time_ns = math.ceil(
        cycles * 1e9 / A100_80GB_FP16.compute_module.clock_freq
    )

    assert time_ns == expected_time_ns
    assert instance.best_cycle_count == cycles
    assert instance.best_time_ns == expected_time_ns


def test_flashattn_cli_qk_main_memory_read_formula_with_hbf_sram_buffer(monkeypatch):
    cli_device = build_device_for_hbm_hbf_architecture_or_raise(
        "H100_SXM",
        h100_cli_architecture(),
    )
    monkeypatch.setattr(
        flash_attention_module,
        "ENABLE_FLASHATTN_CLI_HBF_SRAM_BUFFER",
        True,
    )

    total_bandwidth_per_cycle = (
        cli_device.io_module.total_bandwidth / cli_device.compute_module.clock_freq
    )
    hbm_bandwidth_per_cycle = (
        cli_device.io_module.hbm_bandwidth / cli_device.compute_module.clock_freq
    )
    hbf_bandwidth_per_cycle = (
        cli_device.io_module.hbf_bandwidth / cli_device.compute_module.clock_freq
    )
    qk_mk_only_bytes = 4096
    qk_kn_only_bytes = 8192
    qk_mk_bytes_kn_slower = 4096
    qk_kn_bytes_kn_slower = 8192
    qk_mk_bytes_mk_slower = 8192
    qk_kn_bytes_mk_slower = 4096

    assert flash_attention_module._simulate_flashattn_qk_cli_read_cycle_count(
        cli_device,
        qk_mk_bytes=qk_mk_only_bytes,
        qk_kn_bytes=0,
    ) == math.ceil(qk_mk_only_bytes / total_bandwidth_per_cycle)
    assert flash_attention_module._simulate_flashattn_qk_cli_read_cycle_count(
        cli_device,
        qk_mk_bytes=0,
        qk_kn_bytes=qk_kn_only_bytes,
    ) == math.ceil(qk_kn_only_bytes / hbf_bandwidth_per_cycle)
    assert flash_attention_module._simulate_flashattn_qk_cli_read_cycle_count(
        cli_device,
        qk_mk_bytes=qk_mk_bytes_kn_slower,
        qk_kn_bytes=qk_kn_bytes_kn_slower,
    ) == math.ceil(qk_kn_bytes_kn_slower / hbf_bandwidth_per_cycle)
    assert (
        qk_kn_bytes_kn_slower / hbf_bandwidth_per_cycle
        >= qk_mk_bytes_kn_slower / hbm_bandwidth_per_cycle
    )
    assert flash_attention_module._simulate_flashattn_qk_cli_read_cycle_count(
        cli_device,
        qk_mk_bytes=qk_mk_bytes_mk_slower,
        qk_kn_bytes=qk_kn_bytes_mk_slower,
    ) == math.ceil(
        (qk_mk_bytes_mk_slower + qk_kn_bytes_mk_slower)
        / total_bandwidth_per_cycle
    )
    assert (
        qk_kn_bytes_mk_slower / hbf_bandwidth_per_cycle
        < qk_mk_bytes_mk_slower / hbm_bandwidth_per_cycle
    )


def test_flashattn_cli_qk_main_memory_read_formula_without_hbf_sram_buffer(monkeypatch):
    cli_device = build_device_for_hbm_hbf_architecture_or_raise(
        "H100_SXM",
        h100_cli_architecture(),
    )
    monkeypatch.setattr(
        flash_attention_module,
        "ENABLE_FLASHATTN_CLI_HBF_SRAM_BUFFER",
        False,
    )

    hbm_bandwidth_per_cycle = (
        cli_device.io_module.hbm_bandwidth / cli_device.compute_module.clock_freq
    )
    hbf_bandwidth_per_cycle = (
        cli_device.io_module.hbf_bandwidth / cli_device.compute_module.clock_freq
    )
    qk_mk_only_bytes = 4096
    qk_kn_only_bytes = 8192
    qk_mk_bytes_kn_slower = 4096
    qk_kn_bytes_kn_slower = 8192
    qk_mk_bytes_mk_slower = 8192
    qk_kn_bytes_mk_slower = 4096

    assert flash_attention_module._simulate_flashattn_qk_cli_read_cycle_count(
        cli_device,
        qk_mk_bytes=qk_mk_only_bytes,
        qk_kn_bytes=0,
    ) == math.ceil(qk_mk_only_bytes / hbm_bandwidth_per_cycle)
    assert flash_attention_module._simulate_flashattn_qk_cli_read_cycle_count(
        cli_device,
        qk_mk_bytes=0,
        qk_kn_bytes=qk_kn_only_bytes,
    ) == math.ceil(qk_kn_only_bytes / hbf_bandwidth_per_cycle)
    assert flash_attention_module._simulate_flashattn_qk_cli_read_cycle_count(
        cli_device,
        qk_mk_bytes=qk_mk_bytes_kn_slower,
        qk_kn_bytes=qk_kn_bytes_kn_slower,
    ) == math.ceil(
        max(
            qk_mk_bytes_kn_slower / hbm_bandwidth_per_cycle,
            qk_kn_bytes_kn_slower / hbf_bandwidth_per_cycle,
        )
    )
    assert flash_attention_module._simulate_flashattn_qk_cli_read_cycle_count(
        cli_device,
        qk_mk_bytes=qk_mk_bytes_mk_slower,
        qk_kn_bytes=qk_kn_bytes_mk_slower,
    ) == math.ceil(
        max(
            qk_mk_bytes_mk_slower / hbm_bandwidth_per_cycle,
            qk_kn_bytes_mk_slower / hbf_bandwidth_per_cycle,
        )
    )


def test_flashattn_cli_sv_main_memory_read_write_formula(monkeypatch):
    cli_device = build_device_for_hbm_hbf_architecture_or_raise(
        "H100_SXM",
        h100_cli_architecture(),
    )
    sv_nk_bytes = 8192
    sv_mk_bytes = 4096
    hbf_bandwidth_per_cycle = (
        cli_device.io_module.hbf_bandwidth / cli_device.compute_module.clock_freq
    )
    hbm_bandwidth_per_cycle = (
        cli_device.io_module.hbm_bandwidth / cli_device.compute_module.clock_freq
    )
    total_bandwidth_per_cycle = (
        cli_device.io_module.total_bandwidth / cli_device.compute_module.clock_freq
    )

    monkeypatch.setattr(
        flash_attention_module,
        "ENABLE_FLASHATTN_CLI_HBF_SRAM_BUFFER",
        True,
    )
    assert flash_attention_module._simulate_flashattn_sv_cli_read_cycle_count(
        cli_device,
        sv_nk_bytes=sv_nk_bytes,
    ) == math.ceil(sv_nk_bytes / hbf_bandwidth_per_cycle)
    assert flash_attention_module._simulate_flashattn_sv_cli_write_cycle_count(
        cli_device,
        sv_mk_bytes=sv_mk_bytes,
    ) == math.ceil(sv_mk_bytes / total_bandwidth_per_cycle)

    monkeypatch.setattr(
        flash_attention_module,
        "ENABLE_FLASHATTN_CLI_HBF_SRAM_BUFFER",
        False,
    )
    assert flash_attention_module._simulate_flashattn_sv_cli_read_cycle_count(
        cli_device,
        sv_nk_bytes=sv_nk_bytes,
    ) == math.ceil(sv_nk_bytes / hbf_bandwidth_per_cycle)
    assert flash_attention_module._simulate_flashattn_sv_cli_write_cycle_count(
        cli_device,
        sv_mk_bytes=sv_mk_bytes,
    ) == math.ceil(sv_mk_bytes / hbm_bandwidth_per_cycle)


@pytest.mark.parametrize(
    ("matmul_type", "dim", "architecture"),
    [
        ("QK", (1, 4096, 4096), {"mode": "cli", "hbm_stacks": 2, "hbf_stacks": 3}),
        ("QK", (4096, 4096, 1), {"mode": "cli", "hbm_stacks": 3, "hbf_stacks": 2}),
        ("SV", (1, 4096, 4096), {"mode": "cli", "hbm_stacks": 2, "hbf_stacks": 3}),
        ("SV", (4096, 4096, 1), {"mode": "cli", "hbm_stacks": 3, "hbf_stacks": 2}),
    ],
)
@pytest.mark.parametrize("enable_buffer", [True, False])
def test_flashattn_cli_gemv_compile_cycles_match_expected_formula(
    matmul_type: str,
    dim: tuple[int, int, int],
    architecture: dict[str, object],
    enable_buffer: bool,
    monkeypatch,
):
    device = build_device_for_hbm_hbf_architecture_or_raise("H100_SXM", architecture)
    M, K, N = dim
    word_size = 2
    total_flop_count = 2 * M * N * K
    compute_cycle_count = (
        total_flop_count / device.compute_module.get_total_vector_flops_per_cycle(16)
    )
    hbm_bandwidth_per_cycle = (
        device.io_module.hbm_bandwidth / device.compute_module.clock_freq
    )
    hbf_bandwidth_per_cycle = (
        device.io_module.hbf_bandwidth / device.compute_module.clock_freq
    )
    total_bandwidth_per_cycle = (
        device.io_module.total_bandwidth / device.compute_module.clock_freq
    )

    monkeypatch.setattr(
        flash_attention_module,
        "ENABLE_FLASHATTN_CLI_HBF_SRAM_BUFFER",
        enable_buffer,
    )

    if matmul_type == "QK":
        qk_mk_bytes = M * K * word_size
        qk_kn_bytes = K * N * word_size
        qk_mk_time_hbm = qk_mk_bytes / hbm_bandwidth_per_cycle
        qk_kn_time_hbf = qk_kn_bytes / hbf_bandwidth_per_cycle
        if enable_buffer:
            if qk_kn_time_hbf >= qk_mk_time_hbm:
                io_cycle_count = math.ceil(qk_kn_time_hbf)
            else:
                io_cycle_count = math.ceil(
                    (qk_mk_bytes + qk_kn_bytes) / total_bandwidth_per_cycle
                )
        else:
            io_cycle_count = math.ceil(max(qk_mk_time_hbm, qk_kn_time_hbf))
    elif matmul_type == "SV":
        io_cycle_count = math.ceil(K * N * word_size / hbf_bandwidth_per_cycle)
        io_cycle_count += math.ceil(
            M * N * word_size
            / (
                total_bandwidth_per_cycle
                if enable_buffer
                else hbm_bandwidth_per_cycle
            )
        )
    else:
        raise ValueError(f"Unsupported matmul_type: {matmul_type}")

    expected_cycle_count = math.ceil(max(compute_cycle_count, io_cycle_count))
    actual_cycle_count = FlashAttentionMatMul_Simulation(
        M,
        K,
        N,
        weight_bits=16,
        matmul_type=matmul_type,
    ).compile_and_simulate(
        pcb_module=device,
        compile_mode="heuristic-GPU",
        return_unit="cycle",
    )

    assert actual_cycle_count == expected_cycle_count


@pytest.mark.parametrize("matmul_type", ["QK", "SV"])
@pytest.mark.parametrize("dim", [(1, 4096, 4096), (4096, 4096, 1)])
def test_flashattn_hbm_only_and_csi_gemv_latencies_match_exactly(
    matmul_type: str,
    dim: tuple[int, int, int],
    monkeypatch,
):
    monkeypatch.setattr(
        flash_attention_module,
        "ENABLE_FLASHATTN_CLI_HBF_SRAM_BUFFER",
        True,
    )
    hbm_only_device = build_device_for_hbm_hbf_architecture_or_raise(
        "H100_SXM",
        hbm_only_architecture(),
    )
    csi_device = build_device_for_hbm_hbf_architecture_or_raise(
        "H100_SXM",
        {"mode": "csi"},
    )

    hbm_only_cycle_count = FlashAttentionMatMul_Simulation(
        *dim,
        weight_bits=16,
        matmul_type=matmul_type,
    ).compile_and_simulate(
        pcb_module=hbm_only_device,
        compile_mode="heuristic-GPU",
        return_unit="cycle",
    )
    csi_cycle_count = FlashAttentionMatMul_Simulation(
        *dim,
        weight_bits=16,
        matmul_type=matmul_type,
    ).compile_and_simulate(
        pcb_module=csi_device,
        compile_mode="heuristic-GPU",
        return_unit="cycle",
    )

    assert hbm_only_cycle_count == csi_cycle_count


@pytest.mark.parametrize("matmul_type", ["QK", "SV"])
@pytest.mark.parametrize("dim", [(1, 4096, 4096), (1, 8192, 4096)])
def test_flashattn_weight_heavy_cli_prefers_more_hbf_bandwidth_and_buffer(
    matmul_type: str,
    dim: tuple[int, int, int],
    monkeypatch,
):
    cli_more_hbf_device = build_device_for_hbm_hbf_architecture_or_raise(
        "H100_SXM",
        h100_cli_architecture(),
    )
    cli_more_hbm_device = build_device_for_hbm_hbf_architecture_or_raise(
        "H100_SXM",
        h100_cli_architecture_more_hbm(),
    )

    monkeypatch.setattr(
        flash_attention_module,
        "ENABLE_FLASHATTN_CLI_HBF_SRAM_BUFFER",
        True,
    )
    more_hbf_with_buffer = FlashAttentionMatMul_Simulation(
        *dim,
        weight_bits=16,
        matmul_type=matmul_type,
    ).compile_and_simulate(
        pcb_module=cli_more_hbf_device,
        compile_mode="heuristic-GPU",
        return_unit="cycle",
    )
    more_hbm_with_buffer = FlashAttentionMatMul_Simulation(
        *dim,
        weight_bits=16,
        matmul_type=matmul_type,
    ).compile_and_simulate(
        pcb_module=cli_more_hbm_device,
        compile_mode="heuristic-GPU",
        return_unit="cycle",
    )

    monkeypatch.setattr(
        flash_attention_module,
        "ENABLE_FLASHATTN_CLI_HBF_SRAM_BUFFER",
        False,
    )
    more_hbf_without_buffer = FlashAttentionMatMul_Simulation(
        *dim,
        weight_bits=16,
        matmul_type=matmul_type,
    ).compile_and_simulate(
        pcb_module=cli_more_hbf_device,
        compile_mode="heuristic-GPU",
        return_unit="cycle",
    )

    assert more_hbf_with_buffer <= more_hbf_without_buffer
    assert more_hbf_with_buffer < more_hbm_with_buffer


def test_flashattn_qk_buffer_matches_or_beats_buffer_off_across_cases(monkeypatch):
    cli_device = build_device_for_hbm_hbf_architecture_or_raise(
        "H100_SXM",
        h100_cli_architecture(),
    )

    monkeypatch.setattr(
        flash_attention_module,
        "ENABLE_FLASHATTN_CLI_HBF_SRAM_BUFFER",
        True,
    )
    equal_case_with_buffer = FlashAttentionMatMul_Simulation(
        1,
        4096,
        4096,
        weight_bits=16,
        matmul_type="QK",
    ).compile_and_simulate(
        pcb_module=cli_device,
        compile_mode="heuristic-GPU",
        return_unit="cycle",
    )
    faster_case_with_buffer = FlashAttentionMatMul_Simulation(
        4096,
        4096,
        1,
        weight_bits=16,
        matmul_type="QK",
    ).compile_and_simulate(
        pcb_module=cli_device,
        compile_mode="heuristic-GPU",
        return_unit="cycle",
    )

    monkeypatch.setattr(
        flash_attention_module,
        "ENABLE_FLASHATTN_CLI_HBF_SRAM_BUFFER",
        False,
    )
    equal_case_without_buffer = FlashAttentionMatMul_Simulation(
        1,
        4096,
        4096,
        weight_bits=16,
        matmul_type="QK",
    ).compile_and_simulate(
        pcb_module=cli_device,
        compile_mode="heuristic-GPU",
        return_unit="cycle",
    )
    faster_case_without_buffer = FlashAttentionMatMul_Simulation(
        4096,
        4096,
        1,
        weight_bits=16,
        matmul_type="QK",
    ).compile_and_simulate(
        pcb_module=cli_device,
        compile_mode="heuristic-GPU",
        return_unit="cycle",
    )

    assert equal_case_with_buffer == equal_case_without_buffer
    assert faster_case_with_buffer < faster_case_without_buffer


def test_softmax_cycles_are_architecture_invariant():
    dim = (128, 256)
    hbm_only_device = build_device_for_hbm_hbf_architecture_or_raise(
        "H100_SXM",
        hbm_only_architecture(),
    )
    csi_device = build_device_for_hbm_hbf_architecture_or_raise(
        "H100_SXM",
        {"mode": "csi"},
    )
    cli_device = build_device_for_hbm_hbf_architecture_or_raise(
        "H100_SXM",
        h100_cli_architecture(),
    )

    hbm_only_cycle_count = Softmax_Simulation(dim=dim, weight_bits=16).compile_and_simulate(
        pcb_module=hbm_only_device,
        compile_mode=None,
        return_unit="cycle",
    )
    csi_cycle_count = Softmax_Simulation(dim=dim, weight_bits=16).compile_and_simulate(
        pcb_module=csi_device,
        compile_mode=None,
        return_unit="cycle",
    )
    cli_cycle_count = Softmax_Simulation(dim=dim, weight_bits=16).compile_and_simulate(
        pcb_module=cli_device,
        compile_mode=None,
        return_unit="cycle",
    )

    assert hbm_only_cycle_count == csi_cycle_count
    assert hbm_only_cycle_count == cli_cycle_count


def test_flashattn_cli_io_formula_requires_valid_split_bandwidth(monkeypatch):
    cli_device = build_device_for_hbm_hbf_architecture_or_raise(
        "H100_SXM",
        h100_cli_architecture(),
    )
    broken_hbf_device = clone_device_with_io(
        cli_device,
        total_bandwidth=cli_device.io_module.total_bandwidth,
        hbm_bandwidth=cli_device.io_module.hbm_bandwidth,
        hbf_bandwidth=0.0,
    )
    broken_total_device = clone_device_with_io(
        cli_device,
        total_bandwidth=0.0,
        hbm_bandwidth=cli_device.io_module.hbm_bandwidth,
        hbf_bandwidth=cli_device.io_module.hbf_bandwidth,
    )
    broken_hbm_device = clone_device_with_io(
        cli_device,
        total_bandwidth=cli_device.io_module.total_bandwidth,
        hbm_bandwidth=0.0,
        hbf_bandwidth=cli_device.io_module.hbf_bandwidth,
    )

    with pytest.raises(ValueError):
        flash_attention_module._simulate_flashattn_qk_cli_read_cycle_count(
            broken_hbf_device,
            qk_mk_bytes=0,
            qk_kn_bytes=4096,
        )

    monkeypatch.setattr(
        flash_attention_module,
        "ENABLE_FLASHATTN_CLI_HBF_SRAM_BUFFER",
        True,
    )
    with pytest.raises(ValueError):
        flash_attention_module._simulate_flashattn_sv_cli_write_cycle_count(
            broken_total_device,
            sv_mk_bytes=4096,
        )

    monkeypatch.setattr(
        flash_attention_module,
        "ENABLE_FLASHATTN_CLI_HBF_SRAM_BUFFER",
        False,
    )
    with pytest.raises(ValueError):
        flash_attention_module._simulate_flashattn_qk_cli_read_cycle_count(
            broken_hbm_device,
            qk_mk_bytes=4096,
            qk_kn_bytes=0,
        )


def test_compute_engine_uses_cached_simulator_entrypoints(monkeypatch):
    SimSession.reset()
    SimSession.init()
    engine = xpu_module.ComputeEngine(
        make_config(),
        hbf_sram_intermediate_buffer=True,
        memory_architecture=hbm_only_architecture(),
    )
    matmul_calls: list[tuple[tuple[int, int, int], int]] = []
    flash_calls: list[tuple[tuple[int, int, int, int], int, str]] = []
    softmax_calls: list[tuple[tuple[int, int], int]] = []

    class FakeMatMulSimulation:
        def compile_and_simulate(self, pcb_module, compile_mode, return_unit="cycle"):
            assert return_unit == "time_ns"
            return 321

    class FakeFlashSimulation:
        def __init__(self, time_ns: int):
            self.time_ns = time_ns

        def compile_and_simulate(self, pcb_module, compile_mode, return_unit="cycle"):
            assert return_unit == "time_ns"
            return self.time_ns

    class FakeSoftmaxSimulation:
        def __init__(self, dim, weight_bits):
            softmax_calls.append((dim, weight_bits))

        def compile_and_simulate(self, pcb_module, compile_mode, return_unit="cycle"):
            assert return_unit == "time_ns"
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

    matmul_time_ns = engine.execute_macro_op(MatMulOp(dim=(9, 1, 11), weight_bits=16))
    flash_time_ns = engine.execute_macro_op(
        FlashAttnOp(
            qk_bmm_shape=(2, 4, 8, 6),
            sv_bmm_shape=(2, 4, 6, 3),
            softmax_shape=(4, 6),
            weight_bits=16,
        )
    )

    assert matmul_time_ns == 321
    assert flash_time_ns == 150
    assert matmul_calls == [((9, 1, 11), 16)]
    assert flash_calls == [
        ((2, 4, 8, 6), 16, "QK"),
        ((2, 4, 6, 3), 16, "SV"),
    ]
    assert softmax_calls == [((4, 6), 16)]

    SimSession.reset()


def test_compute_engine_uses_macro_op_weight_bits_for_all_latency_paths(monkeypatch):
    SimSession.reset()
    SimSession.init()

    engine = xpu_module.ComputeEngine(
        make_config(),
        hbf_sram_intermediate_buffer=True,
        memory_architecture=hbm_only_architecture(),
    )

    matmul_calls: list[tuple[tuple[int, int, int], int]] = []
    flash_calls: list[tuple[tuple[int, int, int, int], int, str]] = []
    softmax_calls: list[tuple[tuple[int, int], int]] = []

    class FakeMatMulSimulation:
        def compile_and_simulate(self, pcb_module, compile_mode, return_unit="cycle"):
            assert return_unit == "time_ns"
            return 101

    class FakeFlashSimulation:
        def __init__(self, time_ns: int):
            self.time_ns = time_ns

        def compile_and_simulate(self, pcb_module, compile_mode, return_unit="cycle"):
            assert return_unit == "time_ns"
            return self.time_ns

    class FakeSoftmaxSimulation:
        def __init__(self, dim, weight_bits):
            softmax_calls.append((dim, weight_bits))

        def compile_and_simulate(self, pcb_module, compile_mode, return_unit="cycle"):
            assert return_unit == "time_ns"
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

    matmul_time_ns = engine.execute_macro_op(MatMulOp(dim=(9, 1, 11), weight_bits=8))
    flash_time_ns = engine.execute_macro_op(
        FlashAttnOp(
            qk_bmm_shape=(2, 4, 8, 6),
            sv_bmm_shape=(2, 4, 6, 3),
            softmax_shape=(4, 6),
            weight_bits=8,
        )
    )

    assert matmul_time_ns == 101
    assert flash_time_ns == 151
    assert matmul_calls == [((9, 1, 11), 8)]
    assert flash_calls == [
        ((2, 4, 8, 6), 8, "QK"),
        ((2, 4, 6, 3), 8, "SV"),
    ]
    assert softmax_calls == [((4, 6), 8)]

    SimSession.reset()


def test_compute_engine_vector_time_ns_follow_macro_op_weight_bits():
    SimSession.reset()
    SimSession.init()

    engine = xpu_module.ComputeEngine(
        make_config(),
        hbf_sram_intermediate_buffer=True,
        memory_architecture=hbm_only_architecture(),
    )
    fp16_time_ns = engine.execute_macro_op(
        VectorOp(vector_op_type="rms_norm", vector_shape=[4096, 4096], weight_bits=16)
    )
    fp8_time_ns = engine.execute_macro_op(
        VectorOp(vector_op_type="rms_norm", vector_shape=[4096, 4096], weight_bits=8)
    )

    assert fp16_time_ns > fp8_time_ns

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

    engine = xpu_module.TransferEngine()
    allreduce_calls: list[tuple[int, int, int]] = []

    class FakeAllReduceSimulation:
        def __init__(self, num_gpus, data_size, weight_bits):
            allreduce_calls.append((num_gpus, data_size, weight_bits))

        def compile_and_simulate(
            self,
            pcb_module,
            interconnect_module,
            compile_mode,
            return_unit="cycle",
        ):
            assert return_unit == "time_ns"
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

    allreduce_time_ns = engine.execute_macro_op(
        AllReduceOp(num_ranks=4, data_size=128, weight_bits=16)
    )

    assert allreduce_time_ns == 66
    assert allreduce_calls == [(4, 128, 16)]

    SimSession.reset()


def test_transfer_engine_all2all_uses_estimate_path(monkeypatch):
    SimSession.reset()
    SimSession.init()

    engine = xpu_module.TransferEngine()
    all2all_calls: list[tuple[int, int, int]] = []

    class FakeAllToAllSimulation:
        def __init__(self, num_gpus, data_size, weight_bits):
            all2all_calls.append((num_gpus, data_size, weight_bits))

        def compile_and_simulate(
            self,
            pcb_module,
            interconnect_module,
            compile_mode,
            return_unit="cycle",
        ):
            assert return_unit == "time_ns"
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

    all2all_time_ns = engine.execute_macro_op(
        All2AllOp(num_gpus=4, data_size=128, weight_bits=16)
    )

    assert all2all_time_ns == 77
    assert all2all_calls == [(4, 128, 16)]

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

    execute_start_time_ns: list[int] = []

    def fake_execute_macro_op(macro_op):
        execute_start_time_ns.append(int(SimSession.sim_time.cycle))
        return 3.0

    monkeypatch.setattr(engine, "execute_macro_op", fake_execute_macro_op)

    DelayedFinish(dependency_slot)
    SimSession.scheduler.run()

    assert execute_start_time_ns == [6]
    assert transfer_slot.is_finished is True
    assert int(SimSession.sim_time.cycle) == 10

    SimSession.reset()


def test_compute_engine_waits_for_execute_time_ns(monkeypatch):
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

    dependency_slot = DepSlot(VectorOp(vector_op_type="rms_norm", vector_shape=[2, 16], weight_bits=16))
    compute_slot = DepSlot(MatMulOp(dim=(2, 16, 8), weight_bits=16))
    compute_slot.input_slots = [dependency_slot]

    engine = xpu_module.ComputeEngine(
        make_config(),
        hbf_sram_intermediate_buffer=True,
        memory_architecture=hbm_only_architecture(),
    )
    engine.load_command_queue([compute_slot])

    execute_start_time_ns: list[int] = []

    def fake_execute_macro_op(macro_op):
        execute_start_time_ns.append(int(SimSession.sim_time.cycle))
        return 7.0

    monkeypatch.setattr(engine, "execute_macro_op", fake_execute_macro_op)

    DelayedFinish(dependency_slot)
    SimSession.scheduler.run()

    assert execute_start_time_ns == [6]
    assert compute_slot.is_finished is True
    assert int(SimSession.sim_time.cycle) == 14

    SimSession.reset()


def test_cli_main_memory_read_formula_with_hbf_sram_buffer():
    cli_device = build_device_for_hbm_hbf_architecture_or_raise(
        "H100_SXM",
        h100_cli_architecture(),
    )
    hbm_bandwidth_per_cycle = (
        cli_device.io_module.hbm_bandwidth / cli_device.compute_module.clock_freq
    )
    hbf_bandwidth_per_cycle = (
        cli_device.io_module.hbf_bandwidth / cli_device.compute_module.clock_freq
    )
    total_bandwidth_per_cycle = (
        cli_device.io_module.total_bandwidth / cli_device.compute_module.clock_freq
    )

    mk_bytes = 4096
    kn_bytes = 8192
    mn_read_bytes = 2048
    expected_read_cycles = max(
        math.ceil(mk_bytes / hbm_bandwidth_per_cycle),
        math.ceil(kn_bytes / hbf_bandwidth_per_cycle),
    ) + math.ceil(mn_read_bytes / total_bandwidth_per_cycle)
    expected_write_cycles = math.ceil(mn_read_bytes / total_bandwidth_per_cycle)

    assert matmul_module.ENABLE_CLI_HBF_SRAM_BUFFER is True
    assert matmul_module._simulate_cli_main_memory_read_cycle_count(
        cli_device,
        mk_bytes=mk_bytes,
        kn_bytes=kn_bytes,
        mn_read_bytes=mn_read_bytes,
    ) == expected_read_cycles
    assert matmul_module._simulate_cli_main_memory_write_cycle_count(
        cli_device,
        mn_write_bytes=mn_read_bytes,
    ) == expected_write_cycles


def test_cli_main_memory_read_formula_without_hbf_sram_buffer(monkeypatch):
    cli_device = build_device_for_hbm_hbf_architecture_or_raise(
        "H100_SXM",
        h100_cli_architecture(),
    )
    monkeypatch.setattr(matmul_module, "ENABLE_CLI_HBF_SRAM_BUFFER", False)

    hbm_bandwidth_per_cycle = (
        cli_device.io_module.hbm_bandwidth / cli_device.compute_module.clock_freq
    )
    hbf_bandwidth_per_cycle = (
        cli_device.io_module.hbf_bandwidth / cli_device.compute_module.clock_freq
    )

    mk_bytes = 4096
    kn_bytes = 8192
    mn_read_bytes = 2048
    expected_read_cycles = max(
        math.ceil((mk_bytes + mn_read_bytes) / hbm_bandwidth_per_cycle),
        math.ceil(kn_bytes / hbf_bandwidth_per_cycle),
    )
    expected_write_cycles = math.ceil(mn_read_bytes / hbm_bandwidth_per_cycle)

    assert matmul_module._simulate_cli_main_memory_read_cycle_count(
        cli_device,
        mk_bytes=mk_bytes,
        kn_bytes=kn_bytes,
        mn_read_bytes=mn_read_bytes,
    ) == expected_read_cycles
    assert matmul_module._simulate_cli_main_memory_write_cycle_count(
        cli_device,
        mn_write_bytes=mn_read_bytes,
    ) == expected_write_cycles


def test_hbm_only_and_csi_paths_do_not_use_cli_io_formula(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("CLI IO helper should not be called")

    monkeypatch.setattr(
        matmul_module,
        "_simulate_cli_main_memory_read_cycle_count",
        fail_if_called,
    )
    monkeypatch.setattr(
        matmul_module,
        "_simulate_cli_main_memory_write_cycle_count",
        fail_if_called,
    )

    for device in (
        build_device_for_hbm_hbf_architecture_or_raise(
            "H100_SXM",
            hbm_only_architecture(),
        ),
        build_device_for_hbm_hbf_architecture_or_raise(
            "H100_SXM",
            {"mode": "csi"},
        ),
    ):
        bandwidth_per_cycle = (
            device.io_module.total_bandwidth / device.compute_module.clock_freq
        )
        expected_read_cycles = math.ceil(
            (4096 + 8192 + 2048) / bandwidth_per_cycle
        )
        expected_write_cycles = math.ceil(2048 / bandwidth_per_cycle)

        assert matmul_module._simulate_main_memory_read_cycle_count(
            device,
            mk_bytes=4096,
            kn_bytes=8192,
            mn_read_bytes=2048,
        ) == expected_read_cycles
        assert matmul_module._simulate_main_memory_write_cycle_count(
            device,
            mn_write_bytes=2048,
        ) == expected_write_cycles


def test_cli_main_memory_formula_raises_on_missing_split_bandwidth():
    cli_device = build_device_for_hbm_hbf_architecture_or_raise(
        "H100_SXM",
        h100_cli_architecture(),
    )
    broken_device = Device(
        compute_module=cli_device.compute_module,
        io_module=IOModule(
            bandwidth=cli_device.io_module.total_bandwidth,
            hbm_bandwidth=0.0,
            hbf_bandwidth=cli_device.io_module.hbf_bandwidth,
        ),
        memory_capacity_bytes=cli_device.total_memory_capacity_bytes,
        hbm_memory_capacity_bytes=cli_device.hbm_memory_capacity_bytes,
        hbf_memory_capacity_bytes=cli_device.hbf_memory_capacity_bytes,
        memory_architecture_mode="cli",
        hbm_stack_count=cli_device.hbm_stack_count,
        hbf_stack_count=cli_device.hbf_stack_count,
    )

    with pytest.raises(ValueError):
        matmul_module._simulate_cli_main_memory_read_cycle_count(
            broken_device,
            mk_bytes=4096,
            kn_bytes=8192,
            mn_read_bytes=2048,
        )


def test_cli_matmul_latency_changes_when_hbf_sram_buffer_toggles(monkeypatch):
    cli_device = build_device_for_hbm_hbf_architecture_or_raise(
        "H100_SXM",
        h100_cli_architecture(),
    )
    simulation = MatMul_Simulation.get_instance(dim=(1, 4096, 4096), weight_bits=16)

    MatMul_Simulation.clear_caches()
    monkeypatch.setattr(matmul_module, "ENABLE_CLI_HBF_SRAM_BUFFER", True)
    latency_with_buffer = simulation.compile_and_simulate(
        pcb_module=cli_device,
        compile_mode="heuristic-GPU",
        return_unit="time_ns",
    )

    MatMul_Simulation.clear_caches()
    monkeypatch.setattr(matmul_module, "ENABLE_CLI_HBF_SRAM_BUFFER", False)
    latency_without_buffer = simulation.compile_and_simulate(
        pcb_module=cli_device,
        compile_mode="heuristic-GPU",
        return_unit="time_ns",
    )

    assert latency_with_buffer != latency_without_buffer
    assert latency_with_buffer < latency_without_buffer


@pytest.mark.parametrize(
    ("dim", "architecture"),
    [
        ((1, 4096, 4096), {"mode": "cli", "hbm_stacks": 2, "hbf_stacks": 3}),
        ((1, 8192, 4096), {"mode": "cli", "hbm_stacks": 2, "hbf_stacks": 3}),
        ((4096, 4096, 1), {"mode": "cli", "hbm_stacks": 3, "hbf_stacks": 2}),
    ],
)
@pytest.mark.parametrize("enable_buffer", [True, False])
def test_cli_gemv_compile_cycles_match_expected_formula(
    dim: tuple[int, int, int],
    architecture: dict[str, object],
    enable_buffer: bool,
    monkeypatch,
):
    device = build_device_for_hbm_hbf_architecture_or_raise("H100_SXM", architecture)
    M, K, N = dim
    word_size = 2
    total_flop_count = 2 * M * N * K
    compute_cycle_count = (
        total_flop_count / device.compute_module.get_total_vector_flops_per_cycle(16)
    )
    hbm_bandwidth_per_cycle = (
        device.io_module.hbm_bandwidth / device.compute_module.clock_freq
    )
    hbf_bandwidth_per_cycle = (
        device.io_module.hbf_bandwidth / device.compute_module.clock_freq
    )
    total_bandwidth_per_cycle = (
        device.io_module.total_bandwidth / device.compute_module.clock_freq
    )

    monkeypatch.setattr(
        matmul_module,
        "ENABLE_CLI_HBF_SRAM_BUFFER",
        enable_buffer,
    )

    read_cycle_count = max(
        math.ceil(M * K * word_size / hbm_bandwidth_per_cycle),
        math.ceil(K * N * word_size / hbf_bandwidth_per_cycle),
    )
    if enable_buffer:
        write_cycle_count = math.ceil(M * N * word_size / total_bandwidth_per_cycle)
    else:
        write_cycle_count = math.ceil(M * N * word_size / hbm_bandwidth_per_cycle)
    expected_cycle_count = math.ceil(
        max(compute_cycle_count, read_cycle_count + write_cycle_count)
    )

    MatMul_Simulation.clear_caches()
    actual_cycle_count = MatMul_Simulation.get_instance(
        dim=dim,
        weight_bits=16,
    ).compile_and_simulate(
        pcb_module=device,
        compile_mode="heuristic-GPU",
        return_unit="cycle",
    )

    assert actual_cycle_count == expected_cycle_count


@pytest.mark.parametrize("dim", [(1, 4096, 4096), (1, 8192, 4096), (4096, 4096, 1)])
def test_hbm_only_and_csi_gemv_latencies_match_exactly(dim: tuple[int, int, int], monkeypatch):
    monkeypatch.setattr(matmul_module, "ENABLE_CLI_HBF_SRAM_BUFFER", True)
    hbm_only_device = build_device_for_hbm_hbf_architecture_or_raise(
        "H100_SXM",
        hbm_only_architecture(),
    )
    csi_device = build_device_for_hbm_hbf_architecture_or_raise(
        "H100_SXM",
        {"mode": "csi"},
    )

    MatMul_Simulation.clear_caches()
    hbm_only_cycle_count = MatMul_Simulation.get_instance(
        dim=dim,
        weight_bits=16,
    ).compile_and_simulate(
        pcb_module=hbm_only_device,
        compile_mode="heuristic-GPU",
        return_unit="cycle",
    )
    MatMul_Simulation.clear_caches()
    csi_cycle_count = MatMul_Simulation.get_instance(
        dim=dim,
        weight_bits=16,
    ).compile_and_simulate(
        pcb_module=csi_device,
        compile_mode="heuristic-GPU",
        return_unit="cycle",
    )

    assert hbm_only_cycle_count == csi_cycle_count


@pytest.mark.parametrize("dim", [(1, 4096, 4096), (1, 8192, 4096)])
def test_weight_heavy_gemv_prefers_more_hbf_bandwidth_and_hbf_buffer(
    dim: tuple[int, int, int],
    monkeypatch,
):
    cli_more_hbf_device = build_device_for_hbm_hbf_architecture_or_raise(
        "H100_SXM",
        h100_cli_architecture(),
    )
    cli_more_hbm_device = build_device_for_hbm_hbf_architecture_or_raise(
        "H100_SXM",
        h100_cli_architecture_more_hbm(),
    )

    monkeypatch.setattr(matmul_module, "ENABLE_CLI_HBF_SRAM_BUFFER", True)
    MatMul_Simulation.clear_caches()
    more_hbf_with_buffer = MatMul_Simulation.get_instance(
        dim=dim,
        weight_bits=16,
    ).compile_and_simulate(
        pcb_module=cli_more_hbf_device,
        compile_mode="heuristic-GPU",
        return_unit="cycle",
    )
    MatMul_Simulation.clear_caches()
    more_hbm_with_buffer = MatMul_Simulation.get_instance(
        dim=dim,
        weight_bits=16,
    ).compile_and_simulate(
        pcb_module=cli_more_hbm_device,
        compile_mode="heuristic-GPU",
        return_unit="cycle",
    )

    monkeypatch.setattr(matmul_module, "ENABLE_CLI_HBF_SRAM_BUFFER", False)
    MatMul_Simulation.clear_caches()
    more_hbf_without_buffer = MatMul_Simulation.get_instance(
        dim=dim,
        weight_bits=16,
    ).compile_and_simulate(
        pcb_module=cli_more_hbf_device,
        compile_mode="heuristic-GPU",
        return_unit="cycle",
    )

    assert more_hbf_with_buffer < more_hbf_without_buffer
    assert more_hbf_with_buffer < more_hbm_with_buffer


def test_compute_engine_builds_cli_device_and_forwards_it_to_matmul(monkeypatch):
    SimSession.reset()
    SimSession.init()

    engine = xpu_module.ComputeEngine(
        make_config(),
        hbf_sram_intermediate_buffer=True,
        memory_architecture=h100_cli_architecture(),
        device_name="H100_SXM",
        compile_mode="heuristic-GPU",
    )

    class FakeMatMulSimulation:
        def compile_and_simulate(self, pcb_module, compile_mode, return_unit="cycle"):
            assert compile_mode == "heuristic-GPU"
            assert return_unit == "time_ns"
            assert pcb_module.memory_architecture_mode == "cli"
            assert pcb_module.io_module.hbm_bandwidth > 0
            assert pcb_module.io_module.hbf_bandwidth > 0
            assert (
                pcb_module.io_module.total_bandwidth
                == pcb_module.io_module.hbm_bandwidth
                + pcb_module.io_module.hbf_bandwidth
            )
            return 123

    monkeypatch.setattr(
        xpu_module.MatMul_Simulation,
        "get_instance",
        staticmethod(lambda dim, weight_bits=16: FakeMatMulSimulation()),
    )

    assert engine.device.memory_architecture_mode == "cli"
    assert engine.execute_macro_op(MatMulOp(dim=(2, 16, 8), weight_bits=16)) == 123

    SimSession.reset()


def test_compute_engine_syncs_hbf_sram_intermediate_buffer_to_simulator_modules(
    monkeypatch,
):
    SimSession.reset()
    SimSession.init()

    monkeypatch.setattr(matmul_module, "ENABLE_CLI_HBF_SRAM_BUFFER", True)
    monkeypatch.setattr(
        flash_attention_module,
        "ENABLE_FLASHATTN_CLI_HBF_SRAM_BUFFER",
        True,
    )

    engine = xpu_module.ComputeEngine(
        make_config(),
        hbf_sram_intermediate_buffer=False,
        memory_architecture=hbm_only_architecture(),
    )

    assert engine.hbf_sram_intermediate_buffer is False
    assert matmul_module.ENABLE_CLI_HBF_SRAM_BUFFER is False
    assert flash_attention_module.ENABLE_FLASHATTN_CLI_HBF_SRAM_BUFFER is False

    SimSession.reset()


def test_compute_engine_requires_hbf_sram_intermediate_buffer():
    with pytest.raises(TypeError):
        xpu_module.ComputeEngine(
            make_config(),
            memory_architecture=hbm_only_architecture(),
        )
