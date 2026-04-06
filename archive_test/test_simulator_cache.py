import math
from types import SimpleNamespace

import pytest
from Desim import SimSession

import nandmachine.simulator.hardware.xpu as xpu_module
import nandmachine.simulator.software.flash_attention as flash_attention_module
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
from nandmachine.config.hardware_config import A100_80GB_FP16, get_device_or_raise
from nandmachine.simulator.hardware.utils import DepSlot
from nandmachine.simulator.software.flash_attention import (
    FlashAttn_BatchedMatMul_Simulation,
    FlashMLA_BatchedMatMul_Simulation,
    Softmax_Simulation,
)
from nandmachine.simulator.software.matmul import MatMul_Simulation


def make_nand_config(
    *,
    num_channels: int = 1,
    num_plane: int = 2,
    tRead: float = 4.0,
    page_size: int = 16,
) -> NandConfig:
    return NandConfig(
        num_channels=num_channels,
        num_plane=num_plane,
        num_block=8,
        num_pages=32,
        tRead=tRead,
        tWrite=8.0,
        tErase=16.0,
        page_size=page_size,
        sram_threshold=64,
    )


def hbm_bandwidth_bytes_per_sec(device_name: str = "A100_80GB") -> float:
    return get_device_or_raise(device_name).io_module.bandwidth


def _expected_hbf_bandwidth_bytes_per_sec(nand_config: NandConfig) -> float:
    return (
        nand_config.num_channels
        * nand_config.num_plane
        * (nand_config.page_size_bytes / nand_config.tRead)
        * 1e9
    )


def _reset_and_init_sim_session() -> None:
    SimSession.reset()
    SimSession.init()


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
    nand_config = make_nand_config()
    hbm_bw = hbm_bandwidth_bytes_per_sec()

    first_cycles = instance.compile_and_simulate(
        pcb_module=A100_80GB_FP16,
        nand_config=nand_config,
        hbm_bandwidth_bytes_per_sec=hbm_bw,
        compile_mode="heuristic-GPU",
    )
    second_cycles = instance.compile_and_simulate(
        pcb_module=A100_80GB_FP16,
        nand_config=nand_config,
        hbm_bandwidth_bytes_per_sec=hbm_bw,
        compile_mode="heuristic-GPU",
    )
    cache_info = MatMul_Simulation.compile_and_simulate.cache_info()

    assert first_cycles == second_cycles
    assert instance.best_cycle_count == first_cycles
    assert cache_info.misses == 1
    assert cache_info.hits == 1


def test_matmul_compile_cache_key_includes_nand_config():
    MatMul_Simulation.clear_caches()
    instance = MatMul_Simulation.get_instance(dim=(9, 1, 11), weight_bits=16)
    hbm_bw = hbm_bandwidth_bytes_per_sec()

    instance.compile_and_simulate(
        pcb_module=A100_80GB_FP16,
        nand_config=make_nand_config(num_channels=1),
        hbm_bandwidth_bytes_per_sec=hbm_bw,
        compile_mode="heuristic-GPU",
    )
    instance.compile_and_simulate(
        pcb_module=A100_80GB_FP16,
        nand_config=make_nand_config(num_channels=2),
        hbm_bandwidth_bytes_per_sec=hbm_bw,
        compile_mode="heuristic-GPU",
    )
    cache_info = MatMul_Simulation.compile_and_simulate.cache_info()

    assert cache_info.misses == 2
    assert cache_info.hits == 0


def test_matmul_compile_cache_key_includes_hbm_bandwidth():
    MatMul_Simulation.clear_caches()
    instance = MatMul_Simulation.get_instance(dim=(9, 1, 11), weight_bits=16)
    nand_config = make_nand_config()

    instance.compile_and_simulate(
        pcb_module=A100_80GB_FP16,
        nand_config=nand_config,
        hbm_bandwidth_bytes_per_sec=1.0e12,
        compile_mode="heuristic-GPU",
    )
    instance.compile_and_simulate(
        pcb_module=A100_80GB_FP16,
        nand_config=nand_config,
        hbm_bandwidth_bytes_per_sec=2.0e12,
        compile_mode="heuristic-GPU",
    )
    cache_info = MatMul_Simulation.compile_and_simulate.cache_info()

    assert cache_info.misses == 2
    assert cache_info.hits == 0


def test_matmul_compile_and_simulate_supports_time_ns_and_rehydrates_fields():
    MatMul_Simulation.clear_caches()
    instance = MatMul_Simulation.get_instance(dim=(9, 1, 11), weight_bits=16)
    nand_config = make_nand_config()
    hbm_bw = hbm_bandwidth_bytes_per_sec()

    cycles = instance.compile_and_simulate(
        pcb_module=A100_80GB_FP16,
        nand_config=nand_config,
        hbm_bandwidth_bytes_per_sec=hbm_bw,
        compile_mode="heuristic-GPU",
    )
    time_ns = instance.compile_and_simulate(
        pcb_module=A100_80GB_FP16,
        nand_config=nand_config,
        hbm_bandwidth_bytes_per_sec=hbm_bw,
        compile_mode="heuristic-GPU",
        return_unit="time_ns",
    )
    expected_time_ns = math.ceil(
        cycles * 1e9 / A100_80GB_FP16.compute_module.clock_freq
    )

    assert time_ns == expected_time_ns
    assert instance.best_cycle_count == cycles
    assert instance.best_time_ns == expected_time_ns


def test_matmul_bandwidth_formula_uses_nand_config_and_hbm_bandwidth():
    nand_config = make_nand_config(num_channels=2, num_plane=4, tRead=2.0, page_size=8)
    hbm_bw = 1.2e12
    bandwidth_config_key = matmul_module._build_bandwidth_config_key_or_raise(
        nand_config,
        hbm_bw,
    )

    expected_hbf = _expected_hbf_bandwidth_bytes_per_sec(nand_config)
    total_per_cycle = matmul_module._get_main_memory_bandwidth_per_cycle_or_raise(
        A100_80GB_FP16,
        bandwidth_config_key,
        "total",
    )

    assert matmul_module._get_hbf_bandwidth_bytes_per_sec_or_raise(
        bandwidth_config_key
    ) == pytest.approx(expected_hbf)
    assert total_per_cycle == pytest.approx(
        (hbm_bw + expected_hbf) / A100_80GB_FP16.compute_module.clock_freq
    )


def test_matmul_split_main_memory_formula_with_buffer():
    nand_config = make_nand_config(num_channels=2, num_plane=2, tRead=4.0, page_size=16)
    hbm_bw = 1.0e12
    bandwidth_config_key = matmul_module._build_bandwidth_config_key_or_raise(
        nand_config,
        hbm_bw,
    )
    hbf_bw = _expected_hbf_bandwidth_bytes_per_sec(nand_config)
    hbm_per_cycle = hbm_bw / A100_80GB_FP16.compute_module.clock_freq
    hbf_per_cycle = hbf_bw / A100_80GB_FP16.compute_module.clock_freq
    total_per_cycle = (hbm_bw + hbf_bw) / A100_80GB_FP16.compute_module.clock_freq

    mk_bytes = 4096
    kn_bytes = 8192
    mn_bytes = 2048

    assert matmul_module.ENABLE_CLI_HBF_SRAM_BUFFER is True
    assert matmul_module._simulate_cli_main_memory_read_cycle_count(
        A100_80GB_FP16,
        bandwidth_config_key,
        mk_bytes=mk_bytes,
        kn_bytes=kn_bytes,
        mn_read_bytes=mn_bytes,
    ) == max(
        math.ceil(mk_bytes / hbm_per_cycle),
        math.ceil(kn_bytes / hbf_per_cycle),
    ) + math.ceil(mn_bytes / total_per_cycle)
    assert matmul_module._simulate_cli_main_memory_write_cycle_count(
        A100_80GB_FP16,
        bandwidth_config_key,
        mn_write_bytes=mn_bytes,
    ) == math.ceil(mn_bytes / total_per_cycle)


def test_matmul_split_main_memory_formula_without_buffer(monkeypatch):
    nand_config = make_nand_config(num_channels=2, num_plane=2, tRead=4.0, page_size=16)
    hbm_bw = 1.0e12
    bandwidth_config_key = matmul_module._build_bandwidth_config_key_or_raise(
        nand_config,
        hbm_bw,
    )
    hbf_bw = _expected_hbf_bandwidth_bytes_per_sec(nand_config)
    hbm_per_cycle = hbm_bw / A100_80GB_FP16.compute_module.clock_freq
    hbf_per_cycle = hbf_bw / A100_80GB_FP16.compute_module.clock_freq

    mk_bytes = 4096
    kn_bytes = 8192
    mn_bytes = 2048

    monkeypatch.setattr(matmul_module, "ENABLE_CLI_HBF_SRAM_BUFFER", False)

    assert matmul_module._simulate_cli_main_memory_read_cycle_count(
        A100_80GB_FP16,
        bandwidth_config_key,
        mk_bytes=mk_bytes,
        kn_bytes=kn_bytes,
        mn_read_bytes=mn_bytes,
    ) == max(
        math.ceil((mk_bytes + mn_bytes) / hbm_per_cycle),
        math.ceil(kn_bytes / hbf_per_cycle),
    )
    assert matmul_module._simulate_cli_main_memory_write_cycle_count(
        A100_80GB_FP16,
        bandwidth_config_key,
        mn_write_bytes=mn_bytes,
    ) == math.ceil(mn_bytes / hbm_per_cycle)


def test_matmul_compile_and_simulate_rejects_invalid_bandwidth_inputs():
    instance = MatMul_Simulation.get_instance(dim=(9, 1, 11), weight_bits=16)

    with pytest.raises(ValueError):
        instance.compile_and_simulate(
            pcb_module=A100_80GB_FP16,
            nand_config=make_nand_config(tRead=0.0),
            hbm_bandwidth_bytes_per_sec=1.0e12,
            compile_mode="heuristic-GPU",
        )

    with pytest.raises(ValueError):
        instance.compile_and_simulate(
            pcb_module=A100_80GB_FP16,
            nand_config=make_nand_config(),
            hbm_bandwidth_bytes_per_sec=0.0,
            compile_mode="heuristic-GPU",
        )


def test_flashattn_compile_and_simulate_uses_cached_result():
    FlashAttn_BatchedMatMul_Simulation.clear_caches()
    instance = FlashAttn_BatchedMatMul_Simulation.get_instance(
        dim=(2, 1, 8, 4),
        weight_bits=16,
        matmul_type="QK",
    )
    nand_config = make_nand_config()
    hbm_bw = hbm_bandwidth_bytes_per_sec()

    first_cycles = instance.compile_and_simulate(
        pcb_module=A100_80GB_FP16,
        nand_config=nand_config,
        hbm_bandwidth_bytes_per_sec=hbm_bw,
        compile_mode="heuristic-GPU",
    )
    second_cycles = instance.compile_and_simulate(
        pcb_module=A100_80GB_FP16,
        nand_config=nand_config,
        hbm_bandwidth_bytes_per_sec=hbm_bw,
        compile_mode="heuristic-GPU",
    )
    cache_info = FlashAttn_BatchedMatMul_Simulation.compile_and_simulate.cache_info()

    assert first_cycles == second_cycles
    assert instance.best_cycle_count == first_cycles
    assert cache_info.misses == 1
    assert cache_info.hits == 1


def test_flashattn_compile_cache_key_includes_bandwidth_inputs():
    FlashAttn_BatchedMatMul_Simulation.clear_caches()
    instance = FlashAttn_BatchedMatMul_Simulation.get_instance(
        dim=(2, 1, 8, 4),
        weight_bits=16,
        matmul_type="QK",
    )

    instance.compile_and_simulate(
        pcb_module=A100_80GB_FP16,
        nand_config=make_nand_config(num_channels=1),
        hbm_bandwidth_bytes_per_sec=1.0e12,
        compile_mode="heuristic-GPU",
    )
    instance.compile_and_simulate(
        pcb_module=A100_80GB_FP16,
        nand_config=make_nand_config(num_channels=2),
        hbm_bandwidth_bytes_per_sec=1.0e12,
        compile_mode="heuristic-GPU",
    )
    instance.compile_and_simulate(
        pcb_module=A100_80GB_FP16,
        nand_config=make_nand_config(num_channels=2),
        hbm_bandwidth_bytes_per_sec=2.0e12,
        compile_mode="heuristic-GPU",
    )
    cache_info = FlashAttn_BatchedMatMul_Simulation.compile_and_simulate.cache_info()

    assert cache_info.misses == 3
    assert cache_info.hits == 0


def test_flashattn_compile_and_simulate_supports_time_ns_and_rehydrates_fields():
    FlashAttn_BatchedMatMul_Simulation.clear_caches()
    instance = FlashAttn_BatchedMatMul_Simulation.get_instance(
        dim=(2, 1, 8, 4),
        weight_bits=16,
        matmul_type="QK",
    )
    nand_config = make_nand_config()
    hbm_bw = hbm_bandwidth_bytes_per_sec()

    cycles = instance.compile_and_simulate(
        pcb_module=A100_80GB_FP16,
        nand_config=nand_config,
        hbm_bandwidth_bytes_per_sec=hbm_bw,
        compile_mode="heuristic-GPU",
    )
    time_ns = instance.compile_and_simulate(
        pcb_module=A100_80GB_FP16,
        nand_config=nand_config,
        hbm_bandwidth_bytes_per_sec=hbm_bw,
        compile_mode="heuristic-GPU",
        return_unit="time_ns",
    )
    expected_time_ns = math.ceil(
        cycles * 1e9 / A100_80GB_FP16.compute_module.clock_freq
    )

    assert time_ns == expected_time_ns
    assert instance.best_cycle_count == cycles
    assert instance.best_time_ns == expected_time_ns


def test_flashattn_qk_and_sv_split_bandwidth_formulas(monkeypatch):
    nand_config = make_nand_config(num_channels=2, num_plane=2, tRead=4.0, page_size=16)
    hbm_bw = 1.0e12
    bandwidth_config_key = flash_attention_module._build_bandwidth_config_key_or_raise(
        nand_config,
        hbm_bw,
    )
    hbf_bw = _expected_hbf_bandwidth_bytes_per_sec(nand_config)
    hbm_per_cycle = hbm_bw / A100_80GB_FP16.compute_module.clock_freq
    hbf_per_cycle = hbf_bw / A100_80GB_FP16.compute_module.clock_freq
    total_per_cycle = (hbm_bw + hbf_bw) / A100_80GB_FP16.compute_module.clock_freq

    qk_mk_bytes = 4096
    qk_kn_bytes = 8192
    sv_nk_bytes = 6144
    sv_mk_bytes = 2048

    assert flash_attention_module._simulate_flashattn_qk_cli_read_cycle_count(
        A100_80GB_FP16,
        bandwidth_config_key,
        qk_mk_bytes=qk_mk_bytes,
        qk_kn_bytes=qk_kn_bytes,
    ) == math.ceil((qk_mk_bytes + qk_kn_bytes) / total_per_cycle)
    assert flash_attention_module._simulate_flashattn_sv_cli_read_cycle_count(
        A100_80GB_FP16,
        bandwidth_config_key,
        sv_nk_bytes=sv_nk_bytes,
    ) == math.ceil(sv_nk_bytes / hbf_per_cycle)
    assert flash_attention_module._simulate_flashattn_sv_cli_write_cycle_count(
        A100_80GB_FP16,
        bandwidth_config_key,
        sv_mk_bytes=sv_mk_bytes,
    ) == math.ceil(sv_mk_bytes / total_per_cycle)

    monkeypatch.setattr(
        flash_attention_module,
        "ENABLE_FLASHATTN_CLI_HBF_SRAM_BUFFER",
        False,
    )
    assert flash_attention_module._simulate_flashattn_qk_cli_read_cycle_count(
        A100_80GB_FP16,
        bandwidth_config_key,
        qk_mk_bytes=qk_mk_bytes,
        qk_kn_bytes=qk_kn_bytes,
    ) == math.ceil(max(qk_mk_bytes / hbm_per_cycle, qk_kn_bytes / hbf_per_cycle))
    assert flash_attention_module._simulate_flashattn_sv_cli_write_cycle_count(
        A100_80GB_FP16,
        bandwidth_config_key,
        sv_mk_bytes=sv_mk_bytes,
    ) == math.ceil(sv_mk_bytes / hbm_per_cycle)


def test_flashattn_mla_main_memory_read_and_write_match_expected_semantics():
    nand_config = make_nand_config(num_channels=2, num_plane=2, tRead=4.0, page_size=16)
    hbm_bw = 1.0e12
    bandwidth_config_key = flash_attention_module._build_bandwidth_config_key_or_raise(
        nand_config,
        hbm_bw,
    )

    assert flash_attention_module._simulate_flashattn_main_memory_read_cycle_count(
        A100_80GB_FP16,
        bandwidth_config_key,
        matmul_type="MLA_QK",
        qk_mk_or_sv_mk_bytes=4096,
        qk_kn_or_sv_nk_bytes=8192,
    ) == flash_attention_module._simulate_flashattn_qk_cli_read_cycle_count(
        A100_80GB_FP16,
        bandwidth_config_key,
        qk_mk_bytes=0,
        qk_kn_bytes=8192,
    )
    assert flash_attention_module._simulate_flashattn_main_memory_read_cycle_count(
        A100_80GB_FP16,
        bandwidth_config_key,
        matmul_type="MLA_SV",
        qk_mk_or_sv_mk_bytes=0,
        qk_kn_or_sv_nk_bytes=6144,
    ) == 0
    assert flash_attention_module._simulate_flashattn_main_memory_write_cycle_count(
        A100_80GB_FP16,
        bandwidth_config_key,
        matmul_type="MLA_QK",
        sv_mk_bytes=2048,
    ) == 0
    assert flash_attention_module._simulate_flashattn_main_memory_write_cycle_count(
        A100_80GB_FP16,
        bandwidth_config_key,
        matmul_type="MLA_SV",
        sv_mk_bytes=2048,
    ) == flash_attention_module._simulate_flashattn_sv_cli_write_cycle_count(
        A100_80GB_FP16,
        bandwidth_config_key,
        sv_mk_bytes=2048,
    )


def test_flashattn_batched_sv_merged_batch_extra_write_uses_current_buffer_formula(
    monkeypatch,
):
    FlashAttn_BatchedMatMul_Simulation.clear_caches()

    batch_size = 4
    m_dim = 8
    k_dim = 2
    n_dim = 16
    merged_batch_cycle_count = 10
    per_batch_cycle_count = 100
    nand_config = make_nand_config(num_channels=2, num_plane=2, tRead=4.0, page_size=16)
    hbm_bw = 1.0e12
    bandwidth_config_key = flash_attention_module._build_bandwidth_config_key_or_raise(
        nand_config,
        hbm_bw,
    )
    expected_extra_write_cycles = (
        flash_attention_module._simulate_flashattn_sv_cli_write_cycle_count(
            A100_80GB_FP16,
            bandwidth_config_key,
            sv_mk_bytes=(batch_size - 1) * m_dim * n_dim * 2,
        )
    )

    class FakeMatMulSimulation:
        def __init__(self, M, K, N, weight_bits=16, matmul_type="QK"):
            assert (M, N, weight_bits, matmul_type) == (m_dim, n_dim, 16, "SV")
            self.K = K

        def _build_compile_result(
            self,
            *,
            pcb_module,
            bandwidth_config_key,
            compile_mode,
        ):
            assert pcb_module is A100_80GB_FP16
            assert compile_mode == "heuristic-GPU"
            assert bandwidth_config_key == flash_attention_module._build_bandwidth_config_key_or_raise(
                nand_config,
                hbm_bw,
            )
            if self.K == k_dim:
                return SimpleNamespace(best_cycle_count=per_batch_cycle_count)
            if self.K == k_dim * batch_size:
                return SimpleNamespace(best_cycle_count=merged_batch_cycle_count)
            raise AssertionError(f"Unexpected K dimension: {self.K}")

    monkeypatch.setattr(
        flash_attention_module,
        "MatMul_Simulation",
        FakeMatMulSimulation,
    )

    cycles = FlashAttn_BatchedMatMul_Simulation.get_instance(
        dim=(batch_size, m_dim, k_dim, n_dim),
        weight_bits=16,
        matmul_type="SV",
    ).compile_and_simulate(
        pcb_module=A100_80GB_FP16,
        nand_config=nand_config,
        hbm_bandwidth_bytes_per_sec=hbm_bw,
        compile_mode="heuristic-GPU",
    )

    assert cycles == merged_batch_cycle_count + expected_extra_write_cycles


def test_flashattn_batched_mla_qk_merged_batch_skips_extra_output_write(monkeypatch):
    FlashAttn_BatchedMatMul_Simulation.clear_caches()

    batch_size = 4
    m_dim = 8
    k_dim = 2
    n_dim = 16
    merged_batch_cycle_count = 10
    per_batch_cycle_count = 100
    nand_config = make_nand_config(num_channels=2, num_plane=2, tRead=4.0, page_size=16)
    hbm_bw = 1.0e12

    class FakeMatMulSimulation:
        def __init__(self, M, K, N, weight_bits=16, matmul_type="QK"):
            assert (M, N, weight_bits, matmul_type) == (m_dim, n_dim, 16, "MLA_QK")
            self.K = K

        def _build_compile_result(
            self,
            *,
            pcb_module,
            bandwidth_config_key,
            compile_mode,
        ):
            assert pcb_module is A100_80GB_FP16
            assert compile_mode == "heuristic-GPU"
            assert bandwidth_config_key == flash_attention_module._build_bandwidth_config_key_or_raise(
                nand_config,
                hbm_bw,
            )
            if self.K == k_dim:
                return SimpleNamespace(best_cycle_count=per_batch_cycle_count)
            if self.K == k_dim * batch_size:
                return SimpleNamespace(best_cycle_count=merged_batch_cycle_count)
            raise AssertionError(f"Unexpected K dimension: {self.K}")

    monkeypatch.setattr(
        flash_attention_module,
        "MatMul_Simulation",
        FakeMatMulSimulation,
    )

    cycles = FlashAttn_BatchedMatMul_Simulation.get_instance(
        dim=(batch_size, m_dim, k_dim, n_dim),
        weight_bits=16,
        matmul_type="MLA_QK",
    ).compile_and_simulate(
        pcb_module=A100_80GB_FP16,
        nand_config=nand_config,
        hbm_bandwidth_bytes_per_sec=hbm_bw,
        compile_mode="heuristic-GPU",
    )

    assert cycles == merged_batch_cycle_count


def test_flashattn_batched_mla_sv_merged_batch_reuses_sv_extra_output_write(monkeypatch):
    FlashAttn_BatchedMatMul_Simulation.clear_caches()

    batch_size = 4
    m_dim = 8
    k_dim = 2
    n_dim = 16
    merged_batch_cycle_count = 10
    per_batch_cycle_count = 100
    nand_config = make_nand_config(num_channels=2, num_plane=2, tRead=4.0, page_size=16)
    hbm_bw = 1.0e12
    bandwidth_config_key = flash_attention_module._build_bandwidth_config_key_or_raise(
        nand_config,
        hbm_bw,
    )
    expected_extra_write_cycles = (
        flash_attention_module._simulate_flashattn_sv_cli_write_cycle_count(
            A100_80GB_FP16,
            bandwidth_config_key,
            sv_mk_bytes=(batch_size - 1) * m_dim * n_dim * 2,
        )
    )

    class FakeMatMulSimulation:
        def __init__(self, M, K, N, weight_bits=16, matmul_type="QK"):
            assert (M, N, weight_bits, matmul_type) == (m_dim, n_dim, 16, "MLA_SV")
            self.K = K

        def _build_compile_result(
            self,
            *,
            pcb_module,
            bandwidth_config_key,
            compile_mode,
        ):
            assert pcb_module is A100_80GB_FP16
            assert compile_mode == "heuristic-GPU"
            assert bandwidth_config_key == flash_attention_module._build_bandwidth_config_key_or_raise(
                nand_config,
                hbm_bw,
            )
            if self.K == k_dim:
                return SimpleNamespace(best_cycle_count=per_batch_cycle_count)
            if self.K == k_dim * batch_size:
                return SimpleNamespace(best_cycle_count=merged_batch_cycle_count)
            raise AssertionError(f"Unexpected K dimension: {self.K}")

    monkeypatch.setattr(
        flash_attention_module,
        "MatMul_Simulation",
        FakeMatMulSimulation,
    )

    cycles = FlashAttn_BatchedMatMul_Simulation.get_instance(
        dim=(batch_size, m_dim, k_dim, n_dim),
        weight_bits=16,
        matmul_type="MLA_SV",
    ).compile_and_simulate(
        pcb_module=A100_80GB_FP16,
        nand_config=nand_config,
        hbm_bandwidth_bytes_per_sec=hbm_bw,
        compile_mode="heuristic-GPU",
    )

    assert cycles == merged_batch_cycle_count + expected_extra_write_cycles


def test_flashmla_compile_and_simulate_forwards_mla_matmul_types(monkeypatch):
    expected_nand_config = make_nand_config()
    expected_hbm_bw = 1.0e12
    seen_matmul_types = []

    class FakeBatchedSimulation:
        def compile_and_simulate(
            self,
            *,
            pcb_module,
            nand_config,
            hbm_bandwidth_bytes_per_sec,
            compile_mode,
            return_unit="cycle",
        ):
            assert pcb_module is A100_80GB_FP16
            assert nand_config == expected_nand_config
            assert hbm_bandwidth_bytes_per_sec == expected_hbm_bw
            assert compile_mode == "heuristic-GPU"
            assert return_unit == "cycle"
            return 7

    class FakeSoftmaxSimulation:
        def __init__(self, dim, weight_bits=16):
            self.dim = dim
            self.weight_bits = weight_bits

        @staticmethod
        def get_instance(dim, weight_bits=16):
            return FakeSoftmaxSimulation(dim=dim, weight_bits=weight_bits)

        def compile_and_simulate(self, *, pcb_module, compile_mode, return_unit="cycle"):
            assert pcb_module is A100_80GB_FP16
            assert compile_mode == "heuristic-GPU"
            assert return_unit == "cycle"
            return 5

    def fake_get_instance(dim, weight_bits=16, matmul_type="QK"):
        del dim, weight_bits
        seen_matmul_types.append(matmul_type)
        return FakeBatchedSimulation()

    monkeypatch.setattr(
        flash_attention_module.FlashAttn_BatchedMatMul_Simulation,
        "get_instance",
        staticmethod(fake_get_instance),
    )
    monkeypatch.setattr(
        flash_attention_module,
        "Softmax_Simulation",
        FakeSoftmaxSimulation,
    )

    cycles = FlashMLA_BatchedMatMul_Simulation(
        qk_latent_dim=(2, 1, 4, 2),
        qk_rope_dim=(2, 1, 2, 2),
        sv_latent_dim=(2, 1, 2, 4),
        softmax_dim=(2, 2),
        weight_bits=16,
    ).compile_and_simulate(
        pcb_module=A100_80GB_FP16,
        nand_config=expected_nand_config,
        hbm_bandwidth_bytes_per_sec=expected_hbm_bw,
        compile_mode="heuristic-GPU",
    )

    assert cycles == 17
    assert seen_matmul_types == ["MLA_QK", "MLA_QK", "MLA_SV"]


def test_flashmla_small_shape_does_not_trigger_chunk_plan():
    instance = FlashMLA_BatchedMatMul_Simulation(
        qk_latent_dim=(2, 8, 4, 2),
        qk_rope_dim=(2, 8, 2, 2),
        sv_latent_dim=(2, 8, 2, 4),
        softmax_dim=(16, 2),
        weight_bits=16,
    )

    assert instance._build_chunk_plan_or_none() is None


def test_flashmla_large_shape_chunk_plan_matches_notebook_case():
    instance = FlashMLA_BatchedMatMul_Simulation(
        qk_latent_dim=(5130, 128, 512, 228),
        qk_rope_dim=(5130, 128, 64, 228),
        sv_latent_dim=(5130, 128, 228, 512),
        softmax_dim=(5130, 29184),
        weight_bits=16,
    )

    chunk_plan = instance._build_chunk_plan_or_none()

    assert chunk_plan is not None
    assert chunk_plan.blocks_per_chunk == 32
    assert chunk_plan.num_chunks == 161


def test_flashmla_chunked_simulation_reuses_batched_matmul_and_softmax_caches(
    monkeypatch,
):
    FlashAttn_BatchedMatMul_Simulation.clear_caches()
    Softmax_Simulation.clear_caches()

    def fake_matmul_build_compile_result(
        self,
        *,
        pcb_module,
        bandwidth_config_key,
        compile_mode,
    ):
        del self, pcb_module, bandwidth_config_key, compile_mode
        return matmul_module.MatMul_Simulation.CompileResult(
            best_mapping="fake",
            best_cycle_count=7,
            best_time_ns=7,
        )

    def fake_softmax_build_compile_result(self, *, pcb_module, compile_mode):
        del self, pcb_module, compile_mode
        return flash_attention_module.Softmax_Simulation.CompileResult(
            best_mapping="fake",
            best_cycle_count=5,
            best_time_ns=5,
        )

    monkeypatch.setattr(
        flash_attention_module.MatMul_Simulation,
        "_build_compile_result",
        fake_matmul_build_compile_result,
    )
    monkeypatch.setattr(
        flash_attention_module.Softmax_Simulation,
        "_build_compile_result",
        fake_softmax_build_compile_result,
    )

    instance = FlashMLA_BatchedMatMul_Simulation(
        qk_latent_dim=(5130, 128, 512, 228),
        qk_rope_dim=(5130, 128, 64, 228),
        sv_latent_dim=(5130, 128, 228, 512),
        softmax_dim=(5130, 29184),
        weight_bits=16,
    )
    nand_config = make_nand_config()
    hbm_bw = hbm_bandwidth_bytes_per_sec()

    first_cycles = instance.compile_and_simulate(
        pcb_module=A100_80GB_FP16,
        nand_config=nand_config,
        hbm_bandwidth_bytes_per_sec=hbm_bw,
        compile_mode="heuristic-GPU",
    )
    flash_cache_after_first = (
        FlashAttn_BatchedMatMul_Simulation.compile_and_simulate.cache_info()
    )
    softmax_cache_after_first = Softmax_Simulation.compile_and_simulate.cache_info()

    second_cycles = instance.compile_and_simulate(
        pcb_module=A100_80GB_FP16,
        nand_config=nand_config,
        hbm_bandwidth_bytes_per_sec=hbm_bw,
        compile_mode="heuristic-GPU",
    )
    flash_cache_after_second = (
        FlashAttn_BatchedMatMul_Simulation.compile_and_simulate.cache_info()
    )
    softmax_cache_after_second = Softmax_Simulation.compile_and_simulate.cache_info()

    assert first_cycles == second_cycles
    assert flash_cache_after_first.misses == 6
    assert flash_cache_after_second.misses == 6
    assert flash_cache_after_second.hits > flash_cache_after_first.hits
    assert softmax_cache_after_first.misses == 2
    assert softmax_cache_after_second.misses == 2
    assert softmax_cache_after_second.hits > softmax_cache_after_first.hits


def test_softmax_get_instance_reuses_same_object():
    Softmax_Simulation.clear_caches()

    first = Softmax_Simulation.get_instance(dim=(8, 16), weight_bits=16)
    second = Softmax_Simulation.get_instance(dim=(8, 16), weight_bits=16)
    third = Softmax_Simulation.get_instance(dim=(8, 32), weight_bits=16)

    assert first is second
    assert first is not third


def test_softmax_compile_and_simulate_supports_time_ns():
    instance = Softmax_Simulation(dim=(8, 16), weight_bits=16)

    cycles = instance.compile_and_simulate(
        pcb_module=A100_80GB_FP16,
        compile_mode=None,
    )
    time_ns = instance.compile_and_simulate(
        pcb_module=A100_80GB_FP16,
        compile_mode=None,
        return_unit="time_ns",
    )

    assert time_ns == math.ceil(
        cycles * 1e9 / A100_80GB_FP16.compute_module.clock_freq
    )


def test_compute_engine_forwards_nand_config_and_hbm_bandwidth_to_matmul(monkeypatch):
    nand_config = make_nand_config()
    hbm_bw = 1.23e12
    _reset_and_init_sim_session()
    engine = xpu_module.ComputeEngine(
        nand_config,
        hbm_bandwidth_bytes_per_sec=hbm_bw,
    )

    class FakeMatMulSimulation:
        def compile_and_simulate(
            self,
            *,
            pcb_module,
            nand_config,
            hbm_bandwidth_bytes_per_sec,
            compile_mode,
            return_unit="cycle",
        ):
            assert pcb_module is engine.device
            assert nand_config is engine.config
            assert hbm_bandwidth_bytes_per_sec == hbm_bw
            assert compile_mode == engine.compile_mode
            assert return_unit == "time_ns"
            return 7

    monkeypatch.setattr(
        xpu_module.MatMul_Simulation,
        "get_instance",
        staticmethod(lambda dim, weight_bits=16: FakeMatMulSimulation()),
    )

    assert engine.execute_macro_op(MatMulOp(dim=(2, 16, 8), weight_bits=16)) == 7
    SimSession.reset()


def test_compute_engine_forwards_nand_config_and_hbm_bandwidth_to_flashattn(monkeypatch):
    nand_config = make_nand_config()
    hbm_bw = 1.23e12
    _reset_and_init_sim_session()
    engine = xpu_module.ComputeEngine(
        nand_config,
        hbm_bandwidth_bytes_per_sec=hbm_bw,
    )

    class FakeBatchedSimulation:
        def compile_and_simulate(
            self,
            *,
            pcb_module,
            nand_config,
            hbm_bandwidth_bytes_per_sec,
            compile_mode,
            return_unit="cycle",
        ):
            assert pcb_module is engine.device
            assert nand_config is engine.config
            assert hbm_bandwidth_bytes_per_sec == hbm_bw
            assert compile_mode == engine.compile_mode
            assert return_unit == "time_ns"
            return 11

    class FakeSoftmaxSimulation:
        def __init__(self, dim, weight_bits=16):
            self.dim = dim
            self.weight_bits = weight_bits

        def compile_and_simulate(self, *, pcb_module, compile_mode, return_unit="cycle"):
            assert pcb_module is engine.device
            assert compile_mode == engine.compile_mode
            assert return_unit == "time_ns"
            return 13

    monkeypatch.setattr(
        xpu_module.FlashAttn_BatchedMatMul_Simulation,
        "get_instance",
        staticmethod(
            lambda dim, matmul_type="QK", weight_bits=16: FakeBatchedSimulation()
        ),
    )
    monkeypatch.setattr(xpu_module, "Softmax_Simulation", FakeSoftmaxSimulation)

    assert (
        engine.execute_macro_op(
            FlashAttnOp(
                qk_bmm_shape=(2, 4, 8, 6),
                sv_bmm_shape=(2, 4, 6, 3),
                softmax_shape=(4, 6),
                weight_bits=16,
            )
        )
        == 35
    )
    SimSession.reset()


def test_compute_engine_vector_time_ns_follow_macro_op_weight_bits():
    _reset_and_init_sim_session()
    engine = xpu_module.ComputeEngine(
        make_nand_config(),
        hbm_bandwidth_bytes_per_sec=hbm_bandwidth_bytes_per_sec(),
    )

    vector_16_time_ns = engine.execute_macro_op(
        VectorOp(vector_op_type="rms_norm", vector_shape=[1024, 1024], weight_bits=16)
    )
    vector_8_time_ns = engine.execute_macro_op(
        VectorOp(vector_op_type="rms_norm", vector_shape=[1024, 1024], weight_bits=8)
    )

    assert vector_8_time_ns < vector_16_time_ns
    SimSession.reset()


def test_transfer_engine_accepts_supported_transfer_ops():
    _reset_and_init_sim_session()
    engine = xpu_module.TransferEngine()

    assert engine.execute_macro_op(AllGatherOp(num_ranks=4, data_size=128)) == 1.0
    assert engine.execute_macro_op(ReduceScatterOp(num_ranks=4, data_size=128)) == 1.0
    SimSession.reset()


def test_transfer_engine_allreduce_uses_estimate_path(monkeypatch):
    _reset_and_init_sim_session()
    engine = xpu_module.TransferEngine()
    called = {}

    class FakeAllReduceSimulation:
        def __init__(self, num_gpus, data_size, weight_bits=16):
            called["init"] = (num_gpus, data_size, weight_bits)

        def compile_and_simulate(
            self,
            *,
            pcb_module,
            interconnect_module,
            compile_mode,
            return_unit="cycle",
        ):
            assert pcb_module is engine.device
            assert compile_mode == engine.compile_mode
            assert return_unit == "time_ns"
            called["interconnect"] = interconnect_module
            return 9

    monkeypatch.setattr(xpu_module, "AllReduceSimulation", FakeAllReduceSimulation)

    result = engine.execute_macro_op(
        AllReduceOp(num_ranks=4, data_size=128, weight_bits=16)
    )

    assert result == 9
    assert called["init"] == (4, 128, 16)
    assert called["interconnect"] is not None
    SimSession.reset()


def test_transfer_engine_all2all_uses_estimate_path(monkeypatch):
    _reset_and_init_sim_session()
    engine = xpu_module.TransferEngine()
    called = {}

    class FakeAll2AllSimulation:
        def __init__(self, num_gpus, data_size, weight_bits=16):
            called["init"] = (num_gpus, data_size, weight_bits)

        def compile_and_simulate(
            self,
            *,
            pcb_module,
            interconnect_module,
            compile_mode,
            return_unit="cycle",
        ):
            assert pcb_module is engine.device
            assert compile_mode == engine.compile_mode
            assert return_unit == "time_ns"
            called["interconnect"] = interconnect_module
            return 11

    monkeypatch.setattr(
        xpu_module,
        "AllToAllPrimitive_Simulation",
        FakeAll2AllSimulation,
    )

    result = engine.execute_macro_op(
        All2AllOp(num_gpus=4, data_size=128, weight_bits=16)
    )

    assert result == 11
    assert called["init"] == (4, 128, 16)
    assert called["interconnect"] is not None
    SimSession.reset()


def test_transfer_engine_rejects_compute_ops():
    _reset_and_init_sim_session()
    engine = xpu_module.TransferEngine()

    with pytest.raises(TypeError):
        engine.execute_macro_op(MatMulOp(dim=(2, 16, 8), weight_bits=16))
    SimSession.reset()


def test_compute_engine_waits_for_execute_time_ns(monkeypatch):
    SimSession.reset()
    SimSession.init()

    engine = xpu_module.ComputeEngine(
        make_nand_config(),
        hbm_bandwidth_bytes_per_sec=hbm_bandwidth_bytes_per_sec(),
    )
    slot = DepSlot(VectorOp(vector_op_type="rms_norm", vector_shape=[2, 16], weight_bits=16))
    engine.load_command_queue([slot])

    monkeypatch.setattr(engine, "execute_macro_op", lambda macro_op: 7.0)

    SimSession.scheduler.run()

    assert int(SimSession.sim_time.cycle) == 8
    SimSession.reset()


def test_transfer_engine_waits_for_execute_time_ns(monkeypatch):
    SimSession.reset()
    SimSession.init()

    engine = xpu_module.TransferEngine()
    slot = DepSlot(All2AllOp(num_gpus=4, data_size=128, weight_bits=16))
    engine.load_command_queue([slot])

    monkeypatch.setattr(engine, "execute_macro_op", lambda macro_op: 5.0)

    SimSession.scheduler.run()

    assert int(SimSession.sim_time.cycle) == 6
    SimSession.reset()
