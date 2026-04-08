from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from nandmachine.config.hbm_hbf_architecture import (
    build_device_for_hbm_hbf_architecture_or_raise,
)
from nandmachine.config.model_config import Qwen3MoEModelConfig
from scripts.qwen3_coder_480b_sweep import (
    CSV_FIELDNAMES,
    CSI_BATCH_SIZES_BY_RANKS_BY_SLO_MS,
    HBM_ONLY_BATCH_SIZES_BY_RANKS,
    HARDWARE_SPECS,
    MODEL_CARD_PATH,
    SWEEP_NAME,
    SweepCase,
    build_inference_config,
    build_macro_op_list,
    build_nand_config,
    build_parallel_config,
    build_raw_model_config,
    build_runtime_spec,
    build_summary_csv_path,
    build_sweep_cases,
    build_trace_dir,
    build_trace_root,
    get_hardware_spec_or_raise,
    load_model_card_or_raise,
)


def test_qwen3_coder_480b_sweep_builds_expected_case_count() -> None:
    cases = build_sweep_cases()

    assert len(cases) == 36
    assert sum(1 for case in cases if case.num_ranks == 4) == 12
    assert sum(1 for case in cases if case.num_ranks == 8) == 12
    assert sum(1 for case in cases if case.num_ranks == 16) == 12
    assert all(case.hardware_type != "H200-HBF-CLI" for case in cases)


def test_qwen3_coder_480b_sweep_uses_expected_batch_profiles_by_mode_and_slo() -> None:
    cases = build_sweep_cases()

    hbm_batches_by_rank = {
        rank: tuple(
            case.batch_size
            for case in cases
            if case.hardware_type == "H200-HBM" and case.num_ranks == rank
        )
        for rank in HBM_ONLY_BATCH_SIZES_BY_RANKS
    }
    csi_100_batches_by_rank = {
        rank: tuple(
            case.batch_size
            for case in cases
            if case.hardware_type == "H200-HBF-CSI"
            and case.slo_ms == 100
            and case.num_ranks == rank
        )
        for rank in CSI_BATCH_SIZES_BY_RANKS_BY_SLO_MS[100]
    }
    csi_50_batches_by_rank = {
        rank: tuple(
            case.batch_size
            for case in cases
            if case.hardware_type == "H200-HBF-CSI"
            and case.slo_ms == 50
            and case.num_ranks == rank
        )
        for rank in CSI_BATCH_SIZES_BY_RANKS_BY_SLO_MS[50]
    }

    assert hbm_batches_by_rank == HBM_ONLY_BATCH_SIZES_BY_RANKS
    assert csi_100_batches_by_rank == CSI_BATCH_SIZES_BY_RANKS_BY_SLO_MS[100]
    assert csi_50_batches_by_rank == CSI_BATCH_SIZES_BY_RANKS_BY_SLO_MS[50]


def test_qwen3_coder_480b_sweep_csv_field_order_matches_qwen_layout() -> None:
    assert CSV_FIELDNAMES == [
        "hardware_type",
        "memory_architecture_mode",
        "memory_backend",
        "num_ranks",
        "slo_ms",
        "batch_size",
        "model_throughput_tokens_per_sec",
        "throughput_per_GPU",
        "layer_latency_ns",
        "model_latency_ns",
        "trace_path",
        "effective_hbm_stacks",
        "effective_hbf_stacks",
        "sim_hbm_bandwidth_GBps",
        "derived_hbf_bandwidth_GBps",
        "nand_num_channels",
        "attn_dp_size",
        "attn_tp_size",
        "ffn_tp_size",
        "ffn_ep_size",
        "input_sequence_length",
        "output_sequence_length",
        "macro_op_count",
        "device_name",
        "model_card_path",
        "compile_mode",
        "batch_size_semantics",
        "case_limit",
        "selected_case_count",
        "total_case_count",
        "max_workers",
        "worker_count_source",
        "host_logical_cpu_count",
        "weight_bits",
        "activation_bits",
        "kv_cache_bits",
        "kv_block_size_bytes",
        "omp_num_threads",
        "openblas_num_threads",
        "mkl_num_threads",
        "numexpr_num_threads",
        "blis_num_threads",
        "torch_num_threads",
        "torch_num_interop_threads",
        "nand_num_plane",
        "nand_num_block",
        "nand_num_pages",
        "nand_tRead",
        "nand_tWrite",
        "nand_tErase",
        "nand_page_size_kb",
        "nand_sram_threshold_kb",
    ]


def test_qwen3_coder_480b_sweep_runtime_spec_uses_expected_hbm_semantics() -> None:
    hbm_hardware_spec = get_hardware_spec_or_raise("H200-HBM")
    hbm_nand_config = build_nand_config(hbm_hardware_spec)
    hbm_runtime_spec = build_runtime_spec(hbm_hardware_spec, hbm_nand_config)
    assert hbm_runtime_spec.sim_hbm_bandwidth_GBps == 4800.0

    csi_hardware_spec = get_hardware_spec_or_raise("H200-HBF-CSI")
    csi_nand_config = build_nand_config(csi_hardware_spec)
    csi_runtime_spec = build_runtime_spec(csi_hardware_spec, csi_nand_config)
    assert csi_runtime_spec.sim_hbm_bandwidth_GBps == 4800.0

    cli_hardware_spec = get_hardware_spec_or_raise("H200-HBF-CLI")
    cli_nand_config = build_nand_config(cli_hardware_spec)
    cli_runtime_spec = build_runtime_spec(cli_hardware_spec, cli_nand_config)
    cli_device = build_device_for_hbm_hbf_architecture_or_raise(
        cli_hardware_spec.device_name,
        cli_hardware_spec.memory_architecture,
    )
    expected_cli_hbm_bandwidth_GBps = (
        cli_device.io_module.total_bandwidth / 1e9
        - cli_runtime_spec.derived_hbf_bandwidth_GBps
    )

    assert cli_runtime_spec.sim_hbm_bandwidth_GBps == pytest.approx(
        expected_cli_hbm_bandwidth_GBps
    )


def test_qwen3_coder_480b_sweep_normalizes_model_card_fields() -> None:
    model_card = load_model_card_or_raise()

    assert MODEL_CARD_PATH == Path("model_cards/qwen3-coder-480B.json")
    assert model_card["attention_type"] == "gqa"
    assert model_card["attention_bias"] is False
    assert model_card["shared_expert_intermediate_size"] is None

    raw_model_config = build_raw_model_config(deepcopy(model_card))
    model_config = Qwen3MoEModelConfig.from_config(raw_model_config)

    assert model_config.hidden_size == 6144
    assert model_config.num_hidden_layers == 62
    assert model_config.attention_bias is False
    assert model_config.shared_expert_intermediate_size is None


def test_qwen3_coder_480b_sweep_builds_non_empty_macro_op_list() -> None:
    model_card = load_model_card_or_raise()
    raw_model_config = build_raw_model_config(deepcopy(model_card))
    model_config = Qwen3MoEModelConfig.from_config(raw_model_config)
    hardware_spec = get_hardware_spec_or_raise("H200-HBM")
    parallel_config = build_parallel_config(4)
    nand_config = build_nand_config(hardware_spec)
    inference_config = build_inference_config(
        SweepCase(
            hardware_type="H200-HBM",
            num_ranks=4,
            batch_size=52,
            input_sequence_length=9400,
            output_sequence_length=600,
            slo_ms=None,
        ),
        parallel_config,
        hardware_spec.memory_backend,
    )

    macro_op_list = build_macro_op_list(
        raw_model_config,
        model_config,
        nand_config,
        inference_config,
        parallel_config,
    )

    assert macro_op_list


def test_qwen3_coder_480b_sweep_trace_paths_are_isolated() -> None:
    run_tag = "20260408_1200"
    case = SweepCase(
        hardware_type="H200-HBF-CSI",
        num_ranks=16,
        batch_size=2048,
        input_sequence_length=9400,
        output_sequence_length=600,
        slo_ms=100,
    )

    assert SWEEP_NAME == "qwen3_coder_480b_sweep"
    assert build_trace_root(run_tag) == (
        Path("trace/main") / "qwen3_coder_480b_sweep_20260408_1200"
    )
    assert build_summary_csv_path(run_tag) == (
        Path("trace/main") / "qwen3_coder_480b_sweep_summary_20260408_1200.csv"
    )

    trace_dir = build_trace_dir(case, run_tag)
    assert trace_dir == (
        Path("trace/main")
        / "qwen3_coder_480b_sweep_20260408_1200"
        / "H200-HBF-CSI"
        / "ranks_16"
        / "slo_100ms"
        / "isl_9400_osl_600"
        / "bs_2048"
    )
    assert "qwen3_moe_sweep" not in str(trace_dir)


def test_qwen3_coder_480b_sweep_hardware_specs_include_cli_but_skip_it_in_cases() -> None:
    assert {hardware_spec.hardware_type for hardware_spec in HARDWARE_SPECS} == {
        "H200-HBM",
        "H200-HBF-CLI",
        "H200-HBF-CSI",
    }
    assert all(case.hardware_type != "H200-HBF-CLI" for case in build_sweep_cases())
