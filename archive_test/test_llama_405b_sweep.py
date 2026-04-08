from pathlib import Path

import pytest
from nandmachine.config.hbm_hbf_architecture import (
    build_device_for_hbm_hbf_architecture_or_raise,
)
import scripts.llama_405b_sweep as sweep_module
from scripts.llama_405b_sweep import (
    CSV_FIELDNAMES,
    HARDWARE_SPECS,
    MODEL_CARD_PATH,
    SEQUENCE_CASE_CONFIGS,
    TRACE_ROOT,
    SweepCase,
    build_summary_csv_path,
    build_nand_config,
    build_runtime_spec,
    build_sweep_cases,
    build_trace_dir,
    get_hardware_spec_or_raise,
    load_model_card_or_raise,
)


def _count_hbm_cases() -> int:
    return sum(
        len(batch_sizes)
        for sequence_case_config in SEQUENCE_CASE_CONFIGS
        for batch_sizes in (sequence_case_config.hbm_batch_sizes_by_ranks or {}).values()
    )


def _count_collective_cases(mode: str) -> int:
    if mode == "csi":
        attr_name = "csi_batch_sizes_by_ranks_by_slo_ms"
    elif mode == "cli":
        attr_name = "cli_batch_sizes_by_ranks_by_slo_ms"
    else:
        raise AssertionError(f"Unsupported mode: {mode}")

    return sum(
        len(batch_sizes)
        for sequence_case_config in SEQUENCE_CASE_CONFIGS
        for batch_sizes_by_ranks in (getattr(sequence_case_config, attr_name) or {}).values()
        for batch_sizes in batch_sizes_by_ranks.values()
    )


def _configured_sequence_pairs() -> set[tuple[int, int]]:
    return {
        (sequence_case_config.input_sequence_length, sequence_case_config.output_sequence_length)
        for sequence_case_config in SEQUENCE_CASE_CONFIGS
        if sequence_case_config.hbm_batch_sizes_by_ranks
        or sequence_case_config.csi_batch_sizes_by_ranks_by_slo_ms
        or sequence_case_config.cli_batch_sizes_by_ranks_by_slo_ms
    }


def test_llama_405b_sweep_builds_expected_case_count() -> None:
    cases = build_sweep_cases()

    assert {
        (sequence_case_config.input_sequence_length, sequence_case_config.output_sequence_length)
        for sequence_case_config in SEQUENCE_CASE_CONFIGS
    } == {
        (9400, 600),
        (8000, 1000),
        (20000, 1000),
    }
    assert {(case.input_sequence_length, case.output_sequence_length) for case in cases} == _configured_sequence_pairs()
    assert len(cases) == _count_hbm_cases() + _count_collective_cases("csi") + _count_collective_cases("cli")
    assert sum(1 for case in cases if case.hardware_type == "H200-HBM") == _count_hbm_cases()
    assert sum(1 for case in cases if case.hardware_type == "H200-HBF-CSI") == _count_collective_cases("csi")
    assert sum(1 for case in cases if case.hardware_type == "H200-HBF-CLI") == _count_collective_cases("cli")


def test_llama_405b_sweep_csv_includes_interconnect_topology() -> None:
    assert "interconnect_topology" in CSV_FIELDNAMES


def test_llama_405b_sweep_uses_expected_batch_profiles_by_mode_and_slo() -> None:
    active_sequence_case_config = next(
        sequence_case_config
        for sequence_case_config in SEQUENCE_CASE_CONFIGS
        if sequence_case_config.hbm_batch_sizes_by_ranks
        or sequence_case_config.csi_batch_sizes_by_ranks_by_slo_ms
        or sequence_case_config.cli_batch_sizes_by_ranks_by_slo_ms
    )

    assert active_sequence_case_config.hbm_batch_sizes_by_ranks == {
        8: (64, 32, 16, 8),
    }
    assert active_sequence_case_config.csi_batch_sizes_by_ranks_by_slo_ms == {
        50: {
            4: (28, 16, 8, 4),
            8: (224, 128, 64, 32),
        },
        100: {
            4: (228, 128, 64, 32),
            8: (624, 512, 256, 128),
        },
    }
    assert active_sequence_case_config.cli_batch_sizes_by_ranks_by_slo_ms == {
        50: {
            8: (160, 128, 64, 32),
        },
        100: {
            4: (160, 128, 64, 32),
            8: (496, 256, 128, 64),
        },
    }


def test_llama_405b_sweep_paths_are_anchored_to_repo_root() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_tag = "20260409_0100"

    assert sweep_module.REPO_ROOT == repo_root
    assert MODEL_CARD_PATH == repo_root / "model_cards" / "llama-405B.json"
    assert TRACE_ROOT == repo_root / "trace" / "main"
    assert build_summary_csv_path(run_tag) == (
        TRACE_ROOT / "llama_405b_sweep_summary_20260409_0100.csv"
    )


def test_load_model_card_or_raise_ignores_current_working_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    model_card = load_model_card_or_raise()

    assert model_card["model_type"] == "llama"


def test_llama_405b_sweep_trace_dir_is_built_under_repo_trace_root() -> None:
    run_tag = "20260409_0100"
    trace_dir = build_trace_dir(
        SweepCase(
            hardware_type="H200-HBF-CSI",
            num_ranks=8,
            batch_size=120,
            input_sequence_length=9400,
            output_sequence_length=600,
            slo_ms=100,
        ),
        run_tag,
    )

    assert trace_dir == (
        TRACE_ROOT
        / "llama_405b_sweep_20260409_0100"
        / "H200-HBF-CSI"
        / "ranks_8"
        / "slo_100ms"
        / "isl_9400_osl_600"
        / "bs_120"
    )


def test_llama_405b_sweep_runtime_spec_uses_expected_hbm_bandwidth_by_mode() -> None:
    assert any(
        hardware_spec.hardware_type == "H200-HBF-CLI"
        for hardware_spec in HARDWARE_SPECS
    )

    hbm_hardware_spec = get_hardware_spec_or_raise("H200-HBM")
    hbm_nand_config = build_nand_config(hbm_hardware_spec)
    hbm_runtime_spec = build_runtime_spec(hbm_hardware_spec, hbm_nand_config)
    assert hbm_runtime_spec.sim_hbm_bandwidth_GBps == 4800.0

    csi_hardware_spec = get_hardware_spec_or_raise("H200-HBF-CSI")
    csi_nand_config = build_nand_config(csi_hardware_spec)
    csi_runtime_spec = build_runtime_spec(csi_hardware_spec, csi_nand_config)
    csi_device = build_device_for_hbm_hbf_architecture_or_raise(
        csi_hardware_spec.device_name,
        csi_hardware_spec.memory_architecture,
    )
    csi_residual_hbm_bandwidth_GBps = (
        csi_device.io_module.total_bandwidth / 1e9
        - csi_runtime_spec.derived_hbf_bandwidth_GBps
    )

    assert csi_runtime_spec.sim_hbm_bandwidth_GBps == 4800.0
    assert csi_runtime_spec.sim_hbm_bandwidth_GBps != pytest.approx(
        csi_residual_hbm_bandwidth_GBps
    )

    cli_hardware_spec = get_hardware_spec_or_raise("H200-HBF-CLI")
    cli_nand_config = build_nand_config(cli_hardware_spec)
    cli_runtime_spec = build_runtime_spec(cli_hardware_spec, cli_nand_config)
    cli_device = build_device_for_hbm_hbf_architecture_or_raise(
        cli_hardware_spec.device_name,
        cli_hardware_spec.memory_architecture,
    )
    cli_residual_hbm_bandwidth_GBps = (
        cli_device.io_module.total_bandwidth / 1e9
        - cli_runtime_spec.derived_hbf_bandwidth_GBps
    )

    assert cli_residual_hbm_bandwidth_GBps > 0
    assert cli_runtime_spec.sim_hbm_bandwidth_GBps == pytest.approx(
        cli_residual_hbm_bandwidth_GBps
    )


def test_llama_405b_sweep_excludes_invalid_rank_profiles() -> None:
    cases = build_sweep_cases()

    assert not any(case.num_ranks == 16 for case in cases)
    assert not any(
        case.hardware_type == "H200-HBF-CLI"
        and case.slo_ms == 50
        and case.num_ranks == 4
        for case in cases
    )
