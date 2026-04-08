from pathlib import Path

import pytest
from nandmachine.config.hbm_hbf_architecture import (
    build_device_for_hbm_hbf_architecture_or_raise,
)
import scripts.llama_405b_sweep as sweep_module
from scripts.llama_405b_sweep import (
    HARDWARE_SPECS,
    MODEL_CARD_PATH,
    SUMMARY_CSV_PATH,
    TRACE_ROOT,
    SweepCase,
    build_nand_config,
    build_runtime_spec,
    build_sweep_cases,
    build_trace_dir,
    get_hardware_spec_or_raise,
    load_model_card_or_raise,
)


def test_llama_405b_sweep_builds_expected_case_count() -> None:
    cases = build_sweep_cases()

    assert len(cases) == 6
    assert sum(1 for case in cases if case.hardware_type == "H200-HBM") == 4
    assert sum(1 for case in cases if case.hardware_type == "H200-HBF-CSI") == 2
    assert all(case.hardware_type != "H200-HBF-CLI" for case in cases)


def test_llama_405b_sweep_paths_are_anchored_to_repo_root() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    assert sweep_module.REPO_ROOT == repo_root
    assert MODEL_CARD_PATH == repo_root / "model_cards" / "llama-405B.json"
    assert TRACE_ROOT == repo_root / "trace" / "main"
    assert SUMMARY_CSV_PATH == TRACE_ROOT / "llama_405b_sweep_summary.csv"


def test_load_model_card_or_raise_ignores_current_working_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    model_card = load_model_card_or_raise()

    assert model_card["model_type"] == "llama"


def test_llama_405b_sweep_trace_dir_is_built_under_repo_trace_root() -> None:
    trace_dir = build_trace_dir(
        SweepCase(
            hardware_type="H200-HBF-CSI",
            num_ranks=8,
            batch_size=120,
            input_sequence_length=9400,
            output_sequence_length=600,
            slo_ms=100,
        )
    )

    assert trace_dir == (
        TRACE_ROOT
        / "llama_405b_sweep"
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
