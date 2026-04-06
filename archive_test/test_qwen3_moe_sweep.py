from scripts.qwen3_moe_sweep import (
    HARDWARE_SPECS,
    build_nand_config,
    build_runtime_spec,
    build_sweep_cases,
)


def test_qwen3_moe_sweep_builds_expected_case_count() -> None:
    cases = build_sweep_cases()

    assert len(cases) == 108
    assert sum(1 for case in cases if case.num_ranks == 4) == 36
    assert sum(1 for case in cases if case.num_ranks == 8) == 36
    assert sum(1 for case in cases if case.num_ranks == 16) == 36


def test_qwen3_moe_sweep_cli_runtime_uses_requested_stack_split() -> None:
    cli_hardware_spec = next(
        hardware_spec
        for hardware_spec in HARDWARE_SPECS
        if hardware_spec.hardware_type == "H200-HBF-CLI"
    )

    nand_config = build_nand_config(cli_hardware_spec)
    runtime_spec = build_runtime_spec(cli_hardware_spec, nand_config)

    assert nand_config.num_channels == 40
    assert runtime_spec.normalized_architecture["effective_hbm_stacks"] == 1
    assert runtime_spec.normalized_architecture["effective_hbf_stacks"] == 5
    assert runtime_spec.sim_hbm_bandwidth_GBps == 800.0
    assert runtime_spec.derived_hbf_bandwidth_GBps > 0
