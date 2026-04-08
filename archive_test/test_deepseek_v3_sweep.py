from pathlib import Path

from scripts.deepseek_v3_sweep import (
    CSV_FIELDNAMES,
    HARDWARE_SPECS,
    SEQUENCE_CASE_CONFIGS,
    SWEEP_NAME,
    TRACE_ROOT,
    SweepCase,
    build_summary_csv_path,
    build_sweep_cases,
    build_trace_dir,
    build_trace_root,
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


def test_deepseek_v3_sweep_sequence_case_configs_cover_expected_buckets() -> None:
    assert {
        (sequence_case_config.input_sequence_length, sequence_case_config.output_sequence_length)
        for sequence_case_config in SEQUENCE_CASE_CONFIGS
    } == {
        (9400, 600),
        (8000, 1000),
        (20000, 1000),
    }


def test_deepseek_v3_sweep_csv_includes_interconnect_topology() -> None:
    assert "interconnect_topology" in CSV_FIELDNAMES


def test_deepseek_v3_sweep_builds_cases_from_configured_buckets_only() -> None:
    cases = build_sweep_cases()

    assert {(case.input_sequence_length, case.output_sequence_length) for case in cases} == _configured_sequence_pairs()
    assert len(cases) == _count_hbm_cases() + _count_collective_cases("csi") + _count_collective_cases("cli")
    assert sum(1 for case in cases if case.hardware_type == "H200-HBM") == _count_hbm_cases()
    assert sum(1 for case in cases if case.hardware_type == "H200-HBF-CSI") == _count_collective_cases("csi")
    assert sum(1 for case in cases if case.hardware_type == "H200-HBF-CLI") == _count_collective_cases("cli")


def test_deepseek_v3_sweep_uses_expected_batch_profiles_by_mode_and_slo() -> None:
    expected_hbm = {
        8: (696, 512, 256, 128),
        16: (2416, 2048, 1024, 512),
    }
    expected_csi_50 = {
        4: (440, 256, 128, 64),
        8: (1904, 1024, 512, 256),
        16: (4832, 4096, 2048, 1024),
    }
    expected_csi_100 = {
        4: (1904, 1024, 512, 256),
        8: (4840, 4096, 2048, 1024),
        16: (10704, 8192, 4096, 2048),
    }
    expected_cli_50 = {
        4: (196, 128, 64, 32),
        8: (1416, 1024, 512, 256),
        16: (3856, 2048, 1024, 512),
    }
    expected_cli_100 = {
        4: (1416, 1024, 512, 256),
        8: (3856, 2048, 1024, 512),
        16: (8752, 8192, 4096, 2048),
    }

    cases = build_sweep_cases()
    active_sequence_case_config = next(
        sequence_case_config
        for sequence_case_config in SEQUENCE_CASE_CONFIGS
        if sequence_case_config.hbm_batch_sizes_by_ranks
        or sequence_case_config.csi_batch_sizes_by_ranks_by_slo_ms
        or sequence_case_config.cli_batch_sizes_by_ranks_by_slo_ms
    )
    input_sequence_length = active_sequence_case_config.input_sequence_length
    output_sequence_length = active_sequence_case_config.output_sequence_length

    hbm_batches_by_rank = {
        rank: tuple(
            case.batch_size
            for case in cases
            if case.hardware_type == "H200-HBM"
            and case.num_ranks == rank
            and case.input_sequence_length == input_sequence_length
            and case.output_sequence_length == output_sequence_length
        )
        for rank in (active_sequence_case_config.hbm_batch_sizes_by_ranks or {})
    }
    csi_100_batches_by_rank = {
        rank: tuple(
            case.batch_size
            for case in cases
            if case.hardware_type == "H200-HBF-CSI"
            and case.slo_ms == 100
            and case.num_ranks == rank
            and case.input_sequence_length == input_sequence_length
            and case.output_sequence_length == output_sequence_length
        )
        for rank in (
            active_sequence_case_config.csi_batch_sizes_by_ranks_by_slo_ms or {}
        ).get(100, {})
    }
    csi_50_batches_by_rank = {
        rank: tuple(
            case.batch_size
            for case in cases
            if case.hardware_type == "H200-HBF-CSI"
            and case.slo_ms == 50
            and case.num_ranks == rank
            and case.input_sequence_length == input_sequence_length
            and case.output_sequence_length == output_sequence_length
        )
        for rank in (
            active_sequence_case_config.csi_batch_sizes_by_ranks_by_slo_ms or {}
        ).get(50, {})
    }
    cli_100_batches_by_rank = {
        rank: tuple(
            case.batch_size
            for case in cases
            if case.hardware_type == "H200-HBF-CLI"
            and case.slo_ms == 100
            and case.num_ranks == rank
            and case.input_sequence_length == input_sequence_length
            and case.output_sequence_length == output_sequence_length
        )
        for rank in (
            active_sequence_case_config.cli_batch_sizes_by_ranks_by_slo_ms or {}
        ).get(100, {})
    }
    cli_50_batches_by_rank = {
        rank: tuple(
            case.batch_size
            for case in cases
            if case.hardware_type == "H200-HBF-CLI"
            and case.slo_ms == 50
            and case.num_ranks == rank
            and case.input_sequence_length == input_sequence_length
            and case.output_sequence_length == output_sequence_length
        )
        for rank in (
            active_sequence_case_config.cli_batch_sizes_by_ranks_by_slo_ms or {}
        ).get(50, {})
    }

    assert active_sequence_case_config.hbm_batch_sizes_by_ranks == expected_hbm
    assert active_sequence_case_config.csi_batch_sizes_by_ranks_by_slo_ms == {
        50: expected_csi_50,
        100: expected_csi_100,
    }
    assert active_sequence_case_config.cli_batch_sizes_by_ranks_by_slo_ms == {
        50: expected_cli_50,
        100: expected_cli_100,
    }
    assert hbm_batches_by_rank == (active_sequence_case_config.hbm_batch_sizes_by_ranks or {})
    assert csi_100_batches_by_rank == (
        active_sequence_case_config.csi_batch_sizes_by_ranks_by_slo_ms or {}
    ).get(100, {})
    assert csi_50_batches_by_rank == (
        active_sequence_case_config.csi_batch_sizes_by_ranks_by_slo_ms or {}
    ).get(50, {})
    assert cli_100_batches_by_rank == (
        active_sequence_case_config.cli_batch_sizes_by_ranks_by_slo_ms or {}
    ).get(100, {})
    assert cli_50_batches_by_rank == (
        active_sequence_case_config.cli_batch_sizes_by_ranks_by_slo_ms or {}
    ).get(50, {})


def test_deepseek_v3_sweep_paths_use_dedicated_run_tag_root() -> None:
    run_tag = "20260409_0100"
    case = SweepCase(
        hardware_type="H200-HBF-CSI",
        num_ranks=16,
        batch_size=3808,
        input_sequence_length=9400,
        output_sequence_length=600,
        slo_ms=50,
    )

    assert {hardware_spec.hardware_type for hardware_spec in HARDWARE_SPECS} == {
        "H200-HBM",
        "H200-HBF-CLI",
        "H200-HBF-CSI",
    }
    assert SWEEP_NAME == "deepseek_v3_sweep"
    assert build_trace_root(run_tag) == Path("trace/main") / "deepseek_v3_sweep_20260409_0100"
    assert build_summary_csv_path(run_tag) == (
        Path("trace/main") / "deepseek_v3_sweep_summary_20260409_0100.csv"
    )
    assert build_trace_dir(case, run_tag) == (
        TRACE_ROOT
        / "deepseek_v3_sweep_20260409_0100"
        / "H200-HBF-CSI"
        / "ranks_16"
        / "slo_50ms"
        / "isl_9400_osl_600"
        / "bs_3808"
    )
