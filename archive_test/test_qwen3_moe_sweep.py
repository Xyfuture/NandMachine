from pathlib import Path

import pytest
import scripts.qwen3_moe_sweep as sweep_module
from scripts.qwen3_moe_sweep import (
    CASE_LIMIT_ENV_VAR,
    CSI_BATCH_SIZES_BY_RANKS_BY_SLO_MS,
    CSV_FIELDNAMES,
    DEFAULT_MAX_WORKERS_CPU_DIVISOR,
    HBM_ONLY_BATCH_SIZES_BY_RANKS,
    HARDWARE_SPECS,
    MAX_WORKERS_ENV_VAR,
    SweepCase,
    build_nand_config,
    build_runtime_spec,
    build_sweep_cases,
    build_trace_dir,
    resolve_case_limit,
    resolve_max_workers,
)


def test_qwen3_moe_sweep_builds_expected_case_count() -> None:
    cases = build_sweep_cases()

    assert len(cases) == 36
    assert sum(1 for case in cases if case.num_ranks == 4) == 12
    assert sum(1 for case in cases if case.num_ranks == 8) == 12
    assert sum(1 for case in cases if case.num_ranks == 16) == 12
    assert all(case.hardware_type != "H200-HBF-CLI" for case in cases)


def test_qwen3_moe_sweep_uses_expected_batch_profiles_by_mode_and_slo() -> None:
    cases = build_sweep_cases()

    hbm_batches_by_rank = {
        rank: tuple(case.batch_size for case in cases if case.hardware_type == "H200-HBM" and case.num_ranks == rank)
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


def test_qwen3_moe_sweep_csv_includes_explicit_script_parameters() -> None:
    required_fields = {
        "model_card_path",
        "compile_mode",
        "batch_size_semantics",
        "capacity_rule",
        "case_limit",
        "selected_case_count",
        "total_case_count",
        "max_workers",
        "worker_count_source",
        "host_logical_cpu_count",
        "slo_ms",
        "throughput_per_GPU",
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
        "nand_num_channels",
        "nand_num_plane",
        "nand_num_block",
        "nand_num_pages",
        "nand_tRead",
        "nand_tWrite",
        "nand_tErase",
        "nand_page_size_kb",
        "nand_sram_threshold_kb",
    }

    assert required_fields.issubset(set(CSV_FIELDNAMES))


def test_resolve_max_workers_defaults_to_cpu_count_bounded_by_case_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(MAX_WORKERS_ENV_VAR, raising=False)
    monkeypatch.setattr(sweep_module.os, "cpu_count", lambda: 32)

    assert resolve_max_workers(36) == 32
    assert resolve_max_workers(8) == 8


def test_resolve_case_limit_defaults_to_all_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CASE_LIMIT_ENV_VAR, raising=False)

    assert resolve_case_limit(36) == 36


def test_resolve_case_limit_uses_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CASE_LIMIT_ENV_VAR, "4")

    assert resolve_case_limit(36) == 4
    assert resolve_case_limit(2) == 2


def test_resolve_case_limit_rejects_invalid_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(CASE_LIMIT_ENV_VAR, "abc")
    with pytest.raises(ValueError, match=CASE_LIMIT_ENV_VAR):
        resolve_case_limit(36)

    monkeypatch.setenv(CASE_LIMIT_ENV_VAR, "0")
    with pytest.raises(ValueError, match=CASE_LIMIT_ENV_VAR):
        resolve_case_limit(36)


def test_resolve_max_workers_uses_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MAX_WORKERS_ENV_VAR, "12")
    monkeypatch.setattr(sweep_module.os, "cpu_count", lambda: 32)

    assert resolve_max_workers(36) == 12
    assert resolve_max_workers(4) == 4


def test_resolve_max_workers_rejects_invalid_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(MAX_WORKERS_ENV_VAR, "abc")
    with pytest.raises(ValueError, match=MAX_WORKERS_ENV_VAR):
        resolve_max_workers(36)

    monkeypatch.setenv(MAX_WORKERS_ENV_VAR, "0")
    with pytest.raises(ValueError, match=MAX_WORKERS_ENV_VAR):
        resolve_max_workers(36)


def test_qwen3_moe_sweep_trace_dir_includes_slo_segment() -> None:
    assert build_trace_dir(
        SweepCase(
            hardware_type="H200-HBM",
            num_ranks=8,
            batch_size=88,
            input_sequence_length=9400,
            output_sequence_length=600,
            slo_ms=None,
        )
    ) == (
        Path("trace/main")
        / "H200-HBM"
        / "ranks_8"
        / "slo_none"
        / "isl_9400_osl_600"
        / "bs_88"
    )

    assert build_trace_dir(
        SweepCase(
            hardware_type="H200-HBF-CSI",
            num_ranks=16,
            batch_size=2048,
            input_sequence_length=9400,
            output_sequence_length=600,
            slo_ms=100,
        )
    ) == (
        Path("trace/main")
        / "H200-HBF-CSI"
        / "ranks_16"
        / "slo_100ms"
        / "isl_9400_osl_600"
        / "bs_2048"
    )


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


def test_run_sweep_uses_process_pool_with_resolved_worker_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submitted: list[tuple[SweepCase, int]] = []
    summary_rows: list[dict[str, object]] = []
    context_max_workers: list[int] = []

    cases = [
        SweepCase(
            hardware_type="H200-HBM",
            num_ranks=8,
            batch_size=88,
            input_sequence_length=9400,
            output_sequence_length=600,
            slo_ms=None,
        ),
        SweepCase(
            hardware_type="H200-HBF-CSI",
            num_ranks=16,
            batch_size=2048,
            input_sequence_length=9400,
            output_sequence_length=600,
            slo_ms=100,
        ),
    ]

    class FakeFuture:
        def __init__(self, result: dict[str, object]) -> None:
            self._result = result

        def result(self) -> dict[str, object]:
            return self._result

    class FakeProcessPoolExecutor:
        def __init__(self, *, max_workers: int) -> None:
            context_max_workers.append(max_workers)

        def __enter__(self) -> "FakeProcessPoolExecutor":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def submit(
            self,
            fn,
            case: SweepCase,
            max_workers: int,
            selected_case_count: int,
            total_case_count: int,
        ) -> FakeFuture:
            submitted.append((case, max_workers))
            return FakeFuture(fn(case, max_workers, selected_case_count, total_case_count))

    def fake_build_result_row(
        case: SweepCase,
        max_workers: int,
        selected_case_count: int,
        total_case_count: int,
    ) -> dict[str, object]:
        return {
            "hardware_type": case.hardware_type,
            "num_ranks": case.num_ranks,
            "slo_ms": case.slo_ms,
            "input_sequence_length": case.input_sequence_length,
            "output_sequence_length": case.output_sequence_length,
            "batch_size": case.batch_size,
            "case_limit": None,
            "selected_case_count": selected_case_count,
            "total_case_count": total_case_count,
            "max_workers": max_workers,
        }

    monkeypatch.setattr(sweep_module, "build_sweep_cases", lambda: cases)
    monkeypatch.setattr(sweep_module, "resolve_case_limit", lambda case_count: case_count)
    monkeypatch.setattr(sweep_module, "resolve_max_workers", lambda case_count: 7)
    monkeypatch.setattr(sweep_module, "build_result_row", fake_build_result_row)
    monkeypatch.setattr(sweep_module, "ProcessPoolExecutor", FakeProcessPoolExecutor)
    monkeypatch.setattr(sweep_module, "as_completed", lambda futures: list(futures))
    monkeypatch.setattr(
        sweep_module,
        "write_summary_csv",
        lambda rows: summary_rows.extend(rows),
    )

    rows = sweep_module.run_sweep()

    assert context_max_workers == [7]
    assert len(submitted) == len(cases)
    assert all(max_workers == 7 for _, max_workers in submitted)
    assert all(row["selected_case_count"] == len(cases) for row in rows)
    assert all(row["total_case_count"] == len(cases) for row in rows)
    assert rows == summary_rows
