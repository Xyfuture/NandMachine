from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from scripts.deepseek_v3_sweep import (
    SEQUENCE_CASE_CONFIGS as DEEPSEEK_V3_SEQUENCE_CASE_CONFIGS,
    build_sweep_cases as build_deepseek_v3_sweep_cases,
)
from scripts.llama_405b_sweep import (
    SEQUENCE_CASE_CONFIGS as LLAMA_405B_SEQUENCE_CASE_CONFIGS,
    build_sweep_cases as build_llama_405b_sweep_cases,
)
from scripts.qwen3_coder_480b_sweep import (
    SEQUENCE_CASE_CONFIGS as QWEN3_CODER_480B_SEQUENCE_CASE_CONFIGS,
    build_sweep_cases as build_qwen3_coder_480b_sweep_cases,
)
from scripts.qwen3_moe_sweep import (
    SEQUENCE_CASE_CONFIGS as QWEN3_MOE_SEQUENCE_CASE_CONFIGS,
    build_sweep_cases as build_qwen3_moe_sweep_cases,
)


@dataclass(frozen=True)
class SweepBatchConfigExpectation:
    name: str
    sequence_case_configs: tuple[object, ...]
    build_sweep_cases: Callable[[], list[object]]
    expected_hbm_ranks: set[int]
    expected_csi_ranks_by_slo_ms: dict[int, set[int]]
    expected_cli_ranks_by_slo_ms: dict[int, set[int]]


EXPECTATIONS: tuple[SweepBatchConfigExpectation, ...] = (
    SweepBatchConfigExpectation(
        name="qwen3_moe",
        sequence_case_configs=QWEN3_MOE_SEQUENCE_CASE_CONFIGS,
        build_sweep_cases=build_qwen3_moe_sweep_cases,
        expected_hbm_ranks={4, 8, 16},
        expected_csi_ranks_by_slo_ms={50: {4, 8, 16}, 100: {4, 8, 16}},
        expected_cli_ranks_by_slo_ms={50: {4, 8, 16}, 100: {4, 8, 16}},
    ),
    SweepBatchConfigExpectation(
        name="qwen3_coder_480b",
        sequence_case_configs=QWEN3_CODER_480B_SEQUENCE_CASE_CONFIGS,
        build_sweep_cases=build_qwen3_coder_480b_sweep_cases,
        expected_hbm_ranks={8, 16},
        expected_csi_ranks_by_slo_ms={50: {8, 16}, 100: {4, 8, 16}},
        expected_cli_ranks_by_slo_ms={50: {8, 16}, 100: {4, 8, 16}},
    ),
    SweepBatchConfigExpectation(
        name="llama_405b",
        sequence_case_configs=LLAMA_405B_SEQUENCE_CASE_CONFIGS,
        build_sweep_cases=build_llama_405b_sweep_cases,
        expected_hbm_ranks={8},
        expected_csi_ranks_by_slo_ms={50: {4, 8}, 100: {4, 8}},
        expected_cli_ranks_by_slo_ms={50: {8}, 100: {4, 8}},
    ),
    SweepBatchConfigExpectation(
        name="deepseek_v3",
        sequence_case_configs=DEEPSEEK_V3_SEQUENCE_CASE_CONFIGS,
        build_sweep_cases=build_deepseek_v3_sweep_cases,
        expected_hbm_ranks={8, 16},
        expected_csi_ranks_by_slo_ms={50: {4, 8, 16}, 100: {4, 8, 16}},
        expected_cli_ranks_by_slo_ms={50: {4, 8, 16}, 100: {4, 8, 16}},
    ),
)


def _iter_configured_batch_profiles(
    sequence_case_config: object,
) -> Iterable[tuple[str, int | None, int, tuple[int, ...]]]:
    hbm_batch_sizes_by_ranks = getattr(sequence_case_config, "hbm_batch_sizes_by_ranks")
    if hbm_batch_sizes_by_ranks:
        for num_ranks, batch_sizes in hbm_batch_sizes_by_ranks.items():
            yield ("hbm_only", None, num_ranks, batch_sizes)

    csi_batch_sizes_by_ranks_by_slo_ms = getattr(
        sequence_case_config,
        "csi_batch_sizes_by_ranks_by_slo_ms",
    )
    if csi_batch_sizes_by_ranks_by_slo_ms:
        for slo_ms, batch_sizes_by_ranks in csi_batch_sizes_by_ranks_by_slo_ms.items():
            for num_ranks, batch_sizes in batch_sizes_by_ranks.items():
                yield ("csi", slo_ms, num_ranks, batch_sizes)

    cli_batch_sizes_by_ranks_by_slo_ms = getattr(
        sequence_case_config,
        "cli_batch_sizes_by_ranks_by_slo_ms",
    )
    if cli_batch_sizes_by_ranks_by_slo_ms:
        for slo_ms, batch_sizes_by_ranks in cli_batch_sizes_by_ranks_by_slo_ms.items():
            for num_ranks, batch_sizes in batch_sizes_by_ranks.items():
                yield ("cli", slo_ms, num_ranks, batch_sizes)


def test_sweep_batch_sizes_are_never_smaller_than_rank_count() -> None:
    for expectation in EXPECTATIONS:
        for sequence_case_config in expectation.sequence_case_configs:
            sequence_pair = (
                getattr(sequence_case_config, "input_sequence_length"),
                getattr(sequence_case_config, "output_sequence_length"),
            )
            for mode, slo_ms, num_ranks, batch_sizes in _iter_configured_batch_profiles(
                sequence_case_config
            ):
                assert batch_sizes, (
                    f"{expectation.name} has an empty batch profile for {mode}, "
                    f"slo_ms={slo_ms}, num_ranks={num_ranks}, sequence_pair={sequence_pair}"
                )
                assert all(batch_size >= num_ranks for batch_size in batch_sizes), (
                    f"{expectation.name} has batch_size < num_ranks for {mode}, "
                    f"slo_ms={slo_ms}, num_ranks={num_ranks}, "
                    f"sequence_pair={sequence_pair}, batch_sizes={batch_sizes}"
                )


def test_sweep_sequence_buckets_cover_expected_rank_profiles_and_hardware_modes() -> None:
    for expectation in EXPECTATIONS:
        cases = expectation.build_sweep_cases()

        for sequence_case_config in expectation.sequence_case_configs:
            sequence_pair = (
                getattr(sequence_case_config, "input_sequence_length"),
                getattr(sequence_case_config, "output_sequence_length"),
            )

            assert set(
                (getattr(sequence_case_config, "hbm_batch_sizes_by_ranks") or {}).keys()
            ) == expectation.expected_hbm_ranks
            assert {
                slo_ms: set(batch_sizes_by_ranks.keys())
                for slo_ms, batch_sizes_by_ranks in (
                    getattr(sequence_case_config, "csi_batch_sizes_by_ranks_by_slo_ms") or {}
                ).items()
            } == expectation.expected_csi_ranks_by_slo_ms
            assert {
                slo_ms: set(batch_sizes_by_ranks.keys())
                for slo_ms, batch_sizes_by_ranks in (
                    getattr(sequence_case_config, "cli_batch_sizes_by_ranks_by_slo_ms") or {}
                ).items()
            } == expectation.expected_cli_ranks_by_slo_ms

            assert any(
                case.hardware_type == "H200-HBM"
                and case.input_sequence_length == sequence_pair[0]
                and case.output_sequence_length == sequence_pair[1]
                for case in cases
            )
            assert any(
                case.hardware_type == "H200-HBF-CSI"
                and case.input_sequence_length == sequence_pair[0]
                and case.output_sequence_length == sequence_pair[1]
                for case in cases
            )
            assert any(
                case.hardware_type == "H200-HBF-CLI"
                and case.input_sequence_length == sequence_pair[0]
                and case.output_sequence_length == sequence_pair[1]
                for case in cases
            )
