from __future__ import annotations

import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import scripts.qwen3_coder_480b_sweep as base_sweep
import torch


HardwareSpec = base_sweep.HardwareSpec
RuntimeSpec = base_sweep.RuntimeSpec
SweepCase = base_sweep.SweepCase

MODEL_CARD_PATH = base_sweep.MODEL_CARD_PATH
TRACE_ROOT = base_sweep.TRACE_ROOT
FULL_TRACE_FILE_NAME = base_sweep.FULL_TRACE_FILE_NAME
CONFIG_FILE_NAME = base_sweep.CONFIG_FILE_NAME
COMPILE_MODE = base_sweep.COMPILE_MODE
INTERCONNECT_TOPOLOGY = base_sweep.INTERCONNECT_TOPOLOGY
DEFAULT_MAX_WORKERS_CPU_DIVISOR = base_sweep.DEFAULT_MAX_WORKERS_CPU_DIVISOR
TORCH_NUM_INTEROP_THREADS = base_sweep.TORCH_NUM_INTEROP_THREADS
WEIGHT_BITS = base_sweep.WEIGHT_BITS
ACTIVATION_BITS = base_sweep.ACTIVATION_BITS
KV_CACHE_BITS = base_sweep.KV_CACHE_BITS
KV_BLOCK_SIZE_BYTES = base_sweep.KV_BLOCK_SIZE_BYTES
BASE_NAND_CONFIG = base_sweep.BASE_NAND_CONFIG
_SINGLE_THREAD_RUNTIME_ENV_DEFAULTS = base_sweep._SINGLE_THREAD_RUNTIME_ENV_DEFAULTS

SWEEP_NAME = "qwen3_coder_480b_cli_stack_sweep"
MAX_WORKERS_ENV_VAR = "QWEN3_CODER_480B_CLI_STACK_SWEEP_MAX_WORKERS"
CASE_LIMIT_ENV_VAR = "QWEN3_CODER_480B_CLI_STACK_SWEEP_CASE_LIMIT"
CSV_FIELDNAMES = list(base_sweep.CSV_FIELDNAMES)

WEIGHT_FOOTPRINT_GIB = 960
FULL_MODEL_KV_BYTES_PER_TOKEN = 253952
CLI_GIB_PER_HBF_STACK_BY_SLO_MS = {
    50: 40,
    100: 80,
}
H200_TOTAL_BANDWIDTH_GBPS = 4800.0
H200_PER_STACK_BANDWIDTH_GBPS = 800.0
H200_BASE_HBM_STACK_COUNT = 6
CLI_ALLOWED_RANKS = (4, 8, 16)
CLI_ALLOWED_SLO_MS = (50, 100)

CLI_HBM1_HBF5 = "H200-HBF-CLI-HBM1-HBF5"
CLI_HBM2_HBF4 = "H200-HBF-CLI-HBM2-HBF4"
CLI_HBM3_HBF3 = "H200-HBF-CLI-HBM3-HBF3"
CLI_HBM4_HBF2 = "H200-HBF-CLI-HBM4-HBF2"
CLI_HBM5_HBF1 = "H200-HBF-CLI-HBM5-HBF1"


@dataclass(frozen=True)
class CLIRatioSpec:
    hardware_type: str
    hbm_stacks: int
    hbf_stacks: int
    per_card_capacity_gib_by_slo_ms: dict[int, int]
    hbf_bandwidth_gbps: float
    hbm_bandwidth_gbps: float


@dataclass(frozen=True)
class SequenceCaseConfig:
    input_sequence_length: int
    output_sequence_length: int
    cli_batch_sizes_by_hardware_type_by_slo_ms: dict[
        str, dict[int, dict[int, tuple[int, ...]]]
    ]


def build_trace_root(run_tag: str) -> Path:
    return TRACE_ROOT / f"{SWEEP_NAME}_{run_tag}"


def build_summary_csv_path(run_tag: str) -> Path:
    return TRACE_ROOT / f"{SWEEP_NAME}_summary_{run_tag}.csv"


CLI_RATIO_SPECS: tuple[CLIRatioSpec, ...] = (
    CLIRatioSpec(
        hardware_type=CLI_HBM1_HBF5,
        hbm_stacks=1,
        hbf_stacks=5,
        per_card_capacity_gib_by_slo_ms={50: 200, 100: 400},
        hbf_bandwidth_gbps=4000.0,
        hbm_bandwidth_gbps=800.0,
    ),
    CLIRatioSpec(
        hardware_type=CLI_HBM2_HBF4,
        hbm_stacks=2,
        hbf_stacks=4,
        per_card_capacity_gib_by_slo_ms={50: 160, 100: 320},
        hbf_bandwidth_gbps=3200.0,
        hbm_bandwidth_gbps=1600.0,
    ),
    CLIRatioSpec(
        hardware_type=CLI_HBM3_HBF3,
        hbm_stacks=3,
        hbf_stacks=3,
        per_card_capacity_gib_by_slo_ms={50: 120, 100: 240},
        hbf_bandwidth_gbps=2400.0,
        hbm_bandwidth_gbps=2400.0,
    ),
    CLIRatioSpec(
        hardware_type=CLI_HBM4_HBF2,
        hbm_stacks=4,
        hbf_stacks=2,
        per_card_capacity_gib_by_slo_ms={50: 80, 100: 160},
        hbf_bandwidth_gbps=1600.0,
        hbm_bandwidth_gbps=3200.0,
    ),
    CLIRatioSpec(
        hardware_type=CLI_HBM5_HBF1,
        hbm_stacks=5,
        hbf_stacks=1,
        per_card_capacity_gib_by_slo_ms={50: 40, 100: 80},
        hbf_bandwidth_gbps=800.0,
        hbm_bandwidth_gbps=4000.0,
    ),
)

CLI_RATIO_SPECS_BY_HARDWARE_TYPE = {
    cli_ratio_spec.hardware_type: cli_ratio_spec for cli_ratio_spec in CLI_RATIO_SPECS
}


# Model memory note:
# - Weight footprint: 960 GiB
# - Full-model KV cache per token: 253952 B = 248 KiB/token
# - CLI synthetic capacity per HBF stack: 50ms = 40 GiB, 100ms = 80 GiB
# - Scanned CLI ratios: (1,5), (2,4), (3,3), (4,2), (5,1)
SEQUENCE_CASE_CONFIGS: tuple[SequenceCaseConfig, ...] = (
    SequenceCaseConfig(
        input_sequence_length=9400,
        output_sequence_length=600,
        cli_batch_sizes_by_hardware_type_by_slo_ms={
            CLI_HBM1_HBF5: {
                50: {
                    8: (264, 256),  # total capacity: 1600 GiB, remaining for KV: 640 GiB, max batch size: 270
                    16: (944, 512),  # total capacity: 3200 GiB, remaining for KV: 2240 GiB, max batch size: 947
                },
                100: {
                    4: (268, 256),  # total capacity: 1600 GiB, remaining for KV: 640 GiB, max batch size: 270
                    8: (944, 512),  # total capacity: 3200 GiB, remaining for KV: 2240 GiB, max batch size: 947
                    16: (2288, 2048),  # total capacity: 6400 GiB, remaining for KV: 5440 GiB, max batch size: 2300
                },
            },
            CLI_HBM2_HBF4: {
                50: {
                    8: (128, 64),  # total capacity: 1280 GiB, remaining for KV: 320 GiB, max batch size: 135
                    16: (672, 512),  # total capacity: 2560 GiB, remaining for KV: 1600 GiB, max batch size: 676
                },
                100: {
                    4: (132, 128),  # total capacity: 1280 GiB, remaining for KV: 320 GiB, max batch size: 135
                    8: (672, 512),  # total capacity: 2560 GiB, remaining for KV: 1600 GiB, max batch size: 676
                    16: (1744, 1024),  # total capacity: 5120 GiB, remaining for KV: 4160 GiB, max batch size: 1758
                },
            },
            CLI_HBM3_HBF3: {
                50: {
                    16: (400, 256),  # total capacity: 1920 GiB, remaining for KV: 960 GiB, max batch size: 405
                },
                100: {
                    8: (400, 256),  # total capacity: 1920 GiB, remaining for KV: 960 GiB, max batch size: 405
                    16: (1216, 1024),  # total capacity: 3840 GiB, remaining for KV: 2880 GiB, max batch size: 1217
                },
            },
            CLI_HBM4_HBF2: {
                50: {
                    16: (128, 64),  # total capacity: 1280 GiB, remaining for KV: 320 GiB, max batch size: 135
                },
                100: {
                    8: (128, 64),  # total capacity: 1280 GiB, remaining for KV: 320 GiB, max batch size: 135
                    16: (672, 512),  # total capacity: 2560 GiB, remaining for KV: 1600 GiB, max batch size: 676
                },
            },
            CLI_HBM5_HBF1: {
                100: {
                    16: (128, 64),  # total capacity: 1280 GiB, remaining for KV: 320 GiB, max batch size: 135
                },
            },
        },
    ),
    SequenceCaseConfig(
        input_sequence_length=20000,
        output_sequence_length=1000,
        cli_batch_sizes_by_hardware_type_by_slo_ms={
            CLI_HBM1_HBF5: {
                50: {
                    8: (120, 64),  # total capacity: 1600 GiB, remaining for KV: 640 GiB, max batch size: 128
                    16: (448, 256),  # total capacity: 3200 GiB, remaining for KV: 2240 GiB, max batch size: 451
                },
                100: {
                    4: (124, 64),  # total capacity: 1600 GiB, remaining for KV: 640 GiB, max batch size: 128
                    8: (448, 256),  # total capacity: 3200 GiB, remaining for KV: 2240 GiB, max batch size: 451
                    16: (1088, 1024),  # total capacity: 6400 GiB, remaining for KV: 5440 GiB, max batch size: 1095
                },
            },
            CLI_HBM2_HBF4: {
                50: {
                    8: (56, 32),  # total capacity: 1280 GiB, remaining for KV: 320 GiB, max batch size: 64
                    16: (320, 256),  # total capacity: 2560 GiB, remaining for KV: 1600 GiB, max batch size: 322
                },
                100: {
                    4: (60, 32),  # total capacity: 1280 GiB, remaining for KV: 320 GiB, max batch size: 64
                    8: (320, 256),  # total capacity: 2560 GiB, remaining for KV: 1600 GiB, max batch size: 322
                    16: (832, 512),  # total capacity: 5120 GiB, remaining for KV: 4160 GiB, max batch size: 837
                },
            },
            CLI_HBM3_HBF3: {
                50: {
                    16: (192, 128),  # total capacity: 1920 GiB, remaining for KV: 960 GiB, max batch size: 193
                },
                100: {
                    8: (192, 128),  # total capacity: 1920 GiB, remaining for KV: 960 GiB, max batch size: 193
                    16: (576, 512),  # total capacity: 3840 GiB, remaining for KV: 2880 GiB, max batch size: 579
                },
            },
            CLI_HBM4_HBF2: {
                50: {
                    16: (48, 32),  # total capacity: 1280 GiB, remaining for KV: 320 GiB, max batch size: 64
                },
                100: {
                    8: (56, 32),  # total capacity: 1280 GiB, remaining for KV: 320 GiB, max batch size: 64
                    16: (320, 256),  # total capacity: 2560 GiB, remaining for KV: 1600 GiB, max batch size: 322
                },
            },
            CLI_HBM5_HBF1: {
                100: {
                    16: (48, 32),  # total capacity: 1280 GiB, remaining for KV: 320 GiB, max batch size: 64
                },
            },
        },
    ),
)


def _largest_power_of_two_less_than_or_none(value: int) -> int | None:
    if value <= 1:
        return None
    power = 1 << ((value - 1).bit_length() - 1)
    if power == value:
        power >>= 1
    if power <= 0:
        return None
    return power


def _derive_expected_batch_sizes_or_none(
    *,
    input_sequence_length: int,
    output_sequence_length: int,
    num_ranks: int,
    hbf_stacks: int,
    slo_ms: int,
) -> tuple[int, ...] | None:
    if slo_ms not in CLI_GIB_PER_HBF_STACK_BY_SLO_MS:
        raise ValueError(f"Unsupported slo_ms: {slo_ms}")

    total_capacity_gib = (
        num_ranks * hbf_stacks * CLI_GIB_PER_HBF_STACK_BY_SLO_MS[slo_ms]
    )
    remaining_for_kv_gib = total_capacity_gib - WEIGHT_FOOTPRINT_GIB
    if remaining_for_kv_gib <= 0:
        return None

    total_sequence_length = input_sequence_length + output_sequence_length
    max_batch_size = int(
        (remaining_for_kv_gib * 1024**3)
        // (total_sequence_length * FULL_MODEL_KV_BYTES_PER_TOKEN)
    )
    first_batch_size = max_batch_size - 1
    while first_batch_size >= num_ranks and first_batch_size % num_ranks != 0:
        first_batch_size -= 1
    if first_batch_size < num_ranks:
        return None

    batch_sizes = [first_batch_size]
    second_batch_size = _largest_power_of_two_less_than_or_none(first_batch_size)
    while second_batch_size is not None and second_batch_size > 0:
        batch_sizes.append(second_batch_size)
        if second_batch_size % num_ranks == 0:
            break
        second_batch_size //= 2

    deduplicated_batch_sizes: list[int] = []
    for batch_size in batch_sizes:
        if batch_size < num_ranks:
            continue
        if batch_size not in deduplicated_batch_sizes:
            deduplicated_batch_sizes.append(batch_size)

    if not deduplicated_batch_sizes:
        return None
    return tuple(deduplicated_batch_sizes)


def get_cli_ratio_spec_or_raise(hardware_type: str) -> CLIRatioSpec:
    try:
        return CLI_RATIO_SPECS_BY_HARDWARE_TYPE[hardware_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported CLI hardware_type: {hardware_type}") from exc


def validate_cli_ratio_specs_or_raise() -> None:
    if not CLI_RATIO_SPECS:
        raise ValueError("CLI_RATIO_SPECS must not be empty")

    seen_hardware_types: set[str] = set()
    seen_ratios: set[tuple[int, int]] = set()
    for cli_ratio_spec in CLI_RATIO_SPECS:
        if cli_ratio_spec.hardware_type in seen_hardware_types:
            raise ValueError(
                f"Duplicate CLI hardware_type: {cli_ratio_spec.hardware_type}"
            )
        seen_hardware_types.add(cli_ratio_spec.hardware_type)

        ratio = (cli_ratio_spec.hbm_stacks, cli_ratio_spec.hbf_stacks)
        if ratio in seen_ratios:
            raise ValueError(f"Duplicate CLI ratio: {ratio}")
        seen_ratios.add(ratio)

        if cli_ratio_spec.hbm_stacks + cli_ratio_spec.hbf_stacks != H200_BASE_HBM_STACK_COUNT:
            raise ValueError(
                "CLI ratio must partition the six H200 stacks, "
                f"got hbm_stacks={cli_ratio_spec.hbm_stacks}, "
                f"hbf_stacks={cli_ratio_spec.hbf_stacks}"
            )

        expected_capacity = {
            slo_ms: cli_ratio_spec.hbf_stacks * gib_per_hbf_stack
            for slo_ms, gib_per_hbf_stack in CLI_GIB_PER_HBF_STACK_BY_SLO_MS.items()
        }
        if cli_ratio_spec.per_card_capacity_gib_by_slo_ms != expected_capacity:
            raise ValueError(
                "CLI per-card capacity does not match HBF stack count, "
                f"got {cli_ratio_spec.per_card_capacity_gib_by_slo_ms}, "
                f"expected {expected_capacity}"
            )

        expected_hbf_bandwidth_gbps = (
            cli_ratio_spec.hbf_stacks * H200_PER_STACK_BANDWIDTH_GBPS
        )
        if cli_ratio_spec.hbf_bandwidth_gbps != expected_hbf_bandwidth_gbps:
            raise ValueError(
                "CLI HBF bandwidth does not match HBF stack count, "
                f"got {cli_ratio_spec.hbf_bandwidth_gbps}, "
                f"expected {expected_hbf_bandwidth_gbps}"
            )

        expected_hbm_bandwidth_gbps = (
            H200_TOTAL_BANDWIDTH_GBPS - expected_hbf_bandwidth_gbps
        )
        if cli_ratio_spec.hbm_bandwidth_gbps != expected_hbm_bandwidth_gbps:
            raise ValueError(
                "CLI HBM bandwidth does not match residual bandwidth, "
                f"got {cli_ratio_spec.hbm_bandwidth_gbps}, "
                f"expected {expected_hbm_bandwidth_gbps}"
            )


def validate_sequence_case_configs_or_raise() -> None:
    known_hardware_types = {
        cli_ratio_spec.hardware_type for cli_ratio_spec in CLI_RATIO_SPECS
    }
    for sequence_case_config in SEQUENCE_CASE_CONFIGS:
        actual_hardware_types = set(
            sequence_case_config.cli_batch_sizes_by_hardware_type_by_slo_ms.keys()
        )
        unknown_hardware_types = actual_hardware_types - known_hardware_types
        if unknown_hardware_types:
            raise ValueError(
                "SequenceCaseConfig includes unknown CLI hardware types: "
                f"{sorted(unknown_hardware_types)}"
            )

        for cli_ratio_spec in CLI_RATIO_SPECS:
            expected_batches_by_slo_ms: dict[int, dict[int, tuple[int, ...]]] = {}
            for slo_ms in CLI_ALLOWED_SLO_MS:
                expected_batches_by_ranks: dict[int, tuple[int, ...]] = {}
                for num_ranks in CLI_ALLOWED_RANKS:
                    expected_batch_sizes = _derive_expected_batch_sizes_or_none(
                        input_sequence_length=sequence_case_config.input_sequence_length,
                        output_sequence_length=sequence_case_config.output_sequence_length,
                        num_ranks=num_ranks,
                        hbf_stacks=cli_ratio_spec.hbf_stacks,
                        slo_ms=slo_ms,
                    )
                    if expected_batch_sizes is not None:
                        expected_batches_by_ranks[num_ranks] = expected_batch_sizes
                if expected_batches_by_ranks:
                    expected_batches_by_slo_ms[slo_ms] = expected_batches_by_ranks

            actual_batches_by_slo_ms = (
                sequence_case_config.cli_batch_sizes_by_hardware_type_by_slo_ms.get(
                    cli_ratio_spec.hardware_type
                )
                or {}
            )
            if actual_batches_by_slo_ms != expected_batches_by_slo_ms:
                raise ValueError(
                    "Static CLI batch table does not match configured rules for "
                    f"hardware_type={cli_ratio_spec.hardware_type}, "
                    f"sequence=({sequence_case_config.input_sequence_length}, "
                    f"{sequence_case_config.output_sequence_length}). "
                    f"got={actual_batches_by_slo_ms}, "
                    f"expected={expected_batches_by_slo_ms}"
                )


validate_cli_ratio_specs_or_raise()
validate_sequence_case_configs_or_raise()

HARDWARE_SPECS: tuple[HardwareSpec, ...] = tuple(
    HardwareSpec(
        hardware_type=cli_ratio_spec.hardware_type,
        device_name="H200_SXM",
        memory_architecture={
            "mode": "cli",
            "hbm_stacks": cli_ratio_spec.hbm_stacks,
            "hbf_stacks": cli_ratio_spec.hbf_stacks,
        },
        memory_backend="nand",
    )
    for cli_ratio_spec in CLI_RATIO_SPECS
)


def resolve_worker_count_source() -> str:
    if os.getenv(MAX_WORKERS_ENV_VAR) is not None:
        return "env"
    return f"cpu_div_{DEFAULT_MAX_WORKERS_CPU_DIVISOR}"


def resolve_case_limit(case_count: int) -> int:
    if case_count <= 0:
        raise ValueError(f"case_count must be > 0, got {case_count}")

    env_value = os.getenv(CASE_LIMIT_ENV_VAR)
    if env_value is None:
        return case_count

    try:
        configured_case_limit = int(env_value)
    except ValueError as exc:
        raise ValueError(
            f"{CASE_LIMIT_ENV_VAR} must be an integer, got {env_value!r}"
        ) from exc

    if configured_case_limit <= 0:
        raise ValueError(
            f"{CASE_LIMIT_ENV_VAR} must be > 0, got {configured_case_limit}"
        )

    return min(configured_case_limit, case_count)


def resolve_max_workers(case_count: int) -> int:
    if case_count <= 0:
        raise ValueError(f"case_count must be > 0, got {case_count}")

    env_value = os.getenv(MAX_WORKERS_ENV_VAR)
    if env_value is not None:
        try:
            configured_max_workers = int(env_value)
        except ValueError as exc:
            raise ValueError(
                f"{MAX_WORKERS_ENV_VAR} must be an integer, got {env_value!r}"
            ) from exc
        if configured_max_workers <= 0:
            raise ValueError(
                f"{MAX_WORKERS_ENV_VAR} must be > 0, got {configured_max_workers}"
            )
    else:
        cpu_count = os.cpu_count() or 1
        configured_max_workers = max(1, cpu_count // DEFAULT_MAX_WORKERS_CPU_DIVISOR)

    return min(configured_max_workers, case_count)


def build_sweep_cases() -> list[SweepCase]:
    cases: list[SweepCase] = []
    for hardware_spec in HARDWARE_SPECS:
        for sequence_case_config in SEQUENCE_CASE_CONFIGS:
            batch_sizes_by_slo_ms = (
                sequence_case_config.cli_batch_sizes_by_hardware_type_by_slo_ms.get(
                    hardware_spec.hardware_type
                )
            )
            if not batch_sizes_by_slo_ms:
                continue

            for slo_ms, batch_sizes_by_ranks in batch_sizes_by_slo_ms.items():
                for num_ranks, batch_sizes in batch_sizes_by_ranks.items():
                    for batch_size in batch_sizes:
                        cases.append(
                            SweepCase(
                                hardware_type=hardware_spec.hardware_type,
                                num_ranks=num_ranks,
                                batch_size=batch_size,
                                input_sequence_length=sequence_case_config.input_sequence_length,
                                output_sequence_length=sequence_case_config.output_sequence_length,
                                slo_ms=slo_ms,
                            )
                        )
    return cases


def get_hardware_spec_or_raise(hardware_type: str) -> HardwareSpec:
    for hardware_spec in HARDWARE_SPECS:
        if hardware_spec.hardware_type == hardware_type:
            return hardware_spec
    raise ValueError(f"Unsupported hardware_type: {hardware_type}")


def build_nand_config(hardware_spec: HardwareSpec):
    cli_ratio_spec = get_cli_ratio_spec_or_raise(hardware_spec.hardware_type)
    expected_memory_architecture = {
        "mode": "cli",
        "hbm_stacks": cli_ratio_spec.hbm_stacks,
        "hbf_stacks": cli_ratio_spec.hbf_stacks,
    }
    if hardware_spec.memory_architecture != expected_memory_architecture:
        raise ValueError(
            "HardwareSpec memory_architecture does not match CLI ratio spec, "
            f"got {hardware_spec.memory_architecture}, "
            f"expected {expected_memory_architecture}"
        )
    return base_sweep.build_nand_config(hardware_spec)


def build_runtime_spec(hardware_spec: HardwareSpec, nand_config) -> RuntimeSpec:
    cli_ratio_spec = get_cli_ratio_spec_or_raise(hardware_spec.hardware_type)
    runtime_spec = base_sweep.build_runtime_spec(hardware_spec, nand_config)
    if runtime_spec.sim_hbm_bandwidth_GBps != cli_ratio_spec.hbm_bandwidth_gbps:
        raise ValueError(
            "Runtime HBM bandwidth does not match CLI ratio spec, "
            f"got {runtime_spec.sim_hbm_bandwidth_GBps}, "
            f"expected {cli_ratio_spec.hbm_bandwidth_gbps}"
        )
    if runtime_spec.derived_hbf_bandwidth_GBps != cli_ratio_spec.hbf_bandwidth_gbps:
        raise ValueError(
            "Runtime HBF bandwidth does not match CLI ratio spec, "
            f"got {runtime_spec.derived_hbf_bandwidth_GBps}, "
            f"expected {cli_ratio_spec.hbf_bandwidth_gbps}"
        )
    return runtime_spec


def build_trace_dir(case: SweepCase, run_tag: str) -> Path:
    return (
        build_trace_root(run_tag)
        / case.hardware_type
        / f"ranks_{case.num_ranks}"
        / f"slo_{case.slo_ms}ms"
        / f"isl_{case.input_sequence_length}_osl_{case.output_sequence_length}"
        / f"bs_{case.batch_size}"
    )


def write_json_file(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def build_result_row(
    case: SweepCase,
    max_workers: int,
    selected_case_count: int,
    total_case_count: int,
    run_tag: str,
) -> dict[str, object]:
    if max_workers <= 0:
        raise ValueError(f"max_workers must be > 0, got {max_workers}")
    if selected_case_count <= 0:
        raise ValueError(
            f"selected_case_count must be > 0, got {selected_case_count}"
        )
    if total_case_count <= 0:
        raise ValueError(f"total_case_count must be > 0, got {total_case_count}")
    if selected_case_count > total_case_count:
        raise ValueError(
            "selected_case_count must be <= total_case_count, "
            f"got selected_case_count={selected_case_count}, total_case_count={total_case_count}"
        )

    hardware_spec = get_hardware_spec_or_raise(case.hardware_type)
    cli_ratio_spec = get_cli_ratio_spec_or_raise(case.hardware_type)
    trace_root = build_trace_root(run_tag)
    summary_csv_path = build_summary_csv_path(run_tag)
    model_card = base_sweep.load_model_card_or_raise()
    raw_model_config = base_sweep.build_raw_model_config(deepcopy(model_card))
    model_config = base_sweep.Qwen3MoEModelConfig.from_config(raw_model_config)

    if not isinstance(model_config.num_hidden_layers, int):
        raise TypeError(
            "model_config.num_hidden_layers must be an int, "
            f"got {type(model_config.num_hidden_layers).__name__}"
        )
    if model_config.num_hidden_layers <= 0:
        raise ValueError(
            f"model_config.num_hidden_layers must be > 0, got {model_config.num_hidden_layers}"
        )

    parallel_config = base_sweep.build_parallel_config(case.num_ranks)
    nand_config = build_nand_config(hardware_spec)
    runtime_spec = build_runtime_spec(hardware_spec, nand_config)
    inference_config = base_sweep.build_inference_config(
        case,
        parallel_config,
        hardware_spec.memory_backend,
    )
    macro_op_list = base_sweep.build_macro_op_list(
        raw_model_config,
        model_config,
        nand_config,
        inference_config,
        parallel_config,
    )

    trace_dir = build_trace_dir(case, run_tag)
    trace_path = trace_dir / FULL_TRACE_FILE_NAME
    sim_result = base_sweep.run_macro_ops_with_trace(
        nand_config,
        macro_op_list,
        trace_path=trace_path,
        device_name=hardware_spec.device_name,
        interconnect_topology=base_sweep.resolve_interconnect_topology_or_raise(
            INTERCONNECT_TOPOLOGY
        ),
        compile_mode=COMPILE_MODE,
        hbm_bandwidth_GBps=runtime_spec.sim_hbm_bandwidth_GBps,
    )

    layer_latency_ns = int(sim_result["time_ns"])
    model_latency_ns = layer_latency_ns * model_config.num_hidden_layers
    if model_latency_ns <= 0:
        raise ValueError(f"model_latency_ns must be > 0, got {model_latency_ns}")
    model_throughput_tokens_per_sec = case.batch_size * 1e9 / model_latency_ns
    throughput_per_gpu = model_throughput_tokens_per_sec / case.num_ranks
    host_logical_cpu_count = os.cpu_count() or 1

    row = {
        "hardware_type": hardware_spec.hardware_type,
        "device_name": hardware_spec.device_name,
        "model_card_path": str(MODEL_CARD_PATH),
        "compile_mode": COMPILE_MODE,
        "batch_size_semantics": "global",
        "interconnect_topology": INTERCONNECT_TOPOLOGY,
        "case_limit": os.getenv(CASE_LIMIT_ENV_VAR),
        "selected_case_count": selected_case_count,
        "total_case_count": total_case_count,
        "max_workers": max_workers,
        "worker_count_source": resolve_worker_count_source(),
        "host_logical_cpu_count": host_logical_cpu_count,
        "memory_architecture_mode": runtime_spec.normalized_architecture["mode"],
        "effective_hbm_stacks": runtime_spec.normalized_architecture["effective_hbm_stacks"],
        "effective_hbf_stacks": runtime_spec.normalized_architecture["effective_hbf_stacks"],
        "memory_backend": hardware_spec.memory_backend,
        "num_ranks": case.num_ranks,
        "attn_dp_size": parallel_config.attn_dp_size,
        "attn_tp_size": parallel_config.attn_tp_size,
        "ffn_tp_size": parallel_config.ffn_tp_size,
        "ffn_ep_size": parallel_config.ffn_ep_size,
        "slo_ms": case.slo_ms,
        "batch_size": case.batch_size,
        "input_sequence_length": case.input_sequence_length,
        "output_sequence_length": case.output_sequence_length,
        "weight_bits": WEIGHT_BITS,
        "activation_bits": ACTIVATION_BITS,
        "kv_cache_bits": KV_CACHE_BITS,
        "kv_block_size_bytes": KV_BLOCK_SIZE_BYTES,
        "omp_num_threads": os.environ["OMP_NUM_THREADS"],
        "openblas_num_threads": os.environ["OPENBLAS_NUM_THREADS"],
        "mkl_num_threads": os.environ["MKL_NUM_THREADS"],
        "numexpr_num_threads": os.environ["NUMEXPR_NUM_THREADS"],
        "blis_num_threads": os.environ["BLIS_NUM_THREADS"],
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": TORCH_NUM_INTEROP_THREADS,
        "nand_num_channels": nand_config.num_channels,
        "nand_num_plane": nand_config.num_plane,
        "nand_num_block": nand_config.num_block,
        "nand_num_pages": nand_config.num_pages,
        "nand_tRead": nand_config.tRead,
        "nand_tWrite": nand_config.tWrite,
        "nand_tErase": nand_config.tErase,
        "nand_page_size_kb": nand_config.page_size,
        "nand_sram_threshold_kb": nand_config.sram_threshold,
        "sim_hbm_bandwidth_GBps": runtime_spec.sim_hbm_bandwidth_GBps,
        "derived_hbf_bandwidth_GBps": runtime_spec.derived_hbf_bandwidth_GBps,
        "macro_op_count": len(macro_op_list),
        "layer_latency_ns": layer_latency_ns,
        "model_latency_ns": model_latency_ns,
        "model_throughput_tokens_per_sec": model_throughput_tokens_per_sec,
        "throughput_per_GPU": throughput_per_gpu,
        "trace_path": sim_result["trace_path"],
    }

    config_payload = {
        "script_config": {
            "model_card_path": str(MODEL_CARD_PATH),
            "trace_root": str(trace_root),
            "summary_csv_path": str(summary_csv_path),
            "full_trace_file_name": FULL_TRACE_FILE_NAME,
            "config_file_name": CONFIG_FILE_NAME,
            "compile_mode": COMPILE_MODE,
            "batch_size_semantics": "global",
            "case_limit": os.getenv(CASE_LIMIT_ENV_VAR),
            "case_limit_env_var": CASE_LIMIT_ENV_VAR,
            "selected_case_count": selected_case_count,
            "total_case_count": total_case_count,
            "max_workers": max_workers,
            "max_workers_env_var": MAX_WORKERS_ENV_VAR,
            "worker_count_source": resolve_worker_count_source(),
            "host_logical_cpu_count": host_logical_cpu_count,
            "default_max_workers_cpu_divisor": DEFAULT_MAX_WORKERS_CPU_DIVISOR,
            "weight_bits": WEIGHT_BITS,
            "activation_bits": ACTIVATION_BITS,
            "kv_cache_bits": KV_CACHE_BITS,
            "kv_block_size_bytes": KV_BLOCK_SIZE_BYTES,
            "runtime_thread_env": dict(_SINGLE_THREAD_RUNTIME_ENV_DEFAULTS),
            "resolved_runtime_thread_env": {
                env_name: os.environ[env_name]
                for env_name in _SINGLE_THREAD_RUNTIME_ENV_DEFAULTS
            },
            "torch_runtime": {
                "torch_num_threads": torch.get_num_threads(),
                "torch_num_interop_threads": TORCH_NUM_INTEROP_THREADS,
            },
            "base_nand_config": BASE_NAND_CONFIG,
            "cli_ratio_specs": [asdict(cli_ratio_spec) for cli_ratio_spec in CLI_RATIO_SPECS],
            "sequence_case_configs": [
                asdict(sequence_case_config) for sequence_case_config in SEQUENCE_CASE_CONFIGS
            ],
        },
        "model_config": asdict(model_config),
        "hardware_spec": asdict(hardware_spec),
        "cli_ratio_spec": asdict(cli_ratio_spec),
        "sweep_case": asdict(case),
        "parallel_config": asdict(parallel_config),
        "inference_config": asdict(inference_config),
        "nand_config": asdict(nand_config),
        "runtime_spec": {
            "sim_hbm_bandwidth_GBps": runtime_spec.sim_hbm_bandwidth_GBps,
            "derived_hbf_bandwidth_GBps": runtime_spec.derived_hbf_bandwidth_GBps,
            "normalized_architecture": runtime_spec.normalized_architecture,
        },
        "simulation_result": {
            "layer_latency_ns": layer_latency_ns,
            "model_latency_ns": model_latency_ns,
            "model_throughput_tokens_per_sec": model_throughput_tokens_per_sec,
            "throughput_per_GPU": throughput_per_gpu,
            "macro_op_count": len(macro_op_list),
            "trace_path": sim_result["trace_path"],
            "trace_event_count": sim_result["trace_event_count"],
            "trace_complete_event_count": sim_result["trace_complete_event_count"],
        },
    }
    write_json_file(trace_dir / CONFIG_FILE_NAME, config_payload)

    trace_file = Path(str(sim_result["trace_path"]))
    if not trace_file.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_file}")

    return row


def write_summary_csv(rows: list[dict[str, object]], summary_csv_path: Path) -> None:
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_sweep(run_tag: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    all_cases = build_sweep_cases()
    if not all_cases:
        raise ValueError("Sweep cases must not be empty")
    total_case_count = len(all_cases)
    case_limit = resolve_case_limit(total_case_count)
    cases = all_cases[:case_limit]
    selected_case_count = len(cases)
    max_workers = resolve_max_workers(selected_case_count)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_case = {
            executor.submit(
                build_result_row,
                case,
                max_workers,
                selected_case_count,
                total_case_count,
                run_tag,
            ): case
            for case in cases
        }
        for future in as_completed(future_to_case):
            rows.append(future.result())

    rows.sort(
        key=lambda row: (
            str(row["hardware_type"]),
            int(row["num_ranks"]),
            int(row["slo_ms"]),
            int(row["input_sequence_length"]),
            int(row["output_sequence_length"]),
            int(row["batch_size"]),
        )
    )
    summary_csv_path = build_summary_csv_path(run_tag)
    write_summary_csv(rows, summary_csv_path)

    if len(rows) != len(cases):
        raise ValueError(
            f"CSV row count must match successful case count, got rows={len(rows)}, cases={len(cases)}"
        )

    return rows


def main() -> None:
    run_tag = datetime.now().strftime("%Y%m%d_%H%M")
    rows = run_sweep(run_tag)
    print(f"completed {len(rows)} sweep cases")
    print(f"summary csv: {build_summary_csv_path(run_tag)}")


if __name__ == "__main__":
    main()
