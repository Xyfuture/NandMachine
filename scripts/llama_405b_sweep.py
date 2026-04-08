from __future__ import annotations

import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from math import ceil
from pathlib import Path
from typing import Any

_SINGLE_THREAD_RUNTIME_ENV_DEFAULTS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
}
DEFAULT_MAX_WORKERS_CPU_DIVISOR = 1
TORCH_NUM_INTEROP_THREADS = 1

for _env_name, _env_value in _SINGLE_THREAD_RUNTIME_ENV_DEFAULTS.items():
    os.environ.setdefault(_env_name, _env_value)

import torch
from Desim import SimSession
from torch.fx import GraphModule

from nandmachine.commands.macro import MacroOp
from nandmachine.config.config import NandConfig
from nandmachine.config.hbm_hbf_architecture import (
    build_device_for_hbm_hbf_architecture_or_raise,
    validate_hbm_hbf_architecture_or_raise,
)
from nandmachine.config.hardware_config import get_device_or_raise
from nandmachine.config.inference_config import DenseParallelConfig, InferenceConfig
from nandmachine.config.model_config import LlamaModelConfig
from nandmachine.frontend.core.graph.base import NxGraphMeta, NxTracer
from nandmachine.frontend.core.passes.cod_gen import CodeGenPass
from nandmachine.frontend.core.passes.normalize import NormalizePass
from nandmachine.frontend.network.llama import LlamaDecoderLayer
from nandmachine.frontend.utlis import build_kv_cache_state
from nandmachine.simulator.hardware.xpu import xPU

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_CARD_PATH = REPO_ROOT / "model_cards" / "llama-405B.json"
TRACE_ROOT = REPO_ROOT / "trace" / "main"
SWEEP_NAME = "llama_405b_sweep"
FULL_TRACE_FILE_NAME = "full_simulation.json"
CONFIG_FILE_NAME = "config.json"
COMPILE_MODE = "heuristic-GPU"
MAX_WORKERS_ENV_VAR = "LLAMA_405B_SWEEP_MAX_WORKERS"
CASE_LIMIT_ENV_VAR = "LLAMA_405B_SWEEP_CASE_LIMIT"
INTERCONNECT_TOPOLOGY = "RING"


def build_trace_root(run_tag: str) -> Path:
    return TRACE_ROOT / f"{SWEEP_NAME}_{run_tag}"


def build_summary_csv_path(run_tag: str) -> Path:
    return TRACE_ROOT / f"{SWEEP_NAME}_summary_{run_tag}.csv"


BASE_NAND_CONFIG = {
    "num_channels": 6 * 8,
    "num_plane": 96,
    "num_block": 16,
    "num_pages": 16,
    "tRead": 4000,
    "tWrite": 4000 * 10,
    "tErase": 4000 * 100,
    "page_size": 4,
    "sram_threshold": 1024 * 160,
}
WEIGHT_BITS = 16
ACTIVATION_BITS = 16
KV_CACHE_BITS = 16
KV_BLOCK_SIZE_BYTES = 1024 * 256

@dataclass(frozen=True)
class HardwareSpec:
    hardware_type: str
    device_name: str
    memory_architecture: dict[str, int | str]
    memory_backend: str


@dataclass(frozen=True)
class SweepCase:
    hardware_type: str
    num_ranks: int
    batch_size: int
    input_sequence_length: int
    output_sequence_length: int
    slo_ms: int | None


@dataclass(frozen=True)
class RuntimeSpec:
    sim_hbm_bandwidth_GBps: float
    derived_hbf_bandwidth_GBps: float
    normalized_architecture: dict[str, str | int]


@dataclass(frozen=True)
class SequenceCaseConfig:
    input_sequence_length: int
    output_sequence_length: int
    hbm_batch_sizes_by_ranks: dict[int, tuple[int, ...]] | None
    csi_batch_sizes_by_ranks_by_slo_ms: dict[int, dict[int, tuple[int, ...]]] | None
    cli_batch_sizes_by_ranks_by_slo_ms: dict[int, dict[int, tuple[int, ...]]] | None


# Model memory note:
# - Weight footprint: 810 GiB
# - Full-model KV cache per token: 516096 B = 504 KiB/token
SEQUENCE_CASE_CONFIGS: tuple[SequenceCaseConfig, ...] = (
    SequenceCaseConfig(
        input_sequence_length=9400,
        output_sequence_length=600,
        hbm_batch_sizes_by_ranks={
            8: (64, 32, 16, 8),  # total capacity: 1128 GiB, remaining for KV: 318 GiB, max batch size: 66
        },
        csi_batch_sizes_by_ranks_by_slo_ms={
            50: {
                4: (28, 16, 8, 4),  # total capacity: 960 GiB, remaining for KV: 150 GiB, max batch size: 31
                8: (224, 128, 64, 32),  # total capacity: 1920 GiB, remaining for KV: 1110 GiB, max batch size: 230
            },
            100: {
                4: (228, 128, 64, 32),  # total capacity: 1920 GiB, remaining for KV: 1110 GiB, max batch size: 230
                8: (624, 512, 256, 128),  # total capacity: 3840 GiB, remaining for KV: 3030 GiB, max batch size: 630
            },
        },
        cli_batch_sizes_by_ranks_by_slo_ms={
            50: {
                8: (160, 128, 64, 32),  # total capacity: 1600 GiB, remaining for KV: 790 GiB, max batch size: 164
            },
            100: {
                4: (160, 128, 64, 32),  # total capacity: 1600 GiB, remaining for KV: 790 GiB, max batch size: 164
                8: (496, 256, 128, 64),  # total capacity: 3200 GiB, remaining for KV: 2390 GiB, max batch size: 497
            },
        },
    ),
    SequenceCaseConfig(
        input_sequence_length=8000,
        output_sequence_length=1000,
        hbm_batch_sizes_by_ranks={
            8: (72, 64, 32, 16),  # total capacity: 1128 GiB, remaining for KV: 318 GiB, max batch size: 73
        },
        csi_batch_sizes_by_ranks_by_slo_ms={
            50: {
                4: (32, 16, 8, 4),  # total capacity: 960 GiB, remaining for KV: 150 GiB, max batch size: 34
                8: (248, 128, 64, 32),  # total capacity: 1920 GiB, remaining for KV: 1110 GiB, max batch size: 256
            },
            100: {
                4: (252, 128, 64, 32),  # total capacity: 1920 GiB, remaining for KV: 1110 GiB, max batch size: 256
                8: (696, 512, 256, 128),  # total capacity: 3840 GiB, remaining for KV: 3030 GiB, max batch size: 700
            },
        },
        cli_batch_sizes_by_ranks_by_slo_ms={
            50: {
                8: (176, 128, 64, 32),  # total capacity: 1600 GiB, remaining for KV: 790 GiB, max batch size: 182
            },
            100: {
                4: (180, 128, 64, 32),  # total capacity: 1600 GiB, remaining for KV: 790 GiB, max batch size: 182
                8: (544, 512, 256, 128),  # total capacity: 3200 GiB, remaining for KV: 2390 GiB, max batch size: 552
            },
        },
    ),
    SequenceCaseConfig(
        input_sequence_length=20000,
        output_sequence_length=1000,
        hbm_batch_sizes_by_ranks={
            8: (24, 16, 8),  # total capacity: 1128 GiB, remaining for KV: 318 GiB, max batch size: 31
        },
        csi_batch_sizes_by_ranks_by_slo_ms={
            50: {
                4: (12, 8, 4),  # total capacity: 960 GiB, remaining for KV: 150 GiB, max batch size: 14
                8: (104, 64, 32, 16),  # total capacity: 1920 GiB, remaining for KV: 1110 GiB, max batch size: 109
            },
            100: {
                4: (108, 64, 32, 16),  # total capacity: 1920 GiB, remaining for KV: 1110 GiB, max batch size: 109
                8: (296, 256, 128, 64),  # total capacity: 3840 GiB, remaining for KV: 3030 GiB, max batch size: 300
            },
        },
        cli_batch_sizes_by_ranks_by_slo_ms={
            50: {
                8: (72, 64, 32, 16),  # total capacity: 1600 GiB, remaining for KV: 790 GiB, max batch size: 78
            },
            100: {
                4: (76, 64, 32, 16),  # total capacity: 1600 GiB, remaining for KV: 790 GiB, max batch size: 78
                8: (232, 128, 64, 32),  # total capacity: 3200 GiB, remaining for KV: 2390 GiB, max batch size: 236
            },
        },
    ),
)


def configure_runtime_thread_limits() -> None:
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    torch.set_num_interop_threads(TORCH_NUM_INTEROP_THREADS)


configure_runtime_thread_limits()

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


def _resolve_batch_sizes_by_ranks_by_slo_or_none(
    sequence_case_config: SequenceCaseConfig,
    mode: str,
) -> dict[int | None, dict[int, tuple[int, ...]]] | None:
    if mode == "hbm_only":
        if not sequence_case_config.hbm_batch_sizes_by_ranks:
            return None
        return {None: sequence_case_config.hbm_batch_sizes_by_ranks}

    if mode == "csi":
        return sequence_case_config.csi_batch_sizes_by_ranks_by_slo_ms

    if mode == "cli":
        return sequence_case_config.cli_batch_sizes_by_ranks_by_slo_ms

    raise AssertionError(f"Unhandled memory_architecture mode: {mode}")


HARDWARE_SPECS: tuple[HardwareSpec, ...] = (
    HardwareSpec(
        hardware_type="H200-HBM",
        device_name="H200_SXM",
        memory_architecture={"mode": "hbm_only"},
        memory_backend="hbm",
    ),
    HardwareSpec(
        hardware_type="H200-HBF-CLI",
        device_name="H200_SXM",
        memory_architecture={"mode": "cli", "hbm_stacks": 1, "hbf_stacks": 5},
        memory_backend="nand",
    ),
    HardwareSpec(
        hardware_type="H200-HBF-CSI",
        device_name="H200_SXM",
        memory_architecture={"mode": "csi"},
        memory_backend="nand",
    ),
)

CSV_CASE_OVERVIEW_FIELDNAMES = [
    "hardware_type",
    "memory_architecture_mode",
    "memory_backend",
    "interconnect_topology",
    "num_ranks",
    "slo_ms",
    "batch_size",
]
CSV_CORE_OUTPUT_FIELDNAMES = [
    "model_throughput_tokens_per_sec",
    "throughput_per_GPU",
    "layer_latency_ns",
    "model_latency_ns",
    "trace_path",
]
CSV_UNIQUE_RESULT_FIELDNAMES = [
    "status",
    "error_type",
    "error_message",
]
CSV_CASE_DIFF_FIELDNAMES = [
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
]
CSV_OTHER_OUTPUT_FIELDNAMES = [
    "macro_op_count",
]
CSV_COMMON_FIELDNAMES = [
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
CSV_FIELDNAMES = [
    *CSV_CASE_OVERVIEW_FIELDNAMES,
    *CSV_CORE_OUTPUT_FIELDNAMES,
    *CSV_UNIQUE_RESULT_FIELDNAMES,
    *CSV_CASE_DIFF_FIELDNAMES,
    *CSV_OTHER_OUTPUT_FIELDNAMES,
    *CSV_COMMON_FIELDNAMES,
]


def build_sweep_cases() -> list[SweepCase]:
    cases: list[SweepCase] = []
    for hardware_spec in HARDWARE_SPECS:
        mode = str(hardware_spec.memory_architecture["mode"])
        for sequence_case_config in SEQUENCE_CASE_CONFIGS:
            batch_sizes_by_ranks_by_slo = _resolve_batch_sizes_by_ranks_by_slo_or_none(
                sequence_case_config,
                mode,
            )
            if not batch_sizes_by_ranks_by_slo:
                continue

            for slo_ms, batch_sizes_by_ranks in batch_sizes_by_ranks_by_slo.items():
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


def load_model_card_or_raise() -> dict[str, Any]:
    if not MODEL_CARD_PATH.exists():
        raise FileNotFoundError(f"Model card not found: {MODEL_CARD_PATH}")
    return json.loads(MODEL_CARD_PATH.read_text())


def build_raw_model_config(model_card: dict[str, Any]) -> object:
    return type("Llama405BSweepConfig", (), model_card)()


def build_parallel_config(num_ranks: int) -> DenseParallelConfig:
    return DenseParallelConfig(
        num_ranks=num_ranks,
        tp_size=num_ranks,
        dp_size=1,
    )


def build_nand_config(hardware_spec: HardwareSpec) -> NandConfig:
    normalized_architecture = validate_hbm_hbf_architecture_or_raise(
        hardware_spec.device_name,
        hardware_spec.memory_architecture,
    )

    num_channels = BASE_NAND_CONFIG["num_channels"]
    hbf_stacks = int(normalized_architecture["effective_hbf_stacks"])
    base_hbm_stacks = int(normalized_architecture["base_hbm_stack_count"])

    if hbf_stacks > 0:
        scaled_num_channels = num_channels * hbf_stacks
        if scaled_num_channels % base_hbm_stacks != 0:
            raise ValueError(
                "Scaled num_channels must be divisible by base_hbm_stack_count, "
                f"got scaled_num_channels={scaled_num_channels}, "
                f"base_hbm_stack_count={base_hbm_stacks}"
            )
        num_channels = scaled_num_channels // base_hbm_stacks

    return NandConfig(
        num_channels=num_channels,
        num_plane=BASE_NAND_CONFIG["num_plane"],
        num_block=BASE_NAND_CONFIG["num_block"],
        num_pages=BASE_NAND_CONFIG["num_pages"],
        tRead=BASE_NAND_CONFIG["tRead"],
        tWrite=BASE_NAND_CONFIG["tWrite"],
        tErase=BASE_NAND_CONFIG["tErase"],
        page_size=BASE_NAND_CONFIG["page_size"],
        sram_threshold=BASE_NAND_CONFIG["sram_threshold"],
    )


def calculate_derived_hbf_bandwidth_GBps(nand_config: NandConfig) -> float:
    derived_hbf_bandwidth_bytes_per_sec = (
        nand_config.num_channels
        * nand_config.num_plane
        * (nand_config.page_size_bytes / nand_config.tRead)
        * 1e9
    )
    if derived_hbf_bandwidth_bytes_per_sec <= 0:
        raise ValueError(
            "derived_hbf_bandwidth_bytes_per_sec must be > 0, "
            f"got {derived_hbf_bandwidth_bytes_per_sec}"
        )
    return derived_hbf_bandwidth_bytes_per_sec / 1e9


def build_runtime_spec(hardware_spec: HardwareSpec, nand_config: NandConfig) -> RuntimeSpec:
    device = build_device_for_hbm_hbf_architecture_or_raise(
        hardware_spec.device_name,
        hardware_spec.memory_architecture,
    )
    normalized_architecture = validate_hbm_hbf_architecture_or_raise(
        hardware_spec.device_name,
        hardware_spec.memory_architecture,
    )
    derived_hbf_bandwidth_GBps = calculate_derived_hbf_bandwidth_GBps(nand_config)

    if device.memory_architecture_mode == "hbm_only":
        sim_hbm_bandwidth_GBps = device.io_module.hbm_bandwidth / 1e9
    elif device.memory_architecture_mode == "cli":
        derived_hbf_bandwidth_GBps = device.io_module.hbf_bandwidth / 1e9
        residual_bandwidth_bytes_per_sec = (
            device.io_module.total_bandwidth - device.io_module.hbf_bandwidth
        )
        if residual_bandwidth_bytes_per_sec <= 0:
            raise ValueError(
                "CLI residual HBM bandwidth must be > 0, "
                f"got {residual_bandwidth_bytes_per_sec}"
            )
        sim_hbm_bandwidth_GBps = residual_bandwidth_bytes_per_sec / 1e9
    elif device.memory_architecture_mode == "csi":
        sim_hbm_bandwidth_GBps = device.io_module.hbm_bandwidth / 1e9
    else:
        raise AssertionError(
            f"Unhandled memory_architecture_mode: {device.memory_architecture_mode}"
        )

    return RuntimeSpec(
        sim_hbm_bandwidth_GBps=sim_hbm_bandwidth_GBps,
        derived_hbf_bandwidth_GBps=derived_hbf_bandwidth_GBps,
        normalized_architecture=normalized_architecture,
    )

def build_inference_config(
    case: SweepCase,
    parallel_config: DenseParallelConfig,
    memory_backend: str,
) -> InferenceConfig:
    return InferenceConfig(
        batch_size=case.batch_size,
        input_sequence_length=case.input_sequence_length,
        output_sequence_length=case.output_sequence_length,
        weight_bits=WEIGHT_BITS,
        activation_bits=ACTIVATION_BITS,
        kv_cache_bits=KV_CACHE_BITS,
        kv_block_size_bytes=KV_BLOCK_SIZE_BYTES,
        memory_backend=memory_backend,
        parallel_config=parallel_config,
    )


def build_macro_op_list(
    raw_model_config: object,
    model_config: LlamaModelConfig,
    nand_config: NandConfig,
    inference_config: InferenceConfig,
    parallel_config: DenseParallelConfig,
) -> list[MacroOp]:
    kv_cache_state = build_kv_cache_state(nand_config, model_config, inference_config)
    graph_meta = NxGraphMeta(
        nand_config=nand_config,
        model_config=model_config,
        inference_config=inference_config,
        kv_cache_state=kv_cache_state,
    )

    with torch.device("meta"):
        model = LlamaDecoderLayer(model_config, tp_size=parallel_config.tp_size)
        graph = NxTracer().trace(model)
        graph_module = GraphModule(model, graph)

    NormalizePass().transform(graph_module)
    graph_module.graph.meta = {CodeGenPass.GRAPH_META_KEY: graph_meta}
    CodeGenPass().transform(graph_module)

    macro_op_list = graph_module.graph.meta[CodeGenPass.MACRO_OP_LIST_META_KEY]
    if not macro_op_list:
        raise ValueError("macro_op_list must not be empty")
    return macro_op_list


def run_macro_ops_with_trace(
    nand_config: NandConfig,
    commands: list[MacroOp],
    *,
    trace_path: Path,
    device_name: str,
    compile_mode: str,
    hbm_bandwidth_GBps: float,
) -> dict[str, object]:
    SimSession.reset()
    SimSession.init()

    sim_xpu = xPU(
        nand_config,
        hbm_bandwidth_bytes_per_sec=hbm_bandwidth_GBps * 10**9,
        device_name=device_name,
        compile_mode=compile_mode,
        enable_trace=True,
    )
    sim_xpu.load_command(commands)
    SimSession.scheduler.run()

    final_time_ns = int(SimSession.sim_time.cycle)
    device = get_device_or_raise(device_name)
    final_cycle = ceil(final_time_ns * device.compute_module.clock_freq / 1e9)

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    saved_trace_path = Path(sim_xpu.save_trace_file(str(trace_path)))

    tracer = sim_xpu.tracer
    if tracer is None:
        raise RuntimeError("xPU tracer must be enabled")
    complete_events = [event for event in tracer._events if event["ph"] == "X"]

    return {
        "cycle": final_cycle,
        "time_ns": final_time_ns,
        "trace_path": str(saved_trace_path),
        "trace_event_count": len(tracer._events),
        "trace_complete_event_count": len(complete_events),
    }


def build_trace_dir(case: SweepCase, run_tag: str) -> Path:
    slo_segment = "slo_none" if case.slo_ms is None else f"slo_{case.slo_ms}ms"
    return (
        build_trace_root(run_tag)
        / case.hardware_type
        / f"ranks_{case.num_ranks}"
        / slo_segment
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
    trace_root = build_trace_root(run_tag)
    summary_csv_path = build_summary_csv_path(run_tag)
    model_card = load_model_card_or_raise()
    raw_model_config = build_raw_model_config(deepcopy(model_card))
    model_config = LlamaModelConfig.from_dict(model_card)

    if not isinstance(model_config.num_hidden_layers, int):
        raise TypeError(
            "model_config.num_hidden_layers must be an int, "
            f"got {type(model_config.num_hidden_layers).__name__}"
        )
    if model_config.num_hidden_layers <= 0:
        raise ValueError(
            f"model_config.num_hidden_layers must be > 0, got {model_config.num_hidden_layers}"
        )

    parallel_config = build_parallel_config(case.num_ranks)
    nand_config = build_nand_config(hardware_spec)
    runtime_spec = build_runtime_spec(hardware_spec, nand_config)
    inference_config = build_inference_config(
        case,
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

    trace_dir = build_trace_dir(case, run_tag)
    trace_path = trace_dir / FULL_TRACE_FILE_NAME
    sim_result = run_macro_ops_with_trace(
        nand_config,
        macro_op_list,
        trace_path=trace_path,
        device_name=hardware_spec.device_name,
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
        "status": "ok",
        "error_type": None,
        "error_message": None,
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
        "attn_dp_size": parallel_config.dp_size,
        "attn_tp_size": parallel_config.tp_size,
        "ffn_tp_size": None,
        "ffn_ep_size": None,
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
            "sequence_case_configs": [asdict(sequence_case_config) for sequence_case_config in SEQUENCE_CASE_CONFIGS],
        },
        "model_config": asdict(model_config),
        "hardware_spec": asdict(hardware_spec),
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


def build_error_row(
    case: SweepCase,
    max_workers: int,
    selected_case_count: int,
    total_case_count: int,
    exc: BaseException,
) -> dict[str, object]:
    hardware_spec = get_hardware_spec_or_raise(case.hardware_type)
    return {
        "hardware_type": hardware_spec.hardware_type,
        "device_name": hardware_spec.device_name,
        "model_card_path": str(MODEL_CARD_PATH),
        "compile_mode": COMPILE_MODE,
        "batch_size_semantics": "global",
        "interconnect_topology": INTERCONNECT_TOPOLOGY,
        "status": "error",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "case_limit": os.getenv(CASE_LIMIT_ENV_VAR),
        "selected_case_count": selected_case_count,
        "total_case_count": total_case_count,
        "max_workers": max_workers,
        "worker_count_source": resolve_worker_count_source(),
        "host_logical_cpu_count": os.cpu_count() or 1,
        "memory_architecture_mode": None,
        "effective_hbm_stacks": None,
        "effective_hbf_stacks": None,
        "memory_backend": hardware_spec.memory_backend,
        "num_ranks": case.num_ranks,
        "attn_dp_size": None,
        "attn_tp_size": None,
        "ffn_tp_size": None,
        "ffn_ep_size": None,
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
        "torch_num_threads": None,
        "torch_num_interop_threads": TORCH_NUM_INTEROP_THREADS,
        "nand_num_channels": None,
        "nand_num_plane": None,
        "nand_num_block": None,
        "nand_num_pages": None,
        "nand_tRead": None,
        "nand_tWrite": None,
        "nand_tErase": None,
        "nand_page_size_kb": None,
        "nand_sram_threshold_kb": None,
        "sim_hbm_bandwidth_GBps": None,
        "derived_hbf_bandwidth_GBps": None,
        "macro_op_count": None,
        "layer_latency_ns": None,
        "model_latency_ns": None,
        "model_throughput_tokens_per_sec": None,
        "throughput_per_GPU": None,
        "trace_path": None,
    }


def run_case(
    case: SweepCase,
    max_workers: int,
    selected_case_count: int,
    total_case_count: int,
    run_tag: str,
) -> dict[str, object]:
    try:
        return build_result_row(
            case,
            max_workers,
            selected_case_count,
            total_case_count,
            run_tag,
        )
    except Exception as exc:  # noqa: BLE001
        return build_error_row(
            case,
            max_workers,
            selected_case_count,
            total_case_count,
            exc,
        )


def write_summary_csv(
    rows: list[dict[str, object]],
    summary_csv_path: Path,
) -> None:
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
                run_case,
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
            -1 if row["slo_ms"] is None else int(row["slo_ms"]),
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
