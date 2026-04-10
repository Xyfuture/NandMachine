from __future__ import annotations

import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
from torch.fx import GraphModule

from nandmachine.commands.macro import MacroOp
from nandmachine.config.cache_state import KVCacheState
from nandmachine.config.config import NandConfig
from nandmachine.config.hbm_hbf_architecture import (
    build_device_for_hbm_hbf_architecture_or_raise,
    validate_hbm_hbf_architecture_or_raise,
)
from nandmachine.config.inference_config import InferenceConfig, MoEParallelConfig
from nandmachine.config.model_config import Qwen3MoEModelConfig
from nandmachine.frontend.core.graph.base import NxGraphMeta, NxTracer
from nandmachine.frontend.core.passes.cod_gen import CodeGenPass
from nandmachine.frontend.core.passes.normalize import NormalizePass
from nandmachine.frontend.network.qwen3_moe import Qwen3MoEDecoderLayer
from nandmachine.frontend.utlis import (
    build_imbalanced_kv_cache_state,
    build_kv_cache_state,
)
from nandmachine.simulator.entry_point import universe_run_sim


MODEL_CARD_PATH = Path("model_cards/qwen3-moe-235B.json")
TRACE_ROOT = Path("trace/main")
SWEEP_NAME = "qwen3_moe_ablation_sweep"
CONFIG_FILE_NAME = "config.json"
SUMMARY_FILE_SUFFIX = "summary"
COMPILE_MODE = "heuristic-GPU"
MAX_WORKERS_ENV_VAR = "QWEN3_MOE_ABLATION_SWEEP_MAX_WORKERS"
CASE_LIMIT_ENV_VAR = "QWEN3_MOE_ABLATION_SWEEP_CASE_LIMIT"

BASE_NAND_CONFIG = {
    "num_channels": 6 * 8,
    "num_plane": 96,
    "num_block": 16,
    "num_pages": 16,
    "tRead": 4000,
    "tWrite": 4000 * 10,
    "tErase": 4000 * 100,
    "page_size": 4,
    "sram_threshold": 1024 * 80,
}
WEIGHT_BITS = 16
ACTIVATION_BITS = 16
KV_CACHE_BITS = 16
KV_BLOCK_SIZE_BYTES = 1024 * 256
NUM_RANKS = 8
INPUT_SEQUENCE_LENGTH = 9400
OUTPUT_SEQUENCE_LENGTH = 600
BASELINE_BATCH_SIZES = (256, 400)
HBF_BATCH_SIZES = (256, 2080, 2048, 1024, 512)

XPUType = Literal["default", "vallina"]


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_name: str
    device_name: str
    memory_architecture: dict[str, int | str]
    memory_backend: str
    xpu_type: XPUType
    use_imbalanced_kv_cache: bool
    enable_strict: bool


@dataclass(frozen=True)
class AblationCase:
    experiment_name: str
    batch_size: int
    num_ranks: int
    input_sequence_length: int
    output_sequence_length: int


@dataclass(frozen=True)
class RuntimeSpec:
    sim_hbm_bandwidth_GBps: float
    derived_hbf_bandwidth_GBps: float
    normalized_architecture: dict[str, str | int]


EXPERIMENT_SPECS: tuple[ExperimentSpec, ...] = (
    ExperimentSpec(
        experiment_name="BaselineHBM",
        device_name="H200_SXM",
        memory_architecture={"mode": "hbm_only"},
        memory_backend="hbm",
        xpu_type="default",
        use_imbalanced_kv_cache=False,
        enable_strict=False,
    ),
    ExperimentSpec(
        experiment_name="PlainHBF",
        device_name="H200_SXM",
        memory_architecture={"mode": "csi"},
        memory_backend="nand",
        xpu_type="vallina",
        use_imbalanced_kv_cache=True,
        enable_strict=True,
    ),
    ExperimentSpec(
        experiment_name="WO-Prefetch",
        device_name="H200_SXM",
        memory_architecture={"mode": "csi"},
        memory_backend="nand",
        xpu_type="vallina",
        use_imbalanced_kv_cache=False,
        enable_strict=False,
    ),
    ExperimentSpec(
        experiment_name="WO-WeightLayout",
        device_name="H200_SXM",
        memory_architecture={"mode": "csi"},
        memory_backend="nand",
        xpu_type="default",
        use_imbalanced_kv_cache=False,
        enable_strict=True,
    ),
    ExperimentSpec(
        experiment_name="WO-KVCacheLayout",
        device_name="H200_SXM",
        memory_architecture={"mode": "csi"},
        memory_backend="nand",
        xpu_type="default",
        use_imbalanced_kv_cache=True,
        enable_strict=False,
    ),
    ExperimentSpec(
        experiment_name="FlashAccel",
        device_name="H200_SXM",
        memory_architecture={"mode": "csi"},
        memory_backend="nand",
        xpu_type="default",
        use_imbalanced_kv_cache=False,
        enable_strict=False,
    ),
)

CSV_FIELDNAMES = [
    "experiment_name",
    "device_name",
    "memory_architecture_mode",
    "memory_backend",
    "num_ranks",
    "batch_size",
    "input_sequence_length",
    "output_sequence_length",
    "xpu_type",
    "use_imbalanced_kv_cache",
    "nand_enable_strict",
    "kv_cache_is_imbalance",
    "effective_hbm_stacks",
    "effective_hbf_stacks",
    "sim_hbm_bandwidth_GBps",
    "derived_hbf_bandwidth_GBps",
    "model_throughput_tokens_per_sec",
    "throughput_per_GPU",
    "layer_latency_ns",
    "model_latency_ns",
    "kv_cache_total_size_GB",
    "macro_op_count",
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
    "nand_num_channels",
    "nand_num_plane",
    "nand_num_block",
    "nand_num_pages",
    "nand_tRead",
    "nand_tWrite",
    "nand_tErase",
    "nand_page_size_kb",
    "nand_sram_threshold_kb",
    "config_path",
]


def configure_runtime_thread_limits() -> None:
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    torch.set_num_interop_threads(TORCH_NUM_INTEROP_THREADS)


configure_runtime_thread_limits()


def build_trace_root(run_tag: str) -> Path:
    return TRACE_ROOT / f"{SWEEP_NAME}_{run_tag}"


def build_summary_csv_path(run_tag: str) -> Path:
    return TRACE_ROOT / f"{SWEEP_NAME}_{SUMMARY_FILE_SUFFIX}_{run_tag}.csv"


def build_case_dir(case: AblationCase, run_tag: str) -> Path:
    return (
        build_trace_root(run_tag)
        / case.experiment_name
        / f"ranks_{case.num_ranks}"
        / f"isl_{case.input_sequence_length}_osl_{case.output_sequence_length}"
        / f"bs_{case.batch_size}"
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


def get_experiment_spec_or_raise(experiment_name: str) -> ExperimentSpec:
    for experiment_spec in EXPERIMENT_SPECS:
        if experiment_spec.experiment_name == experiment_name:
            return experiment_spec
    raise ValueError(f"Unsupported experiment_name: {experiment_name}")


def build_sweep_cases() -> list[AblationCase]:
    cases: list[AblationCase] = []
    for experiment_spec in EXPERIMENT_SPECS:
        batch_sizes = (
            BASELINE_BATCH_SIZES
            if experiment_spec.experiment_name == "BaselineHBM"
            else HBF_BATCH_SIZES
        )
        for batch_size in batch_sizes:
            cases.append(
                AblationCase(
                    experiment_name=experiment_spec.experiment_name,
                    batch_size=batch_size,
                    num_ranks=NUM_RANKS,
                    input_sequence_length=INPUT_SEQUENCE_LENGTH,
                    output_sequence_length=OUTPUT_SEQUENCE_LENGTH,
                )
            )
    return cases


def load_model_card_or_raise() -> dict[str, Any]:
    if not MODEL_CARD_PATH.exists():
        raise FileNotFoundError(f"Model card not found: {MODEL_CARD_PATH}")
    model_card = json.loads(MODEL_CARD_PATH.read_text())
    model_card.setdefault("attention_type", "gqa")
    return model_card


def build_raw_model_config(model_card: dict[str, Any]) -> object:
    return type("Qwen3MoeAblationConfig", (), model_card)()


def build_parallel_config(num_ranks: int) -> MoEParallelConfig:
    return MoEParallelConfig(
        num_ranks=num_ranks,
        attn_dp_size=num_ranks,
        attn_tp_size=1,
        ffn_tp_size=1,
        ffn_ep_size=num_ranks,
    )


def build_nand_config(experiment_spec: ExperimentSpec) -> NandConfig:
    normalized_architecture = validate_hbm_hbf_architecture_or_raise(
        experiment_spec.device_name,
        experiment_spec.memory_architecture,
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
        enable_strict=experiment_spec.enable_strict,
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


def build_runtime_spec(
    experiment_spec: ExperimentSpec,
    nand_config: NandConfig,
) -> RuntimeSpec:
    device = build_device_for_hbm_hbf_architecture_or_raise(
        experiment_spec.device_name,
        experiment_spec.memory_architecture,
    )
    normalized_architecture = validate_hbm_hbf_architecture_or_raise(
        experiment_spec.device_name,
        experiment_spec.memory_architecture,
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
    case: AblationCase,
    parallel_config: MoEParallelConfig,
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


def resolve_kv_cache_state(
    experiment_spec: ExperimentSpec,
    nand_config: NandConfig,
    model_config: Qwen3MoEModelConfig,
    inference_config: InferenceConfig,
) -> KVCacheState:
    if experiment_spec.use_imbalanced_kv_cache:
        return build_imbalanced_kv_cache_state(
            nand_config,
            model_config,
            inference_config,
        )
    return build_kv_cache_state(
        nand_config,
        model_config,
        inference_config,
    )


def build_macro_op_list(
    raw_model_config: object,
    model_config: Qwen3MoEModelConfig,
    nand_config: NandConfig,
    inference_config: InferenceConfig,
    parallel_config: MoEParallelConfig,
    kv_cache_state: KVCacheState,
) -> list[MacroOp]:
    graph_meta = NxGraphMeta(
        nand_config=nand_config,
        model_config=model_config,
        inference_config=inference_config,
        kv_cache_state=kv_cache_state,
    )

    with torch.device("meta"):
        model = Qwen3MoEDecoderLayer(raw_model_config, parallel_config)
        graph = NxTracer().trace(model)
        graph_module = GraphModule(model, graph)

    NormalizePass().transform(graph_module)
    graph_module.graph.meta = {CodeGenPass.GRAPH_META_KEY: graph_meta}
    CodeGenPass().transform(graph_module)

    macro_op_list = graph_module.graph.meta[CodeGenPass.MACRO_OP_LIST_META_KEY]
    if not macro_op_list:
        raise ValueError("macro_op_list must not be empty")
    return macro_op_list


def write_json_file(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def build_result_row(
    case: AblationCase,
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
            f"got selected_case_count={selected_case_count}, "
            f"total_case_count={total_case_count}"
        )

    experiment_spec = get_experiment_spec_or_raise(case.experiment_name)
    model_card = load_model_card_or_raise()
    raw_model_config = build_raw_model_config(deepcopy(model_card))
    model_config = Qwen3MoEModelConfig.from_config(raw_model_config)
    parallel_config = build_parallel_config(case.num_ranks)
    nand_config = build_nand_config(experiment_spec)
    runtime_spec = build_runtime_spec(experiment_spec, nand_config)
    inference_config = build_inference_config(
        case,
        parallel_config,
        experiment_spec.memory_backend,
    )
    kv_cache_state = resolve_kv_cache_state(
        experiment_spec,
        nand_config,
        model_config,
        inference_config,
    )
    macro_op_list = build_macro_op_list(
        raw_model_config,
        model_config,
        nand_config,
        inference_config,
        parallel_config,
        kv_cache_state,
    )

    sim_result = universe_run_sim(
        nand_config,
        model_config,
        inference_config,
        macro_op_list,
        hbm_bandwidth_bytes_per_sec=runtime_spec.sim_hbm_bandwidth_GBps * 10**9,
        device_name=experiment_spec.device_name,
        compile_mode=COMPILE_MODE,
        xpu_type=experiment_spec.xpu_type,
        kv_cache_state=kv_cache_state,
    )

    host_logical_cpu_count = os.cpu_count() or 1
    case_dir = build_case_dir(case, run_tag)
    config_path = case_dir / CONFIG_FILE_NAME

    row = {
        "experiment_name": experiment_spec.experiment_name,
        "device_name": experiment_spec.device_name,
        "memory_architecture_mode": runtime_spec.normalized_architecture["mode"],
        "memory_backend": experiment_spec.memory_backend,
        "num_ranks": case.num_ranks,
        "batch_size": case.batch_size,
        "input_sequence_length": case.input_sequence_length,
        "output_sequence_length": case.output_sequence_length,
        "xpu_type": experiment_spec.xpu_type,
        "use_imbalanced_kv_cache": experiment_spec.use_imbalanced_kv_cache,
        "nand_enable_strict": nand_config.enable_strict,
        "kv_cache_is_imbalance": kv_cache_state.is_imbalance,
        "effective_hbm_stacks": runtime_spec.normalized_architecture["effective_hbm_stacks"],
        "effective_hbf_stacks": runtime_spec.normalized_architecture["effective_hbf_stacks"],
        "sim_hbm_bandwidth_GBps": runtime_spec.sim_hbm_bandwidth_GBps,
        "derived_hbf_bandwidth_GBps": runtime_spec.derived_hbf_bandwidth_GBps,
        "model_throughput_tokens_per_sec": sim_result.model_throughput,
        "throughput_per_GPU": sim_result.throughput_per_GPU,
        "layer_latency_ns": sim_result.layer_latency_ns,
        "model_latency_ns": sim_result.model_latency_ns,
        "kv_cache_total_size_GB": sim_result.kv_cache_total_size_GB,
        "macro_op_count": len(macro_op_list),
        "model_card_path": str(MODEL_CARD_PATH),
        "compile_mode": COMPILE_MODE,
        "batch_size_semantics": "global",
        "case_limit": os.getenv(CASE_LIMIT_ENV_VAR),
        "selected_case_count": selected_case_count,
        "total_case_count": total_case_count,
        "max_workers": max_workers,
        "worker_count_source": resolve_worker_count_source(),
        "host_logical_cpu_count": host_logical_cpu_count,
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
        "config_path": str(config_path),
    }

    config_payload = {
        "script_config": {
            "model_card_path": str(MODEL_CARD_PATH),
            "trace_root": str(build_trace_root(run_tag)),
            "summary_csv_path": str(build_summary_csv_path(run_tag)),
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
        },
        "experiment_spec": asdict(experiment_spec),
        "ablation_case": asdict(case),
        "model_config": asdict(model_config),
        "parallel_config": asdict(parallel_config),
        "inference_config": asdict(inference_config),
        "nand_config": asdict(nand_config),
        "runtime_spec": asdict(runtime_spec),
        "kv_cache_state": asdict(kv_cache_state),
        "simulation_result": {
            "layer_latency_ns": sim_result.layer_latency_ns,
            "model_latency_ns": sim_result.model_latency_ns,
            "model_throughput_tokens_per_sec": sim_result.model_throughput,
            "throughput_per_GPU": sim_result.throughput_per_GPU,
            "kv_cache_total_size_GB": sim_result.kv_cache_total_size_GB,
            "macro_op_count": len(macro_op_list),
        },
    }
    write_json_file(config_path, config_payload)
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

    if max_workers == 1 or selected_case_count == 1:
        for case in cases:
            rows.append(
                build_result_row(
                    case,
                    max_workers,
                    selected_case_count,
                    total_case_count,
                    run_tag,
                )
            )
    else:
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

    experiment_order = {
        experiment_spec.experiment_name: index
        for index, experiment_spec in enumerate(EXPERIMENT_SPECS)
    }
    rows.sort(
        key=lambda row: (
            experiment_order[str(row["experiment_name"])],
            int(row["num_ranks"]),
            int(row["input_sequence_length"]),
            int(row["output_sequence_length"]),
            int(row["batch_size"]),
        )
    )
    summary_csv_path = build_summary_csv_path(run_tag)
    write_summary_csv(rows, summary_csv_path)

    if len(rows) != len(cases):
        raise ValueError(
            f"CSV row count must match successful case count, got rows={len(rows)}, "
            f"cases={len(cases)}"
        )

    return rows


def main() -> None:
    run_tag = datetime.now().strftime("%Y%m%d_%H%M")
    rows = run_sweep(run_tag)
    print(f"completed {len(rows)} ablation cases")
    print(f"summary csv: {build_summary_csv_path(run_tag)}")


if __name__ == "__main__":
    main()
