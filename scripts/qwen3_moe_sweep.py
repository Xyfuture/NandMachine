from __future__ import annotations

import csv
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict, dataclass
from math import ceil
from pathlib import Path
from typing import Any

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
from nandmachine.config.inference_config import InferenceConfig, MoEParallelConfig
from nandmachine.config.model_config import Qwen3MoEModelConfig
from nandmachine.frontend.core.graph.base import NxGraphMeta, NxTracer
from nandmachine.frontend.core.passes.cod_gen import CodeGenPass
from nandmachine.frontend.core.passes.normalize import NormalizePass
from nandmachine.frontend.network.qwen3_moe import Qwen3MoEDecoderLayer
from nandmachine.frontend.utlis import build_kv_cache_state
from nandmachine.frontend.validator import validate_batch_size_or_raise
from nandmachine.simulator.hardware.xpu import xPU


MODEL_CARD_PATH = Path("model_cards/qwen3-moe-235B.json")
TRACE_ROOT = Path("trace/main")
SUMMARY_CSV_PATH = TRACE_ROOT / "qwen3_moe_sweep_summary.csv"
FULL_TRACE_FILE_NAME = "full_simulation.json"
CONFIG_FILE_NAME = "config.json"
COMPILE_MODE = "heuristic-GPU"
MAX_WORKERS = min(8, os.cpu_count() or 1)

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

SEQUENCE_LENGTHS: tuple[tuple[int, int], ...] = (
    (8 * 1024, 1 * 1024),
    (9400, 600),
    (20 * 1024, 1 * 1024),
)
BATCH_SIZES_BY_RANKS: dict[int, tuple[int, ...]] = {
    4: (32, 64, 128),
    8: (64, 128, 256),
    16: (128, 256, 512),
}

SIMULATION_LOCK = threading.Lock()


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


@dataclass(frozen=True)
class RuntimeSpec:
    sim_hbm_bandwidth_GBps: float
    derived_hbf_bandwidth_GBps: float
    normalized_architecture: dict[str, str | int]


HARDWARE_SPECS: tuple[HardwareSpec, ...] = (
    HardwareSpec(
        hardware_type="H100-HBM",
        device_name="H100_SXM",
        memory_architecture={"mode": "hbm_only"},
        memory_backend="hbm",
    ),
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

CSV_FIELDNAMES = [
    "hardware_type",
    "device_name",
    "memory_architecture_mode",
    "effective_hbm_stacks",
    "effective_hbf_stacks",
    "memory_backend",
    "num_ranks",
    "attn_dp_size",
    "attn_tp_size",
    "ffn_tp_size",
    "ffn_ep_size",
    "batch_size",
    "input_sequence_length",
    "output_sequence_length",
    "sim_hbm_bandwidth_GBps",
    "derived_hbf_bandwidth_GBps",
    "macro_op_count",
    "layer_latency_ns",
    "model_latency_ns",
    "model_throughput_tokens_per_sec",
    "total_used_bytes",
    "total_weight_bytes",
    "total_kv_cache_bytes",
    "trace_path",
]


def build_sweep_cases() -> list[SweepCase]:
    cases: list[SweepCase] = []
    for hardware_spec in HARDWARE_SPECS:
        for num_ranks, batch_sizes in BATCH_SIZES_BY_RANKS.items():
            for input_sequence_length, output_sequence_length in SEQUENCE_LENGTHS:
                for batch_size in batch_sizes:
                    cases.append(
                        SweepCase(
                            hardware_type=hardware_spec.hardware_type,
                            num_ranks=num_ranks,
                            batch_size=batch_size,
                            input_sequence_length=input_sequence_length,
                            output_sequence_length=output_sequence_length,
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
    model_card = json.loads(MODEL_CARD_PATH.read_text())
    model_card.setdefault("attention_type", "gqa")
    return model_card


def build_raw_model_config(model_card: dict[str, Any]) -> object:
    return type("Qwen3MoeSweepConfig", (), model_card)()


def build_parallel_config(num_ranks: int) -> MoEParallelConfig:
    return MoEParallelConfig(
        num_ranks=num_ranks,
        attn_dp_size=num_ranks,
        attn_tp_size=1,
        ffn_tp_size=1,
        ffn_ep_size=num_ranks,
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
        sim_hbm_bandwidth_GBps = device.io_module.hbm_bandwidth / 1e9
    elif device.memory_architecture_mode == "csi":
        residual_bandwidth_bytes_per_sec = (
            device.io_module.total_bandwidth - derived_hbf_bandwidth_GBps * 1e9
        )
        if residual_bandwidth_bytes_per_sec <= 0:
            raise ValueError(
                "CSI residual HBM bandwidth must be > 0, "
                f"got {residual_bandwidth_bytes_per_sec}"
            )
        sim_hbm_bandwidth_GBps = residual_bandwidth_bytes_per_sec / 1e9
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


def build_macro_op_list(
    raw_model_config: object,
    model_config: Qwen3MoEModelConfig,
    nand_config: NandConfig,
    inference_config: InferenceConfig,
    parallel_config: MoEParallelConfig,
) -> list[MacroOp]:
    kv_cache_state = build_kv_cache_state(nand_config, model_config, inference_config)
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


def build_trace_dir(case: SweepCase) -> Path:
    return (
        TRACE_ROOT
        / case.hardware_type
        / f"ranks_{case.num_ranks}"
        / f"isl_{case.input_sequence_length}_osl_{case.output_sequence_length}"
        / f"bs_{case.batch_size}"
    )


def write_json_file(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def build_result_row(case: SweepCase) -> dict[str, object]:
    hardware_spec = get_hardware_spec_or_raise(case.hardware_type)
    model_card = load_model_card_or_raise()
    raw_model_config = build_raw_model_config(deepcopy(model_card))
    model_config = Qwen3MoEModelConfig.from_config(raw_model_config)

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
    capacity_result = validate_batch_size_or_raise(
        hardware_spec.device_name,
        hardware_spec.memory_architecture,
        model_config,
        inference_config,
    )
    macro_op_list = build_macro_op_list(
        raw_model_config,
        model_config,
        nand_config,
        inference_config,
        parallel_config,
    )

    trace_dir = build_trace_dir(case)
    trace_path = trace_dir / FULL_TRACE_FILE_NAME
    with SIMULATION_LOCK:
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

    row = {
        "hardware_type": hardware_spec.hardware_type,
        "device_name": hardware_spec.device_name,
        "memory_architecture_mode": runtime_spec.normalized_architecture["mode"],
        "effective_hbm_stacks": runtime_spec.normalized_architecture["effective_hbm_stacks"],
        "effective_hbf_stacks": runtime_spec.normalized_architecture["effective_hbf_stacks"],
        "memory_backend": hardware_spec.memory_backend,
        "num_ranks": case.num_ranks,
        "attn_dp_size": parallel_config.attn_dp_size,
        "attn_tp_size": parallel_config.attn_tp_size,
        "ffn_tp_size": parallel_config.ffn_tp_size,
        "ffn_ep_size": parallel_config.ffn_ep_size,
        "batch_size": case.batch_size,
        "input_sequence_length": case.input_sequence_length,
        "output_sequence_length": case.output_sequence_length,
        "sim_hbm_bandwidth_GBps": runtime_spec.sim_hbm_bandwidth_GBps,
        "derived_hbf_bandwidth_GBps": runtime_spec.derived_hbf_bandwidth_GBps,
        "macro_op_count": len(macro_op_list),
        "layer_latency_ns": layer_latency_ns,
        "model_latency_ns": model_latency_ns,
        "model_throughput_tokens_per_sec": model_throughput_tokens_per_sec,
        "total_used_bytes": capacity_result.total_used_bytes,
        "total_weight_bytes": capacity_result.total_weight_bytes,
        "total_kv_cache_bytes": capacity_result.total_kv_cache_bytes,
        "trace_path": sim_result["trace_path"],
    }

    config_payload = {
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
        "capacity_result": asdict(capacity_result),
        "simulation_result": {
            "layer_latency_ns": layer_latency_ns,
            "model_latency_ns": model_latency_ns,
            "model_throughput_tokens_per_sec": model_throughput_tokens_per_sec,
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


def write_summary_csv(rows: list[dict[str, object]]) -> None:
    TRACE_ROOT.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV_PATH.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_sweep() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    cases = build_sweep_cases()
    if not cases:
        raise ValueError("Sweep cases must not be empty")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_case = {
            executor.submit(build_result_row, case): case
            for case in cases
        }
        for future in as_completed(future_to_case):
            rows.append(future.result())

    rows.sort(
        key=lambda row: (
            str(row["hardware_type"]),
            int(row["num_ranks"]),
            int(row["input_sequence_length"]),
            int(row["output_sequence_length"]),
            int(row["batch_size"]),
        )
    )
    write_summary_csv(rows)

    if len(rows) != len(cases):
        raise ValueError(
            f"CSV row count must match successful case count, got rows={len(rows)}, cases={len(cases)}"
        )

    return rows


def main() -> None:
    rows = run_sweep()
    print(f"completed {len(rows)} sweep cases")
    print(f"summary csv: {SUMMARY_CSV_PATH}")


if __name__ == "__main__":
    main()
