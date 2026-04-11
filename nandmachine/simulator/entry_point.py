from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Literal

from Desim import SimSession

from nandmachine.commands.macro import MacroOp
from nandmachine.config.cache_state import KVCacheState
from nandmachine.config.config import NandConfig
from nandmachine.config.hardware_config import get_device_or_raise
from nandmachine.config.inference_config import InferenceConfig
from nandmachine.config.model_config import ModelConfigBase
from nandmachine.frontend.utlis import build_kv_cache_state
from nandmachine.simulator.hardware.vallina_xpu import VallinaXPU
from nandmachine.simulator.hardware.xpu import xPU


@dataclass(frozen=True)
class MacroSimResult:
    cycle: int
    time_ns: int


XPUType = Literal["default", "vallina"]


def _get_xpu_class(xpu_type: XPUType) -> type[xPU]:
    if xpu_type == "default":
        return xPU
    if xpu_type == "vallina":
        return VallinaXPU
    raise ValueError(f"Unsupported xpu_type: {xpu_type}")


def _run_macro_ops_with_xpu(
    nand_config: NandConfig,
    commands: list[MacroOp],
    *,
    hbm_bandwidth_bytes_per_sec: float,
    device_name: str,
    compile_mode: str,
    xpu_type: XPUType,
) -> MacroSimResult:
    sim_xpu_class = _get_xpu_class(xpu_type)

    SimSession.reset()
    SimSession.init()

    sim_xpu = sim_xpu_class(
        nand_config,
        hbm_bandwidth_bytes_per_sec=hbm_bandwidth_bytes_per_sec,
        device_name=device_name,
        compile_mode=compile_mode,
    )
    sim_xpu.load_command(commands)
    SimSession.scheduler.run()

    final_time_ns = int(SimSession.sim_time.cycle)
    device = get_device_or_raise(device_name)
    final_cycle = ceil(final_time_ns * device.compute_module.clock_freq / 1e9)
    return MacroSimResult(cycle=final_cycle, time_ns=final_time_ns)


def run_macro_ops(
    nand_config: NandConfig,
    commands: list[MacroOp],
    *,
    hbm_bandwidth_bytes_per_sec: float,
    device_name: str = "A100_80GB",
    compile_mode: str = "heuristic-GPU",
) -> MacroSimResult:
    return _run_macro_ops_with_xpu(
        nand_config,
        commands,
        hbm_bandwidth_bytes_per_sec=hbm_bandwidth_bytes_per_sec,
        device_name=device_name,
        compile_mode=compile_mode,
        xpu_type="default",
    )


@dataclass
class SimResult:
    layer_latency_ns: int
    model_latency_ns: int
    model_throughput: float  # tokens/s
    throughput_per_GPU: float  # tokens/s/GPU

    kv_cache_total_size_GB: float  # Total KV cache size across all layers.


def _validate_run_sim_inputs(
    model_config: ModelConfigBase,
    inference_config: InferenceConfig,
    commands: list[MacroOp],
) -> tuple[int, int]:
    if not commands:
        raise ValueError("commands must not be empty")

    if inference_config.batch_size <= 0:
        raise ValueError(
            f"batch_size must be > 0, got {inference_config.batch_size}"
        )

    num_ranks = inference_config.parallel_config.num_ranks
    if not isinstance(num_ranks, int):
        raise TypeError(
            "inference_config.parallel_config.num_ranks must be an int, "
            f"got {type(num_ranks).__name__}"
        )
    if num_ranks <= 0:
        raise ValueError(f"num_ranks must be > 0, got {num_ranks}")

    if not hasattr(model_config, "num_hidden_layers"):
        raise ValueError("model_config.num_hidden_layers must be set")

    num_hidden_layers = getattr(model_config, "num_hidden_layers")
    if not isinstance(num_hidden_layers, int):
        raise TypeError(
            "model_config.num_hidden_layers must be an int, "
            f"got {type(num_hidden_layers).__name__}"
        )
    if num_hidden_layers <= 0:
        raise ValueError(
            f"model_config.num_hidden_layers must be > 0, got {num_hidden_layers}"
        )

    return num_ranks, num_hidden_layers


def _resolve_kv_cache_state(
    nand_config: NandConfig,
    model_config: ModelConfigBase,
    inference_config: InferenceConfig,
    kv_cache_state: KVCacheState | None,
) -> KVCacheState:
    if kv_cache_state is None:
        return build_kv_cache_state(
            nand_config,
            model_config,
            inference_config,
        )
    return kv_cache_state


def _build_sim_result(
    *,
    nand_config: NandConfig,
    model_config: ModelConfigBase,
    inference_config: InferenceConfig,
    macro_result: MacroSimResult,
    num_ranks: int,
    num_hidden_layers: int,
    kv_cache_state: KVCacheState | None,
) -> SimResult:
    resolved_kv_cache_state = _resolve_kv_cache_state(
        nand_config,
        model_config,
        inference_config,
        kv_cache_state,
    )
    total_kv_cache_size_per_layer = (
        resolved_kv_cache_state.total_kv_cache_size_per_layer
    )
    if not isinstance(total_kv_cache_size_per_layer, int):
        raise TypeError(
            "kv_cache_state.total_kv_cache_size_per_layer must be an int, "
            f"got {type(total_kv_cache_size_per_layer).__name__}"
        )
    if total_kv_cache_size_per_layer <= 0:
        raise ValueError(
            "kv_cache_state.total_kv_cache_size_per_layer must be > 0, "
            f"got {total_kv_cache_size_per_layer}"
        )

    layer_latency_ns = macro_result.time_ns
    model_latency_ns = layer_latency_ns * num_hidden_layers
    if model_latency_ns <= 0:
        raise ValueError(
            f"model_latency_ns must be > 0, got {model_latency_ns}"
        )

    total_kv_cache_bytes = total_kv_cache_size_per_layer * num_hidden_layers
    model_throughput = inference_config.batch_size * 1e9 / model_latency_ns
    throughput_per_gpu = model_throughput / num_ranks
    kv_cache_total_size_gb = total_kv_cache_bytes / (1024 ** 3)

    return SimResult(
        layer_latency_ns=layer_latency_ns,
        model_latency_ns=model_latency_ns,
        model_throughput=model_throughput,
        throughput_per_GPU=throughput_per_gpu,
        kv_cache_total_size_GB=kv_cache_total_size_gb,
    )


def universe_run_sim(
    nand_config: NandConfig,
    model_config: ModelConfigBase,
    inference_config: InferenceConfig,
    commands: list[MacroOp],
    *,
    hbm_bandwidth_bytes_per_sec: float,
    device_name: str = "A100_80GB",
    compile_mode: str = "heuristic-GPU",
    xpu_type: XPUType = "default",
    kv_cache_state: KVCacheState | None = None,
) -> SimResult:
    num_ranks, num_hidden_layers = _validate_run_sim_inputs(
        model_config,
        inference_config,
        commands,
    )
    macro_result = _run_macro_ops_with_xpu(
        nand_config,
        commands,
        hbm_bandwidth_bytes_per_sec=hbm_bandwidth_bytes_per_sec,
        device_name=device_name,
        compile_mode=compile_mode,
        xpu_type=xpu_type,
    )
    return _build_sim_result(
        nand_config=nand_config,
        model_config=model_config,
        inference_config=inference_config,
        macro_result=macro_result,
        num_ranks=num_ranks,
        num_hidden_layers=num_hidden_layers,
        kv_cache_state=kv_cache_state,
    )


def run_sim(
    nand_config: NandConfig,
    model_config: ModelConfigBase,
    inference_config: InferenceConfig,
    commands: list[MacroOp],
    *,
    hbm_bandwidth_bytes_per_sec: float,
    device_name: str = "A100_80GB",
    compile_mode: str = "heuristic-GPU",
) -> SimResult:
    return universe_run_sim(
        nand_config,
        model_config,
        inference_config,
        commands,
        hbm_bandwidth_bytes_per_sec=hbm_bandwidth_bytes_per_sec,
        device_name=device_name,
        compile_mode=compile_mode,
        xpu_type="default",
    )


__all__ = [
    "MacroSimResult",
    "SimResult",
    "run_macro_ops",
    "run_sim",
    "universe_run_sim",
]
