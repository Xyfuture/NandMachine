from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from Desim import SimSession

from nandmachine.commands.macro import MacroOp
from nandmachine.config.config import NandConfig
from nandmachine.config.hardware_config import get_device_or_raise
from nandmachine.config.inference_config import InferenceConfig
from nandmachine.config.model_config import ModelConfigBase
from nandmachine.frontend.utlis import build_kv_cache_state
from nandmachine.simulator.hardware.xpu import xPU


@dataclass(frozen=True)
class MacroSimResult:
    cycle: int
    time_ns: int


def run_macro_ops(
    nand_config: NandConfig,
    commands: list[MacroOp],
    *,
    hbm_bandwidth_bytes_per_sec: float,
    device_name: str = "A100_80GB",
    compile_mode: str = "heuristic-GPU",
) -> MacroSimResult:
    SimSession.reset()
    SimSession.init()

    xpu = xPU(
        nand_config,
        hbm_bandwidth_bytes_per_sec=hbm_bandwidth_bytes_per_sec,
        device_name=device_name,
        compile_mode=compile_mode,
    )
    xpu.load_command(commands)
    SimSession.scheduler.run()

    final_time_ns = int(SimSession.sim_time.cycle)
    device = get_device_or_raise(device_name)
    final_cycle = ceil(final_time_ns * device.compute_module.clock_freq / 1e9)
    return MacroSimResult(cycle=final_cycle, time_ns=final_time_ns)


@dataclass
class SimResult:
    layer_latency_ns: int
    model_latency_ns: int
    model_throughput: float  # tokens/s

    kv_cache_total_size_GB: float  # Total KV cache size across all layers.


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
    if not commands:
        raise ValueError("commands must not be empty")

    if inference_config.batch_size <= 0:
        raise ValueError(
            f"batch_size must be > 0, got {inference_config.batch_size}"
        )

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

    macro_result = run_macro_ops(
        nand_config,
        commands,
        hbm_bandwidth_bytes_per_sec=hbm_bandwidth_bytes_per_sec,
        device_name=device_name,
        compile_mode=compile_mode,
    )

    layer_latency_ns = macro_result.time_ns
    model_latency_ns = layer_latency_ns * num_hidden_layers
    if model_latency_ns <= 0:
        raise ValueError(
            f"model_latency_ns must be > 0, got {model_latency_ns}"
        )

    kv_cache_state = build_kv_cache_state(
        nand_config,
        model_config,
        inference_config,
    )
    total_kv_cache_bytes = (
        kv_cache_state.total_kv_cache_size_per_layer * num_hidden_layers
    )
    model_throughput = inference_config.batch_size * 1e9 / model_latency_ns
    kv_cache_total_size_gb = total_kv_cache_bytes / (1024 ** 3)

    return SimResult(
        layer_latency_ns=layer_latency_ns,
        model_latency_ns=model_latency_ns,
        model_throughput=model_throughput,
        kv_cache_total_size_GB=kv_cache_total_size_gb,
    )


__all__ = ["MacroSimResult", "SimResult", "run_macro_ops", "run_sim"]
