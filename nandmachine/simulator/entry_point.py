from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from Desim import SimSession

from nandmachine.commands.macro import MacroOp
from nandmachine.config.config import NandConfig
from nandmachine.config.hardware_config import get_device_or_raise
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


__all__ = ["MacroSimResult", "run_macro_ops"]
