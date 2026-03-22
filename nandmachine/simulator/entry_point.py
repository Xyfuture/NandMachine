from __future__ import annotations

from Desim import SimSession

from nandmachine.commands.macro import MacroOp
from nandmachine.config.config import NandConfig
from nandmachine.simulator.hardware.xpu import xPU


def run_macro_ops(
    nand_config: NandConfig,
    commands: list[MacroOp],
    *,
    device_name: str = "A100_80GB_fp16",
    compile_mode: str = "heuristic-GPU",
) -> int:
    SimSession.reset()
    SimSession.init()

    xpu = xPU(
        nand_config,
        device_name=device_name,
        compile_mode=compile_mode,
    )
    xpu.load_command(commands)
    SimSession.scheduler.run()

    return int(SimSession.sim_time.cycle)


__all__ = ["run_macro_ops"]
