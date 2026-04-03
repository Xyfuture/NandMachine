from __future__ import annotations

from Desim import SimModule, SimTime

from nandmachine.commands.macro import (
    FlashAttnOp,
    MacroOp,
    MatMulOp,
    SramPrefetch,
    SramPrefetchRelease,
)
from nandmachine.config.config import NandConfig
from nandmachine.simulator.hardware.utils import DepSlot
from nandmachine.simulator.hardware.xpu import (
    ComputeEngine,
    TRANSFER_OP_TYPES,
    TransferEngine,
    _normalize_time_ns,
    xPU,
)


class VallinaPrefetchEngine(SimModule):
    def __init__(self):
        super().__init__()

        self.prefetch_command_queue: list[DepSlot[MacroOp]] = []
        self.release_command_queue: list[DepSlot[MacroOp]] = []

        self.register_coroutine(self.process)

    def process(self):
        for macro_op_slot in self.prefetch_command_queue:
            assert isinstance(macro_op_slot.payload, SramPrefetch)

            for input_slot in macro_op_slot.input_slots:
                if not input_slot.is_finished:
                    SimModule.wait(input_slot.finish_event)

            SimModule.wait_time(SimTime(1))
            macro_op_slot.is_finished = True
            macro_op_slot.finish_event.notify(SimTime(1))

    def load_command_queue(self, command_queue: list[DepSlot[MacroOp]]):
        self.prefetch_command_queue = command_queue


class VallinaComputeEngine(ComputeEngine):
    def execute_macro_op(self, macro_op: MacroOp) -> float:
        execute_time_ns = super().execute_macro_op(macro_op)

        if isinstance(macro_op, MatMulOp):
            return _normalize_time_ns(
                execute_time_ns + self.config.tRead,
                "vallina_matmul_time_ns",
            )

        if isinstance(macro_op, FlashAttnOp):
            return _normalize_time_ns(
                execute_time_ns + 2 * self.config.tRead,
                "vallina_flashattn_time_ns",
            )

        return execute_time_ns


class VallinaXPU(xPU):
    def __init__(
        self,
        nand_config: NandConfig,
        *,
        device_name: str = "A100_80GB",
        compile_mode: str = "heuristic-GPU",
    ):
        SimModule.__init__(self)

        self.nand_config = nand_config
        self.device_name = device_name
        self.compile_mode = compile_mode

        self.compute_engine = VallinaComputeEngine(
            self.nand_config,
            device_name=device_name,
            compile_mode=compile_mode,
        )
        self.transfer_engine = TransferEngine(
            device_name=device_name,
            compile_mode=compile_mode,
        )
        self.prefetch_engine = VallinaPrefetchEngine()

    def load_command(self, command_list: list[MacroOp]):
        prefetch_engine_slot_list: list[DepSlot[MacroOp]] = []
        compute_engine_slot_list: list[DepSlot[MacroOp]] = []
        transfer_engine_slot_list: list[DepSlot[MacroOp]] = []

        slot_map: dict[int, DepSlot[MacroOp]] = {}
        release_ids = {
            command.id for command in command_list if isinstance(command, SramPrefetchRelease)
        }

        for command in command_list:
            if command.id in slot_map:
                raise ValueError(f"Duplicate macro op id detected: {command.id}")
            slot_map[command.id] = DepSlot(command)

        for command in command_list:
            if any(input_op.id in release_ids for input_op in command.input_ops):
                raise NotImplementedError(
                    "SramPrefetchRelease cannot be used as a dependency yet"
                )

        for command in command_list:
            input_slots: list[DepSlot[MacroOp]] = []
            for input_op in command.input_ops:
                if input_op.id not in slot_map:
                    raise KeyError(
                        f"Input macro op id {input_op.id} was not provided to VallinaXPU.load_command"
                    )
                input_slots.append(slot_map[input_op.id])
            slot_map[command.id].input_slots = input_slots

        for command in command_list:
            slot = slot_map[command.id]

            if isinstance(command, SramPrefetch):
                prefetch_engine_slot_list.append(slot)
                continue
            if isinstance(command, SramPrefetchRelease):
                slot.is_finished = True
                continue
            if isinstance(command, TRANSFER_OP_TYPES):
                transfer_engine_slot_list.append(slot)
                continue
            compute_engine_slot_list.append(slot)

        self.prefetch_engine.load_command_queue(prefetch_engine_slot_list)
        self.transfer_engine.load_command_queue(transfer_engine_slot_list)
        self.compute_engine.load_command_queue(compute_engine_slot_list)


__all__ = [
    "VallinaPrefetchEngine",
    "VallinaComputeEngine",
    "VallinaXPU",
]
