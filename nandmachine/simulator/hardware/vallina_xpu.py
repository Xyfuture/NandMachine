from __future__ import annotations

from pathlib import Path
from typing import Optional

from Desim import SimModule, SimTime
from perf_tracer import PerfettoTracer
from perf_tracer.tracer import TrackInfo

from nandmachine.commands.macro import (
    FlashAttnOp,
    FlashMLAOp,
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
    _get_current_sim_cycle,
    _normalize_time_ns,
    _record_macro_op_trace,
    _validate_trace_binding,
    xPU,
)


class VallinaPrefetchEngine(SimModule):
    def __init__(
        self,
        *,
        tracer: Optional[PerfettoTracer] = None,
        trace_track: Optional[TrackInfo] = None,
    ):
        super().__init__()

        self.tracer = tracer
        self.trace_track = trace_track
        _validate_trace_binding(self.tracer, self.trace_track, self.__class__.__name__)
        self.prefetch_command_queue: list[DepSlot[MacroOp]] = []
        self.release_command_queue: list[DepSlot[MacroOp]] = []

        self.register_coroutine(self.process)

    def process(self):
        for macro_op_slot in self.prefetch_command_queue:
            assert isinstance(macro_op_slot.payload, SramPrefetch)

            for input_slot in macro_op_slot.input_slots:
                if not input_slot.is_finished:
                    SimModule.wait(input_slot.finish_event)

            start_time_ns = _get_current_sim_cycle()
            SimModule.wait_time(SimTime(1))
            end_time_ns = _get_current_sim_cycle()
            _record_macro_op_trace(
                self.tracer,
                self.trace_track,
                macro_op_slot.payload,
                start_time_ns,
                end_time_ns,
                "prefetch",
            )
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

        if isinstance(macro_op, FlashMLAOp):
            return _normalize_time_ns(
                execute_time_ns + 2 * self.config.tRead,
                "vallina_flashmla_time_ns",
            )

        return execute_time_ns


class VallinaXPU(xPU):
    def __init__(
        self,
        nand_config: NandConfig,
        *,
        hbm_bandwidth_bytes_per_sec: float,
        device_name: str = "A100_80GB",
        compile_mode: str = "heuristic-GPU",
        enable_trace: bool = False,
    ):
        SimModule.__init__(self)

        self.nand_config = nand_config
        self.hbm_bandwidth_bytes_per_sec = hbm_bandwidth_bytes_per_sec
        self.device_name = device_name
        self.compile_mode = compile_mode
        self.enable_trace = enable_trace

        self.tracer: Optional[PerfettoTracer] = None
        self.trace_module_name: Optional[str] = None
        self.prefetch_trace_track: Optional[TrackInfo] = None
        self.compute_trace_track: Optional[TrackInfo] = None
        self.transfer_trace_track: Optional[TrackInfo] = None

        if self.enable_trace:
            self.tracer = PerfettoTracer(ns_per_cycle=1.0)
            self.trace_module_name = f"{self.__class__.__name__}:{id(self)}"
            trace_module = self.tracer.register_module(self.trace_module_name)
            self.prefetch_trace_track = self.tracer.register_track("prefetch_engine", trace_module)
            self.compute_trace_track = self.tracer.register_track("compute_engine", trace_module)
            self.transfer_trace_track = self.tracer.register_track("transfer_engine", trace_module)

        self.compute_engine = VallinaComputeEngine(
            self.nand_config,
            hbm_bandwidth_bytes_per_sec=hbm_bandwidth_bytes_per_sec,
            device_name=device_name,
            compile_mode=compile_mode,
            tracer=self.tracer,
            trace_track=self.compute_trace_track,
        )
        self.transfer_engine = TransferEngine(
            device_name=device_name,
            compile_mode=compile_mode,
            tracer=self.tracer,
            trace_track=self.transfer_trace_track,
        )
        self.prefetch_engine = VallinaPrefetchEngine(
            tracer=self.tracer,
            trace_track=self.prefetch_trace_track,
        )

    def save_trace_file(self, file_name: str) -> str:
        if self.tracer is None:
            raise RuntimeError("Tracing is disabled on this xPU instance")
        if not file_name:
            raise ValueError("file_name must be a non-empty string")

        output_path = Path(file_name)
        self.tracer.save(str(output_path))
        return str(output_path)

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
