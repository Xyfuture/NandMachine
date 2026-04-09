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
    VectorOp,
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
    def process(self):
        # Count flash attention ops once so the simplified pipeline rule can
        # identify whether the last flash op must keep the serial softmax tail.
        flash_op_count = sum(
            1 for macro_op_slot in self.command_queue if isinstance(macro_op_slot.payload, FlashAttnOp)
        )
        flash_op_index = 0

        # 做好相关的同步 
        for macro_op_slot in self.command_queue:
            for input_slot in macro_op_slot.input_slots:
                if not input_slot.is_finished:
                    SimModule.wait(input_slot.finish_event)
            
            start_cycle = _get_current_sim_cycle()
            if isinstance(macro_op_slot.payload, FlashAttnOp):
                qk_bmm_time_ns, softmax_time_ns, sv_bmm_time_ns = (
                    self._estimate_flashattn_component_times_ns(macro_op_slot.payload)
                )
                is_last_flash_op = flash_op_index == flash_op_count - 1

                # Pair every flash op with the next one and hide softmax behind
                # the longer SV stage. Only the last tail op keeps serial softmax.
                if flash_op_count % 2 == 1 and is_last_flash_op:
                    execute_time_ns = qk_bmm_time_ns + softmax_time_ns + sv_bmm_time_ns
                else:
                    execute_time_ns = qk_bmm_time_ns + max(softmax_time_ns, sv_bmm_time_ns)
                flash_op_index += 1
            else:
                execute_time_ns = self.execute_macro_op(macro_op_slot.payload)
            if not isinstance(macro_op_slot.payload,VectorOp):
                execute_time_ns += self.config.tRead
            wait_time_ns = _normalize_time_ns(execute_time_ns, "execute_time_ns")
            SimModule.wait_time(SimTime(wait_time_ns))
            end_cycle = _get_current_sim_cycle()
            _record_macro_op_trace(
                self.tracer,
                self.trace_track,
                macro_op_slot.payload,
                start_cycle,
                end_cycle,
                "compute",
            )
            macro_op_slot.finish_event.notify(SimTime(1))
            SimModule.wait(macro_op_slot.finish_event)
            macro_op_slot.is_finished = True

            print(macro_op_slot.payload)


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
