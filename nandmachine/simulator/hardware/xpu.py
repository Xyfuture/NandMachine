import math
from pathlib import Path
from typing import Optional

from Desim import SimModule, SimSession, SimTime
from perf_tracer import PerfettoTracer
from perf_tracer.tracer import TrackInfo

from nandmachine.commands.macro import (
    All2AllOp,
    AllGatherOp,
    AllReduceOp,
    FlashAttnOp,
    FlashMLAOp,
    MacroOp,
    MatMulOp,
    ReduceScatterOp,
    SramPrefetch,
    SramPrefetchRelease,
    VectorOp,
)
from nandmachine.config.config import NandConfig
from nandmachine.config.hardware_config import Device, device_dict, get_device_or_raise
from nandmachine.config.interconnect_config import (
    TopologyType,
    get_interconnect_for_device_or_raise,
)
from nandmachine.simulator.hardware.nand import NandController
from nandmachine.simulator.hardware.utils import DepSlot
from nandmachine.simulator.software.communication_primitives_of_dense import (
    AllReduceSimulation,
)
from nandmachine.simulator.software.communication_primitives_of_MoE import (
    AllToAllPrimitive_Simulation,
)
from nandmachine.simulator.software.flash_attention import (
    FlashAttn_BatchedMatMul_Simulation,
    FlashMLA_BatchedMatMul_Simulation,
    Softmax_Simulation,
)
from nandmachine.simulator.software.matmul import MatMul_Simulation


TRANSFER_OP_TYPES = (AllReduceOp, AllGatherOp, ReduceScatterOp, All2AllOp)


def _validate_trace_binding(
    tracer: Optional[PerfettoTracer],
    trace_track: Optional[TrackInfo],
    engine_name: str,
) -> None:
    if (tracer is None) != (trace_track is None):
        raise ValueError(
            f"{engine_name} trace binding is invalid: tracer and trace_track must both be set or both be None"
        )


def _get_current_sim_cycle() -> int:
    return SimSession.sim_time.cycle


def _format_shape(shape: tuple[int, ...] | list[int]) -> str:
    return "x".join(str(value) for value in shape)


def _format_macro_op_trace_name(macro_op: MacroOp) -> str:
    if isinstance(macro_op, MatMulOp):
        m, k, n = macro_op.shape
        return f"MatMul[id={macro_op.id},m={m},k={k},n={n},bits={macro_op.weight_bits}]"

    if isinstance(macro_op, FlashAttnOp):
        qk_b, qk_m, qk_k, qk_n = macro_op.qk_bmm_input_shape
        sv_b, sv_m, sv_n, sv_k = macro_op.sv_bmm_input_shape
        softmax_m, softmax_n = macro_op.softmax_input_shape
        return (
            "FlashAttn["
            f"id={macro_op.id},"
            f"qk_b={qk_b},qk_m={qk_m},qk_k={qk_k},qk_n={qk_n},"
            f"sv_b={sv_b},sv_m={sv_m},sv_n={sv_n},sv_k={sv_k},"
            f"softmax_m={softmax_m},softmax_n={softmax_n},"
            f"bits={macro_op.weight_bits}"
            "]"
        )

    if isinstance(macro_op, FlashMLAOp):
        latent_b, latent_m, latent_k, latent_n = macro_op.qk_latent_bmm_input_shape
        rope_b, rope_m, rope_k, rope_n = macro_op.qk_rope_bmm_input_shape
        sv_b, sv_m, sv_n, sv_k = macro_op.sv_latent_bmm_input_shape
        softmax_m, softmax_n = macro_op.softmax_input_shape
        return (
            "FlashMLA["
            f"id={macro_op.id},"
            f"latent_b={latent_b},latent_m={latent_m},latent_k={latent_k},latent_n={latent_n},"
            f"rope_b={rope_b},rope_m={rope_m},rope_k={rope_k},rope_n={rope_n},"
            f"sv_b={sv_b},sv_m={sv_m},sv_n={sv_n},sv_k={sv_k},"
            f"softmax_m={softmax_m},softmax_n={softmax_n},"
            f"bits={macro_op.weight_bits}"
            "]"
        )

    if isinstance(macro_op, VectorOp):
        return (
            "Vector["
            f"id={macro_op.id},type={macro_op.vector_op_type},"
            f"shape={_format_shape(macro_op.vector_shape)},bits={macro_op.weight_bits}"
            "]"
        )

    if isinstance(macro_op, SramPrefetch):
        return f"SramPrefetch[id={macro_op.id},pages={macro_op.num_prefetch_pages}]"

    if isinstance(macro_op, AllReduceOp):
        return (
            f"AllReduce[id={macro_op.id},ranks={macro_op.num_ranks},"
            f"bytes={macro_op.data_size},bits={macro_op.weight_bits}]"
        )

    if isinstance(macro_op, AllGatherOp):
        return f"AllGather[id={macro_op.id},ranks={macro_op.num_ranks},bytes={macro_op.data_size}]"

    if isinstance(macro_op, ReduceScatterOp):
        return (
            f"ReduceScatter[id={macro_op.id},ranks={macro_op.num_ranks},bytes={macro_op.data_size}]"
        )

    if isinstance(macro_op, All2AllOp):
        return (
            f"All2All[id={macro_op.id},gpus={macro_op.num_gpus},"
            f"bytes={macro_op.data_size},bits={macro_op.weight_bits}]"
        )

    raise TypeError(f"Unsupported macro op type for trace formatting: {type(macro_op).__name__}")


def _record_macro_op_trace(
    tracer: Optional[PerfettoTracer],
    trace_track: Optional[TrackInfo],
    macro_op: MacroOp,
    start_cycle: int,
    end_cycle: int,
    category: str,
) -> None:
    if tracer is None:
        return
    if trace_track is None:
        raise ValueError("trace_track must be set when tracer is enabled")
    tracer.complete_event(
        trace_track,
        start_ts=float(start_cycle),
        end_ts=float(end_cycle),
        name=_format_macro_op_trace_name(macro_op),
        category=category,
    )


def _cycle_count_to_time_ns(cycle_count: float, device: Device) -> int:
    if not math.isfinite(cycle_count) or cycle_count <= 0:
        raise ValueError(f"cycle_count must be finite and > 0, got {cycle_count}")
    return max(1, math.ceil(cycle_count * 1e9 / device.compute_module.clock_freq))


def _normalize_time_ns(time_ns: float, name: str) -> int:
    if not math.isfinite(time_ns) or time_ns <= 0:
        raise ValueError(f"{name} must be finite and > 0, got {time_ns}")
    return max(1, math.ceil(time_ns))


class PerfetchEngine(SimModule):
    def __init__(
        self,
        nand_controller: NandController,
        *,
        tracer: Optional[PerfettoTracer] = None,
        trace_track: Optional[TrackInfo] = None,
    ):
        super().__init__()

        # 负责处理发射 SramPrefetch 和 Release 请求
        # 向Nand Controller 发射细粒度的请求

        self.nand_controller:NandController = nand_controller
        self.tracer = tracer
        self.trace_track = trace_track
        _validate_trace_binding(self.tracer, self.trace_track, self.__class__.__name__)

        self.prefetch_command_queue:list[DepSlot] = [] 

        self.release_command_queue:list[DepSlot] = []

        # special function to skip first prefetch in long pipeline
        self.is_first_prefetch = True


        self.register_coroutine(self.process)

    def process(self):
        
        
        # 首先从队列中拿出请求
        # 转换为 micro op
        # 一次性发射到 nand controller 中 
        # 执行结束，调用 runtime 中的函数，改写相关的参数
        
        # TODO
        # 暂时忽略 release 指令的执行 


        for macro_op_slot in self.prefetch_command_queue:
            assert isinstance(macro_op_slot.payload,SramPrefetch)


            for input_slot in macro_op_slot.input_slots: # type: ignore
                if not input_slot.is_finished:
                    SimModule.wait(input_slot.finish_event)

            # special function to skip first prefetch in long pipeline
            if self.is_first_prefetch:
                macro_op_slot.is_finished=True
                macro_op_slot.finish_event.notify(SimTime(1))
                self.is_first_prefetch = False
                continue


            # 开始执行
            start_cycle = _get_current_sim_cycle()
            nand_request_slot = DepSlot(macro_op_slot.payload.num_prefetch_pages)

            self.nand_controller.handle_request(nand_request_slot)

            SimModule.wait(nand_request_slot.finish_event)
            end_cycle = _get_current_sim_cycle()
            _record_macro_op_trace(
                self.tracer,
                self.trace_track,
                macro_op_slot.payload,
                start_cycle,
                end_cycle,
                "prefetch",
            )

            macro_op_slot.is_finished = True
            macro_op_slot.finish_event.notify(SimTime(1))

    
                
    def load_command_queue(self,command_queue:list[DepSlot]):

        self.prefetch_command_queue = command_queue



class ComputeEngine(SimModule):
    def __init__(
        self,
        nand_config: NandConfig,
        *,
        hbm_bandwidth_bytes_per_sec: float,
        device_name: str = "A100_80GB",
        compile_mode: str = "heuristic-GPU",
        tracer: Optional[PerfettoTracer] = None,
        trace_track: Optional[TrackInfo] = None,
    ):
        super().__init__()

        # 负责运算指令的进行，同时负责相关运算内部指令的发射

        # 暂时先采取roofline 模型
        # max（计算时间， 访存时间）

        self.config = nand_config
        self.hbm_bandwidth_bytes_per_sec = hbm_bandwidth_bytes_per_sec
        self.device_name = device_name
        self.device: Device = get_device_or_raise(device_name)
        self.compile_mode = compile_mode
        self.tracer = tracer
        self.trace_track = trace_track
        _validate_trace_binding(self.tracer, self.trace_track, self.__class__.__name__)

        
        self.command_queue:list[DepSlot[MacroOp]] = []


        self.register_coroutine(self.process)


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

            

    def _validate_flashattn_shapes(self, macro_op: FlashAttnOp) -> None: # flashattn中的矩阵shape合法性检查
        qk_b, qk_m, qk_k, qk_n = macro_op.qk_bmm_input_shape
        sv_b, sv_m, sv_n, sv_k = macro_op.sv_bmm_input_shape
        softmax_m, softmax_n = macro_op.softmax_input_shape

        dims = {
            "qk_b": qk_b,
            "qk_m": qk_m,
            "qk_k": qk_k,
            "qk_n": qk_n,
            "sv_b": sv_b,
            "sv_m": sv_m,
            "sv_n": sv_n,
            "sv_k": sv_k,
            "softmax_m": softmax_m,
            "softmax_n": softmax_n,
        }
        invalid_dims = [f"{name}={value}" for name, value in dims.items() if value <= 0]
        if invalid_dims:
            raise ValueError(
                "FlashAttnOp has non-positive dims: " + ", ".join(invalid_dims)
            )

        shape_errors = []
        valid_softmax_shapes = {
            (qk_m, qk_n),
            (qk_b * qk_m, qk_n),
        }
        if (softmax_m, softmax_n) not in valid_softmax_shapes:
            shape_errors.append(
                "softmax_shape="
                f"{macro_op.softmax_input_shape} must match either "
                f"(M,N)=({qk_m},{qk_n}) or flattened batched shape "
                f"({qk_b * qk_m},{qk_n})"
            )
        if sv_b != qk_b:
            shape_errors.append(f"sv_b({sv_b}) must equal qk_b({qk_b})")
        if sv_m != qk_m:
            shape_errors.append(f"sv_m({sv_m}) must equal qk_m({qk_m})")
        if sv_n != qk_n:
            shape_errors.append(f"sv_n({sv_n}) must equal qk_n({qk_n})")

        if shape_errors:
            raise ValueError("FlashAttnOp shape mismatch: " + "; ".join(shape_errors))

    def _validate_vector_shape(self, macro_op: VectorOp) -> None:
        invalid_dims = [value for value in macro_op.vector_shape if value <= 0]
        if invalid_dims:
            raise ValueError(
                f"VectorOp has non-positive dims: {macro_op.vector_shape}"
            )

    def _validate_flashmla_shapes(self, macro_op: FlashMLAOp) -> None:
        latent_b, latent_m, latent_k, latent_n = macro_op.qk_latent_bmm_input_shape
        rope_b, rope_m, rope_k, rope_n = macro_op.qk_rope_bmm_input_shape
        sv_b, sv_m, sv_n, sv_k = macro_op.sv_latent_bmm_input_shape
        softmax_m, softmax_n = macro_op.softmax_input_shape

        dims = {
            "latent_b": latent_b,
            "latent_m": latent_m,
            "latent_k": latent_k,
            "latent_n": latent_n,
            "rope_b": rope_b,
            "rope_m": rope_m,
            "rope_k": rope_k,
            "rope_n": rope_n,
            "sv_b": sv_b,
            "sv_m": sv_m,
            "sv_n": sv_n,
            "sv_k": sv_k,
            "softmax_m": softmax_m,
            "softmax_n": softmax_n,
        }
        invalid_dims = [f"{name}={value}" for name, value in dims.items() if value <= 0]
        if invalid_dims:
            raise ValueError(
                "FlashMLAOp has non-positive dims: " + ", ".join(invalid_dims)
            )

        shape_errors = []
        if (rope_b, rope_m, rope_n) != (latent_b, latent_m, latent_n):
            shape_errors.append(
                "qk_rope_bmm_shape batch dims must match qk_latent_bmm_shape"
            )
        valid_softmax_shapes = {
            (latent_m, latent_n),
            (latent_b * latent_m, latent_n),
        }
        # if (softmax_m, softmax_n) not in valid_softmax_shapes:
        #     shape_errors.append(
        #         "softmax_shape="
        #         f"{macro_op.softmax_input_shape} must match either "
        #         f"(M,N)=({latent_m},{latent_n}) or flattened batched shape "
        #         f"({latent_b * latent_m},{latent_n})"
        #     )
        if (sv_b, sv_m, sv_n, sv_k) != (latent_b, latent_m, latent_n, latent_k):
            shape_errors.append(
                "sv_latent_bmm_shape must match "
                f"(B,M,N,K)=({latent_b},{latent_m},{latent_n},{latent_k})"
            )

        if shape_errors:
            raise ValueError("FlashMLAOp shape mismatch: " + "; ".join(shape_errors))

    def _estimate_vector_cycles(self, macro_op: VectorOp) -> float:
        self._validate_vector_shape(macro_op)

        total_elements = math.prod(macro_op.vector_shape)
        vector_flops_per_cycle = self.device.compute_module.get_total_vector_flops_per_cycle(
            macro_op.weight_bits
        )
        exp_flops = self.device.compute_module.core.vector_unit.flops_per_exp

        if macro_op.vector_op_type == "rms_norm":
            total_flops = total_elements * 8
        elif macro_op.vector_op_type == "silu_mul":
            total_flops = total_elements * (exp_flops + 6)
        elif macro_op.vector_op_type == "moe_topk_router":
            total_flops = total_elements * (exp_flops + 8)
        elif macro_op.vector_op_type == "moe_weighted_sum":
            total_flops = total_elements * 4
        else:
            raise TypeError(
                f"Unsupported vector op type: {macro_op.vector_op_type}"
            )

        return max(1.0, total_flops / vector_flops_per_cycle)

    def _bytes_per_value(self, weight_bits: int) -> int:
        supported_weight_bits = {8, 16}
        if weight_bits not in supported_weight_bits:
            raise ValueError(
                f"Unsupported weight_bits={weight_bits}, expected one of {sorted(supported_weight_bits)}"
            )
        return weight_bits // 8

    def _estimate_flashattn_component_times_ns(
        self, macro_op: FlashAttnOp
    ) -> tuple[int, int, int]:
        self._validate_flashattn_shapes(macro_op)

        qk_bmm_sim = FlashAttn_BatchedMatMul_Simulation.get_instance(
            dim=macro_op.qk_bmm_input_shape,
            matmul_type="QK",
            weight_bits=macro_op.weight_bits,
        )
        qk_bmm_time_ns = qk_bmm_sim.compile_and_simulate(
            pcb_module=self.device,
            nand_config=self.config,
            hbm_bandwidth_bytes_per_sec=self.hbm_bandwidth_bytes_per_sec,
            compile_mode=self.compile_mode,
            return_unit="time_ns",
        )

        softmax_sim = Softmax_Simulation(
            dim=macro_op.softmax_input_shape,
            weight_bits=macro_op.weight_bits,
        )
        softmax_time_ns = softmax_sim.compile_and_simulate(
            pcb_module=self.device,
            compile_mode=self.compile_mode,
            return_unit="time_ns",
        )
        sv_bmm_sim = FlashAttn_BatchedMatMul_Simulation.get_instance(
            dim=macro_op.sv_bmm_input_shape,
            matmul_type="SV",
            weight_bits=macro_op.weight_bits,
        )
        sv_bmm_time_ns = sv_bmm_sim.compile_and_simulate(
            pcb_module=self.device,
            nand_config=self.config,
            hbm_bandwidth_bytes_per_sec=self.hbm_bandwidth_bytes_per_sec,
            compile_mode=self.compile_mode,
            return_unit="time_ns",
        )

        return (
            _normalize_time_ns(qk_bmm_time_ns, "qk_bmm_time_ns"),
            _normalize_time_ns(softmax_time_ns, "softmax_time_ns"),
            _normalize_time_ns(sv_bmm_time_ns, "sv_bmm_time_ns"),
        )

    def execute_macro_op(self,macro_op:MacroOp)->float:
        if isinstance(macro_op, MatMulOp):
            matmul_sim = MatMul_Simulation.get_instance(
                dim=macro_op.shape,
                weight_bits=macro_op.weight_bits,
            )
            matmul_time_ns = matmul_sim.compile_and_simulate(
                pcb_module=self.device,
                nand_config=self.config,
                hbm_bandwidth_bytes_per_sec=self.hbm_bandwidth_bytes_per_sec,
                compile_mode=self.compile_mode,
                return_unit="time_ns",
            )
            return _normalize_time_ns(matmul_time_ns, "matmul_time_ns")

        if isinstance(macro_op, FlashAttnOp):
            qk_bmm_time_ns, softmax_time_ns, sv_bmm_time_ns = (
                self._estimate_flashattn_component_times_ns(macro_op)
            )
            flashattn_time_ns = qk_bmm_time_ns + softmax_time_ns + sv_bmm_time_ns
            return _normalize_time_ns(flashattn_time_ns, "flashattn_time_ns")

        if isinstance(macro_op, FlashMLAOp):
            # self._validate_flashmla_shapes(macro_op)

            flashmla_sim = FlashMLA_BatchedMatMul_Simulation(
                qk_latent_dim=macro_op.qk_latent_bmm_input_shape,
                qk_rope_dim=macro_op.qk_rope_bmm_input_shape,
                sv_latent_dim=macro_op.sv_latent_bmm_input_shape,
                softmax_dim=macro_op.softmax_input_shape,
                weight_bits=macro_op.weight_bits,
            )
            flashmla_time_ns = flashmla_sim.compile_and_simulate(
                pcb_module=self.device,
                nand_config=self.config,
                hbm_bandwidth_bytes_per_sec=self.hbm_bandwidth_bytes_per_sec,
                compile_mode=self.compile_mode,
                return_unit="time_ns",
            )
            return _normalize_time_ns(flashmla_time_ns, "flashmla_time_ns")

        if isinstance(macro_op, VectorOp):
            vector_cycles = self._estimate_vector_cycles(macro_op)
            vector_time_ns = _cycle_count_to_time_ns(vector_cycles, self.device)
            return vector_time_ns

        raise TypeError(f"Unsupported macro op type: {type(macro_op).__name__}")



    def load_command_queue(self,command_queue:list[DepSlot]):

        self.command_queue = command_queue



class TransferEngine(SimModule):
    def __init__(
        self,
        *,
        device_name: str = "A100_80GB",
        interconnect_topology: TopologyType = TopologyType.FC,
        compile_mode: str = "heuristic-GPU",
        tracer: Optional[PerfettoTracer] = None,
        trace_track: Optional[TrackInfo] = None,
    ):
        super().__init__()

        if device_name not in device_dict:
            raise ValueError(f"Unsupported device_name: {device_name}")

        self.device_name = device_name
        self.interconnect_topology = interconnect_topology
        self.compile_mode = compile_mode
        self.device: Device = device_dict[device_name]
        self.tracer = tracer
        self.trace_track = trace_track
        _validate_trace_binding(self.tracer, self.trace_track, self.__class__.__name__)
        self.transfer_command_queue:list[DepSlot[MacroOp]] = []


        self.register_coroutine(self.process)
    
    def process(self):
        for macro_op_slot in self.transfer_command_queue:
            for input_slot in macro_op_slot.input_slots:
                if not input_slot.is_finished:
                    SimModule.wait(input_slot.finish_event)

            start_cycle = _get_current_sim_cycle()
            execute_time_ns = self.execute_macro_op(macro_op_slot.payload)
            wait_time_ns = _normalize_time_ns(execute_time_ns, "execute_time_ns")
            SimModule.wait_time(SimTime(wait_time_ns))
            end_cycle = _get_current_sim_cycle()
            _record_macro_op_trace(
                self.tracer,
                self.trace_track,
                macro_op_slot.payload,
                start_cycle,
                end_cycle,
                "transfer",
            )
            macro_op_slot.finish_event.notify(SimTime(1))
            SimModule.wait(macro_op_slot.finish_event)
            macro_op_slot.is_finished = True

    def _bytes_per_value(self, weight_bits: int) -> int:
        supported_weight_bits = {8, 16}
        if weight_bits not in supported_weight_bits:
            raise ValueError(
                f"Unsupported weight_bits={weight_bits}, expected one of {sorted(supported_weight_bits)}"
            )
        return weight_bits // 8

    def _estimate_allreduce_time_ns(self, macro_op: AllReduceOp) -> int:
        if macro_op.num_ranks == 1 or macro_op.data_size == 0:
            return 1

        interconnect = get_interconnect_for_device_or_raise(
            device_name=self.device_name,
            device_count=macro_op.num_ranks,
            topology=self.interconnect_topology,
        )

        word_size = self._bytes_per_value(macro_op.weight_bits)
        if macro_op.data_size % word_size != 0:
            raise ValueError(
                "AllReduceOp.data_size must be divisible by source word size. "
                f"data_size={macro_op.data_size}, source_word_size={word_size}"
            )

        allreduce_sim = AllReduceSimulation(
            num_gpus=macro_op.num_ranks,
            data_size=macro_op.data_size,
            weight_bits=macro_op.weight_bits,
        )
        allreduce_time_ns = allreduce_sim.compile_and_simulate(
            pcb_module=self.device,
            interconnect_module=interconnect,
            compile_mode=self.compile_mode,
            return_unit="time_ns",
        )
        return _normalize_time_ns(allreduce_time_ns, "allreduce_time_ns")

    def _estimate_all2all_time_ns(self, macro_op: All2AllOp) -> int:
        if macro_op.num_gpus == 1 or macro_op.data_size == 0:
            return 1

        interconnect = get_interconnect_for_device_or_raise(
            device_name=self.device_name,
            device_count=macro_op.num_gpus,
        )

        word_size = self._bytes_per_value(macro_op.weight_bits)
        if macro_op.data_size % word_size != 0:
            raise ValueError(
                "All2AllOp.data_size must be divisible by source word size. "
                f"data_size={macro_op.data_size}, source_word_size={word_size}"
            )

        all2all_sim = AllToAllPrimitive_Simulation(
            num_gpus=macro_op.num_gpus,
            data_size=macro_op.data_size,
            weight_bits=macro_op.weight_bits,
        )
        all2all_time_ns = all2all_sim.compile_and_simulate(
            pcb_module=self.device,
            interconnect_module=interconnect,
            compile_mode=self.compile_mode,
            return_unit="time_ns",
        )

        # if macro_op.num_gpus <= 8 :
        #     all2all_time_ns = macro_op.data_size / (450) 
        # else:
        #     all2all_time_ns = macro_op.data_size / (40)

        return _normalize_time_ns(all2all_time_ns, "all2all_time_ns")

    def execute_macro_op(self,macro_op:MacroOp)->float:
        # 使用 llm compass 的模拟器，实现基础通信原语的时间仿真，返回 ns
        if isinstance(macro_op, AllReduceOp):
            return self._estimate_allreduce_time_ns(macro_op)

        if isinstance(macro_op, All2AllOp):
            return self._estimate_all2all_time_ns(macro_op)

        if isinstance(macro_op, TRANSFER_OP_TYPES):
            return 1.0

        raise TypeError(f"Unsupported macro op type: {type(macro_op).__name__}")


    def load_command_queue(self,command_queue:list[DepSlot[MacroOp]]):

        self.transfer_command_queue = command_queue
    


class xPU(SimModule):
    def __init__(
        self,
        nand_config: NandConfig,
        hbm_bandwidth_bytes_per_sec: float,
        device_name: str = "A100_80GB",
        interconnect_topology: TopologyType = TopologyType.FC,
        compile_mode: str = "heuristic-GPU",
        enable_trace: bool = False,
    ):
        super().__init__()

        self.nand_config:NandConfig = nand_config
        self.hbm_bandwidth_bytes_per_sec = hbm_bandwidth_bytes_per_sec
        self.device_name = device_name
        self.interconnect_topology = interconnect_topology
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

        # 在这里初始化 nand controller
        self.nand_controller = NandController(self.nand_config)


        # 异步执行的 engine
        self.compute_engine = ComputeEngine(
            self.nand_config,
            hbm_bandwidth_bytes_per_sec=hbm_bandwidth_bytes_per_sec,
            device_name=device_name,
            compile_mode=compile_mode,
            tracer=self.tracer,
            trace_track=self.compute_trace_track,
        )
        self.transfer_engine = TransferEngine(
            device_name=device_name,
            interconnect_topology=interconnect_topology,
            compile_mode=compile_mode,
            tracer=self.tracer,
            trace_track=self.transfer_trace_track,
        )
        self.prefetch_engine = PerfetchEngine(
            self.nand_controller,
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


    def load_command(self,command_list:list[MacroOp]):
        # 这里要求 command list 里面 command 的顺序一定是 拓扑序的
        # 现阶段不要求 command list 里面的指令能构建成图，都是顺序发射执行的 (MacroOp 本身支持构件图)
        
        # 首先构建 dep slot 
        # 然后分发到不同的 Engine 中去执行

        prefetch_engine_slot_list:list[DepSlot[MacroOp]] = []
        transfer_engine_slot_list:list[DepSlot[MacroOp]] = []
        compute_engine_slot_list:list[DepSlot[MacroOp]] = []

        slot_map:dict[int,DepSlot[MacroOp]] = {}
        release_ids = {
            command.id for command in command_list if isinstance(command, SramPrefetchRelease)
        }

        for command in command_list:
            if command.id in slot_map:
                raise ValueError(f"Duplicate macro op id detected: {command.id}")
            slot = DepSlot(command)
            slot_map[command.id] = slot

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
                        f"Input macro op id {input_op.id} was not provided to xPU.load_command"
                    )
                input_slots.append(slot_map[input_op.id])
            slot_map[command.id].input_slots = input_slots

        for command in command_list:
            slot = slot_map[command.id]

            # 分发到不同的 engine 中 
            if isinstance(command,SramPrefetch):
                prefetch_engine_slot_list.append(slot)
            elif isinstance(command,SramPrefetchRelease):
                slot.is_finished = True
                continue
            elif isinstance(command,TRANSFER_OP_TYPES):
                transfer_engine_slot_list.append(slot)
            else:
                compute_engine_slot_list.append(slot)
        
        # 注入到不同的 engine 中
        self.prefetch_engine.load_command_queue(prefetch_engine_slot_list)
        self.transfer_engine.load_command_queue(transfer_engine_slot_list)
        self.compute_engine.load_command_queue(compute_engine_slot_list)
