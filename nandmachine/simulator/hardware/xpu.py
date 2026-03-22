import math

from Desim import SimModule, SimTime

from nandmachine.commands.macro import (
    FlashAttnOp,
    MacroOp,
    MatMulOp,
    SramPrefetch,
    SramPrefetchRelease,
    VectorOp,
)
from nandmachine.config.config import NandConfig
from nandmachine.config.hardware_config import Device, device_dict
from nandmachine.simulator.hardware.nand import NandController
from nandmachine.simulator.hardware.utils import DepSlot
from nandmachine.simulator.software.matmul import MatMul_Simulation
from nandmachine.simulator.software.flash_attention import FlashAttn_BatchedMatMul_Simulation, Softmax_Simulation


class PerfetchEngine(SimModule):
    def __init__(self,nand_controller:NandController):
        super().__init__()

        # 负责处理发射 SramPrefetch 和 Release 请求
        # 向Nand Controller 发射细粒度的请求

        self.nand_controller:NandController = nand_controller

        self.prefetch_command_queue:list[DepSlot] = [] 

        self.release_command_queue:list[DepSlot] = []

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
            # 开始执行
            nand_request_slot = DepSlot(macro_op_slot.payload.num_prefetch_pages)

            self.nand_controller.handle_request(nand_request_slot)
            SimModule.wait(nand_request_slot.finish_event)

            macro_op_slot.is_finished = True
            macro_op_slot.finish_event.notify(SimTime(1))

    
                
    def load_command_queue(self,command_queue:list[DepSlot]):

        self.prefetch_command_queue = command_queue



class ComputeEngine(SimModule):
    def __init__(
        self,
        nand_config: NandConfig,
        *,
        device_name: str = "A100_80GB_fp16",
        compile_mode: str = "heuristic-GPU",
    ):
        super().__init__()

        # 负责运算指令的进行，同时负责相关运算内部指令的发射

        # 暂时先采取roofline 模型
        # max（计算时间， 访存时间）

        self.config = nand_config
        self.device_name = device_name
        if device_name not in device_dict:
            raise ValueError(f"Unsupported device_name: {device_name}")
        self.device: Device = device_dict[device_name]
        self.compile_mode = compile_mode

        
        self.command_queue:list[DepSlot[MacroOp]] = []


        self.register_coroutine(self.process)


    def process(self):
        # 做好相关的同步 
        for macro_op_slot in self.command_queue:
            for input_slot in macro_op_slot.input_slots:
                if not input_slot.is_finished:
                    SimModule.wait(input_slot.finish_event)
            
            cycles = self.execute_macro_op(macro_op_slot.payload)
            execute_cycles = max(1, math.ceil(cycles))
            SimModule.wait_time(SimTime(execute_cycles))
            macro_op_slot.finish_event.notify(SimTime(1))
            SimModule.wait(macro_op_slot.finish_event)
            macro_op_slot.is_finished = True

            

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

    def _estimate_vector_cycles(self, macro_op: VectorOp) -> float:
        self._validate_vector_shape(macro_op)

        total_elements = math.prod(macro_op.vector_shape)
        vector_flops_per_cycle = self.device.compute_module.total_vector_flops_per_cycle
        exp_flops = self.device.compute_module.core.vector_unit.flops_per_exp

        if macro_op.vector_op_type == "rms_norm":
            total_flops = total_elements * 8
        elif macro_op.vector_op_type == "silu_mul":
            total_flops = total_elements * (exp_flops + 6)
        else:
            raise TypeError(
                f"Unsupported vector op type: {macro_op.vector_op_type}"
            )

        return max(1.0, total_flops / vector_flops_per_cycle)

    def _bytes_per_value(self, weight_bits: int) -> int:
        if weight_bits <= 0 or weight_bits % 8 != 0:
            raise ValueError(
                f"Unsupported weight_bits={weight_bits}, expected a positive multiple of 8"
            )
        return weight_bits // 8

    def _estimate_matmul_cycles(self, macro_op: MatMulOp) -> float:
        m, k, n = macro_op.shape
        bytes_per_value = self._bytes_per_value(macro_op.weight_bits)
        array = self.device.compute_module.core.systolic_array
        systolic_flops_per_cycle = (
            array.array_height
            * array.array_width
            * array.mac_per_cycle
            * 2
            * self.device.compute_module.core.systolic_array_count
            * self.device.compute_module.core_count
        )
        io_bandwidth_per_cycle = (
            self.device.io_module.bandwidth / self.device.compute_module.clock_freq
        )

        total_flops = 2 * m * k * n
        io_bytes = (m * k + k * n + m * n) * bytes_per_value

        compute_cycles = total_flops / systolic_flops_per_cycle
        io_cycles = io_bytes / io_bandwidth_per_cycle
        return max(1.0, compute_cycles, io_cycles)

    def _estimate_flashattn_cycles(self, macro_op: FlashAttnOp) -> float:
        qk_b, qk_m, qk_k, qk_n = macro_op.qk_bmm_input_shape
        _, _, _, sv_k = macro_op.sv_bmm_input_shape
        softmax_m, softmax_n = macro_op.softmax_input_shape
        bytes_per_value = self._bytes_per_value(macro_op.weight_bits)
        vector_flops_per_cycle = self.device.compute_module.total_vector_flops_per_cycle
        exp_flops = self.device.compute_module.core.vector_unit.flops_per_exp
        io_bandwidth_per_cycle = (
            self.device.io_module.bandwidth / self.device.compute_module.clock_freq
        )

        qk_cycles = self._estimate_matmul_cycles(
            MatMulOp(dim=(qk_b * qk_m, qk_k, qk_n), weight_bits=macro_op.weight_bits)
        )
        sv_cycles = self._estimate_matmul_cycles(
            MatMulOp(dim=(qk_b * qk_m, qk_n, sv_k), weight_bits=macro_op.weight_bits)
        )

        softmax_flops = softmax_m * softmax_n * (exp_flops + 4)
        softmax_io_bytes = softmax_m * softmax_n * bytes_per_value * 2
        softmax_compute_cycles = softmax_flops / vector_flops_per_cycle
        softmax_io_cycles = softmax_io_bytes / io_bandwidth_per_cycle
        softmax_cycles = max(1.0, softmax_compute_cycles, softmax_io_cycles)

        return qk_cycles + softmax_cycles + sv_cycles

    def _is_invalid_cycle_count(self, cycles: float) -> bool:
        return not math.isfinite(cycles) or cycles <= 0 or cycles >= 2**63 - 1

    def execute_macro_op(self,macro_op:MacroOp)->float:
        
        """
        TODO yalong

        完善各个算子的时间，返回一个最终的延迟

        """

        if isinstance(macro_op, MatMulOp):
            try:
                matmul_sim = MatMul_Simulation.get_instance(
                    dim=macro_op.shape,
                    weight_bits=macro_op.weight_bits,
                )
                matmul_cycles = matmul_sim.compile_and_simulate(
                    pcb_module=self.device,
                    compile_mode=self.compile_mode,
                )
            except Exception:
                assert False
                return self._estimate_matmul_cycles(macro_op)

            if self._is_invalid_cycle_count(matmul_cycles):
                return self._estimate_matmul_cycles(macro_op)
            return matmul_cycles

        if isinstance(macro_op, FlashAttnOp):
            self._validate_flashattn_shapes(macro_op)

            try:
                qk_bmm_sim = FlashAttn_BatchedMatMul_Simulation.get_instance(
                    dim=macro_op.qk_bmm_input_shape,
                    matmul_type="QK",
                    weight_bits=macro_op.weight_bits,
                )
                qk_bmm_cycles = qk_bmm_sim.compile_and_simulate(
                    pcb_module=self.device,
                    compile_mode=self.compile_mode,
                )

                softmax_sim = Softmax_Simulation(
                    dim=macro_op.softmax_input_shape,
                    weight_bits=macro_op.weight_bits,
                )
                softmax_cycles = softmax_sim.compile_and_simulate(
                    pcb_module=self.device,
                    compile_mode=self.compile_mode,
                )
                sv_bmm_sim = FlashAttn_BatchedMatMul_Simulation.get_instance(
                    dim=macro_op.sv_bmm_input_shape,
                    matmul_type="SV",
                    weight_bits=macro_op.weight_bits,
                )
                sv_bmm_cycles = sv_bmm_sim.compile_and_simulate(
                    pcb_module=self.device,
                    compile_mode=self.compile_mode,
                )
                flashattnop_cycles = qk_bmm_cycles + softmax_cycles + sv_bmm_cycles
            except Exception:
                assert False
                return self._estimate_flashattn_cycles(macro_op)

            if self._is_invalid_cycle_count(flashattnop_cycles):
                return self._estimate_flashattn_cycles(macro_op)
            return flashattnop_cycles

        if isinstance(macro_op, VectorOp):
            return self._estimate_vector_cycles(macro_op)

        raise TypeError(f"Unsupported macro op type: {type(macro_op).__name__}")



    def load_command_queue(self,command_queue:list[DepSlot]):

        self.command_queue = command_queue



class TransferEgine(SimModule):
    def __init__(self):
        super().__init__()


        self.register_coroutine(self.process)
    
    def process(self):
        pass 
    


class xPU(SimModule):
    def __init__(
        self,
        nand_config: NandConfig,
        *,
        device_name: str = "A100_80GB_fp16",
        compile_mode: str = "heuristic-GPU",
    ):
        super().__init__()

        self.nand_config:NandConfig = nand_config
        self.device_name = device_name
        self.compile_mode = compile_mode

        # 在这里初始化 nand controller
        self.nand_controller = NandController(self.nand_config)


        # 异步执行的 engine
        self.compute_engine = ComputeEngine(
            self.nand_config,
            device_name=device_name,
            compile_mode=compile_mode,
        )
        self.prefetch_engine = PerfetchEngine(self.nand_controller)


    def load_command(self,command_list:list[MacroOp]):
        # 这里要求 command list 里面 command 的顺序一定是 拓扑序的
        # 现阶段不要求 command list 里面的指令能构建成图，都是顺序发射执行的 (MacroOp 本身支持构件图)
        
        # 首先构建 dep slot 
        # 然后分发到不同的 Engine 中去执行

        prefetch_engine_slot_list:list[DepSlot[MacroOp]] = []
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
            else:
                compute_engine_slot_list.append(slot)
        
        # 注入到不同的 engine 中
        self.prefetch_engine.load_command_queue(prefetch_engine_slot_list)
        self.compute_engine.load_command_queue(compute_engine_slot_list)
