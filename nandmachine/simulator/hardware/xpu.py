import math
from Desim import SimModule, SimSession, SimTime

from nandmachine.commands.macro import FlashAttnOp, MacroOp, MatMulOp, RuntimeCall, SramPrefetch, SramPrefetchRelease
from nandmachine.commands.micro import DataForward, MemoryOperation, NandPageRead, NandRequest, SramPageRead
from nandmachine.config.config import NandConfig
from nandmachine.simulator.hardware.nand import NandController
from nandmachine.simulator.hardware.utils import DepSlot

from nandmachine.config.hardware_config import device_dict
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


        for macro_op_slot in self.prefetch_commnad_queue:
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

        self.prefetch_commnad_queue = command_queue



class ComputeEngine(SimModule):
    def __init__(self,nand_config:NandConfig):
        super().__init__()

        # 负责运算指令的进行，同时负责相关运算内部指令的发射

        # 暂时先采取roofline 模型
        # max（计算时间， 访存时间）

        self.config = nand_config
        self.device = device_dict["A100_80GB_fp16"]
        self.compile_mode = "exhaustive"

        
        self.command_queue:list[DepSlot[MacroOp]] = []


        self.register_coroutine(self.process)


    def process(self):
        # 做好相关的同步 
        for macro_op_slot in self.command_queue:
            for input_slot in macro_op_slot.input_slots:
                if not input_slot.is_finished:
                    SimModule.wait(input_slot.finish_event)
            
            # 开始处理， 应该只有 matmul 指令
            # assert isinstance(macro_op_slot.payload,MatMul)

            cycles = self.execute_macro_op(macro_op_slot.payload)
            execute_cycles = max(1, math.ceil(cycles))
            macro_op_slot.finish_event.notify(SimTime(execute_cycles))
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
        if (softmax_m, softmax_n) != (qk_m, qk_n):
            shape_errors.append(
                f"softmax_shape={macro_op.softmax_input_shape} must match qk output (M,N)=({qk_m},{qk_n})"
            )
        if sv_b != qk_b:
            shape_errors.append(f"sv_b({sv_b}) must equal qk_b({qk_b})")
        if sv_m != qk_m:
            shape_errors.append(f"sv_m({sv_m}) must equal qk_m({qk_m})")
        if sv_n != qk_n:
            shape_errors.append(f"sv_n({sv_n}) must equal qk_n({qk_n})")

        if shape_errors:
            raise ValueError("FlashAttnOp shape mismatch: " + "; ".join(shape_errors))


    def execute_macro_op(self,macro_op:MacroOp)->float:
        
        """
        TODO yalong

        完善各个算子的时间，返回一个最终的延迟

        """

        if isinstance(macro_op, MatMulOp):
            matmul_sim = MatMul_Simulation(dim = macro_op.shape, weight_bits = macro_op.weight_bits)
            matmul_cycles = matmul_sim.compile_and_simulate(
                pcb_module=self.device,
                compile_mode=self.compile_mode,
            )

            return matmul_cycles

        if isinstance(macro_op, FlashAttnOp):
            self._validate_flashattn_shapes(macro_op)

            qk_bmm_sim = FlashAttn_BatchedMatMul_Simulation(
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
            sv_bmm_sim = FlashAttn_BatchedMatMul_Simulation(
                dim=macro_op.sv_bmm_input_shape,
                matmul_type="SV",
                weight_bits=macro_op.weight_bits,
            )
            sv_bmm_cycles = sv_bmm_sim.compile_and_simulate(
                pcb_module=self.device,
                compile_mode=self.compile_mode,
            )

            flashattnop_cycles = qk_bmm_cycles + softmax_cycles + sv_bmm_cycles
            return flashattnop_cycles

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
    def __init__(self,nand_config:NandConfig):
        super().__init__()

        self.nand_config:NandConfig = nand_config

        # 在这里初始化 runtime manager 和  nand controller
        self.nand_controller = NandController(self.nand_config)


        # 异步执行的 engine
        self.compute_engine = ComputeEngine(self.nand_config)
        self.prefetch_engine = PerfetchEngine(self.nand_controller)


    def load_command(self,command_list:list[MacroOp]):
        # 这里要求 command list 里面 command 的顺序一定是 拓扑序的
        # 现阶段不要求 command list 里面的指令能构建成图，都是顺序发射执行的 (MacroOp 本身支持构件图)
        
        # 首先构建 dep slot 
        # 然后分发到不同的 Engine 中去执行

        prefetch_engine_slot_list:list[DepSlot[MacroOp]] = []
        compute_engine_slot_list:list[DepSlot[MacroOp]] = []

        slot_map:dict[MacroOp,DepSlot[MacroOp]] = {}

        for command in command_list:
            slot = DepSlot(command)
            slot_map[command] = slot

        for command in command_list:
            input_slots = [slot_map[input_op] for input_op in command.input_ops]
            slot_map[command].input_slots = input_slots

        for command,slot in slot_map.items():

            # 分发到不同的 engine 中 
            if isinstance(command,SramPrefetch):
                prefetch_engine_slot_list.append(slot)
            elif isinstance(command,SramPrefetchRelease):
                continue
            else:
                compute_engine_slot_list.append(slot)
        
        # 注入到不同的 engine 中
        self.prefetch_engine.load_command_queue(prefetch_engine_slot_list)
        self.compute_engine.load_command_queue(compute_engine_slot_list)
