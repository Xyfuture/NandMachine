import math
from Desim import SimModule, SimSession, SimTime

from nandmachine.commands.macro import MacroOp, MatMul, NandMmap, RuntimeCall, SramPrefetch
from nandmachine.commands.micro import DataForward, MemoryOperation, NandPageRead, NandRequest, SramPageRead
from nandmachine.config.config import NandConfig
from nandmachine.simulator.hardware.nand import NandController
from nandmachine.simulator.hardware.utils import DepSlot
from nandmachine.simulator.runtime.manager import RuntimeManager
from nandmachine.simulator.runtime.tables import DeviceType



class PerfetchEngine(SimModule):
    def __init__(self,runtime_manager:RuntimeManager,nand_controller:NandController):
        super().__init__()

        # 负责处理发射 SramPrefetch 和 Release 请求
        # 向Nand Controller 发射细粒度的请求

        self.runtime_manager:RuntimeManager = runtime_manager 

        self.nand_controller:NandController = nand_controller

        self.prefetch_commnad_queue:list[DepSlot] = [] 

        self.register_coroutine(self.process)

    def process(self):
        
        
        # 首先从队列中拿出请求
        # 转换为 micro op
        # 一次性发射到 nand controller 中 
        # 执行结束，调用 runtime 中的函数，改写相关的参数
        
        # TODO
        # 需要考虑 Release 指令的执行

        for macro_op_slot in self.prefetch_commnad_queue:
            assert isinstance(macro_op_slot.payload,SramPrefetch)


            for input_slot in macro_op_slot.input_slots: # type: ignore
                if not input_slot.is_finished:
                    SimModule.wait(input_slot.finish_event)
            # 开始执行
            nand_request = self.decode_macro_op(macro_op_slot.payload)
            nand_request_slot = DepSlot(nand_request)

            self.nand_controller.handle_request(nand_request_slot)
            SimModule.wait(nand_request_slot.finish_event)

            self.runtime_manager.SramPrefetchHandler(macro_op_slot.payload)

            macro_op_slot.is_finished = True
            macro_op_slot.finish_event.notify(SimTime(1))

    
    def decode_macro_op(self,macro_op:MacroOp)->NandRequest: # type: ignore

        if isinstance(macro_op,SramPrefetch):
            prefetch_addr = macro_op.prefetch_addr
            num_pages = macro_op.num_pages

            memory_operation_list: list[MemoryOperation] = []

            for i in range(num_pages):
                phy_addr_info = self.runtime_manager.page_table.translate(prefetch_addr+i)            
                if not phy_addr_info:
                    raise RuntimeError(f"Cannot translate prefetch source page {prefetch_addr + i}")
                device_type , nand_addr = phy_addr_info
                if device_type != DeviceType.NAND:
                    raise RuntimeError(f"Prefetch source must be NAND, got {device_type}")

                memory_operation_list.append(
                    MemoryOperation(
                        NandPageRead(nand_addr),
                        DataForward("nand", "base", nand_addr, None),
                    )
                )
            nand_request = NandRequest(memory_operation_list)

            return nand_request
        raise TypeError(f"Unsupported macro op for prefetch engine: {type(macro_op)}")
                
    def load_command_queue(self,command_queue:list[DepSlot]):

        self.prefetch_commnad_queue = command_queue



class ComputeEngine(SimModule):
    def __init__(self,runtime_manager:RuntimeManager,nand_controller:NandController,nand_config:NandConfig):
        super().__init__()

        # 负责运算指令的进行，同时负责相关运算内部指令的发射

        # 暂时先采取roofline 模型
        # max（计算时间， 访存时间）

        self.config = nand_config

        self.runtime_manager:RuntimeManager = runtime_manager
        self.nand_controller:NandController = nand_controller
        
        self.command_queue:list[DepSlot[MacroOp]] = []


        self.register_coroutine(self.process)


    def process(self):
        # 做好相关的同步 
        for macro_op_slot in self.command_queue:
            for input_slot in macro_op_slot.input_slots:
                if not input_slot.is_finished:
                    SimModule.wait(input_slot.finish_event)
            
            # 开始处理， 应该只有 matmul 指令
            assert isinstance(macro_op_slot.payload,MatMul)

            self.execute_macro_op(macro_op_slot.payload)

            macro_op_slot.is_finished =True
            macro_op_slot.finish_event.notify(SimTime(1))

            


    def execute_macro_op(self,macro_op:MacroOp):
        
        if isinstance(macro_op,MatMul):

            assert len(macro_op.dim) == 3

            # 现在使用最简单的 roofline model 来确定执行的时间
            computational_capacity = 100 * (2**40) # 100 TOPS
            
            computation_ops = math.prod(macro_op.dim)

            compute_latency = int((computation_ops / computational_capacity) * 1e9) # ns

            compute_finish_time = SimTime(compute_latency) + SimSession.sim_time


            # 构造 memory access pattern 
            # 首先查询 对应地址
            # 访问 指定数量的 pages
            
            logic_addr_base = macro_op.addr

            matrix_size = macro_op.dim[1] * macro_op.dim[2]
            weight_bytes = matrix_size * macro_op.weight_bits // 8
            num_pages = (weight_bytes + self.config.page_size_bytes - 1) // self.config.page_size_bytes


            memory_operation_list: list[MemoryOperation] = []
            for i in range(num_pages):
                phy_addr_info = self.runtime_manager.page_table.translate(logic_addr_base+i)

                if not phy_addr_info:
                    raise RuntimeError(f"Cannot translate matmul page {logic_addr_base + i}")

                device_type, phy_addr = phy_addr_info

                if device_type == DeviceType.NAND:
                    memory_operation_list.append(
                        MemoryOperation(
                            NandPageRead(phy_addr),
                            DataForward("nand", "base", phy_addr, None),
                            DataForward("base", "xpu", None, None),
                        )
                    )
                elif device_type == DeviceType.SRAM:
                    memory_operation_list.append(
                        MemoryOperation(
                            SramPageRead(phy_addr),
                            DataForward("base", "xpu", None, None),
                        )
                    )
                else:
                    raise NotImplementedError(f"Unsupported matmul input device: {device_type}")

            nand_request = NandRequest(memory_operation_list)

            nand_request_slot = DepSlot(nand_request)

            self.nand_controller.handle_request(nand_request_slot)

            SimModule.wait(nand_request_slot.finish_event)

            if SimSession.sim_time  < compute_finish_time:
                SimModule.wait_time(compute_finish_time - SimSession.sim_time)

        else:
            raise TypeError(f"Unsupported macro op in compute engine: {type(macro_op)}")
            




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
        self.runtime_manager = RuntimeManager(self.nand_config)
        self.nand_controller = NandController(self.nand_config)


        # 异步执行的 engine
        self.compute_engine = ComputeEngine(self.runtime_manager,self.nand_controller,self.nand_config)
        self.prefetch_engine = PerfetchEngine(self.runtime_manager,self.nand_controller)


    def load_command(self,command_list:list[MacroOp]):
        # 这里要求 command list 里面 command 的顺序一定是 拓扑序的
        # 现阶段不要求 command list 里面的指令能构建成图，都是顺序发射执行的 (MacroOp 本身支持构件图)
        
        # 首先构建 dep slot 
        # 然后分发到不同的 Engine 中去执行

        prefetch_engine_slot_list:list[DepSlot[MacroOp]] = []
        compute_engine_slot_list:list[DepSlot[MacroOp]] = []
        manage_engine_slot_list:list[DepSlot[MacroOp]] = []


        pre_slot = None 
        for command in command_list:
            slot = DepSlot(command)
            if pre_slot:
                slot.input_slots = [pre_slot]
            pre_slot = slot

            # 分发到不同的 engine 中 
            if isinstance(command,SramPrefetch):
                prefetch_engine_slot_list.append(slot)
            elif isinstance(command,(NandMmap)):
                manage_engine_slot_list.append(slot)
            else:
                compute_engine_slot_list.append(slot)
        
        # 注入到不同的 engine 中
        self.prefetch_engine.load_command_queue(prefetch_engine_slot_list)
        self.compute_engine.load_command_queue(compute_engine_slot_list)
        self.manage_engine.load_command_queue(manage_engine_slot_list)
