from Desim import SimModule

from nandmachine.commands.micro import HwOp
from nandmachine.simulator.hardware.nand import NandController
from nandmachine.simulator.runtime.manager import RuntimeManager



class PerfetchEngine(SimModule):
    def __init__(self):
        super().__init__()

        # 负责处理发射 SramPrefetch 和 Release 请求
        # 向Nand Controller 发射细粒度的请求

        self.runtime_manager:RuntimeManager = None 

        self.nand_controller:NandController = None

        self.prefetch_commnad_queue:list[HwOp] = None 



    def process(self):
        
        
        # 首先从队列中拿出请求
        # 转换为 micro op
        # 一次性发射到 nand controller 中 
        # 执行结束，调用 runtime 中的函数，改写相关的参数
        
        # TODO
        # 需要考虑 Release 指令的执行
        
        pass 







class ComputeEngine(SimModule):
    def __init__(self):
        super().__init__()

        # 负责运算指令的进行，同时负责相关运算内部指令的发射

        # 暂时先采取roofline 模型
        # max（计算时间， 访存时间）


    def process(self):
        # 做好相关的同步 
        pass 

    def computation(self):
        pass 


    def memory_access(self):
        pass 
    pass


class xPU(SimModule):
    pass 