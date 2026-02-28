from __future__ import annotations
from dataclasses import dataclass

from nandmachine.simulator.runtime.tables import DeviceType


# @dataclass
# class HwOp:
#     macro_op:MacroOp
#     is_finished:bool = False
#     finish_event:Event = Event()

#     input_ops:Optional[list[HwOp]] = None 


@dataclass
class MemoryBasicOpBase:
    pass 

# 拆分为更细粒度的 Read/Write 然后拼接组合实现功能 
# 每次读写都是一个 page 不需要手动指定尺寸了


@dataclass
class NandPageRead(MemoryBasicOpBase):
    addr:int 

@dataclass 
class NandPageWrite(MemoryBasicOpBase):
    addr:int

@dataclass
class NandBlockErase(MemoryBasicOpBase):
    block_addr:int


@dataclass 
class SramPageRead(MemoryBasicOpBase):
    addr:int

@dataclass 
class SramPageWrite(MemoryBasicOpBase):
    addr:int


@dataclass
class DataForward:
    """
    Data forwarding operation between hardware endpoints.

    约定：
    - `src_type` / `dst_type` 描述端点类型（如 nand/base/xpu/sram）
    - `src` / `dst` 保存端点值：
      - nand 端通常传物理地址（int）
      - base/xpu/sram 端通常传 channel（int）
    - src/dst 可以为 None，由执行器回退到上下文 channel
    """
    src_type: DeviceType
    dst_type: DeviceType
    src: int | None = None
    dst: int | None = None

 

class MemoryOperation:
    # 一个 memory operation 是一堆内存操作的集合 -- 顺序发生

    def __init__(self, *args):
        # Support both: MemoryOperation(op1, op2, op3) and MemoryOperation([op1, op2, op3])
        if len(args) == 1 and isinstance(args[0], list):
            self.op_list: list[DataForward | MemoryBasicOpBase] = args[0]
        else:
            self.op_list: list[DataForward | MemoryBasicOpBase] = list(args)

        



class NandRequest:
    # 一条 request 可以包含多个 memory operation 

    def __init__(self,*args):
        if len(args) == 1 and isinstance(args[0],list):
            self.operations = args[0]
        else:
            self.operations: list[MemoryOperation] = list(args)
