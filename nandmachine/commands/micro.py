from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from Desim.Core import Event
from nandmachine.commands.macro import MacroOp


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


class DataForward:

    NandToBase = 0x1
    BaseToXPU= 0x2

    BaseToNand = 0x3
    XPUToBase = 0x4


    def __init__(self,direction):
        self.direction = direction  

 

class MemoryOperation:
    # 一个 memory operation 是一堆内存操作的集合 -- 顺序发生

    def __init__(self, *args):
        # Support both: MemoryOperation(op1, op2, op3) and MemoryOperation([op1, op2, op3])
        self.op_list:list[DataForward|MemoryBasicOpBase] = list(args)

        



class NandRequest:
    # 一条 request 可以包含多个 memory operation 

    def __init__(self,*args):
        if len(args) == 1 and isinstance(args[0],list):
            self.requests = args[0]
        else:
            self. requests:list[MemoryOperation] = list[args]

