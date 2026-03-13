from __future__ import annotations
from dataclasses import dataclass, field
from typing import ClassVar

from typing import TypeAlias


LogicAddr:TypeAlias = int



@dataclass
class MacroOp:
    id: int = field(init=False)
    input_ops:list[MacroOp] = field(default_factory=list,init=False) 


    _global_id_counter: ClassVar[int] = 0



    @classmethod
    def _next_id(cls) -> int:
        MacroOp._global_id_counter += 1
        return MacroOp._global_id_counter

    def __post_init__(self) -> None:
        self.id = self._next_id()

# Runtime 之前的 kernel



@dataclass
class RuntimeCall(MacroOp):
    pass 



# ------  Nand Operation ------


 


# prefetch -- read only 
@dataclass 
class SramPrefetch(RuntimeCall):
    # prefetch_addr:LogicAddr  # logic addr
    # num_pages:int 

    # pre_alloc_logic_addr:LogicAddr     
    prefetch_size:int  # bytes 


@dataclass
class SramPrefetchRelease(RuntimeCall):
    # addr:LogicAddr  
    pass 


# ------ SRAM Operation --------





# ------- DRAM Operation ------ 



# -------- Read/Write Operation -------



# ------- xPU Operation ---------

@dataclass
class MatMulOp(MacroOp):

    shape:tuple[int,int,int] # m,k,n

    # addr:LogicAddr

    # weight_bits:int 


@dataclass 
class FlashAttnOp(MacroOp):
    qk_bmm_shape:tuple[int,int,int,int] # b,m,k,n 
    sv_bmm_shape:tuple[int,int,int,int]
    softmax_shape:tuple[int,int] # batch, length 


@dataclass 
class VectorOp(MacroOp):
    vector_op_type:str

    vector_shape:list[int]


# --------- Transfer Operations -------

@dataclass 
class AllReduceOp(MacroOp):
    pass 


@dataclass 
class All2AllOp(MacroOp):
    pass 








__all__ = [
    "MacroOp",
    "RuntimeCall",
    "NandMmap",
    "NandMunmap",
    "NandGroupArrange",
    "NandGroupMmap",
    "NandGroupMunmap",
    "NandGroupWrite",
    "SramPrefetch",
    "SramPrefetchRelease",
    "SramMalloc",
    "SramFree",
    "DramMalloc",
    "DramFree",
    "MatMul",
]