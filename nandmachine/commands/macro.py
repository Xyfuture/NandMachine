from __future__ import annotations
from dataclasses import dataclass, field
from typing import ClassVar

from typing import Optional


@dataclass
class MacroOp:
    id: int = field(init=False)
    input_ops:Optional[list[MacroOp]] = None 


    _global_id_counter: ClassVar[int] = 0



    @classmethod
    def _next_id(cls) -> int:
        MacroOp._global_id_counter += 1
        return MacroOp._global_id_counter

    def __post_init__(self) -> None:
        self.id = self._next_id()

# Runtime 之前的 kernel


@dataclass
class NandCreateFile(MacroOp):
    # 创建权重文件 -- 仅仅供 mapper 使用
    # 需要单独的指令吗
    pass 


@dataclass
class RuntimeCall(MacroOp):
    pass 



# ------  Nand Operation ------
@dataclass
class NandMmap(RuntimeCall):
    file_id:int 

    pre_alloc_logic_addr:int 


@dataclass
class NandMunmap(RuntimeCall):
    addr:int 


@dataclass
class NandGroupArrange(RuntimeCall):
    pass 


@dataclass
class NandGroupMmap(RuntimeCall):
    pass 



@dataclass
class NandGroupMunmap(RuntimeCall):
    pass



@dataclass 
class NandGroupWrite(RuntimeCall):
    pass 


# prefetch -- read only 
@dataclass 
class SramPrefetch(RuntimeCall):
    prefetch_addr:int  # logic addr
    num_pages:int 

    pre_alloc_logic_addr:int     

@dataclass
class SramPrefetchRelease(RuntimeCall):
    addr:int  


# ------ SRAM Operation --------

@dataclass
class  SramMalloc(RuntimeCall):
    num_pages:int  

    pre_alloc_logic_addr: int 

@dataclass
class SramFree(RuntimeCall):
    addr:int



# ------- DRAM Operation ------ 
@dataclass
class DramMalloc(RuntimeCall):
    pass 


@dataclass
class DramFree(RuntimeCall):
    pass 


# ------- xPU Operation ---------

@dataclass
class MatMul(MacroOp):
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