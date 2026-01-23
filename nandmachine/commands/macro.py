from dataclasses import dataclass, field
from typing import ClassVar


@dataclass
class MacroOp:
    id: int = field(init=False)

    _global_id_counter: ClassVar[int] = 0

    @classmethod
    def _next_id(cls) -> int:
        MacroOp._global_id_counter += 1
        return MacroOp._global_id_counter

    def __post_init__(self) -> None:
        self.id = self._next_id()


@dataclass
class RuntimeCall(MacroOp):
    pass 



# ------  Nand Operation ------
@dataclass
class NandMmap(RuntimeCall):
    file_id:int 


@dataclass
class NandMunmap(RuntimeCall):
    pass 


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
    prefetch_ptr:int 
    prefetch_page_counts:int 
    

class SramPrefetchRelease(RuntimeCall):
    pass 


# ------ SRAM Operation --------

@dataclass
class  SramMalloc(RuntimeCall):
    pass 

@dataclass
class SramFree(RuntimeCall):
    pass



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