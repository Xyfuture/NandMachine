from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, TypeAlias


LogicAddr: TypeAlias = int


@dataclass
class MacroOp:
    id: int = field(init=False)
    input_ops: list[MacroOp] = field(default_factory=list, init=False)

    _global_id_counter: ClassVar[int] = 0

    @classmethod
    def _next_id(cls) -> int:
        MacroOp._global_id_counter += 1
        return MacroOp._global_id_counter

    def __post_init__(self) -> None:
        self.id = self._next_id()

    def with_inputs(self, *ops: MacroOp) -> MacroOp:
        self.input_ops = list(ops)
        return self

    def add_inputs(self, *ops: MacroOp) -> MacroOp:
        self.input_ops.extend(ops)
        return self

@dataclass
class RuntimeCall(MacroOp):
    pass



# -------- Prefetch Operations ----------- 

@dataclass
class SramPrefetch(RuntimeCall):
    num_pages: int


@dataclass
class SramPrefetchRelease(RuntimeCall):
    pass

# -------- Compute Operations ---------
@dataclass
class MatMulOp(MacroOp):
    dim: tuple[int, int, int]
    weight_bits: int

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.dim


@dataclass
class FlashAttnOp(MacroOp):
    qk_bmm_shape: tuple[int, int, int, int]
    sv_bmm_shape: tuple[int, int, int, int]
    softmax_shape: tuple[int, int]


@dataclass
class VectorOp(MacroOp):
    vector_op_type: str
    vector_shape: list[int]


# ---------- Transfer Operationss ------------

@dataclass
class AllReduceOp(MacroOp):
    pass


@dataclass
class All2AllOp(MacroOp):
    pass


__all__ = [
    "LogicAddr",
    "MacroOp",
    "RuntimeCall",
    "SramPrefetch",
    "SramPrefetchRelease",
    "MatMulOp",
    "FlashAttnOp",
    "VectorOp",
    "AllReduceOp",
    "All2AllOp",
]
