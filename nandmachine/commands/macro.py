from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, TypeAlias

if TYPE_CHECKING:
    from nandmachine.config.hardware_config import Device
    from nandmachine.simulator.software.matmul import MatMul


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
    dim: tuple[int, int, int] # M, K, N。M * K和K * N两个矩阵GEMM
    weight_bits: int

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.dim

    @property
    def output_shape(self) -> tuple[int, int]:
        m, _, n = self.dim
        return m, n


@dataclass
class FlashAttnOp(MacroOp):
    qk_bmm_shape: tuple[int, int, int, int] # B，M，K，N。qk阶段的B个M * K和K * N矩阵GEMM
    sv_bmm_shape: tuple[int, int, int, int] # B，M，N，K。sv阶段的B个M * N和N * K矩阵GEMM
    softmax_shape: tuple[int, int] # M，N。sotmax的输入矩阵M *。N
    weight_bits: int

    @property
    def shape(
        self,
    ) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], tuple[int, int]]:
        return self.qk_bmm_shape, self.sv_bmm_shape, self.softmax_shape

    @property
    def qk_bmm_input_shape(
        self,
    ) -> tuple[int, int, int, int]:
        return self.qk_bmm_shape

    @property
    def qk_bmm_output_shape(self) -> tuple[int, int, int]:
        b, m, _, n = self.qk_bmm_shape
        return b, m, n

    @property
    def sv_bmm_input_shape(
        self,
    ) -> tuple[int, int, int, int]:
        return self.sv_bmm_shape

    @property
    def sv_bmm_output_shape(self) -> tuple[int, int, int]:
        b, m, _, k = self.sv_bmm_shape
        return b, m, k

    @property
    def softmax_input_shape(self) -> tuple[int, int]:
        return self.softmax_shape

    @property
    def softmax_output_shape(self) -> tuple[int, int]:
        return self.softmax_shape


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
