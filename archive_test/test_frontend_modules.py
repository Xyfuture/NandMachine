import pytest


torch = pytest.importorskip("torch")

from nandmachine.frontend.core.graph.base import NxTracer
from nandmachine.frontend.modules.modules import (
    Attention,
    ColumnParallelLinear,
    LinearBase,
    RMSNorm,
    RotaryEmbedding,
    RowParallelLinear,
)


def test_rms_norm_forward_keeps_shape():
    module = RMSNorm(hidden_size=16)
    x = torch.randn(2, 3, 16)

    y = module(x)

    assert y.shape == x.shape


def test_column_parallel_linear_forward_updates_last_dim():
    module = ColumnParallelLinear(input_size=16, output_size=24, bias=True)
    x = torch.randn(2, 5, 16)

    y = module(x)

    assert y.shape == (2, 5, 24)


def test_row_parallel_linear_forward_updates_last_dim():
    module = RowParallelLinear(input_size=24, output_size=16, bias=False)
    x = torch.randn(2, 5, 24)

    y = module(x)

    assert y.shape == (2, 5, 16)


def test_linear_base_uses_plain_weight_shape():
    module = LinearBase(input_size=16, output_size=24, bias=False)

    assert module.weight.shape == (24, 16)


def test_parallel_linear_modules_keep_local_weight_shape_and_tp_info():
    column = ColumnParallelLinear(input_size=16, output_size=24, tp_size=2)
    row = RowParallelLinear(input_size=24, output_size=16, tp_size=2)

    assert column.weight.shape == (12, 16)
    assert column.tp_size == 2
    assert column.tp_rank == 0
    assert column.tp_dim == 0

    assert row.weight.shape == (16, 12)
    assert row.tp_size == 2
    assert row.tp_rank == 0
    assert row.tp_dim == 1


def test_rotary_embedding_forward_keeps_qk_shapes():
    module = RotaryEmbedding(
        head_size=8,
        rotary_dim=8,
        max_position_embeddings=64,
        base=10000.0,
    )
    positions = torch.arange(6)
    q = torch.randn(6, 4, 8)
    k = torch.randn(6, 2, 8)

    q_out, k_out = module(positions, q, k)

    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


def test_attention_forward_matches_query_shape():
    module = Attention(
        num_heads=4,
        head_dim=8,
        scale=8 ** -0.5,
        num_kv_heads=2,
    )
    q = torch.randn(6, 4, 8)
    k = torch.randn(6, 2, 8)
    v = torch.randn(6, 2, 8)

    y = module(q, k, v)

    assert y.shape == q.shape


def test_nx_tracer_keeps_hook_modules_as_call_module_nodes():
    class HookModulePipeline(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = RMSNorm(hidden_size=16)
            self.proj = ColumnParallelLinear(input_size=16, output_size=24)
            self.row = RowParallelLinear(input_size=24, output_size=16)
            self.rotary = RotaryEmbedding(
                head_size=8,
                rotary_dim=8,
                max_position_embeddings=64,
                base=10000.0,
            )
            self.attn = Attention(
                num_heads=4,
                head_dim=8,
                scale=8 ** -0.5,
                num_kv_heads=2,
            )

        def forward(
            self,
            x: torch.Tensor,
            positions: torch.Tensor,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            x = self.norm(x)
            x = self.proj(x)
            x = self.row(x)
            q, k = self.rotary(positions, q, k)
            y = self.attn(q, k, v)
            return x, y

    tracer = NxTracer()
    graph = tracer.trace(HookModulePipeline())

    call_module_targets = {
        node.target
        for node in graph.nodes
        if node.op == "call_module"
    }

    assert call_module_targets == {"norm", "proj", "row", "rotary", "attn"}
