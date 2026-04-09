from dataclasses import dataclass
from typing import Literal


@dataclass
class ParallelConfig:
    num_ranks: int


@dataclass
class DenseParallelConfig(ParallelConfig):
    tp_size: int
    dp_size: int

    def __post_init__(self) -> None:
        if self.num_ranks <= 0:
            raise ValueError(f"num_ranks must be > 0, got {self.num_ranks}")
        if self.tp_size <= 0:
            raise ValueError(f"tp_size must be > 0, got {self.tp_size}")
        if self.dp_size <= 0:
            raise ValueError(f"dp_size must be > 0, got {self.dp_size}")
        if self.dp_size != 1:
            raise ValueError(
                f"DenseParallelConfig requires dp_size == 1, got {self.dp_size}"
            )
        if self.num_ranks != self.dp_size * self.tp_size:
            raise ValueError(
                "DenseParallelConfig must satisfy num_ranks == dp_size * tp_size, "
                f"got num_ranks={self.num_ranks}, dp_size={self.dp_size}, "
                f"tp_size={self.tp_size}"
            )


@dataclass
class MoEParallelConfig(ParallelConfig):
    attn_dp_size: int
    attn_tp_size: int
    ffn_tp_size: int
    ffn_ep_size: int

    def __post_init__(self) -> None:
        if self.num_ranks <= 0:
            raise ValueError(f"num_ranks must be > 0, got {self.num_ranks}")
        if self.attn_dp_size <= 0:
            raise ValueError(f"attn_dp_size must be > 0, got {self.attn_dp_size}")
        if self.attn_tp_size <= 0:
            raise ValueError(f"attn_tp_size must be > 0, got {self.attn_tp_size}")
        if self.attn_tp_size != 1:
            raise ValueError(
                "MoEParallelConfig requires attn_tp_size == 1, "
                f"got {self.attn_tp_size}"
            )
        if self.ffn_tp_size <= 0:
            raise ValueError(f"ffn_tp_size must be > 0, got {self.ffn_tp_size}")
        if self.ffn_ep_size <= 0:
            raise ValueError(f"ffn_ep_size must be > 0, got {self.ffn_ep_size}")

        attn_world_size = self.attn_dp_size * self.attn_tp_size
        ffn_world_size = self.ffn_ep_size * self.ffn_tp_size
        if self.num_ranks != attn_world_size or self.num_ranks != ffn_world_size:
            raise ValueError(
                "MoEParallelConfig must satisfy "
                "num_ranks == attn_dp_size * attn_tp_size == ffn_ep_size * ffn_tp_size, "
                f"got num_ranks={self.num_ranks}, attn_dp_size={self.attn_dp_size}, "
                f"attn_tp_size={self.attn_tp_size}, ffn_ep_size={self.ffn_ep_size}, "
                f"ffn_tp_size={self.ffn_tp_size}"
            )


@dataclass
class InferenceConfig:
    batch_size: int  # global batch size
    input_sequence_length: int
    output_sequence_length: int
    weight_bits: int
    activation_bits: int
    kv_cache_bits: int
    kv_block_size_bytes: int
    memory_backend: Literal["nand", "hbm"]
    parallel_config: ParallelConfig

    def __post_init__(self) -> None:
        supported_backends = {"nand", "hbm"}
        if self.memory_backend not in supported_backends:
            raise ValueError(
                f"Unsupported memory_backend={self.memory_backend}, "
                f"expected one of {sorted(supported_backends)}"
            )


def resolve_batch_partition_size_or_raise(parallel_config: ParallelConfig | None) -> int:
    if parallel_config is None:
        return 1

    if isinstance(parallel_config, DenseParallelConfig):
        return parallel_config.dp_size

    if isinstance(parallel_config, MoEParallelConfig):
        return parallel_config.attn_dp_size


    raise TypeError(f"Unsupported parallel_config type: {type(parallel_config).__name__}")


def resolve_local_batch_size_or_raise(inference_config: InferenceConfig) -> int:
    global_batch_size = inference_config.batch_size
    if global_batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {global_batch_size}")

    partition_size = resolve_batch_partition_size_or_raise(
        inference_config.parallel_config
    )
    if partition_size <= 0:
        raise ValueError(f"partition_size must be > 0, got {partition_size}")
    if global_batch_size % partition_size != 0:
        raise ValueError(
            "global batch_size must be divisible by batch partition size, "
            f"got batch_size={global_batch_size}, partition_size={partition_size}"
        )

    return global_batch_size // partition_size
