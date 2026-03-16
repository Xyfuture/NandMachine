from nandmachine.config.interconnect_config import (
    InterConnectModule,
    MOE_DEFAULT_PARAMS,
    TopologyType,
)
from typing import Dict
from math import ceil


def _word_size_from_weight_bits(weight_bits: int) -> int:
    if weight_bits <= 0 or weight_bits % 8 != 0:
        raise ValueError(
            f"Unsupported weight_bits={weight_bits}, expected a positive multiple of 8"
        )
    return weight_bits // 8


class CommunicationPrimitive:
    def __init__(self, weight_bits: int = 16) -> None:
        self.weight_bits = weight_bits
        self.word_size = _word_size_from_weight_bits(weight_bits)
        # simulation results
        self.latency = None


class AllToAll_Simulation(CommunicationPrimitive):
    def __init__(
        self,
        tokens_per_device: float,
        hidden_size: float,
        weight_bits: int = 16,
        moe_params: Dict[str, float] = MOE_DEFAULT_PARAMS,
    ) -> None:
        super().__init__(weight_bits)
        self.tokens_per_device = float(tokens_per_device)
        self.hidden_size = float(hidden_size)
        if self.tokens_per_device <= 0:
            raise ValueError("tokens_per_device must be > 0.")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be > 0.")
        self.moe_params: Dict[str, float] = dict(MOE_DEFAULT_PARAMS)
        self.moe_params.update(moe_params)

    def _single_layer_alltoall_phase_latency(
        self,
        participant_count: int,
        link_bandwidth_per_direction: float,
        link_latency: float,
        header_size: float,
        max_payload_size: float,
        link_count_per_device: float,
        bytes_per_device: float,
        imbalance_factor: float,
    ) -> float:
        if participant_count <= 1 or bytes_per_device <= 0:
            return 0.0

        bytes_per_peer = bytes_per_device / (participant_count - 1)
        effective_bytes_per_peer = (
            header_size
            + ceil(bytes_per_peer / max_payload_size) * header_size
            + bytes_per_peer
        )

        edge_bandwidth_per_direction = (
            link_bandwidth_per_direction
            * link_count_per_device
            / (participant_count - 1)
        )
        return imbalance_factor * (
            link_latency + effective_bytes_per_peer / edge_bandwidth_per_direction
        )

    def simulate(
        self,
        interconnect_module: InterConnectModule,
    ) -> float:
        device_count = interconnect_module.device_count
        if device_count <= 1:
            self.latency = 0.0
            return self.latency

        if interconnect_module.topology != TopologyType.FC:
            raise NotImplementedError("MoE mode currently supports FC topology only.")

        # 分支1: <=8 卡使用单层模型；分支2: >8 卡使用分层模型

        link_bandwidth_per_direction = (
            interconnect_module.link_module.bandwidth_per_direction
        )  # 单向链路带宽（B/s）
        link_latency = interconnect_module.link_module.latency  # 单次消息固定时延（s）
        header_size = interconnect_module.link_module.header_size  # packet 头部字节（已按 flit 对齐）
        max_payload_size = interconnect_module.link_module.max_payload_size  # 单个 packet 的最大 payload（B）
        link_count_per_device = (
            interconnect_module.link_count_per_device
        )  # 每张卡可用互连链路条数

        tokens_per_device = self.tokens_per_device
        hidden_size = self.hidden_size
        top_k = self.moe_params["top_k"]  # 每个 token 路由到的 expert 数
        remote_ratio = self.moe_params["remote_ratio"]  # 需要跨卡发送的 token 比例
        if remote_ratio < 0 or remote_ratio > 1:
            raise ValueError("remote_ratio must be in [0, 1].")
        # 修复：默认占位值 0.75 只对 4 卡接近合理；在其他卡数会偏离均匀路由比例
        # 仅当当前值仍是占位默认值时，自动改为 (N-1)/N
        if (
            device_count > 1  # 多卡场景启用
            and abs(remote_ratio - MOE_DEFAULT_PARAMS["remote_ratio"]) < 1e-12  # 仅当当前值仍是默认值
        ):
            remote_ratio = (device_count - 1) / device_count  # 均匀路由近似：本地占比约 1/N，跨卡占比约 1-1/N
        load_imbalance_factor = self.moe_params["load_imbalance_factor"]  # 负载不均衡放大系数
        dispatch_meta_bytes = self.moe_params["dispatch_metadata_bytes_per_token"]  # dispatch 每 token 元数据字节
        combine_meta_bytes = self.moe_params["combine_metadata_bytes_per_token"]  # combine 每 token 元数据字节

        # MoE 两次 all-to-all: dispatch + combine
        payload_bytes_per_device = (
            tokens_per_device
            * top_k
            * hidden_size
            * self.word_size
            * remote_ratio
        )
        dispatch_total_bytes_per_device = (
            payload_bytes_per_device
            + tokens_per_device * top_k * dispatch_meta_bytes * remote_ratio
        )
        combine_total_bytes_per_device = (
            payload_bytes_per_device
            + tokens_per_device * top_k * combine_meta_bytes * remote_ratio
        )

        if device_count <= 8:
            dispatch_latency = self._single_layer_alltoall_phase_latency(
                participant_count=device_count,
                link_bandwidth_per_direction=link_bandwidth_per_direction,
                link_latency=link_latency,
                header_size=header_size,
                max_payload_size=max_payload_size,
                link_count_per_device=link_count_per_device,
                bytes_per_device=dispatch_total_bytes_per_device,
                imbalance_factor=load_imbalance_factor,
            )
            combine_latency = self._single_layer_alltoall_phase_latency(
                participant_count=device_count,
                link_bandwidth_per_direction=link_bandwidth_per_direction,
                link_latency=link_latency,
                header_size=header_size,
                max_payload_size=max_payload_size,
                link_count_per_device=link_count_per_device,
                bytes_per_device=combine_total_bytes_per_device,
                imbalance_factor=load_imbalance_factor,
            )
        else:
            # >8 卡：分层模型（节点内 + 节点间），仅作用于 MoE all-to-all
            gpus_per_node = int(self.moe_params["gpus_per_node"])
            if gpus_per_node <= 1:
                raise ValueError("gpus_per_node must be > 1 for hierarchical MoE modeling.")
            if device_count % gpus_per_node != 0:
                raise ValueError(
                    "For device_count > 8, device_count must be divisible by gpus_per_node."
                )
            node_count = device_count // gpus_per_node

            intra_share = (gpus_per_node - 1) / (device_count - 1)
            inter_share = (device_count - gpus_per_node) / (device_count - 1)

            dispatch_intra_bytes = dispatch_total_bytes_per_device * intra_share
            combine_intra_bytes = combine_total_bytes_per_device * intra_share
            dispatch_inter_bytes = dispatch_total_bytes_per_device * inter_share
            combine_inter_bytes = combine_total_bytes_per_device * inter_share

            # 节点内仍沿用当前 interconnect 的链路参数
            dispatch_intra_latency = self._single_layer_alltoall_phase_latency(
                participant_count=gpus_per_node,
                link_bandwidth_per_direction=link_bandwidth_per_direction,
                link_latency=link_latency,
                header_size=header_size,
                max_payload_size=max_payload_size,
                link_count_per_device=link_count_per_device,
                bytes_per_device=dispatch_intra_bytes,
                imbalance_factor=load_imbalance_factor,
            )
            combine_intra_latency = self._single_layer_alltoall_phase_latency(
                participant_count=gpus_per_node,
                link_bandwidth_per_direction=link_bandwidth_per_direction,
                link_latency=link_latency,
                header_size=header_size,
                max_payload_size=max_payload_size,
                link_count_per_device=link_count_per_device,
                bytes_per_device=combine_intra_bytes,
                imbalance_factor=load_imbalance_factor,
            )

            inter_dispatch_latency = self._single_layer_alltoall_phase_latency(
                participant_count=node_count,
                link_bandwidth_per_direction=self.moe_params[
                    "inter_node_bandwidth_per_direction"
                ],
                link_latency=self.moe_params["inter_node_latency"],
                header_size=self.moe_params["inter_node_header_size"],
                max_payload_size=self.moe_params["inter_node_max_payload_size"],
                link_count_per_device=self.moe_params[
                    "inter_node_link_count_per_device"
                ],
                bytes_per_device=dispatch_inter_bytes,
                imbalance_factor=load_imbalance_factor
                * self.moe_params["inter_node_oversubscription_factor"],
            )
            inter_combine_latency = self._single_layer_alltoall_phase_latency(
                participant_count=node_count,
                link_bandwidth_per_direction=self.moe_params[
                    "inter_node_bandwidth_per_direction"
                ],
                link_latency=self.moe_params["inter_node_latency"],
                header_size=self.moe_params["inter_node_header_size"],
                max_payload_size=self.moe_params["inter_node_max_payload_size"],
                link_count_per_device=self.moe_params[
                    "inter_node_link_count_per_device"
                ],
                bytes_per_device=combine_inter_bytes,
                imbalance_factor=load_imbalance_factor
                * self.moe_params["inter_node_oversubscription_factor"],
            )

            # dispatch/combine 各由“节点内阶段 + 节点间阶段”主导，使用 max 近似瓶颈
            dispatch_latency = max(dispatch_intra_latency, inter_dispatch_latency)  # 近似假设节点内与节点间阶段可高度重叠（乐观估计）
            combine_latency = max(combine_intra_latency, inter_combine_latency)  # 同上；若重叠不足，真实时延会更接近两者求和

        internal_copy_bytes = self.moe_params["internal_copy_multiplier"] * (
            dispatch_total_bytes_per_device + combine_total_bytes_per_device
        )
        internal_latency = (
            internal_copy_bytes
            / interconnect_module.internal_link_bandwidth_per_direction
        )

        self.latency = dispatch_latency + combine_latency + internal_latency
        return self.latency


class AllReduceMultiPCB(AllToAll_Simulation):
    # 为兼容旧调用名，MoE 文件里将该类映射到 MoE all-to-all 模型
    pass


class Broadcast:
    def __init__(self):
        self.src = None
        self.tensor = None

    def __call__(self, src: int, tensor: object):
        self.src = src
        self.tensor = tensor
