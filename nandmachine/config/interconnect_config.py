from enum import Enum, auto
from math import ceil
from typing import Dict

from nandmachine.config.GPU_config.A100 import (
    A100_FLIT_SIZE_BYTES,
    A100_HEADER_SIZE_BYTES,
    A100_LINK_LATENCY_SEC,
    A100_MAX_PAYLOAD_SIZE_BYTES,
    A100_NVLINK_BW_BIDIR_BPS,
    A100_NVLINK_BW_PER_DIRECTION_BPS,
    A100_NVLINK_LINK_COUNT_PER_DEVICE,
)
from nandmachine.config.GPU_config.H100 import (
    H100_FLIT_SIZE_BYTES,
    H100_HEADER_SIZE_BYTES,
    H100_LINK_LATENCY_SEC,
    H100_MAX_PAYLOAD_SIZE_BYTES,
    H100_NVLINK_BW_BIDIR_BPS,
    H100_NVLINK_BW_PER_DIRECTION_BPS,
    H100_PCIE_NVLINK_LINK_COUNT_PER_DEVICE,
    H100_SXM_NVLINK_LINK_COUNT_PER_DEVICE,
)
from nandmachine.config.GPU_config.H200 import (
    H200_FLIT_SIZE_BYTES,
    H200_HEADER_SIZE_BYTES,
    H200_LINK_LATENCY_SEC,
    H200_MAX_PAYLOAD_SIZE_BYTES,
    H200_NVLINK_BW_BIDIR_BPS,
    H200_NVLINK_BW_PER_DIRECTION_BPS,
    H200_NVL_NVLINK_LINK_COUNT_PER_DEVICE,
    H200_SXM_NVLINK_LINK_COUNT_PER_DEVICE,
)


class TopologyType(Enum):
    RING = auto()
    FC = auto()


class LinkModule:
    def __init__(
        self,
        bandwidth_per_direction: float,  # B/s。单向带宽
        bandwidth_both_direction: float,  # B/s。双向带宽
        latency: float,  # s。链路固定时延
        flit_size: int,  # B。链路最小传输粒度
        max_payload_size: int,  # B。单个packet能装载的最大有效数据
        header_size: int,  # B。packet的头部开销
    ) -> None:
        self.bandwidth_per_direction = bandwidth_per_direction
        self.bandwidth_both_direction = bandwidth_both_direction
        self.latency = latency
        self.flit_size = flit_size
        self.max_payload_size = max_payload_size
        self.header_size = ceil(header_size / flit_size) * flit_size


link_module_dict = {
    "NVLinkV3": LinkModule(
        A100_NVLINK_BW_PER_DIRECTION_BPS,
        A100_NVLINK_BW_BIDIR_BPS,
        A100_LINK_LATENCY_SEC,
        A100_FLIT_SIZE_BYTES,
        A100_MAX_PAYLOAD_SIZE_BYTES,
        A100_HEADER_SIZE_BYTES,
    ),
    "NVLinkV4": LinkModule(
        H100_NVLINK_BW_PER_DIRECTION_BPS,
        H100_NVLINK_BW_BIDIR_BPS,
        H100_LINK_LATENCY_SEC,
        H100_FLIT_SIZE_BYTES,
        H100_MAX_PAYLOAD_SIZE_BYTES,
        H100_HEADER_SIZE_BYTES,
    ),
}


class InterConnectModule:
    def __init__(
        self,
        device_count: int,
        topology,
        link_module: LinkModule,
        link_count_per_device: int,
        internal_link_bandwidth_per_direction: float = float("inf"),
    ) -> None:
        self.device_count = device_count
        self.topology = topology
        self.link_module = link_module
        self.link_count_per_device = link_count_per_device
        self.internal_link_bandwidth_per_direction = (
            internal_link_bandwidth_per_direction
        )
        pass


interconnect_module_dict = {
    "NVLinkV3_FC_4": InterConnectModule(
        4, TopologyType.FC, link_module_dict["NVLinkV3"], A100_NVLINK_LINK_COUNT_PER_DEVICE
    ),
    "H100_SXM_NVLinkV4_FC_4": InterConnectModule(
        4, TopologyType.FC, link_module_dict["NVLinkV4"], H100_SXM_NVLINK_LINK_COUNT_PER_DEVICE
    ),
    "H100_PCIE_NVLinkV4_FC_4": InterConnectModule(
        4, TopologyType.FC, link_module_dict["NVLinkV4"], H100_PCIE_NVLINK_LINK_COUNT_PER_DEVICE
    ),
    "H200_SXM_NVLinkV4_FC_4": InterConnectModule(
        4, TopologyType.FC, link_module_dict["NVLinkV4"], H200_SXM_NVLINK_LINK_COUNT_PER_DEVICE
    ),
    "H200_NVL_NVLinkV4_FC_4": InterConnectModule(
        4, TopologyType.FC, link_module_dict["NVLinkV4"], H200_NVL_NVLINK_LINK_COUNT_PER_DEVICE
    ),
}

device_interconnect_profile_dict: dict[str, tuple[str, int]] = {
    "A100_80GB": ("NVLinkV3", A100_NVLINK_LINK_COUNT_PER_DEVICE),
    "H100_SXM": ("NVLinkV4", H100_SXM_NVLINK_LINK_COUNT_PER_DEVICE),
    "H100_PCIE": ("NVLinkV4", H100_PCIE_NVLINK_LINK_COUNT_PER_DEVICE),
    "H200_SXM": ("NVLinkV4", H200_SXM_NVLINK_LINK_COUNT_PER_DEVICE),
    "H200_NVL": ("NVLinkV4", H200_NVL_NVLINK_LINK_COUNT_PER_DEVICE),
}


def get_interconnect_for_device_or_raise(
    device_name: str,
    device_count: int,
    topology: TopologyType = TopologyType.FC,
) -> InterConnectModule:
    if device_name not in device_interconnect_profile_dict:
        raise ValueError(f"Unsupported device_name for interconnect: {device_name}")
    if device_count <= 0:
        raise ValueError(f"device_count must be positive, got {device_count}")
    link_key, link_count_per_device = device_interconnect_profile_dict[device_name]
    if link_key not in link_module_dict:
        raise ValueError(f"Unsupported link_key={link_key} for device_name={device_name}")
    return InterConnectModule(
        device_count=device_count,
        topology=topology,
        link_module=link_module_dict[link_key],
        link_count_per_device=link_count_per_device,
    )


MoEParamDict = Dict[str, float]


# MoE routing behavior defaults
MOE_ROUTING_DEFAULTS: MoEParamDict = {
    "top_k": 2.0,  # 每个 token 路由到的 expert 数量（top-1/top-2）
    "remote_ratio": 0.75,  # 被路由到远端设备的 token 比例（0~1）
    "load_imbalance_factor": 1.15,  # 负载不均衡放大因子（>1 代表最慢链路/最忙 expert 拖慢整体）
}

# MoE payload and metadata defaults
MOE_PAYLOAD_DEFAULTS: MoEParamDict = {
    "dispatch_metadata_bytes_per_token": 8.0,  # dispatch 阶段每个 token 的额外元数据字节（如 expert id / offset）
    "combine_metadata_bytes_per_token": 0.0,  # combine 阶段每个 token 的额外元数据字节（通常可比 dispatch 更小）
    "internal_copy_multiplier": 1.0,  # 设备内部搬运量系数（用于近似片上/板内 copy 开销）
}

# MoE hierarchical inter-node communication defaults
MOE_INTER_NODE_DEFAULTS: MoEParamDict = {
    "gpus_per_node": 8.0,  # >8 卡时的分层建模参数：每个节点内 GPU 数
    "inter_node_bandwidth_per_direction": 50e9,  # 节点间单向带宽（B/s，按每 GPU 有效带宽近似）
    "inter_node_latency": 5e-6,  # 节点间固定时延（s）
    "inter_node_header_size": 64.0,  # 节点间每 packet 头部字节
    "inter_node_max_payload_size": 4096.0,  # 节点间每 packet 最大 payload（B）
    "inter_node_link_count_per_device": 1.0,  # 节点间每 GPU 可并行链路数（占位）
    "inter_node_oversubscription_factor": 1.2,  # 节点间拥塞/过订阅放大因子
}

# Final default set used by MoE communication simulator
MOE_DEFAULT_PARAMS: MoEParamDict = {
    **MOE_ROUTING_DEFAULTS,
    **MOE_PAYLOAD_DEFAULTS,
    **MOE_INTER_NODE_DEFAULTS,
}

# Backward compatibility for existing imports/usages.
MOE_PLACEHOLDER_PARAMS = MOE_DEFAULT_PARAMS
