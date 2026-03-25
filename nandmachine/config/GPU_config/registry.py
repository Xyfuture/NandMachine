from __future__ import annotations

from nandmachine.config.GPU_config.A100 import A100_80GB_FP16
from nandmachine.config.GPU_config.H100 import H100_PCIE_FP16, H100_SXM_FP16
from nandmachine.config.GPU_config.H200 import H200_NVL_FP16, H200_SXM_FP16
from nandmachine.config.GPU_config.schema import Device

device_dict: dict[str, Device] = {
    "A100_80GB": A100_80GB_FP16,
    "H100_SXM": H100_SXM_FP16,
    "H100_PCIE": H100_PCIE_FP16,
    "H200_SXM": H200_SXM_FP16,
    "H200_NVL": H200_NVL_FP16,
}


def get_device_or_raise(device_name: str) -> Device:
    if device_name not in device_dict:
        raise ValueError(f"device_name {device_name} not found.")
    return device_dict[device_name]


__all__ = [
    "device_dict",
    "get_device_or_raise",
]
