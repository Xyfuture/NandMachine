from __future__ import annotations

import math

from nandmachine.config.hardware_config import device_dict
from nandmachine.config.interconnect_config import get_interconnect_for_device_or_raise


def _tensor_dense_tflops(device_name: str) -> float:
    device = device_dict[device_name]
    compute_module = device.compute_module
    systolic = compute_module.core.systolic_array
    flops_per_cycle = (
        systolic.array_height
        * systolic.array_width
        * systolic.mac_per_cycle_fp16
        * 2
        * compute_module.core.systolic_array_count
        * compute_module.core_count
    )
    return flops_per_cycle * compute_module.clock_freq / 1e12


def _all2all_latency_sec(device_name: str, num_ranks: int, bytes_per_peer: int) -> float:
    interconnect = get_interconnect_for_device_or_raise(
        device_name=device_name,
        device_count=num_ranks,
    )
    link = interconnect.link_module
    packet_count = math.ceil(bytes_per_peer / link.max_payload_size)
    effective_bytes_per_peer = (
        link.header_size + packet_count * link.header_size + bytes_per_peer
    )
    edge_bandwidth = (
        link.bandwidth_per_direction
        * interconnect.link_count_per_device
        / (num_ranks - 1)
    )
    return link.latency + effective_bytes_per_peer / edge_bandwidth


def test_gpu_device_registry_includes_h100_h200() -> None:
    assert "A100_80GB" in device_dict
    assert "H100_SXM" in device_dict
    assert "H100_PCIE" in device_dict
    assert "H200_SXM" in device_dict
    assert "H200_NVL" in device_dict


def test_interconnect_profile_is_gpu_specific() -> None:
    a100 = get_interconnect_for_device_or_raise("A100_80GB", 8)
    h100_sxm = get_interconnect_for_device_or_raise("H100_SXM", 8)
    h100_pcie = get_interconnect_for_device_or_raise("H100_PCIE", 8)
    h200_sxm = get_interconnect_for_device_or_raise("H200_SXM", 8)
    h200_nvl = get_interconnect_for_device_or_raise("H200_NVL", 8)

    assert a100.link_count_per_device == 12
    assert h100_sxm.link_count_per_device == 18
    assert h100_pcie.link_count_per_device == 12
    assert h200_sxm.link_count_per_device == 18
    assert h200_nvl.link_count_per_device == 18


def test_h100_h200_sxm_expected_speedup_over_a100() -> None:
    a100_tflops = _tensor_dense_tflops("A100_80GB")
    h100_tflops = _tensor_dense_tflops("H100_SXM")
    h200_tflops = _tensor_dense_tflops("H200_SXM")

    assert h100_tflops > a100_tflops
    assert h200_tflops > a100_tflops

    a100_all2all = _all2all_latency_sec("A100_80GB", 8, 8 * 1024 * 1024)
    h100_all2all = _all2all_latency_sec("H100_SXM", 8, 8 * 1024 * 1024)
    h200_all2all = _all2all_latency_sec("H200_SXM", 8, 8 * 1024 * 1024)

    assert h100_all2all < a100_all2all
    assert h200_all2all < a100_all2all
