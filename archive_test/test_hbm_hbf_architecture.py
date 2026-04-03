from __future__ import annotations

import math

import pytest

from nandmachine.config.GPU_config.registry import get_device_or_raise
from nandmachine.config.hbm_hbf_architecture import (
    build_device_for_hbm_hbf_architecture_or_raise,
    validate_hbm_hbf_architecture_or_raise,
)


def test_base_devices_keep_hbm_only_split_fields() -> None:
    a100 = get_device_or_raise("A100_80GB")
    h100 = get_device_or_raise("H100_SXM")
    h200 = get_device_or_raise("H200_SXM")

    assert a100.memory_architecture_mode == "hbm_only"
    assert a100.memory_capacity_bytes == a100.total_memory_capacity_bytes
    assert a100.total_memory_capacity_bytes == a100.hbm_memory_capacity_bytes
    assert a100.hbf_memory_capacity_bytes == 0
    assert a100.io_module.bandwidth == a100.io_module.total_bandwidth
    assert a100.io_module.hbm_bandwidth == a100.io_module.total_bandwidth
    assert a100.io_module.hbf_bandwidth == 0

    assert h100.memory_architecture_mode == "hbm_only"
    assert h100.hbm_stack_count == 5
    assert h100.hbf_stack_count == 0

    assert h200.memory_architecture_mode == "hbm_only"
    assert h200.hbm_stack_count == 6
    assert h200.hbf_stack_count == 0


@pytest.mark.parametrize(
    ("device_name", "raw_config"),
    [
        ("A100_80GB", {"mode": "csi"}),
        ("A100_80GB", {"mode": "cli", "hbm_stacks": 1, "hbf_stacks": 4}),
        ("H100_PCIE", {"mode": "csi"}),
        ("H100_PCIE", {"mode": "cli", "hbm_stacks": 2, "hbf_stacks": 3}),
        ("H200_NVL", {"mode": "csi"}),
        ("H200_NVL", {"mode": "cli", "hbm_stacks": 3, "hbf_stacks": 3}),
    ],
)
def test_only_sxm_h100_h200_support_hbf_architectures(
    device_name: str,
    raw_config: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="only supported on H100_SXM/H200_SXM"):
        validate_hbm_hbf_architecture_or_raise(device_name, raw_config)


def test_h100_sxm_hbm_only_keeps_legacy_total_fields() -> None:
    device = build_device_for_hbm_hbf_architecture_or_raise(
        "H100_SXM",
        {"mode": "hbm_only"},
    )

    assert device.memory_architecture_mode == "hbm_only"
    assert device.total_memory_capacity_bytes == 80 * 1024**3
    assert device.hbm_memory_capacity_bytes == 80 * 1024**3
    assert device.hbf_memory_capacity_bytes == 0
    assert device.memory_capacity_bytes == device.total_memory_capacity_bytes
    assert device.io_module.total_bandwidth == 3.35e12
    assert device.io_module.hbm_bandwidth == 3.35e12
    assert device.io_module.hbf_bandwidth == 0
    assert device.io_module.bandwidth == device.io_module.total_bandwidth


def test_h100_sxm_csi_derives_capacity_and_keeps_total_bandwidth() -> None:
    device = build_device_for_hbm_hbf_architecture_or_raise(
        "H100_SXM",
        {"mode": "csi"},
    )

    assert device.memory_architecture_mode == "csi"
    assert device.hbm_stack_count == 5
    assert device.hbf_stack_count == 5
    assert device.hbm_memory_capacity_bytes == 5 * 16 * 1024**3
    assert device.hbf_memory_capacity_bytes == 5 * 256 * 1024**3
    assert device.total_memory_capacity_bytes == (
        device.hbm_memory_capacity_bytes + device.hbf_memory_capacity_bytes
    )
    assert device.io_module.total_bandwidth == 3.35e12
    assert device.io_module.hbm_bandwidth == 3.35e12
    assert device.io_module.hbf_bandwidth == 0


def test_h100_sxm_cli_derives_split_capacity_and_bandwidth() -> None:
    device = build_device_for_hbm_hbf_architecture_or_raise(
        "H100_SXM",
        {"mode": "cli", "hbm_stacks": 2, "hbf_stacks": 3},
    )

    assert device.memory_architecture_mode == "cli"
    assert device.hbm_stack_count == 2
    assert device.hbf_stack_count == 3
    assert device.hbm_memory_capacity_bytes == 2 * 16 * 1024**3
    assert device.hbf_memory_capacity_bytes == 3 * 256 * 1024**3
    assert device.total_memory_capacity_bytes == (
        device.hbm_memory_capacity_bytes + device.hbf_memory_capacity_bytes
    )
    assert math.isclose(device.io_module.hbm_bandwidth, 2 * 670e9)
    assert math.isclose(device.io_module.hbf_bandwidth, 3 * 670e9)
    assert math.isclose(device.io_module.total_bandwidth, 3.35e12)
    assert math.isclose(
        device.io_module.total_bandwidth,
        device.io_module.hbm_bandwidth + device.io_module.hbf_bandwidth,
    )


def test_h200_sxm_hbm_only_keeps_current_capacity() -> None:
    device = build_device_for_hbm_hbf_architecture_or_raise(
        "H200_SXM",
        {"mode": "hbm_only"},
    )

    assert device.memory_architecture_mode == "hbm_only"
    assert device.total_memory_capacity_bytes == 141 * 1024**3
    assert device.hbm_memory_capacity_bytes == 141 * 1024**3
    assert device.hbf_memory_capacity_bytes == 0
    assert device.io_module.total_bandwidth == 4.8e12
    assert device.io_module.hbm_bandwidth == 4.8e12
    assert device.io_module.hbf_bandwidth == 0


def test_h200_sxm_csi_derives_capacity_and_keeps_total_bandwidth() -> None:
    device = build_device_for_hbm_hbf_architecture_or_raise(
        "H200_SXM",
        {"mode": "csi"},
    )

    assert device.memory_architecture_mode == "csi"
    assert device.hbm_stack_count == 6
    assert device.hbf_stack_count == 6
    assert device.hbm_memory_capacity_bytes == 6 * ((47 * 1024**3) // 2)
    assert device.hbf_memory_capacity_bytes == 6 * 256 * 1024**3
    assert device.total_memory_capacity_bytes == (
        device.hbm_memory_capacity_bytes + device.hbf_memory_capacity_bytes
    )
    assert device.io_module.total_bandwidth == 4.8e12
    assert device.io_module.hbm_bandwidth == 4.8e12
    assert device.io_module.hbf_bandwidth == 0


def test_h200_sxm_cli_derives_split_capacity_and_bandwidth() -> None:
    device = build_device_for_hbm_hbf_architecture_or_raise(
        "H200_SXM",
        {"mode": "cli", "hbm_stacks": 3, "hbf_stacks": 3},
    )

    assert device.memory_architecture_mode == "cli"
    assert device.hbm_stack_count == 3
    assert device.hbf_stack_count == 3
    assert device.hbm_memory_capacity_bytes == 3 * ((47 * 1024**3) // 2)
    assert device.hbf_memory_capacity_bytes == 3 * 256 * 1024**3
    assert device.total_memory_capacity_bytes == (
        device.hbm_memory_capacity_bytes + device.hbf_memory_capacity_bytes
    )
    assert math.isclose(device.io_module.hbm_bandwidth, 3 * 800e9)
    assert math.isclose(device.io_module.hbf_bandwidth, 3 * 800e9)
    assert math.isclose(device.io_module.total_bandwidth, 4.8e12)
    assert math.isclose(
        device.io_module.total_bandwidth,
        device.io_module.hbm_bandwidth + device.io_module.hbf_bandwidth,
    )
