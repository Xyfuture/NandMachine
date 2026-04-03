from __future__ import annotations

from math import isclose

from nandmachine.config.GPU_config.H100 import (
    H100_HBF_STACK_BYTES,
    H100_HBM_STACK_BYTES,
    H100_HBM_STACK_COUNT,
    H100_MEMORY_BANDWIDTH_PER_STACK_BYTES_PER_SEC,
)
from nandmachine.config.GPU_config.H200 import (
    H200_HBF_STACK_BYTES,
    H200_HBM_STACK_BYTES,
    H200_HBM_STACK_COUNT,
    H200_MEMORY_BANDWIDTH_PER_STACK_BYTES_PER_SEC,
)
from nandmachine.config.GPU_config.registry import get_device_or_raise
from nandmachine.config.GPU_config.schema import Device, IOModule


_DEVICE_NAME_TO_BASE_HBM_STACKS: dict[str, int] = {
    "A100_80GB": 5,
    "H100_SXM": H100_HBM_STACK_COUNT,
    "H100_PCIE": H100_HBM_STACK_COUNT,
    "H200_SXM": H200_HBM_STACK_COUNT,
    "H200_NVL": H200_HBM_STACK_COUNT,
}

_HBF_SUPPORTED_DEVICE_NAMES = {"H100_SXM", "H200_SXM"}

_CLI_ALLOWED_STACK_RATIOS: dict[str, set[tuple[int, int]]] = {
    "H100": {
        (1, 4),
        (2, 3),
        (3, 2),
        (4, 1),
    },
    "H200": {
        (1, 5),
        (2, 4),
        (3, 3),
        (4, 2),
        (5, 1),
    },
}

_DEVICE_FAMILY_TO_HBM_STACK_BYTES: dict[str, int] = {
    "H100": H100_HBM_STACK_BYTES,
    "H200": H200_HBM_STACK_BYTES,
}

_DEVICE_FAMILY_TO_HBF_STACK_BYTES: dict[str, int] = {
    "H100": H100_HBF_STACK_BYTES,
    "H200": H200_HBF_STACK_BYTES,
}

_DEVICE_FAMILY_TO_PER_STACK_BANDWIDTH: dict[str, float] = {
    "H100": H100_MEMORY_BANDWIDTH_PER_STACK_BYTES_PER_SEC,
    "H200": H200_MEMORY_BANDWIDTH_PER_STACK_BYTES_PER_SEC,
}


def _resolve_device_family_or_raise(device_name: str) -> str:
    get_device_or_raise(device_name)

    if device_name not in _DEVICE_NAME_TO_BASE_HBM_STACKS:
        raise ValueError(
            f"Unsupported device for memory architecture: {device_name}"
        )

    return device_name.split("_", 1)[0]


def _require_dict(raw_config: object) -> dict[str, object]:
    if not isinstance(raw_config, dict):
        raise ValueError("memory_architecture must be a dict with a 'mode' field")
    return raw_config


def _require_exact_keys(
    config: dict[str, object],
    *,
    expected_keys: set[str],
    mode: str,
) -> None:
    actual_keys = set(config.keys())
    missing_keys = expected_keys - actual_keys
    extra_keys = actual_keys - expected_keys

    if missing_keys or extra_keys:
        raise ValueError(
            "Invalid memory_architecture keys for "
            f"mode={mode}: missing={sorted(missing_keys)}, extra={sorted(extra_keys)}"
        )


def _require_mode(config: dict[str, object]) -> str:
    mode = config.get("mode")
    if not isinstance(mode, str):
        raise ValueError("memory_architecture.mode must be a string")
    return mode


def _require_positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"memory_architecture.{field_name} must be an int")
    if value <= 0:
        raise ValueError(f"memory_architecture.{field_name} must be > 0")
    return value


def _require_hbf_supported_device(device_name: str, mode: str) -> None:
    if device_name not in _HBF_SUPPORTED_DEVICE_NAMES:
        raise ValueError(
            f"mode={mode} is only supported on H100_SXM/H200_SXM devices, "
            f"got {device_name}"
        )


def _build_normalized_result(
    *,
    mode: str,
    device_name: str,
    device_family: str,
    base_hbm_stack_count: int,
    effective_hbm_stacks: int,
    effective_hbf_stacks: int,
) -> dict[str, str | int]:
    return {
        "mode": mode,
        "device_name": device_name,
        "device_family": device_family,
        "base_hbm_stack_count": base_hbm_stack_count,
        "effective_hbm_stacks": effective_hbm_stacks,
        "effective_hbf_stacks": effective_hbf_stacks,
    }


def validate_hbm_hbf_architecture_or_raise(
    device_name: str,
    raw_config: object,
) -> dict[str, str | int]:
    config = _require_dict(raw_config)
    mode = _require_mode(config)
    device_family = _resolve_device_family_or_raise(device_name)
    base_hbm_stack_count = _DEVICE_NAME_TO_BASE_HBM_STACKS[device_name]

    if mode == "hbm_only":
        _require_exact_keys(config, expected_keys={"mode"}, mode=mode)
        return _build_normalized_result(
            mode=mode,
            device_name=device_name,
            device_family=device_family,
            base_hbm_stack_count=base_hbm_stack_count,
            effective_hbm_stacks=base_hbm_stack_count,
            effective_hbf_stacks=0,
        )

    if mode == "csi":
        _require_exact_keys(config, expected_keys={"mode"}, mode=mode)
        _require_hbf_supported_device(device_name, mode)
        return _build_normalized_result(
            mode=mode,
            device_name=device_name,
            device_family=device_family,
            base_hbm_stack_count=base_hbm_stack_count,
            effective_hbm_stacks=base_hbm_stack_count,
            effective_hbf_stacks=base_hbm_stack_count,
        )

    if mode == "cli":
        _require_exact_keys(
            config,
            expected_keys={"mode", "hbm_stacks", "hbf_stacks"},
            mode=mode,
        )
        _require_hbf_supported_device(device_name, mode)

        hbm_stacks = _require_positive_int(config["hbm_stacks"], "hbm_stacks")
        hbf_stacks = _require_positive_int(config["hbf_stacks"], "hbf_stacks")

        if (hbm_stacks, hbf_stacks) not in _CLI_ALLOWED_STACK_RATIOS[device_family]:
            allowed_ratios = sorted(_CLI_ALLOWED_STACK_RATIOS[device_family])
            raise ValueError(
                f"Invalid CLI stack ratio for {device_family}: "
                f"hbm_stacks={hbm_stacks}, hbf_stacks={hbf_stacks}, "
                f"allowed={allowed_ratios}"
            )

        return _build_normalized_result(
            mode=mode,
            device_name=device_name,
            device_family=device_family,
            base_hbm_stack_count=base_hbm_stack_count,
            effective_hbm_stacks=hbm_stacks,
            effective_hbf_stacks=hbf_stacks,
        )

    raise ValueError(
        "memory_architecture.mode must be one of ['hbm_only', 'csi', 'cli']"
    )


def _build_device(
    *,
    base_device: Device,
    total_memory_capacity_bytes: int,
    hbm_memory_capacity_bytes: int,
    hbf_memory_capacity_bytes: int,
    total_bandwidth: float,
    hbm_bandwidth: float,
    hbf_bandwidth: float,
    memory_architecture_mode: str,
    hbm_stack_count: int,
    hbf_stack_count: int,
) -> Device:
    io_module = IOModule(
        bandwidth=total_bandwidth,
        hbm_bandwidth=hbm_bandwidth,
        hbf_bandwidth=hbf_bandwidth,
    )
    return Device(
        compute_module=base_device.compute_module,
        io_module=io_module,
        memory_capacity_bytes=total_memory_capacity_bytes,
        hbm_memory_capacity_bytes=hbm_memory_capacity_bytes,
        hbf_memory_capacity_bytes=hbf_memory_capacity_bytes,
        memory_architecture_mode=memory_architecture_mode,
        hbm_stack_count=hbm_stack_count,
        hbf_stack_count=hbf_stack_count,
    )


def build_device_for_hbm_hbf_architecture_or_raise(
    device_name: str,
    raw_config: object,
) -> Device:
    normalized_architecture = validate_hbm_hbf_architecture_or_raise(
        device_name,
        raw_config,
    )
    base_device = get_device_or_raise(device_name)
    mode = normalized_architecture["mode"]
    device_family = normalized_architecture["device_family"]
    hbm_stack_count = int(normalized_architecture["effective_hbm_stacks"])
    hbf_stack_count = int(normalized_architecture["effective_hbf_stacks"])

    if mode == "hbm_only":
        return _build_device(
            base_device=base_device,
            total_memory_capacity_bytes=base_device.total_memory_capacity_bytes,
            hbm_memory_capacity_bytes=base_device.hbm_memory_capacity_bytes,
            hbf_memory_capacity_bytes=base_device.hbf_memory_capacity_bytes,
            total_bandwidth=base_device.io_module.total_bandwidth,
            hbm_bandwidth=base_device.io_module.hbm_bandwidth,
            hbf_bandwidth=base_device.io_module.hbf_bandwidth,
            memory_architecture_mode="hbm_only",
            hbm_stack_count=hbm_stack_count,
            hbf_stack_count=hbf_stack_count,
        )

    hbm_stack_bytes = _DEVICE_FAMILY_TO_HBM_STACK_BYTES[device_family]
    hbf_stack_bytes = _DEVICE_FAMILY_TO_HBF_STACK_BYTES[device_family]
    hbm_memory_capacity_bytes = hbm_stack_count * hbm_stack_bytes
    hbf_memory_capacity_bytes = hbf_stack_count * hbf_stack_bytes
    total_memory_capacity_bytes = (
        hbm_memory_capacity_bytes + hbf_memory_capacity_bytes
    )

    if mode == "csi":
        return _build_device(
            base_device=base_device,
            total_memory_capacity_bytes=total_memory_capacity_bytes,
            hbm_memory_capacity_bytes=hbm_memory_capacity_bytes,
            hbf_memory_capacity_bytes=hbf_memory_capacity_bytes,
            total_bandwidth=base_device.io_module.total_bandwidth,
            hbm_bandwidth=base_device.io_module.total_bandwidth,
            hbf_bandwidth=0.0,
            memory_architecture_mode="csi",
            hbm_stack_count=hbm_stack_count,
            hbf_stack_count=hbf_stack_count,
        )

    if mode != "cli":
        raise AssertionError(f"Unhandled memory architecture mode: {mode}")

    per_stack_bandwidth = _DEVICE_FAMILY_TO_PER_STACK_BANDWIDTH[device_family]
    hbm_bandwidth = hbm_stack_count * per_stack_bandwidth
    hbf_bandwidth = hbf_stack_count * per_stack_bandwidth
    total_bandwidth = hbm_bandwidth + hbf_bandwidth
    if not isclose(total_bandwidth, base_device.io_module.total_bandwidth):
        raise ValueError(
            "Derived CLI total bandwidth does not match base device total "
            f"bandwidth for {device_name}: derived={total_bandwidth}, "
            f"base={base_device.io_module.total_bandwidth}"
        )

    return _build_device(
        base_device=base_device,
        total_memory_capacity_bytes=total_memory_capacity_bytes,
        hbm_memory_capacity_bytes=hbm_memory_capacity_bytes,
        hbf_memory_capacity_bytes=hbf_memory_capacity_bytes,
        total_bandwidth=total_bandwidth,
        hbm_bandwidth=hbm_bandwidth,
        hbf_bandwidth=hbf_bandwidth,
        memory_architecture_mode="cli",
        hbm_stack_count=hbm_stack_count,
        hbf_stack_count=hbf_stack_count,
    )


__all__ = [
    "validate_hbm_hbf_architecture_or_raise",
    "build_device_for_hbm_hbf_architecture_or_raise",
]
