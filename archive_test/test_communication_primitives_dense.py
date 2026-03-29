from __future__ import annotations

import math

import pytest

from nandmachine.config.hardware_config import A100_80GB_FP16
from nandmachine.config.interconnect_config import (
    InterConnectModule,
    TopologyType,
    get_interconnect_for_device_or_raise,
)
from nandmachine.simulator.software.communication_primitives_of_dense import AllReduceSimulation


def test_constructor_rejects_unsupported_weight_bits() -> None:
    with pytest.raises(ValueError, match="Unsupported weight_bits"):
        AllReduceSimulation(num_gpus=4, data_size=128, weight_bits=4)


def test_constructor_rejects_invalid_data_size_alignment() -> None:
    with pytest.raises(ValueError, match="divisible by word_size"):
        AllReduceSimulation(num_gpus=4, data_size=3, weight_bits=16)


def test_simulate_returns_zero_for_single_gpu_or_zero_data() -> None:
    interconnect_single = get_interconnect_for_device_or_raise(
        device_name="A100_80GB",
        device_count=1,
    )
    sim_single = AllReduceSimulation(num_gpus=1, data_size=64, weight_bits=16)
    assert sim_single.simulate(interconnect_single) == 0.0

    interconnect_multi = get_interconnect_for_device_or_raise(
        device_name="A100_80GB",
        device_count=4,
    )
    sim_zero = AllReduceSimulation(num_gpus=4, data_size=0, weight_bits=16)
    assert sim_zero.simulate(interconnect_multi) == 0.0


def test_simulate_raises_for_non_fc_topology() -> None:
    fc_interconnect = get_interconnect_for_device_or_raise(
        device_name="A100_80GB",
        device_count=4,
    )
    ring_interconnect = InterConnectModule(
        device_count=fc_interconnect.device_count,
        topology=TopologyType.RING,
        link_module=fc_interconnect.link_module,
        link_count_per_device=fc_interconnect.link_count_per_device,
        internal_link_bandwidth_per_direction=fc_interconnect.internal_link_bandwidth_per_direction,
    )

    sim = AllReduceSimulation(num_gpus=4, data_size=128, weight_bits=16)
    with pytest.raises(NotImplementedError, match="FC topology only"):
        sim.simulate(ring_interconnect)


def test_simulate_raises_for_device_count_mismatch() -> None:
    interconnect = get_interconnect_for_device_or_raise(
        device_name="A100_80GB",
        device_count=8,
    )
    sim = AllReduceSimulation(num_gpus=4, data_size=128, weight_bits=16)

    with pytest.raises(ValueError, match="must equal num_gpus"):
        sim.simulate(interconnect)


def test_compile_and_simulate_raises_for_invalid_compile_mode() -> None:
    interconnect = get_interconnect_for_device_or_raise(
        device_name="A100_80GB",
        device_count=4,
    )
    sim = AllReduceSimulation(num_gpus=4, data_size=128, weight_bits=16)

    with pytest.raises(ValueError, match="compile_mode .* not supported"):
        sim.compile_and_simulate(
            pcb_module=A100_80GB_FP16,
            interconnect_module=interconnect,
            compile_mode="invalid",
        )


def test_compile_and_simulate_matches_latency_ceiling() -> None:
    interconnect = get_interconnect_for_device_or_raise(
        device_name="A100_80GB",
        device_count=4,
    )
    sim = AllReduceSimulation(num_gpus=4, data_size=128, weight_bits=16)

    latency_sec = sim.simulate(interconnect)
    cycles = sim.compile_and_simulate(
        pcb_module=A100_80GB_FP16,
        interconnect_module=interconnect,
        compile_mode="heuristic-GPU",
    )

    assert cycles == math.ceil(latency_sec * A100_80GB_FP16.compute_module.clock_freq)


def test_hierarchical_gpus_per_node_validation() -> None:
    interconnect = get_interconnect_for_device_or_raise(
        device_name="A100_80GB",
        device_count=16,
    )

    sim_bad_size = AllReduceSimulation(
        num_gpus=16,
        data_size=128,
        weight_bits=16,
        allreduce_params={"gpus_per_node": 6},
    )
    with pytest.raises(ValueError, match="must be divisible by gpus_per_node"):
        sim_bad_size.simulate(interconnect)

    sim_bad_value = AllReduceSimulation(
        num_gpus=16,
        data_size=128,
        weight_bits=16,
        allreduce_params={"gpus_per_node": 1},
    )
    with pytest.raises(ValueError, match="must be > 1"):
        sim_bad_value.simulate(interconnect)
