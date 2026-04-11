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


def _expected_phase_latency(
    participant_count: int,
    link_bandwidth_per_direction: float,
    link_latency: float,
    header_size: float,
    max_payload_size: float,
    link_count_per_device: float,
    bytes_per_device: float,
    imbalance_factor: float = 1.0,
) -> float:
    bytes_per_peer = bytes_per_device / (participant_count - 1)
    effective_bytes_per_peer = (
        header_size
        + math.ceil(bytes_per_peer / max_payload_size) * header_size
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


def _expected_ring_allreduce_latency(
    num_gpus: int,
    link_bandwidth_per_direction: float,
    link_latency: float,
    header_size: float,
    max_payload_size: float,
    link_count_per_device: float,
    bytes_per_hop: float,
    imbalance_factor: float = 1.0,
) -> float:
    effective_bytes_per_hop = (
        header_size
        + math.ceil(bytes_per_hop / max_payload_size) * header_size
        + bytes_per_hop
    )
    edge_bandwidth_per_direction = (
        link_bandwidth_per_direction * link_count_per_device
    )
    startup_latency = 2 * link_latency
    data_latency = (
        2
        * (num_gpus - 1)
        * effective_bytes_per_hop
        / edge_bandwidth_per_direction
    )
    return imbalance_factor * (startup_latency + data_latency)


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


def test_simulate_supports_ring_topology() -> None:
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
    assert sim.simulate(ring_interconnect) > 0.0


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


def test_compile_and_simulate_supports_time_ns() -> None:
    interconnect = get_interconnect_for_device_or_raise(
        device_name="A100_80GB",
        device_count=4,
    )
    sim = AllReduceSimulation(num_gpus=4, data_size=128, weight_bits=16)

    cycles = sim.compile_and_simulate(
        pcb_module=A100_80GB_FP16,
        interconnect_module=interconnect,
        compile_mode="heuristic-GPU",
    )
    time_ns = sim.compile_and_simulate(
        pcb_module=A100_80GB_FP16,
        interconnect_module=interconnect,
        compile_mode="heuristic-GPU",
        return_unit="time_ns",
    )

    assert time_ns == math.ceil(
        cycles * 1e9 / A100_80GB_FP16.compute_module.clock_freq
    )
    assert sim.cycle_count == cycles
    assert sim.time_ns == time_ns


def test_simulate_uses_full_local_tensor_bytes_for_small_fc_allreduce() -> None:
    base_interconnect = get_interconnect_for_device_or_raise(
        device_name="A100_80GB",
        device_count=4,
    )
    interconnect = InterConnectModule(
        device_count=base_interconnect.device_count,
        topology=base_interconnect.topology,
        link_module=base_interconnect.link_module,
        link_count_per_device=base_interconnect.link_count_per_device,
        internal_link_bandwidth_per_direction=1e9,
    )
    sim = AllReduceSimulation(
        num_gpus=4,
        data_size=128,
        weight_bits=16,
        allreduce_params={"internal_copy_multiplier": 1.5},
    )

    phase_bytes_per_device = 128 * (4 - 1) / 4
    phase_latency = _expected_phase_latency(
        participant_count=4,
        link_bandwidth_per_direction=interconnect.link_module.bandwidth_per_direction,
        link_latency=interconnect.link_module.latency,
        header_size=interconnect.link_module.header_size,
        max_payload_size=interconnect.link_module.max_payload_size,
        link_count_per_device=interconnect.link_count_per_device,
        bytes_per_device=phase_bytes_per_device,
    )
    internal_latency = 1.5 * (2 * phase_bytes_per_device) / 1e9
    expected_latency = 2 * phase_latency + internal_latency

    assert sim.simulate(interconnect) == pytest.approx(expected_latency)


def test_simulate_uses_correct_phase_bytes_for_hierarchical_allreduce() -> None:
    base_interconnect = get_interconnect_for_device_or_raise(
        device_name="A100_80GB",
        device_count=16,
    )
    interconnect = InterConnectModule(
        device_count=base_interconnect.device_count,
        topology=base_interconnect.topology,
        link_module=base_interconnect.link_module,
        link_count_per_device=base_interconnect.link_count_per_device,
        internal_link_bandwidth_per_direction=2e9,
    )
    sim = AllReduceSimulation(
        num_gpus=16,
        data_size=128,
        weight_bits=16,
        allreduce_params={
            "gpus_per_node": 8,
            "internal_copy_multiplier": 0.25,
            "inter_node_bandwidth_per_direction": 40e9,
            "inter_node_latency": 5e-6,
            "inter_node_header_size": 64.0,
            "inter_node_max_payload_size": 4096.0,
            "inter_node_link_count_per_device": 1.0,
            "inter_node_oversubscription_factor": 1.5,
        },
    )

    phase_bytes_per_device = 128 * (16 - 1) / 16
    intra_share = (8 - 1) / (16 - 1)
    inter_share = (16 - 8) / (16 - 1)
    intra_bytes_per_device = phase_bytes_per_device * intra_share
    inter_bytes_per_device = phase_bytes_per_device * inter_share

    intra_phase_latency = _expected_phase_latency(
        participant_count=8,
        link_bandwidth_per_direction=interconnect.link_module.bandwidth_per_direction,
        link_latency=interconnect.link_module.latency,
        header_size=interconnect.link_module.header_size,
        max_payload_size=interconnect.link_module.max_payload_size,
        link_count_per_device=interconnect.link_count_per_device,
        bytes_per_device=intra_bytes_per_device,
    )
    inter_phase_latency = _expected_phase_latency(
        participant_count=2,
        link_bandwidth_per_direction=40e9,
        link_latency=5e-6,
        header_size=64.0,
        max_payload_size=4096.0,
        link_count_per_device=1.0,
        bytes_per_device=inter_bytes_per_device,
        imbalance_factor=1.5,
    )
    phase_latency = max(intra_phase_latency, inter_phase_latency)
    internal_latency = 0.25 * (2 * phase_bytes_per_device) / 2e9
    expected_latency = 2 * phase_latency + internal_latency

    assert sim.simulate(interconnect) == pytest.approx(expected_latency)


def test_ring_allreduce_h200_8gpu_latency_regression() -> None:
    interconnect = get_interconnect_for_device_or_raise(
        device_name="H200_SXM",
        device_count=8,
        topology=TopologyType.RING,
    )
    sim = AllReduceSimulation(num_gpus=8, data_size=2097152, weight_bits=16)

    assert sim.simulate(interconnect) * 1e6 == pytest.approx(10.665813333333332)


def test_ring_allreduce_matches_fc_in_current_idealized_model() -> None:
    fc_interconnect = get_interconnect_for_device_or_raise(
        device_name="H200_SXM",
        device_count=8,
        topology=TopologyType.FC,
    )
    ring_interconnect = get_interconnect_for_device_or_raise(
        device_name="H200_SXM",
        device_count=8,
        topology=TopologyType.RING,
    )

    fc_sim = AllReduceSimulation(num_gpus=8, data_size=2097152, weight_bits=16)
    ring_sim = AllReduceSimulation(num_gpus=8, data_size=2097152, weight_bits=16)

    assert ring_sim.simulate(ring_interconnect) == pytest.approx(
        fc_sim.simulate(fc_interconnect)
    )


def test_ring_allreduce_startup_and_data_latency_breakdown() -> None:
    ring_interconnect = get_interconnect_for_device_or_raise(
        device_name="H200_SXM",
        device_count=8,
        topology=TopologyType.RING,
    )
    fc_interconnect = get_interconnect_for_device_or_raise(
        device_name="H200_SXM",
        device_count=8,
        topology=TopologyType.FC,
    )
    ring_sim = AllReduceSimulation(num_gpus=8, data_size=2097152, weight_bits=16)
    fc_sim = AllReduceSimulation(num_gpus=8, data_size=2097152, weight_bits=16)

    bytes_per_hop = 2097152 / 8
    ring_startup_latency = 2 * ring_interconnect.link_module.latency
    ring_total_latency = _expected_ring_allreduce_latency(
        num_gpus=8,
        link_bandwidth_per_direction=ring_interconnect.link_module.bandwidth_per_direction,
        link_latency=ring_interconnect.link_module.latency,
        header_size=ring_interconnect.link_module.header_size,
        max_payload_size=ring_interconnect.link_module.max_payload_size,
        link_count_per_device=ring_interconnect.link_count_per_device,
        bytes_per_hop=bytes_per_hop,
    )
    ring_data_latency = ring_total_latency - ring_startup_latency

    fc_phase_bytes = 2097152 * (8 - 1) / 8
    fc_phase_latency = _expected_phase_latency(
        participant_count=8,
        link_bandwidth_per_direction=fc_interconnect.link_module.bandwidth_per_direction,
        link_latency=fc_interconnect.link_module.latency,
        header_size=fc_interconnect.link_module.header_size,
        max_payload_size=fc_interconnect.link_module.max_payload_size,
        link_count_per_device=fc_interconnect.link_count_per_device,
        bytes_per_device=fc_phase_bytes,
    )
    fc_total_latency = 2 * fc_phase_latency
    fc_startup_latency = 2 * fc_interconnect.link_module.latency
    fc_data_latency = fc_total_latency - fc_startup_latency

    assert ring_startup_latency * 1e6 == pytest.approx(2.0)
    assert ring_data_latency * 1e6 == pytest.approx(8.665813333333333)
    assert fc_startup_latency * 1e6 == pytest.approx(2.0)
    assert fc_data_latency * 1e6 == pytest.approx(8.665813333333333)
    assert ring_sim.simulate(ring_interconnect) == pytest.approx(ring_total_latency)
    assert fc_sim.simulate(fc_interconnect) == pytest.approx(fc_total_latency)


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
