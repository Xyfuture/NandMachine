from __future__ import annotations


class VectorUnit:
    def __init__(
        self,
        total_vector_flops_per_cycle_fp16: int,
        total_vector_flops_per_cycle_fp8: int,
        flops_per_exp: int,
    ) -> None:
        self.total_vector_flops_per_cycle_fp16 = total_vector_flops_per_cycle_fp16
        self.total_vector_flops_per_cycle_fp8 = total_vector_flops_per_cycle_fp8
        self.total_vector_flops_per_cycle = total_vector_flops_per_cycle_fp16
        self.total_vector_flops_per_cycle_by_weight_bits = {
            16: total_vector_flops_per_cycle_fp16,
            8: total_vector_flops_per_cycle_fp8,
        }
        self.flops_per_exp = flops_per_exp

    def get_total_vector_flops_per_cycle(self, weight_bits: int) -> int:
        if weight_bits not in self.total_vector_flops_per_cycle_by_weight_bits:
            raise ValueError(
                f"Unsupported weight_bits={weight_bits}, expected one of {sorted(self.total_vector_flops_per_cycle_by_weight_bits)}"
            )
        return self.total_vector_flops_per_cycle_by_weight_bits[weight_bits]


class SystolicArray:
    def __init__(
        self,
        array_height: int,
        array_width: int,
        mac_per_cycle_fp16: int,
        mac_per_cycle_fp8: int,
        input_word_size: int,
        output_word_size: int,
    ) -> None:
        self.array_height = array_height
        self.array_width = array_width
        self.mac_per_cycle_fp16 = mac_per_cycle_fp16
        self.mac_per_cycle_fp8 = mac_per_cycle_fp8
        self.mac_per_cycle = mac_per_cycle_fp16
        self.mac_per_cycle_by_weight_bits = {
            16: mac_per_cycle_fp16,
            8: mac_per_cycle_fp8,
        }
        self.input_word_size = input_word_size
        self.output_word_size = output_word_size

    def get_mac_per_cycle(self, weight_bits: int) -> int:
        if weight_bits not in self.mac_per_cycle_by_weight_bits:
            raise ValueError(
                f"Unsupported weight_bits={weight_bits}, expected one of {sorted(self.mac_per_cycle_by_weight_bits)}"
            )
        return self.mac_per_cycle_by_weight_bits[weight_bits]


class Core:
    def __init__(
        self,
        vector_unit: VectorUnit,
        systolic_array: SystolicArray,
        systolic_array_count: int,
        SRAM_size: int,
    ) -> None:
        self.vector_unit = vector_unit
        self.systolic_array = systolic_array
        self.systolic_array_count = systolic_array_count
        self.SRAM_size = SRAM_size


class ComputeModule:
    def __init__(
        self,
        core: Core,
        core_count: int,
        clock_freq: float,
        l2_size: int,
        l2_bandwidth_per_cycle: int,
    ) -> None:
        self.core = core
        self.core_count = core_count
        self.clock_freq = clock_freq
        self.l2_size = int(l2_size)
        self.l2_bandwidth_per_cycle = l2_bandwidth_per_cycle
        self.total_vector_flops_per_cycle = (
            core.vector_unit.total_vector_flops_per_cycle * core_count
        )
        self.total_vector_flops_per_cycle_by_weight_bits = {
            16: core.vector_unit.get_total_vector_flops_per_cycle(16) * core_count,
            8: core.vector_unit.get_total_vector_flops_per_cycle(8) * core_count,
        }

    def get_total_vector_flops_per_cycle(self, weight_bits: int) -> int:
        if weight_bits not in self.total_vector_flops_per_cycle_by_weight_bits:
            raise ValueError(
                f"Unsupported weight_bits={weight_bits}, expected one of {sorted(self.total_vector_flops_per_cycle_by_weight_bits)}"
            )
        return self.total_vector_flops_per_cycle_by_weight_bits[weight_bits]


class IOModule:
    def __init__(
        self,
        bandwidth: float,
        *,
        hbm_bandwidth: float | None = None,
        hbf_bandwidth: float = 0.0,
    ) -> None:
        if bandwidth < 0:
            raise ValueError(f"bandwidth must be >= 0, got {bandwidth}")
        if hbm_bandwidth is not None and hbm_bandwidth < 0:
            raise ValueError(
                f"hbm_bandwidth must be >= 0, got {hbm_bandwidth}"
            )
        if hbf_bandwidth < 0:
            raise ValueError(f"hbf_bandwidth must be >= 0, got {hbf_bandwidth}")

        self.total_bandwidth = bandwidth
        self.hbm_bandwidth = bandwidth if hbm_bandwidth is None else hbm_bandwidth
        self.hbf_bandwidth = hbf_bandwidth

        # Keep the legacy field as an alias of total bandwidth.
        self.bandwidth = self.total_bandwidth


class Device:
    def __init__(
        self,
        compute_module: ComputeModule,
        io_module: IOModule,
        memory_capacity_bytes: int,
        *,
        hbm_memory_capacity_bytes: int | None = None,
        hbf_memory_capacity_bytes: int = 0,
        memory_architecture_mode: str = "hbm_only",
        hbm_stack_count: int | None = None,
        hbf_stack_count: int = 0,
    ) -> None:
        if memory_capacity_bytes < 0:
            raise ValueError(
                f"memory_capacity_bytes must be >= 0, got {memory_capacity_bytes}"
            )
        if hbf_memory_capacity_bytes < 0:
            raise ValueError(
                "hbf_memory_capacity_bytes must be >= 0, "
                f"got {hbf_memory_capacity_bytes}"
            )
        if hbm_memory_capacity_bytes is None:
            hbm_memory_capacity_bytes = memory_capacity_bytes - hbf_memory_capacity_bytes
        if hbm_memory_capacity_bytes < 0:
            raise ValueError(
                "hbm_memory_capacity_bytes must be >= 0, "
                f"got {hbm_memory_capacity_bytes}"
            )
        if hbm_memory_capacity_bytes + hbf_memory_capacity_bytes != memory_capacity_bytes:
            raise ValueError(
                "HBM/HBF capacity split must sum to total capacity, got "
                f"total={memory_capacity_bytes}, "
                f"hbm={hbm_memory_capacity_bytes}, "
                f"hbf={hbf_memory_capacity_bytes}"
            )
        if hbm_stack_count is not None and hbm_stack_count < 0:
            raise ValueError(
                f"hbm_stack_count must be >= 0, got {hbm_stack_count}"
            )
        if hbf_stack_count < 0:
            raise ValueError(f"hbf_stack_count must be >= 0, got {hbf_stack_count}")

        self.compute_module = compute_module
        self.io_module = io_module
        self.total_memory_capacity_bytes = memory_capacity_bytes
        self.hbm_memory_capacity_bytes = hbm_memory_capacity_bytes
        self.hbf_memory_capacity_bytes = hbf_memory_capacity_bytes
        self.memory_architecture_mode = memory_architecture_mode
        self.hbm_stack_count = hbm_stack_count
        self.hbf_stack_count = hbf_stack_count

        # Keep the legacy field as an alias of total memory capacity.
        self.memory_capacity_bytes = self.total_memory_capacity_bytes
