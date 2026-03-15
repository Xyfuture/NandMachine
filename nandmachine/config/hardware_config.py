class VectorUnit:
    def __init__(self, total_vector_flops_per_cycle: int, flops_per_exp: int):
        self.total_vector_flops_per_cycle = total_vector_flops_per_cycle
        self.flops_per_exp = flops_per_exp


class SystolicArray:
    def __init__(
        self,
        array_height: int,
        array_width: int,
        mac_per_cycle: int,
        input_word_size: int,
        output_word_size: int,
    ) -> None:
        self.array_height = array_height
        self.array_width = array_width
        self.mac_per_cycle = mac_per_cycle
        self.input_word_size = input_word_size
        self.output_word_size = output_word_size


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


class IOModule:
    def __init__(self, bandwidth: float) -> None:
        self.bandwidth = bandwidth


class Device:
    def __init__(self, compute_module: ComputeModule, io_module: IOModule) -> None:
        self.compute_module = compute_module
        self.io_module = io_module


# A100 scalar parameters
A100_CORE_COUNT = 108
A100_CLOCK_FREQ_HZ = 1.41e9
A100_L2_SIZE_BYTES = 40 * 1024**2
A100_L2_BW_BYTES_PER_CYCLE = 5120
A100_IO_BW_BYTES_PER_SEC = 2039e9
A100_FLOPS_PER_EXP = 35

# A100 vector/tensor core parameters
A100_VECTOR_UNIT_FP16 = VectorUnit(512, A100_FLOPS_PER_EXP)
A100_SYSTOLIC_ARRAY_FP16 = SystolicArray(16, 16, 1, 2, 2)
A100_CORE_FP16 = Core(
    vector_unit=A100_VECTOR_UNIT_FP16,
    systolic_array=A100_SYSTOLIC_ARRAY_FP16,
    systolic_array_count=4,
    SRAM_size=192 * 1024,
)

# A100 module objects
A100_COMPUTE_MODULE_FP16 = ComputeModule(
    core=A100_CORE_FP16,
    core_count=A100_CORE_COUNT,
    clock_freq=A100_CLOCK_FREQ_HZ,
    l2_size=A100_L2_SIZE_BYTES,
    l2_bandwidth_per_cycle=A100_L2_BW_BYTES_PER_CYCLE,
)
A100_IO_MODULE = IOModule(bandwidth=A100_IO_BW_BYTES_PER_SEC)
A100_80GB_FP16 = Device(
    compute_module=A100_COMPUTE_MODULE_FP16,
    io_module=A100_IO_MODULE,
)

# Dict-style access for compatibility.
device_dict = {"A100_80GB_fp16": A100_80GB_FP16}

__all__ = [
    "VectorUnit",
    "SystolicArray",
    "Core",
    "ComputeModule",
    "IOModule",
    "Device",
    "A100_CORE_COUNT",
    "A100_CLOCK_FREQ_HZ",
    "A100_L2_SIZE_BYTES",
    "A100_L2_BW_BYTES_PER_CYCLE",
    "A100_IO_BW_BYTES_PER_SEC",
    "A100_FLOPS_PER_EXP",
    "A100_VECTOR_UNIT_FP16",
    "A100_SYSTOLIC_ARRAY_FP16",
    "A100_CORE_FP16",
    "A100_COMPUTE_MODULE_FP16",
    "A100_IO_MODULE",
    "A100_80GB_FP16",
    "device_dict",
]
