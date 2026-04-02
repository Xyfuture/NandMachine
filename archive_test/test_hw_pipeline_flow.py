import json
import math
from pathlib import Path

from Desim import SimSession

from nandmachine.commands.macro import (
    All2AllOp,
    FlashAttnOp,
    MatMulOp,
    SramPrefetch,
    SramPrefetchRelease,
    VectorOp,
)
from nandmachine.config.config import NandConfig
from nandmachine.config.hardware_config import get_device_or_raise
from nandmachine.simulator.entry_point import MacroSimResult, run_macro_ops
from nandmachine.simulator.hardware.xpu import xPU


def make_config() -> NandConfig:
    return NandConfig(
        num_channels=1,
        num_plane=2,
        num_block=8,
        num_pages=32,
        tRead=4.0,
        tWrite=8.0,
        tErase=16.0,
        page_size=16,
        sram_threshold=64,
    )


def test_xpu_instances_keep_their_own_compute_settings():
    config = make_config()

    SimSession.reset()
    SimSession.init()

    fast_xpu = xPU(config, compile_mode="heuristic-GPU")
    throughput_xpu = xPU(config, compile_mode="heuristic-our-throughput")

    assert fast_xpu.compute_engine.compile_mode == "heuristic-GPU"
    assert throughput_xpu.compute_engine.compile_mode == "heuristic-our-throughput"
    assert fast_xpu.compute_engine.device_name == "A100_80GB"
    assert throughput_xpu.compute_engine.device_name == "A100_80GB"

    SimSession.reset()


def test_xpu_routes_transfer_ops_to_transfer_engine():
    config = make_config()

    prefetch = SramPrefetch(num_prefetch_pages=2)
    transfer = All2AllOp(num_gpus=4, data_size=128, weight_bits=16).with_inputs(prefetch)
    matmul = MatMulOp(dim=(2, 16, 8), weight_bits=16).with_inputs(transfer)

    SimSession.reset()
    SimSession.init()

    sim_xpu = xPU(config)
    sim_xpu.load_command([prefetch, transfer, matmul])

    assert [slot.payload for slot in sim_xpu.prefetch_engine.prefetch_command_queue] == [
        prefetch
    ]
    assert [slot.payload for slot in sim_xpu.transfer_engine.transfer_command_queue] == [
        transfer
    ]
    assert [slot.payload for slot in sim_xpu.compute_engine.command_queue] == [matmul]
    assert sim_xpu.transfer_engine.transfer_command_queue[0].input_slots == [
        sim_xpu.prefetch_engine.prefetch_command_queue[0]
    ]
    assert sim_xpu.compute_engine.command_queue[0].input_slots == [
        sim_xpu.transfer_engine.transfer_command_queue[0]
    ]

    SimSession.reset()


def test_hw_pipeline_flow_runs_with_current_macro_ops():
    config = make_config()

    vector_norm = VectorOp(vector_op_type="rms_norm", vector_shape=[2, 16], weight_bits=16)
    prefetch_linear = SramPrefetch(num_prefetch_pages=4).with_inputs(vector_norm)
    matmul = MatMulOp(dim=(2, 16, 8), weight_bits=16).with_inputs(prefetch_linear)
    vector_act = VectorOp(vector_op_type="silu_mul", vector_shape=[2, 8], weight_bits=16).with_inputs(
        matmul
    )
    prefetch_attn = SramPrefetch(num_prefetch_pages=2).with_inputs(vector_act)
    flash_attn = FlashAttnOp(
        qk_bmm_shape=(4, 2, 4, 2),
        sv_bmm_shape=(4, 2, 2, 4),
        softmax_shape=(2, 2),
        weight_bits=16,
    ).with_inputs(prefetch_attn)
    release = SramPrefetchRelease().with_inputs(flash_attn)

    result = run_macro_ops(
        config,
        [
            vector_norm,
            prefetch_linear,
            matmul,
            vector_act,
            prefetch_attn,
            flash_attn,
            release,
        ],
    )

    assert result.cycle > 0
    assert result.time_ns > 0


def test_hw_pipeline_flow_runs_without_prefetch_or_release():
    config = make_config()

    vector_norm = VectorOp(vector_op_type="rms_norm", vector_shape=[2, 16], weight_bits=16)
    matmul = MatMulOp(dim=(2, 16, 8), weight_bits=16).with_inputs(vector_norm)
    vector_act = VectorOp(vector_op_type="silu_mul", vector_shape=[2, 8], weight_bits=16).with_inputs(
        matmul
    )
    flash_attn = FlashAttnOp(
        qk_bmm_shape=(4, 2, 4, 2),
        sv_bmm_shape=(4, 2, 2, 4),
        softmax_shape=(8, 2),
        weight_bits=16,
    ).with_inputs(vector_act)

    result = run_macro_ops(
        config,
        [
            vector_norm,
            matmul,
            vector_act,
            flash_attn,
        ],
    )

    assert result.cycle > 0
    assert result.time_ns > 0


def test_h100_heuristic_gpu_runs_large_output_matmul_without_asserting():
    config = make_config()

    result = run_macro_ops(
        config,
        [MatMulOp(dim=(2, 4096, 9216), weight_bits=16)],
        device_name="H100_SXM",
        compile_mode="heuristic-GPU",
    )

    assert result.cycle > 0
    assert result.time_ns > 0


def test_hw_pipeline_flow_runs_with_moe_vector_ops_on_h100():
    config = make_config()

    router = VectorOp(
        vector_op_type="moe_topk_router",
        vector_shape=[2, 8, 2],
        weight_bits=16,
    )
    dispatch = All2AllOp(num_gpus=4, data_size=128, weight_bits=16).with_inputs(router)
    expert = MatMulOp(dim=(2, 16, 32), weight_bits=16).with_inputs(dispatch)
    combine = All2AllOp(num_gpus=4, data_size=128, weight_bits=16).with_inputs(expert)
    weighted_sum = VectorOp(
        vector_op_type="moe_weighted_sum",
        vector_shape=[2, 16],
        weight_bits=16,
    ).with_inputs(combine)

    result = run_macro_ops(
        config,
        [router, dispatch, expert, combine, weighted_sum],
        device_name="H100_SXM",
        compile_mode="heuristic-GPU",
    )

    assert result.cycle > 0
    assert result.time_ns > 0


def test_run_macro_ops_returns_cycle_and_time_ns_for_a100():
    config = make_config()

    result = run_macro_ops(
        config,
        [VectorOp(vector_op_type="rms_norm", vector_shape=[2, 16], weight_bits=16)],
        device_name="A100_80GB",
        compile_mode="heuristic-GPU",
    )

    device = get_device_or_raise("A100_80GB")

    assert isinstance(result, MacroSimResult)
    assert result.time_ns == int(SimSession.sim_time.cycle)
    assert result.cycle == math.ceil(
        result.time_ns * device.compute_module.clock_freq / 1e9
    )


def test_run_macro_ops_returns_cycle_and_time_ns_for_h100():
    config = make_config()

    result = run_macro_ops(
        config,
        [VectorOp(vector_op_type="rms_norm", vector_shape=[2, 16], weight_bits=16)],
        device_name="H100_SXM",
        compile_mode="heuristic-GPU",
    )

    device = get_device_or_raise("H100_SXM")

    assert isinstance(result, MacroSimResult)
    assert result.time_ns == int(SimSession.sim_time.cycle)
    assert result.cycle == math.ceil(
        result.time_ns * device.compute_module.clock_freq / 1e9
    )


def test_notebook_entrypoints_use_macro_sim_result_and_drop_weight_bits():
    repo_root = Path(__file__).resolve().parents[1]
    notebook_paths = [
        repo_root / "frontend_pipeline.ipynb",
        repo_root / "llama_pipeline.ipynb",
        repo_root / "qwen3_moe_pipeline.ipynb",
    ]

    for notebook_path in notebook_paths:
        notebook = json.loads(notebook_path.read_text())
        simulator_config_cells = []
        run_cells = []
        for cell in notebook["cells"]:
            source = "".join(cell.get("source", []))
            if "simulator_config = {" in source:
                simulator_config_cells.append(source)
            if "run_macro_ops(" in source:
                run_cells.append(source)

        assert simulator_config_cells
        assert run_cells
        assert all("\"weight_bits\"" not in cell for cell in simulator_config_cells)
        assert all("sim_result = run_macro_ops(" in cell for cell in run_cells)
        assert all("assert sim_result.cycle > 0" in cell for cell in run_cells)
        assert all("sim_result.time_ns" in cell for cell in run_cells)
