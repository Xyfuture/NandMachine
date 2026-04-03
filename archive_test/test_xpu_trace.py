import json

import pytest
from Desim import SimSession, SimTime

from nandmachine.commands.macro import All2AllOp, SramPrefetch, VectorOp
from nandmachine.config.config import NandConfig
from nandmachine.config.hardware_config import get_device_or_raise
from nandmachine.simulator.hardware.vallina_xpu import VallinaXPU
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


def hbm_bandwidth_bytes_per_sec(device_name: str = "A100_80GB") -> float:
    return get_device_or_raise(device_name).io_module.bandwidth


def test_xpu_trace_records_all_three_engines_and_saves_file(tmp_path, monkeypatch):
    SimSession.reset()
    SimSession.init()

    vector_norm = VectorOp(vector_op_type="rms_norm", vector_shape=[2, 16], weight_bits=16)
    prefetch = SramPrefetch(num_prefetch_pages=2).with_inputs(vector_norm)
    transfer = All2AllOp(num_gpus=4, data_size=128, weight_bits=16).with_inputs(prefetch)

    sim_xpu = xPU(
        make_config(),
        hbm_bandwidth_bytes_per_sec=hbm_bandwidth_bytes_per_sec(),
        enable_trace=True,
    )
    sim_xpu.load_command([vector_norm, prefetch, transfer])

    monkeypatch.setattr(sim_xpu.compute_engine, "execute_macro_op", lambda macro_op: 3.0)
    monkeypatch.setattr(sim_xpu.transfer_engine, "execute_macro_op", lambda macro_op: 5.0)

    def fake_handle_request(request_slot):
        request_slot.is_finished = True
        request_slot.finish_event.notify(SimTime(4))

    monkeypatch.setattr(sim_xpu.nand_controller, "handle_request", fake_handle_request)

    SimSession.scheduler.run()

    tracer = sim_xpu.tracer
    assert tracer is not None

    thread_events = [
        event for event in tracer._events if event["ph"] == "M" and event["name"] == "thread_name"
    ]
    assert {event["args"]["name"] for event in thread_events} == {
        "prefetch_engine",
        "compute_engine",
        "transfer_engine",
    }

    complete_events = [event for event in tracer._events if event["ph"] == "X"]
    assert len(complete_events) == 3
    assert complete_events[0]["name"] == (
        f"Vector[id={vector_norm.id},type=rms_norm,shape=2x16,bits=16]"
    )
    assert complete_events[1]["name"] == f"SramPrefetch[id={prefetch.id},pages=2]"
    assert complete_events[2]["name"] == (
        f"All2All[id={transfer.id},gpus=4,bytes=128,bits=16]"
    )
    assert [event["cat"] for event in complete_events] == ["compute", "prefetch", "transfer"]
    assert [event["dur"] for event in complete_events] == [pytest.approx(0.003), pytest.approx(0.004), pytest.approx(0.005)]

    monkeypatch.chdir(tmp_path)
    output_path = sim_xpu.save_trace_file("xpu_trace.json")
    assert output_path == "xpu_trace.json"

    with tmp_path.joinpath("xpu_trace.json").open("r", encoding="utf-8") as fh:
        trace_doc = json.load(fh)

    assert trace_doc["displayTimeUnit"] == "ns"
    assert len(trace_doc["traceEvents"]) == len(tracer._events)

    SimSession.reset()


def test_xpu_save_trace_file_requires_enabled_tracing():
    SimSession.reset()
    SimSession.init()

    sim_xpu = xPU(
        make_config(),
        hbm_bandwidth_bytes_per_sec=hbm_bandwidth_bytes_per_sec(),
        enable_trace=False,
    )

    assert sim_xpu.tracer is None
    with pytest.raises(RuntimeError):
        sim_xpu.save_trace_file("disabled_trace.json")

    SimSession.reset()


def test_vallina_xpu_trace_uses_same_save_interface(tmp_path, monkeypatch):
    SimSession.reset()
    SimSession.init()

    vector_norm = VectorOp(vector_op_type="rms_norm", vector_shape=[2, 16], weight_bits=16)
    prefetch = SramPrefetch(num_prefetch_pages=1).with_inputs(vector_norm)

    sim_xpu = VallinaXPU(
        make_config(),
        hbm_bandwidth_bytes_per_sec=hbm_bandwidth_bytes_per_sec(),
        enable_trace=True,
    )
    sim_xpu.load_command([vector_norm, prefetch])

    monkeypatch.setattr(sim_xpu.compute_engine, "execute_macro_op", lambda macro_op: 2.0)

    SimSession.scheduler.run()

    tracer = sim_xpu.tracer
    assert tracer is not None

    complete_events = [event for event in tracer._events if event["ph"] == "X"]
    assert [event["name"] for event in complete_events] == [
        f"Vector[id={vector_norm.id},type=rms_norm,shape=2x16,bits=16]",
        f"SramPrefetch[id={prefetch.id},pages=1]",
    ]

    output_path = tmp_path / "vallina_trace.json"
    assert sim_xpu.save_trace_file(str(output_path)) == str(output_path)
    assert output_path.exists()

    SimSession.reset()
