from collections import deque
from typing import Optional

from Desim.Core import SimModule, Event, SimTime
from Desim import EventQueue, SimSession

from nandmachine.commands.macro import MacroOp, SramPrefetch
from nandmachine.commands.micro import (
    DataForward,
    MemoryBasicOpBase,
    MemoryOperation,
    NandBlockErase,
    NandPageRead,
    NandPageWrite,
    NandRequest,
    SramPageRead,
    SramPageWrite,
)
from nandmachine.config.config import NandConfig
from nandmachine.simulator.hardware.utils import DepSlot
from nandmachine.simulator.runtime.addr import NandAddress



class NandController(SimModule):
    def __init__(self,nand_config:NandConfig):
        super().__init__()

        self.nand_config = nand_config

        self.waiting_requests_queue:deque[DepSlot[NandRequest]] = deque()


        self.core_event_queue:EventQueue = EventQueue()

        self.nand_sim_core:NandSimCore =  NandSimCore(self.nand_config) 

        self.register_coroutine(self.process)
        

    
    def process(self):
        while True:
            SimModule.wait(self.core_event_queue.event)

            # 处理新请求
            while self.waiting_requests_queue:
                cur_slot = self.waiting_requests_queue.popleft()

                nand_request  = cur_slot.payload


                finish_time = self.nand_sim_core.handle_request(nand_request,SimSession.sim_time.cycle)

                cur_slot.is_finished = True
                cur_slot.finish_event.notify(SimTime(int(finish_time-SimSession.sim_time.cycle)))
        

    def handle_request(self,nand_request_slot:DepSlot[NandRequest]):
        self.waiting_requests_queue.append(nand_request_slot)
        self.core_event_queue.next_notify(SimTime(1))





class NandSimCore:
    """
    Non-preemptive NAND timing simulator core.

    模型说明：
    - 每类资源维护 free_time，表示该资源下一次可用时刻
    - 本阶段支持的资源：
      1) plane 读取电路（按 channel+plane）
      2) nand -> base 通路（按 channel）
      3) base -> xpu 通路（按 channel）
    - 一条 request 内 memory operation 串行执行
    - operation 内各 micro op 也按顺序执行
    """
    def __init__(
        self,
        nand_config: NandConfig,
    ) -> None:
        self.nand_config = nand_config
        self.t_nand_to_base = 100
        self.t_base_to_xpu = 100
        self.t_sram_read = 10

        self.plane_free_time: dict[tuple[int, int], float] = {}
        self.nand_to_base_free_time: dict[int, float] = {}
        self.base_to_xpu_free_time: dict[int, float] = {}

    def handle_request(self, request: NandRequest, arrive_time: float) -> float:
        """
        Handle one request and return its finish timestamp (ns).

        不同 MemoryOperation 在同一 arrive_time 并行发起，
        request 完成时间为所有 operation 完成时间的最大值。
        """
        finish_time = arrive_time
        for memory_op in request.operations:
            op_finish = self._handle_memory_operation(memory_op, arrive_time)
            finish_time = max(finish_time, op_finish)
        return finish_time

    def _handle_memory_operation(self, mem_op: MemoryOperation, start_time: float) -> float:
        """
        Execute one MemoryOperation sequentially.

        `last_channel` records the most recent resolved channel in this
        operation and is used by DataForward fallback logic.
        """
        cur_time = start_time
        last_channel: Optional[int] = None

        for op in mem_op.op_list:
            if isinstance(op, DataForward):
                cur_time, last_channel = self._handle_data_forward(op, cur_time, last_channel)
                continue

            if isinstance(op, MemoryBasicOpBase):
                cur_time, last_channel = self._handle_basic_op(op, cur_time, last_channel)
                continue

            raise TypeError(f"Unsupported operation type: {type(op)}")

        return cur_time

    def _handle_basic_op(
        self,
        basic_op: MemoryBasicOpBase,
        start_time: float,
        last_channel: Optional[int],
    ) -> tuple[float, Optional[int]]:
        if isinstance(basic_op, NandPageRead):
            return self._simulate_nand_page_read(basic_op.addr, start_time)

        if isinstance(basic_op, SramPageRead):
            return self._simulate_sram_page_read(basic_op.addr, start_time, last_channel)

        if isinstance(basic_op, (NandPageWrite, NandBlockErase, SramPageWrite)):
            self._raise_unsupported_op(basic_op)

        raise TypeError(f"Unsupported memory basic operation: {type(basic_op)}")

    def _handle_data_forward(
        self,
        forward_op: DataForward,
        start_time: float,
        last_channel: Optional[int],
    ) -> tuple[float, int]:
        """
        Handle a DataForward micro-op and return (finish_time, resolved_channel).

        路径支持：
        - nand -> base
        - base -> xpu
        """
        channel = self._resolve_forward_channel(forward_op, last_channel)

        if forward_op.src_type == "nand" and forward_op.dst_type == "base":
            end_time = self._simulate_nand_to_base(channel, start_time)
            return end_time, channel

        if forward_op.src_type == "base" and forward_op.dst_type == "xpu":
            end_time = self._simulate_base_to_xpu(channel, start_time)
            return end_time, channel

        raise NotImplementedError(
            f"Unsupported data forward path: {forward_op.src_type}->{forward_op.dst_type}"
        )

    def _resolve_forward_channel(
        self,
        forward_op: DataForward,
        last_channel: Optional[int],
    ) -> int:
        """
        Resolve channel for DataForward.

        规则：
        1) 优先使用 src/dst 端点解析出的 channel
        2) src/dst 缺失时回退到 last_channel
        3) src/dst 两端若冲突，报错
        4) 与 last_channel 冲突，报错
        """
        src_channel = self._resolve_endpoint_channel(forward_op.src_type, forward_op.src)
        dst_channel = self._resolve_endpoint_channel(forward_op.dst_type, forward_op.dst)

        endpoint_channels = [channel for channel in (src_channel, dst_channel) if channel is not None]

        if endpoint_channels and len(set(endpoint_channels)) != 1:
            raise ValueError(
                f"DataForward endpoint channel conflict: src={src_channel}, dst={dst_channel}"
            )

        resolved_channel = endpoint_channels[0] if endpoint_channels else last_channel
        if resolved_channel is None:
            raise ValueError("Cannot resolve channel for DataForward without endpoint or context")

        if last_channel is not None and resolved_channel != last_channel:
            raise ValueError(
                f"DataForward channel mismatch with previous operation: {resolved_channel} != {last_channel}"
            )

        return resolved_channel

    def _resolve_endpoint_channel(self, endpoint_type: str, endpoint_value: Optional[int]) -> Optional[int]:
        if endpoint_value is None:
            return None

        if endpoint_type == "nand":
            channel, _ = self._decode_nand_addr(endpoint_value)
            return channel

        if endpoint_type in {"base", "xpu", "sram"}:
            if not isinstance(endpoint_value, int):
                raise TypeError(f"Endpoint channel must be int for {endpoint_type}")
            self._validate_channel(endpoint_value)
            return endpoint_value

        raise NotImplementedError(f"Unsupported endpoint type: {endpoint_type}")

    def _simulate_nand_page_read(self, addr: int, start_time: float) -> tuple[float, int]:
        """
        Simulate one NAND page read on (channel, plane).

        同一个 (channel, plane) 串行，不同 plane 可以并行。
        """
        channel, plane = self._decode_nand_addr(addr)
        key = (channel, plane)
        begin_time = max(start_time, self.plane_free_time.get(key, 0.0))
        end_time = begin_time + self.nand_config.tRead
        self.plane_free_time[key] = end_time
        return end_time, channel

    def _simulate_sram_page_read(
        self,
        addr: int,
        start_time: float,
        last_channel: Optional[int],
    ) -> tuple[float, int]:
        """
        Simulate SRAM read with fixed latency.

        SRAM 地址当前不解码 channel；优先继承 last_channel，缺失时默认 channel=0。
        """
        _ = addr
        channel = 0 if last_channel is None else last_channel
        self._validate_channel(channel)
        end_time = start_time + self.t_sram_read
        return end_time, channel

    def _simulate_nand_to_base(self, channel: int, start_time: float) -> float:
        """Simulate channel-local nand->base transfer."""
        self._validate_channel(channel)
        begin_time = max(start_time, self.nand_to_base_free_time.get(channel, 0.0))
        end_time = begin_time + self.t_nand_to_base
        self.nand_to_base_free_time[channel] = end_time
        return end_time

    def _simulate_base_to_xpu(self, channel: int, start_time: float) -> float:
        """Simulate channel-local base->xpu transfer."""
        self._validate_channel(channel)
        begin_time = max(start_time, self.base_to_xpu_free_time.get(channel, 0.0))
        end_time = begin_time + self.t_base_to_xpu
        self.base_to_xpu_free_time[channel] = end_time
        return end_time

    def _decode_nand_addr(self, addr: int) -> tuple[int, int]:
        nand_addr = NandAddress(addr, self.nand_config)
        self._validate_channel(nand_addr.channel)
        if not 0 <= nand_addr.plane < self.nand_config.num_plane:
            raise ValueError(f"Invalid plane index: {nand_addr.plane}")
        return nand_addr.channel, nand_addr.plane

    def _validate_channel(self, channel: int) -> None:
        if not 0 <= channel < self.nand_config.num_channels:
            raise ValueError(
                f"Channel {channel} out of range [0, {self.nand_config.num_channels})"
            )

    def _raise_unsupported_op(self, op: object) -> None:
        raise NotImplementedError(f"Operation not implemented yet: {type(op).__name__}")

    def reset(self) -> None:
        self.plane_free_time.clear()
        self.nand_to_base_free_time.clear()
        self.base_to_xpu_free_time.clear()
