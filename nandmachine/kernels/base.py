

from nandmachine.commands.macro import MacroOp
from nandmachine.kernels.utils import PageTableAddrPreAllocator


class NandKernelBase:
    def __init__(self) -> None:
        
        self.command_buffer:list[MacroOp] = []
        self.global_command_buffer:list[MacroOp] = []

        self.pre_addr_allocator = PageTableAddrPreAllocator.get_instance()

    