import torch.fx as fx

from nandmachine.commands.macro import * 
from nandmachine.frontend.core.graph.base import NxGraph
from nandmachine.frontend.network.torch_kernels import RowParallelLinear
from nandmachine.kernels.base import NandKernelBase
from nandmachine.kernels.utils import PageTableAddrPreAllocator




class LinearNandKernel(NandKernelBase):

    def __init__(self) -> None:
        super().__init__()


    def lowering(
            self,
            node:fx.Node,

    ):
        # 开放资源
        nand_file_mmap_ptr = self.pre_addr_allocator.allocate(100)
        # 需要提前拿到 file id 从 node 中获取， 通过 MapperPass 来记录这个 file_id 
        self.global_command_buffer.append(
            NandMmap(1,nand_file_mmap_ptr)
        )

        # prefetch 

        sram_ptr = self.pre_addr_allocator.allocate(100)
        self.command_buffer.append(
            SramPrefetch(nand_file_mmap_ptr,100,sram_ptr)
        )        
        
        # pass  -- 运算



        # 额外的通信操作
        if isinstance(node.meta['target_obj'],RowParallelLinear):
            
            pass 
        
        self.command_buffer.append(
            SramPrefetchRelease(sram_ptr)
        )

