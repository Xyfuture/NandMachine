import math

import torch.fx as fx

from nandmachine.commands.macro import * 
from nandmachine.config.config import NandConfig
from nandmachine.frontend.core.graph.base import NxGraph
from nandmachine.frontend.network.torch_kernels import RowParallelLinear
from nandmachine.kernels.base import NandKernelBase
from nandmachine.kernels.utils import PageTableAddrPreAllocator





class LinearNandKernel(NandKernelBase):

    def __init__(self) -> None:
        super().__init__()


    @classmethod
    def lowering(
            cls,
            m: int,
            k: int,
            n: int,
            weight_bits: int,
            input_bits: int,
            nand_config: NandConfig,
    ):
        macro_op_list: list[MacroOp] = []
        sram_threshold = nand_config.sram_threshold * 1024 # KB->Bytes

        n_slice = sram_threshold // (k * (weight_bits // 8))                                                                                                  
        num_splits = math.ceil(n / n_slice)                                                                                                                   
                                                                                                                                                                
        n_shape_list = []                                                                                                                                     
        for i in range(num_splits):                                                                                                                           
            n_i = min(n_slice, n - i * n_slice)                                                                                                               
            n_shape_list.append(n_i) 


        for cur_n in n_shape_list:
            prefetch_pages = math.ceil( ((k*cur_n*weight_bits+7)//8)/ nand_config.page_size )
            
            sram_prefetch = SramPrefetch(prefetch_pages)

            matmul = MatMulOp((m,k,cur_n),weight_bits=weight_bits).with_inputs(sram_prefetch)

            sram_release = SramPrefetchRelease().with_inputs(matmul)

            macro_op_list.extend([
                sram_prefetch,
                matmul,
                sram_release,
            ])


        return macro_op_list





        

    
