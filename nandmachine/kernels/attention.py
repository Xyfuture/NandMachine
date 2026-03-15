import math

import torch.fx as fx



from nandmachine.commands.macro import FlashAttnOp, MacroOp, SramPrefetch,SramPrefetchRelease
from nandmachine.config.config import NandConfig
from nandmachine.kernels.base import NandKernelBase


class GQANandKernel(NandKernelBase):
    def __int__(self):
        super().__init__()

    def lowering(
            self,
            group_size:int , # GQA 的力度
            num_kv_heads:int, # 多少个 KV head
            head_dim:int, # 每个头的维度
            num_kv_blocks:int  ,# 总共多少个 Block
        
            kv_block_size:int , # 单个 kv block 的 seq len， 多少个 token
            block_bytes:int , # 一个 block 占有多少的 bytes 

            kv_cache_bits:int,
            input_bits:int,
            nand_config:NandConfig

    ):

        # 需要什么参数
        

        macro_op_list:list[MacroOp] = [] 

        hyper_page_size = nand_config.num_plane * nand_config.page_size_bytes

        num_hyper_pgaes:int = math.ceil(block_bytes * num_kv_blocks / nand_config.page_size_bytes)
        
        num_blocks_per_hyper_page = math.floor(hyper_page_size / block_bytes)

        b = num_blocks_per_hyper_page * num_kv_heads
        m = group_size
        k = head_dim
        n = kv_block_size 



        for i in range(num_hyper_pgaes):
            sram_prefetch = SramPrefetch(nand_config.num_plane)
            
            flash_attn = FlashAttnOp(
                qk_bmm_shape=(b,m,k,n),
                sv_bmm_shape=(b,m,n,k),
                softmax_shape=(b*m,n),
            ).with_inputs(sram_prefetch)

            sram_release = SramPrefetchRelease().with_inputs(flash_attn)

            macro_op_list.extend([
                sram_prefetch,
                flash_attn,
                sram_release,
            ])

            