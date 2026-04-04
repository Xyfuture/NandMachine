import math

import torch.fx as fx



from nandmachine.commands.macro import FlashAttnOp, MacroOp, SramPrefetch,SramPrefetchRelease
from nandmachine.config.config import NandConfig
from nandmachine.kernels.base import HBMKernelBase, NandKernelBase


class GQANandKernel(NandKernelBase):
    def __int__(self):
        super().__init__()

    @classmethod
    def lowering(
            cls,
            group_size:int , # GQA 的力度
            num_kv_heads:int, # 多少个 KV head
            head_dim:int, # 每个头的维度
            num_kv_blocks:int  ,# 总共多少个 Block
        
            kv_block_size:int , # 单个 kv block 的 seq len， 多少个 token
            block_bytes:int , # 一个 block 占有多少的 bytes 

            kv_cache_bits:int,
            input_bits:int,
            nand_config:NandConfig

    )->list[MacroOp]:

        # 需要什么参数
        

        macro_op_list:list[MacroOp] = [] 

        hyper_page_size = nand_config.num_channels * nand_config.num_plane * nand_config.page_size_bytes

        num_hyper_pgaes:int = math.ceil(block_bytes * num_kv_blocks / hyper_page_size)
        
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
                weight_bits=kv_cache_bits,
            ).with_inputs(sram_prefetch)

            sram_release = SramPrefetchRelease().with_inputs(flash_attn)

            macro_op_list.extend([
                sram_prefetch,
                flash_attn,
                sram_release,
            ])
        
        return macro_op_list
        


class GQAHBMKernel(HBMKernelBase):
    def __init__(self) -> None:
        super().__init__()
    

    @classmethod
    def lowering(
            cls,
            group_size:int,
            num_kv_heads:int,
            head_dim:int,
            num_kv_blocks:int,
            kv_block_size:int,
            block_bytes:int,
            kv_cache_bits:int,
            input_bits:int,
            nand_config:NandConfig
    )->list[MacroOp]:
        del block_bytes, input_bits, nand_config

        dims = {
            "group_size": group_size,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "num_kv_blocks": num_kv_blocks,
            "kv_block_size": kv_block_size,
        }
        invalid_dims = {name: value for name, value in dims.items() if value <= 0}
        if invalid_dims:
            raise ValueError(f"GQAHBMKernel expects positive dims, got {invalid_dims}")
        if kv_cache_bits <= 0 or kv_cache_bits % 8 != 0:
            raise ValueError(
                f"kv_cache_bits must be a positive multiple of 8, got {kv_cache_bits}"
            )

        b = num_kv_blocks * num_kv_heads
        m = group_size
        k = head_dim
        n = kv_block_size

        return [
            FlashAttnOp(
                qk_bmm_shape=(b, m, k, n),
                sv_bmm_shape=(b, m, n, k),
                softmax_shape=(b * m, n),
                weight_bits=kv_cache_bits,
            )
        ]
