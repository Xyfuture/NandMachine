import torch.fx as fx



from nandmachine.commands.macro import FlashAttnOp, MacroOp, SramPrefetch,SramPrefetchRelease
from nandmachine.config.config import NandConfig
from nandmachine.kernels.base import NandKernelBase


class GQANandKernel(NandKernelBase):
    def __int__(self):
        super().__init__()

    def lowering(
            self,
            node:fx.Node
    ):

        nand_config = NandConfig()

        macro_op_list:list[MacroOp] = [] 

        num_hyper_pgaes:int = 1024 


        for i in range(num_hyper_pgaes):
            sram_prefetch = SramPrefetch(nand_config.num_plane)
            
            flash_attn = FlashAttnOp(
                qk_bmm_shape=(1,2,3,4),
                sv_bmm_shape=(1,2,3,4),
                softmax_shape=(12,3),
            ).with_inputs(sram_prefetch)

            sram_release = SramPrefetchRelease().with_inputs(flash_attn)

            macro_op_list.extend([
                sram_prefetch,
                flash_attn,
                sram_release,
            ])

            