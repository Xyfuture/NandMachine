from dataclasses import dataclass


@dataclass
class NandConfig:

    num_stacks:int 
    num_channels_per_stack:int 
    num_planes_per_channel:int
    num_blocks_per_plane:int
    num_pages_per_block:int     

    nand_page_size:int # btyes 
    
    tR:float = 0

    pass 