from dataclasses import dataclass


@dataclass 
class MemoryConfig:
    pass 


@dataclass 
class NandConfig(MemoryConfig):
    num_channels:int
    num_plane:int  #  per channel
    num_block:int  #  per plane
    num_pages:int  # per block 

    tRead: float # ns
    tWrite: float 
    tErase: float 


@dataclass
class DramConfig(MemoryConfig):
    pass 


@dataclass 
class SramConfig(MemoryConfig):
    pass 


# TODO
# 补全后面的 xpu config 