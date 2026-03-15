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

    page_size:int # KB

    sram_threshold:int # KB

    @property
    def page_size_bytes(self) -> int:
        """Page size in bytes."""
        return self.page_size * 1024


@dataclass
class DramConfig(MemoryConfig):
    pass 


@dataclass 
class SramConfig(MemoryConfig):
    pass 




# TODO
# 补全后面的 xpu config 
