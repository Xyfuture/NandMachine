from dataclasses import dataclass


@dataclass 
class MemoryConfig:
    pass 


@dataclass 
class NandConfig(MemoryConfig):
    num_channels:int # 一个 channel 有很多 plane，plane 最基础的读写单元就是 一个  page
    num_plane:int  #  per channel
    num_block:int  #  per plane -- block 是 nand 的erase单元， 暂时没什么用
    num_pages:int  # per block  -- page 是一个 plane 最基础的单元。暂时没什么用 

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
