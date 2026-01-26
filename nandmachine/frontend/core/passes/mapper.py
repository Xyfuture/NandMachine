# 运行 得出映射表
import torch
import torch.fx as fx
import torch.nn as nn 


from nandmachine.config.config import NandConfig
from nandmachine.frontend.core.passes.base import GraphPass
from nandmachine.frontend.network.torch_kernels import LinearBase
from nandmachine.simulator.runtime.addr import NandAddress, NandBlockAddress
from nandmachine.simulator.runtime.tables import NandFileEntry, NandFileTable, NandFreeTable, Permission


class NandTableManager:
    # 记录 NandFileTable 和 NandFreeTable 
    def __init__(self,config:NandConfig=None) -> None:
        
        self.config = config

        self.nand_file_table = NandFileTable()
        self.nand_free_table = NandFreeTable(self.config)
        
        self.next_block_addr = NandBlockAddress(0,self.config) # 后续继续改进吧，这个不是很对
        pass 

    def create_new_file(self,num_pages)->int:

        cur_file_id = self.nand_file_table.get_new_file_id()

        allocated_pages = []

        # 首先更新 channel，再更新 plane，最后更新 page
        for i in range(num_pages):
            next_addr = self.nand_free_table.allocate(self.next_block_addr)
            while not next_addr:
                # 应该每个 plane 之类的维护一个 
                self.next_block_addr+= 1
                next_addr = self.nand_free_table.allocate(self.next_block_addr)
            
            allocated_pages.append(next_addr)

            self.next_page_addr += 1

        file_entry = NandFileEntry(cur_file_id,allocated_pages,Permission.READ,'weight')

        self.nand_file_table.add_entry(file_entry)

        return cur_file_id 
    

        



class MapperPass(GraphPass):
    def __init__(self) -> None:
        super().__init__()

        self._to_map_type = {
            LinearBase
        }

        self.nand_table_manager = NandTableManager() 

        # 需要创建 nand table
        # 两部分， 一个是 nand table 相关的 一个 class
        # 另一部分是 拓扑排序相关的部分 


    def transform(self,graph:fx.Graph):
        if not hasattr(graph,'meta'):
            graph.meta = {}  # type: ignore
        
        for node in graph.nodes:
            if "nand_store_pages" in node.meta:
                pages = node.meta['nand_store_pages']
                file_id = self.nand_table_manager.create_new_file(pages)

                node.meta['file_id'] = file_id 

        # 转化结束后，给 graph 中记录上 page_table
        # graph.meta['nand_file_table'] = 

        


