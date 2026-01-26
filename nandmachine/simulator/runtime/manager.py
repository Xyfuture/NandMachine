
from nandmachine.simulator.runtime.entries import NandMmapEntry, RuntimeResourceTable
from nandmachine.simulator.runtime.tables import * 
from nandmachine.commands.macro import *



class RuntimeManager():
    def __init__(self):
        

        # 所有的 table 都在 这里面管理， table 之间不会联动， 这里有点类似 os 的感觉了


        self.nand_file_table:NandFileTable = None 
        
        self.nand_free_table:NandFreeTable = None


        self.dram_free_table:DRAMFreeTable = None
        self.sram_free_table:SRAMFreeTable = None

        self.page_table: PageTable = PageTable(page_size=4096)


        self.resource_usage_table = RuntimeResourceTable()

        # TODO
        # - page table design
        # - config design
        # - kernel design  

        
        # Procedure
        # - 初始化 这一系列的 table
        # - 完成 nand file table 初步赋值
        # - 实现





    def NandMmapHandler(self,command:NandMmap):
        # 从 nand file table 中拿一个 file
        
        file_id = command.file_id

        nand_file_entry = self.nand_file_table.get_file_by_id(file_id)
        
        assert nand_file_entry


        # 创建 nand_mmap_entry
        logic_addr_base = self.page_table.get_next_free_addr()


        nand_mmap_entry = NandMmapEntry(
            logic_addr_base,nand_file_entry.num_nand_pages,file_id,Permission.WRITE,4096
        )
        
        self.resource_usage_table.add_entry(nand_mmap_entry)


        # 开始映射
        for i,nand_page_id in enumerate(nand_file_entry.nand_pages):
            self.page_table.map_page(
                logic_addr_base+i,DeviceType.NAND,nand_page_id,Permission.READ
            )

        


    def NandMunmapHandler(self, command:NandMunmap):
        logic_addr_base = command.addr

        nand_mmap_entry = self.resource_usage_table.get_entry(logic_addr_base)

        assert isinstance(nand_mmap_entry,NandMmapEntry)

        for i in range(nand_mmap_entry.size):
            self.page_table.unmap_page(logic_addr_base+i)
        
        self.resource_usage_table.remove_entry(logic_addr_base)




    def SramPrefetchHandler(self):
        # 这个是最为麻烦的， 因为涉及到映射失败的情况。
        # 需要回撤一下操作，可能会出现一部分在 sram 中， 一部分在 nand 中

        pass 



    def SramPrefetchReleaseHandler(self, command:SramPrefetchRelease):
        logic_addr_base = command.addr
        
        sram_prefetch_entry = self.resource_usage_table.get_entry(logic_addr_base)

        # 需要释放 sram 资源，这里资源的记录需要处理一下

        pass  


    def DramMallocHandler(self):
        pass 

    def DramFreeHandler(self):
        pass 

    def SramMallocHandler(self):
        # 分配失败的话， 要进行报错的操作 
        
        pass 

    def SramFreeHandler(self):
        pass 


    


    