
from nandmachine.simulator.runtime.tables import * 
from nandmachine.commands.macro import *



class RuntimeManager():
    def __init__(self):
        
        self.nand_file_table:NandFileTable = None 
        
        self.nand_free_table:NandFreeTable = None


        self.dram_free_table:DRAMFreeTable = None
        self.sram_free_table:SRAMFreeTable = None

        self.page_table: PageTable = PageTable(page_size=4096)

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

        

        pass  


    def NandMunmapHandler(self):
        pass

    def SramPrefetchHandler(self):

        pass 



    def SramPrefetchReleaseHandler(self):
        pass 


    def DramMallocHandler(self):
        pass 

    def DramFreeHandler(self):
        pass 

    def SramMallocHandler(self):
        pass 

    def SramFreeHandler(self):
        pass 


    


    