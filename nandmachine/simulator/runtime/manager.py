
from nandmachine.simulator.runtime.tables import * 




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

    


    