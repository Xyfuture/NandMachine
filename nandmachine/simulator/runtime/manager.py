
from typing import Optional
from nandmachine.config.config import NandConfig
from nandmachine.simulator.runtime.entries import NandMmapEntry, PrefetchEntry, RuntimeResourceTable
from nandmachine.simulator.runtime.tables import * 
from nandmachine.simulator.runtime.addr import NandBlockAddress
from nandmachine.commands.macro import *
from nandmachine.simulator.runtime.tables import NandFileEntry
from nandmachine.simulator.runtime.tables import NandFileMeta



class NandFileSystem:
    def __init__(self,nand_config:NandConfig) -> None:
        self.nand_config = nand_config


        self.nand_file_table:NandFileTable = NandFileTable(self.nand_config) 
        
        self.nand_free_table:NandFreeTable = NandFreeTable(self.nand_config)

        # 要在这里面维护好 static file 的地址逻辑，保证地址是连续的， block 之间的关系不用管
        # 模拟器层次不考虑这些信息
        self._total_plane_slots = self.nand_config.num_channels * self.nand_config.num_plane
        self._next_plane_slot = 0
        self._slot_next_block: dict[int, int] = {
            slot: 0 for slot in range(self._total_plane_slots)
        }


    def create_static_file(self,nand_file_meta:NandFileMeta) -> int:
        assert nand_file_meta.num_pages > 0
        assert self.nand_file_table.get_file_by_id(nand_file_meta.file_id) is None
        assert self._total_plane_slots > 0

        start_slot = self._next_plane_slot
        allocated_pages: list[int] = []

        for i in range(nand_file_meta.num_pages):
            slot = (start_slot + i) % self._total_plane_slots
            channel = slot // self.nand_config.num_plane
            plane = slot % self.nand_config.num_plane
            block = self._slot_next_block.get(slot, 0)

            while True:
                if block >= self.nand_config.num_block:
                    assert False

                block_addr = NandBlockAddress.from_components(
                    channel=channel,
                    plane=plane,
                    block=block,
                    config=self.nand_config,
                )
                page_addr = self.nand_free_table.allocate(block_addr)

                if page_addr is not None:
                    allocated_pages.append(page_addr)
                    self._slot_next_block[slot] = block
                    break

                block += 1
                self._slot_next_block[slot] = block

        file_entry = NandFileEntry(nand_file_meta, allocated_pages)
        self.nand_file_table.add_entry(file_entry)

        self._next_plane_slot = (
            start_slot + nand_file_meta.num_pages
        ) % self._total_plane_slots

        return file_entry.file_id

    
    # TODO 细化一下相关的描述
    def create_kv_cache(self,):
        pass




class RuntimeManager():
    def __init__(self,nand_config:NandConfig):
        

        # 所有的 table 都在 这里面管理， table 之间不会联动， 这里有点类似 os 的感觉了

        self.nand_config = nand_config


        self.nand_file_table:NandFileTable = NandFileTable(self.nand_config) 
        
        self.nand_free_table:NandFreeTable = NandFreeTable(self.nand_config)


        self.dram_free_table:DRAMFreeTable = DRAMFreeTable()
        self.sram_free_table:SRAMFreeTable = SRAMFreeTable()

        self.page_table: PageTable = PageTable(page_size=self.nand_config.page_size_bytes)


        self.resource_usage_table = RuntimeResourceTable()

        # TODO
        # - page table design
        # - config design
        # - kernel design  

        
        # Procedure
        # - 初始化 这一系列的 table
        # - 完成 nand file table 初步赋值
        # - 实现


    def load_nand_file_system(self,nand_file_table:NandFileTable,nand_free_table:NandFreeTable):
        self.nand_file_table = nand_file_table
        self.nand_free_table = nand_free_table





    def NandMmapHandler(self,command:NandMmap):
        # 从 nand file table 中拿一个 file
        
        file_id = command.file_id

        nand_file_entry:Optional[NandFileEntry] = self.nand_file_table.get_file_by_id(file_id)
        
        if not nand_file_entry:
            assert False
        

        # 创建 nand_mmap_entry
        logic_addr_base = command.pre_alloc_logic_addr # 使用好预先分配的地址，不用每次重新找了


        nand_mmap_entry = NandMmapEntry(
            logic_addr_base,
            nand_file_entry.num_nand_pages,
            file_id,
            Permission.READ,
            self.nand_config.page_size_bytes,
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



    def SramPrefetchHandler(self,command:SramPrefetch):
        # 这个是最为麻烦的， 因为涉及到映射失败的情况。
        # 需要回撤一下操作，可能会出现一部分在 sram 中， 一部分在 nand 中

        src_base = command.prefetch_addr
        dst_base = command.pre_alloc_logic_addr
        num_pages = command.num_pages

        allocated_sram_pages: list[int] = []
        mapped_dst_pages: list[int] = []
        source_mapping: dict[int, int] = {}

        sram_allocate = self.sram_free_table.allocate_page
        sram_free = self.sram_free_table.free_page
        map_page = self.page_table.map_page
        unmap_page = self.page_table.unmap_page
        translate = self.page_table.translate

        for offset in range(num_pages):
            src_page = src_base + offset
            dst_page = dst_base + offset

            src_phy_info = translate(src_page)
            if not src_phy_info:
                for page in mapped_dst_pages:
                    unmap_page(page)
                for phy_page in allocated_sram_pages:
                    sram_free(phy_page)
                assert False

            src_device_type, _ = src_phy_info
            assert src_device_type == DeviceType.NAND

            sram_phy_page = sram_allocate()
            if sram_phy_page == -1:
                for page in mapped_dst_pages:
                    unmap_page(page)
                for phy_page in allocated_sram_pages:
                    sram_free(phy_page)
                assert False

            allocated_sram_pages.append(sram_phy_page)

            if not map_page(dst_page, DeviceType.SRAM, sram_phy_page, Permission.READ):
                for page in mapped_dst_pages:
                    unmap_page(page)
                for phy_page in allocated_sram_pages:
                    sram_free(phy_page)
                assert False

            mapped_dst_pages.append(dst_page)
            source_mapping[dst_page] = src_page

        prefetch_entry = PrefetchEntry(
            dst_base,
            num_pages,
            source_mapping,
            self.nand_config.page_size_bytes,
        )

        if not self.resource_usage_table.add_entry(prefetch_entry):
            for page in mapped_dst_pages:
                unmap_page(page)
            for phy_page in allocated_sram_pages:
                sram_free(phy_page)
            assert False



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


    


    
