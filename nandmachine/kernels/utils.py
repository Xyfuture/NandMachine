


class PageTableAddrPreAllocator:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        self.base_addr = 2**30
        self.base_addr_step = 2**30
        self._initialized = True

    def allocate(self, num_pages: int) -> int:
        self.base_addr += self.base_addr_step
        return self.base_addr

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of PageTableAddrPreAllocator"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance