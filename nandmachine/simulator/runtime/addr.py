from nandmachine.config.config import NandConfig, DramConfig, SramConfig


class AddressTranslatorBase:
    def translate(self, addr: int) -> 'AddressBase':
        raise NotImplementedError("Subclasses must implement translate() method")


class AddressBase:
    def __init__(self, device_type: str) -> None:
        self.device_type = device_type


class NandAddressTranslator(AddressTranslatorBase):
    def __init__(self, config: NandConfig):
        self.config = config

    def translate(self, addr: int) -> 'NandAddress':
        return NandAddress(addr, self.config)


class NandAddress(AddressBase):
    # 这里面的地址就到 page 结束了， page 内部的地址不关心 
    def __init__(self, addr: int, config: NandConfig):
        super().__init__("nand")
        self.addr = addr
        self.config = config


    def _encode(self, channel: int, plane: int, page: int, block: int) -> int:
        """Encode components using continuous mixed-radix layout."""
        blocks_per_page = self.config.num_block
        blocks_per_plane = self.config.num_block * self.config.num_pages
        blocks_per_channel = blocks_per_plane * self.config.num_plane

        addr = block
        addr += page * blocks_per_page
        addr += plane * blocks_per_plane
        addr += channel * blocks_per_channel
        return addr

    @property
    def channel(self) -> int:
        """Extract channel from address"""
        blocks_per_channel = self.config.num_block * self.config.num_pages * self.config.num_plane
        return self.addr // blocks_per_channel

    @property
    def plane(self) -> int:
        """Extract plane from address"""
        blocks_per_plane = self.config.num_block * self.config.num_pages
        return (self.addr // blocks_per_plane) % self.config.num_plane

    @property
    def page(self) -> int:
        """Extract page from address"""
        return (self.addr // self.config.num_block) % self.config.num_pages

    @property
    def block(self) -> int:
        """Extract block from address"""
        return self.addr % self.config.num_block

    @channel.setter
    def channel(self, value: int):
        """Set channel component of address"""
        if not 0 <= value < self.config.num_channels:
            raise ValueError(f"Channel {value} out of range [0, {self.config.num_channels})")
        current_plane = self.plane
        current_page = self.page
        current_block = self.block
        self.addr = self._encode(value, current_plane, current_page, current_block)

    @plane.setter
    def plane(self, value: int):
        """Set plane component of address"""
        if not 0 <= value < self.config.num_plane:
            raise ValueError(f"Plane {value} out of range [0, {self.config.num_plane})")
        current_channel = self.channel
        current_page = self.page
        current_block = self.block
        self.addr = self._encode(current_channel, value, current_page, current_block)

    @page.setter
    def page(self, value: int):
        """Set page component of address"""
        if not 0 <= value < self.config.num_pages:
            raise ValueError(f"Page {value} out of range [0, {self.config.num_pages})")
        current_channel = self.channel
        current_plane = self.plane
        current_block = self.block
        self.addr = self._encode(current_channel, current_plane, value, current_block)

    @block.setter
    def block(self, value: int):
        """Set block component of address"""
        if not 0 <= value < self.config.num_block:
            raise ValueError(f"Block {value} out of range [0, {self.config.num_block})")
        current_channel = self.channel
        current_plane = self.plane
        current_page = self.page
        self.addr = self._encode(current_channel, current_plane, current_page, value)

    def is_valid(self) -> bool:
        """Check if address components are within valid ranges"""
        if (
            self.config.num_channels <= 0
            or self.config.num_plane <= 0
            or self.config.num_pages <= 0
            or self.config.num_block <= 0
        ):
            return False
        total = self.config.num_channels * self.config.num_plane * self.config.num_pages * self.config.num_block
        return 0 <= self.addr < total


    def __add__(self, other: int) -> 'NandAddress':
        """Add an integer to the address and return a new NandAddress."""
        if not isinstance(other, int):
            raise TypeError(f"Cannot add {type(other)} to NandAddress")
        if other < 0:
            raise ValueError("Cannot add negative value to address")
        total = self.config.num_channels * self.config.num_plane * self.config.num_pages * self.config.num_block
        next_addr = self.addr + other
        if next_addr >= total:
            raise OverflowError("Address overflow: result exceeds maximum address")
        return NandAddress(next_addr, self.config)

    def to_block_address(self) -> 'NandBlockAddress':
        """
        Convert this NandAddress to a NandBlockAddress.
        Only preserves channel, plane, and block components (discards page).
        """
        return NandBlockAddress.from_components(
            channel=self.channel,
            plane=self.plane,
            block=self.block,
            config=self.config
        )

    def __repr__(self) -> str:
        return (f"NandAddress(addr={self.addr}, "
                f"channel={self.channel}, plane={self.plane}, "
                f"block={self.block}, page={self.page})")


class NandBlockAddress(NandAddress):
    """
    Block-level NAND address that only tracks channel-plane-block.
    Used for NAND free table to track next writable page per block.
    """
    def __init__(self, addr: int, config: NandConfig):
        # Initialize with addr, but we'll override encoding
        super().__init__(addr, config)

    def _encode(self, channel: int, plane: int, block: int) -> int:
        """Encode channel-plane-block using continuous mixed-radix layout."""
        blocks_per_plane = self.config.num_block
        blocks_per_channel = self.config.num_block * self.config.num_plane

        addr = block
        addr += plane * blocks_per_plane
        addr += channel * blocks_per_channel
        return addr

    @property
    def channel(self) -> int:
        """Extract channel from block address"""
        blocks_per_channel = self.config.num_block * self.config.num_plane
        return self.addr // blocks_per_channel

    @property
    def plane(self) -> int:
        """Extract plane from block address"""
        return (self.addr // self.config.num_block) % self.config.num_plane

    @property
    def block(self) -> int:
        """Extract block from block address"""
        return self.addr % self.config.num_block

    @channel.setter
    def channel(self, value: int):
        """Set channel component of block address"""
        if not 0 <= value < self.config.num_channels:
            raise ValueError(f"Channel {value} out of range [0, {self.config.num_channels})")
        current_plane = self.plane
        current_block = self.block
        self.addr = self._encode(value, current_plane, current_block)

    @plane.setter
    def plane(self, value: int):
        """Set plane component of block address"""
        if not 0 <= value < self.config.num_plane:
            raise ValueError(f"Plane {value} out of range [0, {self.config.num_plane})")
        current_channel = self.channel
        current_block = self.block
        self.addr = self._encode(current_channel, value, current_block)

    @block.setter
    def block(self, value: int):
        """Set block component of block address"""
        if not 0 <= value < self.config.num_block:
            raise ValueError(f"Block {value} out of range [0, {self.config.num_block})")
        current_channel = self.channel
        current_plane = self.plane
        self.addr = self._encode(current_channel, current_plane, value)

    @classmethod
    def from_components(cls, channel: int, plane: int, block: int, config: NandConfig) -> 'NandBlockAddress':
        """Create NandBlockAddress from channel, plane, block components"""
        addr = cls(0, config)
        addr.addr = addr._encode(channel, plane, block)
        return addr

    def is_valid(self) -> bool:
        """Check if block address components are within valid ranges"""
        if self.config.num_channels <= 0 or self.config.num_plane <= 0 or self.config.num_block <= 0:
            return False
        total = self.config.num_channels * self.config.num_plane * self.config.num_block
        return 0 <= self.addr < total

    def __repr__(self) -> str:
        return (f"NandBlockAddress(addr={self.addr}, "
                f"channel={self.channel}, plane={self.plane}, "
                f"block={self.block})")

    def __add__(self, other: int) -> 'NandBlockAddress':
        """Add an integer to the block address and return a new NandBlockAddress."""
        if not isinstance(other, int):
            raise TypeError(f"Cannot add {type(other)} to NandBlockAddress")
        if other < 0:
            raise ValueError("Cannot add negative value to address")
        total = self.config.num_channels * self.config.num_plane * self.config.num_block
        next_addr = self.addr + other
        if next_addr >= total:
            raise OverflowError("Block address overflow: result exceeds maximum address")
        return NandBlockAddress(next_addr, self.config)


class DramAddressTranslator(AddressTranslatorBase):
    def __init__(self, config: DramConfig):
        self.config = config

    def translate(self, addr: int) -> 'DramAddress':
        return DramAddress(addr, self.config)


class DramAddress(AddressBase):
    def __init__(self, addr: int, config: DramConfig):
        super().__init__("dram")
        self.addr = addr
        self.config = config

    def __repr__(self) -> str:
        return f"DramAddress(addr={self.addr})"


class SramAddressTranslator(AddressTranslatorBase):
    def __init__(self, config: SramConfig):
        self.config = config

    def translate(self, addr: int) -> 'SramAddress':
        return SramAddress(addr, self.config)


class SramAddress(AddressBase):
    def __init__(self, addr: int, config: SramConfig):
        super().__init__("sram")
        self.addr = addr
        self.config = config

    def __repr__(self) -> str:
        return f"SramAddress(addr={self.addr})"


__all__ = [
    "AddressTranslatorBase",
    "AddressBase",
    "NandAddressTranslator",
    "NandAddress",
    "NandBlockAddress",
    "DramAddressTranslator",
    "DramAddress",
    "SramAddressTranslator",
    "SramAddress",
]
