from typing import Optional
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
    def __init__(self, addr: int, config: NandConfig):
        super().__init__("nand")
        self.addr = addr
        self.config = config
        # self.channel: Optional[int] = None
        # self.plane: Optional[int] = None
        # self.block: Optional[int] = None
        # self.page: Optional[int] = None

    def _get_block_digits(self) -> int:
        """Get number of decimal digits for block component"""
        return len(str(self.config.num_block - 1)) if self.config.num_block > 0 else 1

    def _get_page_digits(self) -> int:
        """Get number of decimal digits for page component"""
        return len(str(self.config.num_pages - 1)) if self.config.num_pages > 0 else 1

    def _get_plane_digits(self) -> int:
        """Get number of decimal digits for plane component"""
        return len(str(self.config.num_plane - 1)) if self.config.num_plane > 0 else 1

    def _get_channel_digits(self) -> int:
        """Get number of decimal digits for channel component"""
        return len(str(self.config.num_channels - 1)) if self.config.num_channels > 0 else 1

    def _encode(self, channel: int, plane: int, page: int, block: int) -> int:
        """Encode components into a single address integer"""
        block_digits = self._get_block_digits()
        page_digits = self._get_page_digits()
        plane_digits = self._get_plane_digits()

        addr = block
        addr += page * (10 ** block_digits)
        addr += plane * (10 ** (block_digits + page_digits))
        addr += channel * (10 ** (block_digits + page_digits + plane_digits))

        return addr

    @property
    def channel(self) -> int:
        """Extract channel from address"""
        block_digits = self._get_block_digits()
        page_digits = self._get_page_digits()
        plane_digits = self._get_plane_digits()
        divisor = 10 ** (block_digits + page_digits + plane_digits)
        return self.addr // divisor

    @property
    def plane(self) -> int:
        """Extract plane from address"""
        block_digits = self._get_block_digits()
        page_digits = self._get_page_digits()
        plane_digits = self._get_plane_digits()
        addr_without_block = self.addr // (10 ** block_digits)
        addr_without_page = addr_without_block // (10 ** page_digits)
        return addr_without_page % (10 ** plane_digits)

    @property
    def page(self) -> int:
        """Extract page from address"""
        block_digits = self._get_block_digits()
        page_digits = self._get_page_digits()
        addr_without_block = self.addr // (10 ** block_digits)
        return addr_without_block % (10 ** page_digits)

    @property
    def block(self) -> int:
        """Extract block from address"""
        block_digits = self._get_block_digits()
        return self.addr % (10 ** block_digits)

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
        try:
            return (0 <= self.channel < self.config.num_channels and
                    0 <= self.plane < self.config.num_plane and
                    0 <= self.page < self.config.num_pages and
                    0 <= self.block < self.config.num_block)
        except:
            return False


    def __add__(self, other: int) -> 'NandAddress':
        """
        Add an integer to the address with proper carry logic.
        Creates a new NandAddress instance.
        """
        if not isinstance(other, int):
            raise TypeError(f"Cannot add {type(other)} to NandAddress")
        if other < 0:
            raise ValueError("Cannot add negative value to address")

        # Extract current components
        block = self.block
        page = self.page
        plane = self.plane
        channel = self.channel

        # Add to block and handle carries
        block += other

        # Carry to page
        if block >= self.config.num_block:
            page += block // self.config.num_block
            block = block % self.config.num_block

        # Carry to plane
        if page >= self.config.num_pages:
            plane += page // self.config.num_pages
            page = page % self.config.num_pages

        # Carry to channel
        if plane >= self.config.num_plane:
            channel += plane // self.config.num_plane
            plane = plane % self.config.num_plane

        # Check overflow
        if channel >= self.config.num_channels:
            raise OverflowError(f"Address overflow: result exceeds maximum address")

        # Create new address
        new_addr = NandAddress(0, self.config)
        new_addr.addr = new_addr._encode(channel, plane, page, block)

        return new_addr

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
        """Encode channel-plane-block into a single address integer (no page)"""
        block_digits = self._get_block_digits()
        plane_digits = self._get_plane_digits()

        addr = block
        addr += plane * (10 ** block_digits)
        addr += channel * (10 ** (block_digits + plane_digits))

        return addr

    @property
    def channel(self) -> int:
        """Extract channel from block address"""
        block_digits = self._get_block_digits()
        plane_digits = self._get_plane_digits()
        divisor = 10 ** (block_digits + plane_digits)
        return self.addr // divisor

    @property
    def plane(self) -> int:
        """Extract plane from block address"""
        block_digits = self._get_block_digits()
        plane_digits = self._get_plane_digits()
        addr_without_block = self.addr // (10 ** block_digits)
        return addr_without_block % (10 ** plane_digits)

    @property
    def block(self) -> int:
        """Extract block from block address"""
        block_digits = self._get_block_digits()
        return self.addr % (10 ** block_digits)

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
        try:
            return (0 <= self.channel < self.config.num_channels and
                    0 <= self.plane < self.config.num_plane and
                    0 <= self.block < self.config.num_block)
        except:
            return False

    def __repr__(self) -> str:
        return (f"NandBlockAddress(addr={self.addr}, "
                f"channel={self.channel}, plane={self.plane}, "
                f"block={self.block})")

    def __add__(self, other: int) -> 'NandBlockAddress':
        """
        Add an integer to the block address with proper carry logic.
        Creates a new NandBlockAddress instance.
        """
        if not isinstance(other, int):
            raise TypeError(f"Cannot add {type(other)} to NandBlockAddress")
        if other < 0:
            raise ValueError("Cannot add negative value to address")

        # Extract current components
        block = self.block
        plane = self.plane
        channel = self.channel

        # Add to block and handle carries
        block += other

        # Carry to plane
        if block >= self.config.num_block:
            plane += block // self.config.num_block
            block = block % self.config.num_block

        # Carry to channel
        if plane >= self.config.num_plane:
            channel += plane // self.config.num_plane
            plane = plane % self.config.num_plane

        # Check overflow
        if channel >= self.config.num_channels:
            raise OverflowError(f"Block address overflow: result exceeds maximum address")

        # Create new address
        new_addr = NandBlockAddress(0, self.config)
        new_addr.addr = new_addr._encode(channel, plane, block)

        return new_addr


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
