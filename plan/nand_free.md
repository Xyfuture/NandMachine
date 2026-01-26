
我要实现 nand free table 这个表 (位于 @nandmachine/simulator/runtime/tables.py 中)，这个表的功能是这样的：
- 记录 nand 中哪些页是可写的
- 需要和 nand 的硬件机制对齐
    - 每个 channel，每个 plane， 每个 block 下记录一个 page id 
    - 这个 page id是这个 channel-plane-block 下唯一下一个可写的 page， 其他的 page 在这个 block 中都是不可以写入的，这个 table 中主要就记录这个值
- block_addr 的定义位于 @nandmachine/simulator/runtime/addr.py 中， 用于提供没有 page 的地址，普通地址是 int 的，可以转换为 NandAddress
- 提供如下几个函数
    - allocate(self,block_addr) , 给这个分配下一个 page , 调用 NandAddress 返回一个int 地址
    - free(self,block_addr), 一次性 free 所有的 block， 重置 page id
    - check_free_page(self,addr), 检测这个地址(含有 page 的)是否是能写入的

相关代码实现到 @nandmachine/simulator/runtime/tables.py 中
