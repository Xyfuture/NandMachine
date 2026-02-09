我要对 @nandmachine\simulator\runtime\addr.py 进行一下重构, 将这里面的地址表示方式改为连续递增的方式, 现在这里面是一种方便阅读的方式,要改变这个. 这是原始方便阅读方式使用的 prompt

"
我要实现 addr 解析机制，实现到 @nandmachine/simulator/runtime/addr.py 中，会和 @nandmachine/config/config.py 有关
功能描述
nand 的层次架构是按照 channel-plane-block-page 的层次布置的，可以类比一下的内存的hierarchy，我们要根据这个分配地址(实际的地址分配可能不是按照上面那个顺序进行的)。我想采用一个比较易读的方式，比如一个block 一共有 2048 个 page，那么十进制的地址后四位只会给 page，其他的层次类似，不会有 2049 这样的地址。
然后基于这样的地址分配逻辑，完成对于 @nandmachine/simulator/runtime/addr.py 中的 NandAddress 的解析操作, 要求如下：
- addr 要满足易读地址的设计
- 地址的顺序是  block -> page -> plane -> channel 
- channel,plane,block,page 都可以用@property 的方式获取和设置(set)的
- 支持一个 “加一" 操作，从小地址逐渐往上加，但是要能满足易读地址的设计
"





