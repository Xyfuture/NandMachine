


前期准备
- 制定 nand 读取的指令
- 设计细节的操作方式


模拟方式
- 延迟
- 带宽 

需要设置 每个 channel 的速度 以及排队的情况
怎么设置完成情况呢？ 


我要设计一个 Nand 时序模拟器, 我会给出你一些背景和设计方案，你帮我完成代码的编写
## 代码参考
@nandmachine\simulator\hardware\nand.py 这里面放置的是 Nand 仿真的核心逻辑
@nandmachine\simulator\runtime\addr.py 这里面放置的是 nand 地址相关的操作 
@nandmachine\config\config.py 这里放置了一些nand 参数的配置 


## Nand 的组织结构
在这个模拟器中 nand 是按照层次结构来组织的, 从大的角度上看, 从上到下依次分别是 channel->plane->block->page, 每个plane共享一个读写电路, 因此读取操作每次只能从一个plane中读取一个 Page, 每次读取的时间是 tRead, 不同的Plane是可以同时读取不同Page的. 
这个层次中的 Block 在读取操作中没有意义, 其是针对写入操作进行的,  暂时不需要考虑写入操作.

我们将nand参照 hbm 的方式组织结构, 我们拥有一个 base die 在底部, 上层多个 nand die, 在这种组织架构中, 我们有两个数据通路, 一个是 nand->base die ,另一个 nand -> xpu(GPU or TPU 就是核心的运算单元). 对于这两个通路, 我们也要简单仿真一下, 资源占用情况, 有时候指令是要求数据从 nand->base die , 有时候是要求数据从 base die -> xpu.

我们在 base die 中还放置了 sram 用于缓存一些数据, 这些数据会走  base die -> xpu 的通路

我在 @nandmachine\commands\micro.py 中定义了一下 nand 和 sram 相关操作需要的基本指令。指令分为 MemoryBasicOpBase 和  DataForward，分别描述基本的操作和取出来的数据需要走的通路。这两个按照顺序组合到MemoryOperation中，表示一条完整的指令， 然后一个request 中可能不只一个 MemoryOperation, 因此是一个组合。 

## 模拟方式
我现在要通过最简单的仿真方式, 假设不会出现指令抢占的情况, 每个资源都可以维护一个自己的free time 表示在那个时刻以后是空闲的, 然后可以直接给出一条指令到来后的完成时间.
我需要模拟的资源有
- plane 读出 page 的控制
- nand-> base die 通路
- base die -> xpu 通路

在模拟器中 NandSimCore 是需要和 NandController 配合的，NandController 向 NandSimCore发出 Request， 然后NandSimCore 负责返回这个 Request 所有的 Operation 都完成之后的时间。



## 实现相关的要求 
代码实现在  @nandmachine\simulator\hardware\nand.py 中 填充 class NandSimCore 相关内容
- 实现的每个函数都要与我讨论, 不要实现没用的代码








# misc not for agent 

Nand Controller + Nand die 模拟

首先实现为 非抢占式，然后实现为抢占式, 双方的接口定义好
在指令中预埋同步操作 or 放一个回调函数

Prefetch Engine + Compute Engine -> Nand Controller -> Nand Sim Core




我现在要实现一个NandController 和 NandSimCore 类，用于实现 Nand 访问的仿真，
NandController 类来自 





NandController 接口
- request 是 一堆 page 级别的访问治理， 但是一次一个组 level --> 需要定义新的接口
- 
