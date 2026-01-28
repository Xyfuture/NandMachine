


前期准备
- 制定 nand 读取的指令
- 设计细节的操作方式


模拟方式
- 延迟
- 带宽 

需要设置 每个 channel 的速度 以及排队的情况
怎么设置完成情况呢？ 


我要设计一个 Nand 时序模拟器，




Nand Controller + Nand die 模拟

首先实现为 非抢占式，然后实现为抢占式, 双方的接口定义好
在指令中预埋同步操作 or 放一个回调函数

Prefetch Engine + Compute Engine -> Nand Controller -> Nand Sim Core




我现在要实现一个NandController 和 NandSimCore 类，用于实现 Nand 访问的仿真，
NandController 类来自 





NandController 接口
- request 是 一堆 page 级别的访问治理， 但是一次一个组 level --> 需要定义新的接口
- 
