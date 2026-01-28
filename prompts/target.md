v 0.0.5
- 跑通转换的流程
- 跑通第一个 ffn 网络
- 支持 nand file open 系列的操作
- hw 部分直接用 roofline 模型，但是 nand 部分用比较精确的访问时间
- sram 不考虑 channel 的分配
- dram 部分直接不考虑分析的事情
- nand 的部分不过度区分 channel， 甚至认为是一个大 channel？
- nand 采用 free time 的形式，不支持抢占模式



v 0.1



v0.2