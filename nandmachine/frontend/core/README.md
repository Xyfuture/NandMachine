在这里面写 graph 相关的逻辑？ 

- graph 
  - 写图相关的逻辑
- kernels 
  - 留存关键的 macro op 的 kernel 
- mapper
  - 写几个 pass， 用于生成 nand 上的最优 mapping 



最终给到 runtime 的东西是一个 packle 的序列化的东西

- 基于 torch fx graph 的图结构
- nand file table --> 由 mapper 生成的映射表
- kernels，直接把 可执行的 kernels 送进去， kernels 不执行运算，而是输出指令 -- 这里有讨论的空间
  - 可以 借用 python 的 runtime
  - 也可以 在 runtime module 中 手动解析
