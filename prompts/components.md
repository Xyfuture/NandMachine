需要支持 read - write 次数的追踪 


功耗的数据的统计 


mapper 的支持   



指令的支持 


现在的文件还是随便分的，之后需要重新改进




代码设计
- nand 和 sram 要同时进行设计
- dram 部分暂时先用 延迟和带宽反映一下， 后续更新为 full support 
- 需要提供原生的 table 控制 --> 暂时启动名为 manager 
- runtime 下面放那几个函数，支持分配的操作. 





实现一下 NandFileTable，其实一个接近 文件系统的东西， 主要记录 一个文件 id 对应到具体 nand page id， 最后还有记录一个 offset， 表示最后一个 page 的 offset
其要支持一个功能：
- 支持向某一个文件，写入到某一个特定的地址





如何支持软件的操作 ？ 
- weigh 的声明部分
- 接入 nano vllm 在上层进行调度 --> 大后期的工作
- kv cache 的声明部分
- kernel 应该怎么写 ？ 
    - 支持细粒度的访问操作 
- 怎么支持 EP 
    - 




先生成一波指令，然后模拟器执行具体的指令

地址部分的生成与分配靠提前的一波 pass

模拟器跑起来，需要两个东西：
- 指令，生成的指令文件
- 映射关系，由编译器生成，可以 save 到内部
- 注意，只需要跑单层的，多层的重复就好了，--> 资源需要按照层缩减


怎么接入 runtime 的支持呢？
- 部分指令类似于 系统调用， 用于提供 runtime 支持
- 还要主动发出 runtime 调用相关的指令，这个应该是框架层干的活，应该支持外面也对模拟器中发射指令
- 是否考虑 runtime 和 hardware 合二为一， 因为 runtime 操作也是有延迟的



# TODO
- [ ] Hook 一下 dist 的一些方法，实现并行操作
- [ ] 实现 meta kernel 级别的操作 
- [ ] 转换出需要的 计算图





我现在想通过 monkey patch 的方式实现对 torch.distribute 主要方法的 hook，实现在只有一个 GPU 的情况下，将代码跑起来，不需要让结果正确，只需要让代码能够正常运作
主要 hook 如下的函数
- get_rank
- get_world_size
- all_reduce

要设定一个参数 表示当前有多少个 GPU (WorldSize), 当前 rank 总保持为 0 就好了
对于 all reduce这个函数，直接返回一个 zeros 或者 ones 就可以了，此外要额外支持 meta kernel 能够正常返回维度的信息。
相关代码实现到 hook_dist.py 中, @nandmachine\frontend\network\qwen3.py 中关于 torch dist 调用注释一下, 换成 fake 的 调用 
