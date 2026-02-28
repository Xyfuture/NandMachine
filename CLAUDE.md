请使用中文进行与我的交互，但是代码中的注释和其他内容请使用英文.
我是 molly，每次你和我对话，都要说 molly 加油! 

这个项目旨在编写一套模拟器，实现对 high Bandwidth flash 在 llm 上的运行进行模拟。

模拟器分为几个部分，分别由前端，kernels 和后端组成.

nandmachine/ 下有多个文件夹，我简单描述一下其对应的功能：
- commands 中存储了一些 op ，其中 macro op 是比较高层次的 op，用于输入到硬件模拟器中， micro op 则是细粒度的 op， 用于模拟器内部的多个组件之间的协作，硬件模拟器会在内部解析 macro op，然后转换为 micro op，用于细节的模拟
- frontedn 负责 torch代码，计算图，以及一些 计算图 pass 相关的逻辑
- kernels 复杂将 计算图中的一些节点生成 最终的macro op，这一部分由手写完成相关工作
- config 用于存储软硬件的一些配置信息
- simulator 模拟器的部分，分为 hardware 和 runtime，hardware中描述了各个硬件的行为，runtime 主要用来管理 页表，hardware 每次进行访问的时候，需要根据页表的值来获取到实际的地址，因此需要 runtime 来参与管理


- 前端 : 位于 nandmachine/frontend 目录下
    - 主要负责计算图的转换工作
    - 用户编写一个基于 torch 的程序，然后经过 torch.fx.graph 功能转换为计算图，然后交由一个 nanovllm 对计算图进行调度(这一部分的代码还没有实现完全，实现结束后是类似 vllm 的东西，每个执行一次 graph 就是一个 llm inference 的 step， 在 graph 执行前会配置一些参数，用户 kv cache 和 batch
    的管理等等)
    - 目前实现的代码仅支持构建一个 graph，然后执行一行，不涉及 continuous batching 的内容
    - 我要在 front end 中实现一系列 pass ，实现一些功能
        - mapping pass: 用于实现 weight 的 mapping 过程
            - 输入: 
            - 输出:
            - 描述:
        - code gen pass: 这个理论是是一系列 pass，用于最终生成 commands 中干的 macro op 用于 hardware sim 的执行，目前还没有完全实现好，还在筹划阶段。
            - 输入：
            - 输出：
            - 描述：
        - 


- 硬件模拟器：位于 nandmachine/simulator 目录下
    - 主要负责输入 marco op 和 当前文件系统，然后仿真得出 macro op 的执行时间
    - 我要实现一个 资源管理机制， 这个机制主要基于 page table 进行， 因此我实现了一套 runtime 用于维护 runtime 的正确性
    - 模拟相关的核心逻辑放置到了 hardware 目录下， 其中有一个 xpu 作为 toplevel， 其内部有多个异步并行的 engine，通过指令之间的依赖实现同步。NandController 作为接口，用于访问 nand ，NandSimCore 则提供 nand 操作延迟的内在仿真功能，基于Desim 进行开发。
    - micro op 是硬件内部访问 nand 操作核心的 op，多个 micro op 组成一个 memory operation ， 然后多个 operation 能够组成一个 nand request， 用于向 nand controller 发起请求 

    