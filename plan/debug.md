请你单独写一个 ipynb file ，仿照 llama_pipeline.ipynb ，但是 code gen pass 跑完之后我们要遍历每个 node， 拿到 node 的 macro op list 而不是全局的，针对每个 node 的 macro op list，我们单独跑模拟器，输出 node 的信息， macro op 的信息 和 模拟的时间
这个要能支持配置不同的 tp size 

- 使用 logging 在 xpu.py 中添加输出 -- 估计时间的部分提供 输出 


参考 perfettotracer，利用这个库，在 nandmachine/simulator/hardware/xpu.py 中的 三个 engine ，每个都抓一下 trace 的信息，记录 指令的信息，需要你更改一下：
- xpu 中记录一下 Tracer，然后传到 engine 中。记录一下是否启用，如果不启用，就传一个 none，不用记录了
- xpu 中添加一个 save trace file 的功能， 支持自定义名字 
- 每个 macro op 的名字要尽可能记录多的信息
    - op id
    - 各种有关维度的信息也要记录一下 
你有什么拿不准的问题找我讨论一下 






你看一下，现在的代码代码中有关于 FlashAttention 的逻辑，包括生成的 macro op 和 针对 macro op 模拟的 模拟器相关的逻辑，我现在想引入对 deepseek v3 中的 mla 的支持，实现一个用于 decode phase的 FlashMLA 的模拟，需要配置的东西，首先生成专门的 macro op和针对该 macro op 的模拟逻辑，包括类似 nandmachine/kernels/attention.py 的 MLA code gen kernel 。然后参考 nandmachine/frontend/network/qwen3_moe.py 和 model_cards/deepseek-v3.json 中的逻辑和格式，构建一个 deepseek v3 的 model 类，需要仔细思考有关 mla 部分的实现，要正确处理 macro op 的生成 。 你仔细思考一下，如果有问题的话，及时问我。 大致需要改动的功能:
- 能够类似 nandmachine/frontend/network/qwen3_moe.py 构建出 支持 dp/ep 的 model 基本架构，能够跑通 normalize pass 和 macro op code gen pass
- 能够类似 nandmachine/kernels/attention.py 中提供 macro op 转换的功能
    - 需要提前定义 FlashMLA 的 macro op
- 能够类似 nandmachine/simulator/software/flash_attention.py 支持对 flashmla 的仿真
- 编写一个类似 frontend_pipeline_moe.ipynb 的测试 pipeline 
- 你考虑一下 build kv cache 部分是否需要同步的改动
- MLA 的计算方式要采用 absorb 之后的形式
- 拿不准的地方要和我讨论



支持消融实验

请你仿照 nandmachine/simulator/hardware/xpu.py 中的内容，新建一个类，实现没有 prefetch 下的延迟仿真，具体要求如下:
- Prefetch Engine 要去掉，prefetch 内部直接通过指令，不需要仿真延迟 
- Compute Engine 对于 gemm ， Attention 相关的指令，都加上一个 nand config 中 tR 的延迟。
- 一个全新的 xpu类，可以继承自原有的 class，但是支持独立的初始化操作 
- 拿不准的地方要找我讨论 




可以添加一个消灭通信延迟的版本 



支持 kv cache layout 的仿真 

请你参考 nandmachine/frontend/utlis.py 中的 build_kv_cache_state，新加一个 构建不均衡 kv cache state 的函数，我的要求如下
- 首先使用普通的 build_kv_cache_state 的函数，统计出 每个 layer 有多少个 KV block， 视作 balls，然后根据 nand config 中单个 plane 的大小，kv block size 的大小，和 plane 的数量计算出，有多少个 bins。
- 按照前面的模型进行 balls into bins 仿真， 假设每个 balls 投到 bins 中的概率是相同的，然后找出 load 最大的 bin 中多少个 ball， 用这个值做为 kv cache state 中的 num hyper pages 的参数。




给出一个写入量统计功能 

在 kv cache state 中加上 写入量的统计，然后根据 write 的速度直接算出来 总共写入需要多长的时间




需要评估 offloading 机制带来的开销 
应该是一个专门的实验来验证这个的开销，仿真一段时间内的请求情况，然后评估 整体的写入量， 然后计算这个时间的消耗
假设一个分布 ， 暂时先不处理erase的情况，一天只有 50 多次 erase 操作  



简化思路:

关于 xpu.py 中 class xPU 的签名
我们只关心 HBF 和 HBM 的带宽，不关心其他的内容 
- HBF 的带宽，通过 nand config 可以通过 num_planes * num_channels * (page_size / tR) -- 要注意单位的问题，统一为 bytes/ns 或者和 llmcompass 中对应的带宽单位
- HBM 的带宽，通过一个单独的参数传进来 -- 直接标明是多少 GB/s 就 ok 了
- hbf_sram_intermediate_buffer / memory_architecture  之类的参数就不要了， 不需要 hbm only 的东西



nand 部分添加一个 feature ，等待容量空闲的特性 




你看一下 nandmachine/frontend/utlis.py 中的关于 build kv cache 的逻辑，进行重写，只计算整体的 batch 情况，不区分并行的事情，并行相关的处理已经放到后面进行了，不要在因为 TP 之类的拆分 kv cache 了




仿照 dense_node_pipeline.ipynb ， 写一个针对 deepseek v3 测试的 ipynb ，输出每个 node 的信息，同时输出 trace 文件



mla 中有一个问题
bmm 中的不同的 matmul 实际上是共享 一个 input 的 / 需要修正这个 bug 





测试消融实验

你看一下 nandmachine/simulator/hardware/vallina_xpu.py 中现在的实现， 这个的主要目的是 bypass prefetch 的功能。现在这个是可以跑的吗



我现在希望进行一个消融实验，在同一个 config 下, 跑 5 个情况
- 完全没有任何优化的情况
    - 使用 vallina_xpu 运行
    - 使用 build_imbalanced_kv_cache_state 构建 kv cache 状态

