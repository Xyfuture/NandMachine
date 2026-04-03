请你单独写一个 ipynb file ，仿照 llama_pipeline.ipynb ，但是 code gen pass 跑完之后我们要遍历每个 node， 拿到 node 的 macro op list 而不是全局的，针对每个 node 的 macro op list，我们单独跑模拟器，输出 node 的信息， macro op 的信息 和 模拟的时间
这个要能支持配置不同的 tp size 





你看一下，现在的代码代码中有关于 FlashAttention 的逻辑，包括生成的 macro op 和 针对 macro op 模拟的 模拟器相关的逻辑，我现在想引入对 deepseek v3 中的 mla 的支持，实现一个用于 decode phase的 FlashMLA 的模拟，需要配置的东西，首先生成专门的 macro op和针对该 macro op 的模拟逻辑，包括类似 nandmachine/kernels/attention.py 的 MLA code gen kernel 。然后参考 nandmachine/frontend/network/qwen3_moe.py 和 model_cards/deepseek-v3.json 中的逻辑和格式，构建一个 deepseek v3 的 model 类，需要仔细思考有关 mla 部分的实现，要正确处理 macro op 的生成 。 你仔细思考一下，如果有问题的话，及时问我。




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




