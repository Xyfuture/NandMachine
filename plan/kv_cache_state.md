帮我实现一个用于计算 KV Cache 的所占的空间和 nand page 数量的函数，放置到 @nandmachine/frontend/utlis.py 下面，核心的功能描述如以下:
- 输入配置
    - nandmachine/config/config.py 中提供的 nand 的基本信息
    - nandmachine/config/model_config.py 中提供模型的基本信息
    - nandmachine/config/inference_config.py 中提供这次运行的batch size, prefill/decode length 等信息
- 输出内容：
    - 要输出一个 nandmachine/config/cache_state.py 中的 KVCacheState 信息
        - 每个 layer 总计的 KV cache 大小 (聚合所有的 Batch 和 Sequence length)
        - 存储上述这些 kv cache 一共需要多少的 nand page
        - 存储上述这些 kv cache 一共需要多少的 hyper page
            - hyper page 就是 num_planes 个 nand page 的总大小
- 要求
    - 要能正确的实现 输出 的内容
    - 正确区分 MHA，GQA，MLA(暂时不需要) 的 KV Cache 计算逻辑
    - 能够从inference config 中拿到 kv cache 的精度信息，做出对应的操作。 




