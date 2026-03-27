近期工作目标
- 模拟器
    - 实现 llama 405b 的 model 
    - 实现 dense model 的 tp
        - all to all 通信
        - all reduce 通信
    - 实现 moe model 
        - 直接把 dp 和 tp 一块做了
    - 实现 mla 的 Attention kernel 和 model layer
    - 写入开销统计
    - 不均情况统计
- Paper
    - weight in HBF 部分删除 算法图
    - 完善 system 部分
        - file system -- 急需完成
        - weight / kv 的接口
        - 图 x 3 
    - 完善 data layout 部分
        - 先跳过 weight 部分
        - 图 x 2
            - kv cache 分布图
        - 3.26 down 掉
    - 




# MISC
- 支持 tp 的功能 和 ep 和 dp 功能 -- qwen3
- 代码仓库的重整， 部分 qwen3 moe 的代码 挪到 modules 中去 