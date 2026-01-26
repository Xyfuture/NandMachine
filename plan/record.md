我要实现一个 针对 torch fx graph 的 pass, 这个 pass 主要用来记录 node 中的各类信息
目前是针对 @nandmachine/frontend/network/qwen3.py 捕获的 pass ，其中有一个 Linear，Attention 类，重点是捕获这些类里面的一些参数加入到 node meta 中

我有一个大致的捕获信息
- 记录 输入/输出的 shape --> 多个输入/输出情况的特判 --> 我会在运行这个 pass 之前运行一下 FakeProp 这个 pass 来捕获一些输入输出 shape 信息， 但是我不知道 FakeProp 是怎么处理输入输出的特判的 
- module_type -> 从 graph module 中获取，最后记录下来
- linear_info 针对 linear 类型单独的记录
    - weight_shape 
    - require_all_reduce ? --> 有无更好的方式记录这个信息
- attention_info 针对 attention 类型单独的记录
    - pass -> 先不记录了， 后面想好了再重新记录
- nand_store_pages -> 记录这个 node 需要存储多少 nand pages, 不需要存储数据的可以跳过
    - 目前对于 linear 需要进行存储， 按照 fp16 计算出 weight 的 size， 然后按照一个 page 是 4k 页的情况，计算需要多少页

相关的代码实现在 @nandmachine/frontend/core/passes/recorder.py 中



