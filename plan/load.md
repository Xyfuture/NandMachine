你帮我探究一个问题，首先参考 nano vllm 的实现，抽象出来一个 block 分配机制的模拟，包括paged Attention 那一套策略和替换机制。
然后模拟出大量的请求，假设为 1w 个，每个请求的 context length 都是从 gemma 分布中采样的一个值，prefill 长度是从 mean 为 1k 的 gemma 分布中采样的，而 decode 长度则是从 mean 为 512 的 gemma 分布中采样的。
一个token 的 KV cache 尺寸是 4 KB (所有的数据都是依据单层来处理的，我们不考虑层数的影响)
用于存储 KV Cache 的内存容量是 16 GB
每个 inference step 的最大 batch size 为 100，
这样让 Request 进进出出，kv cache 在 block pool 中替换多次后，我们探究 KV Cache 的分布情况
KV Block size 设置为 16。
存储 KV Cache 的内存一共有 1024 个 Bank，每个 Bank 的 Page Size 是 64 KB。
输出一下最后一个 step 中，其拥有多少 batch size，这些 Batch Size 对应的 KV Cache Block 在 Bank 上的分布是什么样子的，输出这些 Batch 所有的 KV cache 平均每个Bank上应该存储多少个 Page，和实际情况下， 拥有 page 数最多的 Bank 存储了多少的 Page。

