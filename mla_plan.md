# DeepSeek V3 FlashMLA Decode 完整方案

## Summary
目标是在现有 NandMachine 架构内，完整接入 DeepSeek V3 的 MLA decode 模拟链路，使其能够像当前 `Qwen3MoE` 一样完成：

- 构建支持 `attn_dp/attn_tp/ffn_ep/ffn_tp` 的 decoder layer 基本架构
- 跑通 `NxTracer -> NormalizePass -> CodeGenPass`
- 生成符合 DeepSeek V3 MLA absorb 设计的 macro op 序列
- 对生成出的 `FlashMLA` macro op 做 cycle/time 仿真
- 提供一个可复现的 notebook pipeline

本方案固定采用这些决策：
- 只做 `decode`，不做 prefill
- 采用 MLA 的真实压缩 KV cache
- 采用 `Design B`：`absorb matmul + FlashMLA core + up_proj matmul`
- 不做兼容逻辑，不做兜底，遇到未支持路径直接报错
- 不实现 MTP
- 不实现 shared experts
- attention 必须支持 `attn_dp_size > 1`

---

## MLA 算子与数学方案

### 1. DeepSeek V3 MLA decode 的计算分解
对单层 attention，在 decode phase 下固定拆成三段：

1. `Absorb`
- 从 query 的 `nope` 分量构造 absorbed query：
- `q_absorb = q_nope @ W_UK^T`
- 输出留在 latent 空间，维度是 `kv_lora_rank`

2. `FlashMLA Core`
- 读取历史 cache：
  - `c_kv_cache`
  - `k_rope_cache`
- 分别计算：
  - `score_latent = q_absorb @ c_kv_cache^T`
  - `score_rope = q_rope @ k_rope_cache^T`
- 两项相加后做 online softmax
- 再执行：
  - `o_latent = softmax(score) @ c_kv_cache`

3. `Output Up-Projection`
- 将 latent 输出恢复到 value 空间：
- `o = o_latent @ W_UV`

### 2. 运算分类
普通 `matmul`：
- `q_absorb = q_nope @ W_UK^T`
- `o = o_latent @ W_UV`
- query / kv projection 侧的普通线性层

`batched matmul`：
- `q_absorb @ c_kv_cache^T`
- `q_rope @ k_rope_cache^T`
- `softmax(score) @ c_kv_cache`

非 matmul：
- RoPE
- score 累加
- online softmax
- running max / running sum / running output 更新

### 3. Flash 化约束
`FlashMLA core` 必须严格按 block streaming 建模：
- 每次只处理一个 KV block
- 不回写完整 `score`
- 不回写完整 `prob`
- 不回写逐 token attention 中间结果
- 仅维护 online softmax 的状态和最终 `o_latent`

这条约束必须直接体现在 macro op 设计和 simulator 行为里，不能只作为注释说明。

---

## 配置与类型设计

### 1. 模型配置
在 `nandmachine/config/model_config.py` 新增：

- `DeepseekV3ModelConfig`

建议字段最少包含：

- `hidden_size`
- `num_attention_heads`
- `max_position_embeddings`
- `intermediate_size`
- `moe_intermediate_size`
- `num_hidden_layers`
- `num_experts_per_tok`
- `n_routed_experts`
- `first_k_dense_replace`
- `q_lora_rank`
- `kv_lora_rank`
- `qk_nope_head_dim`
- `qk_rope_head_dim`
- `v_head_dim`
- `rms_norm_eps`
- `attention_bias`
- `rope_theta`
- `n_shared_experts`
- `hidden_act`
- `num_nextn_predict_layers`
- `attention_type="mla"`

同时提供：
- `from_dict()`
- 对 `model_cards/deepseek-v3.json` 的严格字段校验
- 对不支持字段的直接报错规则：
  - `n_shared_experts != 0` 报错
  - `num_nextn_predict_layers != 1` 也报错或严格按卡片值校验
  - `hidden_act != "silu"` 报错

### 2. macro op 类型
在 `nandmachine/commands/macro.py` 新增：

- `FlashMLAOp(MacroOp)`

字段固定为：

- `qk_latent_bmm_shape: tuple[int, int, int, int]`
- `qk_rope_bmm_shape: tuple[int, int, int, int]`
- `sv_latent_bmm_shape: tuple[int, int, int, int]`
- `softmax_shape: tuple[int, int]`
- `weight_bits: int`

语义固定：
- `qk_latent_bmm_shape = (B, M, K, N)`
- `qk_rope_bmm_shape = (B, M, K, N)`
- `sv_latent_bmm_shape = (B, M, N, K)`
- `softmax_shape = (B * M, N)`

这里：
- `B = local_num_heads * num_kv_blocks`
- `M = local_batch_size`
- latent QK 的 `K = kv_lora_rank`
- rope QK 的 `K = qk_rope_head_dim`
- SV latent 的输出维 `K = kv_lora_rank`
- `N = kv_block_size_tokens`

`FlashMLAOp` 不包含 absorb 和 up-proj 形状，这两部分始终由普通 `MatMulOp` 表达。

---

## KV Cache 方案

### 1. cache 物理语义
MLA 的历史 cache 不再存展开后的 K/V，而是存：

- `c_kv_cache`: 每 token `kv_lora_rank`
- `k_rope_cache`: 每 token `qk_rope_head_dim`

因此每 token cache 大小固定为：

- `kv_lora_rank + qk_rope_head_dim`

乘上 bit precision 后得到 bytes。

### 2. KV cache 计算规则
在 `nandmachine/frontend/utlis.py` 中，把 `attention_type == "mla"` 从未实现改为正式实现。

固定公式：

- `per_token_kv_values = kv_lora_rank + qk_rope_head_dim`
- `per_token_kv_bytes = ceil(per_token_kv_values * kv_cache_bits / 8)`

总 cache 大小：
- `total_bytes = batch_size * peak_sequence_length * per_token_kv_bytes`

其中：
- `peak_sequence_length = input_sequence_length + output_sequence_length`

block token 数：
- `kv_block_size_tokens = ceil(kv_block_size_bytes / per_token_kv_bytes)`

block 数：
- `num_kv_blocks = ceil(total_bytes / kv_block_size_bytes)`，`total_bytes == 0` 时为 0

### 3. 并行切分规则
attention 的 local 视角固定为：

- `local_batch_size = global_batch_size / attn_dp_size`
- `local_num_heads = num_attention_heads / attn_tp_size`

若不能整除，直接报错。

注意：
- MLA cache 不按 head 复制存储，所以 `attn_tp` 不改变每 rank 持有的 token cache 总量推导逻辑
- `attn_dp` 会影响每 rank 的 local batch，对 codegen 的 local block 数有影响
- 全局 `KVCacheState` 仍保留全局视角
- MLA attention module 内部需要从全局 state 推出 local block 视角，不能直接照搬 GQA 的 `num_kv_heads/head_dim` 公式

### 4. 容量估算同步修改
在 `nandmachine/frontend/validator.py` 中：
- `_resolve_attention_layout()` 对 `mla` 改成正式实现
- `_resolve_weight_spec()` 对 `mla` 改成正式实现
- Qwen3MoE 专用 validator 不复用 DeepSeekV3；新增 DeepSeekV3 的容量估算路径

MLA attention 权重估算至少要包含：
- q projection low-rank/down + up
- kv projection down
- absorb 使用的 `W_UK`
- output up-proj 使用的 `W_UV`
- `o_proj`
- norm 参数

如果某部分在当前 frontend 模型里不建真实线性层，也必须在 validator 中用一致的参数口径估算，不允许口径漂移。

---

## Frontend 模型方案

### 1. 模块层设计
在 `nandmachine/frontend/modules/modules.py` 中新增：

- `MLAAttention(HookModuleBase)`

它只负责 MLA attention 对应的 macro op codegen，不负责完整 query/kv/o_proj 模块组织。

`MLAAttention` 构造参数固定包含：

- `num_heads`
- `q_lora_rank`
- `kv_lora_rank`
- `qk_nope_head_dim`
- `qk_rope_head_dim`
- `v_head_dim`
- `tp_size`
- `dp_size`

内部保存：
- `local_num_heads = num_heads / tp_size`

### 2. MLAAttention 的 codegen 输出
`macro_code_gen()` 必须固定生成三段：

1. absorb `MatMulOp`
- `dim = (local_batch_size * local_num_heads, qk_nope_head_dim, kv_lora_rank)`

2. `FlashMLAOp`
- `qk_latent_bmm_shape = (local_num_heads * local_num_kv_blocks, local_batch_size, kv_lora_rank, kv_block_size_tokens)`
- `qk_rope_bmm_shape = (local_num_heads * local_num_kv_blocks, local_batch_size, qk_rope_head_dim, kv_block_size_tokens)`
- `sv_latent_bmm_shape = (local_num_heads * local_num_kv_blocks, local_batch_size, kv_block_size_tokens, kv_lora_rank)`
- `softmax_shape = (local_num_heads * local_num_kv_blocks * local_batch_size, kv_block_size_tokens)`

3. up-proj `MatMulOp`
- `dim = (local_batch_size * local_num_heads, kv_lora_rank, v_head_dim)`

依赖关系：
- `FlashMLAOp.with_inputs(absorb_matmul or prefetch)`
- `up_proj_matmul.with_inputs(flashmla_core or release)`

### 3. local KV block 规则
`MLAAttention` 不能再调用 `build_gqa_kernel_param()`，而是必须新增专门方法，例如：

- `build_mla_kernel_param(graph_meta)`

规则：
- `local_batch_size = global_batch_size / attn_dp_size`
- `local_total_kv_bytes = local_batch_size * peak_sequence_length * per_token_kv_bytes`
- `local_num_kv_blocks = ceil(local_total_kv_bytes / kv_block_size_bytes)`

这里 local block 数必须基于 local batch 推导，不能沿用全局 block 数。

### 4. 网络层结构
新增 `nandmachine/frontend/network/deepseek_v3.py`：

- `DeepseekV3Attention`
- `DeepseekV3MLP`
- `DeepseekV3MoE`
- `DeepseekV3DecoderLayer`

设计原则：
- 整体结构参考 `qwen3_moe.py`
- attention 子系统替换为 MLA 版本
- MLP / MoE 继续复用现有 `FusedMoE`、`MergedColumnParallelLinear`、`RowParallelLinear`、`RMSNorm`、`SiluAndMul` 能复用的部分
- dense / moe 的层选择由 `first_k_dense_replace` 控制

`DeepseekV3Attention` 推荐结构：
- query low-rank path 的线性层
- q norm
- q up-proj，产出 `q_nope + q_rope`
- kv down-proj，产出 cache 语义上的 latent/rope 分量
- kv norm
- `MLAAttention` hook module
- `o_proj`

前向只需要满足：
- meta device 可 trace
- shape 合法
- 能让 normalize pass 保留关键 hook module
- 不要求真实数值

### 5. MoE 路径
DeepSeek V3 decoder 中：
- 前 `first_k_dense_replace` 层使用 dense MLP
- 后续层使用 routed MoE

实现方式：
- 在 `DeepseekV3DecoderLayer(layer_idx, config, parallel_config)` 中根据 `layer_idx` 决定使用：
  - `DeepseekV3MLP`
  - 或 `FusedMoE`

对不支持项直接报错：
- `n_shared_experts != 0`
- 其他需要 shared expert 的路径

### 6. NormalizePass 兼容要求
`NormalizePass` 当前只保留 `HookModuleBase` 子模块。只要 `MLAAttention`、新增 norm/投影模块继承策略与现有 hook module 一致，就不需要改 pass 逻辑。

唯一要求：
- 必须保证最终 FX graph 中 attention 核心仍是 `call_module` 节点，而不是被展开成普通算子。

---

## Kernel 方案

### 1. 新增 MLA kernel
在 `nandmachine/kernels/attention.py` 中新增：

- `MLANandKernel`
- `MLAHBMKernel`

其职责仅是生成 `FlashMLAOp(core)`。

### 2. HBM kernel
`MLAHBMKernel.lowering()`：
- 校验所有维度正数
- 返回单个 `FlashMLAOp`

### 3. NAND kernel
`MLANandKernel.lowering()`：
- 依据 `kv_block_size_bytes` 和 `nand_config` 推导 hyper page / prefetch 数
- 每个 hyper page 生成：
  - `SramPrefetch`
  - `FlashMLAOp`
  - `SramPrefetchRelease`

注意：
- 这里只包裹 core/
- absorb / up-proj 不放进 prefetch/release 包裹内

### 4. 建议的 frontend 组装方式
`MLAAttention.macro_code_gen()`：
- 先生成 absorb `MatMulOp`
- 调用 MLA kernel 生成 FlashMLA core macro op 列表
- 给 kernel 返回的第一个 op 加上 absorb 依赖
- 再生成 up-proj `MatMulOp`
- up-proj 依赖 kernel 返回链路的最后一个有效 op

这样可以保留现有 macro op DAG 组织方式，不需要改 `CodeGenPass`。

---

## Simulator 方案

### 1. software simulator
在 `nandmachine/simulator/software/flash_attention.py` 中新增 MLA 专用 simulation 逻辑，建议新增：

- `FlashMLA_BatchedMatMul_Simulation`

它的职责只覆盖 FlashMLA core：

- latent QK
- rope QK
- online softmax
- latent SV

不包含：
- absorb
- up-proj

这两部分继续使用现有 `MatMul_Simulation`。

### 2. 仿真时间构成
`FlashMLAOp` 的总时间固定建模为：

- `latent_qk_time_ns`
- `rope_qk_time_ns`
- `softmax_time_ns`
- `latent_sv_time_ns`

总和得到 `flashmla_core_time_ns`

### 3. 内存访问建模
与 FlashAttention 一样，FlashMLA core 的 IO 要按 block streaming 口径处理：

对每个 block：
- 读取 `c_kv_block`
- 读取 `k_rope_block`
- 不回写 `score/prob`
- 仅在最终阶段输出 `o_latent` 到下游 up-proj

因此 simulator 中不能把 FlashMLA 退化成：
- 先完整 QK
- 再完整写回 score
- 再读回做 softmax
- 再完整写回 prob

如果这么建模，就与 flash 设计冲突，必须禁止。

### 4. xPU 接入
在 `nandmachine/simulator/hardware/xpu.py` 中：
- `import FlashMLAOp`
- `_format_macro_op_trace_name()` 新增 `FlashMLAOp`
- `ComputeEngine` 增加：
  - `_validate_flashmla_shapes()`
  - `execute_macro_op(FlashMLAOp)` 分支

执行规则：
- `MatMulOp(absorb)` 走普通 matmul
- `FlashMLAOp` 走 MLA core simulation
- `MatMulOp(up_proj)` 走普通 matmul

`load_command()` 不需要改调度结构，只要新 op 被视为 compute op 即可。

---

## Notebook 方案

新增一个新 notebook，建议命名为：

- `frontend_pipeline_deepseek_v3_mla.ipynb`

内容固定包含：

1. 加载 `model_cards/deepseek-v3.json`
2. 构造 `DeepseekV3ModelConfig`
3. 构造 `NandConfig`
4. 构造 `MoEParallelConfig`
5. 构造 `InferenceConfig`
6. 调用 `build_kv_cache_state()`
7. 构造单层 `DeepseekV3DecoderLayer`
8. `NxTracer().trace(model)`
9. `NormalizePass`
10. 写入 `NxGraphMeta`
11. `CodeGenPass`
12. 打印：
- FX graph node summary
- macro op summary
- attention 相关 op 明细
- 明确看到：
  - `MatMulOp(absorb)`
  - `FlashMLAOp`
  - `MatMulOp(up_proj)`

这个 notebook 默认只生成和检查 macro op，不强制跑 cycle 仿真；如需再加一个 cell 跑 `run_macro_ops()`，可以加，但不是必须。

---

## 测试方案

### 1. model config
新增 `archive_test/test_frontend_deepseek_v3.py` 或等价测试文件，覆盖：

- `DeepseekV3ModelConfig.from_dict()` 正确读取 `deepseek-v3.json`
- `attention_type == "mla"`
- `q_lora_rank/kv_lora_rank/qk_nope_head_dim/qk_rope_head_dim/v_head_dim` 正确
- `n_shared_experts != 0` 报错
- 不支持字段组合时报错

### 2. KV cache
新增测试覆盖：

- MLA per-token cache bytes 正确
- block token 数正确
- `num_kv_blocks` 正确
- `attn_dp_size > 1` 的 local batch 影响正确
- 非整除 batch 直接报错
- 不再对 `mla` 抛 `NotImplementedError`

### 3. frontend network
覆盖：

- `DeepseekV3Attention` 在 meta device 上保持 hidden shape
- `DeepseekV3DecoderLayer` 在 dense 层和 moe 层都能保持 hidden shape
- `layer_idx < first_k_dense_replace` 选 dense
- `layer_idx >= first_k_dense_replace` 选 MoE

### 4. codegen
覆盖：

- `NormalizePass` 保留 `MLAAttention`
- `CodeGenPass` 能生成完整 macro op 列表
- attention 子序列顺序正确：
  - absorb `MatMulOp`
  - `FlashMLAOp`
  - up-proj `MatMulOp`
- `attn_tp_size > 1` 时 local head 维度正确
- `attn_dp_size > 1` 时 local batch / local blocks 正确

### 5. kernel
覆盖：

- `MLAHBMKernel.lowering()` 只返回一个 `FlashMLAOp`
- `MLANandKernel.lowering()` 返回 prefetch/core/release
- `FlashMLAOp` 非法 shape 校验会报错

### 6. simulator
覆盖：

- `ComputeEngine.execute_macro_op(FlashMLAOp)` 会调用 MLA simulation 路径
- `run_macro_ops()` 能运行含 `MatMulOp + FlashMLAOp + MatMulOp` 的序列
- trace name 格式包含 FlashMLA 形状信息

---

## 文件改动范围

主要改动这些位置：

- `nandmachine/commands/macro.py`
- `nandmachine/config/model_config.py`
- `nandmachine/frontend/modules/modules.py`
- `nandmachine/frontend/network/deepseek_v3.py`
- `nandmachine/kernels/attention.py`
- `nandmachine/frontend/utlis.py`
- `nandmachine/frontend/validator.py`
- `nandmachine/simulator/software/flash_attention.py`
- `nandmachine/simulator/hardware/xpu.py`
- `model_cards/deepseek-v3.json` 不改内容，只作为读取输入
- `frontend_pipeline_deepseek_v3_mla.ipynb`
- 对应 `archive_test/` 下新增测试

---

## Public APIs / Interfaces
本次会新增的公开接口：

- `nandmachine.commands.macro.FlashMLAOp`
- `nandmachine.config.model_config.DeepseekV3ModelConfig`
- `nandmachine.frontend.modules.modules.MLAAttention`
- `nandmachine.frontend.network.deepseek_v3.DeepseekV3Attention`
- `nandmachine.frontend.network.deepseek_v3.DeepseekV3DecoderLayer`
- `nandmachine.kernels.attention.MLAHBMKernel`
- `nandmachine.kernels.attention.MLANandKernel`

本次会改变行为的已有接口：

- `build_kv_cache_state()`
- `calculate_kv_cache_state()`
- `validator` 中的 MLA 分支
- xPU compute dispatch 对 compute macro op 类型的支持集合

---

## Explicit Assumptions
- 只做 decode，不做 prefill。
- `FlashMLAOp` 只表示 core streaming attention。
- absorb 与 up-proj 固定拆成独立 `MatMulOp`。
- MLA cache 固定按 `kv_lora_rank + qk_rope_head_dim` 建模。
- `attn_dp_size` 必须能整除 batch，否则直接报错。
- 不实现 MTP，不实现 shared experts，不实现 shared expert fallback。
- 不保留 GQA 路径对 MLA 的复用入口，MLA 必须走独立 attention codegen 逻辑。
