# Kimi-K3.1-Pro 双机 BF16 训推一致性验收

2026-11-28，93 层、56 个路由专家、每 token 选 16。两机各 8 张 Ascend 950DT。Actor / Reference 为 FSDP16、EP8、expert FSDP2，TP/CP/PP 均为 1。vLLM 有两个完整副本，每个副本 TP8、DP1、EP8。两侧每个 EP rank 各负责 7 个专家。

## 推理 BI 与 PD

| 固定什么、改变什么 | 原路径与首个差异 | 实际修改及验证 |
| --- | --- | --- |
| BI 固定目标请求，改变同批请求的数量及顺序 | TP 的 attention `o_proj` 按 K 维分片做局部乘法再规约。固定轨迹位置 1024 出现首差异。单独切换它后，下一处出现在位置 1025 的 dense MLP `down_proj` | 两处投影分别开启 `KIMI_ALIGNMENT_FULL_O_PROJ` 和 `KIMI_ALIGNMENT_DENSE_MLP`。在当前 TP 组收集输入和权重的完整 K 维，每张卡用一次 BF16 `F.linear` 算自己负责的 token 行，统一每行乘加顺序。相关 worker 的确定性等级为 3。重复、混批、调序和 batch16 四组各比较 64 个有效 token，全部相等 |
| PD 固定前缀和待评分 token，改变一次送入的 token 数及是否读取缓存 | KDA prefill 用 `run_chunk_kda` 按块展开，decode 用 `run_recurrent_kda` 逐 token 读写 state。同一递推式在 BF16 下可能因中间舍入顺序不同而给出不同值 | prefill 与 decode 共用 `canonical_kda`，它每个请求每个 token 调一次已有的 AscendC `recurrent_kda`。prefill 按请求取初始状态、结束时写回；decode 由 `serving_decode` 取真实 cache slot 的 FP32 state，再写回该 slot。直接 decode 与完整前缀、实际分成 511 + 434 的 prefill 与完整 prefill、长回答偏移 1023/1024/1025/2047 均相等 |

以上为组合验收。每项开关的独立效果以固定失败输入的单项复验为准。投影代码见 [full_k_sequence_projection](runtime/linear_alignment.py)，KDA 见 [canonical_kda 与 serving_decode](runtime/kda_alignment.py)。

## 训练前向怎样与推理对齐

| 位置 | 谁向谁对齐 |
| --- | --- |
| KDA | old 重算、带梯度 actor、推理 prefill 和 decode 都从各自输入调用共用的逐 token 实现。加载及 FSDP 分组保留 checkpoint 内的 FP32 门控参数 `A_log` 和 `dt_bias` |
| 最终 token logprob | old 和 actor 分别用自己前向产生的 logits、对应因果位置上的 label 调用推理侧 Ascend V2 的 `compute_token_logprobs`。rollout 直接复用开关关闭。见 [V2 打分与梯度](runtime/kimi_scoring.py) |
| 投影与中间舍入 | 推理侧向训练侧的完整 K 维投影、卷积输出转 BF16 后再做激活等计算顺序对齐。还要统一 DSA 的归一化及缓存布局、路由选专家和归一化所用的精度。调用同名算子仍须核对输入、规约顺序和 state |
| MoE 空专家 | 训练侧原先有任一本地专家未分到 token 就同时改变专家乘法和 EP 分发/合并路径。修复后只对本地专家乘法做回退，各 rank 的分发/合并保持一致。固定输入先用原版、再用修复版、再恢复原版，16 个分数中的差异数依次为 16、0、16 |
| 权重发布 | 保留 expert FSDP 分片的专家归属，重组到实际 EP rank。词嵌入和输出头按 rollout TP rank 发送对应词表行，接收端校验起止行、类型和形状。TP8 时每份 280 MiB，避免发布时汇集完整词表矩阵。当前验证采用连续 TP rank 排列。见 [词表分片传输](runtime/vocab_alignment.py) |

## 带梯度前向之后怎样反传

1. **KDA**。actor 的前向算出逐 token 对齐值，同时调用原生训练 `chunk_kda` 建立梯度图。`_ServingValueNativeVJP.forward(native, serving)` 返回 serving 的值。反向收到上游梯度 g 时返回 `(g, None)`，梯度沿原生 chunk 的 backward 回到 Q、K、V 和门控参数。独立 old 前向无需这条梯度图。三路比较验证的是输出 logprob，梯度使用原生训练实现的雅可比。
2. **V2 打分**。前向保存本次 logits、label 和忽略位置。反向分块计算 FP32 的 `g × (one_hot(label) − softmax(logits))`，忽略位置置零，最后转回输入梯度的类型。这里只需重算局部 softmax。
3. **区分两类重算**。KDA 原生分支的 `skip_kda_recompute=True` 选择保留反向所需中间量。另一个开关 FSDPTurbo 的 `memory.recompute=true` 对选定语言层及视觉块启用激活重算；此次 verl 模型通用 `enable_gradient_checkpointing=false`。层级重算服务于显存，KDA 的梯度来源始终是原生 backward。
4. **96 头 DSA 兼容**。稀疏注意力反向拆成两组 48 头，拼回 dQ，对共享 KV 梯度用 FP32 求和。Indexer KL 遇到不受融合算子支持的 96 个 teacher heads 时，在当前 NPU 上走可微 PyTorch 参考实现。这两处是已有的训练兼容路径。

## 本轮新增与验收边界

- **verl** 增加初次发布、真实更新及末次固定轨迹核验，保存独立三路打分证据，修正分片发布及 packed 输入处理。**FSDPTurbo** 保留门控 FP32 分组、按实际 mesh 生成专家编号，随仓提供本轮使用的 checkpoint MoE 模块。该模块应位于私有 checkpoint 代码视图中。仓内模型目录的 MoE 与 MC2 文件还带入此前版本的实现。
- **vLLM** 增加按实际 TP 词表范围装载和调度分块记录。**vLLM-Ascend** 增加这次全层的 BI、PD、权重版本探针和结果。逐 token KDA、完整 K 投影及 V2 评分沿用已有实现。
- 运行全层探针时，将 vLLM-Ascend 的 `examples/kimi_alignment/full93` 放在本仓 `examples/kimi_alignment/runtime` 之前，两者在 driver 和 worker 的 Python 路径上都须可见。`KIMI_ALIGNMENT_LONG_TRAJECTORY` 指向长轨迹 JSON，短轨迹文件 `single.json` 位于同一目录。
- 三步分别有 5,555、2,910、5,557 个有效回答 token，合计 48 条回答、14,022 个 token。三组两两比较的非有限值数、差异数、平均和最大绝对差均为 0。每步均观测到非零梯度和实际参数变化。末次发布后的版本 3 另取 8 段历史，在两个副本共 16 个评分位置与独立 old 比较，全部相等。
- 长度上限为 1024 输入、2048 输出。训练批里实际提示长度 50–53，多模态展开后为 305–308，3 条回答达到 2048。单独的长文本探针覆盖 1024 输入到 2048 输出。32 机仍需按实际拓扑独立验收。[三路与更新证据](evidence/full93-56e-acceptance.json)
