# Kimi-K3.1 四层 BF16 训推一致性：代码改动、BI/PD 与验证结果

记录日期：2026 年 11 月 24 日。

本文对象是我们在 **423** 上开展的 **Kimi-K3.1** 任务，使用 **verl + FSDPTurbo + vLLM / vLLM-Ascend**。文档按现有日志、源码及第34轮验收记录整理，本次没有启动模型或运行测试。

## 当前可用的个人开发仓与分支

下表给出你的个人开发仓/fork分支，不使用上游主仓的 main 作为查看入口。四个远端分支提交均已回读核对。

| 仓库 | 在本项目中的职责 | 分支 | 固定提交 | 代码入口 |
| --- | --- | --- | --- | --- |
| vllm-ascend | 推理 KDA、DSA、KV/状态缓存、MoE 通信、norm、视觉与投影路径 | `codex/kimi-k31-bf16-upstream-20261124` | [`9adb3321`](https://github.com/yeyunu/vllm-ascend/commit/9adb33214f126b77f982663a5064cbe34106ba8a) | [个人仓分支](https://github.com/yeyunu/vllm-ascend/tree/codex/kimi-k31-bf16-upstream-20261124) |
| FSDPTurbo | 训练模型层、KDA 接入、稀疏算子反向兼容及独立 checkpoint 代码 | `codex/kimi-k31-alignment-main-20261124` | [`36d763bd`](https://gitcode.com/qq_49552963/FSDPTurbo/commit/36d763bd2735c0f56f11eda0daa1355ff5f72a27) | [个人仓分支](https://gitcode.com/qq_49552963/FSDPTurbo/tree/codex/kimi-k31-alignment-main-20261124) |
| verl | RL 编排、独立 old logprob、actor 前向审计、V2 评分反向及共享 helper | `codex/kimi-k31-bf16-upstream-20261124` | [`b14d9f4b`](https://github.com/yeyunu/verl/commit/b14d9f4bd7278e51d794937fd12e99b1c18c8be1) | [个人仓分支](https://github.com/yeyunu/verl/tree/codex/kimi-k31-bf16-upstream-20261124) |
| vllm | Kimi/视觉模型的执行路径与中间量诊断入口 | `codex/generic-sampling-observer-20261124` | [`5391773c`](https://github.com/yeyunu/vllm/commit/5391773c029dbb478bb510398abe0796d04a0d65) | [个人仓分支](https://github.com/yeyunu/vllm/tree/codex/generic-sampling-observer-20261124) |

这些分支保留了服务器时代的基线和实际代码，已撤掉为最新 main 做的目录、接口和配置重构。分支名中的 `upstream`、`main`、`generic-sampling-observer` 是历史名称，不代表当前仍使用此前的移植或通用化实现。

仓库权限说明：以上入口是你的个人开发仓，当前查询为公开可见；“个人仓/fork”不等于平台的 Private 权限。本文没有修改任何仓库的可见性。

还有一个集中保存源码覆盖文件、实验回执和历史报告的个人归档仓：

- [kimi-k31-alignment：codex/kimi-k31-alignment 分支](https://gitcode.com/qq_49552963/kimi-k31-alignment/tree/codex/kimi-k31-alignment)
- 当前远端提交：[`51c29f6a`](https://gitcode.com/qq_49552963/kimi-k31-alignment/commit/51c29f6abb9e15d449fa6bb62703d5c2a4f4e114)。该归档包含后续第35轮诊断材料；本文已验证源码以表内四个分支和第34轮 manifest 为准。

### 版本与证据如何对应

当前代码交付绑定已完成的 **第34轮**：`kimi31_4layer_alignment_423_20261123_202005`。当轮 manifest 记录的 **53/53 个文件**已与交付文件逐一核对，哈希全部匹配。其他补充源码记录的是后续只读采集时的哈希，不能追认为当轮全部文件均有历史哈希证明。

第34轮的原始 `serving_invariance_probe.py` 已按当轮哈希恢复；第35轮新增、尚未执行成功的整模型分段探针与 scheduler-budget 改动没有纳入这份已验证版本。

vLLM-Ascend 原安装版本为 `0.30.0rc2.dev252+g9a0b3fbd6`，对应的完整公开提交未找到。因此其交付保留公开 Git 基底并补齐实际服务器源码，不能把公开基底 SHA 当作原安装版本的精确提交。

## 当前已获得的结果

截至 2026 年 11 月 24 日，完成的是 **四层、16 专家、每 token 选择16专家、BF16** 模型的短测。第32、33、34轮各两步，每步1,024个有效 response token，累计6,144个受检 token 的独立旧 logprob 与 rollout 差异为0。

第34轮还在真实训练入口核对了 old 重算与带梯度 actor 前向：32对 response、2,048个有效 token 相等。当前结果属于固定策略下的前向一致性证据；reward、advantage 和 actor grad norm 为0，尚不能作为有效策略更新验收。

| 项目 | 当前配置或范围 |
| --- | --- |
| 模型 | Kimi-K3.1 四层版本；16专家、top16；BF16，不量化 |
| 框架 | verl 负责 RL；FSDPTurbo 负责训练和独立评分；vLLM / vLLM-Ascend 负责 rollout |
| 资源 | 423，单机8张 Ascend 950DT |
| 训练 Actor / Reference | FSDP=8、EP=8、eFSDP=1 |
| 推理 | TP=8、EP=8、DP=1；短测使用 eager 路径 |
| 短测工作量 | 每轮2步；训练 batch=8、rollout n=2；每步16条 response、每条64个 token |
| 数据入口 | 原启动脚本的 CountBenchQA-lite 图文数据；短测覆盖不等于数据集完整验收 |
| 四层权重 | `/mnt/share/g00815994/kimi-k31/models/Kimi-K3.1-top16-pruned-bf16-4layers` |
| 工作区 | `/mnt/share/y00938198/kimi-k31/test_wxx_skill` |
| 后续全层模型 | 93层、32专家、top16；当前没有该规模的通过结论 |

## 一、训推不一致的定位和解决思路

Kimi 同时涉及 KDA 线性注意力、DSA 稀疏注意力、视觉输入、MoE 和 Attention Residual。训练主要处理完整序列；推理区分 prefill 和逐 token decode，并通过分页 KV 和 recurrent state 延续历史。即使权重与输入相同，算子的分块方式、中间 BF16 舍入、TP 归约、MoE 通信分支和状态写入位置不同，也会使下一层看到不同输入。

定位时先保存同一权重版本、token 历史、图文输入、因果位置和有效 mask。分别捕获输入、attention、投影、norm、MoE、残差与逐层输出，逐步缩小第一个分叉位置。固定真实操作数用于单模块对照，再回到真实训练/rollout 入口复验。

形成的修复方向有三类：

1. **共同前向。** KDA 的训练与推理前向独立调用同一个逐请求、逐 token recurrent 实现。
2. **推理匹配训练。** 视觉 attention、卷积与激活边界、norm、DSA、投影、MoE 和残差主要在推理侧对齐训练算术。
3. **训练评分匹配推理。** 训练使用自身 logits 调用 V2 token-logprob 评分前向，再通过独立实现的解析反向传回 logits。

这里的“共同实现”仍然是两侧各自计算；没有拿 rollout 输出 Tensor 或 rollout 分数替代训练计算。

## 二、逐层对齐前向与反向：具体改了什么

| 模块 | 改动方向 | 当前实现与验证边界 |
| --- | --- | --- |
| KDA 核心 | 训练和推理采用共同 recurrent 前向 | `canonical_kda` 按请求、按 token 调用已有 AscendC `recurrent_kda`。推理 prefill/decode 与训练独立重算统一前向组织，避免 chunk/recurrent 的浮点运算顺序差异。 |
| KDA 训练梯度 | 共同前向值配合 native chunk VJP | 带梯度 actor 额外构造 native chunk KDA 图，前向输出 recurrent 值，反向将上游梯度传入 native 图。固定操作数探针验证了 native VJP 的保留；尚未证明它与 BF16 recurrent 执行轨迹的梯度逐位等价。 |
| 因果卷积与 SiLU | 推理匹配训练的舍入边界 | prefill 的 `causal_conv1d_fn` 与 decode 的 `causal_conv1d_update` 均关闭内融合 SiLU，先得到 BF16 卷积结果，再执行 `F.silu`。 |
| 输入 RMSNorm 与 gated norm | 推理按训练边界拆分 | 分开处理 norm、gate、sigmoid 和乘法，保留中间转换与舍入顺序，避免融合路径产生不同结果。 |
| Attention Residual | 推理复用训练残差算术 | 保留推理 native prefix 加法与 block 状态写入，接入训练侧残差混合，再独立做 RMSNorm。 |
| DSA Q/KV norm 与通道 | 推理匹配训练参数和排列 | Q/KV norm 的 epsilon 从 `1e-5` 对齐到实际训练使用的 `1e-6`；额外64个 NoPE 通道保留 checkpoint 排列，避免重新排列引起归约差异。 |
| DSA KV 写入 | 修复推理状态写入 | norm 后依据实际 storage offset、page/token stride 写入 KV，避免对非连续页视图展平或复制后写入而没有更新原缓存。 |
| 视觉 attention | 推理采用训练对应的 FA 路径 | 使用匹配训练的原生 FlashAttention TND 路径；真实图像操作数探针检查过 Q/K/V、attention 与投影。局部结果不单独代表整个视觉塔验收。 |
| O projection | 推理统一完整 K 维投影 | 收集完整 K 维输入及权重，按当前 rank 行片执行 `F.linear`，对齐训练投影和 TP 归约边界。 |
| Dense MLP、MoE gate 与 shared expert | 推理匹配训练精度及顺序 | 对齐 BF16 gate、激活、专家权重归一化、shared expert 完整 K 投影与 combine 边界；小 batch 通信路径另见 BI 改动。 |
| 所选 token 的 logprob | 训练采用推理 V2 评分前向 | 训练自身 logits 与 labels 输入 `compute_token_logprobs`，输出 FP32 分数；新增解析 VJP，按 `upstream × (one_hot(y) − softmax(z))` 传播梯度，处理 ignored label 与不可生成 token mask。 |
| 96头 indexer KL | 训练反向兼容 | fused CANN 不支持该 teacher head 数时使用可微 KL 参考表达式，保留 teacher detach 与 indexer 参数梯度语义。 |
| 96头共享 KV 稀疏注意力 | 训练反向兼容 | backward 分两次处理48个查询头，拼接 dQ；共享 dK/dV 等先 FP32 求和再转换回目标 dtype，保留前向 output/max/sum。 |

KDA 带梯度前向额外执行 native chunk 前向来建立反向图；无梯度 old 重算不需要这份额外图。因此两条路径虽然前向数值可相等，计算量与保存的中间量并不相同。前向零差不能替代反向及有效更新验证。

## 三、BI 和 PD 的具体改动

BI 指同一请求在不同 batch、邻居和实际打包位置下保持结果；PD 指同一因果历史在完整 prefill、分段 prefill、逐步 decode 下保持结果。需要同时检查算术和状态，不能仅根据一个确定性开关或一次 logpdiff=0 宣布全部通过。

### BI：批次不变性

| 修改点 | 偏差来源 | 实际改动 | 已验证范围 |
| --- | --- | --- | --- |
| 小 batch MoE 通信 | batch 大小可能触发 MC2、FUSED_MC2 等不同通信/计算分支，decode 曾从第1层开始偏离 | 在 `ascend_forward_context.py` 的 `KIMI_ALIGNMENT_ALLTOALL=1` 分支，将候选 MC2/FUSED_MC2 路径改为 ALLTOALL | 原失败路径复验及组合短测通过；尚未把差异细分到融合内核内部的首个舍入点 |
| KDA 请求隔离 | packed 邻居、序列边界或共用 state 槽会影响同一请求 | 按 `cu_seqlens` 分离请求、独立执行逐 token recurrence；decode 检查 live state slot 不重复并按请求写回 | 单模块 packed 邻居与状态探针通过；整模型实际 batch1/2 短测通过 |
| 投影和 MoE 算术 | 不同打包下的投影、分片归约和专家加权边界可能改变结果 | 完整 K 维 O projection/shared expert 投影；对齐 gate、归一化与 combine 顺序 | 作为组合修复通过当前8卡短测，尚未单独证明任意 batch/拓扑下均不变 |
| norm、卷积和视觉算子 | 融合路径的中间舍入或算子选路不同 | 拆分 norm/gate/激活，使用匹配训练的视觉 attention 路径 | 固定输入探针和组合短测有证据；更长序列、更大 batch 尚未完成 |
| 确定性配置 | 同一输入重复执行本身可能波动 | 一致性配置使用确定性等级3，并核对实际 worker 路径 | 用于建立 A/A 重现性；不能单独作为 BI 通过证明 |
| 真实调度探针 | 提交了两条请求不一定实际形成目标 batch；反向提交不一定改变目标行 | `serving_invariance_probe` 暂停引擎、加入目标和邻居、恢复调度，记录实际 batch 成员、目标行、逐位置 logits 与选中分数 | 第34轮实际 batch1 对 batch2，8个生成位置相等；“重排”组实际目标行仍为0，重排行位置验收未完成 |

**BI 当前结论：**单图单 prompt、固定策略版本、8个生成位置的 A/A 和实际 batch1/2 对照通过。batch16、更长轨迹和实际改变目标行位置的完整重排验证仍未完成。

### PD：prefill/decode 阶段一致性

| 修改点 | 偏差来源 | 实际改动 | 已验证范围 |
| --- | --- | --- | --- |
| KDA prefill/decode 前向 | prefill chunk 与 decode recurrent 的组织和舍入顺序不同 | 在 `kimi_kda.py` 中，prefill 转入 `canonical_kda`，decode 经 `serving_decode` 转入同一核心 | 单模块整段、分段延续及 prefill/decode 输出和终态相等；不据此推定整模型分段 prefill 已通过 |
| recurrent state 延续 | 请求到状态槽的映射、初态或写回不一致 | 使用 FP32 state；校验形状、序列边界与 slot；从对应 state 读入，返回终态并写回相同请求槽 | 固定真实操作数的状态延续探针通过 |
| 因果卷积阶段边界 | prefill/decode 的卷积入口不同，融合 SiLU 的 BF16 舍入点可能不同 | 两条路径都先得到 BF16 卷积结果，再独立 SiLU；配套拆分 gated norm | 原操作数对照及整模型组合修复验证；整模型长序列、分段场景仍待扩展 |
| DSA 分页 KV 与通道 | decode 读取分页历史；写入非连续视图或重排通道会改变后续计算 | 用真实 storage offset/stride 写原 KV 存储，保留 NoPE 通道顺序，统一 norm epsilon | 固定输入与算子级对照已验证；完整缓存/调度矩阵未完成 |
| AttnRes 与 prefix 状态 | prefill 和 decode 的 prefix 累加、block 状态及待处理 MLP 输出边界不同 | 保留推理的 prefix/block 状态协议，在对应位置对齐训练残差混合与 RMSNorm | 逐层定位和组合短测支持当前路径；未宣称所有调度状态均通过 |
| 相同历史的整模型对照 | 比较不同历史位置会产生无意义差异 | 固定相同 token 历史，分别执行完整 prefill 与逐步 decode，核对因果位置和 logits | 第34轮 position196：完整 prefill 为197个 query，decode 为1个 query，logits相等 |

**PD 当前结论：**KDA 单模块状态延续已通过；整模型仅完成一个对应历史下的完整 prefill/decode 对照。整模型 chunked prefill、更多前缀长度和缓存/调度组合仍待验证。

第35轮试图扩展实际重排和整模型分段探针，但在引擎初始化时发生 HCCL AllGather 错误并退出，未获得新增数值结论。该轮未通过的诊断改动不能计入第34轮的已验证版本。

### BI/PD 的代码入口

共享 helper 在交付仓中的位置为 `verl/examples/kimi_alignment/runtime/`，对应服务器的 `source/alignment_runtime/`。这是源码存放位置映射，没有改写服务器导入方式。

| 主题 | 仓库与入口 |
| --- | --- |
| KDA 共同前向、请求隔离、状态延续、训练 VJP | verl：`examples/kimi_alignment/runtime/kda_alignment.py`，`canonical_kda` / `serving_decode` / `training_kda` |
| KDA prefill/decode 与卷积、gated norm | vLLM-Ascend：`vllm_ascend/ops/kimi_kda.py` |
| MoE 通信分支 | vLLM-Ascend：`vllm_ascend/ascend_forward_context.py` |
| DSA cache 写入与通道 | verl：`examples/kimi_alignment/runtime/dsa_alignment.py` |
| 完整 K 投影与残差 | verl：`examples/kimi_alignment/runtime/linear_alignment.py`、`residual_alignment.py` |
| 真实调度与逐位置验收 | verl：`examples/kimi_alignment/runtime/serving_invariance_probe.py` |
| 训练 KDA 接入与稀疏 backward | FSDPTurbo：`fsdp_turbo/ops/kda.py`、`fsdp_turbo/ops/npu/` 下对应实现 |

## 四、其他影响训推一致性的因素

| 因素 | 处理方式 | 当前结论 |
| --- | --- | --- |
| 图文输入、有效位置与输出词表 | 保留相同图像和展开 token，明确 response mask、因果位置、不可生成 token 屏蔽及评分语义 | 输入与评分边界是独立比较的前提，不能只比较两条自然生成的回答 |
| rollout 分数复用 | 保持训练侧独立 old 重算，并在 actor 真实前向采集分数 | 没有用 rollout 分数赋值给训练分数来制造零差；性能复用开关不能替代独立验收 |
| 带梯度 actor 路径 | 从实际训练入口核对 old/grad，保留相同策略版本与有效 mask | 第34轮32对 response、2,048个 token 相等；不等同于反向与更新全部通过 |
| Reference 路径 | 同权重、同输入检查 actor/reference 分数 | 第34轮32对 response、2,048个 token 相等 |
| 模型加载路径 | package 模型与 checkpoint Python 代码分开保留，MC2 adapter 不从 checkpoint 强行合并进 package | 交付补齐独立 checkpoint 代码，保留服务器真实加载结构 |
| 代码与运行版本绑定 | 记录实际脚本、源码和运行 manifest；区分第34轮与第35轮诊断版本 | 第34轮53个文件哈希全部匹配，补充文件另列采集来源 |
| 更新与同步 | 需要真实非零训练信号、参数变化、推理端加载新权重，再做同版本评分 | 当前最终短测 reward、advantage、grad norm为0，有效更新与更新后同步验收仍未完成 |

## 五、原始记录与当前结果

下表展示已有审计记录。不同轮次可能使用不同自然生成轨迹，因此这些统计不能直接用来计算某一个补丁的因果收益；单项修复的判断以固定操作数和原失败场景对照为准。

| 记录 | 有效 token | 差异 token | 平均绝对差 | 最大绝对差 | 说明 |
| --- | --- | --- | --- | --- | --- |
| 2026-11-21 14:04:44 启动轮次的首份前向审计 | 1,024 | 1,024 | 0.0154245491 | 0.0653505325 | 最早可用的独立前向差异记录；该轮随后遇到反向兼容错误 |
| 2026-11-23 19:25:14 启动轮次，第1步 | 1,024 | 1,005 | 0.0114118014 | 0.0623979568 | 最终零差前的历史记录 |
| 第32轮，第1/2步分别统计 | 每步1,024 | 每步0 | 每步0 | 每步0 | 独立 old logprob 对 rollout |
| 第33轮，第1/2步分别统计 | 每步1,024 | 每步0 | 每步0 | 每步0 | 独立 old logprob 对 rollout |
| 第34轮，第1/2步分别统计 | 每步1,024 | 每步0 | 每步0 | 每步0 | 独立 old logprob 对 rollout；非有限值数同样为0 |

| 第34轮追加核对 | 结果 | 覆盖边界 |
| --- | --- | --- |
| old 重算对带梯度 actor 前向 | 32对 response、2,048 token，差异0 | 相同策略与实际训练入口 |
| Actor 对 Reference | 32对 response、2,048 token，差异0 | 相同权重及输入 |
| A/A 与 BI | logits与选中分数相等 | 单图/prompt，8个生成位置，实际batch1/2 |
| PD | 对应位置logits相等 | 一条对应历史的完整prefill与decode |
| 有效更新 | 未验收 | 最终短测没有非零有效更新证据 |

第32、33、34轮的6,144个受检 token 是三次短测比较数之和，不表示6,144个互不重复的 token 内容，也不构成长跑或全量模型验收。

## 六、性能代价与后续优化

目前没有完成“原生基线—对齐版”的固定负载、同条件性能配对，不能给出可信的吞吐提升或性能损失百分比。

| 路径 | 当前可能的开销来源 | 实测状态 |
| --- | --- | --- |
| KDA共同前向 | 逐请求、逐token recurrence，是用于对齐的参考实现 | 未完成与原生chunk路径的正式性能配对 |
| KDA带梯度actor | 同时计算共同前向值和native chunk前向图，并保存反向中间量 | 有额外计算与缓存，尚无有效更新下的性能验收 |
| 完整K投影 | 收集输入/权重后投影，增加通信或内存访问 | 未给出独立消融耗时 |
| ALLTOALL MoE | 更改原MC2/FUSED_MC2通信路径 | 零差修复证据不等于性能收益，性能待测 |
| 审计与探针 | logits、状态和中间量的CPU拷贝及落盘 | 属于诊断负载，不能直接用诊断轮耗时代表正式吞吐 |

后续顺序应先补齐当前四层模型的实际重排、长轨迹、batch16和整模型分段prefill；再按授权范围完成反向及有效更新、权重同步后的同版本对照。四层门禁通过后，推进93层、32专家、top16版本，并重新核对资源和并行策略。性能优化要保留同输入零差复验，不能只依据新一轮自然生成的误差变小或耗时下降。

## 证据与查看入口

- [第34轮 gate-status：原始验收记录](https://github.com/yeyunu/verl/blob/b14d9f4bd7278e51d794937fd12e99b1c18c8be1/examples/kimi_alignment/evidence/attempt34-gate-status.json)
- [第34轮 manifest：原始文件哈希](https://github.com/yeyunu/verl/blob/b14d9f4bd7278e51d794937fd12e99b1c18c8be1/examples/kimi_alignment/evidence/attempt34-manifest.json)
- [53个文件与四仓的映射核对](https://github.com/yeyunu/verl/blob/b14d9f4bd7278e51d794937fd12e99b1c18c8be1/examples/kimi_alignment/evidence/attempt34-hash-verification.json)
- [共享KDA实现与native VJP桥接](https://github.com/yeyunu/verl/blob/b14d9f4bd7278e51d794937fd12e99b1c18c8be1/examples/kimi_alignment/runtime/kda_alignment.py)
- [真实同侧BI/PD探针](https://github.com/yeyunu/verl/blob/b14d9f4bd7278e51d794937fd12e99b1c18c8be1/examples/kimi_alignment/runtime/serving_invariance_probe.py)
- [V2 logprob前向和解析反向](https://github.com/yeyunu/verl/blob/b14d9f4bd7278e51d794937fd12e99b1c18c8be1/examples/kimi_alignment/runtime/kimi_scoring.py)
- [FSDPTurbo独立checkpoint代码](https://gitcode.com/qq_49552963/FSDPTurbo/tree/codex/kimi-k31-alignment-main-20261124/examples/kimi_alignment/checkpoint-code)

四个上游草稿PR仅用于审阅： [verl #8818](https://github.com/verl-project/verl/pull/8818)、[vLLM-Ascend #23003](https://github.com/vllm-project/vllm-ascend/pull/23003)、[vLLM #69526](https://github.com/vllm-project/vllm/pull/69526)、[FSDPTurbo #399](https://gitcode.com/Ascend/FSDPTurbo/pull/399)。实际查看和复现版本以开头的个人仓分支与固定提交为准。
