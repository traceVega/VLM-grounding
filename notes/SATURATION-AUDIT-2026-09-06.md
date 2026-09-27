# 多模态 grounding benchmark 饱和度核查与提分路线（2026-09-06）

方法：10 个扫描 agent 按家族检索一手来源，13 个对抗 agent 逐条反驳（两个镜头：**过时数字** 和 **判定错误**），3 路查漏 + 12 个候选核实，4 组方法杠杆调研，5 路策略方案 + 2 位独立评审 + 1 个完备性批评者。共 50 个 agent，约 2,600 次工具调用。

证据等级：**P** = 本轮读了一手来源（arXiv HTML/PDF 文本、官方 repo、HF model card）；**S** = 二手（搜索摘要、第三方榜单转引）。凡标 UNVERIFIED 的不要引用。

---

## 1. 结论先行

你的理解**方向正确，但有五处需要收窄或修正**：

1. **RefCOCO/RefCOCOg 的"饱和"只在一个很窄的范围内成立**：`Acc@0.5` + "训练里含 RefCOCO 的开源通才模型"。同一批测试集在 `Acc@0.75/0.9` 上完全没饱和，而且一个 7B 专用模型仍比所有通才旗舰高 2.5 到 4 分。
2. **RefCOCO 的分割版（RES/mask）没有饱和**，2025-26 还在以 2 到 4 分的步长往上跳。
3. **Ref-Adv 和 GroundingME 确实未饱和**，但两者各有一个会让你误判的结构性问题（Ref-Adv 只公开 1,142/5,000 且有同名数据集冲突；GroundingME 有 20% 的分数是格式化打分的拒绝维度，且小目标维度被分辨率管线污染）。
4. **"很多多模态理解数据集都饱和了"这个说法对 grounding 太宽**。本轮核了约 90 个 benchmark，确证饱和的只有 7 个；其余绝大多数未饱和。
5. **所有"未饱和"判定都只针对 2025 代模型**。没有任何 2026 前沿模型（Gemini 3.x、GPT-5.x/6、Claude 4.8/5、Seed 2.x、Qwen3.5+）在 GroundingME、Ref-Adv、Ref-L4、OpenRef 上有过公开数字。这是本轮最大的证据缺口。

---

## 2. 饱和度总表

### 2.1 已确证饱和（7 个，别再当 headline 用）

| Benchmark | 指标 | 最佳 | 为什么算饱和 | 来源 |
|---|---|---|---|---|
| RefCOCO val/testA | Acc@0.5 | InternVL3.5-241B 94.1/96.3 | 通才前五名 1.5 分内；2026 model card 已不再报 per-split，Qwen3.7-Plus（2026-06）起整行删除 | P 2508.18265；P Qwen3.7-Plus blog |
| RefCOCOg val/test | Acc@0.5 | Youtu-VL-4B 92.2/92.9 | 前二名 0.7 分内；一个 4B 追平 78B | P 2601.19798 |
| RefCOCO-avg（8 split 均值） | Acc@0.5 | Qwen3.5-Plus-Instruct **95.2**；开源 Qwen3.6-27B 92.5 | 2024-12 到 2026-04 全部旗舰落在 92.0-92.5 的 1.5 分窗口内 | P 2604.15804；P HF Qwen3.6-27B |
| Flickr30k Entities | R@1 | Grounding DINO-T 88.1 | 2022 年起 top-3 在 0.7 分内，前沿模型已弃用 | P 2401.02361 |
| ScreenSpot v1 / v2 | 点击命中率 | Qwen-UI-Agent-27B 97.5（v2） | 8 个 8B-397B 模型落在 1.7 分内；v2 审计仍有 10.8% 标注错误 | P 2607.28227；P 2512.16501 |
| GroundUI-1K | 点击命中率 | Holo2 85.5 | 3B 到 235B 全部 82-85.5，4B 追平 235B | P HF Holo2 card |
| V\* Bench | 准确率 | LOCI k=8 **96.0** vs 人类 98.95 | 191 题，1 题 = 0.52 分；6 个方法在 92-96 | P 2608.30959；P 2312.14135 |
| POPE | F1 | ~90.5 | RePOPE 显示 9.3%/1.7% 标签错误会改排名 | P 2504.15707 |
| CountBench | 准确率 | Qwen3.5/3.6 97.8 | **仅在 Qwen 的多选协议下**；开放式协议下 2026 开源 4B 仍在 78-89 | P HF cards；P 2601.19798 |

### 2.2 你点名的两个"未饱和"benchmark：核实通过，但有坑

**GroundingME**（arXiv 2512.17495，CVPR 2026，1,005 题）

| 协议 | 最佳 | 分维度（Dis/Spa/Lim/Rej） |
|---|---|---|
| 单遍非思考 | Qwen3-VL-235B-A22B **45.1** | 69.6 / 49.7 / 54.0 / **0.0** |
| 单遍思考 | 同模型 **49.8** | 65.2 / 73.7 / 45.0 / 5.5 |
| Best-of-16 + 轨迹判官 | **54.3** | 66.7 / 79.7 / 46.3 / 15.9 |

25 个模型里 **20 个在 Rejection 维度得分恰好 0.0**。三个必须知道的坑：

- **n=1,005 给 95% CI ±3.1**，所以第一名和第二名的差距（45.1 vs 42.6）是噪声。它区分的是模型**代际**，不是同代排名。
- **Rejection 是格式打分**：官方 `evaluate.py` 只认字面 `{"bbox_2d": null}`，任何无法解析的输出都算作正确拒绝。报 Rejection 分数时必须同时报解析失败率。（P，官方 evaluate.py）
- **Limited-Small 呈反尺度**：同一批图上 4B 得 48.7、8B 得 **16.0**、32B 得 7.3、A3B 得 52.0。官方脚本不控制分辨率，直接把 JPEG 发给 OpenAI 兼容 API。这几乎肯定是**管线问题不是能力问题**——也正是最便宜的一个提分点。
- 人类天花板：只有一个 100 题的二分类拒绝探针（91%），**804 道正样本没有任何人类分数**。

**Ref-Adv 2026**（arXiv 2602.23898，ICLR 2026，5,000 题 / 公开子集 Ref-Adv-s 1,142 题）

| 协议 | 最佳 | 人类 |
|---|---|---|
| 全集 5,000，需 Set-of-Marks | GPT-4o + CoT + SoM **63.7** | — |
| 全集，原生框 | Qwen2.5-VL-72B + CoT 58.3 | — |
| 公开子集 Ref-Adv-s | Qwen3.5-397B **68.0** | 专家 **90.3**（高）/ 80.6（中） |

- **同名冲突**：Akula et al. ACL 2020 也叫 Ref-Adv（arXiv 2005.01655，RefCOCOg 派生，SOTA 82.8）。引用时必须带 arXiv id。
- 公开子集只占 23%，77% 来自 OpenImages、40% 含否定，**不是全集的代表性抽样**，CI ±2.7。
- **规模已经耗尽**：Qwen3.5-27B 67.3 / 122B 67.2 / 397B 68.0，跨度 1.3 分，全部离人类 22 分。
- 2026 年 8 月起已有 RL 论文（PointRL 2608.25299）把 Ref-Adv-s 当作验证集用，**污染窗口正在关闭**。

### 2.3 未饱和的其余部分（按暴露的失败模式分组）

**长/难 REC**

| Benchmark | 规模 | 最佳 | 缺口 |
|---|---|---|---|
| Ref-L4 | 45,341 标注 | Ovis2.5-9B 90.29 @0.5；Qwen3-VL-8B **88.53 / 77.62 / 55.79 / mAcc 72.95** | Acc@0.9 只有 56，小目标 bin 78.9 vs 大目标 92.7 |
| PR-Bench（RefBench-PRO 后继） | 6,000 | Qwen3.5-27B 65.3 mAcc；Motto-4B 74.1 | 拒绝子任务无模型超过 47.5 |
| HumanRef | 6,000 表达 | Rex-Thinker-GRPO DF1 83.5 | 通才最好 59.5；Rejection 顶到 68 |
| RSC / ScenGround | ~4k 域内测试 | ScenGround Acc@0.5 60.9 ID / 38.1 OOD | 现成 MLLM 全部 <28；标注天花板 ~96% |

**泛化 REC（多目标 / 无目标 / 部件级）— 这是最被低估的一块**

| Benchmark | 最佳 | 缺口 |
|---|---|---|
| gRefCOCO GREC（框） | RC-GRPO-II Pr **75.7** / N-acc 85.3 | 四分之一的表达仍全错（集合级严格指标） |
| gRefCOCO GRES（掩码） | SetCon gIoU 78.4 | N-acc 顶到 ~80，五分之一无目标表达仍吐出假掩码 |
| **OpenRef** | Qwen3-VL-8B F1 **63.7**（同模型 RefCOCO 94.1） | **Multi 35.5、Ordinal 27.7** |
| RefCOCOm / MRES（部件级） | UniRES++ mIoU **27.7** | 全线低于 40 |
| Ref-ZOM / R-RefCOCO / FineCops-Ref / D3 / OmniLabel / OVDEval | 见报告表 | OVDEval 属性子任务（Color 4.5 / Material 9.7）接近零 |

**高分辨率 / 小目标 / 推理分割**

| Benchmark | 最佳 | 人类/上限 |
|---|---|---|
| HR-Bench 8K | S1-VL-32B-RL 93.5（agentic）；单遍最好 ~85 | 人类 86.8 |
| TreeBench | Seed2.0 Pro **64.7** | 无人类基线；对抗 o3/Gemini2.5 构造 |
| ZoomBench | Gemini-3-Flash 59.3 | 论文自证上限（给出证据裁剪）~73 |
| ReasonSeg | StAR+MV test gIoU 72.7 | 作者称标签有噪声，val n=200 排不了名 |
| SOREC（小目标驾驶 REC） | 45.1 mAcc（3.5M 参数适配器） | **7B MLLM 零样本崩到 ~0，微调后也只有 1.9-3.8** |
| MultihopSpatial | Gemini-3-Pro **40.6** Acc@50IoU | 3-hop 自我中心格只有 18.8 |

**GUI**

| Benchmark | 单遍最佳 | 多步/工具最佳 |
|---|---|---|
| ScreenSpot-Pro | 开源 KV-Ground-8B 73.2；专有 Opus 4.8 82.3（无工具） | 87.9（带工具）；GPT-6 Astra 92.7（S，协议未知） |
| OSWorld-G（原始指令） | **可交叉验证的只有 ~70-71**（H Company 自测的 76.9-79.4 第三方复测低 10 分） | LookAgain 82.8（OSWorld-G-R 变体） |
| UI-Vision | Qwen-UI-Agent-27B 70.0 | — |
| GUI-Primitives | Claude Opus 4.7 **32.4%** | 人类二选一 96.9% |

**指点 / 空间 / 视频 / 3D / 检测计数**：Point-Bench 77.2 vs 人类 89.1；Where2Place 和 RefSpatial-Bench 未饱和但**样本量太小排不了名**（n=100/277，CI ±8.5 到 ±11）；Charades-STA R@0.7 最好 56.9（但 34.9% 标注有问题）；ScanRefer Acc@0.5 59.5-64.1；Multi3DRefer F1@0.5 60.7；MLLM 在 COCO/LVIS 的 **F1@0.95 只有 15.9-31.1**，VisDrone ≤3.9；FSC-147 上 MLLM ~21 MAE vs 专家 6.43。

---

## 3. 你遗漏的重要 benchmark（按优先级）

**必须加入的三个**

1. **Ref-L4** — 45,341 标注，唯一同时提供 Acc@0.75/0.9/mAcc 和 小/中/大 尺寸分档的大规模 REC 集。它把 RefCOCO 隐藏的头部空间暴露出来：同一个 Qwen3-VL-8B 在 Acc@0.5 上 88.5，在 Acc@0.9 上只有 55.8。**还带一个天然的污染探针**：COCO 来源的行 92.32/63.87，Objects365 来源的行 84.09/50.28，差 8.2/13.6 分。（P 2608.19553 Table 33）
2. **OpenRef** — 唯一同时覆盖多目标、无目标、专名、多义、序数的开放世界 REC 集，且用了无人机/暗光/恶劣天气图像。它给出本轮最刺眼的一个对比：**同一模型 RefCOCO 94.1、OpenRef 63.7**。
3. **gRefCOCO GREC/GRES + RC-GRPO 那条线** — 集合级严格指标 `Pr@(F1=1, IoU≥0.5)`，这是"多目标 + 拒绝"唯一有成熟基线的地方。

**值得加入的（各暴露一个别处没有的失败模式）**

- **PR-Bench**（2607.24407）：RefBench-PRO 的公开后继，六个子任务，拒绝无模型超 47.5。**注意：RefBench-PRO 报的 Qwen3-VL-8B 71.4 与同一作者 PR-Bench 榜上的 59.4 矛盾，用后者。**
- **HumanRef**：多实例 + 拒绝，坐标式 MLLM 在这里塌方（Qwen2.5-VL-7B DF1 56.2 vs 检索式 RexSeek 82.3）。
- **SOREC**：小目标 REC，7B MLLM 几乎完全失效，是"分辨率/token pitch"假设的最强判别集。
- **MultihopSpatial**：多跳空间 grounding，前沿模型 40.6。
- **GUI-Primitives**：对比式最小对，19 个 VLM 全部 ≤32% vs 人类 96.9%——GUI 空间关系的失败模式 ScreenSpot-Pro 测不出来。
- **FineCops-Ref / D3 / OVDEval / OmniLabel**：组合性、缺失描述、属性检测。
- **VRT-Bench、FindIt、WinDeskGround、FineState-Bench、GUI-360-Bench、ChartREG++、UniRef-UAV、AgroVG、FPCO-Dialog、Interactive Visual Grounding**：2026 年新出，各占一个细分失败模式。

**本轮没能覆盖、但批评者判定应当补的四个家族**：时空视频 grounding（HC-STVG、VidSTG）、航拍/无人机 grounding、跨视角对应、文档文本区域定位（DOGR-Bench、TRIG-Bench、OCRBench v2）。

---

## 4. 被推翻或修正的说法（逐条）

对抗核查推翻了扫描阶段和仓库里 `verifier-anchors.md` 的若干条目。**下面这些是要改的**：

| 说法 | 事实 | 来源 |
|---|---|---|
| RefCOCO 最好是通才 92-94 | **PaDT Pro 7B 96.6 / 97.4 / 95.6**（8-split 平均 94.5），一个 7B 专用模型；Qwen3.5-Plus-Instruct avg 95.2 | P 2510.01954；P 2604.15804 |
| RefCOCO 饱和因为"在 14% 标签噪声带内" | **非因果推断**。14% 错标不定义 86% 天花板（模型考 94-96）。清洗后的实际位移是 +2.0/+2.7/+1.6 分 | P 2406.16866 Table 2 |
| Ref-L4 上 Qwen3-VL-8B = 81.70 | 该格与同表 CogVLM 完全相同，可疑。直接测量是 **88.53** | P 2608.19553 |
| RefBench-PRO Qwen3-VL-8B Acc_p 71.4 | 同一作者的 PR-Bench 榜复报 **59.4** | P 2607.24407 |
| gRefCOCO STAMP-7B val gIoU 73.6 / cIoU 77.6 | **列反了**，应为 gIoU 77.6 / cIoU 73.6。`verifier-anchors.md` 里也是错的 | P STAMP PDF Table 3 |
| OSWorld-G 接近饱和（76.9-79.4） | 那一簇**全部是 H Company 自测**；第三方测同样模型低约 10 分。可交叉验证的单遍最好是 ~70-71 | P UI-Venus-1.5 Tables 13-14 |
| Where2Place 最好 76.0（GR-ER 1.5）；RefSpatial 72.2 | **误归因**。Google 自报 GR-ER 1.5 是 Where2Place 59.0 / RefSpatial 48.5 | P Gemini Robotics 1.5 Table 19 |
| V\* Bench 未饱和 | **已饱和**：LOCI k=8 得 96.0，人类 98.95，191 题 | P 2608.30959 |
| ReasonSeg 训练 SOTA ~68 gIoU | 已过时，test 现在 **72.7**（StAR+MV），且 top-5 在 0.9 分内 | P 2603.14382 |
| ScanRefer 最好 65.9/59.5 | TDVR 零样本报 **70.85 / 64.06** | P 2608.03763 |
| Nr3D/Sr3D 未饱和 | **Sr3D 已弱饱和**（top-3 在 0.4 分内，人类 86.1）；Nr3D 未饱和（75.1-76.1 vs 人类 92.2） | P 2606.31148；P ReferIt3D |
| GroundingME 未进任何评测框架 | 已进 **lmms-eval**（PR #949）和 **VLMEvalKit**（PR #1382），复现值 31.2 / 31.3 vs 论文 31.0 | P GitHub |
| GroundingME thinking 增益 4.7% 到 7.4% | 论文正文与自己的表冲突（GLM-4.5V 实际 +1.9）。用表不用正文 | P 2512.17495v2 Tables 3/6 |

---

## 5. 什么方法真的推动过这些未饱和的榜

### 5.1 免训练杠杆（先做这些）

| 杠杆 | from → to | 基座 | 代价 | 陷阱 |
|---|---|---|---|---|
| **换 Thinking checkpoint** | Ref-Adv-s 47.2 → **59.5**（8B）；4B 42.5→57.6；32B 53.4→65.6 | Qwen3-VL | 推理 token 5-10x | **不是开关**，是另一个做过 RL 的 checkpoint |
| Thinking mode（GroundingME） | 45.1→49.8（A22B）；39.5→46.9（32B）；31.0→34.3（8B） | Qwen3-VL | 同上 | **Discriminative 每个尺寸都掉**（8B 61.3→52.5），Limited 也掉。**必须路由，不能全开** |
| **LFPR 无标签边界细化** | Ref-L4 Acc@0.5 88.53→89.73，**Acc@0.75 77.62→80.79，Acc@0.9 55.79→61.14**，mAcc 72.95→76.01 | Qwen3-VL-8B | ~2x 前向 | 无护栏版本**每个指标都掉**（-2.9/-4.0/-5.7）；阈值是在 test 上选的 |
| 仅分辨率路由 | RefCOCO 小目标 bin Acc@0.9 41.96→**52.09** | Qwen3-VL-8B | 7% 的样本走第二遍 | 自然图像只有 7% 的框够小 |
| OpenRef MCC（计数-检测一致性） | F1 58.9→66.9（GLM-4.6V）；N3R 38.1→**91.9**（2B） | 多个 | 2-3 遍 | **只在弱拒绝模型上验证过**，最强模型未测；GLM-4.6V 的 N3R 反而掉了 |
| Best-of-16 + 轨迹判官 | GroundingME 49.8→54.3 | 235B thinking | **~30x** | 30B-A3B 判官零增益；Discriminative 完全不动（66.6→66.7） |
| GUI 两遍 zoom | ScreenSpot-Pro 67.9→73.5；ZoomClick 54.0→72.1 | 多个 | 2-4x | **在自然图像上只有 +0.6 到 +1.2**，不是 +7 到 +25 |
| 拒绝任务改写成是否问句 | RefBench-PRO Reject 3.1→55.1 / 15.8→64.2 | 多个 | 一次额外查询 | 作者自己说"接近随机（约 50%）" |

### 5.2 训练杠杆

| 杠杆 | from → to | 代价 | 附带损害 |
|---|---|---|---|
| **RC-GRPO 拒绝校准 RL**（强制 None rollout + 过度拒绝惩罚 + 负优势缩放） | gRefCOCO Pr/N-acc 38.2/6.4→**64.9/71.5**（7B）；72.4/77.2→75.7/85.3（Qwen3-VL-4B） | **仅 2,000 样本，LoRA，2×A800，500 步** | MMBench 82.6→82.5，POPE 不变。**本轮性价比最高的训练杠杆** |
| GroundingME 拒绝数据混合 SFT | Rejection 0→27.9（2:1 比例） | 30k 样本，3 epoch | **正样本崩**：Discriminative 61.3→40.2，Limited 36.0→17.0，正样本整体约 25.5 |
| 坐标表示（Hi-Token / PaDT / Rex-Omni 0-999） | RefCOCO P@0.95 14.4→**31.1**（Rex-Omni）；PaDT-3B 89.1→93.2 Acc@0.5 | 需要重训 | 格式效应本身有好几分，**和很多 RL-vs-SFT 的宣称增益同量级** |
| 检测器提案 + 选择（RexSeek / Rex-Thinker） | HumanRef DF1 56.2→**82.3** | 多阶段大规模训练 | Rex-Thinker 去掉框提示掉 13.2 Recall；CoT 6.68s vs 1.13s |
| IoU 奖励 GRPO | RefCOCO 88.7→90.55；**OOD LISA-Grounding 56.51→63.14**（SFT 反而掉到 54.82） | 600 步 @3B | 域内增益 1-2 分**在格式效应之内**；纯 IoU 奖励会拉低 P@0.95 |

### 5.3 已证伪的路子（不要浪费时间）

- **类别缺失负样本不能训出 GroundingME 式拒绝**：Qwen3-VL-4B 在 gRefCOCO no-target 上 N-acc 已经 77.2，在 GroundingME Rejection 上仍是 **0.0**。前者是"类别不在图里"，后者是"物体在但属性不符"，是两个任务。
- **规模不解决拒绝**：Qwen3-VL 从 2B 到 32B 再到 235B MoE，非思考模式下 Rejection 全部恰好 0.0。
- **agentic crop/zoom 救不了小目标**：Claude-Sonnet-4.5 + PyVision 在 GroundingME Limited-Small 上只有 7.3，总分 12.4，作者归因于裁剪后的坐标偏移。因果审计（2608.06270）显示 DeepEyes 有 72.8% 的 rollout 是"调用而不看"。
- **迭代自我修正是 oracle 假象**：505 题的 RefCOCOg/Ref-Adv-S/Ref-L4 混合集上，单次 79.6，oracle 最佳步 82.0，但**每一个可部署的停止规则都不如不迭代**（最好 75.6），置信度-正确率相关只有 r≈0.22。
- **CoT 在 GUI 点击和指点上是负的**：GUI-G2 ScreenSpot-v2 93.3（无思考）vs 88.7（思考）；Point-Bench 上 GPT-4o -2.9、Gemini-2.5-Flash -16。
- **无护栏的裁剪重定位有害**：LFPR 的无护栏对照在每个指标上都比不裁剪差。
- **别把 GUI 配方原样搬到自然图像**：thinking 的符号在两个域是相反的；zoom 在 GUI 上 +7 到 +25，在自然图像上 +1。

---

## 6. 建议路线（按信息成本排序的合并方案）

两位评审独立都选了「按失败模式拆解」方案作为主干（41-42 分），并一致要求嫁接：「纯推理」方案的两个开局实验（43 分，最决定性）、「数据」方案的污染协议（污染分 10/10）、「RL」方案的奖励 A/B 和回归防御、「架构」方案的 SAM 3 细化器和注意力门控。

硬约束：**永不在 benchmark 测试图上训练**；每个数字都要报 image-clean 子集；任何 RL 跑完都要出通用能力回归表（MMMU/MMBench/POPE/DocVQA/AI2D）；每一步都有停止规则。

### 阶段 0：开始前必须先解决的两件事（批评者标为最高优先级）

- **基座选择没验证**。五个方案都硬编码 Qwen3-VL-4B/8B，但本轮自己的表显示更新的小模型更强：PR-Bench 上 Qwen3.5-9B 63.7 vs Qwen3-VL-8B 59.4。**先花 2 GPU-h 把 Qwen3.5/3.6 的 4-9B 类基座跑一遍这些榜**，否则所有"from"数字都是错的。
- **没有 2026 前沿模型的数字**。如果 Gemini 3.x 在 GroundingME 上已经 70+，那你追的是代际差不是任务难度差。用 API 跑一次（几十美元）就能定性。

### 阶段 1：本地、免训练、最决定性（约 8-15 GPU-h，RTX 5090）

| # | 动作 | 代价 | 停止规则 |
|---|---|---|---|
| 1 | 协议锁定 + 基线复现 + pHash/DINOv2 污染索引 | 6-12 GPU-h | Qwen3-VL-8B 的 GroundingME 必须落在 31.0±2.5 且 Rejection 恰好 0.0，否则先修 harness |
| 2 | **GroundingME Limited-Small 像素预算扫描**（150 题 × {1.05, 4.19, 16.7 MP} × {4B, 8B}） | **<1 GPU-h** | 某个预算把 Small 提升 ≥8 分（CI ±7.6）才保留；否则该缺口不是像素问题，转去查坐标缩放 |
| 3 | Ref-Adv-s：direct vs cot vs Thinking checkpoint | 0.3-2 GPU-h | cot−direct <+2.0 就别做 CoT；Thinking 8B 必须 ≥55（论文 59.5），否则是 harness bug |
| 4 | LFPR：仅路由 vs 护栏裁剪（Ref-L4 **val** 3,000 题，绝不碰 test） | 1 GPU-h | 仅路由的小目标 Acc@0.9 增益 <+4 就停；护栏裁剪不如仅路由就只发布仅路由 |
| 5 | OpenRef MCC 在 8B 上复现 | 1 GPU-h | F1 ≥+2.0 或 N3R ≥+3.0 才保留 |
| 6 | Best-of-N **带完整基线阶梯**（greedy / 随机挑 / oracle pass@N / IoU-medoid） | 1.5-2.5 GPU-h | oracle−greedy <+8 说明无可选；medoid−greedy <+1.5 就关掉。**GroundingME 那个 +4.5 从来没报过这三行** |
| 7 | 按查询类型路由 Thinking（正则匹配序数/关系/否定词） | 0（复用步骤 1 的输出） | 必须捕获 ≥50% 的 oracle Spatial 增益，且 Discriminative 掉幅 ≤2.0 |

阶段 1 的预期合计：GroundingME 31.0 → 38-40，Ref-Adv-s 47.2 → 57-60，Ref-L4 mAcc 72.9 → 75-76。**全部不改权重、不碰测试集。**

### 阶段 2：本地训练（4B LoRA，约 25-40 GPU-h）

| # | 动作 | 停止规则 |
|---|---|---|
| 8 | **RC-GRPO 拒绝校准**，3-4k 条 COCO-train/Objects365 图上构造的属性不符负样本（**不是**类别缺失负样本），跨族听者（Molmo2-8B + Qwen3-VL-4B）验证 | Rejection ≥10.0 **且** 无拒绝维度掉幅 ≤2.0 **且** RefCOCOg val 掉幅 ≤1.0 |
| 9 | 奖励形状 A/B（同数据同种子）：IoU-only vs Ref-R1 动态阈值 vs Hi-GAR 分级容差 vs mAcc 形状 | 在 Ref-L4 val 上某个变体 Acc@0.9 ≥+2.0 且 Acc@0.5 掉幅 ≤0.5 |
| 10 | OpenRef 多目标集合值 GRPO（匈牙利匹配集合 F1 + 基数惩罚） | Multi F1 ≥+5.0 且 Single F1 掉幅 ≤1.0 |
| 11 | 难干扰物 RL（**最后做，最可能是 null**） | Ref-Adv-s ≥+3.0 且 Discriminative 不掉；否则"没有便宜的解法"本身就是结论 |

### 阶段 3：整合与云端（RunPod H100，约 20 GPU-h）

合并存活的 LoRA，8B 上做一次联合巩固；跑完整评测 + 通用能力回归表 + image-clean 分层报告（Ref-L4 按 COCO vs Objects365、Ref-Adv-s 按 OpenImages vs COCO、GroundingME 按哈希排除后的 SA-1B 部分）。

**巩固前的门槛**：先把阶段 1 的推理策略 + 步骤 8 的 LoRA 叠加，重跑 GroundingME。总分 ≥38.0 且 Discriminative ≥59.3 才值得做合并训练；否则说明各维度增益不可加。

---

## 7. 不要做的事

1. 不要全局打开 thinking——它在 204 道 Discriminative 题上掉 8.8 分来换总分 +3.3。**路由它。**
2. 不要用高负样本比例做 SFT——2:1 换来 Rejection 27.9，代价是 Discriminative 61.3→40.2。
3. 不要用类别缺失负样本训拒绝——那是另一个任务（4B 在 gRefCOCO 上 77.2，在 GroundingME 上 0.0）。
4. 不要在自然图像上做无护栏裁剪或迭代自我修正。
5. 不要为小目标上 agentic zoom 工具循环——先修像素预算。
6. 不要复刻 235B best-of-16 + 判官作为主要杠杆（30x 成本，小判官零增益）。
7. 不要把 RefCOCO/+/g Acc@0.5、ScreenSpot-v2、V\*、POPE 当 headline——只当回归表用。
8. 不要引用 RefBench-PRO 的 71.4 或它的 Ref-L4 81.70。
9. 不要在没有解析失败率的情况下解读 GroundingME Rejection 或 PR-Bench N-Acc。
10. 不要把单遍和多遍数字放同一列。
11. 不要试图在 5090 上同时常驻 8B Instruct + 8B Thinking（2×17 GB > 32 GB）——顺序跑，离线合并。
12. 不要在 WSL 下预载 GroundingME 的 8K 图或 Ref-L4 的 9,735 张图——真实上限约 23 GB，超了发行版会被回收并杀掉后台训练。

---

## 8. 未解决的问题

1. **人类天花板几乎全部缺失**。GroundingME 的 804 道正样本、PR-Bench、OpenRef、Ref-L4、gRefCOCO、HumanRef 都没有人类分数。所以现在的"未饱和"只意味着"低于 100"，不是"低于人类"。若这些集合有 15-20% 的歧义样本（ReasonSeg 是 13%、OSWorld-G 是 14.2%），可达头部空间会大幅缩水。**建议在阶段 1 之后花 300-500 美元做一次 500 题的人类基线**，这决定整个项目值不值得做。
2. 没有任何 2026 前沿模型在自然图像难 REC 上的数字（见阶段 0）。
3. 四个 grounding 家族本轮未覆盖：时空视频 grounding、航拍、跨视角、文档文本区域。
4. GroundingME Small 的反尺度现象成因未定（像素预算 vs 坐标归一化失败）——这是阶段 1 第 2 步要回答的，也是最便宜的一个提分点。
5. 若干 S 级数字仍未核实：Chain-of-Ground 68.4、GPT-6 Astra 92.7、llm-stats 上全部 26 条 ScreenSpot-Pro 条目（该站自己标注"0 条已验证"）。
6. OpenRef 的 HF 数据集是**受限访问**的，train/val 划分和规模论文没给，需要先申请。
