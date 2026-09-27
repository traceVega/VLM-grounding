# 拒绝回答（Rejection）深度调研：文献、缺陷、方案（2026-09-16）

方法：11 个 agent，三阶段。第一阶段 4 个调研 agent 按切片读一手来源（grounding 本领域 / 通用弃答与校准 / 相邻领域与机制 / 前沿实验室技术报告，共约 170 篇），1 个诊断 agent 在本地表格上重算；第二阶段 Verifier 对 42 条承重数字回查原文（31 VERIFIED、10 PARTIALLY、1 REFUTED），Skeptic 裁决争议并给出 17 条陷阱，3 个 brainstorm agent 从训练目标 / 推理架构 / 数据评测三个角度出 20 个方案；第三阶段 Skeptic 逐条审查、合并、排序。所有中间文件在 `team/rejection/`（章程 `CHARTER.md`，调研 `survey-A..D`，诊断 `diagnosis-local.md`，核查 `verifier.md`，审查 `skeptic-survey.md` / `skeptic-ideas.md`，方案 `ideas-1..3`）。标 **P** 的数字是本轮在本地表格上实算的。

---

## 1. 结论先行

**你们要修的不是"模型不会说 null"，而是"模型不核对从句"。而且现有的两条主要证据链都被本轮修正了。**

1. **决策 token 上的 p(null) 在 GroundingME 负样本上没有任何缺席信号，这不是 operating point 问题。** 全 benchmark AUROC 0.298 [0.265, 0.335]，对同图源同风格的 Discriminative 正样本 0.468，长度匹配上界 0.565。把 OpenImages 上校准的 5% 误弃答阈值搬到 GME 只抓到 2/201 个负样本，却在 804 个正样本上误弃答 126 个；要抓到一半负样本得牺牲 611 个正样本。`ABSTAIN-SIGNAL.md` 的标题"模型总是知道"只在类别标签移除上成立。**[P]**
2. **`CORRECTION-2026-09-05.md` 的"长度"变量是错的，变量是子轴。** 那 28 个"≤20 词 Discriminative 移除"逐条正好就是 28 个 Discriminative/**Text** 项（Text 中位 13 词，其他子轴 46–49 词）。非 Text 的 87 个 Discriminative 移除里长度与信号无关（ρ=+0.06，p=0.61），46–71 词的 Text 项仍然移动 +5 到 +14 个数量级。Verifier 独立重算确认。但 Skeptic 指出这个否定结论只覆盖 23–86 词（非 Text 项没有短的），且 9 个长 Text 项移动 +8 个数量级却 0% 弃答——**一个变量能同时解释两组数据：最佳剩余候选满足从句的比例（稀释度）**。子轴 vs 稀释度是本轮留下的第一争议，两个 0.15–0.5 GPU 小时的探针能定（§4 P1、P10）。**[P + Verifier V-R35]**
3. **GroundingME 的 201 个负样本仅凭表达式文本就能以 AUROC 0.92 从正样本里分出来**（对 Discriminative 0.81，对 Limited 1.00；负样本词汇偏向 young / shirt / sleeved / jacket）。任何在 GME 上报的"缺席 vs 在场 AUROC"、任何探针、任何训练后的拒绝增益，都必须先过这个盲基线，或者带灰图对照跑。20 个方案里 9 个没有灰图对照，6 个把 AUROC 门槛定在这个地板以下。**[P: Skeptic]**
4. **RL 的前提被算术确认：GRPO 自己采不出 null。** 201 个负样本上 p(null) 之和 3.4e-4；k=16、T=1 下期望有 null rollout 的负样本数 0.005；T=4 才到 1.9 个，代价是 87/804 个正样本也采出 null。强制 / 离策略 null，或者一条独立通道，是必选项。**[P]**
5. **"简单负样本已解决、困难负样本为零"这个表述要改。** 章程写的"gRefCOCO 负样本 = 类别缺席"是错的：gRefCOCO 的默认规则 1 是"类别在场、属性不符"（"the kid in blue"），规则 2（他图表达式）是标注员的备选，比例未公布。所以 Qwen3-VL-4B 在 gRefCOCO 上的 77.2 N-acc 可能恰恰说明**短的、单从句的属性不符负样本模型已经能拒**——这支持稀释度读法。0.2 GPU 小时的 COCO 真值分层能定（P10）。**[Verifier V-R36]**
6. **文献里唯一"负样本对定位免费"的证据全部来自把 presence 放在坐标流之外的架构**（SAM 3 presence head：0→30 个对抗挖掘的硬负样本/图，IL_MCC 0.44→0.68，pmF1 62.4→62.8 不动；MolmoPoint 的 no-more-points 类；Molmo 文本"This isn't in the image."；Rex-Omni `COORDS: None`）。**每一个把 "none" 塞进同一条自回归坐标流的 SFT 都付了 4–21 分正样本**（GroundingME 1:8 就掉 4 分 Discriminative，2:1 掉 21），或者什么都没换到（Jedi 266 万条 = 没训过的 OS-Atlas）。**没有任何前沿实验室训练 GroundingME 打分的那种"同一 JSON key、值为 null"形式。**
7. **前沿实验室没有人在优化这个能力。** 15 份技术报告 / 模型卡里只有 Qwen2.5-VL 一句话提到"合成了不存在的物体类别"（且它是自家系里最差的拒绝者）；Qwen3-VL 报告全文 grep 零命中。Qwen 2.5→3 代际间简单负样本指标涨了（HumanRef Rej 7.1→47.9，OpenRef N3R 14.1→84.6），硬负样本没动（RefBench-PRO 3.1→15.8，GME 0.5→0.0），Qwen3.5 在简单负样本上还**退步**了（HumanRef 10.8/13.4）而正样本上涨。GroundingME 排行榜上每一个非零的非思考模式拒绝分都属于弱 grounder（Llama-4-Maverick 6.0 @ 总分 13.0，Gemini-2.5-Pro 7.0 @ 20.7）——这是"解析失败计为正确拒绝"的签名。
8. **RC-GRPO 从来没在 GroundingME 上跑过。** 它是唯一带过度拒绝惩罚、强制拒绝 rollout 和通用能力回归表的配方，但 +65 的 headline 是在一个根本不会输出拒绝串的 7B 上测的（Qwen3-VL-4B 上 +8.1），OOD 只往梯子**下方**测过（FineCops-Ref → gRefCOCO），trade-off 是一个旋钮（α：N-acc 19.6–74.5 vs P-acc 73.6–60.9）。它在 L4 上的数字是这条线上最明显的缺失实验，也就是 `RL-DESIGN` 的 S3 本身。

**预算应该怎么花：先 ~10 GPU 小时本地探针（§4.2），结果直接决定训练押注里哪几个死掉；然后第一个训练实验是 SFT 重定价臂 + RL-DESIGN S3 作为对照臂的同数据 A/B（§4.3）。**

---

## 2. 调研：近期有影响力的工作

### 2.1 负样本难度阶梯（本轮把各 benchmark 的负样本按构造方式排序）

| 阶 | 负样本怎么造 | 零样本 MLLM | 训练后能到 |
|---|---|---|---|
| L0 随机无关文本 | 随机 caption / 类别 | Qwen3-VL-8B AgroVG 88.9 | ~85 |
| L1 他图表达式 | gRefCOCO 规则 2、OmniLabel、UniRef-UAV | Qwen3-VL-4B gRefCOCO **77.2**（Qwen2.5-VL-7B 6.4 是格式效应：它不会输出拒绝串） | 专家 ~85（RC-GRPO-II 4B 85.3） |
| L1.5 类别在场、短属性不符、人写 | **gRefCOCO 规则 1（默认）**、UniRef-UAV | 从未与 L1 分开报过；UniRef-UAV 纯文本 Qwen3.6-27B 0.03，给图像范例后 0.87 | 未知 |
| L2 LLM 扰动正样本表达式的一个元素 | FP-RefCOCO、FineCops-Ref、HumanRef、RefBench-PRO、OpenRef 颜色/朝向、Ground-V | HumanRef Qwen2.5-VL-7B 7.1；RefBench-PRO Qwen3-VL-8B 15.8；FineCops 5.3 | **域内 58–71**（Ref-R1 58.2、RC-GRPO 61–71、Rex-Thinker 68.2）；域外塌（Ground-V/PSALM-G5：gRefCOCO 83.7 → 自家属性集 33.9） |
| L3 编辑图像、表达式不变 | FineCops 负图、HalluSegBench、我们的移除 | 类别标签移除 50% 弃答、AUROC 0.964 **[P]**；SESAME 在编辑图上仍 43% 幻觉 | — |
| **L4 人工改假长描述里的一个从句、真照片、物体在场** | **GroundingME Rejection**（201 条，中位 **54 词**不是 39）、PR-Bench Reject | **20/25 模型 0.0**；最好 9.5（32B thinking） | 2:1 SFT 27.9，代价 Disc −21 / Lim −19 / RefCOCOg −5；Motto-2B PR-Bench 46.9 |
| 正交轴：否定 | Ref-Adv（21% 否定，目标存在）、D3 ABS | 72B 58–67 vs RefCOCOg ~90 | 不是拒绝问题：模型选了在场的干扰物 |

两个锚点：同一 checkpoint L1 77.2 / L4 0.0（Qwen3-VL-4B）；PSALM-G5 L1 83.7 / L2 33.9。**规律：每个把 L2 拒绝拉 50 分以上的配方都在同分布上训和测；仅有的两次 OOD 测试，往下（RC-GRPO L2→L1）活，往上（GroundingME L2→L4）只剩 27.9 且付 20 分。没人在 L4 风格负样本上训过——GroundingME 没有训练集，它 SFT 用的负样本构造也没公布。**

### 2.2 grounding 本领域最有影响力的五篇

1. **gRefCOCO / GRES**（2306.00968）— 发明 no-target 样本和 N-acc / T-acc 一对指标；此后所有 MLLM 拒绝数字都在它的词汇里。规则 1 / 规则 2 比例未公布，是"简单 vs 困难"至今混在一起的根源。
2. **FP-RefCOCO / SESAME**（2312.08366）— 第一个 LLM 扰动的 false-premise 集；诊断出"grounding 微调让模型忘了怎么说不"（基座 LLaVA 本来会说不）；2026 年的编辑图测试上仍是最好的先前模型。
3. **HumanRef / RexSeek → Rex-Thinker**（2503.08507, 2506.04034）— 第一个带干净前后对比的硬负样本拒绝子集（0 → 54 → 68），第一次证明 CoT 以零正样本代价买到拒绝（+13.8，GRPO 再 +0.9）；也最清楚地展示了零框指标被过度拒绝刷分（无提示时 71.7，作者自认）。
4. **GroundingME**（2512.17495）— 悬崖：在 gRefCOCO 上 77–85 的模型在人工改假的长描述上 0.0；域内 97 到域外 28 且付 20 分正样本。定义了本项目的目标。
5. **RC-GRPO**（2608.04698）— 第一个带过度拒绝惩罚、强制拒绝 rollout（constrained beam search 出 "There are none"）、通用能力回归表和诚实单旋钮 trade-off 曲线的拒绝配方；自然的下一个实验（在 L4 上跑）没人跑。

**曾经移动过 L2+ 拒绝数字的方法（全表见 `survey-A` §4）：** SESAME LoRA SFT（FP-detect 51→80，RefCOCOg −1.8）；GSVA `[REJ]` token（N-acc 2.7→56.5，正样本无损）；RexSeek 拒绝数据（0→54）；Rex-Thinker CoT-SFT+GRPO（53.5→68.2，DF1 +1.2）；Ground-V 幻觉数据（gRefCOCO 24.5→83.7，自家硬集 33.9）；Ref-R1（SFT 3.1→58.5，RL 再 −0.3——**RL 对 Reject 贡献为零**）；CRS 检测器候选 + "None" 选项（5.2→62.3，+12.5 P@1 来自检测器）；GroundingME 2:1 SFT；RC-GRPO；MCC 训练无关计数一致性（2B N3R 38→92，GLM-4.6V 79→73 **反而掉**）；FINER-tuning DPO（VQA）。

### 2.3 通用弃答文献里能借的和不能借的

**弃答失败的五种机制，各自的证据：**（a）解码 operating point——TRAPSBench 探针 0.91 vs 自发克制 0.292；HALP 生成前探针均值 0.89；我们类别标签上的 0.964。（b）知识/感知缺失——TRAPSBench"文本不可能性比视觉证据缺失容易检测约 4×（中位，范围 3–197×）"；MM-AQA 前沿 VLM 对不可答题只弃答 1.2–4.9%，对矛盾证据"试图调和"；FPCO-Dialog 假前提纠正 identity 0.56 > attribute 0.41 > **location 0.15**。**部分匹配这个 regime 在所有弃答 benchmark 上都失败。**（c）评测激励——Kalai 2509.04664：主流评测不给 IDK 分；但 GroundingME 自己就给 null 计分，20/25 仍是 0，**所以 grounding 的激励问题在训练数据里，不在评测里**。（d）先验过度作答——POPE 时代 LLaVA yes 率 95–99%；在类别负样本上已消失，在硬负样本上还在。（e）RL 双向崩溃——Hallucination Tax：仅在可解题上 RFT 后拒绝率 0.30→0.08，混 10% 不可答题恢复到 0.85、GSM8K 只掉 0.05；AbstentionBench：推理微调平均掉 24% 弃答、规模无效；反向：AWA-RL "r_ref=0.05 就导致灾难性 reward hacking（99.9% 拒绝）"，TIAR λ=0.3 崩，TruthRL 三元 37.2 vs 二元 20.8。

**不崩正样本的目标函数的共同点**：要么未知题由模型自己采样打标（Alignment-for-Honesty −0.4、IDK token TriviaQA recall 0），要么 RL 里误弃答被显式惩罚且正确答案仍拿最高奖励（Abstain-R1 U-Ref 9.4→68.1 **且**可答准确率 48.8→57.2；TruthRL +7.8）。外部强加的负样本 SFT 代价随比例单调：10% → −0.05；20+20% → −6 到 −8；67%（GroundingME 2:1）→ −21。**KoNA 不是零代价（0.77→0.70，Verifier 修正）。**

**置信度信号里过 AUROC 0.85 的只有**：隐状态/注意力上的**有监督探针**（HALP 0.89、HaloProbe 0.935、HARMONY VizWiz 0.87–0.89）、图像消融似然对比（BCEA 存在性 0.86–0.88）、区域注意力对框正确性（Propose-and-Attend MTLA 0.89 vs token log-prob 0.69，Qwen3-VL-8B/COCO），以及我们自己的决策 token 在 Text 子轴上。所有 verbalized / 自我验证 / 采样一致性信号都在 0.55–0.82（IVT r=0.22；SafeGround 0.70–0.82；MM-AQA MCC 0.11）；证据被移除时 verbalized 置信度不动（Gemini 18 帧和 6 帧中位都 0.9）。

**不能直接借的**："知识边界"框架假设未知题是采样时会答错的题，而硬负样本是模型**一致地、高置信地**答错的题，采样找不到它；yes/no 存在性作为代理——RefBench-PRO Table 5 同一批硬负样本 grounding 格式 3–28、yes/no 格式 47–64，作者说"接近随机"，**但表里没有正样本侧的 yes/no 率，所以既不能读成"格式买了 40–50 分"也不能读成"验证能力不存在"**（Verifier V-R05/06；survey B 把 InternVL3-8B 抄反了，正确是 21.3 / 46.8）；conformal 保证需要部署分布的校准集；规模和 thinking 在所有来源里都是死杠杆。

### 2.4 相邻领域：谁把"不存在"做对了

**检测器**：缺席 = 每候选一个分数 + 负类自己的损失。SAM 3 的 presence token 是一个全局 BCE 训练的 null 决策，负样本上逐 query 监督被屏蔽；Table 10 里"只在概念在场时监督 mask 分"（54.0 cgF1）优于 DETR 式在负图上惩罚每个候选（52.2）。**数据项远大于架构项**：head 本身 +1.5 cgF1 / +0.05 IL_MCC（pmF1 65.4→63.4，论文自己注明消融来自不同长度的训练，不可直接比），而每图 0→30 个硬负样本 +14.7 cgF1 / +0.24 IL_MCC。它的硬负样本定义："当前模型会为之预测 mask、且 mask 落在在场物体上的可混淆概念"，**已经被当前模型拒绝的候选直接丢弃**——这是 GroundingME"物体在场、一个从句为假"的类别级孪生。Rex-Omni（最强的纯 next-token 检测器）在论文里承认"多模态模型缺乏可靠置信度，AP 不适用"，改报 F1。

**移植进 MLLM 的**（全表见 `survey-C` §1.2）：GSVA `[REJ]`（gRefCOCO N-acc 65.4，正样本无损）；PostAlign `<REJ>` + BCE（HaloQuest 假前提 2.0→33.2）；ROD-MLLM（OWLv2 候选做 anchor token + 字面 `None`，anchor token 概率当 AP 分：D3 ABS 28.7 vs 自家定位器 24.7，RefCOCO 90.2）；ChatRex（有打分候选但没有 null 选项 → HumanRef 拒绝 **0.0**）；Rex-Thinker（逐候选 CoT，68.2）；Molmo-7B-D 零样本 HumanRef 68.6（PixMo 里有"不在图里"的指点数据；但任何非指点输出都计为拒绝）。**没有一个在属性不符负样本上连同正样本代价一起测过。**

**GUI**：Jedi 的 266 万拒绝样本（仓库 `refusal.py`：把另一张截图的指令复制过来，标签 `wait 10`，无对抗过滤）→ OSWorld-G refusal 7.4 = 完全没做拒绝数据的 OS-Atlas-7B。VenusBench-GD（900 条单属性编辑的 GUI 负样本）：所有只用正样本做 grounding SFT 的专家模型 0–51（Holo1.5-72B **0.00**，UI-TARS-72B **0.00**），基座 Qwen2.5-VL-72B 78.1、Qwen3-VL-8B 63.8——**正样本 grounding SFT 删掉残余弃答，在第二个领域得到确认**。OSWorld-G 上唯一过 40 的是 MEGA-GUI 的独立 refuser 阶段：同一个 Gemini 2.5 Pro 单发 38.9，先推理再决定 68.5 @ 3.3% FPR——**仅靠 prompt 把 presence 决策从坐标决策里解耦就值 +30**。过度拒绝无人测：WebArena 里给拒绝提示后 GPT-4 把 54.9% 的可行任务判为不可能。

**机制**：三个独立的 2025–26 结果说 VLM 在中层编码了缺席但不输出——TRAPSBench（Qwen3-VL-8B 探针 0.91 vs 行为 0.29；**但它的"层 20 转向后 75% 弃答"是在可答视频上诱导出来的，即无差别转向**，Verifier V-R08）、VA neurons（中层 FFN 激活高而 p("No") 低，Qwen2-VL）、Knowledge-vs-Verbalization 方向分离。2410.02707：信号在"精确答案 token"处，且**探针不跨任务类型迁移**——正好预测我们类别标签 0.964 → 段落 0.685 的塌陷。2509.22449 的弃答转向方向 α=2 时对 95% 可答输入也弃答。**没人跑过的实验**：在坐标输出 MLLM 的决策 token 隐状态上训练探针（用移除数据），在 GME Rejection 上测，再加门控转向。

### 2.5 评测伪影（每个数字旁边必须带的东西）

| 指标 | 静默奖励/隐藏的 |
|---|---|
| GroundingME Rejection | **不可解析输出计为正确** [P: 读过 evaluate.py]；201 条 CI ±6.2；每子轴 50 |
| HumanRef Rejection Score | "无框即拒绝"，解析失败计入；Rex-Thinker 无提示时靠过度拒绝到 71.7 |
| N-acc（gRefCOCO 系） | 一律拒绝得 100；必须配 T-acc / P-acc，多数 MLLM 论文不报 |
| OpenRef N3R | 坐标 token 平均概率的软乘积，一个低置信框仍拿大部分分；名词交换（易）与颜色/朝向交换（难）混池——这就是与 GME 相关 −0.304 的来源；Mistral-3 F1 5.9 拿 70.9 |
| RefBench-PRO RejAcc | 无框**或**明确说不存在，judge 未说明 |
| FineCops-Ref pairwise | 从不要求输出弃答，一直吐框也能得分 |
| thinking 模式的所有非零数 | 解析伪影 + 先验位移签名（每个模型 Rejection 上升同时 Discriminative 下降：8B 61.3→52.5，32B 75.0→65.7，与 2:1 SFT 同形） |

---

## 3. 缺陷分析：到底哪里错了

Diagnostician 的 D1–D8 加 Skeptic 补的 D9–D16，按证据强度。全部本地数字来自 `tables/gme_original.jsonl`（1,005 条）和 `tables/pilot_abstain.jsonl`（332×3），Qwen3-VL-8B-Instruct。

| # | 缺陷 | 证据 | 强度 | 最便宜的锐化探针 |
|---|---|---|---|---|
| D1 | **LM-head 上的 null logit 在 L4 负样本上无信号**（不是 operating point） | AUROC 0.298；vs Discriminative 0.468；5% 预算阈值抓 2/201、误弃答 126/804 **[P]** | 强（对 logit）；对中层表示**未测** | P5 隐状态探针（0.5 GPU-h + 灰图 0.4） |
| D2 | **部分匹配接受 / 从句不进决策**：同类候选在场、一个从句为假时照样框 | Rejection 内子轴 KW p=0.58，Text>Appearance AUROC 0.496，词数 ρ=−0.09；仅有的 3 个 ≤20 词 Rejection 项全是 Text，p(null) 1e−16..1e−18，框是车牌/螺栓/飞机大小；Rejection 框中位 5% 图面、p(box) ≥ 0.9996 全 201 条 **[P]**；2605.09090 在 TransVG/SwimVG 上报同样的"近似行为"（不是 MLLM 特有） | 模式强；机制（D2a 可感知但忽略 vs D2b 分辨率下不可感知）**未分开** | P1 表达式形式阶梯（0.5）、P6 逐从句隔离测试（Text 子轴 0.5） |
| D3 | 输出通道里没有逐候选分数；p(null) 也不是正确性分 | wrong-vs-right AUROC 0.573，Spatial 0.494 **[P]**；**但** Propose-and-Attend 的 MTLA 在 Qwen3-VL-8B 上对框正确性到 0.89——候选级分数**存在**，缺席才是未测的 | 强（对 p(null)） | 并入 P5 的 eager 前向 |
| D4 | 格式打分伪影 | 官方 scorer 计不可解析为正确；我们的 harness 读决策 token，0/201 是真决策零 **[P]** | 强 | thinking 跑 4B 版（1.5–2 GPU-h），延后 |
| D5 | **过度弃答是可见性，不是缺席**：null 目前的含义是"看不见" | 41 个误弃答里 40 个 Limited/Small，真值中位面积 0.005% 图面；面积 <0.05% 弃答 28.4%，>0.1% ≤1.2%；ρ(log p(null), 面积) −0.48，Limited 内 −0.69；弃答是**自信的**（中位 0.9994）**[P]**。含义：null 通道是被训过的（多半是微小物体或 PixMo 式"不在图里"数据），任何阈值/转向方法首先继承这个含义 | 强 | 已计划的像素预算扫描改读弃答率 |
| D6 | 拒绝指标跨 benchmark 不自洽 | 同一模型同一决策 token：类别标签移除 0.964 / 50%，GME 0.298 / 0%；跨模型 GME vs N3R ρ −0.304；各 benchmark 负样本还各带自己的文本签名（D14） | 强 | 不需要：永远按负样本类型分报 |
| D7 | 分数的零对规模不变 | 2B→235B 非思考全 0.0；表示是否随规模变**未测** | 分数强，表示未测 | 2B/4B 本地 + 32B 云（~$3） |
| D8 | **RL 前提：null 的 pass@k ≈ 0** | Σ p(null)=3.4e−4；k=16、T=1 期望 0.005 条；T=8 才有 28 条采到，代价 255/804 正样本 **[P]** | 强（假设前缀近确定，p(box)+p(null)≈1 支持） | P8 自由生成 pass@8（0.5，同时数散文式拒绝——官方 scorer 会给它计分） |
| D9 | **没有任何已发表方法在 GME 正样本上测过过度拒绝** | Table 3/6/7 只报维度准确率；Discriminative 掉 8.8 可以全是误 null 也可以全是错框 | 强 | 任何一次跑：804 条按维度的字面 null 率 |
| D10 | "中心名词在场"是 benchmark 构造的假设，201 条上没测 | 若一部分 Rejection 图里根本没有中心名词实例，那是伪装的类别缺席负样本 | 未测 | P2：Molmo2 指点 201 图（0.1 GPU-h）+ 50 条人工 |
| D11 | thinking 数字双重可疑：解析伪影 + 先验位移；Thinking checkpoint 与 Instruct+CoT 混淆 | Table 6；CI ±4 内 32B 9.5 vs 8B 4.5 无差 | 强 | 同 D4 |
| D12 | **没有人类天花板** | 唯一人类数字是 100 题二分类探针 91%；"50 分"要对着 ~90 读 | — | 201 + 201 匹配正样本 × 3 标注员 ≈ $550 或 2 个人日（图是研究许可，不能公开众包） |
| D13 | 测试集窄：98.5% >20 词，99/201 是人或车，服装词汇，单一图源 | **[P]** | — | 任何增益在 PR-Bench Reject（1,000）、RefBench-PRO Reject（1,000）、VenusBench-GD（900）上复现后才叫能力 |
| D14 | **benchmark 文本可分性 0.92** | NB 文本分类器 5 折 CV **[P: Skeptic]** | 强 | 每个方法带灰图对照 + 文本基线行 |
| D15 | Rejection/Text 的引号串在 P21 像素上限下是否可读未知 | 49/50 Text 项带引号串 | 未测 | 50 条 OCR 读回（0.05 GPU-h），cap 与 2× cap |
| D16 | D2 应拆成 D2a（可感知但忽略）与 D2b（不可感知）——修法不同（目标函数 vs 分辨率/数据） | — | — | P6 按子轴报"假从句上的 yes/no 准确率 vs 同类真从句" |

**两句话的机制模型**：`null` 现在是一个带可见性项的**类别在场检测器**，不是**描述满足检测器**；Text 串和微小孤立斑块被当作"中心名词 gestalt"的一部分，所以它们没了会触发 null，而同类候选带一个假从句不会。GME 的两种失败（338 个大小正确、物体错误、满置信的正样本错框，和 201 个自信的负样本框）是**同一个忽略从句的选择操作**。

**三个争议的裁决状态**（Skeptic + Verifier）：
- (a) 子轴 vs 长度 vs 稀释度：Diagnostician 在"CORRECTION 的变量搞错了"上胜出；"子轴是变量"过度延伸；稀释度（最佳剩余候选满足从句的比例：短 Text 移除 = 0/1 满足 → null；长移除 ≈ (k−2)/k → 框；Rejection = (k−1)/k → 框）能同时拟合。**P1 + P10 定。**对数据设计的含义不管结果如何都成立：负样本要故意跨稀释度（1 假 among k 真，k ∈ {0,1,2,4}），不要让 Text 主导，假从句必须在评测分辨率下可被基座单独验证。
- (b) 格式 vs 负样本类型：RefBench-PRO Table 5 定不了（两列地板差 50 分且无正样本对照）；对我们的模型"输出地板"论证本来就不成立（它在正样本上输出过 41 次字面 null）。**P4（带正样本对照和灰图的 yes/no 探针）定。**
- (c) D1"不是 operating point"vs"是 read-out 不是表示"：同一个词，不同层，都可以对。**P5 定**，四种结果各对应哪些方案存活见 §4.2。

---

## 4. Brainstorm：怎么提升

20 个方案经 Skeptic 合并成 8 个簇（无 REJECT；5 PROMISING，9 MINOR，6 MAJOR）。原则：每个数字带灰图对照和文本地板行；主 AUROC 用 GME 唯一免疫文本地板的对比——同图的 F0（原段落）vs F5（假从句改真）对；永远不把 Rejection 与正样本维度放进一个加权总分。

### 4.1 八个簇

| 簇 | 保留的形式 | 合并掉的 |
|---|---|---|
| A 整句 presence 通道 | 零样本 yes/no 探针（P4，带灰图）→ **冻结隐状态上的 presence head**（T2，SAM 3 移植）→ 仅当 head 失败但 P4 盲修正后通过时才做 **两轮 presence-turn**（T3：yes/no 走自己的 LoRA，框那一轮跑基座权重，坐标流从不见负样本） | I2-7(e) 两次调用；RefBench-PRO 缺的正样本对照 |
| B 逐从句核对 | **I2-2 隔离 vs 整合分解**：把表达式拆原子从句，对模型自己框的物体逐从句问 yes/no，按子轴报 AUROC；Molmo2 跨族行 | I1-4 自验证蒸馏、I1-6 的奖励 oracle 都要先读它 |
| C 表达式形式阶梯（EFL） | **一次 0.5 GPU-h 的跑**：Text 行 = 50 个 Disc/Text 正样本换引号串（N0，无编辑）+ 49 个 Rejection/Text 缩成"中心名词 + 引号串"；Appearance 行 = 50 个 Disc/Appearance 正样本短翻转 + 50 个人工核过的 Rejection/Appearance 缩短；F4 正对照；F5 同图正样本；全部再跑灰图 | I3-2、I3-3、Skeptic 的 2×2、I2-5 FLIP 臂、I3-4 的推理阶梯 |
| D 评测卡 | **I3-5 九行 + 第 10 行文本地板**：字面 null / 解析失败 / 官方分三个数；负样本类型分层；按尺寸分层的过度拒绝（加 ≥36 词的 134 条长 Discriminative 层）；同图 F0-vs-F5 AUROC；协议交换；灰图；套件一致性；DocVQA/AI2D/MathVista/CharXiv；人类天花板。**用两个已知作弊臂验证卡本身**（宽松 prompt 臂必须读成误 null 上升 + 盲增益 >0；坏 JSON 臂必须读成官方分升 / 字面 null 不动） | I2 的四列、Diagnostician D4/D9 |
| E 硬度门控负样本引擎 | **I3-1 策略对抗挖掘（PAM）**：OpenImages 实例库（9,692 场景已有），≥3 同类实例，每正样本 8 个单从句翻转（每子轴 2 个，标注装饰性 / 承重性），**先过门 3**（基座 p(box) ≥ 0.99 且框落在目标或同类兄弟上）再过跨族门；每个子轴的盲可检测性（纯文本 judge AUROC ≤ 0.60，否则 SugarCrepe 式重生成）；输出基座 log10 p(null)、翻转类型、从句数 | RL-DESIGN S0 门 1–4、I3-7 的门 |
| F 坐标流内目标函数 | **同数据同种子一次 A/B**：T0 RL-DESIGN S3（对照）· T4 配对奖励（同图 (e⁺, e⁻) 文本对，无编辑器，**null 单独永不得分**，只有对刚框过的孪生做出决策翻转才有分；k 变化）· T5 name-the-clause 作为胜者的第二阶段 · T1 SFT 重定价臂 · T6 转向 rollout 仅当 P5 通过 | I1-2/3/5/6 |
| G thinking / prompt | I2-7 (b) 显式 null 指令 + (e) 先推理后决定，在 405 条上先跑；thinking 4B 版后跑 | I2-6、Diagnostician 探针 3、5 |
| H 图像侧敏感性 | **删掉** I2-5 图像侧（`gme_remove` 已答：94.3% 的移除会重新框一个替代物；物体大小的框修复干净率 ~46%）；I2-4 的贪心 in-box 注意力质量并入 P5 | — |

**新的观点（本轮之前没有的）**：GME 硬负样本是把一段**唯一可定位**描述里的**装饰性**从句改假（一个候选匹配其余一切，只差一个细节），不是把承重从句改假（所有候选都部分失败）。两者都是合法的零满足，但是不同的阶，要分开挖掘和报告；`RL-DESIGN` 门 4 的"承重率"是正样本表达式的必要条件，不是该翻转的对象。

### 4.2 前 ~10 GPU 小时（全部本地 5090，8B-Instruct 在 P21 上限，一次一个模型驻留）

预注册阈值写死；"仅仪器"= 结果只验证工具不验证主张。每行必带：文本地板（对 Discriminative 0.81，对全部 0.92，同图 F0-vs-F5 ≈ 0.5）和灰图对照。

| # | 探针（簇） | 规模 / GPU-h | 预注册阈值 | 测什么 |
|---|---|---|---|---|
| P0 | CPU：文本地板（已做）；pHash 每个候选图源 vs GME/RefCOCOg/Ref-L4/Ref-Adv-s/PR-Bench；冻结卡的 1–4、6、10 行 | 0 | — | 仅仪器 |
| **P1** | **EFL**（C）：Text 行 + Appearance 行 + F4 ×100 + F5 + 全部灰图 | ≈400 prompts + Gemma4 核对 ≈ **0.5**（+1 h 人工） | "可单独验证"当且仅当弃答 ≥25%（Wilson 下界 >15%）**或**中位移动 ≥+5 数量级；"不可"当且仅当弃答 ≤2/50 **且**移动 <+2；F4 必须 ≥80% 框对，否则形式无效；灰图各形式弃答 ≤5% | **主张**（争议 a：子轴 / 稀释 / 都不是） |
| P2 | D15 可读性（50 个 Rejection/Text 串 OCR 读回，cap 与 2× cap）+ D10 中心名词在场（Molmo2 指点 201 图；50 条人工） | ≈300 prompts，**0.3** | 可读 ≥80%，否则 P1 Text 行在 2× cap 重跑；中心名词在场 ≥90%，否则 GME Rejection 一律按"类别在场 / 类别缺席"两层报 | 仅仪器 |
| P3 | I2-7 (b)+(e) 在 201 + 204 Discriminative + 134 长 Discriminative；最佳变体跑灰图 | **0.6** | 变体"算数"当且仅当字面 null ≥10% **且** 204 与 134 条上误 null ≤5% **且**灰 null ≤ 有图的 ⅓；否则 prompt 杠杆已耗尽，争议 (b) 在 GME 上判为负样本类型 | 仅仪器（基线行）+ 争议 b |
| **P4** | 零样本 yes/no："是否存在满足描述每一部分的物体"，201 + 204，读 p("no")；灰图同跑 | 810 prompts，**0.3** | AUROC vs Discriminative ≥0.85 且灰 ≤0.60 → 存在 presence 通道；0.65–0.85 且灰 ≤0.60 → 弱通道；≤0.65 或灰 ≥0.75 → 无通道 / 文本先验 | **主张**（簇 A 前提；RefBench-PRO 缺的对照） |
| **P5** | 隐状态转储（1,005 GME + 333 移除 + 632 pilot + 108 not_clean；4 位置 × 37 层），**eager attention** 顺带算 MTLA；再转储 1,005 张灰图；CPU 探针 S1/S2 → GME、GME 内 CV、灰图 CV | **1.5** + **0.4** | "read-out 缺陷"当且仅当某 (层, 位置) 用 S1/S2 训练在 absent-vs-Discriminative 上 ≥0.70 **且**灰图探针 ≤0.60；CV 天花板只有 >0.81 且灰 CV ≤0.60 才算；"表示缺陷"当且仅当各处 ≤0.65。MTLA：absent-vs-Disc ≥0.65 推翻 D3 的拆分；wrong-vs-right ≥0.70 = 仅正确性仪器 | **主张**（争议 c） |
| **P6** | I2-2 Text 子轴隔离：50 Rejection/Text（自己的框）+ 50 Disc/Text（真值框），≈600 从句 prompt；灰图同问；通过后再跑其余三子轴（≈1.0）和 Molmo2 行（≈1.0） | **0.5**（+2.0） | Text 隔离 AUROC ≥0.80 且灰 ≤0.60 → 检查存在（D2a）；<0.70 → 不存在（D2b，与 P2 一起读）；扩展子轴各 ≥0.70 才算 | **主张**（D2 隔离 vs 整合） |
| P7 | I3-1 先门 3：200 OpenImages 场景 × 8 翻转，写手 + 策略；翻转上跑纯文本 judge | **0.7** | 存活 ≥30%（p(box) ≥0.99 且框在目标或兄弟上）**且**存活者中位 log10 p(null) ≤ −12 → 库能喂训练；盲可检测性每子轴 ≤0.60，否则该子轴重生成 | 仅仪器（数据可行性） |
| **P8** | null pass@8，T=1 自由生成，严格解析器 + 散文式拒绝检测器：50 Rejection + 50 Limited/Small + 50 类别标签 REMOVE + P1 的 50 短 Text N0 | 1,600 次生成，**0.5** | Rejection pass@8 ≤2% 确认 D8（强制 / 转向 / 配对信号必选）；短 Text N0 pass@8 ≥20% → 存在有在策略 null 的底阶；散文拒绝单独报 | **主张**（D8）+ 课程底阶 |
| P9 | 评测卡基线跑（1–4、6、10 行，官方解析器，自由生成）+ 两个作弊臂 | **1.2** | 宽松 prompt 臂读成误 null 升 + 盲增益 >0；坏 JSON 臂读成官方分升 / 字面 null 平；否则在读任何训练数字前重设计卡 | 仅仪器 |
| **P10** | **gRefCOCO 规则 1 分层**（新；需 200 张 COCO train2014，**下载前先问**）：Qwen3-VL-4B 在 200 条 gRefCOCO val no-target 上，按 COCO 真值标"中心名词类别在场"，分层 N-acc | **0.2**（+下载） | 类别在场项 N-acc ≥60% → 短的单从句属性不符负样本零样本就能拒 → **稀释度胜出**，"gRefCOCO = 简单"部分是长度效应；≤20% → 77.2 来自规则 2，子轴读法成立 | **主张**（争议 a，公开集带真值） |

合计 ≈ 9.5–11.5 GPU-h。**顺序：P0 → P1 → P2 → P4 → P5 → P6 → P8 → P10 → P3 → P7 → P9**（前六项约 4 GPU-h、4 个人日，答完所有三个争议）。

**结果映射（哪个探针结果杀掉 / 提升哪个训练押注）**：

| 结果 | 死 | 活 / 提升 |
|---|---|---|
| P1 Text 可单独验证、Appearance 不可 | Text 为主的负样本配比；I3-7 对 Text 的前提 | T4/T5 在非 Text 负样本上；I3-4 阶梯在 App/State/Cmp 上；B 簇作 Text-only 教师 |
| P1 两行都可（稀释度是变量） | RL-DESIGN"优先短可核对从句"的原文表述；I1-3 的移除→编辑课程轴 | **I3-4 支持率课程成为核心数据设计**；T4 配对奖励 k 变化；I2-2 集合逻辑管线升为训练无关基线；T8 获资 |
| P1 都不可（且 P2 可读 ≥80%） | I1-4（无教师）；I2-2 管线；I1-1 的"起点优势" | T5 name-the-clause / RL-DESIGN 第二阶段（目标函数必须**创造**这个检查）；T2 只作有监督数据押注 |
| P2 可读 <80% | P1、P6 的 Text 行在 cap 下无效 | 两者在 2× cap 重跑；D15 进入每个 Text 数字 |
| P2 中心名词缺席 >10% | "GME = 单一硬负样本层" | 所有 GME Rejection 数字分层报 |
| P3 某变体算数 | "prompt 已耗尽"；排行榜 0.0 作为纯能力数 | 该变体成为强制基线行；T3 失去未训 vs 训练的差值 |
| P4 ≥0.85 且灰 ≤0.60 | — | T2、T3 提升；survey D 的读法得到正样本对照 |
| P4 ≤0.65 或灰 ≥0.75 | T3 的"起点优势"；I1-4 整句前提 | T2 仅作数据押注；F 簇目标函数 |
| P5 read-out（≥0.70，灰 ≤0.60） | "表示缺陷"框架 | T6 转向 rollout 活；T2 强提升；门控转向作训练无关臂 |
| P5 表示缺陷（各处 ≤0.65） | T6；门控转向；"表示了但没输出"这篇论文 | T0/T1/T4/T5（只剩目标函数 / 数据）；T2 仅用 S3 特征 |
| P5 有图高**且**灰图高 | 至今所有 GME 上的 AUROC 主张 | 先重设计负样本（I3-1 带盲门）再引用任何探针 |
| P6 Text ≥0.80 | — | I2-2 全跑；T7 对通过的子轴活；T5 奖励 oracle 有效 |
| P6 Text <0.70 | T7；I2-2 管线；I1-6 对 Text 的 oracle | T4/T5 用 N0′ 负样本；T2 |
| P7 存活 <30% 或中位 >−8 | 用 OpenImages 挖掘的负样本训练 | I3-7（SA-1B）或 GME 图编辑作源；F 簇全部延后 |
| P8 Rejection pass@8 >2% | D8 原表述 | 纯 GRPO 变得可能，强制 None 可选 |
| P10 类别在场 N-acc ≥60% | 子轴读法；"简单 vs 困难"作为负样本**类型**的阶梯 | 稀释度；**T8 提升为第一个训练实验** |

### 4.3 探针之后的训练押注（4B LoRA 本地，数据来自 I3-1；GPU-h 不含造数据）

| 押注 | 内容 | 前置 | GPU-h | 哪里 |
|---|---|---|---|---|
| **T0** | RL-DESIGN S3 原样（集合 F1 GRPO、强制 `{"boxes": []}`、α=0.5、过度拒绝惩罚、3–5k 挖掘样本）——survey A 说缺的 RC-GRPO-on-L4 | P7 过 | 10–15 | 本地 |
| T1 | I1-5 SFT 重定价：臂 (i) LoRA 2:1、(iii) 全参 lr 1e−6 20% 负样本、(v) = (iii) + 10% 通用 VL 回放；长翻转作独立层；负样本过盲门 | P7；**先找到 CAP-MECH §4 那条"lr÷10 消除 17.6/32.1pp 回归"的来源 id**（本轮无人核过） | 2 本地 + 6–9 云 + 6 评测 | 本地 + 1×80 GB |
| T2 | I2-3 冻结隐状态 presence head（线性 → MLP → 注意力 query），在 I3-1 池上训，GME 上报 AURC | P7；P5（特征） | 2–4 | 本地 |
| T3 | I1-1 presence-turn（yes/no LoRA；框轮跑基座）+ 匹配的 null-in-stream 对照臂 | P4 盲修正 ≥0.65；P7 | 3–5 | 本地 |
| T4 | I1-3 配对奖励**文本对形式**（同图 e⁺/e⁻，无固定 null 奖励）、k 变化；N0′ 图像对作第二臂（编辑器需先问） | P7；P8 底阶；e⁻ 过盲门 | 8–12/臂 | 本地 |
| T5 | I1-6 name-the-clause 作为 F 簇胜者的第二阶段（null 只在 rollout 说出失败从句且与翻转记录一致时得分） | T0/T4 结果；P6；盲门 | 12–18 | 本地 |
| T6 | I1-2 转向 rollout（转向 / 强制 / 无 三臂） | P5 read-out **且**灰 ≤0.60 | 18–27 | 本地或 3×80 GB |
| T7 | I1-4 自验证 DPO | P6 ≥0.80 于 ≥2 子轴；跨族一致 ≥70% | 10–13 | 本地 |
| T8 | I3-4 支持率 A/B（仅 L1 vs L1–L4 vs 仅 L4）用 F 簇胜者的目标函数 | P1 稀释或 P10 ≥60%；I3-1 L4 产出 ≥15% | 30–45 | 本地两周或 3×80 GB ≈ $40 |
| T9 | I3-7 SA-1B GME 风格训练集 + 人工编辑臂 | P1 稀释**且** I3-1 L4 产出 <15% | 8–10 建 + 20–30 A/B + $450 | 本地 + 可选云 |

**排序一：每人周信息量（主排序）**

| 名次 | 押注 | 理由 |
|---|---|---|
| 1 | **T1 SFT 重定价** | ≈1 人周；任何结果都给整个组合重定价——若臂 (iii) 到 Rejection ≥13 且 Discriminative ≤ −2，就没有 RL 项目 |
| 2 | **T0 RL-DESIGN 基线** | ≈1.5 人周；它就是缺失的 RC-GRPO-on-L4，是其他目标函数的对照臂，也是硬负样本上第一条诚实的 α 曲线；带护栏的否定结果可发表 |
| 3 | T2 presence head | ≈1 人周（数据之后）；正样本代价按构造只剩误 null 一个数；之后每个臂的 operating-point 基线 |
| 4 | T4 文本对奖励 | ≈1.5 人周；唯一完全没有 null 奖励且不需要编辑器的目标；k 变化在训练内部回答稀释度 |
| 5 | T8 支持率 A/B | ≈2 人周；第一次测 (k−1)/k regime 必须在训练集里吗 |
| 6 | T3 | 信息在误 no 率不在增益；除非 P4 很强否则被 T2 取代 |
| 7 | T5 | 质量最高、每周信息最低；需要 P6 和盲门 |
| 8 | T7 | P6 之外信息很少 |
| 9 | T6 | 作者自己给前置失败 55% |
| 10 | T9 | ≈3 人周 + $450，双重条件 |

**排序二：全部通过时的论文质量**

| 名次 | 押注 | 主张 |
|---|---|---|
| 1 | T5 on T4（+ k>1 迁移测试） | 拒绝即经验证的从句核对：必须说出理由的弃答，多目标迁移作机制级验收 |
| 2 | T2 + P5 | "缺席在中层被表示但没被输出"：冻结骨干上的 presence head 以零正样本代价恢复硬负样本拒绝——**仅当 P5 的灰图对照过** |
| 3 | T4 + T8 | 无 null 奖励的配对奖励 + REC 里第一次支持率（稀释度）课程测量 |
| 4 | T9 + I3-1 | 图源不相交、策略对抗、跨族验证的 GME regime 训练集，人写 vs LLM 翻转的比较 |
| 5 | **T0** | 缺失的实验；硬负样本上带全护栏的 α 曲线——无论结果都扎实，不是 headline |
| 6 | T3 | SAM 3 配方作两轮 MLLM 协议，带实测误 no 代价 |
| 7 | T1 | 重定价结果 |
| 8 | T7 / T6 | 难与 MCC / R-Tuning / ICPO 区分 |

**RL-DESIGN 现有设计（T0）按信息量排第二、按质量排第五。信息量上只有 SFT 重定价臂在它前面；质量上排在它前面的三个（T5-on-T4、T2+P5、T4+T8）全是复用它数据引擎和护栏的扩展，不是替代。**

### 4.4 每个实验的硬件（估计值）

| 实验 | 数据 | 模型 | GPU 显存 | 主机内存 | GPU-h | 位置 |
|---|---|---|---|---|---|---|
| P1–P4、P6、P8 | GME 405–1,005 条、50–100 条编辑/翻转 | Qwen3-VL-8B-Instruct（Gemma4-12B 核对 24 GB 顺序驻留） | 20–24 GB | ≤18 GB（图流式） | 各 0.3–0.6 | 本地 |
| P5 | 2,078 条 + 1,005 灰图，4 位置 × 37 层（2.8 GB 落盘 ext4） | 8B-Instruct，eager attention | ~22 GB | ~20 GB | 1.9 | 本地 |
| P7 | 200 场景 × 8 翻转 | Qwen3.5-9B 19 GB → 4B 9 GB → Molmo2-8B 17 GB → Gemma4-12B 24 GB 顺序 | ≤24 GB | ≤20 GB | 0.7（全门 2.0） | 本地 |
| P10 | 200 gRefCOCO val + COCO 图 | Qwen3-VL-4B | ~10 GB | ~18 GB | 0.2 | 本地（下载先问） |
| I3-1 全量 S1 | 3–5k 已验证样本 | 同 P7 | ≤24 GB | ≤20 GB | 15–30 | 本地 |
| T0 / T4 / T5 | I3-1 池 | 4B LoRA GRPO + co-located vLLM | 28–31 GB | ≤20 GB | 见上表 | 本地 |
| T1 (ii)–(v) | 6k 条 | 4B 全参，8-bit Adam + grad ckpt | ~40 GB | — | 5 × 2–3 | 云 1×80 GB（≈$2–3/h） |
| 8B 复现 | 同上 | 8B LoRA / 全参 | 30 GB / 2×80 GB | — | 3–8 | 云 1–2×80 GB |
| 人类天花板 | 201 + 201 × 3 标注 | — | — | — | 0 | ≈$550 外包（NDA）或 2 个人日 |

---

## 5. 需要修正的现有文件（本轮结论推翻或收窄了它们）

| 文件 | 原表述 | 改为 |
|---|---|---|
| `notes/ABSTAIN-SIGNAL.md` 标题与结论 | "模型总是知道；贪心解码不能作用于它" | 只在类别标签移除（L3）上成立；GME L4 上决策 token 无信号（AUROC 0.298） |
| `notes/CORRECTION-2026-09-05.md` §1 | "≤20 词的短可核对从句 → 43% / 0.972，>35 词的渐变外观段落 → 0% / 0.685；变量是表达式（长度）" | 那 28 条恰是 Discriminative/**Text** 项；Appearance / Component / State 在任何长度都 0–7% / 0.63–0.69；非 Text 在 <23 词无样本，长度在那里未测；子轴 vs 稀释度待 P1/P10 |
| `notes/RL-DESIGN-CANDIDATE-VERIFICATION.md` §1.2 | "优先造离散可核对从句"的理由是"短从句携带信号" | 理由改为**标签可验证性**；带一个假从句的长段落必须进训练分布；负样本要跨稀释度 k∈{0,1,2,4}；Text 不能主导（它是最容易的硬负样本）；门 4 的承重率是正样本的必要条件，不是翻转对象；§3.3 的强制 None 是离策略 SFT（CAP-MECH 规则 3），T6 三臂是它的检验 |
| `team/rejection/CHARTER.md` 及所有引用 | "gRefCOCO 负样本 = 类别缺席"；"GME 描述 39 词" | gRefCOCO 默认规则 1 = 在场类别属性不符，规则 2 备选，比例未公布；GME Rejection 中位 54 词（分位 38/46/54/64/74） |
| `notes/SATURATION-AUDIT` §"不要做" 第 3 条 | "不要用类别缺失负样本训拒绝（4B 在 gRefCOCO 77.2、GME 0.0）" | gRefCOCO 不是干净的类别缺失集；这条的证据要等 P10 |
| `notes/CAPABILITY-MECHANISMS` §2 表首行 | RC-GRPO "+3.3 Pr / +8.1 N-acc" 作为能力证据 | 加注：从未在 L4 测过；OOD 只往下测过；trade-off 单旋钮 |
| 任何引用 RefBench-PRO Table 5 的地方 | "yes/no ≈ 随机" 或 "格式买 40–50 分" | 都不成立：该表无正样本侧 yes/no 率；InternVL3-8B 是 21.3 / 46.8 |

---

## 6. 不要做的

- 不要在 GME 上报任何不带灰图对照和文本地板行的 AUROC——benchmark 文本可分性 0.92。
- 不要把 p(null) 的阈值当方法——它在 GME 上抓 2/201、误弃 126/804；先掉的是 Limited/Small。
- 不要把 Rejection 与正样本维度放进加权总分；不要把 Discriminative 掉分当作"不是误 null"——没人测过。
- 不要用 Text 主导的负样本配比；不要只用短单从句负样本（k=0/1 是模型已半会的 regime）。
- 不要引用 CFCamo 的具体数字（仅摘要，两份读法不一致）、KoNA"零代价"（实际 −7）、TRAPSBench"75%"作为定向转向先例（它是对可答输入的无差别弃答）、Motto top-K 作为拒绝证据（N-Acc 在 K 上平）、SAM 3 head "−2.0" 作为设计依据（论文自注不可比）。
- 不要在找到 CAP-MECH §4 那条 lr÷10 的来源之前租 80 GB 卡跑 T1 的全参臂。
- 不要用策略同族的模型验证自己的负样本（`lineage.py` 已强制）；写手同族可以，但加一个 Gemma 写手臂比对产出。
- 不要在 GME 的 201 条上选 α / 阈值 / prompt——先冻结 100/101 的 dev/test 划分，或在 PR-Bench 上选。
- 不要下载 COCO / SA-1B 或跑属性编辑器（B 臂）而不先问。

---

## 7. 未解决 / 未核实

1. **子轴 vs 稀释度**（P1 + P10，≈0.7 GPU-h）和 **read-out vs 表示**（P5，≈1.9 GPU-h）——本轮最贵的两个未知数，也是最便宜能买到答案的。
2. GroundingME 自己 SFT 用的负样本构造（"modifying the description"）未公布；若是 L4 风格，27.9 就是域内数字。
3. gRefCOCO 规则 1 / 规则 2 比例（三篇论文都没有）。
4. Rejection 图上中心名词是否真的在场（D10）；Text 串在 P21 cap 下是否可读（D15）；255 条未审的 GME 移除的干净率。
5. Ideas3 报的"只有 6 张 Rejection 图与正样本共图"未被复核（与 1,005 条 / 879 张唯一图一致）。
6. RefBench-PRO 正样本侧 yes/no 率；PR-Bench Reject 的构造和接受规则；FINER 各级的纯负样本准确率（其 80→20 曲线在每从句技能不变时也会机械衰减为 q^(k+1)）。
7. 没有任何 2026 前沿模型（Gemini 3.x / GPT-5.x / Claude 5 / Seed 2.0）在任何拒绝 benchmark 上有公开数字；Gemini-2.5-Pro 是唯一同时在 GUI（OSWorld-G 38.9）和照片（GME 7.0）上拒绝的前沿模型，但正样本弱到无法与不合规区分。
8. 灰图前向下决策 token 的位置是否保持（前缀可能变）——P5 前用 20 条检查。
9. 各方案文件的 GPU-h 全是估计，基准是 1.4 s/条的 8B 决策 token 前向；自由生成按 2–3×，thinking 按 20–40×，eager attention 按 1.5–2× 计。
