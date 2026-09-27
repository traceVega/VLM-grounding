> **Revised in part by `REJECTION-DEEP-DIVE-2026-09-16.md` §4/§5** — §1.2's justification ("short clauses carry the signal") is withdrawn; negatives must span dilution k∈{0,1,2,4}, Text must not dominate, gate 4's load-bearing rate applies to the positive not to the flipped clause; S3 is kept as the control arm T0 of a shared-data A/B.

# RL 设计：教模型按从句筛选候选物体（2026-09-08）

目标：让模型学会**逐候选核对描述里的每个从句**，从而同时解决三件事——挑对那一个、说"没有"、说"有好几个"。

标注约定：**[E]** = 有实测证据支持（给出处）；**[D]** = 本文的设计选择，没人跑过，属于要验证的假设。

---

## 0. 支点：三个毛病是同一个操作

你们现在面对的三个数字看起来无关：

| 症状 | 现状 |
|---|---|
| 框放在错的物体上 | 338/804（其中 88.2% 与真值零重叠，框的大小还是对的） |
| 说不出"没有" | GroundingME Rejection **0.0**，25 个模型里 20 个恰好 0 |
| 多目标 | OpenRef Multi **35.5**（同模型 Single 74.7） |

**它们是同一个操作在 k=1 / k=0 / k>1 三种情况下的表现：**

> 对图里每个候选区域，逐条核对描述里的从句。通过的挑出来。
> 恰好一个通过 → 输出那个框
> 零个通过 → 输出空
> k 个通过 → 输出 k 个框

模型现在做的不是这个。它做的是"找一个长得像中心名词的东西"。**从句根本没进入决策。**

这个支点有两个后果：训练目标只需要教一件事；而且**验收测试是免费的**——只用 k=0 和 k=1 训练，去测 k>1，一条多目标数据都不喂。涨了，说明学到的是核对能力；没涨，说明只学了个弃答开关。**[D]，但这是本方案最强的单项证据，代价为零。**

---

## 1. 数据设计

### 1.1 起点：找"必须靠从句才能区分"的图

从无标注图像出发（COCO train / Objects365 train / OpenImages / SA-1B），用 SAM 3 拿实例。

**筛选条件：同类实例 ≥ 3 个。** 只有这种图，光说中心名词是歧义的，从句才必须承载信息。这是整个设计的地基——在只有一个杯子的图里，"红色的杯子"和"杯子"没有区别，模型走捷径也能对，那个样本教不了任何东西。

### 1.2 从句类型要对齐目标失败模式

GroundingME 的拒绝维度有四个轴，直接照抄：

| 类型 | 例子 | 对应 |
|---|---|---|
| **外观 appearance** | 蓝色的杯子 | Rejection/App |
| **部件 component** | 把手有缺口的杯子 | Rejection/Cmp |
| **文字 text** | 车身写着 "2500" 的车 | Rejection/Txt |
| **状态 state** | 倒着的杯子 | Rejection/Sta |
| **关系 relation** | 笔记本电脑左边的杯子 | Spatial/Relationship |

**优先造"离散可核对"的从句，不要造"渐变外观"的段落。** 这有你们自己的实测支持：在同一批 GroundingME 图上、同样的移除操作，**≤20 词的短从句表达式上弃答信号是 16 个数量级、AUROC 0.972、43% 弃答；>35 词的渐变外观段落上是 0% 弃答、AUROC 0.685。** 变量是表达式，不是场景。[E，`notes/CORRECTION-2026-09-05.md`]

也就是说：**"车身写着 2500"这种是可核对的，"淡蓝色玻璃质感的高层建筑"这种不是。** 先在可核对的regime里把能力建起来，再考虑要不要碰渐变描述。

### 1.3 一个源样本生成五种训练项

这是设计的核心。从一个 `(图, 目标实例, 表达式)` 出发：

```
P1  正样本，k=1
    图：原图        表达式：原句           答案：目标框

N0  零满足（改表达式）
    图：原图        表达式：翻转一个从句   答案：[]
    "蓝色的杯子" → "红色的杯子"（图里没有红杯子）

N0' 零满足（改图）
    图：把目标的属性改掉  表达式：逐字节不变  答案：[]
    把那个蓝杯子涂成绿色，句子一个字不动

M   多满足
    图：原图        表达式：删掉区分性从句  答案：所有同类框
    "笔记本左边的蓝杯子" → "杯子"

C   控制
    图：把非目标物体做同样的编辑  表达式：原句  答案：目标框（不变）
```

**为什么 N0 和 N0' 都要有。** 它们失效的方式不同，互相堵对方的捷径：

- N0 便宜、无限量，但翻转后的句子可能不自然，模型可能学到"句子怪 → 拒绝"。
- N0' 句子逐字节不变，不可能是文本捷径，但贵，而且编辑器会留痕。

**两者的差值本身就是一个诊断**：如果模型在 N0 上好、在 N0' 上差，它学的是文本启发式，不是视觉核对。[D]

**C 是必须的（对 N0' 而言）。** 它证明模型响应的是"目标的属性变了"，不是"这张图被编辑过"。你们在移除实验里已经跑过这个设计（REMOVE / CONTROL_OBJ），而且结果是干净的：**316/316 的控制编辑没有让框移动，0/316 误弃答。** [E，`notes/ABSTAIN-SIGNAL.md`] 属性编辑版本需要重新验一遍，但设计是现成的。

### 1.3b 第三条路：配对生成（不用编辑器）

不改真图，而是**用同一个种子、同一句生成提示词、只改一个词，生成一对图**。

```
生成提示词 A："a cluttered kitchen counter with five mugs, the leftmost one is BLUE"   → 正样本
生成提示词 B："a cluttered kitchen counter with five mugs, the leftmost one is GREEN"  → 负样本（问句仍问蓝的）
```

两个优点是编辑做不到的 **[D]**：

1. **真值免费且完美。** 场景是你生成的，每个物体在哪、什么属性你全知道。不需要 SAM 3 标实例，不需要听者验证框的位置。
2. **不需要 C 控制条件。** 编辑方案里负样本被动过而正样本没动，"这张图被编辑过吗"本身携带信息，所以必须有 C 去抵消。配对生成里**两张都是合成的，唯一差别就是那个属性**，痕迹不对称从根上不存在。

风险有两个，都是已知的：

- **合成到真实的迁移。** 有文献记录：CLEVR 类合成 benchmark 会饱和而不迁移到真图；加合成数据反而降低某些是非题分数；室内训练的空间推理不迁移到开放世界场景。[E，`team/landscape/skeptic-crowded-map.md` §5] 风险不只是像素真实度，是构图、光照、摆放和纹理统计的分布差异。
- **生成模型不擅长属性绑定和计数。** "五个杯子恰好一个是蓝的"正是扩散模型出名不可靠的地方。所以逐张验证跑不掉，只是从"验编辑对不对"变成"验生成对不对"——后者稍容易，因为有生成提示词当参照。

**结论：不要在编辑和生成之间二选一，把它们做成 S0 的两条臂，用数据回答。**

### 1.4 验证：这一步决定数据有没有用

**每一条都要过四关，不过就丢。** 这是"2000 条打赢 266 万条"的原因所在。

**关一：唯一性。** 用一个**不同族**的冻结听者（Molmo2-8B，其训练混合里没有 RefCOCO）从表达式定位目标。定位不到，或者定位到别的实例，说明表达式信息量不够，丢弃。

> 听者必须换族，否则你是在蒸馏自己的偏见。

**关二：零满足核对。** 对 N0 / N0' 项，把**每一个同类实例**都问一遍是非题："这个区域满足『<从句>』吗？"必须全部答否。任何一个说是，这条负样本是错的，丢弃。

**关三：难度门。** 原始基座必须在这条负样本上**输出一个框**（≥90% 的情况）。如果基座本来就会拒绝，这条样本教不了东西。

**关四：从句承重率。** 这一关是新加的，也是回答"图够不够复杂"的那个可测量判据。

一句 40 词的描述，如果只有 3 个词在起区分作用、其余 37 个是装饰，那它对模型来说等于一句短描述。GroundingME 难，是因为它的长描述里**多个从句同时承重**。

> **承重率 = 承重从句数 / 总从句数。**
> 把表达式拆成原子从句，逐个删掉一个，看跨族听者还能不能唯一定位到目标。
> 删掉后答案不变 → 那个从句是装饰。删掉后定位失败或落到兄弟实例 → 承重。

为什么这一关比"图像复杂度"更贴切：**驱动拒绝失败的是表达式长度，不是场景复杂度。** 同一个 8B 模型在 OpenRef 的负样本上 N3R 得 84.6，而 OpenRef 含无人机、暗光、恶劣天气图，图像并不简单。[E] 而你们自己的实测把变量钉死了——同一批图、同一个编辑器、同一个移除操作，只按长度分组：

| 表达式长度 | 弃答率 | AUROC |
|---|---:|---:|
| ≤ 20 词 | **43%** | **0.972** |
| 21–35 词 | 12% | 0.758 |
| > 35 词 | **0%** | 0.685 |

"变量是表达式，不是场景。" [E，`notes/CORRECTION-2026-09-05.md`]

但两者是有关联的：**要让从句承重，必须有足够多的同类干扰物**，让每一条从句都消掉一部分候选。图像密度是承重率的必要条件，不是充分条件——所以量承重率，不要量"复杂度"。

**目标：训练集的承重率分布覆盖 GroundingME 的分布。** GroundingME 的描述长度四分位是 18 / 40 / 58 词，实例面积四分位 0.16% / 1.0% / 2.7%。[E]

**关一到关三的必要性有直接证据**：GroundingME 的负样本和 FineCops-Ref 的 LLM 扰动负样本**都能造出接近完美的域内拒绝，也都在域外失效**——失败不是因为负样本太容易。[E，机制调研] 所以问题在验证，不在难度设计。

### 1.5 规模和配比

**目标 3,000 到 5,000 条已验证样本。不是 10 万条。**

| 证据 | 规模 | 结果 |
|---|---:|---|
| RC-GRPO | **2,000** | gRefCOCO Pr 38.2 → 64.9，N-acc 6.4 → 71.5 [E] |
| Jedi | **2,666,124** | OSWorld-G Refusal **7.4/100**，与零拒绝训练的模型同分 [E] |

差了三个数量级，结果反过来。**杠杆在目标函数，不在语料。** 另有 grounding 数据的规模拐点实测在约 10⁶ 元素处，之后 10 倍数据只值 1 到 3 分。[E]

配比 **[D]，正负约 1:1**：

| 类型 | 占比 |
|---|---:|
| P1 正样本 | 40% |
| N0 改句负样本 | 25% |
| N0' 改图负样本 | 15% |
| M 多满足 | 15% |
| C 控制 | 5% |

**不要用负样本占多数的配比。** GroundingME 自己的 2:1 实验：Rejection 到 27.9，代价是 Discriminative 61.3 → **40.2**、Limited 36.0 → **17.0**。[E]

另外配比调优本身不值钱——VLM-VG 扫了五种四路混合比例，RefCOCO 均值只在 49.4 到 51.3 之间。**真正值钱的是"表达式说什么"**：同一篇把零样本 RefCOCO 均值从 46.2 推到 58.7，靠的是加入空间关系从句，不是加量。[E]

---

## 2. 输出格式

**一个格式覆盖三种情况**，这样模型学的是一个操作，不是三个。

```json
{"boxes": [[x1,y1,x2,y2]]}        // k=1
{"boxes": []}                      // k=0
{"boxes": [[...],[...],[...]]}     // k=3
```

坐标沿用基座的 0-1000 归一化。你们已经验证过这条路径是干净的：**3,856 个发出坐标全部落在 0-1000 网格的精确整数上，最大残差 2.3e-13，0 个越界。** [E]

**注意这不是 GroundingME 的 `{"bbox_2d": null}` 格式，这是故意的。** 评测时做一次格式转换，而**这个转换本身就是协议交换测试**：

> 增益若在换成 benchmark 原生格式后存活，**且 Acc@0.75/0.9 上的提升不小于 Acc@0.5 上的**，就是目标选择能力，会迁移。
> 若只在你自己的格式里存在，或者 Δ(Acc@0.5) 远大于 Δ(Acc@0.9)，就是格式遵从，不会迁移。[E，该判据由 PointRL / VLM-R1 / RC-GRPO 通过，由匹配数据 SFT / GroundingME 负样本混合 SFT / LFPR 裁剪阶段证伪]

**要不要让模型先写逐候选核对再给答案？** 做成消融臂，不要做默认。理由：在 CoT 里强制写框往往不如纯文本 CoT；thinking 在感知重的维度上是负的（GroundingME Discriminative 每个尺寸都掉，8B 61.3 → 52.5）。[E]

---

## 3. 奖励函数

### 3.1 主体：集合级 F1

对一次 rollout 产生的预测集合 `P` 和真值集合 `G`：

```python
def reward(P, G, lam_refuse=0.5, lam_fmt=0.1, parseable=True):
    r_fmt = lam_fmt if parseable else 0.0
    if not parseable:
        return 0.0                          # 解析失败直接 0

    if len(G) == 0:
        r = 1.0 if len(P) == 0 else 0.0     # 该拒绝时：拒对得 1，吐框得 0
        return r + r_fmt

    if len(P) == 0:
        return 0.0 - lam_refuse + r_fmt     # 过度拒绝：负分

    # 匈牙利匹配最大化总 IoU，IoU >= 0.5 记一个命中
    m = hungarian_matches(P, G, thr=0.5)
    prec, rec = m / len(P), m / len(G)
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return f1 + r_fmt
```

**`lam_refuse` 是防胆小的那道闸。** 它让"该答不答"严格劣于"答错"。这正是 GroundingME 那次 SFT 没有的东西——SFT 只说"这些例子上输出 null"，模型没有任何途径学到*什么时候*，只能整体把先验往拒绝挪。

**格式奖励保持小。** GUI-G2 实测显式格式奖励值 −0.1，不是承重件，够解析就行。[E]

### 3.2 可选：严格 IoU 整形

把单阈值 F1 换成 IoU 0.5 到 0.9 上的平均 F1，形状对齐 mAcc 而不是 Acc@0.5。

但要盯着：**纯 IoU 奖励的 GRPO 会拉低 P@0.95**（Hi-Token 31.7 → 30.3，加了结构化多尺度奖励才回到 33.4）。[E] 建议做成 A/B 臂，不要默认开。

### 3.3 三个校准机制的具体实现

**机制一：强制 None rollout**

```python
# 在算优势之前，对负样本组做后处理
if is_negative(item) and all(len(p) > 0 for p in rollouts):
    idx = argmin(logprob)                 # 换掉最差的那个
    rollouts[idx] = {"boxes": []}
    forced_none_count += 1                 # 必须记日志
```

**为什么必须有。** GRPO 的优势是组内的 `(r_i - mean(r)) / std(r)`。负样本上如果 8 个 rollout 全吐框，8 个奖励全是 0，`std = 0`，优势为零或未定义，**这一组什么都学不到**。而基座在这类题上 pass@k ≈ 0（25 个模型 20 个恰好 0.0），所以这种情况是常态不是例外。[E]

**实现注意**：vLLM 没有约束束搜索，所以不能在解码时强制，只能在 rollout buffer 里替换。

**必须记 `forced_none_count` 的触发率。** 训练后期若仍在 >80% 的负样本组上触发，说明策略始终没学会自己采出拒绝，这是要停下来看的信号。

**机制二：负优势缩放**

```python
A = (r - r.mean()) / (r.std() + 1e-6)
A[is_negative] *= alpha        # alpha = 0.5
```

负样本的奖励是双峰的（0 或 1），优势天然大。不压制的话它会主导 batch，把策略拖向"一律拒绝"。α=0.5 是 RC-GRPO 的取值；**若正样本上的拒绝率超过 5%，把 α 降到 0.3。** [E]

**机制三：在线过度拒绝探针**

每 50 步在一个留出的正样本集上测拒绝率。**这是金丝雀。** 超过 5% 就说明正在往 GroundingME 那次 SFT 的坑里走，在跑完之前就该干预，而不是等最后看结果。[D，但是从那次失败反推出来的必要监控]

---

## 4. 训练配方

**基座：Qwen3-VL-4B-Instruct，本地。** 两个 4B checkpoint（各约 8.2 GB）能同时装进 32 GB；LoRA GRPO 配 co-located vLLM 占 28-31 GB。4B 通过之后再上 RunPod 的 8B。

**要不要 SFT 冷启动？** 证据两边都有：

| 证据 | 结论 |
|---|---|
| UniVG-R1：CoT-SFT 后 GRPO 64.50，纯 RL 57.43 | 冷启动值 +7.07 [E] |
| FineCops-Ref 拒绝：SFT-then-GRPO 58.2 Pr，纯 GRPO **62.5** | 冷启动有害 [E] |

调和两者的规律：**当 RL 必须学一个它采不出来的输出结构时，冷启动有用；当 SFT 先把策略压成退化模式时，冷启动有害。**

你们这里正好在边界上：模型已经会发 `{"boxes": [一个框]}`，但**采不出 `[]`**。所以：

> **[D] 跑一个 500 条的纯格式 SFT**，只教 JSON 列表格式（含空集），不教判断。然后进 GRPO。跑之前先测一下策略能不能自己采出 `[]`——能采出来就跳过这步。

**超参**（照 RC-GRPO 的配方）：

| 项 | 值 |
|---|---|
| LoRA rank | 32-64 |
| rollouts / prompt | 8 |
| 步数 | 每阶段 500 |
| batch | 8 |
| KL 惩罚 | **不要**（三个独立来源一致：不防遗忘，只影响稳定性，还拖慢适应）[E] |

**另外加一个 lr=1e-6 的全参臂，同步数。** 实测：全参微调学习率降 10 倍，目标任务只掉 0.19pp，却消掉 17.6/32.1pp 的域外回归——**在同一张表里两个轴都赢过 LoRA**。几乎免费，可能直接占优。[E]

**两阶段**：
- **阶段 I**：上面的集合 F1 + 拒绝校准
- **阶段 II**：加对比项，明确奖励模型区分最小对（同图的 P1 vs N0'），逼它注意到被翻转的那个从句

---

## 5. 评测设计（这是一半的工作量）

### 5.1 主指标与护栏

**主**：GroundingME Rejection（n=201，CI ±6.2，所以**低于约 13 都和零区分不开**）。

> **每个 Rejection 数字旁边必须写解析失败率。** 官方 `evaluate.py` 把无法解析的输出当成正确拒绝。不写这个数，这一列不可解读。[E，读过源码]

**护栏，三条同时成立才算过**：

| 项 | 阈值 |
|---|---|
| GroundingME 非拒绝维度 | 掉幅 ≤ 2.0 |
| RefCOCOg val | 掉幅 ≤ 1.0 |
| **DocVQA、AI2D** | 掉幅 ≤ 2.0 |

**通用能力套件要换。** MME / MMBench / POPE 是最不敏感的三列。实测：同一批 SFT checkpoint 在 CoT 提示下 MMMU −12.0、无 CoT −8.8，POPE −5.2 vs −0.7——**仅评测提示就让"遗忘"量摆动 7 倍**。[E] 加 DocVQA、AI2D、MathVista、CharXiv。

### 5.2 五个迁移测试（分辨能力和刷榜的地方）

1. **元数泛化（最强，且免费）**。只用 k=0 和 k=1 训练，测 OpenRef Multi。一条多目标数据都不喂。涨了就说明学到的是逐候选核对。
2. **协议交换**。训练用 `{"boxes": [...]}`，评测走 GroundingME 的 `{"bbox_2d": null}`。增益必须存活，且 Δ(Acc@0.75/0.9) ≥ Δ(Acc@0.5)。
3. **负样本类型迁移**。训练用 COCO/Objects365 的编辑；测 GroundingME Rejection——那是**未编辑的** SA-1B 照片、人写的 39 词描述、物体从来就不存在。图源、负样本机制、撰写团队三者全不同。
4. **N0 与 N0' 的差值**。差得多说明学的是文本启发式。
5. **灰图对照**。把图换成纯灰再跑一遍。**在盲态下仍然存活的增益不是 grounding 增益。** 一次额外前向的代价。已经抓到过实例：某个已发表的训练无关"修复"在盲态下 +19、在正常视觉下 +21。[E]

### 5.3 套件一致性检查

训练后重算 GroundingME / Ref-Adv / Ref-L4 / OpenRef 之间的秩相关。**掉了就说明增益是榜不是能力。** grounding 专门化的后训练实测平均把 ρ 从 0.82 打到 0.20（p=0.021）。[E]

按经验预算 **3 到 10 倍衰减**：自己管线上涨 X 分的方法，到独立 benchmark 上预期只剩 X/3 到 X/9。

---

## 6. 分阶段执行与停止规则

| 阶段 | 内容 | 代价 | 停止规则 |
|---|---|---:|---|
| **S0** | 可行性探针：200 条手工/半自动样本 | ~2 GPU-h | 见下 |
| **S1** | 造 3-5k 条已验证样本 | 15-30 GPU-h | — |
| **S2** | 500 条格式冷启动 SFT | ~1 GPU-h | 策略能自采 `[]` 则跳过 |
| **S3** | 阶段 I GRPO（4B LoRA） | 10-15 GPU-h | 三条护栏同时成立 |
| **S4** | 元数泛化测试 | ~1 GPU-h | 免费的能力判据 |
| **S5** | 阶段 II 对比训练 | 10-15 GPU-h | 仅 S3 通过后 |
| **S6** | 8B，RunPod | 20-30 H100-h | 仅 4B 通过后 |

本地合计约 **40 到 65 GPU 小时**，一张 5090 上两周左右。

### S0 的四个门

完整的实施规格（数据量、模型、提示词、infra、显存核算）在**第 9 节**。四个门：

1. **头部空间**：基座在负样本上输出框的比例 ≥ 90%？（本来就会拒绝 → 没得教）
2. **表达式质量**：跨族听者唯一定位正样本的比例 ≥ 80%？（否则生成器不合格）
3. **承重率**：中位承重率 ≥ 0.6，且长表达式（>35 词）的承重从句数中位 ≥ 3？（否则你的长句是装饰，教不出 GroundingME 那种难度）
4. **数据源质量**：三条臂各自的产出合格率。编辑臂人工检查 50 条，生成臂自动 + 抽检。

> 你们移除实验的实测干净率是 **46%**，属性交换文献报 60-80%，**小目标是公认弱项**。[E] 若编辑臂在关键尺寸区间低于约 50%，砍掉 N0'，走改句臂和生成臂。这是真实分叉，不是形式主义。

### S3 的停止规则（三条同时）

- GroundingME Rejection **≥ 10.0**
- 任何非拒绝维度掉幅 **≤ 2.0**
- RefCOCOg val 掉幅 **≤ 1.0**

若拒绝涨了但正样本掉超过 2：把负样本比例减半，重试一次。再不行就关掉这条杠杆——**这本身是个可发表的结论**，因为它会是第一个带完整护栏的否定结果。

---

## 7. 已知的五个失败模式

1. **编辑器在小目标上不行。** 有文献记录的弱项，而 GroundingME 的 Limited-Small 正好在这个区间。S0 第 3 门专门查这个。
2. **负样本太容易。** 翻转后的从句如果荒谬得明显，模型学到的是废话。S0 第 1 门（难度门）挡这个。
3. **模型学到"图被编辑过 → 拒绝"。** 控制项和 N0/N0' 差值测这个。
4. **正样本还是崩了。** 过度拒绝惩罚和 α 缩放是防御，但不保证。在线金丝雀让你在跑完之前就发现。
5. **有效但不迁移。** 那你得到的是一个 benchmark 结果，而元数泛化测试会诚实地告诉你。这也是可发表的。

---

## 8. 为什么值得押这一个

**这是前沿实验室用蛮力明确失败过的一格。** Jedi 造了 266 万个合成拒绝样本，在 OSWorld-G Refusal 上得 7.4 分，**和完全没做过拒绝训练的 OS-Atlas-7B 同分**。三代模型、六个尺寸、每一家的 GroundingME Rejection 都恰好是 0.0——**没有任何实验室在代际之间改动的东西碰到了硬负样本拒绝。** [E]

而有效的解法只要 2,000 个样本、两张卡。**并且造这批数据的机器只有你们有**——你们的 K1 门控编辑流水线加上已经验证过的 REMOVE/CONTROL 对照设计，是现成的。

---

## 9. S0 实施规格

### 9.0 端到端管线（A 臂，六个阶段，五次模型驻留）

阶段之间只通过 parquet 传递，任何时刻只有一个模型在显存里。这是仓库既有的模式（`judges.yaml`：先跑策略、落盘、再换验证器进来）。

```
S0.0  选场景                                        [无 GPU]
      in   已有 instances.parquet（9,692 场景 / 311,282 实例行）
      op   n_head_noun_instances >= 3
           面积分位对齐 GroundingME（0.16% / 1.0% / 2.7%）
           pHash 排除所有评测集
      out  scenes.parquet            200 行

S0.1  写表达式 + 翻转从句 + 拆原子从句              [Qwen3.5-9B  19 GB]
      in   scenes.parquet + 目标画红框的图
      op   expr_write_clausal   -> expr           （n_clauses 条可核对从句）
           expr_flip_clause     -> expr_neg       （只改一条，且合理不荒谬）
           expr_decompose       -> clauses[]      （原子从句列表）
      out  expressions.parquet       200 行
           clauses.parquet           约 600 行

S0.2  门2 唯一性 + 门3 承重率                       [Molmo2-8B  17 GB]
      in   expressions + clauses
      op   完整 expr 定位一次        -> unique_hit（点落在目标掩码内且不在兄弟实例内）
           逐条删一个从句再定位      -> load_bearing[i]
      out  listener.parquet          200 + 600 行

S0.3  零满足核对                                    [Gemma4-12B  24 GB]
      in   expr_neg 的 changed_clause × 每个同类实例的 crop
      op   verifier_clause_instance -> yes / no / unclear
           全部为 no 才保留这条负样本
      out  verify.parquet            约 800 行

S0.4  门1 头部空间                                  [Qwen3-VL-4B  9 GB]
      in   正样本和负样本各 200
      op   grounding_qwen3vl_primary（GroundingME 原生 prompt，逐字节不改）
      out  policy.parquet            400 行

S0.5  汇总判定                                      [无 GPU]
      out  tables/s0_gates.md        四个门的数字 + 95% CI
```

**B 臂**：在 S0.0 和 S0.4 之间插一个编辑阶段（编辑器 + SAM 3，约 12–18 GB），产出 N0' 和 C 两种条件，其余不变。50 张，约 90 分钟。

**C 臂**：S0.0 和 S0.1 换成生成（生成器，约 8–16 GB），种子固定、只改一个词出一对图。**但 S0.2 到 S0.4 一条都不能省**——生成模型的属性绑定不可靠，"五个杯子恰好一个蓝"必须逐张验。C 臂省掉的是编辑成本和 C 控制条件，不是验证。

### 9.0b 为什么是三个模型，不是一个

四个角色，**每个角色的错误会造成不同的坏结果，而相关的错误是最贵的那种**：

| 角色 | 模型 | 谱系 | 它出错会怎样 |
|---|---|---|---|
| **写手** | Qwen3.5-9B | qwen | 表达式有歧义或说错 → 被听者挡下 |
| **听者** | Molmo2-8B | molmo | 好表达式被误弃（浪费）或坏表达式漏过（污染数据） |
| **核对者** | Gemma4-12B | gemma | **无效负样本漏过 → 训模型在该答时拒绝**，直接制造过度拒绝 |
| **被测策略** | Qwen3-VL-4B | qwen | 就是你要训的那个 |

三条不能合并的理由：

**一、写手不能自己检查自己。** Qwen3.5 写了"左边那个蓝杯子"，再问 Qwen3.5"能找到左边那个蓝杯子吗"——它当然说能，它就是看着那个杯子写的。这个检查什么都没验证，只验证了"Qwen3.5 同意 Qwen3.5"。仓库的 `lineage.py` 已经把这条写成加载时强制的规则。

**二、策略不能造自己的训练数据。** 如果 Qwen3-VL-4B 写表达式，它的盲区会原样成为数据集的盲区——**它描述不出来的情况就不会进训练集，而那恰恰是你要修的情况**。核对同理：策略验自己的负样本，它检测不出的负样本会被当成"无效"丢掉，等于系统性地筛掉最难的样本。

**三、听者和核对者也要分开，因为它们的错误方向相反且会互相掩盖。** 同一个模型看不见的属性，会同时让听者定位失败（该留的被丢）和让核对者误答"无实例满足"（该丢的被留）。分成两个族，这两类错误就不相关了。

**Molmo2 被选为听者还有一个特定理由**：它的公开训练混合里**没有 RefCOCO**。所以它对"这句话能不能唯一定位"的判断，不是从 REC 数据里背下来的。[E]

**成本几乎为零**：三个模型你们全都下载并 pin 好了（`judges.yaml` 带 revision，`serve.sh` 和缓存 client 都在）。多用一个的边际代价是**每阶段约 2 分钟的换模型时间**，不是下载或搭建。

**可以退到两个吗？** 可以：把核对者合进听者（都用 Molmo2）。省一次驻留，代价是上面第三条的相关错误。**建议先按三个跑 S0，在报告里记录核对者和听者的分歧率**；分歧率很低就说明合并是安全的，S1 再合。这是可测量的决定，不用先争。

### 9.1 三条臂

| 臂 | 负样本怎么来 | 主要风险 | 需要新下载 |
|---|---|---|---|
| **A 改句** | 真图，翻转一个从句 | 文本捷径（句子变怪就拒绝） | 否 |
| **B 编辑** | 真图，改目标的属性，句子不动 | 编辑痕迹、小目标编不动（实测干净率 46%） | **是**（指令式图像编辑模型） |
| **C 生成** | 配对生成，同种子只改一个词 | 合成到真实不迁移 | **是**（文生图模型） |

三条臂共用同一批表达式模板、同样的四个门、同样的 200 条规模。A 臂零新增依赖，先跑。

### 9.2 数据量

**每条臂 200 个源场景。** 每个源场景产出 1 正 + 1 负 = 400 次 grounding 调用。

各门的统计功效（二项正态近似）：

| 门 | n | 判据 | 95% CI 半宽 |
|---|---:|---|---:|
| 1 头部空间 | 200 | ≥ 90% | ±4.2% |
| 2 听者唯一性 | 200 | ≥ 80% | ±5.5% |
| 3 承重率 | 200 表达式 × ~3 从句 = 600 次消融 | 中位 ≥ 0.6 | — |
| 4 编辑干净率 | **50**（人工） | ≥ 50% | ±13.9% |

门 4 只要 50 条，因为它花的是人力不是算力，而 ±13.9% 足够把 50% 和 25% 分开。**B 臂在 S0 只编辑 50 张**，过了门 4 再补到 200。

后续（不属于 S0）：S1 造 3,000 到 5,000 条已验证样本，配比见 1.5 节。

### 9.3 用什么模型

**被测基座：Qwen3-VL-4B-Instruct**（不是 8B）。五个理由：

1. **两个 4B checkpoint 能同时装进 32 GB**，8B 那一对装不下。后面 Instruct/Thinking 路由实验要用到。
2. **4B 的 Thinking 增益更大**（Ref-Adv-s 上 +15.7 vs 8B 的 +12.3），检验功效更好。[E]
3. **4B LoRA GRPO 本地放得下**（28 到 31 GB），8B 放不下。
4. **有可复现的参照值**：GroundingME 论文 Table 3 给了 4B 的 33.9。[E]
5. 剩下的显存够和 SAM 3 共驻。

> 需要新建 `configs/models/qwen3vl-4b-instruct.yaml`，并重跑 Q-3 坐标约定探针。不能假设 4B 继承 8B 的 `relative_1000`。Qwen2.5-VL 到 Qwen3-VL 就换过一次约定且没有 ablation。复用 `tables/q3_coordinate_probe.json` 的方法：同一张图两个 max_pixels，看坐标动不动。

**其余角色全部已 pin，配置已存在：**

| 角色 | 模型 | 谱系 | 显存 |
|---|---|---|---|
| 写表达式 + 翻转从句 | judge_a `Qwen3.5-9B` | qwen | 19 GB |
| **听者**（唯一性、承重率） | `Molmo2-8B` | molmo | 约 17 GB |
| 逐实例是非核对 | judge_b `Gemma4-12B` | gemma | 约 24 GB |
| 实例分割 | SAM 3 | 无 | 约 5 GB |
| 移除类编辑 | big-LaMa | 无 | 小 |

**谱系约束**（`shared/judges/lineage.py` 会在加载时强制）：听者必须与被测策略不同族。Qwen3-VL-4B 是 qwen 族，所以**听者不能用 judge_a**，必须是 Molmo2。写手用 judge_a 可以，因为验证它的是 Molmo2，写手与验证者不同族。若想要写手、听者、策略三方全不同族，写手改用 judge_b。

### 9.4 需要的 prompt

沿用 `configs/prompts/` 的格式（`# status:` / `# role:` / `# fields:` 头，内容哈希进 manifest）。

**复用现有的三个，一个字都不要改：**

- `grounding_qwen3vl_primary.txt` 被测的那个（VERIFIED，与 GroundingME 的 evaluator 逐字节相同）
- `pointing_molmo2_primary.txt` 听者
- `verifier_headnoun.txt` / `verifier_expression_full.txt` 核对的起点

**新增五个：**

**(1) `expr_write_clausal.txt`** 写表达式

```
# status: DRAFT
# role: clausal expression writer (arm A/B source expressions)
# fields: category, n_siblings, n_clauses
The image contains {n_siblings} objects of the category "{category}". One of them
is outlined in red.

Write one referring expression that identifies the outlined object and no other.

Requirements:
- Use the head noun "{category}" plus exactly {n_clauses} distinguishing clauses.
- Every clause must be independently checkable by looking at one region and
  answering yes or no. Good: "has the number 2500 on its side", "is lying on its
  side", "has a chipped handle", "is to the left of the laptop". Bad: "looks
  elegant", "is the nicest one", "seems out of place".
- At least one clause must describe the object itself (appearance, a component,
  text on it, or its state) rather than only its position.
- Every clause must be true of the outlined object and false of at least one
  other object of the same category.
- Do not use ordinal position alone ("the third from the left") as the only
  distinguishing clause.
- Output the expression only, on one line, with no quotation marks and no
  explanation.
```

**(2) `expr_flip_clause.txt`** 翻转一个从句

```
# status: DRAFT
# role: single-clause negation for zero-satisfier negatives
# fields: expr, category, n_siblings
This expression describes one object in an image:
{expr}

The image contains {n_siblings} objects of the category "{category}".

Rewrite the expression so that it describes NO object in the image, by changing
exactly one clause and nothing else.

Requirements:
- Change exactly one clause. Every other word must be identical, including
  punctuation and word order.
- The replacement must be PLAUSIBLE for this category and this kind of scene: a
  property such an object could easily have, but that none of the objects in
  this image actually has. Change "blue" to "red", not to "invisible".
- The result must be a grammatical, natural sentence that a person could have
  written in good faith.
- Do not negate with "not", "no" or "without". Substitute a different value.
- Output two lines: first the rewritten expression, then the single clause you
  changed, in the form  CHANGED: <original clause> -> <new clause>
```

> "PLAUSIBLE 而非荒谬"这一条是门 1 通过与否的关键。荒谬的翻转会让模型靠"句子离谱"来拒绝，那学到的是废话。

**(3) `expr_decompose.txt`** 拆原子从句（承重率用）

```
# status: DRAFT
# role: atomic clause decomposition for the load-bearing test
# fields: expr
Split this referring expression into its head noun and its atomic distinguishing
clauses:
{expr}

Rules:
- One clause per line. Each line must be a single checkable predicate.
- Keep the wording verbatim from the expression; do not paraphrase.
- The head noun goes on the first line, prefixed "HEAD: ".
- Each clause line is prefixed "CLAUSE: ".
- A conjunction of two independent properties is two clauses.
- Output nothing else.
```

**(4) `verifier_clause_instance.txt`** 逐实例核对（零满足检查用）

```
# status: DRAFT
# role: per-instance clause verification (zero-satisfier certification)
# fields: clause
Look only at the region outlined in red.

Does this statement hold for the outlined object?
{clause}

Answer with exactly one word: yes, no, or unclear.
Answer "unclear" only if the region is too occluded or too small to judge.
```

**(5) `gen_scene_pair.txt`** C 臂配对生成

```
# status: DRAFT
# role: paired scene generation spec (arm C)
# fields: category, n_instances, attr_name, attr_pos, attr_neg, setting
A photograph of {setting}. It contains {n_instances} {category} objects of
similar size, arranged naturally and partly overlapping. Exactly one of them
{attr_name} is {attr_pos}; all the others differ in this property.
Photorealistic, natural lighting, cluttered background, shot on a DSLR.
```

> 正负两张只把 `{attr_pos}` 换成 `{attr_neg}`，**种子、步数、CFG、其余每个词全部不动**。这就是配对的全部机制。

### 9.5 Infra：绝大部分已经有了

**可以直接复用（已建成、已 pin、已测）：**

| 现有件 | S0 里做什么 |
|---|---|
| `idea91/instances/` + `store.py` | 场景和实例。`StoredScene.n_head_noun_instances` **已经是字段**，"同类实例 >= 3"直接就是一个 parquet 查询 |
| **9,692 个已建实例的 OpenImages 场景**（3.6 小时建好，311,282 实例行） | **A 臂的选场景步骤是零 GPU**，直接查已有的 bank |
| `shared/harness/prompts.py` | 提示词版本化加哈希，`UNVERIFIED` 不能进 kill run |
| `shared/harness/model_config.py` | `PIN_REQUIRED` 强制，4B 配置补齐前跑不了 |
| `shared/harness/parsers.py` | 两种坐标约定都已定妥（Q-1、Q-2） |
| `shared/judges/` client 加 `lineage.py` 加 `serve.sh` | 谱系规则在加载时强制，一次只服务一个判官 |
| `schema.assert_dev_slice_disjoint` | pHash 排除，防污染 |
| `idea91/edits/`（window/composite/sampler/inpaint） | B 臂的编辑骨架，`assert_in_mask` 等检查现成 |
| `idea91/human/app.py` | **门 4 的人工检查 UI 现成**，本地绑 127.0.0.1，答案键不下发，顺序按标注者打乱 |
| `freeze.py` | 冻结阈值，防止事后改判据 |

**需要新写的（delta）：**

| 新件 | 内容 | 工作量 |
|---|---|---|
| `configs/models/qwen3vl-4b-instruct.yaml` | 新 pin 加 Q-3 探针 | 半天（含 GPU 探针） |
| 上面五个 prompt 文件 | 见 9.4 | 半天 |
| `select.py` | 场景筛选：同类实例 >=3、面积分位对齐 GroundingME | 半天（纯 parquet） |
| `expressions.py` | 写手、翻转、拆解三个调用的封装 | 一天 |
| `certify.py` | 四个门做成流水线，逐条打标签 | 一天 |
| `clause_ablation.py` | 承重率 | 半天 |
| `edits/attribute.py` | **属性编辑后端**（现有 `inpaint.py` 是 big-LaMa，只会移除） | 需下载 |
| `generate.py` | C 臂配对生成，种子固定 | 需下载 |
| `run_s0.py` | 分阶段驱动：显式 load/unload、可续跑、按阶段落盘 | 一天 |

> **两个新下载是决策点，不是默认。** 按仓库惯例（backend 契约，以及"备用后端和下载要先问"），属性编辑模型和文生图模型都需要先定型号、先问过再拉。**A 臂完全不依赖这两个，可以立刻开跑。**

### 9.6 本机能不能跑：能，但必须一次只驻一个模型

这条约束你们已经写进设计了。`shared/judges/judges.yaml` 里就有："32 GB cannot hold a policy and a judge together，所以 K2 先跑策略、落盘、再换验证器进来。" S0 照同样的模式。

**显存（32 GB VRAM），逐阶段：**

| 阶段 | 常驻模型 | 显存 | 余量 |
|---|---|---:|---:|
| S0.1 选场景 | 无（parquet 查询） | 0 | 无 |
| S0.2 写表达式加翻转 | Qwen3.5-9B | 约 19 GB | 13 |
| S0.3 门 1 头部空间 | Qwen3-VL-4B | 约 9 GB 加 KV | 大 |
| S0.4 门 2 唯一性 | Molmo2-8B | 约 17 GB | 15 |
| S0.5 零满足核对 | Gemma4-12B | 约 24 GB | 8 |
| S0.6 门 3 承重率 | Molmo2-8B | 约 17 GB | 15 |
| S0.7 B 臂编辑 | 编辑器加 SAM 3 | 约 12 到 18 GB | 中 |
| S0.8 C 臂生成 | 生成器 | 约 8 到 16 GB | 中 |

**每个阶段单独都放得下，没有任何阶段需要共驻。** 阶段间换模型约 1 到 2 分钟，六次切换约 10 分钟，可忽略。

**系统内存（WSL 下约 23 GB 可用）才是真正的风险。** 你们的记忆里已经有这一条：超了发行版会被回收，连带杀掉后台任务，而且 `/proc/uptime` 看不出来。

三条硬规则写进 `run_s0.py`：

1. **流式读图，绝不批量解码。** 同时解码的图上限设成 8，用信号量卡住。
2. **解码后立刻缩放**到该阶段的 `max_pixels` 以下再进后续处理。一张 7680x5120 的 RGB 图 uint8 就是 118 MB，float32 是 472 MB，十张就把你顶穿了。
3. **中间结果落 ext4**（`~/vlmg-data`），不留在内存。每个阶段写自己的 parquet，下一阶段读它。

好消息：你们那 9,692 场景的实例 pass 就是这么跑的，3.6 小时没死，**流式模式是已验证的**。另外 C 臂的图是 1024 px 的，反而是三条臂里最省内存的。

长任务照旧用 `wsl -d Ubuntu`。

### 9.7 时间预估

用你们自己实测的速率，不是拍的：

| 步骤 | 依据 | A 臂 | B 臂 | C 臂 |
|---|---|---:|---:|---:|
| 选场景 | 实例 bank 已有，parquet 查询 | **0** | 0 | 不适用 |
| 写表达式加翻转 | 200 次长输出调用 | 10 min | 10 min | 10 min |
| 门 1 头部空间 | 400 次 grounding，约 1 s（8B 实测 1.4 s） | 7 min | 7 min | 7 min |
| 门 2 唯一性 | 200 次 pointing | 5 min | 5 min | 5 min |
| 零满足核对 | 200 负 x 约 4 同类实例 = 800 次 crop-VQA | 15 min | 15 min | 15 min |
| 门 3 承重率 | 600 次消融 | 10 min | 10 min | 10 min |
| 造图 | B：50 编辑，约 33 对每 GPU 小时；C：400 张，3 到 6 s | 无 | **90 min** | **30 min** |
| **合计** |  | **约 50 min** | **约 2.3 h** | **约 1.3 h** |

**三条臂全跑约 4.5 GPU 小时**，加门 4 的人工检查（50 条，用现成的标注 UI）。

**建议的开跑顺序**（按依赖和成本）：

1. **先把 4B 配置和 Q-3 探针做掉**（半天）。没有它后面每个数字都是悬的。
2. **只跑 A 臂**（50 分钟，零新依赖）。它同时回答门 1、门 2、门 3，也就是"COCO/OpenImages 够不够复杂"这个问题的可测量版本。
3. **看门 3 的承重率。** 不达标就换图源（Objects365 或 SA-1B，注意 SA-1B 要 pHash 排除 GroundingME 那 1,005 张），不要急着上编辑器和生成器。
4. **门 1 到 3 都过了，再决定要不要拉那两个模型**，跑 B 和 C。

这个顺序的意义：**A 臂的 50 分钟就能告诉你数据源合不合格**，而那正是你三个问题里的第一个。编辑和生成的钱要等这个答案出来再花。
