# 方案审核综合报告（提议 → 逐候选观察与判断 → 模型自答，一致性门奖励），2026-09-26

五个独立审核视角（奖励钻空子、忠实性与通用性、数据与标签、评测与统计、实现与成本）+ 综合裁决。

**审核综合报告（供今日决策）**

## 1. 总判决

**有条件批准。** 五位评审一致认为方向正确（逐候选独立裁定、先写观察值再判断、由模型自己给最终答案），但按现稿直接跑，标签、提示词、奖励三处都会把"看图"这条路重新绕开，算力估计低 2–6 倍；改完第 2 节 7 条阻断项、零训练探针过线后再进 SFT。

五个视角的判决：reward-hacking 有条件批准；faithfulness-generality 有条件批准；data-labels 有条件批准；evaluation-statistics 有条件批准；implementation-cost 有条件批准。

我对评审引用的代码做了复核，均属实：`configs/prompts/grounding_coa_audit.txt` 含 "{k} of {n}"、[type] 标签与"relation 永不拒绝"；`train/coa.py` render_audit 第 113–118 行仍跳过 relation、第 111 行用 claim 回填 seen，而 named_mismatches 以 EXEMPTIONS=False 运行；`train/grpo_lora.py` 第 99 行 s_pos 为 None 时 base=1.0，`--r-pos-null` 默认 −0.5，第 526 行 need_pos 要求 matrix；`train/eval_suite.py` _coa_turn2 的 sub_batch 硬编码 8，且把 COA.decide 直接写成 <answer>；`shared/harness/tokens.py` MAX_PIXELS=2,457,600；SFT 与 eval 的 audit 视图均为 [原图, 描红图, 裁片] 三图。

## 2. 阻断级问题（必须改才能跑）

**B1 标签与推导规则自相矛盾**（3 方）。机制：render_audit 对 relation 免拒、无一致 rationale 的 no 格回填 claim（echo），而推导规则不免 relation、忽略 echo；verdict 与 decide() 不一致的行 327/4,622（另一口径 83/1,012），5–9/240 项的标签推导答案≠标签答案；提示词文件仍带 [type]/relation 规则，与 v2 文本不同。修法：删 relation 跳过与 [type]；无反值的 no 格渲染为 unsure；重渲后断言 verdict=='fits' ⇔ named_mismatches 为空、首个 fits 候选 == 标签答案，100% 通过否则丢行/项并打印计数；提示词文件改成 v2 文本，SFT 与 eval 共用。

**B2 名次泄露**（3 方）。机制：SFT 中描述对象永远 k=1，k=1 行 fits 42%，k≥2 行 99.9% 被排除，提示词又告诉模型 k——Night-4 "22/23 错选都在第 1 行"从表格搬进了提示词。修法：删 "{k} of {n}"；audit 样例按场景种子随机序；探针把 GT 候选误控率按名次拆分，rank1 与 rank≥2 差 >10 点即未除。

**B3 不看图也能拿空答案分**（3 方）。机制：K=0（b400 上已有 3–12/99 正例）、单个诱饵框、全行 mismatch 三种都过一致性门并得 +max(s+,0.5)；158 个 RL 负例中 118 个（cross/alt/mosaic）无 matrix、无 flip、无孪生，代码路径付 1.0，比孪生负例还高；+0.5 指控只查行位不查 seen，"逐行全指"稳拿。修法：K=0 → 格式 −1；有标注实例的负例，null 分仅当某候选 IoU≥0.5 命中该实例、其 audit 恰一行 mismatch、落在对齐子句、seen 含 flip_from 实词；无标注负例 K≥1 且封顶 0.5，孪生负例占负例 ≥ 一半（alt@20、cross@10、去 mosaic 负例）；注入触发改为"组内无人得 outcome 分"。

**B4 "seen≠claim"是空规则**（3 方）。机制："not visible"/"different" 都通过；+3.8..+11.8 的证据来自要求 not/instead/without 的 CONTRAST 正则，套到 isolated 运行只有 24/35/17/6.8，低于 V3d 的 18.9。修法：推导侧去掉否定词与停用表后 seen 实词 ⊄ claim 实词才算 named；奖励侧用 names_value 校验 flip_from。

**B5 K 个 audit 的 AND 封顶拒绝率**（1 方，faithfulness 呼应）。机制：null 需每个候选都被点名，rej ≤ d·q^(K−1)；GME 上 q≈0.6–0.8、K̄≈3.5，q=0.7 时 d=0.8 也只到 33，q=0.6 封顶 22；正例奖大 K，负例不罚，K 只涨。修法：探针存全部 audit，离线按前 K 个候选重推 K=1..6；某 K 达标才继续 K>1，否则退到 K=1（同一 audit 头接 base hint）。

**B6 答案步与 RL 路径不存在**（4 方）。机制：eval 把 decide() 当答案，consistent≈0.98 是同义反复；expand_coa 无答案样例；sampler/reward/注入全是表格版；"追加在最后一个 audit 后"不可行。修法：固定为第三次纯文本调用（turn-1 文本 + K 个 audit 块 + 框坐标，无图）；SFT 加 _coa_k=−1 样例；写 sample_coa / coa_reward / inject_coa。

**B7 算力低估 2–6 倍**（5 方）。机制：每个 audit 带两张 2.46 MP 满图（各 ~2,400 视觉 token）+ 裁片，K=4 每 rollout 9–12.6k token、K+2 次前向；旧协议实测 4.1–4.3 min/step → 8–20 min/step。修法：audit 上下文只 [描红满图, 原生裁片]（build_example_coa 与 _coa_turn2 同一提交改）；RL rollout K≤4；GME 上 sub_batch 4；首个 RL ≤25 步。

## 3. 重要问题（应改）

**M1 指控项需分段优势**：标量优势摊到 500–900 token，信号 ≈1/(K·J)，grpo_v5_obs 60 步 neg_ev 停在 0.05–0.15。segs 已分离：outcome 优势全段，accusation 优势只落在命中标注实例的 audit 段（~20 行改动）。

**M2 一致性门是对称的**：几步内 ≈100%，"不上升"无信息。改为每步记录标签校验指标：任一 IoU≥0.5 命中标注行的 audit 在 label-yes 格的误控率、flip 格命中率、unsure 率（负例 flip 格 unsure >30% 视作漏控）；−0.5 扩到任何命中标注行的误控（正负例皆然，每 rollout 一次）；kill 改为一致性低于 SFT−5。

**M3 SFT 数据结构错位**：GME 型行（对的对象、恰一处错）仅 357/4,622（7.7%，分半后 ~180）；非目标行 85% 过半 mismatch、39% 全 mismatch；2.1× 重复；330 行 echo-no、40 条肯定式 rationale、~30% match 行照抄；"<claimed detail> |" 占每行一半 loss。修法：clause_bank 零 API 重组单错行（341 实例、2,289 行），单错行 ≥40%；全 mismatch 行每项 ≤1；按（实例,条件,verdict）去重；丢 echo-no 与肯定式行；≥5 词 flip 用词级 diff；mask 照抄段；用 rationales_gemini 并丢跨 ≥3 子句重复句；audit 样例框 50% 抖动到 IoU 0.5–0.8。

**M4 接受漂移**：正负 1:1（记录实际比）；--r-pos-null 显式设 0；每 20 步记录按类均 K、注入比例；均 K 涨 >1.0、own neg-null 低于 SFT 10 点、注入 >40% 步即停。

**M5 mini-GME 作停机 = 在测试集上选点**，且 n=20 SE≈10。停机只看 own-val 48 + own gray 48 + 训练批统计；GME 曲线只记不用；全集另报从未筛查的 806 项（101 rej + 705 pos）。

**M6 筛查正例偏高 ~5 点**（base hint 正确 44.4% vs 全集 39.9%；V3d 43.4 vs 38.3）：turn-1 单遍测 804 正例召回（~20 min），正例按 recall_804 × P(对|GT∈候选) 换算全集单位；RefCOCO 30/split SE≈6.5，改 100/split。

**M7 不可解析处理**：任一 audit 不可解析整条 −1 在 K=6、T=1 下约 12% rollout 中招；改为该候选记 unsure、rollout −0.2，turn-1/答案不可解析才 −1；eval 把 K=0 空答、不可解析空答、推导空答分列，报 turn-1 截断率（b400 已截断 91/199）。

**M8 损失归一化**：loss 除以 max_new（现为单 audit 预算），COA rollout 700–1,000 token，梯度尺度 2–3× 且随 K 涨；改除固定 1,024，记录 clip 前梯度范数。

## 4. 评审分歧与裁决

**[type] 标签留不留**：evaluation-statistics 主张保留并用槽值差异定义 named；其余三方要删（正则误标 "no visible logo"→count，v2 设计无标签）。裁决：删标签；named 用 B4 的实词规则 + 奖励侧 flip_from 校验；探针额外报告 seen≠claim / CONTRAST / 槽值三种通过率，作为诊断而非规则。

**主指标是模型自答还是推导答案**：设计稿定自答为主；evaluation-statistics 指出一致性门使二者收敛为同一数，"忠实度差"不可能成为结果。裁决：第三次纯文本调用固定后两者同报；差 >3 点即门失效，以推导数为准；忠实度差是诊断。

**配对信用 s+ 的 0.5 下限**：reward-hacking 说方向惰性（never 让 null 比 box 差），evaluation-statistics 要 1:1 配比 + pos-null=0。裁决：用 B3 的"描述实例门"替代配对信用（同一意图但可校验），不两者并存；配比 1:1；pos-null 0。

**审计行数口径**：data-labels 4,622 行/156 场景，implementation-cost 4,767 行/240 矩阵（含 extras）。机制已在代码确认，具体数无关紧要；修后以 100% 断言为准。

**算力倍数 2×/3–4×/4–6×**：implementation-cost 有实测（旧 4.1–4.3 min/step、isolated eval 20–22 s/item），最可信；三方修法一致（去一张满图）。裁决：先改上下文，第 3 步实测定预算，不采用任何估算。

**K>1 是否值得**：evaluation-statistics 的 AND 封顶是唯一无人反驳的结构性风险；faithfulness 提议"null 侧保留 AND、box 侧由模型在无 mismatch 候选中自选"。裁决：两者都做——探针 K 曲线定去留，K>1 继续时采用 box 侧放宽。

**一致性作 kill 信号**：三方认为几步内饱和。裁决：一致性是格式检查，kill 改为低于 SFT−5。

## 5. 各阶段必须达到的数字

**零训练探针**（turn-1 = sft_v2_half_b400 提案，audit 用去 k/n 的新提示词，两图上下文，100+99 GME 存全部 audit，~1 GPU-h）：(1) 格式：audit 可解析 ≥90%，mismatch 行 echo ≤30%，单一泛化 seen 串占 mismatch ≤20%；(2) 正例含 GT 候选的 ~73 项：GT audit 候选级误控 f ≤25%（>40% 杀），其他候选点名率 q ≥80%（<65% 杀）；(3) 拒绝项 top-1 audit 点名率 c1 ≥45%，c1−f ≥25 点（<15 杀：头无判别力，训练救不了）；(4) K 曲线：某 K 使推导 rej ≥36 / pos ≥38 / pos-null ≤15 / net ≥14（V3d 同 199 项 31.0/43.4/12.1/18.9，V3e 40.0/41.4/16.2/23.8）；最佳 K 的 net <8 杀；只有 K=1 过则改建 K=1；(5) 规则：seen≠claim 通过 >95% mismatch 行即换规则；(6) 名次：f 在 rank1 与 rank≥2 相差 ≤10 点；(7) own 孪生场景每候选 8 采样 T=1：负例描述实例的 audit "恰一行 mismatch、对齐子句、seen 含 flip_from" 至少一次的比例 ≥40%（<25% 无物可放大）；(8) 实测三图与两图每 8-rollout 项的墙钟时间。

**SFT 筛查**（正例用全集单位）：标签重渲 100% 一致；单错行 ≥40%；own 留出唯一行：flip 行点名 ≥80% 且 seen 含 flip_from 实词 ≥70%，全 yes 行误控 ≤5%，单错兄弟行落对 ≥60%，|P(fits|k=1)−P(fits|k≥3)| <5，mismatch 行 seen==claim <5%，unsure 行 <30%。GME：一致性 ≥90，f ≤15%，q ≥85%，c1 ≥50%，推导 rej ≥38 / pos ≥38 / pos-null ≤20 / net ≥ 探针+5 且 ≥14；own gray gap ≤8；纯文本地板 ≤10；RefCOCO 100×3 ≥84/74/81。

**RL 筛查**（≤25 步先测速）：min/step ≤7、显存 ≤29 GB；注入 <40% 步；按类均 K 平（±0.5）；own neg-null ≥ SFT−5；own gray gap ≤8；一致性 ≥90 且不低于 SFT−5；推导 rej ≥42 / pos ≥39 / pos-null ≤15 / net ≥24（Δnet vs SFT ≥10 或 rejection McNemar p<0.1）；RefCOCO ≥84/74/81；筛查内任何差 <5 点的判断先复跑种子再读数。

**全集判定**：两个种子 SFT+RL，每个种子 rej ≥40.3 且 201 项上 McNemar p<0.05 胜 V3d，pos ≥37.7（配对差 ≥−0.6），GME gray gap ≤5，文本地板 ≤10，RefCOCO 100×3 ≥88/78/85，两种子平均 net ≥23；补跑 V3d 第二种子（~5 GPU-h）做 2 vs 2，V3d 两种子差 >4 则双方加第三种子；806 项未筛查子集单独报。RC-GRPO：同 base、同数据、同 RL 预算（步×组）、同 LoRA 秩、2 种子，α 在 own-val 调到正例与我们相差 ≤2；主张 = 匹配正例下 rejection ≥ RC-GRPO+15 且 RefCOCO 不低；旁列论文 4B gRefCOCO 77.2→85.3 与 GME-SFT 复现 30.9/30.0 vs 论文 27.9。通用性：POPE 1,000 与 base 差 ≤1.5，MMStar 1,500 差 ≤2.5，每种子全集阶段各跑一次。

## 6. 开跑前冒烟清单（GPU 被 grpo_v3 占用期间即可在 CPU 完成）

1. **渲染器闭环**（240 训练矩阵）：decide(parse(render_audit)) == 标签答案 100%；mismatch 行 norm(seen)==norm(claim) 为 0；verdict 行与规则不一致为 0；打印丢弃行/项数；每位置 fits 率在所用顺序下相差 <5 点；单错行占比 ≥40%；去重后行数。
2. **提示词同一性**：build_example_coa 的 audit 用户文本与 _COA_TMPL.render 字节相同（已无 k/n、[type]、relation 规则）；turn-1 提示词 == eval 模板；上下文改为 [描红图, 裁片] 后 u1 一图、u2 一图。
3. **token 前缀**：20 个 audit + 10 个答案样例，prompt ids == full ids[:n_p]，tok.decode(labels) 与目标文本逐字相同（上下文改动后重跑；现 16/16 通过）；"<claimed detail> |" mask 生效。
4. **奖励单元测试**（六条手写 rollout）：正例 GT 首位全 match + 正确框 → 1.1；答案≠推导 → 0.1；负例 K 个泛化 "seen: not visible" → 无 null 分且触发监控；负例对齐子句点名 flip_from → +0.5 叠加；任一 audit 不可解析 → 该候选 unsure、rollout −0.2；K=0 → −1；正例 GT audit 误控 → −0.5；无标注负例封顶 0.5；accusation 优势只落在命中段。
5. **eval 干跑**：对已存 audit 文本 + 一个零框项解析：不崩，K=0 记录，model_answer / derived / consistent / null_by_k0 / null_by_unparsable 字段齐全，一致性不再恒 0.98。
6. **监控项就位**：mismatch 行 top-5 seen 串与无新实词占比；正负例分列的 unsure 率；按类均 K；注入比例；clip 前梯度范数；flip 子句对齐率（<80% 视作噪声项）。
7. **首个 GPU 分钟**：2 场景 GRPO 冒烟（--max-scenes 2 --epochs 1），记录 min/step、每 rollout 完成 token、max_memory_allocated；>7 min/step 或 >29 GB 即停。探针 k1_coa_sft（train/queues/v5.txt 第 3 行）须改用重写后的提示词与 sub_batch 4 再排队；GME 上第一个 sub-batch 后记录显存并在 29 GB 以上中止。