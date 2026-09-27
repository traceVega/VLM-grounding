# PoC 训练与评测方案（2026-09-23）

目的只有一个：证明 `poc_v1.jsonl` 里的 93 条难负样本能让模型在"原本以 p≈1 框错"的难负样本上学会拒答，且正样本不塌、不是靠读字。回答不了"GroundingME 提升多少"。

## 模型

Qwen3-VL-4B-Instruct，LoRA。4B 是最终要训的模型，GRPO 在 5090 上能舒服地跑（8B 要拆批、缩图）。
所有难度门此前在 8B 上量的，换模型后先用 4B 重跑 policy 阶段（308 条，约 10 分钟）：记录 p(null)、框结果；
4B 已经拒答的负样本剔除，已经框错的正样本记为困难正样本。这一步同时给出训练前基线。

## 数据

- `~/vlmg-data/datagen/exports/poc_v1.jsonl`：308 条，正样本 110、兄弟正样本 105、负样本 93，116 个场景。
- 切分按场景（`group`），不按条目：留 18 个场景（约 45 条）做验证，其余训练。固定种子。
- 不做类别平衡（Person 三成是源分布）。不上采样负样本（约 30%，GroundingME 是 20%）。
- 格式：GroundingME 原 prompt（`configs/prompts/grounding_qwen3vl_primary.txt`，逐字）；答案 `{"bbox_2d": [x1, y1, x2, y2]}`
  用 Qwen3-VL 的 0–1000 相对坐标（与 harness 的 `parse_qwen3vl(RELATIVE_1000)` 一致），负样本逐字 `{"bbox_2d": null}`。
  只在答案 token 上算损失。不加思维链。

## 第一阶段：SFT 冷启动（必做）

- LoRA r=16, alpha=32，作用于语言模型的 q/k/v/o 和 MLP，视觉塔和 merger 冻结。
- lr 1e-4，cosine，warmup 5%，3 epoch，有效 batch 8（batch 2 × 累积 4），图按 harness 的像素上限。
- 同一场景的三条尽量同批（对比信号来自共享的图）。
- 每 20 步在验证集上量决策 token 的 p(null)（负样本）和 p(box)（正样本）：解锁曲线。
- 产出：LoRA 权重、曲线、验证集指标。本地约 10 分钟。

## 第二阶段：GRPO（可选，看解锁曲线）

- 只有第一阶段负样本 p(null) 抬到 0.1 以上才做；否则采不到 null，RL 没有梯度。
- 奖励：正样本 IoU≥0.5 得 1；负样本输出 null 得 1；格式不合法 −1；其余 0。
- 每 prompt 采 8 条，温度 1.0，训练集全部 prompt，3 轮；参考模型 = 关掉 LoRA 的同一份权重；KL 系数 0.02。
- 本地 4B 估计 30–45 分钟。

## 评测（训练前 = 4B 基线，训练后同一套）

| 评测 | 指标 | 数据 |
|---|---|---|
| GroundingME 拒答 201 | 准确率（基座 0）；决策 token p(null) 总和与分布（基座 3.4e-4，8B） | 本地 |
| GroundingME 非拒答 804，按类别 | 准确率（IoU≥0.5）；相对基线的掉点 | 本地，全集不抽样（SE≈1.8 点） |
| 自家验证集 18 场景 | 负样本拒答率、正/兄弟正样本准确率、p(null) | 本地 |
| 灰图对照 | 同样的验证负样本配灰图：拒答率应远低于真图 | 本地 |
| RefCOCO / RefCOCO+ / RefCOCOg val 各 300 条 | 短表达式准确率保持（训练集全是长句） | 下载中：`~/vlmg-data/raw/refcoco/`（各取 val 第一分片，采样 300） |
| 困难正样本 | 4B 基线框错、训练后是否框对 | 自家数据 |

GroundingME 的图和文本只在本地评测，不进训练集、不发 API。

## 成败判据（先定，后看）

- 解锁：验证负样本 p(null) 中位数从 ~1e-14 抬到 ≥0.1，且 GroundingME 拒答 201 上 p(null) 总和明显上升。否则数据量不够或方法不对，转看前缀路线。
- 代价：GroundingME 非拒答掉 ≤5 点可接受；>5 点 → 降负样本比例或加 RefCOCO 类正样本回放。
- 捷径：灰图拒答率接近真图拒答率 → 文本捷径，对策是把改假值同时放进正样本对子里。
- 短句：RefCOCO 系列掉 >3 点 → 加短句正样本回放。

## 工程

- 环境：`~/vlmg-env` 加 `peft`；不用 TRL，自己写 `train/sft_lora.py`（两百行），chat 模板、坐标、解析都走现有 harness。
- 评测复用 `scripts/pilot_abstain_signal`（决策 token）和 datagen 的 policy 解析；GroundingME 用 K2 的 harness。
- 顺序：下载 → 4B policy 重跑 + 基线评测 → 切分 → SFT + 曲线 → 评测 → 决定 GRPO。
