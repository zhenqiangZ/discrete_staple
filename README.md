# Discrete Stapled Peptide Design

基于掩码离散扩散模型，为已知抗菌肽（AMP）设计订书肽（stapled peptide）的计算设计项目。

模型学习「原始 AMP → 引入 S5/R8 订书桥」的条件映射，支持四种条件：

| cond | 含义 | 订书方式 |
|---|---|---|
| 1 | 原始序列 | 不做修饰（identity） |
| 2 | i,i+4 单订书 | 两个 S5 |
| 3 | i,i+7 单订书 | S5 + R8 |
| 4 | 双订书 | 两组不重叠的订书对 |

## 方法概览

1. **数据增强**：从 AMP 序列枚举螺旋面兼容的 i,i+4 / i,i+7 位点，替换为 S5/R8，生成配对训练数据。
2. **监督预训练**：24 词表（20 AA + S5 + R8 + PAD + MASK）的 Transformer，随机掩码去噪，条件向量经 cross-attention 注入。
3. **候选枚举**：对每个父序列枚举全部合法订书位点，按参考模型 log-prob 排序取 Top-K。
4. **多目标奖励**：三模型 LLM 评审团（评分 → 互评 → 终审），结合 AMP/毒性分类器的逐位置梯度对齐。
5. **GRPO 微调**：组内标准化优势 + clip + KL 约束，优化生成策略。
6. **奖励代理（可选）**：用本地 Transformer 回归 LLM 分数，降低 API 成本。

## 目录结构

```
.
├── TRAINING_PLAN.md          # ★ 迁移到 GPU 机器的训练 runbook（先读这个）
├── config.py                 # 全局配置（模型/训练/RL/LLM）
├── data/
│   ├── peptide_data.csv      # 原始 AMP 数据（44,618 条）
│   ├── paired_data.csv       # 配对训练数据（199,099 对，当前主线）
│   ├── paired_data_legacy.csv# 旧配对数据（82,603 对，复现旧检查点用）
│   ├── make_holdout_split.py # 按父序列切 train/holdout（真 held-out 评估）
│   ├── tokenizer.py          # 24 词表编解码
│   ├── dataset.py            # SFT Dataset / DataLoader（按父序列分组划分）
│   ├── build_paired_data.py  # 生成配对数据（主线使用）
│   ├── augment_stapled.py    # 位点采样 + 订书增强
│   └── raw/                  # 上游原始数据（DRAMP 导出、过滤前后 JSON）
├── model/
│   ├── network.py            # StapleDiffusionNetwork
│   ├── diffusion.py          # 随机掩码
│   └── sampling.py           # 迭代去掩码采样
├── train.py                  # 监督预训练（SFT）
├── train_models.py           # AMP / 毒性分类器
├── predictors.py             # 分类器加载 + 逐位置梯度对齐
├── reward.py                 # 张量版位置对齐奖励（re-export 规则）
├── reward_rules.py           # torch-free 规则奖励（最大匹配配对/孤儿率/保真度）
├── train_rl_grpo.py          # GRPO 微调（主线）
├── design.py                 # 候选枚举 + 按 log-prob 排序（训练/评估共用）
├── eval_core.py              # torch-free 评估指标 + 随机基线
├── evaluate.py               # 评估入口：SFT / GRPO / random 三方对比
├── llm_judge/                # 多模型评审团
├── collect_judge_data.py     # 评审数据采集
├── train_reward_proxy.py     # 奖励代理训练
├── sample.py                 # 无条件采样
├── tests/                    # 纯 Python 单测（无需 torch）
├── reports/                  # 评估报告（*.md / *.json 纳入版本控制）
├── checkpoints/              # 模型权重（已 gitignore，需单独传输）
├── logs/llm_judge_collection/# 评审记录（纳入版本控制，训奖励代理用）
├── legacy/                   # 归档：早期 RL、采样原型、无关任务
└── toxdata/                  # 毒性模型训练数据
```

## 环境

```bash
pip install -r requirements.txt
```

> 需要 Python 3.10+。GPU 可选，CPU 可跑推理，训练建议 CUDA。

## 数据

| 文件 | 规模 | 说明 |
|---|---|---|
| `data/peptide_data.csv` | 44,618 | 22,176 正 / 22,442 负，长度 4–50 |
| `data/paired_data.csv` | 199,099 | cond1 22,176 / cond2 79,862 / cond3 66,090 / cond4 30,971（每父序列 3.6 / 3.0 / 1.4 个目标） |
| `data/paired_data_legacy.csv` | 82,603 | 2026-10-19 之前的版本，每父序列每种条件仅 1 个目标；现有检查点是用它训练的，保留以便复现 |
| `toxdata/` | 多源合并 | 毒性分类器训练数据（TXP / CAPTP / Toxtel / BTL 等） |

重新生成配对数据：

```bash
python data/build_paired_data.py
```

生成器对每个父序列从**全部合法候选**中均匀采样多个订书位点（`SAMPLES_PER_COND = 4`、`SAMPLES_DOUBLE = 2`），并过滤掉末端锚定与会产生孤儿 S5/R8 的组合。脚本结束会打印 `targets/parent`，该值若为 1.0 说明训练集又退化成了唯一解。

## 快速开始

> 完整的迁移/训练流程（含显存对照、时间估算、验收标准、成本估算）见 **[TRAINING_PLAN.md](TRAINING_PLAN.md)**。
> 下面是骨架。

```bash
# 0. 配置 LLM 密钥（可选，仅 RL 的 LLM 评审阶段需要）
cp .env.example .env   # 填入 A/B/C 三个 key

# 1. 训练 AMP / 毒性分类器（产物 checkpoints/{amp_classifier,toxicity}_best.pt）
python train_models.py

# 2. （推荐）按父序列切出真 held-out，供最终评估用
python data/make_holdout_split.py --frac 0.15

# 3. 监督预训练 —— 注意显式给 --epochs，默认值 1000 是历史遗留
python train.py --data data/paired_train.csv --out_dir checkpoints/sft_v2 \
                --epochs 30 --batch_size 256 --lr 1e-4

# 4. 关卡：SFT 必须打赢随机基线，否则先回头查数据
python evaluate.py --n_parents 200 --k 5 --conds 2,3,4 \
  --sft_ckpt checkpoints/sft_v2/best.pt --systems sft,random \
  --parents_csv data/paired_holdout.csv --holdout 1.0 --out reports/eval_sft_v2

# 5. GRPO 微调 —— 必须显式给 --steps_per_epoch，默认 20 只够动 260 步
python train_rl_grpo.py --ckpt checkpoints/sft_v2/best.pt \
  --save_dir checkpoints/rl_grpo_v2 \
  --epochs 20 --steps_per_epoch 125 --batch_size 16 --k 8 --w_llm 0

# 6. 评估：SFT / GRPO / 随机基线三方对比
python evaluate.py --n_parents 200 --k 5 --conds 2,3,4 \
  --sft_ckpt checkpoints/sft_v2/best.pt \
  --grpo_ckpt checkpoints/rl_grpo_v2/rl_grpo_best.pt \
  --parents_csv data/paired_holdout.csv --holdout 1.0 --out reports/eval_final

# 7. 运行单测（无需 torch，117 项断言）
for f in tests/test_core.py tests/test_eval_core.py tests/test_record_store.py tests/test_reward_rules.py; do
    python "$f"
done
```

设计订书肽的入口是 `design.py`（`design_candidates`），早期独立采样脚本已归档到 `legacy/early_sampling/`。

## 评估

`evaluate.py` 在固定父序列集上对比三个系统，输出控制台表格 + `reports/eval_*.json` + `reports/eval_*.md`。

| 系统 | 说明 |
|---|---|
| `sft` | 监督预训练检查点 |
| `grpo` | GRPO 微调检查点 |
| `random` | 模型无关基线：在父序列上均匀随机放置订书 |

两种生成模式：

- `--mode design`（默认）：枚举全部合法订书位点，按模型 log-prob 排序取 top-K。**与 GRPO 训练时的 `design_candidates` 完全同一条推理路径**，因此评估的就是模型被优化的那个目标。
- `--mode generate`：掩码父序列后让模型重建，探查原始生成策略。`--mask_rate` 默认 1.0（全掩码）——部分掩码时订书位点通常未被掩掉、会保留原残基，合规率就被掩码几何卡死，而不是反映模型能力。此模式下 `Pres` 预期很低（父序列信息已被掩掉），**不是**质量指标；应看 `Compl`、`Orph%`、`Pairs`。

指标分四组：

| 组 | 指标 |
|---|---|
| 结构 | cond 合规率、订书位点合法性、全部合法率、保真度、订书数、末端锚定率、回退率 |
| 预测功能 | AMP 概率均值 / 阳性率、毒性概率均值 |
| 物化性质 | 净电荷、⟨H⟩、疏水矩 μH、分子量 |
| 多样性 | 唯一序列比例、平均两两同一性 |

> **注意**：`--holdout` 用 md5 分桶保证父序列集可复现，但**不是真正的 held-out**——现有检查点是在全部 `data/paired_data.csv` 上训练的。SFT/GRPO 的数字属于同分布结果，应视为上界。

`回退率（Fall%）` 指设计与父序列完全相同的比例，即枚举找不到 K 个不同候选时用原序列占位。该值偏高说明该方法在该父序列/条件下实际没有产出设计。

## 评估结果（2026-10-19）

200 条父序列（长度 10–20）× K=5 × cond 2/3/4，每个系统 1000 个设计。完整数据见 `reports/eval_design.md`、`reports/eval_generate.md`。

### design 模式（部署路径）

| 系统 | cond | Compl | Valid | Valid% | Rew | Term% | Offset | AMP |
|---|---|---|---|---|---|---|---|---|
| sft | 2 | 1.000 | 0.778 | 77.8% | 0.967 | 22.2% | 0.33 | 0.840 |
| grpo | 2 | 1.000 | 0.780 | 78.0% | 0.967 | 22.0% | 0.32 | 0.841 |
| random | 2 | 1.000 | **0.792** | **79.2%** | **0.969** | **20.8%** | 0.36 | **0.846** |
| sft | 3 | 0.968 | 0.705 | 70.5% | 0.940 | 29.5% | 0.23 | 0.848 |
| grpo | 3 | 0.968 | 0.706 | 70.6% | 0.940 | 29.4% | 0.23 | **0.849** |
| random | 3 | 1.000 | 0.705 | 70.5% | **0.956** | 29.5% | 0.24 | 0.836 |
| sft | 4 | 0.984 | 0.627 | 37.7% | 0.936 | 62.3% | 0.32 | 0.769 |
| grpo | 4 | 0.980 | 0.616 | 36.8% | 0.932 | 63.2% | 0.32 | 0.769 |
| random | 4 | 0.954 | **0.755** | **56.5%** | **0.940** | **43.5%** | 0.28 | **0.801** |

三条结论：

1. **GRPO 相对 SFT 没有任何可测量的改进。** 所有结构指标差异 ≤ 0.011，AMP/毒性差异 ≤ 0.009。检查点权重确实变了（223 个张量平均相对变化 0.79%，最大 9.9%），所以不是加载错误——是优化幅度不足以改变行为（lr=5e-6、13 epoch × 20 update ≈ 260 步）。

2. **随机基线在多数指标上不输甚至优于两个模型。** cond 3/4 的规则奖励（`Rew`）随机更高；cond 4 的位点合法性随机 0.755 vs 模型 0.62–0.63，AMP 概率随机 0.801 vs 模型 0.769。

3. **cond 4（双订书）最差。** 模型有 62–63% 的设计把订书锚定在末端残基上（随机只有 43.5%），导致 `Valid%` 仅 37%——结构上不可行。

### 根因：训练数据只有唯一解

`data/build_paired_data.py` 对 cond 2/3 只取 `enumerate_staple_candidates()` 排序后的**第一个**候选（`s4[:1]` / `s7[:1]`），而该排序按 `-abs(i - L/4) * 0.5` 加权，即偏好四分之一处。

实测 `data/paired_data.csv`：

- cond 2：21,301 个父序列对应 21,301 个目标 —— **每个父序列只有 1 个解**
- 订书位置与 `n/4` 的差值集中在 `-0.5 ~ +0.2`

于是 SFT 学到的不是「订书位点分布」，而是「把 S5 放在 round(n/4)」这条确定性映射。这解释了全部现象：`Offset` 恒为 0.32 左右、同父序列内多样性接近 0、短肽上 `n/4` 落到末端从而拉低 `Valid%`、以及 GRPO 缺乏可探索的分布（top-K 只是在同一位置附近的噪声）。

### generate 模式（全掩码，原始生成策略）

| 系统 | cond | Compl | Pairs | Orph% | Rew |
|---|---|---|---|---|---|
| sft | 2 | 0.752 | 1.06 | 50.4% | 0.553 |
| grpo | 2 | 0.810 | 1.15 | 45.0% | 0.586 |
| sft | 3 | 0.165 | 0.40 | 72.4% | 0.274 |
| grpo | 3 | 0.265 | 0.55 | 67.4% | 0.319 |
| sft | 4 | 0.618 | 1.41 | 52.6% | 0.466 |
| grpo | 4 | 0.667 | 1.52 | 49.7% | 0.491 |

模型能生成 i,i+4（合规 75–81%），但 i,i+7 只有 16–27%；**45–72% 的 S5/R8 残基是孤儿**（未配对成桥）。说明间距约束没有被真正学会——design 路径下由枚举强行保证，所以看不出来。GRPO 在此模式下反而略优于 SFT（孤儿率低 3–5 个百分点），是唯一能看到 GRPO 起作用的指标。

### 建议的修复顺序

1. ~~`build_paired_data.py`：每个父序列采样 5–10 个随机合法位点（而非 `[:1]`），并去掉 `abs(i - L/4)` 这个位置先验项。~~ → 已修复，见下节。
2. ~~`compute_reward`：`w_valid` 从 0.15 提高。~~ → 已提到 0.35。
3. ~~`design_candidates` cond 4：拒绝会产生孤儿 S5/R8 的放置。~~ → 已加 `staple_matching_size` 检查。
4. 重新生成配对数据 → 重训 SFT → 再跑 GRPO，然后重跑本评估。

### 已执行的修复（2026-10-19）

| 项 | 改动 |
|---|---|
| 数据生成 | `build_paired_data.py` 改为复用 `augment_stapled.augment_sequence`；`SAMPLES_PER_COND=4`、`SAMPLES_DOUBLE=2`，从全部合法候选中**均匀无放回**采样 |
| 位置先验 | `augment_stapled._site_score` 删除 `-abs(i - L/4) * 0.5`，只保留螺旋倾向项 |
| 位点合法性 | `is_valid_staple_position` 增加端部余量，与 `staple_site_validity` 对齐 |
| 配对解析 | `detect_staple_pairs` 由贪心扫描改为**最大匹配**。旧实现遇到 `S5@0, S5@4, R8@7, S5@8` 会只报 1 对并把 R8 留成孤儿；最大匹配能正确还原成 2 对 |
| 孤儿残基 | 新增 `orphan_staple_rate` / `staple_matching_size`，生成器据此拒绝无法成键的放置 |
| 奖励权重 | `W_ACCURACY=0.45 / W_VALIDITY=0.35 / W_PRESERVATION=0.20`；末端锚定从 0.85 降到 0.65 |
| 数据划分 | `create_loaders` 改为**按父序列分组**划分。旧 `random_split` 按行随机切分，同一父序列的 2–4 个订书变体会同时进入训练集和验证集 |

新 `paired_data.csv` 实测：4 个条件**零孤儿残基、cond 合规率 100%、位点合法性 100%**；cond 2 的订书位置分布标准差由 0.195 升到 0.214，最密集的十分位占比由 34% 降到 6.6%。每个父序列的目标数（cond 2/3/4）为 3.75 / 3.61 / 1.85，不再是唯一解。

新划分：Train 179,199（19,959 父序列）/ Val 19,900（2,217 父序列），**父序列重叠 0**。

> **现有检查点（`best.pt` / `rl_grpo_best.pt`）仍是在旧数据上训练的**，所以上表里的模型指标不会因为这次修改而变化。需要重训 SFT 后再评估才能看到效果。

## 配置要点

- `config.py` 中的 LLM key 通过环境变量读取；本地复制 `.env.example` 为 `.env` 并填入密钥（`.env` 已 gitignore）。
- `RECORD_DIR` 控制评审记录输出目录，`RecordStore` 启动时会加载全部历史记录，并把旧 z-score 记录还原为绝对 0–10 分。
- `PROXY_ENABLED=False` 时直接调用 LLM API；开启前需先训练 `checkpoints/reward_proxy/best.pt`。开启后奖励按「缓存命中 → 代理模型 → LLM API」三级回退。

## 已知问题

P0 正确性问题已修复（2026-10-19）：

- ~~`train.py` 训练 loss 未按 mask 取子集~~ → 已改为 mask-only，训练/验证口径一致。
- ~~`model/sampling.py` 硬编码 `MAX_LEN = 18`~~ → 已改为 `DEFAULT_SAMPLE_LEN = 20`，并支持 `--max_len`。
- ~~`compute_properties.py` 逐字符解析 S5/R8~~ → 已改为 `split_residues()` 按残基解析。
- ~~API key 硬编码~~ → 已移至 `.env`（gitignored），`config.py` 只读环境变量。

已修复（2026-10-19，阶段 2B）：

- `preservation_score` 不再把 S5/R8 位点当作突变，分母改为 scaffold 位点数。
- `staple_site_validity` 增加端部余量与间距检查，末端订书不再得分。

已修复（2026-10-19，阶段 2C —— GRPO 信号链路）：

- importance ratio 只对**突变位**求和并 `clamp(-10, 10)`，不再随序列长度放大。
- 引入 `judged` 掩码：组内优势归一化与 `pg_loss` 只在真正被评审的候选上计算，未评审组不再稀释梯度。
- 未到评审步的 batch 在 forward 之前 `continue`，每 epoch 的更新次数由 `update_step` 单独计数，`global_step` 每 batch 递增。
- 评审失败返回 `NaN` 而不是全 0，失败组被剔除而非当作「低分」学习。
- 评审记录不再写入失败组，`NaN` 不会进入 JSONL。
- 补上了此前断掉的信号链路：`RecordStore` 缓存命中 → 奖励代理 → LLM API 三级回退（`--proxy_enabled` 默认关闭，行为不变）。

已修复（2026-10-19，阶段 2D —— 奖励量纲统一）：

- `JudgeRecord` 新增 `score_scale` 字段，新记录统一标记为 `raw`（绝对 0–10）。
- 旧记录的 `final_scores` 是组内 z-score，无法直接作为回归目标；新增 `llm_judge/score_scale.py`，从记录内嵌的 phase-3 `final_score`（或 phase-1 A/B 均值）**还原绝对分**，实测仓库内 3/5 条旧记录可完整还原，另 2 条因采集时 LLM 调用失败、payload 为空而无数据可用。
- `RecordStore.lookup_raw()` 只返回绝对分；`train_reward_proxy.py` 只训练绝对分样本。
- `llm_judge/__init__.py` 改为惰性导出，`import llm_judge.record` 不再强制拉入 numpy / HTTP 客户端。

已修复（2026-10-19，阶段 3 后 —— 数据与结构约束）：

- `build_paired_data.py` 每个父序列只取一个排序首位的目标（`s4[:1]` / `s7[:1]`），且排序含 `abs(i - L/4)` 位置先验 → 训练集退化成唯一解。已改为均匀采样多个合法位点，详见「评估结果 → 已执行的修复」。
- `detect_staple_pairs` 贪心配对会漏报交叉键 → 改为最大匹配。
- `compute_reward` 的 `w_valid` 仅 0.15 → 提到 0.35。
- `design_candidates` cond 4 可能产生孤儿 S5/R8 → 加 `staple_matching_size` 检查。
- `data/dataset.py` 的 `random_split` 按行划分，同一父序列的订书变体会同时落在训练/验证集 → 改为**按父序列分组划分**（`create_loaders`）。新数据每父序列最多 4 个目标，不分组的话泄漏会被放大 4 倍。当前划分：Train 179,199（19,959 父序列）/ Val 19,900（2,217 父序列），父序列重叠为 0。

仍待处理（P1，会影响 RL 结果）：

1. **检查点需要重训。** 数据与奖励都改了，但 `checkpoints/best.pt`、`checkpoints/rl_grpo/rl_grpo_best.pt` 仍是旧数据训练的产物；不重训就无法验证修复效果。本机只有 MX250（2GB 显存），SFT 199k 样本在此不现实。
2. 现有评审记录仅 5 组 / 50 条，其中可用 3 组，远不足以训练奖励代理。
3. 仓库尚未 `git init`，`.gitignore` 已就位。

## 归档

`legacy/hnav1_7_peptide_eval/` 是独立的 hNav1.7 四肽抑制潜力评估任务，与订书肽主线无关，详见该目录下的 README。
