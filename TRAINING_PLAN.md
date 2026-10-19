# 训练计划（迁移到 GPU 机器执行）

> 本文件是可直接照做的 runbook。目标不是「把模型训一遍」，而是**验证 2026-10-19 的数据与奖励修复是否真的解决了「模型不优于随机基线」这个问题**。
> 如果只关心命令，跳到 [第 5 节](#5-sft从零训练)。

---

## 0. 你接手的是什么状态

### 已经改完的（不需要再动）

| 项 | 现状 |
|---|---|
| 配对数据 | `data/paired_data.csv`，199,099 条，每父序列 1.85–3.75 个目标（不再是唯一解） |
| 数据质量 | 4 个条件孤儿残基 0、cond 合规 100%、位点合法性 100% |
| 奖励函数 | `W_ACCURACY=0.45 / W_VALIDITY=0.35 / W_PRESERVATION=0.20` |
| 配对解析 | 二部图最大匹配（旧贪心会漏报交叉键） |
| 候选生成 | `design.py` 的 cond 4 已过滤孤儿残基 |
| 数据划分 | 按父序列分组，父序列重叠 0 |
| 单测 | 4 个脚本、117 项断言，全部通过 |

### 没有改的（本计划的全部工作量）

**检查点是旧的。** `checkpoints/best.pt` 和 `checkpoints/rl_grpo/rl_grpo_best.pt` 都是用旧的 82,603 条 `paired_data_legacy.csv` 训练的，学的是「把 S5 放在 round(n/4)」这条确定性映射。

所以：**任何评估结论在重训之前都无效。** 现有 `reports/eval_*.md` 里的模型数字反映的是旧模型的缺陷，不是新代码的能力。

### 这次要回答的三个问题

1. 新数据训出的 SFT，`Rew` 能否终于高于 random 基线？（旧：不能，cond 3/4 随机更高）
2. cond 4 的末端锚定能否从 62–63% 降下来？（random 基线 43.5%）
3. GRPO 在结构指标上能否产生 >0.02 的可见改进？（旧：≤0.011，等于没有）

---

## 1. 机器要求

| 项 | 最低 | 推荐 | 说明 |
|---|---|---|---|
| GPU 显存 | 8 GB | 24 GB+ | 8 GB 需把 `--batch_size` 降到 64–128 |
| Python | 3.10 | 3.11 / 3.12 | |
| CUDA | 11.8 | 12.1+ | `torch>=2.0` |
| 磁盘 | 5 GB | 20 GB | 数据 15 MB，检查点每个 ~430 MB |
| 网络 | 需要访问 LLM API（仅 GRPO 的 LLM 评审阶段） | | 可先跳过，见 [7.2](#72-先跑纯规则版本推荐) |

CPU 训练不可行：199k 样本 × 30 epoch 在 CPU 上要数十小时。

---

## 2. 传输清单

### 2.1 git 里已经有的（`git clone` 即可）

仓库共 88 个文件，约 34 MB，包含：

- 全部源码（`train.py` / `train_rl_grpo.py` / `evaluate.py` / `design.py` / `reward_rules.py` / `eval_core.py` / `model/` / `data/` / `llm_judge/`）
- `data/paired_data.csv`（199,099 条，训练主数据）
- `data/paired_data_legacy.csv`（82,603 条，复现旧检查点用）
- `data/peptide_data.csv`（配对数据的输入）
- `toxdata/`（毒性分类器训练数据）
- `tests/`（4 个单测脚本）
- `reports/eval_*.md`（旧基线数字，用于对照）
- `logs/llm_judge_collection/` + `test_records/`（评审记录，训奖励代理用）

### 2.2 git 里没有的 —— **必须单独传**

`.gitignore` 排除了 `*.pt`，但有两个文件是 **GRPO 和评估的硬依赖**，缺了会直接报错：

```bash
# 必须传（12.4 MB）
checkpoints/amp_classifier_best.pt
checkpoints/toxicity_best.pt

# 建议传（对照用，各 ~430 MB）
checkpoints/best.pt                    # 旧 SFT，作为「修复前」基线
checkpoints/rl_grpo/rl_grpo_best.pt    # 旧 GRPO，同上
```

传输方式（在源机器执行）：

```bash
# 小文件走 scp
scp checkpoints/amp_classifier_best.pt checkpoints/toxicity_best.pt user@host:~/discrete_staple/checkpoints/

# 大文件建议 rsync 断点续传
rsync -avP checkpoints/best.pt checkpoints/rl_grpo/ user@host:~/discrete_staple/checkpoints/
```

### 2.3 绝对不能传的

- `.env`（含 API 密钥）。目标机器上从 `.env.example` 重建。

---

## 3. 环境搭建

```bash
git clone <repo> && cd discrete_staple

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 只有要跑 LLM 评审才需要
pip install -r requirements_llm.txt

cp .env.example .env   # 填入 LLM_A/B/C_API_KEY；纯规则 GRPO 可以不填
```

### 3.1 自检（必做，2 分钟）

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 117 项断言，全部应通过。不需要 GPU。
for f in tests/test_core.py tests/test_eval_core.py tests/test_record_store.py tests/test_reward_rules.py; do
    python "$f" || echo "FAILED: $f"
done

# 确认数据和配置路径正确（注意 config 用的是绝对路径，可在任意目录运行）
python -c "
from config import PAIRED_CSV
from data.dataset import create_loaders
tr, va = create_loaders(PAIRED_CSV, batch_size=64)
print('train batches', len(tr), '| val batches', len(va))
"
```

预期输出：

```
Train: 179199 (19959 parents), Val: 19900 (2217 parents)
train batches: 2800 | val batches: 311
```

如果 batch 数对不上，说明数据文件不对，**先停下**，不要继续。

---

## 4. 数据准备：要不要切 holdout

两种路线，选一个。

### 路线 A —— 快速验证（不切）

直接用 `data/paired_data.csv`。`create_loaders` 会按父序列切 10% 做 val。

- 优点：一步到位
- 缺点：`evaluate.py` 的父序列来自同一个文件，模型训练时见过 → **评估是同分布结果，只能当上界看**

### 路线 B —— 真正的 held-out（推荐）

```bash
python data/make_holdout_split.py --frac 0.15
```

输出：

```
train    rows=169,288  parents=18,850  cond1=18,850 cond2=67,891 cond3=56,207 cond4=26,340
holdout  rows= 29,811  parents= 3,326  cond1=3,326 cond2=11,971 cond3=9,883 cond4=4,631
parent overlap: 0
```

生成 `data/paired_train.csv` 和 `data/paired_holdout.csv`，父序列完全不重叠。

代价是训练数据少了 15%，但换来一个**能对外讲的泛化数字**。既然这个项目的核心质疑就是「模型是不是只会背训练集」，建议走 B。

> 两个文件都不大（13 MB / 2 MB），可以提交进 git 固化下来，避免下次又切出不同的划分。

---

## 5. SFT：从零训练

**不要**用旧 `best.pt` 做 warm start —— 它带着旧数据的位置先验偏差，会让「修复是否生效」这个问题永远说不清。从零训，旧的留作对照。

### 5.1 路线 A 的命令

```bash
python train.py --epochs 30 --batch_size 256 --lr 1e-4
```

### 5.2 路线 B 的命令

```bash
python train.py \
  --data data/paired_train.csv \
  --out_dir checkpoints/sft_v2 \
  --epochs 30 --batch_size 256 --lr 1e-4
```

### 5.3 参数说明

| 参数 | 建议 | 理由 |
|---|---|---|
| `--epochs` | 30 | 默认值 `EPOCHS=1000` 是历史遗留，不要用。778 batch/epoch × 30 ≈ 23,000 步足够收敛 |
| `--batch_size` | 256（≥16 GB 显存）<br>128（8–12 GB） | 显存不够时优先降这个，别降 lr |
| `--lr` | 1e-4 | AdamW + weight_decay 0.01，无 warmup、无 scheduler（`WARMUP_STEPS` 定义了但代码里没用） |
| `--data` | 见上 | 新增参数，指向训练集 |
| `--out_dir` | `checkpoints/sft_v2` | 新增参数，避免覆盖旧 `best.pt` |

损失只算在被 mask 的 token 上（`cross_entropy(logits[mask], tgt[mask])`），mask rate 在 `[0, 0.6]` 随机采样。

### 5.4 时间估算

每个 epoch 778 个 batch（bs=256）。**先只跑 1 个 epoch 实测**，再决定总 epoch 数：

| GPU | 约每 epoch | 30 epoch 总计 |
|---|---|---|
| A100 40 GB | 1.5–2.5 min | 45–75 min |
| RTX 4090 24 GB | 3–5 min | 1.5–2.5 h |
| RTX 3090 24 GB | 4–6 min | 2–3 h |
| RTX 4060 Ti 16 GB | 6–9 min | 3–4.5 h |
| T4 16 GB (bs=128) | 12–18 min | 6–9 h |

粗估，实际以第一个 epoch 的打印为准。

### 5.5 看什么

每个 epoch 会打印：

```
Epoch   1 | train_loss=2.3412 val_loss=1.9876 train_acc=0.4123 val_acc=0.4987 | T_sacc=0.2103 V_sacc=0.3345 | 118.3s
```

- `val_loss` 持续下降 → 正常
- `V_sacc`（订书位点准确率）是这个任务的关键指标，应从 ~0.2 爬到 0.5+。如果一直停在 0.2 附近，说明模型没学到位置分布，**后面的 GRPO 也不会有用**
- 检查点按 `val_loss` 最低保存。注意 val 的 mask 是每轮随机的，`val_loss` 有噪声，不要因为单轮抖动就停

---

## 6. 评估 SFT（GRPO 之前必做）

这一步是**分水岭**：如果 SFT 本身已经不输随机基线，GRPO 才有意义；如果 SFT 仍然打不过 random，先回头查数据，别急着上 RL。

```bash
# 路线 A
python evaluate.py --n_parents 200 --k 5 --conds 2,3,4 \
  --sft_ckpt checkpoints/best.pt \
  --systems sft,random \
  --out reports/eval_sft_v2

# 路线 B（真 held-out）
python evaluate.py --n_parents 200 --k 5 --conds 2,3,4 \
  --sft_ckpt checkpoints/sft_v2/best.pt \
  --parents_csv data/paired_holdout.csv --holdout 1.0 \
  --systems sft,random \
  --out reports/eval_sft_v2_ho
```

产出 `reports/eval_sft_v2.md` + `.json`。**把这份 md 存下来**，它是后面 GRPO 的对照基线。

### 判断标准（对照旧数字）

| 指标 | 旧 SFT | random | 新 SFT 应该 |
|---|---|---|---|
| `Rew`（cond 2/3/4） | 与 random 持平或更低 | — | **高于 random**，至少 cond 2 要赢 |
| cond 4 `Term%` | 62–63% | 43.5% | **< 50%** |
| cond 4 `Valid%` | 37% | — | **> 60%** |
| cond 2 `Offset` 标准差 | 0.195（旧数据） | — | **> 0.20** |
| `Uniq`（同父序列内多样性） | 接近 0 | — | **明显 > 0** |

任一项没达标 → 先看 `reports/eval_sft_v2.md` 的 `Fall%`（回退率）。`Fall%` 高说明枚举找不到 K 个不同候选，通常是父序列太短或 `MIN_TERMINAL_MARGIN` 把合法位点砍太多。

---

## 7. GRPO

### 7.1 关键前提

`train_rl_grpo.py` 的默认值 `RL_STEPS_PER_EPOCH=20` 会让每个 epoch 只更新 20 步（20×16 = 320 个 AMP），而 `SUBSAMPLE_PER_EPOCH=2000` 采样的数据 **84% 用不到**。旧 GRPO 只训了约 260 步，这是「权重平均变化仅 0.79%」的直接原因。

本次新增了 `--steps_per_epoch` 参数。**必须显式调大**，否则会重犯旧错误：

| 参数 | 旧值 | 建议 | 说明 |
|---|---|---|---|
| `--steps_per_epoch` | 20 | **60–125** | 125 = 2000/16，把采样数据用满 |
| `--subsample_per_epoch` | 2000 | 2000 | 与上面配套 |
| `--epochs` | 20 | 20 | 20 × 125 = 2500 步，比旧版多 10 倍 |
| `--k` | 8 | 8 | 每个 AMP 的候选数 |
| `--lr` | 5e-6 | 5e-6 | RL 阶段不要放大 |
| `--mask_rate` | 0.1 | 0.1 | |
| `--clip_eps` | 0.2 | 0.2 | |
| `--kl_beta` | 0.05 | 0.05 | |

### 7.2 先跑纯规则版本（推荐）

**第一步不接 LLM。** 理由：LLM 评审是主要的时间和费用瓶颈（见 7.4），而我们要先验证的是结构指标（`Compl` / `Valid` / `Orph` / `Term`），这些全部来自规则奖励，不需要 LLM。

```bash
python train_rl_grpo.py \
  --ckpt checkpoints/sft_v2/best.pt \
  --save_dir checkpoints/rl_grpo_v2 \
  --epochs 20 --steps_per_epoch 125 --batch_size 16 --k 8 \
  --w_llm 0 \
  --amp_ckpt checkpoints/amp_classifier_best.pt \
  --tox_ckpt checkpoints/toxicity_best.pt
```

`--w_llm 0` 关掉 LLM 评审，奖励只用规则项。这一步约 20–40 分钟（GPU），产出 `checkpoints/rl_grpo_v2/rl_grpo_best.pt`。

跑完立刻评估：

```bash
python evaluate.py --n_parents 200 --k 5 --conds 2,3,4 \
  --sft_ckpt checkpoints/sft_v2/best.pt \
  --grpo_ckpt checkpoints/rl_grpo_v2/rl_grpo_best.pt \
  --parents_csv data/paired_holdout.csv --holdout 1.0 \
  --out reports/eval_grpo_ruleonly
```

如果这一步 GRPO 相对 SFT 的结构指标提升 < 0.02，**不要往下走**——先诊断，否则加 LLM 只是在给一个不动的策略加噪声。

### 7.3 再跑带 LLM 评审的版本

只有 7.2 有效才做这一步。

```bash
python train_rl_grpo.py \
  --ckpt checkpoints/sft_v2/best.pt \
  --save_dir checkpoints/rl_grpo_v2_llm \
  --epochs 20 --steps_per_epoch 125 --batch_size 16 --k 8 \
  --w_llm 0.3 \
  --judge_every_n_steps 5 \
  --judge_subsample_groups 4 \
  --judge_lite \
  --amp_ckpt checkpoints/amp_classifier_best.pt \
  --tox_ckpt checkpoints/toxicity_best.pt
```

`--judge_lite` 跳过 A/B 互评，每组 3 次调用而非 5 次，省 40% 费用。

### 7.4 LLM 评审的成本

`--steps_per_epoch 125` × `--epochs 20` = 2500 步，每 5 步触发一次评审、每次 4 组：

- 触发次数：2500 / 5 = **500 次**
- 评审组数：500 × 4 = **2000 组**
- API 调用：2000 × 3（lite）= **6,000 次**；非 lite 为 **10,000 次**

每组要送完整的肽序列 + 结构描述给 A/B/C 三个模型（A=`gpt-5.1`，B=`deepseek-v4-pro`，C=`deepseek-v4-flash`）。按当前配置粗估 **$30–150**，取决于 prompt 长度和实际单价。先跑 `--judge_lite`，并考虑把 `--judge_every_n_steps` 调到 10（费用减半）。

评审记录会写进 `--record_dir`（默认 `logs/llm_judge/`）。这些记录是将来训奖励代理的训练数据，**不要删**。

---

## 8. 最终评估

```bash
python evaluate.py --n_parents 200 --k 5 --conds 2,3,4 \
  --sft_ckpt  checkpoints/sft_v2/best.pt \
  --grpo_ckpt checkpoints/rl_grpo_v2_llm/rl_grpo_best.pt \
  --parents_csv data/paired_holdout.csv --holdout 1.0 \
  --out reports/eval_final
```

再加一组 `generate` 模式，看原始生成策略（不看 `Pres`，看 `Compl` / `Orph%` / `Pairs`）：

```bash
python evaluate.py --mode generate --mask_rate 1.0 --n_parents 200 --k 5 \
  --sft_ckpt  checkpoints/sft_v2/best.pt \
  --grpo_ckpt checkpoints/rl_grpo_v2_llm/rl_grpo_best.pt \
  --parents_csv data/paired_holdout.csv --holdout 1.0 \
  --out reports/eval_final_generate
```

`generate` 模式是旧版唯一能看到 GRPO 起作用的地方（孤儿率低 3–5 个百分点）。新版应该看到更大差距。

---

## 9. 验收标准

| # | 检查项 | 通过条件 | 不通过怎么办 |
|---|---|---|---|
| 1 | 环境自检 | 117 项断言全过 + batch 数 = 2800/311 | 检查数据文件是否被覆盖 |
| 2 | SFT 收敛 | `V_sacc` > 0.45，`val_loss` 单调下降趋势 | 降 lr 到 5e-5，或查数据 |
| 3 | **SFT 打赢 random** | cond 2/3/4 的 `Rew` 均 > random | **停下**，回到数据生成查分布 |
| 4 | cond 4 末端锚定 | `Term%` < 50%（旧 62–63%） | 检查 `W_VALIDITY` 是否生效 |
| 5 | cond 4 合法性 | `Valid%` > 60%（旧 37%） | 同上 |
| 6 | 位置分布 | cond 2 `Offset` std > 0.20 | 数据仍偏 n/4，重查 `augment_stapled` |
| 7 | GRPO 有改进 | 结构指标相对 SFT 提升 ≥ 0.02 | 加 `--steps_per_epoch`，或查优势是否全 0 |
| 8 | 孤儿残基 | design 模式 `Orph%` < 1% | 检查 `design.py` 的 `staple_matching_size` 过滤 |

第 3 条是核心。**如果新数据训出的 SFT 仍然打不过 random，说明问题不在训练而在数据或奖励，不要靠加训练量硬顶。**

---

## 10. 显存不足时的降级顺序

按这个顺序降，别乱改：

1. `--batch_size 256 → 128 → 64`（SFT）
2. 确认没有别的进程占显存：`nvidia-smi`
3. `--batch_size` 降到 32 以下时，把 `--lr` 同步减半（1e-4 → 5e-5），否则梯度噪声太大
4. 仍 OOM：检查是否误把 `MAX_LEN` 调大了（应保持 50）
5. 最后手段：`torch.cuda.amp` 混合精度（当前代码未启用，需要改 `train.py`）

---

## 11. 已知的坑

| 坑 | 现象 | 处理 |
|---|---|---|
| `EPOCHS=1000` 默认值 | 跑起来停不下来 | 必须显式 `--epochs 30` |
| `RL_STEPS_PER_EPOCH=20` 默认值 | GRPO 训 20 epoch 只动了 260 步 | 必须显式 `--steps_per_epoch 125` |
| 缺分类器权重 | GRPO / evaluate 报 `FileNotFoundError: amp_classifier_best.pt` | 见 [2.2](#22-git-里没有的--必须单独传) |
| `WARMUP_STEPS` 无效 | 以为有 warmup，实际没有 | 代码里没实现，lr 全程恒定 |
| 评估非 held-out | 数字虚高 | 用 `make_holdout_split.py` + `--holdout 1.0` |
| `.env` 没配 | GRPO 的 LLM 阶段报鉴权失败 | 用 `--w_llm 0` 先跳过 |
| 评审记录目录 | 每次运行都往 `logs/llm_judge/` 追加 | 保留，是训奖励代理的数据 |
| Windows → Linux 换行符 | 已由 `.gitattributes` 统一为 LF | 无需处理 |

---

## 12. 执行顺序速查

```bash
# 0) 环境
pip install -r requirements.txt && cp .env.example .env
python tests/test_reward_rules.py

# 1) 切 holdout（推荐）
python data/make_holdout_split.py --frac 0.15

# 2) SFT
python train.py --data data/paired_train.csv --out_dir checkpoints/sft_v2 \
                --epochs 30 --batch_size 256 --lr 1e-4

# 3) 评估 SFT —— 关卡：必须打赢 random
python evaluate.py --n_parents 200 --k 5 --conds 2,3,4 \
  --sft_ckpt checkpoints/sft_v2/best.pt \
  --parents_csv data/paired_holdout.csv --holdout 1.0 \
  --systems sft,random --out reports/eval_sft_v2

# 4) GRPO（纯规则）
python train_rl_grpo.py --ckpt checkpoints/sft_v2/best.pt \
  --save_dir checkpoints/rl_grpo_v2 \
  --epochs 20 --steps_per_epoch 125 --batch_size 16 --k 8 --w_llm 0

# 5) 评估 GRPO —— 关卡：结构指标提升 ≥ 0.02
python evaluate.py --n_parents 200 --k 5 --conds 2,3,4 \
  --sft_ckpt checkpoints/sft_v2/best.pt \
  --grpo_ckpt checkpoints/rl_grpo_v2/rl_grpo_best.pt \
  --parents_csv data/paired_holdout.csv --holdout 1.0 \
  --out reports/eval_grpo_ruleonly

# 6) GRPO（带 LLM，仅当第 5 步有效）
python train_rl_grpo.py --ckpt checkpoints/sft_v2/best.pt \
  --save_dir checkpoints/rl_grpo_v2_llm \
  --epochs 20 --steps_per_epoch 125 --batch_size 16 --k 8 \
  --w_llm 0.3 --judge_lite --judge_every_n_steps 5

# 7) 最终评估
python evaluate.py --n_parents 200 --k 5 --conds 2,3,4 \
  --sft_ckpt checkpoints/sft_v2/best.pt \
  --grpo_ckpt checkpoints/rl_grpo_v2_llm/rl_grpo_best.pt \
  --parents_csv data/paired_holdout.csv --holdout 1.0 \
  --out reports/eval_final
```

把第 3、5、7 步产出的 `reports/*.md` 带回来，就能判断这次修复到底成不成。

---

## 13. 如果机器资源紧张

按性价比排序，最值得先做的两件事：

1. **只做第 2–3 步**（SFT + 评估）。这一步就能回答「数据修复是否有效」这个最关键的问题，成本 1–2 小时。GRPO 是在此之上的增量，如果 SFT 都打不过 random，GRPO 必然也打不过。
2. **GRPO 只跑 `--w_llm 0`**。结构指标不依赖 LLM，能省掉几乎全部 API 费用，且能在 40 分钟内给出「GRPO 到底有没有用」的答案。

奖励代理（`train_reward_proxy.py`）暂时不用碰：现有可用评审记录只有 3 组 / 30 条，远不够训练。等第 6 步跑完攒够数据再说。
