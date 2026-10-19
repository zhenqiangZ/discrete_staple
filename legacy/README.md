# legacy/ —— 归档区

这里放的是**不再参与主线流程**的历史代码与数据。保留目的是可追溯（论文/报告里引用的旧数字能对上），
不要在此基础上继续开发。

> 这些脚本的 `sys.path` 与相对导入在移入本目录后**已失效**，直接运行会报 `ModuleNotFoundError`。
> 如需复现，请把它们复制回仓库根目录再运行。

## 目录说明

### `hnav1_7_peptide_eval/`
独立的 hNav1.7 四肽抑制潜力评估任务，与订书肽主线无关。含自己的 README。
归档时间：2026-10-19（阶段 1 工程整理）。

### `rl_discrete/`
早期「离散 RL」路径，已被 GRPO 取代。

| 文件 | 说明 |
|---|---|
| `train_rl_discrete.py` | 旧 RL 训练脚本，无任何模块引用它 |
| `checkpoints_rl/rl_best.pt` | 上述脚本的产物（19 MB，2026-06-26） |

归档原因：`train_rl_grpo.py` 是当前唯一在用的 RL 实现，两条路径的奖励/优势计算逻辑不同，
并存容易误用。当前基线是 `checkpoints/best.pt`（SFT）+ `checkpoints/rl_grpo/rl_grpo_best.pt`（GRPO）。

### `early_sampling/`
早期探索期的独立采样/测试脚本，均无模块引用。

| 文件 | 说明 |
|---|---|
| `sample_amp_design.py` | 基于 logit 梯度做 AMP 定向采样的原型 |
| `sample_amp_vote.py` | 多候选投票选优的原型 |
| `sample_based_amp.py` | 按 AMP 分类器分数做 rejection sampling 的原型 |
| `test_llm_judge.py` | LLM 评审团的冒烟测试脚本 |

归档原因：功能已被 `design.py`（候选枚举 + log-prob 排序）和 `evaluate.py` 覆盖。

## 不在本目录、但同样属于历史产物的文件

- `data/paired_data_legacy.csv`（82,603 条）：旧版配对数据，**现有检查点是用它训练的**。
  重训时用它可复现旧结果；新训练一律用 `data/paired_data.csv`。
- `data/raw/`：上游原始数据（DRAMP 导出、过滤前后的 JSON）。当前训练链只用
  `data/peptide_data.csv`，这三个文件仅用于数据溯源。
