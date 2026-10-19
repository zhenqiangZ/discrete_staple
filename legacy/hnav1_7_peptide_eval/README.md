# 归档：hNav1.7 四肽抑制潜力评估

本目录是从主仓库迁出的独立任务，与订书肽设计主线**无关**，仅作留存。

## 内容

| 文件 | 说明 |
|---|---|
| `test_4.py` | 内置 10 条四肽的批量评估脚本 |
| `test_4_1.py` | 从 `sample.csv` 批量评估，支持断点续跑（JSON 累积 → CSV 输出） |
| `sample.csv` | 999 条四肽输入 |
| `results.csv` / `results.json` | 第三次运行结果（UTF-8） |
| `results-0.csv` / `results-0.json` | 第一次运行结果（GBK 编码） |
| `results-1.csv` / `results-1.json` | 第二次运行结果（GBK 编码） |

## 运行

在本目录下执行（脚本默认使用相对路径）：

```bash
python test_4_1.py --csv sample.csv --output results.csv --json results.json
```

## 说明

- 评估目标是 hNav1.7 的 Site 1（TTX/STX 位点）与 Site E（KIIIA 位点），输出 0–10 抑制潜力评分。
- 三次运行使用了不同 prompt / 模型，均值分别约为 2.69、3.54、5.06，横向对比时需注意。
- `results-0.*` 与 `results-1.*` 为 GBK 编码，读取时需指定 `encoding='gbk'`。
- 脚本内含明文 API key，仅作归档；如要复用，请改为环境变量读取。
