#!/usr/bin/env python3
"""Batch evaluate four-peptides from sample.csv for hNav1.7 inhibitory
potential. Results are accumulated in a dict and persisted as JSON after each
batch; the CSV is written only at the very end, in the original input order.

Property-based prompt (no explicit amino-acid bias). Uses the openai SDK.
"""
import argparse
import csv
import json
import os

from openai import OpenAI

SYSTEM_PROMPT = """# Role

你是一位顶级的离子通道结构药理学专家，擅长基于结构的药物设计（SBDD），特别是在电压门控钠通道（Nav）毒素结合位点分析方面。

# Task

请评估给定的多条四肽序列是否可能抑制人源 Nav1.7 (hNav1.7) 通道。由于四肽尺寸极小，你需要分别评估每条序列结合在 **Site 1 (TTX/STX 位点)** 和 **Site E (孔阻滞剂受体位点)** 的可能性，并给出综合抑制潜力评分。

# Background Knowledge

## 1. hNav1.7 关键结合位点序列与特征

### Site 1 (TTX/STX 结合口袋)

*   **位置**：选择性滤器（SF）正上方，由四个重复单元（DI-DIV）的 P1-P2 胞外环围成。
*   **核心基序**：由四个重复单元各贡献一个关键残基组成的选择性滤器环（其中含一个天冬氨酸、一个谷氨酸、一个赖氨酸、一个丙氨酸）。
*   **关键互作特征**：
    *   一个位于 DI 的芳香残基，可与配体暴露的胍基形成 π-阳离子相互作用，是决定此类毒素敏感性的关键开关（某些抗性亚型此处为半胱氨酸，导致敏感性丧失）。
    *   一个位于 DIII 的疏水残基（hNav1.7 独有特征，其他亚型多为酸性残基）：该疏水环境破坏了某些胍基类毒素在其他亚型中形成的氢键网络，使 hNav1.7 对这类毒素极不敏感，但对另一类胍基毒素高度敏感。
*   **配体应具备的性质**：需要刚性骨架、一个暴露的强正电中心（如胍基），以及 2-3 个精确定位的氢键供体（如羟基）。

### Site E (KIIIA 结合口袋)

*   **位置**：孔结构域的胞外前庭（Outer Vestibule），由 DI-DIII 的胞外环围成，**不与 DIV 接触**（偏心结合）。
*   **关键互作特征（基于 Nav1.2 结构外推至 hNav1.7）**：
    *   **DI**：一个芳香残基（与配体芳香侧链形成 π-π 堆积），一个酸性残基（氢键）。
    *   **DII**：一个酸性残基（与配体碱性侧链形成盐桥，也是上述选择性滤器环的一部分），其邻近另一个酸性残基。
    *   **DIII**：两个芳香残基（与配体碱性 / 芳香侧链形成阳离子-π 堆积），以及若干决定亚型选择性的疏水残基。
*   **配体应具备的性质**：需要一个正电侧链（与通道酸性残基形成盐桥），以及一个芳香侧链（与通道 Tyr 形成堆积）。

## 2. 参考分子

*   **Site E 经典肽类毒素（约 16 肽）**：其核心药效特征为——一个插入选择性滤器的正电基团 + 一个与 Tyr 堆积的芳香残基 + 1-2 个与酸性残基成盐桥的碱性基团；对 Nav1.7 阻断不完全（约 90%）。
*   **Site 1 经典小分子毒素 A**：核心为胍基 + 多个羟基；对 hNav1.7 在 nM 级有效，依赖于胍基与选择性滤器环酸性残基的盐桥。
*   **Site 1 经典小分子毒素 B**：核心同样为胍基；但因 hNav1.7 DIII 疏水残基的排斥作用，几乎不阻断 hNav1.7。

# Evaluation Criteria for 4-peptides

## Site 1 倾向性判据

*   **正电中心**：需要暴露的强正电中心（如胍基或质子化胺基），能与选择性滤器环的酸性残基形成稳定盐桥。
*   **氢键能力**：需要 2-3 个精确定位的氢键供体（羟基或酰胺基团）。
*   **刚性需求**：四肽柔性极高，不利于 Site 1 的精确几何匹配；较低的构象柔性（如通过环化或刚性骨架获得）是重要加分项。

## Site E 倾向性判据

*   **电荷互补**：至少一个可在生理 pH 质子化的碱性侧链，提供正电荷与通道酸性残基成盐桥。
*   **芳香堆积**：至少一个富电子芳香侧链，与通道 Tyr 形成 π-π / 阳离子-π 堆积。
*   **几何匹配**：正电侧链与芳香侧链的相对位置应允许同时接触 DII 酸性残基与 DI 芳香残基。

考虑氨基酸顺序以及空间几何构象的影响

# Input

以下是用户输入的四肽序列列表（共 N 条，必须全部依次独立评估，不得遗漏任何一条）。示例格式：

1. xxxx
2. xxxx
...

# Output Format

请严格输出如下 JSON 对象（不要输出任何额外文字或 Markdown 围栏，{"results": [...]} 是唯一输出）：

{
  "results": [
    {
      "peptide": "RWDK",
      "inhibition_score": 0-10,
      "favored_site": "Site E | Site 1 | 不明确",
      "site1_analysis": "结合 Site 1 的可能性与依据（基于性质判断）",
      "siteE_analysis": "结合 Site E 的可能性与依据（基于性质判断）",
      "key_residue_mapping": "根据四肽侧链的物理化学性质（电荷 / 芳香性 / 氢键能力 / 疏水性）推断其可能与 hNav1.7 哪些关键残基（酸性 / 芳香 / 疏水）发生盐桥、氢键、π-阳离子或疏水作用，而非仅按残基名称归类",
      "comparison": "与 Site E / Site 1 经典毒素核心药效性质的对比",
      "optimization": "为增强抑制活性，在性质层面建议如何修饰（如增加正电中心、引入芳香侧链、环化降低柔性、端基修饰等）"
    }
  ]
}

results 数组长度必须等于输入序列条数，且顺序与输入编号一致。
"""

FIELDS = [
    "peptide", "inhibition_score", "favored_site",
    "site1_analysis", "siteE_analysis", "key_residue_mapping",
    "comparison", "optimization",
]

norm = lambda s: (s or "").strip().lower()


def read_peptides(path):
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if row and row[0].strip():
                out.append(row[0].strip())
    return out


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def error_row(pep, msg=""):
    return {
        "peptide": pep, "inhibition_score": "", "favored_site": "error",
        "site1_analysis": "", "siteE_analysis": "",
        "key_residue_mapping": f"error: {msg}" if msg else "未返回",
        "comparison": "", "optimization": "",
    }


def main():
    ap = argparse.ArgumentParser(description="从 sample.csv 批量评测四肽（字典累积+JSON持久化+末态转CSV）")
    ap.add_argument("--csv", default="sample.csv")
    ap.add_argument("--output", default="results.csv", help="最终 CSV（按输入顺序）")
    ap.add_argument("--json", default="results.json", help="中间字典对象持久化文件")
    ap.add_argument("--batch-size", type=int, default=10)
    
    ap.add_argument("--api-key", default=os.environ.get("LLM_C_API_KEY", ""))
    ap.add_argument("--base-url", default="https://api.deepseek.com")
    
    # ap.add_argument("--api-key", default=os.environ.get("LLM_B_API_KEY", ""))
    # ap.add_argument("--base-url", default="https://opencode.ai/zen/go/v1/")
    
    # ap.add_argument("--api-key", default=os.environ.get("LLM_C_API_KEY", ""))
    # ap.add_argument("--base-url", default="https://inferaiapi.com/v1/")
    
    ap.add_argument("--model", default="deepseek-v4-flash")
    # ap.add_argument("--model", default="deepseek-v4-pro")
    
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--retries", type=int, default=2, help="每批失败重试次数")
    ap.add_argument("--no-json-mode", action="store_true")
    ap.add_argument("--resume", default=True, action="store_true", help="续跑：加载已有 JSON，error 条目重算")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    order = read_peptides(args.csv)
    print(f"读取 {len(order)} 条肽，批次={args.batch_size}")

    # 1) 载入中间字典对象
    results = {}
    if args.resume and os.path.exists(args.json):
        with open(args.json, encoding="utf-8") as f:
            loaded = json.load(f)
        # 保留非 error 的条目，error 条目留给本轮重算
        results = {k: v for k, v in loaded.items() if v.get("favored_site") != "error"}
        print(f"resume: 已有 {len(results)} 条成功结果")

    todo = [p for p in order if norm(p) not in results]
    print(f"待处理 {len(todo)} 条")

    # 2) dry-run：仅打印首批 prompt
    if args.dry_run:
        batch0 = todo[:args.batch_size]
        seq_block = "\n".join(f"{i+1}. {p}" for i, p in enumerate(batch0))
        user_content = (
            f"四肽序列列表（共 {len(batch0)} 条，必须全部依次独立评估，"
            f"不得遗漏任何一条）：\n{seq_block}\n\n"
            "请对列表中的每一条四肽序列都独立完成分析，并严格按上述 Output Format 的 JSON 对象格式输出。"
        )
        print("===== DRY-RUN 首批 MESSAGES =====")
        print("[system]\n" + SYSTEM_PROMPT)
        print("\n[user]\n" + user_content)
        return

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    # 3) 逐批累积字典 + 每批持久化 JSON
    for batch in chunks(todo, args.batch_size):
        last_err = ""
        for attempt in range(args.retries + 1):
            seq_block = "\n".join(f"{i+1}. {p}" for i, p in enumerate(batch))
            user_content = (
                f"四肽序列列表（共 {len(batch)} 条，必须全部依次独立评估，"
                f"不得遗漏任何一条）：\n{seq_block}\n\n"
                "请对列表中的每一条四肽序列都独立完成分析，并严格按上述 Output Format 的 JSON 对象格式输出。"
            )
            kwargs = dict(
                model=args.model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": user_content}],
                temperature=args.temperature,
                extra_body={"thinking": {"type": "disabled"}},
            )
            if not args.no_json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs)
            raw = resp.choices[0].message.content
            try:
                data = json.loads(raw)
                items = data.get("results", []) if isinstance(data, dict) else []
                by_pep = {norm(r.get("peptide")): r for r in items}
                last_err = ""
                break
            except json.JSONDecodeError:
                last_err = "JSON 解析失败"
                continue

        for p in batch:
            key = norm(p)
            r = by_pep.get(key) if last_err == "" else None
            results[key] = r if r is not None else error_row(p, last_err)

        # 每批结束立即持久化字典对象
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  批次完成（首肽 {batch[0]}），已持久化 {len(results)} 条 -> {args.json}")

    # 4) 最后按 sample.csv 原始顺序写出 CSV
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for p in order:
            r = results.get(norm(p), error_row(p))
            writer.writerow({k: r.get(k, "") for k in FIELDS})
    print(f"\n完成，CSV 已按输入顺序写入 {args.output}（共 {len(order)} 行）")


if __name__ == "__main__":
    main()