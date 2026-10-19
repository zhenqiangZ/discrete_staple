#!/usr/bin/env python3
"""Batch evaluation of 10 four-peptides for hNav1.7 inhibitory potential.

Property-based prompt (no explicit amino-acid bias). Uses the official
openai SDK (OpenAI-compatible). Outputs a JSON object {"results": [...]}.

Usage:
    python tests/test_nav1_7_batch.py --dry-run
    python tests/test_nav1_7_batch.py \
        --api-key sk-xxx --base-url https://api.deepseek.com --model deepseek-v4-flash
"""
import argparse
import json
import os

from openai import OpenAI

PEPTIDES = [
    "RWDK", "WKRR", "KWWR", "RWEK", "WKRD",
    "RKWP", "WWKR", "RERW", "KWRK", "RWYK",
]

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

## 2. 参考分子（Benchmark，仅以性质描述）

*   **Site E 经典肽类毒素（约 16 肽）**：其核心药效特征为——一个插入选择性滤器的正电基团 + 一个与 Tyr 堆积的芳香残基 + 1-2 个与酸性残基成盐桥的碱性基团；对 Nav1.7 阻断不完全（约 90%）。
*   **Site 1 经典小分子毒素 A**：核心为胍基 + 多个羟基；对 hNav1.7 在 nM 级有效，依赖于胍基与选择性滤器环酸性残基的盐桥。
*   **Site 1 经典小分子毒素 B**：核心同样为胍基；但因 hNav1.7 DIII 疏水残基的排斥作用，几乎不阻断 hNav1.7。

# Evaluation Criteria for 4-peptides

## Site 1 倾向性判据 (高难度)

*   **正电中心**：需要暴露的强正电中心（如胍基或质子化胺基），能与选择性滤器环的酸性残基形成稳定盐桥。
*   **氢键能力**：需要 2-3 个精确定位的氢键供体（羟基或酰胺基团）。
*   **刚性需求**：四肽柔性极高，不利于 Site 1 的精确几何匹配；较低的构象柔性（如通过环化或刚性骨架获得）是重要加分项。
*   **结论**：四肽结合 Site 1 的可能性极低，除非通过环化增加刚性；若结合，预计效力很弱 (µM-mM)。

## Site E 倾向性判据 (更可行)

*   **电荷互补**：至少一个可在生理 pH 质子化的碱性侧链，提供正电荷与通道酸性残基成盐桥。
*   **芳香堆积**：至少一个富电子芳香侧链，与通道 Tyr 形成 π-π / 阳离子-π 堆积。
*   **几何匹配**：正电侧链与芳香侧链的相对位置应允许同时接触 DII 酸性残基与 DI 芳香残基。
*   **结论**：四肽更可能结合 Site E，虽尺寸远小于经典肽类毒素，仍可能占据部分前庭产生弱抑制或作为分子探针。

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


def main():
    ap = argparse.ArgumentParser(description="hNav1.7 四肽批量抑制潜力评估（性质导向提示词）")
    ap.add_argument("--peptides", nargs="*", default=PEPTIDES,
                    help="四肽序列列表，默认内置 10 条")
    ap.add_argument("--api-key", default=os.environ.get("LLM_C_API_KEY", ""))
    ap.add_argument("--base-url", default="https://inferaiapi.com/v1/")
    # ap.add_argument("--base-url", default="https://opencode.ai/zen/go/v1/")
    ap.add_argument("--model", default="deepseek-v4-pro")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--no-json-mode", action="store_true",
                    help="关闭 response_format=json_object（当后端不支持时）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印拼装后的 prompt，不调用 API")
    args = ap.parse_args()

    seq_block = "\n".join(f"{i+1}. {p}" for i, p in enumerate(args.peptides))
    user_content = (
        f"四肽序列列表（共 {len(args.peptides)} 条，必须全部依次独立评估，"
        f"不得遗漏任何一条）：\n{seq_block}\n\n"
        "请对列表中的每一条四肽序列都独立完成分析，并严格按上述 Output Format 的 JSON 对象格式输出。"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    if args.dry_run:
        print("===== COMPOSED MESSAGES =====")
        for m in messages:
            print(f"\n--- [{m['role']}] ---\n{m['content']}")
        return

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    kwargs = dict(model=args.model, messages=messages, temperature=args.temperature)
    if not args.no_json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    resp = client.chat.completions.create(**kwargs)
    raw = resp.choices[0].message.content

    try:
        data = json.loads(raw)
        results = data.get("results", data) if isinstance(data, dict) else data
        print(f"解析到 {len(results)} 条结果：\n")
        for r in results:
            print(f"{r.get('peptide'):<6} 评分={r.get('inhibition_score')} "
                  f"位点={r.get('favored_site')}")
            print(f"       关键映射: {r.get('key_residue_mapping')}")
            print(f"       优化建议: {r.get('optimization')}\n")
    except json.JSONDecodeError:
        print("模型未返回合法 JSON，原始输出：\n", raw)


if __name__ == "__main__":
    main()