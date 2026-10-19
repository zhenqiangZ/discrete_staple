import glob
import json
import os
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from llm_judge.score_scale import SCALE_RECOVERED, resolve_scores


def load_latest_record(record_dir: str) -> dict:
    files = sorted(glob.glob(os.path.join(record_dir, "judge_records_*.jsonl")))
    if not files:
        raise FileNotFoundError(f"No judge records found in {record_dir}")
    with open(files[-1], "r", encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        raise ValueError(f"Empty record file: {files[-1]}")
    return json.loads(lines[-1])


def _extract_scores(results, key="score"):
    if isinstance(results, dict):
        for v in results.values():
            if isinstance(v, list):
                results = v
                break
    out = {}
    for item in results:
        if isinstance(item, dict):
            idx = int(item.get("idx", -1))
            s = float(item.get(key, item.get("final_score", 5.0)))
            out[idx] = s
    return out


def _extract_reasonings(results, key="reasoning"):
    if isinstance(results, dict):
        for v in results.values():
            if isinstance(v, list):
                results = v
                break
    out = {}
    for item in results:
        if isinstance(item, dict):
            idx = int(item.get("idx", -1))
            out[idx] = item.get(key, "")
    return out


def _get_variant_labels(record):
    return [c.get("seq", f"V{i}") for i, c in enumerate(record.get("candidates", []))]


def plot_score_matrix(record, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    variants = _get_variant_labels(record)
    K = len(variants)
    p1 = record.get("phase1", {})
    p3 = record.get("phase3", {})

    a_results = p1.get("llm_a", {}).get("results", {})
    b_results = p1.get("llm_b", {}).get("results", {})
    c_results = p3.get("llm_c", {}).get("results", {})

    scores_a = _extract_scores(a_results)
    scores_b = _extract_scores(b_results)
    scores_c = _extract_scores(c_results, "final_score")

    raw = np.full((K, 3), np.nan)
    for i in range(K):
        raw[i, 0] = scores_a.get(i, np.nan)
        raw[i, 1] = scores_b.get(i, np.nan)
        raw[i, 2] = scores_c.get(i, np.nan)

    fig, ax = plt.subplots(figsize=(max(6, K * 1.2), 5))
    im = ax.imshow(raw, cmap="RdYlGn", vmin=0, vmax=10, aspect="auto")

    ax.set_xticks(range(3))
    ax.set_xticklabels(["LLM A", "LLM B", "LLM C"], fontsize=10)
    ax.set_yticks(range(K))
    yticklabels = [f"V{i}\n{s[:12]}..." for i, s in enumerate(variants)]
    ax.set_yticklabels(yticklabels, fontsize=8)

    for i in range(K):
        for j in range(3):
            v = raw[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        fontsize=9, color="black" if 3 < v < 7 else "white")

    ax.set_title("Score Matrix: Raw Scores per Expert", fontsize=12)
    fig.colorbar(im, ax=ax, label="Score (0-10)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_score_evolution(record, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    variants = _get_variant_labels(record)
    K = len(variants)
    p1 = record.get("phase1", {})
    p3 = record.get("phase3", {})

    a_results = p1.get("llm_a", {}).get("results", {})
    b_results = p1.get("llm_b", {}).get("results", {})
    c_results = p3.get("llm_c", {}).get("results", {})

    scores_a = _extract_scores(a_results)
    scores_b = _extract_scores(b_results)
    scores_c = _extract_scores(c_results, "final_score")
    fin = record.get("final_scores", [])
    resolved, fin_scale = resolve_scores(record)
    if resolved is not None:
        fin = resolved
    fin_label = "Final (recovered)" if fin_scale == SCALE_RECOVERED else "Final"

    x = np.arange(K)
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(8, K * 1.5), 5))
    ax.bar(x - 1.5 * width, [scores_a.get(i, 0) for i in range(K)], width, label="LLM A")
    ax.bar(x - 0.5 * width, [scores_b.get(i, 0) for i in range(K)], width, label="LLM B")
    ax.bar(x + 0.5 * width, [scores_c.get(i, 0) for i in range(K)], width, label="LLM C")
    ax.bar(x + 1.5 * width, fin if len(fin) == K else [0] * K, width, label=fin_label)

    ax.set_xticks(x)
    ax.set_xticklabels([f"V{i}" for i in range(K)], fontsize=9)
    ax.set_ylabel("Score (0-10)")
    ax.set_title("Score Evolution: A → B → C → Final")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _parse_disagreements(review_dict):
    items = []
    for entry in review_dict.get("disagreements", []):
        if isinstance(entry, dict):
            items.append(entry)
        elif isinstance(entry, str):
            items.append({"text": entry})
    return items


def plot_cross_review(record, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    variants = _get_variant_labels(record)
    K = len(variants)
    p1 = record.get("phase1", {})
    p2 = record.get("phase2", {})

    a_results = p1.get("llm_a", {}).get("results", {})
    b_results = p1.get("llm_b", {}).get("results", {})

    scores_a = _extract_scores(a_results)
    scores_b = _extract_scores(b_results)

    review_ab = p2.get("review_ab", {})
    review_ba = p2.get("review_ba", {})

    x = np.arange(K)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, K * 1.5), 5))
    bars_a = ax.bar(x - width / 2, [scores_a.get(i, 0) for i in range(K)], width, label="LLM A", color="#4C72B0")
    bars_b = ax.bar(x + width / 2, [scores_b.get(i, 0) for i in range(K)], width, label="LLM B", color="#DD8452")

    dis_ab = _parse_disagreements(review_ab)
    dis_ba = _parse_disagreements(review_ba)

    flagged_a = set()
    flagged_b = set()
    notes = []
    for d in dis_ab:
        vi = d.get("variant_idx")
        if vi is not None:
            flagged_b.add(int(vi))
            notes.append((int(vi), "A→B", d.get("text", d.get("reason", ""))[:60]))
    for d in dis_ba:
        vi = d.get("variant_idx")
        if vi is not None:
            flagged_a.add(int(vi))
            notes.append((int(vi), "B→A", d.get("text", d.get("reason", ""))[:60]))

    y_max = max(
        max([scores_a.get(i, 0) for i in range(K)] + [0]),
        max([scores_b.get(i, 0) for i in range(K)] + [0]),
    ) + 1.5

    for i in range(K):
        if i in flagged_a:
            ax.plot(i - width / 2, scores_a.get(i, 0) + 0.3, "rv", markersize=8)
        if i in flagged_b:
            ax.plot(i + width / 2, scores_b.get(i, 0) + 0.3, "r^", markersize=8)

    if notes:
        note_str = "\n".join(
            f"  V{v} {src}: {txt}" for v, src, txt in notes
        )
        ax.text(0.02, 0.98, f"Disagreements:\n{note_str}", transform=ax.transAxes,
                fontsize=7, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels([f"V{i}\n{s[:10]}" for i, s in enumerate(variants)], fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title("Cross-Review: A vs B with Disagreement Flags")
    ax.legend(fontsize=8)
    ax.set_ylim(0, y_max)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_expert_agreement(record, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    p1 = record.get("phase1", {})
    p3 = record.get("phase3", {})
    a_results = p1.get("llm_a", {}).get("results", {})
    b_results = p1.get("llm_b", {}).get("results", {})
    c_results = p3.get("llm_c", {}).get("results", {})
    scores_a = _extract_scores(a_results)
    scores_b = _extract_scores(b_results)
    scores_c = _extract_scores(c_results, "final_score")

    keys = sorted(set(scores_a) | set(scores_b) | set(scores_c))
    va = np.array([scores_a.get(k, np.nan) for k in keys])
    vb = np.array([scores_b.get(k, np.nan) for k in keys])
    vc = np.array([scores_c.get(k, np.nan) for k in keys])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    pairs = [("A vs B", va, vb), ("A vs C", va, vc), ("B vs C", vb, vc)]
    for ax, (title, x, y) in zip(axes, pairs):
        valid = ~(np.isnan(x) | np.isnan(y))
        ax.scatter(x[valid], y[valid], c="#4C72B0", s=40)
        min_v = min(np.nanmin(x), np.nanmin(y))
        max_v = max(np.nanmax(x), np.nanmax(y))
        ax.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=0.8)
        ax.set_xlabel(title.split(" vs ")[0])
        ax.set_ylabel(title.split(" vs ")[1])
        ax.set_title(title, fontsize=10)
        ax.set_aspect("equal")
        for i, k in enumerate(keys):
            if valid[i]:
                ax.annotate(f"V{k}", (x[i], y[i]), fontsize=6, alpha=0.7)
    fig.suptitle("Expert Agreement (diagonal = perfect agreement)", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def export_reasoning_report(record, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    variants = _get_variant_labels(record)
    K = len(variants)
    p1 = record.get("phase1", {})
    p2 = record.get("phase2", {})
    p3 = record.get("phase3", {})
    fin = record.get("final_scores", [])
    resolved, fin_scale = resolve_scores(record)
    if resolved is not None:
        fin = resolved
    fin_label = "Final score (recovered)" if fin_scale == SCALE_RECOVERED else "Final score"
    parent = record.get("parent_seq", "")
    cond_desc = record.get("cond_desc", "")

    a_results = p1.get("llm_a", {}).get("results", {})
    b_results = p1.get("llm_b", {}).get("results", {})
    c_results = p3.get("llm_c", {}).get("results", {})
    reas_a = _extract_reasonings(a_results)
    reas_b = _extract_reasonings(b_results)
    reas_c = _extract_reasonings(c_results, "reasoning")
    scores_a = _extract_scores(a_results)
    scores_b = _extract_scores(b_results)
    scores_c = _extract_scores(c_results, "final_score")

    review_ab = p2.get("review_ab", {})
    review_ba = p2.get("review_ba", {})

    lines = []
    lines.append("=" * 72)
    lines.append("LLM JUDGE REASONING REPORT")
    lines.append("=" * 72)
    lines.append(f"Parent: {parent}")
    lines.append(f"Cond:   {cond_desc}")
    lines.append(f"Variants: {K}")
    lines.append(f"Models: A={p1.get('llm_a', {}).get('model','')} | "
                 f"B={p1.get('llm_b', {}).get('model','')} | "
                 f"C={p3.get('llm_c', {}).get('model','')}")
    lines.append("=" * 72)

    for i in range(K):
        lines.append(f"\n── Variant {i}: {variants[i][:50]}")
        if i < len(fin):
            lines.append(f"    {fin_label}: {fin[i]:.2f}")
        lines.append("")
        for expert, reas, scores, label in [
            ("LLM A", reas_a, scores_a, "Score"),
            ("LLM B", reas_b, scores_b, "Score"),
            ("LLM C", reas_c, scores_c, "Final"),
        ]:
            s = scores.get(i, "N/A")
            r = reas.get(i, "(no reasoning)")
            if isinstance(s, float):
                lines.append(f"  [{label}] {expert}: {s:.2f}")
            else:
                lines.append(f"  [{label}] {expert}: {s}")
            wrapped = textwrap.fill(r, width=68, initial_indent="    ", subsequent_indent="    ")
            lines.append(wrapped)
            lines.append("")

    lines.append(f"\n{'=' * 72}")
    lines.append("CROSS-REVIEW (Phase 2)")
    lines.append(f"{'=' * 72}")
    lines.append(f"\nLLM A reviewing LLM B:")
    lines.append(f"  Verdict: {review_ab.get('overall_verdict', 'N/A')}")
    for d in _parse_disagreements(review_ab):
        vi = d.get("variant_idx", "?")
        txt = d.get("text", d.get("reason", ""))
        lines.append(f"  - V{vi}: {txt}")
    lines.append(f"\nLLM B reviewing LLM A:")
    lines.append(f"  Verdict: {review_ba.get('overall_verdict', 'N/A')}")
    for d in _parse_disagreements(review_ba):
        vi = d.get("variant_idx", "?")
        txt = d.get("text", d.get("reason", ""))
        lines.append(f"  - V{vi}: {txt}")

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
