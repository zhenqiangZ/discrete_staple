"""
Test LLM judge pipeline with parent gradient data.

Edit variables below, then:
    python test_llm_judge.py
"""
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ── Edit these ──────────────────────────────────────────
PARENT_SEQ = "FIHHIIGGLFSAGKAIHRLIRRRRR"
CANDIDATES = [
    "FIIIKKSGGLFKKKAGAS5KKKS5IKK",
    "FIIIKKSGGLS5KKKS5GAS5KKKS5IKK",
    "FIIS5KKSS5GLFKKKAGAS5KKKS5IKK",
    "FIIS5KKSS5GLS5KKKS5GAS5KKKS5IKK",
    # "GLFAVIKKVAS5VIKS5L",
]
COND = 4                # 2=i,i+4, 3=i,i+7, 4=double
RECORD_DIR = "test_records"  # "" = no recording, or dir path like "test_records"

# Optional: load AMP/tox models to compute parent gradients for the prompt.
# Set to None to use original prompts (no gradient data).
AMP_CKPT = "checkpoints/amp_classifier_best.pt"
TOX_CKPT = "checkpoints/toxicity_best.pt"
# ─────────────────────────────────────────────────────────

import os
import sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_judge.evaluator import create_evaluator
from llm_judge.visualize import (
    load_latest_record,
    plot_score_matrix,
    plot_score_evolution,
    plot_cross_review,
    plot_expert_agreement,
    export_reasoning_report,
)
from predictors import load_predictor, batch_position_alignment
from data.tokenizer import encode, pad
from config import MAX_LEN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

evaluator = create_evaluator(lite_mode=False, record_dir=RECORD_DIR)

# Compute parent gradients if model checkpoints are available
extra_data = None
if AMP_CKPT and TOX_CKPT and os.path.exists(AMP_CKPT) and os.path.exists(TOX_CKPT):
    print("  Loading AMP/tox models for gradient computation...")
    amp_model = load_predictor(AMP_CKPT, device)
    tox_model = load_predictor(TOX_CKPT, device)

    parent_tokens = torch.tensor([pad(encode(PARENT_SEQ), MAX_LEN)], dtype=torch.long, device=device)
    
    # with torch.no_grad():
    amp_align = batch_position_alignment(amp_model, parent_tokens)
    tox_align = batch_position_alignment(tox_model, parent_tokens)

    seq_len = len(PARENT_SEQ)
    extra_data = {
        'amp_alignments': amp_align[0, :seq_len].tolist(),
        'tox_alignments': tox_align[0, :seq_len].tolist(),
    }
    print(f"  Gradients computed for {seq_len} positions")
else:
    print("  No AMP/tox checkpoints found, using original prompts (no gradient data)")

print(f"\nFULL mode | cond={COND} | extra_data={'yes' if extra_data else 'no'}")
print(f"  Parent: {PARENT_SEQ}")
for i, s in enumerate(CANDIDATES):
    print(f"  Variant {i}: {s}")
print()

scores = evaluator.evaluate_group(PARENT_SEQ, COND, CANDIDATES, extra_data)

print("── Results ──")
for i, (seq, s) in enumerate(zip(CANDIDATES, scores)):
    print(f"  Variant {i}: {seq}  →  score = {s:.4f}")
print()

# ── Visualize ─────────────────────────────────────
if RECORD_DIR and os.path.exists(RECORD_DIR):
    viz_dir = "visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    try:
        record = load_latest_record(RECORD_DIR)
        plot_score_matrix(record, os.path.join(viz_dir, "score_matrix.png"))
        plot_score_evolution(record, os.path.join(viz_dir, "score_evolution.png"))
        plot_cross_review(record, os.path.join(viz_dir, "cross_review.png"))
        plot_expert_agreement(record, os.path.join(viz_dir, "expert_agreement.png"))
        export_reasoning_report(record, os.path.join(viz_dir, "reasoning_report.txt"))
        print(f"  Visualizations saved to {viz_dir}/")
    except Exception as e:
        print(f"  Visualization skipped: {e}")
print()
