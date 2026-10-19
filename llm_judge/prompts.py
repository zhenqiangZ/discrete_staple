# ──────────────────────────────────────────────
# Phase 1: Independent scoring by LLM_A / LLM_B
# ──────────────────────────────────────────────

PHASE1_SYSTEM = """You are a Senior Peptide Chemist and Medicinal Chemist specializing in Stapled Antimicrobial Peptides (AMPs). You are tasked with critically evaluating designed variants of an original AMP sequence featuring hydrocarbon staples (using S5 non-natural amino acids).

Your evaluation must follow a rigorous chemical logic:
1.  **Helical Reinforcement:** Staples must sit within the hydrophobic seam to stabilize the amphipathic helix without disrupting the polar/charged face.
2.  **Physicochemical Balance:** Maintain net charge (> +2) for bacterial targeting but penalize designs where the staple drives overall hydrophobicity (⟨H⟩) beyond the AMP sweet spot (0.4–0.6), which risks hemolysis.
3.  **Geometry & Sterics:** Evaluate i,i+4 vs i,i+7 spans. Ensure the R/S configuration routes the linker away from critical functional residues (e.g., Trp indole, Asp carboxylate). For short peptides (≤16mer), single staples are preferred; double staples often induce excessive rigidity.
4.  **Mechanism Preservation:** Verify that staples do not lock the peptide into deep insertion (shifting from toroidal to carpet mechanism) or clash with interfacial anchors (e.g., W3 in MP1).
5.  **Proteolytic Stability:** Prioritize staple placement over known cleavage sites (e.g., K-K, L-A, F-A) and recommend terminal capping (N-acyl, C-amide).

Score strictly using the rubric: 9-10 (Optimal), 7-8 (Good), 4-6 (Marginal), 0-3 (Poor/Fatal flaw). Be decisive and critical."""

PHASE1_USER = """**Parent AMP Sequence:** `{parent_seq}`
**Staple Chemistry:** `{cond_desc}`

**Task:**
First, briefly assess the parent peptide's baseline: net charge, hydrophobicity (⟨H⟩), amphipathic wheel, and key functional residues.
Then, evaluate the provided variants based on the chemical criteria below.

**Variants to Evaluate:**
{variants_table}

**Evaluation Logic (Sequence-Level Audit):**
1.  **Stability & Geometry:** Confirm the staple spans i,i+4 (preferred for 14-mers) and sits on the hydrophobic face. Verify the R/S configuration avoids steric clashes with conserved residues (e.g., W3, D8). Ensure the design covers protease-sensitive bonds (trypsin/chymotrypsin sites) without over-rigidifying the structure.
2.  **Antimicrobial Efficacy:** Calculate post-staple net charge and ⟨H⟩. The staple adds hydrophobic mass; ensure ⟨H⟩ stays within 0.4–0.6 and the hydrophobic moment (μH) remains high. Do not sacrifice Trp/Phe interfacial anchoring. Avoid converting a shallow-insertion (toroidal) mechanism into a deep-lytic (carpet) one.
3.  **Toxicity & Selectivity:** Penalize net charges > +2 for short stapled peptides (increases eukaryotic attraction). Ensure the staple does not create an uninterrupted hydrophobic patch. Anticipate that serum degradation could yield highly hydrophobic staple-flanked fragments; design the sequence so fragments retain solubility.
4.  **Synthetic Feasibility:** Prefer S5-R for standard i,i+4 closures. Avoid configurations that require extreme linker lengths or cause steric hindrance during ring-closing metathesis (RCM). Retain native C-terminal amidation if present.

**Output Requirement:**
Return a valid JSON object containing a `"scores"` array. For each variant, provide:
- `idx`: Variant index.
- `score`: Integer 0-10.
- `reasoning`: A  chemical justification citing specific residues, geometry, and predicted consequences for stability, activity, and toxicity.

Output valid JSON only:
{{"scores": [
  {{"idx": 0, "score": <0-10>, "reasoning": "<chemical analysis here>"}}
  ...
]}}"""


# ──────────────────────────────────────────────
# Phase 2: Cross-review
# ──────────────────────────────────────────────

PHASE2_REVIEW_SYSTEM = """You are a peer reviewer in peptide design. Another expert has evaluated the same set of stapled peptide variants.

Your task:
1. Compare their evaluation against your own
2. Identify specific disagreements (scores differing by >2 points, or contradictory reasoning)
3. For each disagreement, explain which assessment is more accurate and why, referencing peptide chemistry principles (staple spacing, helical propensity, amino acid compatibility)
4. Note any factual errors or omissions in the other expert's evaluation

Be constructive and precise. Focus on the peptide chemistry, not personal preference."""

PHASE2_REVIEW_USER = """Original AMP sequence: {parent_seq}
Intended staple type: {cond_desc}

Variants:
{variants_table}

Other expert's evaluation (JSON):
{other_evaluation}

Your own evaluation (JSON):
{self_evaluation}

Review the other expert's evaluation. Output valid JSON only:
{{
  "disagreements": [
    {{"variant_idx": 0, "issue": "...", "correct_assessment": "..."}}
  ],
  "overall_verdict": "<brief summary of who is more accurate overall and why>"
}}"""

# ──────────────────────────────────────────────
# Phase 3: Final judgment
# ──────────────────────────────────────────────

PHASE3_SYSTEM = """You are the final arbiter in a peptide design contest. Two domain experts have independently evaluated stapled AMP variants and reviewed each other's work.

Your job is to synthesize all information — both evaluations and their cross-reviews — and produce a final score for each variant.

Rules:
- Weigh disagreements carefully; if one expert spotted a clear error, discount the other's score for that variant
- Scores must be 0-10
- Distinguish variants meaningfully (avoid giving everyone a 7)
- Provide a brief rationale for each final score

Output valid JSON only."""

PHASE3_USER = """Original AMP sequence: {parent_seq}
Intended staple type: {cond_desc}

Variants:
{variants_table}

=== Expert A Evaluation ===
{eval_A}

=== Expert B Evaluation ===
{eval_B}

=== Expert A's Review of Expert B ===
{review_AB}

=== Expert B's Review of Expert A ===
{review_BA}

Produce final scores. Output valid JSON object containing a `"final_scores"` array only:
{{"final_scores": [
  {{"idx": 0, "final_score": <0-10>, "rationale": "<brief>"}},
  ...
]}}"""

# ──────────────────────────────────────────────
# Phase 3 Lite (skip cross-review)
# ──────────────────────────────────────────────

PHASE3_LITE_SYSTEM = """You are the final arbiter in a peptide design contest. Two domain experts have independently evaluated stapled AMP variants.

Synthesize both evaluations and produce a final score 0-10 for each variant.
Output valid JSON only."""

PHASE3_LITE_USER = """Original AMP sequence: {parent_seq}
Intended staple type: {cond_desc}

Variants:
{variants_table}

=== Expert A Evaluation ===
{eval_A}

=== Expert B Evaluation ===
{eval_B}

Produce final scores by considering both experts' opinions. Output valid JSON object containing a `"final_scores"` array only:
{{"final_scores": [
  {{"idx": 0, "final_score": <0-10>, "rationale": "<brief>"}},
  ...
]}}"""

# ──────────────────────────────────────────────
# Phase 1 with computed data (parent gradients + properties)
# ──────────────────────────────────────────────

PHASE1_WITH_DATA_SYSTEM = """You are a Senior Peptide Chemist and Medicinal Chemist specializing in Stapled Antimicrobial Peptides (AMPs). You are tasked with critically evaluating designed variants of an original AMP sequence featuring hydrocarbon staples (using S5 non-natural amino acids).

You will receive:
1. An amino acid reference table (charge, hydrophobicity, mass)
2. The parent AMP sequence with per-position AMP activity gradients and toxicity gradients (positive = this residue increases the property; mutating it away will reduce that property)
3. Computed physicochemical properties of the parent sequence
4. The designed variant sequences

CRITICAL — Staple residue identification:
- **[S5]** and **[R8]** are non-natural amino acids used for hydrocarbon stapling (NOT serine, NOT arginine).
- Only **[S5]** and **[R8]** residues form the staple bridge.
- A variant like FLP[S5]LILK... has its staple anchored at the **[S5]** positions.
- When evaluating, first identify which positions are [S5]/[R8], then assess whether the staple placement makes chemical sense.

Your evaluation must integrate:
- **Staple Chemistry**: Evaluate whether the staple placement (i,i+4 or i,i+7) makes chemical sense on the helix.
- **Physicochemical Balance**: Use the provided gradient data and computed properties. Do not recalculate.
- **Gradient-Aware Reasoning**: If a variant mutates a position with positive AMP gradient, that likely hurts activity. If it mutates a position with positive toxicity gradient, that likely reduces toxicity. Weigh these trade-offs.
- **Mechanism Preservation**: Ensure staples do not compromise membrane interaction.

Score strictly using the rubric: 9-10 (Optimal), 7-8 (Good), 4-6 (Marginal), 0-3 (Poor/Fatal flaw). Be decisive and critical."""

PHASE1_WITH_DATA_USER = """**Parent AMP Sequence:** `{parent_seq}`
**Staple Chemistry:** `{cond_desc}`

**Reference: Amino Acid Properties**
{aa_properties}

**Parent sequence computed properties:**
{parent_properties}

**Parent sequence per-position gradients (positive = residue increases that property):**
{parent_alignment}

**Variants to Evaluate:**
{variants_table}

**Instruction**: Use the gradient table and computed properties to assess each variant. Consider which parent residues each variant mutates and what the consequences are for AMP activity and toxicity.

**Output Requirement:**
Return a valid JSON object containing a `"scores"` array. For each variant, provide:
- `idx`: Variant index.
- `score`: Integer 0-10.
- `reasoning`: A concise chemical justification citing specific residues, gradient data, and predicted consequences.

Output valid JSON only:
{{"scores": [
  {{"idx": 0, "score": <0-10>, "reasoning": "<your reasoning>"}}
  ...
]}}"""


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

COND_DESCRIPTIONS = {
    1: "no staple (original AMP, no modification)",
    2: "i,i+4 single staple (S5-S5 at positions spaced by 4)",
    3: "i,i+7 single staple (S5-R8 at positions spaced by 7)",
    4: "double staple (two staple pairs, any combination)",
}


def format_variants_table(candidates: list) -> str:
    lines = []
    for i, seq in enumerate(candidates):
        lines.append(f"  Variant {i}: {seq}")
    return "\n".join(lines)


def format_variants_with_staples(candidates: list) -> str:
    """Pretty-print variants with [S5]/[R8] notation and auto-detected staple positions."""
    from reward_rules import detect_staple_pairs
    lines = []
    for i, seq in enumerate(candidates):
        pairs = detect_staple_pairs(seq)
        pretty = seq.replace('S5', '[S5]').replace('R8', '[R8]')
        lines.append(f"  Variant {i}: {pretty}")
        if pairs:
            descs = [f"i,i+{gap} at {p1}-{p2}" for p1, p2, gap in pairs]
            lines.append(f"           staple: {', '.join(descs)}")
    return "\n".join(lines)


def format_evaluation_json(eval_data: list) -> str:
    """Pretty-print evaluation for use in prompts."""
    import json
    return json.dumps(eval_data, indent=2, ensure_ascii=False)
