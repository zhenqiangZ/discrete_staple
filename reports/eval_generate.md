# Stapled peptide design — evaluation report

- Generated: 2026-10-19 12:56:46
- Mode: `generate`
- Parents: 200 (seed=42, holdout=0.0)
- Candidates per parent: K=2
- Conditions: 2, 3, 4
- Runtime: 107.0s

## Caveat

Masked reconstruction with `mask_rate=1.0`. With partial masking the staple sites usually stay unmasked, so compliance is bounded by mask geometry rather than model capability — full masking (1.0) is the default for that reason.

`Pres` is **not** a quality signal here: masking removes the parent information, so a low value is expected.

All parents come from `data/paired_data.csv`, which the checkpoints were trained on. SFT/GRPO numbers are in-distribution and should be read as an upper bound.

## Structure

| System | Cond | N | Compl | Valid | Valid% | Pres | Rew | Pairs | Term% | Offset | Orph% | Fall% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sft | 2 | 400 | 0.752 | 0.810 | 74.8% | 0.156 | 0.553 | 1.06 | 25.2% | 0.21 | 50.4% | 0.0% |
| sft | 3 | 400 | 0.165 | 0.892 | 88.2% | 0.165 | 0.274 | 0.40 | 11.8% | 0.19 | 72.4% | 0.0% |
| sft | 4 | 400 | 0.618 | 0.668 | 55.5% | 0.163 | 0.466 | 1.41 | 44.5% | 0.28 | 52.6% | 0.0% |
| grpo | 2 | 400 | 0.810 | 0.846 | 77.8% | 0.155 | 0.586 | 1.15 | 22.2% | 0.24 | 45.0% | 0.0% |
| grpo | 3 | 400 | 0.265 | 0.881 | 86.2% | 0.154 | 0.319 | 0.55 | 13.8% | 0.18 | 67.4% | 0.0% |
| grpo | 4 | 400 | 0.667 | 0.665 | 52.5% | 0.165 | 0.491 | 1.52 | 47.5% | 0.28 | 49.7% | 0.0% |

## Predicted function and properties

| System | Cond | N | AMP | AMP+ | Tox | Chg | H | uH |
|---|---|---|---|---|---|---|---|---|
| sft | 2 | 400 | 0.812 | 79.8% | 0.728 | +3.73 | -0.055 | 5.170 |
| sft | 3 | 400 | 0.814 | 82.5% | 0.712 | +3.99 | -0.096 | 5.303 |
| sft | 4 | 400 | 0.821 | 83.8% | 0.692 | +3.28 | +0.043 | 4.742 |
| grpo | 2 | 400 | 0.778 | 76.8% | 0.700 | +3.61 | -0.046 | 5.289 |
| grpo | 3 | 400 | 0.815 | 82.0% | 0.689 | +3.76 | -0.044 | 5.537 |
| grpo | 4 | 400 | 0.812 | 82.2% | 0.670 | +3.00 | +0.099 | 4.394 |

## Diversity

| System | Cond | N | Uniq | Ident |
|---|---|---|---|---|
| sft | 2 | 400 | 0.86 | 0.19 |
| sft | 3 | 400 | 0.88 | 0.14 |
| sft | 4 | 400 | 0.98 | 0.22 |
| grpo | 2 | 400 | 0.84 | 0.20 |
| grpo | 3 | 400 | 0.88 | 0.15 |
| grpo | 4 | 400 | 0.97 | 0.24 |

## Metric definitions

| Metric | Definition |
|---|---|
| Compl | `staple_accuracy` — match between requested cond and design |
| Valid | `staple_site_validity` — gap in {4,7}, terminal margin, same helical face |
| Valid% | Fraction of designs with every staple pair valid |
| Pres | `preservation_score` — scaffold residues kept unchanged |
| Rew | `compute_reward` — rule-based scalar the RWR baseline optimises |
| Pairs | Mean number of detected staple pairs |
| Term% | Fraction with a staple touching residue 0 or L-1 |
| Offset | Mean normalised staple start position (0=N-term, 1=C-term) |
| Orph% | Fraction of S5/R8 residues in no detected pair (broken crosslink) |
| Fall% | Fraction identical to the parent (no design produced) |
| AMP | Mean AMP-classifier probability (higher is better) |
| AMP+ | Fraction predicted AMP-positive (p >= 0.5) |
| Tox | Mean toxicity-classifier probability (lower is better) |
| Chg | Mean net charge at pH 7 |
| H | Mean Eisenberg hydrophobicity <H> |
| uH | Mean hydrophobic moment (helical wheel) |
| Uniq | Distinct sequences / total designs |
| Ident | Mean pairwise residue identity (1 - diversity) |
