# Stapled peptide design — evaluation report

- Generated: 2026-10-19 12:54:54
- Mode: `design`
- Parents: 200 (seed=42, holdout=0.0)
- Candidates per parent: K=5
- Conditions: 2, 3, 4
- Runtime: 90.8s

## Caveat

All parents come from `data/paired_data.csv`, which the checkpoints were trained on. SFT/GRPO numbers are in-distribution and should be read as an upper bound.

## Structure

| System | Cond | N | Compl | Valid | Valid% | Pres | Rew | Pairs | Term% | Offset | Orph% | Fall% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sft | 2 | 1000 | 1.000 | 0.778 | 77.8% | 1.000 | 0.967 | 1.00 | 22.2% | 0.33 | 0.0% | 0.0% |
| sft | 3 | 1000 | 0.968 | 0.705 | 70.5% | 1.000 | 0.940 | 0.97 | 29.5% | 0.23 | 0.0% | 3.2% |
| sft | 4 | 1000 | 0.984 | 0.627 | 37.7% | 1.000 | 0.936 | 1.97 | 62.3% | 0.32 | 1.6% | 0.0% |
| grpo | 2 | 1000 | 1.000 | 0.780 | 78.0% | 1.000 | 0.967 | 1.00 | 22.0% | 0.32 | 0.0% | 0.0% |
| grpo | 3 | 1000 | 0.968 | 0.706 | 70.6% | 1.000 | 0.940 | 0.97 | 29.4% | 0.23 | 0.0% | 3.2% |
| grpo | 4 | 1000 | 0.980 | 0.616 | 36.8% | 1.000 | 0.932 | 1.96 | 63.2% | 0.32 | 2.0% | 0.0% |
| random | 2 | 1000 | 1.000 | 0.792 | 79.2% | 1.000 | 0.969 | 1.00 | 20.8% | 0.36 | 0.0% | 0.0% |
| random | 3 | 1000 | 1.000 | 0.705 | 70.5% | 1.000 | 0.956 | 1.00 | 29.5% | 0.24 | 0.0% | 0.0% |
| random | 4 | 1000 | 0.954 | 0.755 | 56.5% | 1.000 | 0.940 | 1.91 | 43.5% | 0.28 | 4.6% | 0.0% |

## Predicted function and properties

| System | Cond | N | AMP | AMP+ | Tox | Chg | H | uH |
|---|---|---|---|---|---|---|---|---|
| sft | 2 | 1000 | 0.840 | 84.8% | 0.681 | +3.01 | +0.022 | 5.249 |
| sft | 3 | 1000 | 0.848 | 85.2% | 0.696 | +3.00 | +0.061 | 5.392 |
| sft | 4 | 1000 | 0.769 | 77.4% | 0.628 | +2.52 | +0.131 | 4.823 |
| grpo | 2 | 1000 | 0.841 | 84.8% | 0.690 | +3.01 | +0.022 | 5.248 |
| grpo | 3 | 1000 | 0.849 | 85.3% | 0.697 | +3.00 | +0.060 | 5.365 |
| grpo | 4 | 1000 | 0.769 | 77.7% | 0.625 | +2.52 | +0.135 | 4.820 |
| random | 2 | 1000 | 0.846 | 85.4% | 0.679 | +3.00 | +0.036 | 5.326 |
| random | 3 | 1000 | 0.836 | 83.4% | 0.685 | +3.04 | +0.067 | 5.502 |
| random | 4 | 1000 | 0.801 | 80.6% | 0.633 | +2.56 | +0.166 | 4.856 |

## Diversity

| System | Cond | N | Uniq | Ident |
|---|---|---|---|---|
| sft | 2 | 1000 | 1.00 | 0.10 |
| sft | 3 | 1000 | 0.98 | 0.09 |
| sft | 4 | 1000 | 1.00 | 0.13 |
| grpo | 2 | 1000 | 1.00 | 0.09 |
| grpo | 3 | 1000 | 0.98 | 0.09 |
| grpo | 4 | 1000 | 1.00 | 0.13 |
| random | 2 | 1000 | 0.83 | 0.10 |
| random | 3 | 1000 | 0.74 | 0.09 |
| random | 4 | 1000 | 0.98 | 0.11 |

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
