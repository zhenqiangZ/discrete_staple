"""Format computed data for LLM judge prompts."""

from data.tokenizer import split_residues

AA_PROPERTIES_TABLE = """残基  单字母  净电荷  疏水性(Eisenberg)  质量(Da)
A      A       0      +0.62              71.0
R      R      +1      -2.53             156.1
N      N       0      -0.78             114.0
D      D      -1      -0.90             115.0
C      C       0      +0.29             103.0
Q      Q       0      -0.85             128.1
E      E      -1      -0.74             129.0
G      G       0      +0.48              57.0
H      H      +0.5    -0.40             137.1
I      I       0      +1.38             113.1
L      L       0      +1.06             113.1
K      K      +1      -1.50             128.1
M      M       0      +0.64             131.0
F      F       0      +1.19             147.1
P      P       0      +0.12              97.1
S      S       0      -0.18              87.0
T      T       0      -0.05             101.1
W      W       0      +0.81             186.1
Y      Y       0      +0.26             163.1
V      V       0      +1.08              99.1
S5     S5      0      +0.64             157.2
R8     R8      0      +1.06             199.3"""


def format_parent_alignment(parent_seq: str, amp_aligns: list, tox_aligns: list) -> str:
    """Format a per-position gradient table for the parent sequence.

    Uses residue-level splitting so S5/R8 stay aligned with one gradient value
    each.  If the gradient lists are shorter than the sequence, only the
    overlapping prefix is rendered and a warning line is appended.
    """
    residues = split_residues(parent_seq)
    n = min(len(residues), len(amp_aligns), len(tox_aligns))
    lines = ["位置  残基  AMP梯度     毒性梯度"]
    for i in range(n):
        lines.append(f"  {i:3d}  {residues[i]:4s}  {amp_aligns[i]:+8.4f}  {tox_aligns[i]:+8.4f}")
    if n < len(residues):
        lines.append(
            f"  [warn] only {n}/{len(residues)} positions have gradient values"
        )
    return "\n".join(lines)


def format_parent_properties(props: dict) -> str:
    """Format parent computed properties as a single line."""
    return (f"净电荷: {props['charge']:+g} | "
            f"平均疏水性 ⟨H⟩: {props['hydrophobicity']:.4f} | "
            f"疏水矩 μH: {props['hmoment']:.4f} | "
            f"分子量: {props['mw']:.1f} Da")
