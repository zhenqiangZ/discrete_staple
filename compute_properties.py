"""Physicochemical property computation from peptide sequence."""

import math

from data.tokenizer import split_residues

# Eisenberg consensus hydrophobicity scale (normalized)
EISENBERG = {
    'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90,
    'C': 0.29, 'Q': -0.85, 'E': -0.74, 'G': 0.48,
    'H': -0.40, 'I': 1.38, 'L': 1.06, 'K': -1.50,
    'M': 0.64, 'F': 1.19, 'P': 0.12, 'S': -0.18,
    'T': -0.05, 'W': 0.81, 'Y': 0.26, 'V': 1.08,
    'S5': 0.64,   # S5 ≈ Leu-like (hydrocarbon side chain)
    'R8': 1.06,   # R8 ≈ Leu/Ile-like (longer hydrocarbon chain)
}

# Monoisotopic masses (Da)
MASSES = {
    'A': 71.04, 'R': 156.10, 'N': 114.04, 'D': 115.03,
    'C': 103.01, 'Q': 128.06, 'E': 129.04, 'G': 57.02,
    'H': 137.06, 'I': 113.08, 'L': 113.08, 'K': 128.10,
    'M': 131.04, 'F': 147.07, 'P': 97.05, 'S': 87.03,
    'T': 101.05, 'W': 186.08, 'Y': 163.06, 'V': 99.07,
    'S5': 157.21, 'R8': 199.29,
}


def compute_net_charge(seq: str) -> float:
    """Net charge at pH 7. K/R=+1, D/E=-1, H=+0.5."""
    c = 0.0
    for aa in split_residues(seq):
        if aa in ('K', 'R'):
            c += 1.0
        elif aa in ('D', 'E'):
            c -= 1.0
        elif aa == 'H':
            c += 0.5
    return c


def compute_hydrophobicity(seq: str) -> float:
    """Mean Eisenberg consensus hydrophobicity ⟨H⟩."""
    residues = split_residues(seq)
    vals = [EISENBERG.get(aa, 0.0) for aa in residues]
    return sum(vals) / len(vals) if vals else 0.0


def compute_hydrophobic_moment(seq: str, angle: float = 100.0) -> float:
    """Eisenberg hydrophobic moment μH (helical wheel projection)."""
    rad = math.radians(angle)
    x = y = 0.0
    for i, aa in enumerate(split_residues(seq)):
        h = EISENBERG.get(aa, 0.0)
        theta = i * rad
        x += h * math.cos(theta)
        y += h * math.sin(theta)
    return math.sqrt(x * x + y * y)


def compute_molecular_weight(seq: str) -> float:
    """Sum of residue monoisotopic masses (N-term + C-term H2O not added)."""
    return sum(MASSES.get(aa, 0.0) for aa in split_residues(seq))


def compute_all(seq: str) -> dict:
    """Compute all physicochemical properties for a sequence."""
    return {
        'charge': compute_net_charge(seq),
        'hydrophobicity': compute_hydrophobicity(seq),
        'hmoment': compute_hydrophobic_moment(seq),
        'mw': compute_molecular_weight(seq),
    }
