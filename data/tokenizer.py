from typing import List

AMINO_ACIDS = {aa: i + 1 for i, aa in enumerate(
    ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
     'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
     'S5', 'R8']
)}
ID_TO_AA = {v: k for k, v in AMINO_ACIDS.items()}
PAD_TOKEN = 0
MASK_TOKEN = 23
VOCAB_SIZE = 24


RESIDUE_TOKENS = ('S5', 'R8')


def split_residues(seq: str) -> List[str]:
    """Split a peptide sequence into residue tokens.

    S5 and R8 are two-character non-natural residues and must stay intact;
    every other residue is one character.
    """
    residues = []
    i = 0
    while i < len(seq):
        if seq[i:i + 2] in RESIDUE_TOKENS:
            residues.append(seq[i:i + 2])
            i += 2
        else:
            residues.append(seq[i])
            i += 1
    return residues


def encode(seq: str) -> List[int]:
    return [AMINO_ACIDS.get(res, PAD_TOKEN) for res in split_residues(seq)]


def decode(tokens: List[int]) -> str:
    chars = []
    for t in tokens:
        if t == PAD_TOKEN:
            break
        if t == MASK_TOKEN:
            chars.append('?')
        else:
            chars.append(ID_TO_AA.get(t, '-'))
    return ''.join(chars)

def decode_pretty(tokens: List[int]) -> str:
    chars = []
    for t in tokens:
        if t == PAD_TOKEN:
            break
        if t == MASK_TOKEN:
            chars.append('?')
        elif t == 21:
            chars.append('[S5]')
        elif t == 22:
            chars.append('[R8]')
        else:
            chars.append(ID_TO_AA.get(t, '-'))
    return ''.join(chars)


def pad(seq: List[int], max_len: int) -> List[int]:
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [PAD_TOKEN] * (max_len - len(seq))
