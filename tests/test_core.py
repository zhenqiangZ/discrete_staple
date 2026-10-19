"""Pure-Python unit tests for residue handling and property computation.

These tests intentionally avoid torch so they can run in any environment.

Run:
    python tests/test_core.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.tokenizer import split_residues, encode, decode
from compute_properties import (
    compute_net_charge,
    compute_hydrophobicity,
    compute_hydrophobic_moment,
    compute_molecular_weight,
    compute_all,
)
from format_prompt_data import format_parent_alignment


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    print(f"  ok  {name}")


def main():
    print("[1] residue splitting")
    check("plain sequence", split_residues("ACDE") == ["A", "C", "D", "E"])
    check("S5/R8 stay intact", split_residues("ACS5R8Y") == ["A", "C", "S5", "R8", "Y"])
    check("adjacent staples", split_residues("S5R8") == ["S5", "R8"])
    check("encode staples", encode("S5R8") == [21, 22])
    check("decode roundtrip", decode(encode("ACS5R8Y")) == "ACS5R8Y")

    print("[2] physicochemical properties")
    check("staples carry no charge", compute_net_charge("S5R8") == 0.0)
    check("K/R charge", compute_net_charge("S5KR") == 2.0)
    check("hydrophobicity S5", abs(compute_hydrophobicity("S5") - 0.64) < 1e-9)
    check("hydrophobicity R8", abs(compute_hydrophobicity("R8") - 1.06) < 1e-9)
    check(
        "molecular weight S5+R8",
        abs(compute_molecular_weight("S5R8") - (157.21 + 199.29)) < 1e-6,
    )
    check("hydrophobic moment is finite", compute_hydrophobic_moment("S5R8") >= 0.0)
    check(
        "compute_all keys",
        set(compute_all("S5R8KR")) == {"charge", "hydrophobicity", "hmoment", "mw"},
    )

    print("[3] prompt alignment table")
    table = format_parent_alignment("AS5K", [0.1, 0.2, 0.3], [-0.1, -0.2, -0.3])
    lines = table.splitlines()
    check("header + 3 residues", len(lines) == 4)
    check("S5 aligned as one row", "S5" in lines[2])
    short = format_parent_alignment("AS5K", [0.1], [0.2])
    check("short gradient list warns", "warn" in short)

    print("\nAll core tests passed.")


if __name__ == "__main__":
    main()
