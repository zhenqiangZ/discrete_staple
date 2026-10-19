"""Pure-Python unit tests for the evaluation core.

Covers metric definitions and the model-free random baseline.  No torch
required.

Run:
    python tests/test_eval_core.py
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_core import (
    count_staples, flatten_parents, has_terminal_anchor, mean_pairwise_identity,
    mean_staple_offset, orphan_staple_rate, random_designs,
    random_staple_placement, sequence_identity, summarize_designs,
    summarize_diversity, summarize_properties, unique_ratio,
)


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    print(f"  ok  {name}")


def main():
    print("[1] sequence identity")
    check("identical", sequence_identity("ACDEF", "ACDEF") == 1.0)
    check("disjoint", sequence_identity("AAAAA", "CCCCC") == 0.0)
    check("half", sequence_identity("ACDEF", "ACDGG") == 3 / 5)
    check("staples are single residues",
          sequence_identity("S5CDES5", "S5CDES5") == 1.0)
    check("length mismatch uses overlap",
          sequence_identity("ACDEF", "ACD") == 1.0)

    print("[2] diversity")
    check("unique ratio all distinct", unique_ratio(["A", "C", "D"]) == 1.0)
    check("unique ratio duplicates", unique_ratio(["A", "A", "C"]) == 2 / 3)
    check("single sequence identity is 1", mean_pairwise_identity(["ACDEF"]) == 1.0)
    check("identical pair identity is 1",
          mean_pairwise_identity(["ACDEF", "ACDEF"]) == 1.0)
    check("disjoint pair identity is 0",
          mean_pairwise_identity(["AAAAA", "CCCCC"]) == 0.0)

    print("[3] staple counting and terminal anchors")
    check("no staple", count_staples("ACDEF") == 0)
    check("one i,i+4", count_staples("S5CDES5") == 1)
    check("one i,i+7", count_staples("S5CDEFGHR8") == 1)
    check("N-terminal anchor flagged", has_terminal_anchor("S5CDES5FG") is True)
    check("internal pair not flagged", has_terminal_anchor("AS5CDS5FG") is False)
    check("no staple not flagged", has_terminal_anchor("ACDEFG") is False)

    print("[3b] staple position offset")
    check("N-terminal pair offset is 0", mean_staple_offset("S5CDES5GHIK") == 0.0)
    check("internal pair offset is 1/8",
          abs(mean_staple_offset("AS5DEFS5HIK") - 1 / 8) < 1e-9)
    nan = mean_staple_offset("ACDEFG")
    check("no staple yields NaN", nan != nan)

    print("[3c] orphan staple residues")
    check("paired design has no orphan",
          orphan_staple_rate("AS5DEFS5HIK") == 0.0)
    check("no staple has no orphan", orphan_staple_rate("ACDEFG") == 0.0)
    # S5@0, S5@4, R8@7 — three staple residues, only one pair can form, so
    # one residue is left without a crosslink.
    orphan_seq = "S5AAAS5AAR8"
    check("odd staple count leaves an orphan",
          abs(orphan_staple_rate(orphan_seq) - 1 / 3) < 1e-9)
    check("one pair detected", count_staples(orphan_seq) == 1)
    # Maximum matching resolves this into two real crosslinks, not one.
    check("ambiguous placement is resolvable",
          orphan_staple_rate("S5AAAS5AAR8S5A") == 0.0)

    print("[4] summarize_designs")
    parents = ["ACDEFGHIK", "ACDEFGHIK"]
    designs = ["AS5DEFS5HIK", "ACDEFGHIK"]
    m = summarize_designs(designs, parents, cond=2)
    check("count", m["n"] == 2)
    check("compliance averaged", m["cond_compliance"] == 0.5)
    check("preservation is 1.0", m["preservation"] == 1.0)
    check("fallback detected", m["fallback_rate"] == 0.5)
    check("mean pairs", m["mean_pairs"] == 0.5)
    check("terminal anchor absent", m["terminal_anchor_rate"] == 0.0)
    check("offset ignores staple-free designs",
          abs(m["mean_staple_offset"] - 1 / 8) < 1e-9)
    check("rule reward in [0,1]", 0.0 <= m["rule_reward"] <= 1.0)
    check("orphan rate present", m["orphan_rate"] == 0.0)

    empty = summarize_designs([], [], cond=2)
    check("empty set is zeroed", empty["n"] == 0 and empty["cond_compliance"] == 0.0)

    print("[5] summarize_properties / diversity")
    props = summarize_properties(["AAAAK", "AAAAK"])
    check("charge for one K", props["charge"] == 1.0)
    check("mw positive", props["mw"] > 0)
    div = summarize_diversity(["ACDEF", "ACDEF", "CCCCC"])
    check("unique ratio", abs(div["unique_ratio"] - 2 / 3) < 1e-9)
    check("identity between 0 and 1", 0.0 <= div["mean_pairwise_identity"] <= 1.0)

    print("[6] random baseline respects the condition")
    rng = random.Random(0)
    parent = "ACDEFGHIKLMN"
    for _ in range(20):
        d2 = random_staple_placement(parent, 2, rng)
        assert d2 != parent, "cond 2 produced no staple"
    check("cond 2 always staples", True)

    from reward_rules import staple_accuracy
    ok2 = all(staple_accuracy(random_staple_placement(parent, 2, rng), 2) == 1.0
              for _ in range(20))
    ok3 = all(staple_accuracy(random_staple_placement(parent, 3, rng), 3) == 1.0
              for _ in range(20))
    check("cond 2 compliant", ok2)
    check("cond 3 compliant", ok3)
    ok4 = all(staple_accuracy(random_staple_placement(parent, 4, rng), 4) >= 0.5
              for _ in range(20))
    check("cond 4 at least one pair", ok4)

    print("[7] random baseline keeps length and scaffold")
    d = random_staple_placement(parent, 2, rng)
    check("same residue count", len(d.replace("S5", "x")) == len(parent))
    check("only staple positions change",
          sequence_identity(d, parent) >= (len(parent) - 2) / len(parent) - 1e-9)

    print("[8] short sequences are returned unchanged")
    check("cond 2 too short", random_staple_placement("ACDE", 2, rng) == "ACDE")
    check("cond 3 too short", random_staple_placement("ACDEFG", 3, rng) == "ACDEFG")
    check("cond 4 too short", random_staple_placement("ACDEFGH", 4, rng) == "ACDEFGH")
    check("cond 1 passthrough", random_staple_placement("ACDEF", 1, rng) == "ACDEF")

    print("[9] helpers")
    check("flatten parents", flatten_parents(["A", "C"], 2) == ["A", "A", "C", "C"])
    ds = random_designs(["ACDEFGHIK", "ACDEFGHIK"], 2, 3, rng)
    check("k designs per parent", len(ds) == 6)

    print("\nAll eval-core tests passed.")


if __name__ == "__main__":
    main()
