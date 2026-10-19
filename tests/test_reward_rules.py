"""Unit tests for the torch-free reward rules.

Run:
    python tests/test_reward_rules.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reward_rules import (
    compute_reward,
    detect_staple_pairs,
    orphan_staple_rate,
    preservation_score,
    staple_accuracy,
    staple_matching_size,
    staple_site_validity,
)


def check(name, condition):
    if not condition:
        raise AssertionError(name)
    print(f"  ok  {name}")


def main():
    print("[1] staple detection")
    check("no staple", detect_staple_pairs("ACDEFG") == [])
    check("i,i+4", detect_staple_pairs("S5ACDS5") == [(0, 4, 4)])
    check("i,i+7", detect_staple_pairs("AS5ACDEFGR8") == [(1, 8, 7)])
    check(
        "double staple",
        detect_staple_pairs("S5ACDS5ACDS5ACDS5") == [(0, 4, 4), (8, 12, 4)],
    )

    print("[1b] maximum matching beats the old greedy scan")
    # S5@0, S5@4, R8@7, S5@8: greedy paired (0,4) then reported one pair and
    # left the R8 dangling.  The true answer is (0,7) + (4,8) = two crosslinks.
    ambiguous = "S5AAAS5AAR8S5A"
    check("two crosslinks recovered",
          detect_staple_pairs(ambiguous) == [(0, 7, 7), (4, 8, 4)])
    check("no orphan in ambiguous case", orphan_staple_rate(ambiguous) == 0.0)
    check("cond4 accuracy now 1.0", staple_accuracy(ambiguous, 4) == 1.0)

    print("[2] condition accuracy")
    check("cond1 pass", staple_accuracy("ACDEFG", 1) == 1.0)
    check("cond1 fail", staple_accuracy("S5ACDS5", 1) == 0.0)
    check("cond2 pass", staple_accuracy("S5ACDS5", 2) == 1.0)
    check("cond3 pass", staple_accuracy("AS5ACDEFGR8", 3) == 1.0)
    check("cond4 pass", staple_accuracy("S5ACDS5ACDS5ACDS5", 4) == 1.0)
    check("cond4 partial", staple_accuracy("S5ACDS5", 4) == 0.5)

    print("[2b] orphan staple residues")
    check("no staple has no orphan", orphan_staple_rate("ACDEFG") == 0.0)
    check("clean pair has no orphan", orphan_staple_rate("AS5DEFS5HIK") == 0.0)
    # S5@0, S5@4, R8@7 — three staple residues, only one pair can form.
    check("odd staple count leaves an orphan",
          abs(orphan_staple_rate("S5AAAS5AAR8") - 1 / 3) < 1e-9)

    print("[2c] placement feasibility check")
    check("two clean pairs accepted",
          staple_matching_size({0: 'S5', 4: 'S5', 8: 'S5', 12: 'S5'}) == 2)
    check("mixed gap-4/gap-7 accepted",
          staple_matching_size({0: 'S5', 4: 'S5', 7: 'R8', 8: 'S5'}) == 2)
    check("lone staple rejected",
          staple_matching_size({0: 'S5', 5: 'S5'}) == 0)

    print("[3] preservation")
    check("identical", preservation_score("ACDEFG", "ACDEFG") == 1.0)
    check(
        "single point mutation",
        abs(preservation_score("ACDEYG", "ACDEFG") - 5 / 6) < 1e-9,
    )
    check(
        "staple sites are not mutations",
        preservation_score("S5ACDS5", "AACDAA") == 1.0,
    )
    check(
        "staple sites skipped in parent too",
        preservation_score("S5ACDS5", "S5ACDS5") == 1.0,
    )
    check(
        "scaffold mutation still counted",
        abs(preservation_score("S5ACDS5", "AGCDAA") - 2 / 3) < 1e-9,
    )

    print("[4] site validity")
    check("internal pair", staple_site_validity("AS5ACDS5A") == 1.0)
    check("no staple is valid", staple_site_validity("ACDEFG") == 1.0)
    check("N-terminal anchor rejected", staple_site_validity("S5ACDS5A") == 0.0)
    check("C-terminal anchor rejected", staple_site_validity("AS5ACDS5") == 0.0)

    print("[5] combined reward")
    r = compute_reward("S5ACDS5", "AACDAA", 2)
    check("reward in [0,1]", 0.0 <= r <= 1.0)

    # A clean internal staple satisfies every term.
    clean = compute_reward("AS5DEFS5HA", "ACDEFGHA", 2)
    check("clean internal design scores 1.0", abs(clean - 1.0) < 1e-9)

    # A terminal-anchored staple must lose the validity term.  Under the old
    # 0.15 weight it still scored 0.85; it should now land at 0.65.
    anchored = compute_reward("S5ACDS5A", "AACDSAA", 2)
    check("terminal anchor scores 0.65", abs(anchored - 0.65) < 1e-9)
    check("validity term actually bites", anchored < 0.70)
    check("clean beats anchored", clean > anchored)

    print("\nAll reward-rule tests passed.")


if __name__ == "__main__":
    main()
