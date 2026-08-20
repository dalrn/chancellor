"""Sweep the heuristic policy constants through the arena.

The three constants in :class:`~loveletter.policy.HeuristicPolicy` are guesses
about how people play, not facts about the game.  This asks the arena whether
any of them measurably changes results, and reports when they don't.

Run: ``python sweep.py [pairs]``
"""

from __future__ import annotations

import sys
import time

from loveletter.agents import BASELINE, BeliefAgent
from loveletter.arena import compare
from loveletter.policy import HeuristicPolicy


def sweep_constant(
    name: str, values: list[float], pairs: int, seed: int = 500
) -> None:
    """Measure each value of one constant against the frozen baseline.

    Every arm is measured against the same opponent on the same seeds, so
    the arms are comparable with each other as well as with the baseline.
    """
    print(f"\n{'=' * 68}")
    print(f"  {name}   ({pairs} pairs per value, vs frozen baseline)")
    print(f"{'=' * 68}")
    print(
        f"  {'value':>8}  {'diff':>9}  {'95% CI':>9}  {'verdict':<14}  ess"
    )

    rows = []
    for value in values:
        policy = HeuristicPolicy(**{name: value})
        agent = BeliefAgent(name=f"belief[{name}={value}]", policy=policy)
        result = compare(agent, BASELINE, pairs=pairs, seed=seed)
        verdict = "ahead" if result.conclusive else "inconclusive"
        print(
            f"  {value:>8.3f}  {result.token_diff:>+9.4f}  "
            f"{result.token_ci:>9.4f}  {verdict:<14}  "
            f"{result.seconds:.0f}s"
        )
        rows.append((value, result))

    # Does the choice of value matter at all? Compare the extremes to each
    # other rather than each to the baseline.
    lo_v, lo = rows[0]
    hi_v, hi = rows[-1]
    spread = abs(hi.token_diff - lo.token_diff)
    combined_ci = lo.token_ci + hi.token_ci
    print()
    if spread > combined_ci:
        print(
            f"  -> {name} MATTERS: {lo_v} and {hi_v} differ by "
            f"{spread:.4f}, wider than the combined interval "
            f"{combined_ci:.4f}"
        )
    else:
        print(
            f"  -> {name} shows NO DETECTABLE EFFECT at {pairs} pairs: "
            f"extremes differ by {spread:.4f}, within the combined "
            f"interval {combined_ci:.4f}."
        )
        need = max(lo.required_pairs(max(spread, 0.005)), 1)
        print(
            f"     ~{need} pairs would be needed to resolve a difference "
            f"this small. Do not trust precision in these constants."
        )
        
def main() -> None:
    pairs = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    started = time.time()

    print(f"Sweeping HeuristicPolicy constants at {pairs} pairs each.")
    print("Baseline is frozen; only the policy feeding the tracker varies.")

    sweep_constant("countess_bluff", [0.01, 0.05, 0.20, 0.50], pairs)
    sweep_constant("keep_high_bias", [0.0, 0.30, 0.60, 1.20], pairs)
    sweep_constant("self_destruct_penalty", [0.001, 0.02, 0.10, 0.30], pairs)

    print(f"\nTotal: {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
