"""
Do near-ties resolve at 50,000 rollouts, or stay flat?
If they resolve, the heuristic playout policy is the bottleneck and a learned
one earns its place. If they stay flat, the positions really are equivalent.
"""
import random, sys, time
from loveletter.engine import (new_round, apply_unchecked, unchecked_transitions,
                               legal_actions, Card)
from loveletter.agents import BASELINE

def playout(state, rng, agents):
    st = state
    while not st.round_over:
        st = apply_unchecked(st, agents[st.current].choose(st, st.current, rng))
    return st

def distinct_decisions(state):
    """Collapse Guard guesses; drop pure-information ties (Priest targets)."""
    seen, out = set(), []
    for a in legal_actions(state):
        key = (a.card, a.target)
        if key in seen:
            continue
        seen.add(key)
        out.append(a)
    return out

def score(state, action, me, agents, n, seed0=0):
    wins = 0
    for i in range(n):
        end = playout(apply_unchecked(state, action), random.Random(seed0 + i), agents)
        wins += (me in end.winners)
    return wins / n

def find_positions(n_players, want, rng_seed=0):
    """Mid-round positions with >=2 genuinely different actions."""
    found = []
    for seed in range(400):
        if len(found) >= want:
            break
        rng = random.Random(rng_seed * 1000 + seed)
        st = new_round(n_players, rng)
        depth = rng.randrange(1, 6)
        for _ in range(depth):
            if st.round_over:
                break
            st = apply_unchecked(st, rng.choice(legal_actions(st)))
        if st.round_over:
            continue
        acts = distinct_decisions(st)
        # Require two actions playing DIFFERENT cards: Priest-vs-Priest is
        # identical under any information-blind playout, so it is not a tie
        # to investigate, it is a known blind spot.
        if len({a.card for a in acts}) < 2:
            continue
        found.append((st, acts))
    return found

def main():
    n_players = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    small, large = 300, 50_000
    agents = [BASELINE] * n_players
    positions = find_positions(n_players, 14, rng_seed=n_players)
    print(f"{n_players}p: {len(positions)} positions, screening at {small} rollouts")

    near_ties = []
    with unchecked_transitions():
        for st, acts in positions:
            me = st.current
            sc = sorted(((score(st, a, me, agents, small), a) for a in acts),
                        reverse=True, key=lambda x: x[0])
            gap = sc[0][0] - sc[1][0]
            if gap < 0.05:
                near_ties.append((st, sc[0][1], sc[1][1], gap))
    print(f"  near-ties found (<5pp at {small}): {len(near_ties)}")

    print(f"\n  re-running each at {large:,} rollouts")
    print(f"  {'small gap':>10}  {'large gap':>10}  {'95% CI':>8}  verdict")
    resolved = flat = 0
    with unchecked_transitions():
        for st, a1, a2, small_gap in near_ties:
            me = st.current
            t = time.time()
            s1 = score(st, a1, me, agents, large, seed0=1_000_000)
            s2 = score(st, a2, me, agents, large, seed0=2_000_000)
            big_gap = abs(s1 - s2)
            ci = 1.96 * ((0.25 / large) ** 0.5 * 2 ** 0.5)
            verdict = "RESOLVED" if big_gap > max(0.05, 2 * ci) else "still flat"
            if verdict == "RESOLVED":
                resolved += 1
            else:
                flat += 1
            print(f"  {small_gap*100:>9.1f}pp  {big_gap*100:>9.1f}pp  "
                  f"{ci*100:>7.2f}pp  {verdict}  ({time.time()-t:.0f}s)")

    print(f"\n  RESOLVED {resolved} / still flat {flat}")
    if flat > resolved:
        print("  => near-ties are genuine: the positions really are equivalent.")
        print("     A better playout policy would return the same verdict sooner.")
    else:
        print("  => near-ties RESOLVE at high N: the playout policy is the")
        print("     bottleneck, and a learned one would earn its place.")

if __name__ == "__main__":
    main()
