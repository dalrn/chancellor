"""Phase 2.5: the evaluation harness.

Without this there is no way to tell whether a change helps.  Every claim
about strength has to come through here.

Pairing
-------
Love Letter has enormous variance: the Guard is 6 cards in 21 and a correct
guess is frequently just luck.  Comparing two agents on independent samples
needs roughly 10,000 games per arm to see a 2-percentage-point edge.

So games are *paired*.  Each pair uses one seed, so both arms get the identical
shuffle, and the agents swap seats between the two games of the pair.  What is
then measured is the within-pair difference, which cancels both the deal and
seat advantage.  The variance of that difference is far smaller than the
variance of either arm, which is what buys back the sample size.

Usable player counts
--------------------
Belief-driven agents get sharply more expensive per decision as hidden hands
multiply -- the enumerated world count grows roughly 8x per additional
opponent (2p: 9 worlds, 6p: ~47,000).  Measured time for a statistically
useful 4,000-pair comparison, with :class:`~loveletter.agents.BeliefAgent`:

===  ==========  ============================
n    pairs/sec   4,000 pairs
===  ==========  ============================
2    ~560        7 s
3    ~200        20 s
4    ~18         4 min
5    ~2.4        28 min       (borderline)
6    ~0.2        5 h          (not practical)
===  ==========  ============================

Three tiers, and they mean different things:

**2-4 measurable** (:data:`EVALUABLE_PLAYERS`) -- sweep freely.

**5 playable, expensive to measure** (:data:`SLOW_PLAYERS`) -- fast enough to
use at the table (belief worst case 0.6s), but 28 minutes per 4,000-pair
comparison.  Budget for a 5-player run deliberately; a casual three-constant
sweep at four values each would be six hours.

**6 unvalidated** (:data:`UNVALIDATED_PLAYERS`) -- the engine and tracker are
correct, but it is over the 2s recommendation budget *and* impractical to
evaluate.  Make no strength claim for it.

Agents that do not consult the belief tracker (the baseline, random) run at
full speed at every count -- the limit is the tracker, not the arena.

Reading the result
------------------
:class:`ArenaResult` reports the paired difference with a confidence interval
and says plainly whether it is :attr:`~ArenaResult.conclusive`.  A comparison
run on a few hundred games will usually not be, and it says so rather than
offering a number that is not real.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

from .agents import Agent
from .config import STANDARD, GameConfig
from .engine import (
    GameState,
    apply_unchecked,
    legal_actions,
    new_round,
)

#: Two-sided 95% normal quantile.
Z95 = 1.959964

#: Tier 1 -- measurable. A 4,000-pair comparison finishes in seconds to
#: minutes, so these counts can be swept freely and re-measured on a whim.
EVALUABLE_PLAYERS = (2, 3, 4)

#: Tier 2 -- playable, expensive to measure. The tool is fast enough to *use*
#: at 5 players (belief worst case 0.6s, comfortably inside the 2s budget),
#: but a 4,000-pair comparison takes about 28 minutes. Budget for a 5-player
#: run deliberately; never fold one into a casual sweep, where three constants
#: at four values each would be six hours.
SLOW_PLAYERS = (5,)

#: Tier 3 -- unvalidated. 6 players runs, and the rules engine and tracker are
#: correct there, but it is both over the 2s recommendation budget (mean 1.13s,
#: 30% of recommendations above 2s, worst 4.9s) and impractical to evaluate
#: (~5 hours for 4,000 pairs). No strength claim should be made for it.
UNVALIDATED_PLAYERS = (6,)


def pair_seed(seed: int, index: int) -> int:
    """Deterministic seed for pair ``index`` of run ``seed``.

    Written out rather than using ``hash((seed, index))``: tuple hashing of
    ints happens to be stable across processes, but that is an implementation
    detail and reproducible runs are a hard requirement here. This is plain
    arithmetic and will still mean the same thing in five years.
    """
    return (seed * 1_000_003 + index * 31 + 17) & 0x7FFF_FFFF


@dataclass(frozen=True, slots=True)
class RoundOutcome:
    """What one round produced for one arm."""

    tokens: tuple[int, ...]
    winners: tuple[int, ...]
    turns: int


def play_round(
    agents: Sequence[Agent],
    rng: random.Random,
    *,
    config: GameConfig = STANDARD,
    first_player: int = 0,
    max_turns: int = 200,
) -> RoundOutcome:
    """Play one round to completion. ``agents[i]`` acts for seat ``i``.

    The deck and the agents draw from **separate** streams, both derived from
    ``rng``. Sharing one stream silently destroys the pairing: a PIMC agent
    consumes thousands of draws per decision while the baseline consumes one,
    so from the first decision onward the two games of a "paired" comparison
    are dealing different cards -- and the divergence is largest for exactly
    the comparisons the arena exists to run. The deck stream is drawn first
    and used only for the deal, so it is identical across arms regardless of
    what the agents do.

    Uses :func:`apply_unchecked`: every action comes straight from
    ``legal_actions``, so re-validating it here would be millions of
    repetitions of a question already answered.
    """
    deck_rng = random.Random(rng.getrandbits(64))
    agent_rng = random.Random(rng.getrandbits(64))
    state = new_round(
        len(agents), deck_rng, config=config, first_player=first_player
    )
    turns = 0
    while not state.round_over:
        actions = legal_actions(state)
        seat = state.current
        action = agents[seat].choose(state, seat, agent_rng)
        if action not in actions:
            raise ValueError(
                f"{agents[seat].name} returned an illegal action: {action}"
            )
        state = apply_unchecked(state, action)
        turns += 1
        if turns > max_turns:
            raise RuntimeError("round did not terminate")
    return RoundOutcome(
        tokens=tuple(p.tokens for p in state.players),
        winners=tuple(state.winners),
        turns=turns,
    )


@dataclass(slots=True)
class ArenaResult:
    """A paired comparison of two agents.

    ``diff_*`` fields are always (agent A) minus (agent B), averaged over
    pairs.  Positive means A is ahead.
    """

    name_a: str
    name_b: str
    pairs: int
    #: Per-pair differences in tokens gained, A minus B.
    token_diffs: list[float] = field(default_factory=list)
    #: Per-pair differences in round wins, A minus B.
    win_diffs: list[float] = field(default_factory=list)
    tokens_a: float = 0.0
    tokens_b: float = 0.0
    wins_a: float = 0.0
    wins_b: float = 0.0
    seconds: float = 0.0

    # ------------------------------------------------------------- summaries

    @staticmethod
    def _mean_ci(xs: Sequence[float]) -> tuple[float, float]:
        """Mean and 95% half-width, from the *paired* differences."""
        n = len(xs)
        if n < 2:
            return (xs[0] if xs else 0.0), float("inf")
        mean = sum(xs) / n
        var = sum((x - mean) ** 2 for x in xs) / (n - 1)
        return mean, Z95 * math.sqrt(var / n)

    @property
    def token_diff(self) -> float:
        return self._mean_ci(self.token_diffs)[0]

    @property
    def token_ci(self) -> float:
        return self._mean_ci(self.token_diffs)[1]

    @property
    def win_diff(self) -> float:
        return self._mean_ci(self.win_diffs)[0]

    @property
    def win_ci(self) -> float:
        return self._mean_ci(self.win_diffs)[1]

    @property
    def conclusive(self) -> bool:
        """Does the token-difference interval exclude zero?

        The honest headline. When this is False the run did not distinguish
        the agents, and the point estimate should not be quoted as if it had.
        """
        return abs(self.token_diff) > self.token_ci

    def required_pairs(self, effect: float = 0.02) -> int:
        """Roughly how many pairs would be needed to detect ``effect``.

        Uses the observed per-pair spread, so it answers "how much more of
        *this* comparison would I need", not a textbook figure.
        """
        n = len(self.token_diffs)
        if n < 2:
            return 0
        mean = sum(self.token_diffs) / n
        var = sum((x - mean) ** 2 for x in self.token_diffs) / (n - 1)
        if var <= 0:
            return 1
        return int(math.ceil((Z95 * Z95 * var) / (effect * effect)))

    def summary(self) -> str:
        lines = [
            f"{self.name_a}  vs  {self.name_b}",
            f"  pairs           {self.pairs}  ({self.pairs * 2} rounds)"
            f"  in {self.seconds:.1f}s",
            f"  tokens/round    {self.tokens_a:.4f}  vs  {self.tokens_b:.4f}",
            f"  win rate        {self.wins_a:.4f}  vs  {self.wins_b:.4f}",
            f"  paired diff     {self.token_diff:+.4f} tokens"
            f"  +/- {self.token_ci:.4f}  (95%)",
            f"  paired win diff {self.win_diff:+.4f}  +/- {self.win_ci:.4f}",
        ]
        if self.conclusive:
            ahead = self.name_a if self.token_diff > 0 else self.name_b
            lines.append(f"  VERDICT         {ahead} is ahead (interval excludes 0)")
        else:
            need = self.required_pairs()
            lines.append(
                "  VERDICT         INCONCLUSIVE at this sample size -- "
                "the interval includes 0"
            )
            lines.append(
                f"                  ~{need} pairs would be needed to detect "
                f"a 0.02 token/round edge"
            )
        return "\n".join(lines)


def compare(
    agent_a: Agent,
    agent_b: Agent,
    *,
    pairs: int = 2000,
    n_players: int = 2,
    seed: int = 0,
    config: GameConfig = STANDARD,
    progress: Callable[[int, int], None] | None = None,
) -> ArenaResult:
    """Play ``pairs`` paired matchups and report the difference.

    Each pair plays the same deal twice with seats swapped, so the deal and
    seat advantage both cancel in the within-pair difference.  With more than
    two players the remaining seats are filled by ``agent_b``, which makes the
    comparison "one A among Bs" -- a different question from heads-up, and
    worth being explicit about when quoting a number.

    One subtlety: both agents draw from the same per-round RNG stream.  Two
    agents that make identical choices therefore consume it identically and
    produce a difference of *exactly* zero, with a zero-width interval.  That
    is a correct answer to "are these the same agent", not a bug -- but it
    means a zero-width interval indicates identical play, not infinite
    precision.  Agents that differ at all consume the stream differently and
    the interval widens to something realistic.
    """
    if pairs < 1:
        raise ValueError("need at least one pair")
    if n_players not in EVALUABLE_PLAYERS and pairs > 200:
        import warnings

        tier = (
            "playable but expensive to measure (~28 min for 4,000 pairs) -- "
            "budget for this run deliberately"
            if n_players in SLOW_PLAYERS
            else "unvalidated (~5 h for 4,000 pairs, and over the 2s "
            "recommendation budget)"
        )
        warnings.warn(
            f"{n_players} players is {tier}. "
            f"Routinely measurable counts are {EVALUABLE_PLAYERS}.",
            stacklevel=2,
        )
    result = ArenaResult(name_a=agent_a.name, name_b=agent_b.name, pairs=pairs)

    import time

    started = time.time()
    tot_a = tot_b = 0.0
    win_a = win_b = 0.0

    for i in range(pairs):
        # Both games of a pair share a seed, so both arms see the same deal.
        seat_a_first = [agent_a] + [agent_b] * (n_players - 1)
        seat_b_first = [agent_b] + [agent_a] + [agent_b] * (n_players - 2)

        ps = pair_seed(seed, i)
        r1 = play_round(seat_a_first, random.Random(ps), config=config)
        r2 = play_round(seat_b_first, random.Random(ps), config=config)

        # Arm A: seat 0 in game 1, seat 1 in game 2. Arm B mirrors it.
        a_tokens = r1.tokens[0] + r2.tokens[1]
        b_tokens = r1.tokens[1] + r2.tokens[0]
        a_wins = (0 in r1.winners) + (1 in r2.winners)
        b_wins = (1 in r1.winners) + (0 in r2.winners)

        result.token_diffs.append((a_tokens - b_tokens) / 2.0)
        result.win_diffs.append((a_wins - b_wins) / 2.0)
        tot_a += a_tokens / 2.0
        tot_b += b_tokens / 2.0
        win_a += a_wins / 2.0
        win_b += b_wins / 2.0

        if progress and (i + 1) % 500 == 0:
            progress(i + 1, pairs)

    result.tokens_a = tot_a / pairs
    result.tokens_b = tot_b / pairs
    result.wins_a = win_a / pairs
    result.wins_b = win_b / pairs
    result.seconds = time.time() - started
    return result
