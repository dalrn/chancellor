"""Phase 3: the PIMC evaluator.

Perfect Information Monte Carlo.  For each action available to me: sample a
world from the belief posterior, assume it is the truth, play the round out
with a cheap policy, and record whether I won.  Average over many samples.

What PIMC cannot do
-------------------
Each sampled world is treated as fully observable during the rollout, so the
rollout policy plays as if it already knew every hidden card.  Two consequences,
both real and both surfaced in the output rather than hidden:

**It cannot value information.**  A Priest reveals a hand; a policy that already
"knows" every hand gains nothing from looking.  Every Priest target therefore
scores *identically* -- to zero variance, not approximately -- which would be
reported as a tie between actions that are obviously not equivalent.  So Priest
targets are ranked by expected information gain from the posterior instead
(:meth:`~loveletter.belief.Belief.information_gain`), which needs no rollout,
and the output says that is what happened.

**It never bluffs.**  Playing the Countess without being compelled only pays off
against an opponent who draws an inference from it.  The rollout policy draws no
inferences, so a bluff is never worth anything and PIMC will never suggest one.

Reading the output
------------------
Most Love Letter positions have genuinely near-equivalent actions -- measured
median gaps between the best and second-best action are a few percentage points,
below what the per-turn rollout budget can resolve.  So :class:`Recommendation`
reports ties as ties.  That is the honest answer, not a hedge: when two actions
really are equivalent, saying so is more useful at a table than inventing a
ranking the evidence does not support.
"""

from __future__ import annotations

import math
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from .agents import BASELINE, Agent
from .belief import Belief, World
from .config import STANDARD, Card, GameConfig
from .engine import (
    Action,
    GameState,
    PlayerId,
    apply_unchecked,
    legal_actions,
    state_from_hands,
    unchecked_transitions,
)
from .observation import observe

#: Two-sided 95% normal quantile.
Z95 = 1.959964


class FastPlayoutPolicy:
    """A cheap, *stochastic* playout policy. Kept for experiments; NOT the
    default.

    Measured head-to-head, PIMC using this as the rollout policy scored a
    0.475 win rate against the frozen baseline, while the identical search
    using the baseline itself as the playout scored 0.525 and closed most of
    the token gap.  The mechanism: in a rollout world where the opponent plays
    near-randomly, nobody punishes mistakes, so protection (Handmaid) looks
    unnecessary and greedy plays (Spy) look safe -- PIMC then over-played Spy
    by 3.7pp and under-played Handmaid by 2.9pp at the real table.  The
    playout is the search's opponent model, and its competence is part of the
    value estimate, not an implementation detail.

    The randomness is also not decoration.  An earlier version returned the
    first acceptable action and ignored ``rng`` entirely, which made every
    rollout from a determinized world identical.  The search then explored one
    fixed line per world instead of the space of continuations, and lost to
    the baseline by 0.11 tokens/round.  A deterministic playout does not
    sample anything -- it evaluates one arbitrary future very precisely.
    """

    name = "fast_playout"

    def choose(
        self, state: GameState, me: PlayerId, rng: random.Random
    ) -> Action:
        actions = legal_actions(state)
        if len(actions) == 1:
            return actions[0]
        safe = [
            a
            for a in actions
            if a.card is not Card.PRINCESS
            and not (a.card is Card.PRINCE and a.target == me)
        ]
        return rng.choice(safe or actions)


@dataclass(slots=True)
class ActionValue:
    """What the search learned about one action."""

    action: Action
    rollouts: int = 0
    wins: float = 0.0
    #: Set when the action was ranked by information rather than by rollouts.
    information_bits: float | None = None

    #: Sum of squared outcomes, for the variance of a non-binary score.
    sq: float = 0.0

    @property
    def win_rate(self) -> float:
        """Expected tokens gained from this action, per round.

        Named ``win_rate`` for continuity, but it is tokens: a round can award
        a token to each of several tied players, and the Spy bonus can award
        one to a player who did not win. Values above 1.0 are possible when a
        round win and the Spy bonus land together.
        """
        return self.wins / self.rollouts if self.rollouts else 0.0

    @property
    def ci(self) -> float:
        """95% half-width on the expected token estimate.

        Computed from the observed spread rather than a Bernoulli formula --
        the score is 0, 1 or 2 tokens, not a coin flip, so assuming binomial
        variance would understate the interval whenever the Spy bonus is live.
        """
        if self.rollouts < 2:
            return float("inf")
        mean = self.win_rate
        var = max(self.sq / self.rollouts - mean * mean, 0.0)
        return Z95 * math.sqrt(var / self.rollouts)

    def overlaps(self, other: "ActionValue") -> bool:
        """Do the two intervals overlap? Then the order is not established."""
        return abs(self.win_rate - other.win_rate) <= (self.ci + other.ci)


@dataclass(slots=True)
class Recommendation:
    """The evaluator's answer, including what it could not determine."""

    values: list[ActionValue]
    seconds: float = 0.0
    worlds_sampled: int = 0
    belief_worlds: int = 0
    #: True when the top actions were separated only by information gain,
    #: because the rollouts scored them identically.
    information_ranked: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def best(self) -> ActionValue:
        return self.values[0]

    @property
    def tied(self) -> list[ActionValue]:
        """Actions the search could not separate from the best one."""
        if not self.values:
            return []
        top = self.values[0]
        return [v for v in self.values[1:] if top.overlaps(v)]

    @property
    def conclusive(self) -> bool:
        """Is the top action distinguishable from the runner-up?"""
        return not self.tied

    def explain(self, limit: int = 6) -> str:
        """A plain reason for the top pick, and an honest note when there
        isn't one. The explanation is the product; do not shorten this."""
        if not self.values:
            return "no legal actions"
        lines: list[str] = []
        top = self.values[0]

        if self.information_ranked:
            lines.append(
                f"RECOMMEND  {top.action}   "
                f"({top.information_bits:.2f} bits of information)"
            )
            lines.append(
                "  Ranked by information gain, not by simulated win rate: "
                "the rollouts scored every target identically because a "
                "perfect-information playout gains nothing from looking at a "
                "hand it already knows. This targets the player whose hand is "
                "least certain."
            )
        else:
            lines.append(
                f"RECOMMEND  {top.action}   "
                f"{top.win_rate:.3f} +/- {top.ci:.3f} tokens"
            )
            if self.conclusive:
                runner = self.values[1] if len(self.values) > 1 else None
                if runner is not None:
                    lines.append(
                        f"  Clear of the next option ({runner.action}, "
                        f"{runner.win_rate:.3f}) by more than the margin of "
                        f"error."
                    )
                else:
                    lines.append("  The only action available.")
            else:
                names = ", ".join(str(v.action) for v in self.tied[:3])
                lines.append(
                    f"  NOT SEPARATED from: {names}. Their intervals overlap, "
                    f"so this ranking is not established -- treat them as "
                    f"equivalent and choose on grounds the search cannot see."
                )

        lines.append("")
        lines.append(f"  {'action':<34} {'E[tokens]':>9}  {'95% CI':>8}  n")
        for v in self.values[:limit]:
            if v.information_bits is not None and self.information_ranked:
                lines.append(
                    f"  {str(v.action):<34} {v.information_bits:>8.2f}b  "
                    f"{'--':>8}  --"
                )
            else:
                lines.append(
                    f"  {str(v.action):<34} {v.win_rate:>9.3f}  "
                    f"{v.ci:>8.3f}  {v.rollouts}"
                )
        if len(self.values) > limit:
            lines.append(f"  ... and {len(self.values) - limit} more")

        lines.append("")
        lines.append(
            f"  {self.worlds_sampled} rollouts over {self.belief_worlds} "
            f"candidate worlds in {self.seconds:.2f}s"
        )
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


def _distinct(actions: Sequence[Action], belief: Belief) -> list[Action]:
    """Reduce to the actions worth spending rollouts on.

    Non-Guard actions pass through.  Guard *targets* are all kept.  For each
    target the guesses are pruned to those the posterior says are possible,
    which is usually a large saving and never discards a live option.

    Guesses are deliberately **not** collapsed to one per target.  An earlier
    version did that, on the reasoning that the tracker ranks guesses exactly
    and rollouts would only divide the budget nine ways to re-answer a settled
    question.  That was wrong twice over: measured guess values against one
    opponent spanned 32.8% to 44.0%, an 11-point spread thrown away by keeping
    whichever guess sorted first; and where the posterior is uniform -- common
    early, when little is public -- it has no opinion to contribute, so
    "ranked exactly" meant "picked arbitrarily".
    """
    out: list[Action] = []
    seen_chancellor = False
    for a in actions:
        if a.card is Card.CHANCELLOR:
            # Collapse every return-order to one "play the Chancellor" option.
            # The return decision cannot be made at recommendation time: the
            # two cards are drawn *after* the Chancellor is played, so ranking
            # specific return-orders would describe a choice the user has not
            # been offered yet, using cards they have not seen. Each rollout
            # still resolves a concrete return per sampled world -- what is
            # collapsed is the reported option, not the simulation.
            if seen_chancellor:
                continue
            seen_chancellor = True
            out.append(Action(Card.CHANCELLOR))
            continue
        if a.card is not Card.GUARD or a.target is None or a.guess is None:
            out.append(a)
            continue
        marginals = belief.hand_marginals().get(a.target, {})
        if marginals.get(a.guess, 0.0) > 0.0:
            out.append(a)
    # If pruning removed every Guard (possible when the posterior has already
    # collapsed), keep them all rather than silently dropping a legal card.
    if not any(a.card is Card.GUARD for a in out) and any(
        a.card is Card.GUARD for a in actions
    ):
        out.extend(a for a in actions if a.card is Card.GUARD)
    return out


def _rebind(action: Action, determinized: GameState) -> Action | None:
    """Re-resolve ``action`` against a sampled world.

    A Chancellor's return-order names specific cards drawn from the deck, and
    every sampled world puts different cards on top -- so an action built
    against the real deck is meaningless in a determinization. Guard guesses
    are likewise re-picked, since ``_distinct`` collapsed them away.

    Returns the matching legal action in this world, or None if there is none.
    """
    options = legal_actions(determinized)
    exact = [
        a
        for a in options
        if a.card is action.card and a.target == action.target
    ]
    if not exact:
        return None
    if action.card is Card.CHANCELLOR:
        if action.chancellor_return is not None:
            # Resolution mode: the drawn cards are known and pinned in every
            # sampled world, so this specific return-order exists in each and
            # must be matched exactly -- taking any variant would collapse the
            # very distinction being evaluated.
            for a in exact:
                if a.chancellor_return == action.chancellor_return:
                    return a
            return None
        # Pre-play mode: the collapsed option carries no return-order; each
        # world offers its own, and averaging over them is exactly the value
        # of "play it".
        return exact[0]
    if action.guess is not None:
        for a in exact:
            if a.guess == action.guess:
                return a
        return None  # this guess is not offered in this world
    return exact[0]


def _world_to_state(state: GameState, world: World, me: PlayerId) -> GameState:
    """Build a fully-determined state from a sampled world.

    The viewer's own hand and everything public stay as they are; hidden hands
    and the deck are filled in from the world.
    """
    hands: list[list[Card]] = []
    for pid, player in enumerate(state.players):
        if player.out:
            hands.append([])
        elif pid == me:
            hands.append(list(player.hand))
        else:
            hands.append(list(world.hands[pid]))

    pinned = dict(world.slots)
    pool = list(world.deck_pool)
    deck: list[Card] = []
    for slot in state.slots:
        if slot in pinned:
            deck.append(pinned[slot])
        elif pool:
            deck.append(pool.pop())
    set_aside = world.set_aside if world.set_aside is not None else (
        pool.pop() if pool else None
    )

    return state_from_hands(
        hands,
        deck,
        set_aside,
        config=state.config,
        faceup=state.faceup,
        current=state.current,
        discards=[p.discards for p in state.players],
        out=[p.out for p in state.players],
        protected=[p.protected for p in state.players],
        tokens=[p.tokens for p in state.players],
    )


def evaluate(
    state: GameState,
    me: PlayerId,
    *,
    belief: Belief | None = None,
    rng: random.Random | None = None,
    budget_seconds: float = 1.4,
    max_rollouts_per_action: int = 2000,
    policy: object | None = None,
    playout: Agent | None = None,
    config: GameConfig = STANDARD,
) -> Recommendation:
    """Recommend an action for ``me``, with a win probability for each option.

    ``belief`` is built once per turn and reused across every rollout.  That is
    a requirement, not an optimisation: constructing it costs milliseconds at
    small tables and seconds at six players, while sampling a world from a
    built one costs about a microsecond.  Rebuilding per rollout would leave no
    time to roll out at all.
    """
    rng = rng or random.Random(0)
    # The baseline is the default playout: the playout is the search's
    # opponent model, and a too-weak one systematically misprices defensive
    # plays. See FastPlayoutPolicy's docstring for the measurement.
    playout = playout or BASELINE
    started = time.time()

    actions = legal_actions(state)
    if not actions:
        return Recommendation(values=[], seconds=0.0)
    if len(actions) == 1:
        return Recommendation(
            values=[ActionValue(action=actions[0], rollouts=0)],
            seconds=time.time() - started,
            notes=["only one legal action"],
        )

    if belief is None:
        belief = Belief.from_log(
            observe(state.log, me),
            me,
            state.n_players,
            config=config,
            faceup=state.faceup,
            policy=policy,
        )
    worlds = belief.worlds()

    candidates = _distinct(actions, belief)

    # Priest targets are ranked by information, not by rollouts -- see the
    # module docstring. Handled before the search so the budget is not spent
    # on a question rollouts provably cannot answer.
    priests = [a for a in candidates if a.card is Card.PRIEST and a.target is not None]
    if priests and len(candidates) == len(priests):
        values = [
            ActionValue(
                action=a,
                information_bits=belief.information_gain(a.target),  # type: ignore[arg-type]
            )
            for a in priests
        ]
        values.sort(key=lambda v: -(v.information_bits or 0.0))
        return Recommendation(
            values=values,
            seconds=time.time() - started,
            belief_worlds=len(worlds),
            information_ranked=True,
            notes=[
                "every option was a Priest look; PIMC scores these identically "
                "(zero variance), so they are ranked by expected bits learned"
            ],
        )

    values = [ActionValue(action=a) for a in candidates]
    per_action = max(
        1,
        min(
            max_rollouts_per_action,
            _estimate_budget(state, worlds, me, playout, rng, budget_seconds)
            // len(candidates),
        ),
    )

    total_rollouts = per_action * len(candidates)
    sampled = _search(
        state, me, values, worlds, playout, rng, started, budget_seconds,
        total_rollouts,
    )

    values.sort(key=lambda v: -v.win_rate)
    rec = Recommendation(
        values=values,
        seconds=time.time() - started,
        worlds_sampled=sampled,
        belief_worlds=len(worlds),
    )
    _add_priest_note(rec, candidates, belief)
    if not belief.exact:
        rec.notes.append(
            f"belief was sampled, not exact ({belief.total_worlds} worlds "
            f"existed); probabilities carry extra error"
        )
    return rec


def _search(
    state: GameState,
    me: PlayerId,
    values: list[ActionValue],
    worlds: Sequence[World],
    playout: Agent,
    rng: random.Random,
    started: float,
    budget_seconds: float,
    total_rollouts: int,
) -> int:
    """Spend the rollout budget where it can still change the answer.

    Splitting the budget evenly is the obvious approach and it is wrong when
    the option count is large: a Guard offers nine guesses per target, so an
    even split gives each about a ninth of the samples and the intervals come
    out too wide to separate any of them.  Measured on one position, the even
    split picked the wrong guess at a 1.2s budget and the right one only at
    6s -- the search was correct but starved.

    So: seed every option with a small equal sample, then give the remainder
    to the options that are still plausibly best (their interval overlaps the
    leader's).  Clearly-losing options stop consuming budget, which is where
    the samples for separating the real contenders come from.
    """
    if not values:
        return 0
    sampled = 0
    seed_each = max(8, total_rollouts // (len(values) * 4))

    def roll(value: ActionValue) -> bool:
        """One rollout for ``value``. False if the budget ran out."""
        nonlocal sampled
        if time.time() - started > budget_seconds:
            return False
        world = worlds[rng.randrange(len(worlds))]
        determinized = _world_to_state(state, world, me)
        action = _rebind(value.action, determinized)
        if action is None:
            return True  # not offered in this world; skip, do not stop
        end = _play_out(apply_unchecked(determinized, action), playout, rng)
        # Expected *tokens*, not round wins. Winning is not binary-exclusive:
        # tied players each take a token, and the Spy bonus is a token that
        # can go to someone who lost the round. Scoring `me in end.winners`
        # throws that away -- measured at 0.164 tokens/round, larger than the
        # margin the search is trying to find.
        gained = float(end.player(me).tokens)
        value.wins += gained
        value.sq += gained * gained
        value.rollouts += 1
        sampled += 1
        return True

    with unchecked_transitions():
        for value in values:
            for _ in range(seed_each):
                if not roll(value):
                    return sampled

        while sampled < total_rollouts:
            leader = max(values, key=lambda v: v.win_rate)
            live = [v for v in values if v is leader or leader.overlaps(v)]
            if len(live) < 2:
                live = [leader]
            for value in live:
                if not roll(value):
                    return sampled
    return sampled


def evaluate_chancellor(
    state: GameState,
    me: PlayerId,
    drawn: Sequence[Card],
    *,
    belief: Belief,
    rng: random.Random | None = None,
    budget_seconds: float = 1.4,
) -> Recommendation:
    """Rank the keep/return options of an already-played Chancellor.

    This is the second half of the Chancellor's two decisions.  ``evaluate``
    handles the first (play it or not) and deliberately refuses to rank
    return-orders, because the cards are unknown at that point.  Here they
    are known: ``drawn`` was physically picked up, so it is hard information
    -- the top deck slots are pinned to those cards in every sampled world,
    and the unknown pool shrinks by them, which also sharpens what opponents
    can hold.

    ``state`` is the mid-resolution position: the Chancellor already in the
    play area, my hand holding the one card I kept back plus the Chancellor
    re-inserted so ``legal_actions`` can enumerate the concrete return-orders
    against a deck whose top cards are the real draws.
    """
    rng = rng or random.Random(0)
    started = time.time()

    # Pin the drawn cards to the slots they came from, and take them out of
    # the unknown pool. Mutating a copy of the belief's constraints keeps the
    # caller's posterior untouched.
    import copy as _copy

    pinned = Belief(
        _copy.deepcopy(belief.constraints),
        Counter(belief.unknown_pool),
        config=belief.config,
        rng=rng,
        max_worlds=belief.max_worlds,
        policy=belief.policy,
    )
    for slot, card in zip(state.slots, drawn):
        pinned.constraints.known_slot[slot] = card
        pinned.unknown_pool[card] -= 1
        if pinned.unknown_pool[card] <= 0:
            del pinned.unknown_pool[card]
    worlds = pinned.worlds()

    candidates = [
        a for a in legal_actions(state) if a.card is Card.CHANCELLOR
    ]
    values = [ActionValue(action=a) for a in candidates]
    if len(values) == 1:
        values[0].rollouts = 0
        return Recommendation(
            values=values, seconds=time.time() - started,
            belief_worlds=len(worlds), notes=["only one way to resolve"],
        )
    per_action = max(
        1,
        min(
            2000,
            _estimate_budget(state, worlds, me, BASELINE, rng, budget_seconds)
            // max(len(candidates), 1),
        ),
    )
    sampled = _search(
        state, me, values, worlds, BASELINE, rng, started, budget_seconds,
        per_action * len(candidates),
    )
    values.sort(key=lambda v: -v.win_rate)
    return Recommendation(
        values=values,
        seconds=time.time() - started,
        worlds_sampled=sampled,
        belief_worlds=len(worlds),
        notes=[
            "drawn cards are pinned as known information; opponents' "
            "possible hands shrink accordingly"
        ],
    )


def _add_priest_note(
    rec: Recommendation, candidates: Sequence[Action], belief: Belief
) -> None:
    """Flag Priest options as unrankable by rollout, and give their bits."""
    priests = [a for a in candidates if a.card is Card.PRIEST and a.target is not None]
    if len(priests) < 2:
        return
    for value in rec.values:
        if value.action in priests and value.action.target is not None:
            value.information_bits = belief.information_gain(value.action.target)
    best = max(priests, key=lambda a: belief.information_gain(a.target))  # type: ignore[arg-type]
    rec.notes.append(
        f"Priest targets score identically in rollouts (PIMC cannot value "
        f"information); by expected bits learned the best look is "
        f"{best} at {belief.information_gain(best.target):.2f} bits"  # type: ignore[arg-type]
    )


def _estimate_budget(
    state: GameState,
    worlds: Sequence[World],
    me: PlayerId,
    playout: Agent,
    rng: random.Random,
    budget_seconds: float,
) -> int:
    """Time a few rollouts to decide how many the budget affords."""
    if not worlds:
        return 1
    probe = min(12, len(worlds))
    started = time.time()
    with unchecked_transitions():
        for i in range(probe):
            world = worlds[rng.randrange(len(worlds))]
            determinized = _world_to_state(state, world, me)
            actions = legal_actions(determinized)
            _play_out(apply_unchecked(determinized, actions[0]), playout, rng)
    elapsed = time.time() - started
    if elapsed <= 0:
        return 5000
    per = elapsed / probe
    remaining = max(budget_seconds - elapsed, 0.05)
    return max(len(worlds) and 8, int(remaining / per))


def _play_out(
    state: GameState, playout: Agent, rng: random.Random, max_turns: int = 60
) -> GameState:
    """Play a determinized state to the end of the round."""
    st = state
    turns = 0
    while not st.round_over:
        st = apply_unchecked(
            st, playout.choose(st, st.current, rng)
        )
        turns += 1
        if turns > max_turns:
            raise RuntimeError("rollout did not terminate")
    return st
