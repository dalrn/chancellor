"""Phase 2a: the belief tracker, hard constraints only.

What this maintains
-------------------
A posterior over the *joint* hidden state: what each opponent holds, what sits
in each deck slot, and what the facedown set-aside card is.  Per-player
marginals are not a sufficient representation -- opponents' hands are
correlated through the fixed deck composition, and asking "could P1 and P2 both
hold the Princess?" must come back no.  So a world here is a complete
assignment, and marginals are derived from the world set, never tracked
directly.

Representation
--------------
Exact enumeration over consistent worlds.  The unknown pool is small enough in
practice (it shrinks with every public discard) and being exact makes the
log-replay test an equality rather than a convergence check.  Above
``max_worlds`` the set is reservoir-sampled with the injected RNG, which keeps
it unbiased where truncation would not; :attr:`Belief.exact` says which
happened and :attr:`Belief.total_worlds` how many worlds really exist.

Only hands are enumerated.  Deck slots are exchangeable given the leftover
multiset, so enumerating their orderings would multiply the world count by 15!
without adding a distinguishable outcome -- see :meth:`Belief._enumerate`.

Hard constraints only
---------------------
Everything here is a logical certainty derivable from public information plus
the viewer's own private knowledge.  No opponent model, no assumption about how
anyone plays.  The Countess ambiguity is deliberately *not* resolved -- that
needs a policy model and belongs to Phase 2b.

The Priest field is read only for the viewer's own looks.  An opponent's look
tells this phase nothing; it is in the log for Phase 2b to pick up, with no
change to the log format.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import Iterable, Iterator, Sequence

from .config import STANDARD, Card, GameConfig
from .policy import OpponentPolicy, PlayContext
from .events import (
    BaronCompare,
    Dealt,
    ChancellorExchange,
    Drew,
    Eliminated,
    Event,
    EventLog,
    GuardResult,
    KingTrade,
    Played,
    PlayerId,
    PriestLook,
    PrinceDiscard,
    RoundEnded,
)

#: Where a card can be. Deck slots are keyed by their stable slot id.
SET_ASIDE = -1


@dataclass(frozen=True, slots=True)
class World:
    """One complete, consistent assignment of every hidden card.

    ``hands`` maps player -> the cards they hold (1, or 2 mid-turn).
    ``slots`` maps deck slot id -> card.  ``set_aside`` is the facedown card.
    """

    hands: tuple[tuple[Card, ...], ...]
    #: Slots whose occupant is pinned (Chancellor returns we can identify).
    slots: tuple[tuple[int, Card], ...]
    set_aside: Card | None
    #: The unordered remainder: cards filling every unpinned hidden position
    #: (unknown deck slots and the set-aside, when unknown). Every ordering of
    #: these is equally likely, so storing one is storing all of them.
    deck_pool: tuple[Card, ...] = ()
    #: Relative plausibility of this world, from Phase 2b's opponent model.
    #: 1.0 under hard constraints alone, where every consistent world is
    #: equally likely. Marginals are weighted by this, so leaving it at 1.0
    #: reproduces the Phase 2a posterior exactly.
    weight: float = 1.0

    def slot_map(self) -> dict[int, Card]:
        return dict(self.slots)


@dataclass(slots=True)
class Constraints:
    """Everything hard-known about the hidden state, accumulated from the log.

    This is the intermediate form: the log is replayed into constraints, then
    constraints are enumerated into worlds.  Keeping the two apart is what
    makes the replay test meaningful -- constraints are a pure function of the
    projected log.
    """

    n_players: int
    #: Cards known to be in a player's hand right now (the viewer's own, or a
    #: card whose location is forced).
    known_hand: dict[PlayerId, tuple[Card, ...]] = field(default_factory=dict)
    #: Cards a player is known *not* to hold at all, from a Guard miss on a
    #: hand they have not changed since.
    excluded: dict[PlayerId, set[Card]] = field(default_factory=lambda: defaultdict(set))
    #: Ceilings on how many copies a player can hold, from a Guard miss that
    #: has since been diluted by a draw.  A miss on a 1-card hand followed by a
    #: draw means "at most 1 copy": the drawn card could be that card, the card
    #: they already held provably is not.
    at_most: dict[PlayerId, dict[Card, int]] = field(default_factory=dict)
    #: Slot id -> card, for slots whose occupant is known.
    known_slot: dict[int, Card] = field(default_factory=dict)
    #: Players out of the round; they hold nothing.
    out: set[PlayerId] = field(default_factory=set)
    #: Live deck slot ids, top to bottom.
    deck_slots: tuple[int, ...] = ()
    #: Cards publicly accounted for: discards and the 2-player faceup cards.
    public: list[Card] = field(default_factory=list)
    set_aside_known: Card | None = None
    #: Strict inequalities between hands from Baron: (higher, lower) at the
    #: time of comparison, kept only while both hands are unchanged since.
    greater_than: list[tuple[PlayerId, PlayerId]] = field(default_factory=list)
    #: Equalities from a Baron tie, same freshness rule.
    equal_to: list[tuple[PlayerId, PlayerId]] = field(default_factory=list)
    #: Cards a player is known to still hold, without their whole hand being
    #: known. A Priest sighting degrades to this the moment they draw: they
    #: gained a card but did not lose the one that was seen.
    must_hold: dict[PlayerId, tuple[Card, ...]] = field(default_factory=dict)
    #: Baron facts diluted by a later draw. The compared card is still in the
    #: hand somewhere, so the relation must hold for *some* card they hold
    #: rather than for the hand's maximum.
    weak_greater: list[tuple[PlayerId, PlayerId]] = field(default_factory=list)
    weak_equal: list[tuple[PlayerId, PlayerId]] = field(default_factory=list)
    #: Baron wins against an eliminated opponent, whose card became public.
    #: ``(winner, value)``: the winner held something strictly above ``value``.
    #: Kept as a value because the loser's hand is gone -- comparing against an
    #: empty hand would make the constraint vacuously true.
    beats_value: list[tuple[PlayerId, Card]] = field(default_factory=list)
    #: How many cards each player holds. Normally 1; the actor holds 2 between
    #: their draw and their play, and enumeration must fill both.
    sizes: dict[PlayerId, int] = field(default_factory=dict)
    #: Set between a BaronCompare and the Eliminated event that reveals the
    #: loser's card, so the winner's bound can be pinned to a real value.
    pending_baron_win: tuple[PlayerId, PlayerId] | None = None
    #: Observed opponent plays, for Phase 2b reweighting: the player, the card
    #: they played, the public context at the time, and the card they were
    #: known to still hold afterwards (None when it was hidden). Recorded
    #: unconditionally; the tracker only reads them when a policy is set.
    observed_plays: list[tuple[PlayerId, Card, PlayContext, Card | None]] = field(
        default_factory=list
    )

    def hand_size(self, pid: PlayerId) -> int:
        return 0 if pid in self.out else self.sizes.get(pid, 1)

    def clear_player_inferences(self, pid: PlayerId) -> None:
        """``pid``'s hand was *replaced*: drop everything time-indexed.

        Used for a Prince discard-and-redraw and for a King trade, where the
        card every earlier observation spoke about is simply gone.  Nothing
        survives in weakened form, because nothing of the old hand remains.

        Contrast :meth:`note_draw`, where the hand only grew.
        """
        self.excluded.pop(pid, None)
        self.known_hand.pop(pid, None)
        self.must_hold.pop(pid, None)
        self.at_most.pop(pid, None)
        self.greater_than = [p for p in self.greater_than if pid not in p]
        self.equal_to = [p for p in self.equal_to if pid not in p]
        self.weak_greater = [p for p in self.weak_greater if pid not in p]
        self.weak_equal = [p for p in self.weak_equal if pid not in p]
        self.beats_value = [p for p in self.beats_value if p[0] != pid]
        self.observed_plays = [r for r in self.observed_plays if r[0] != pid]

    def note_draw(self, pid: PlayerId) -> None:
        """``pid`` drew: they gained a card and lost none, so facts weaken.

        Every observation about the old hand is still true *of a card they
        still hold*; it is merely no longer true of the hand as a whole.  So
        each one demotes rather than disappearing:

        * a known hand becomes "must still hold these";
        * a Guard miss becomes "at most one copy" -- the card they had is
          provably not the named one, the drawn card might be;
        * a Baron result becomes "some held card satisfies the relation".

        Discarding these outright is the tempting simplification and it throws
        away most of what the tracker knows in the mid-game.
        """
        known = self.known_hand.pop(pid, None)
        if known:
            self.must_hold[pid] = known

        banned = self.excluded.pop(pid, None)
        if banned:
            caps = dict(self.at_most.get(pid, {}))
            for card in banned:
                caps[card] = min(caps.get(card, 99), 1)
            self.at_most[pid] = caps

        self.weak_greater += [p for p in self.greater_than if pid in p]
        self.weak_equal += [p for p in self.equal_to if pid in p]
        self.greater_than = [p for p in self.greater_than if pid not in p]
        self.equal_to = [p for p in self.equal_to if pid not in p]


class Belief:
    """The posterior, reconstructed by replaying a projected event log."""

    def __init__(
        self,
        constraints: Constraints,
        unknown_pool: Counter[Card],
        *,
        config: GameConfig = STANDARD,
        rng: random.Random | None = None,
        max_worlds: int = 20_000,
        policy: "OpponentPolicy | None" = None,
    ) -> None:
        self.constraints = constraints
        self.unknown_pool = unknown_pool
        self.config = config
        self.rng = rng or random.Random(0)
        self.max_worlds = max_worlds
        #: Phase 2b opponent model. ``None`` disables soft reweighting, leaving
        #: the Phase 2a posterior untouched so the two can be compared.
        self.policy = policy
        self.exact = True
        #: How many consistent worlds exist, even when only a sample is kept.
        self.total_worlds = 0
        self._worlds: list[World] | None = None

    # ------------------------------------------------------------- building

    @classmethod
    def from_log(
        cls,
        log: EventLog,
        viewer: PlayerId,
        n_players: int,
        *,
        config: GameConfig = STANDARD,
        faceup: Sequence[Card] = (),
        viewer_hand: Sequence[Card] = (),
        initial_slots: Sequence[int] | None = None,
        rng: random.Random | None = None,
        policy: "OpponentPolicy | None" = None,
    ) -> "Belief":
        """Rebuild the posterior from scratch by replaying ``log``.

        ``log`` must already be projected through
        :func:`~loveletter.observation.observe` for ``viewer``.  Replaying the
        engine's raw log here would make the tracker omniscient.

        ``viewer_hand`` is the card ``viewer`` was *dealt* at setup, not the
        hand they hold now -- subsequent draws come from the log's own Drew
        events, and supplying the current hand would count them twice.
        """
        c = replay(
            log,
            viewer,
            n_players,
            config=config,
            faceup=faceup,
            viewer_hand=viewer_hand,
            initial_slots=initial_slots,
        )
        pool = unknown_pool(c, config)
        return cls(c, pool, config=config, rng=rng, policy=policy)

    # -------------------------------------------------------------- worlds

    def worlds(self) -> list[World]:
        """Every consistent world, or an unbiased sample if there are too many.

        Truncating enumeration would bias the posterior toward whatever the
        recursion happens to visit first -- for Love Letter's card ordering
        that means low-value hands. So when the count exceeds ``max_worlds``
        the worlds are reservoir-sampled instead, giving every consistent
        world an equal chance of being kept. :attr:`exact` records which
        happened, and marginals from a sampled set carry sampling error.
        """
        if self._worlds is None:
            self._worlds = self._sample_worlds()
        return self._worlds

    def _weight_of(self, world: World) -> float:
        """How plausible ``world`` is, given the plays we actually saw.

        For each observed opponent play we know the card that was played and,
        from this world, the card they kept.  Together those are the 2-card
        hand they were choosing from, so the policy can say how likely that
        choice was.  Multiplying across plays gives the world's weight.

        This is the whole of Phase 2b: hard constraints say which worlds are
        *possible*, the policy says which are *likely*.
        """
        if self.policy is None:
            return 1.0
        c = self.constraints
        weight = 1.0
        for actor, played, context, _ in c.observed_plays:
            kept = world.hands[actor]
            if not kept:
                # They are out now, so the card they kept is public and the
                # play carries no hidden information for this world.
                continue
            # The hand they chose from: what they played plus what they kept.
            # Only each player's most recent play is retained, so the kept
            # card is the one they held when they made it -- unless they have
            # drawn since, which is the residual approximation and the reason
            # 2b reweights rather than constrains.
            hand = (played,) + tuple(kept[:1])
            weight *= self.policy.play_probability(hand, played, context)
        return max(weight, 0.0)

    def _sample_worlds(self) -> list[World]:
        """Enumerate, reservoir-sampling down to ``max_worlds`` if needed."""
        reservoir: list[World] = []
        seen = 0
        for world in self._enumerate():
            seen += 1
            if len(reservoir) < self.max_worlds:
                reservoir.append(world)
            else:
                # Reservoir sampling: world i survives with probability k/i,
                # which keeps the sample uniform over everything enumerated.
                j = self.rng.randrange(seen)
                if j < self.max_worlds:
                    reservoir[j] = world
        self.exact = seen <= self.max_worlds
        self.total_worlds = seen
        return reservoir

    def _slot_targets(self) -> list[int]:
        """Deck slots whose occupant is unknown, plus the set-aside if unknown."""
        c = self.constraints
        targets = [s for s in c.deck_slots if s not in c.known_slot]
        if c.set_aside_known is None:
            targets.append(SET_ASIDE)
        return targets

    def _hand_targets(self) -> list[tuple[PlayerId, int]]:
        """(player, count) for each player whose hand is not fully known."""
        c = self.constraints
        out: list[tuple[PlayerId, int]] = []
        for pid in range(c.n_players):
            if pid in c.out:
                continue
            known = c.known_hand.get(pid)
            if known is not None:
                continue
            out.append((pid, c.hand_size(pid)))
        return out

    def _enumerate(self) -> Iterator[World]:
        """Enumerate consistent worlds.

        Only *hands* are enumerated card by card.  There are at most a handful
        of hidden hand positions and they carry every hard constraint worth
        pruning on -- Guard exclusions, Baron inequalities, Priest sightings --
        so branching there is both small and productive.

        The deck is deliberately *not* enumerated.  Given a hand assignment,
        the leftover multiset fills the remaining slots, and every arrangement
        of it is equally likely; enumerating those orderings would multiply the
        world count by 15! without adding a single distinguishable outcome.
        A world therefore carries the deck as a multiset plus the slots it
        fills, and :meth:`slot_marginals` derives per-slot probabilities
        analytically.  Chancellor-constrained slots stay pinned via
        ``known_slot``, which is what the ordered representation buys.
        """
        c = self.constraints
        hand_slots = self._hand_targets()
        n_hand_cards = sum(n for _, n in hand_slots)
        pool = Counter(self.unknown_pool)

        if n_hand_cards > sum(pool.values()):
            raise ValueError("not enough unknown cards to fill the hands")

        count = 0
        for assignment, leftover in self._assign_hands(hand_slots, pool, 0):
            hands: list[tuple[Card, ...]] = [() for _ in range(c.n_players)]
            for pid in range(c.n_players):
                if pid in c.out:
                    continue
                known = c.known_hand.get(pid)
                if known is not None:
                    hands[pid] = known
            for pid, cards in assignment:
                hands[pid] = cards
            if not _relations_hold(c, hands):
                continue
            count += 1
            world = World(
                hands=tuple(hands),
                slots=tuple(sorted(c.known_slot.items())),
                set_aside=c.set_aside_known,
                deck_pool=tuple(sorted(leftover.elements())),
            )
            yield replace(world, weight=self._weight_of(world))

    def _assign_hands(
        self,
        targets: list[tuple[PlayerId, int]],
        pool: Counter[Card],
        i: int,
    ) -> Iterator[tuple[tuple[tuple[PlayerId, tuple[Card, ...]], ...], Counter[Card]]]:
        """Recursively deal the pool into the unknown hands, pruning as we go."""
        if i == len(targets):
            yield (), pool
            return
        pid, size = targets[i]
        c = self.constraints
        banned = c.excluded.get(pid, set())
        required = c.must_hold.get(pid, ())
        for combo in _multiset_combinations(pool, size):
            if any(card in banned for card in combo):
                continue
            if required:
                have = Counter(combo)
                if any(have[card] < n for card, n in Counter(required).items()):
                    continue
            caps = c.at_most.get(pid)
            if caps:
                have = Counter(combo)
                if any(have[card] > cap for card, cap in caps.items()):
                    continue
            rest = pool.copy()
            for card in combo:
                rest[card] -= 1
                if rest[card] == 0:
                    del rest[card]
            for tail, leftover in self._assign_hands(targets, rest, i + 1):
                yield ((pid, combo),) + tail, leftover

    # ------------------------------------------------------------ marginals

    def hand_marginals(self) -> dict[PlayerId, dict[Card, float]]:
        """P(player holds at least one copy of card), per live player.

        This is a probability, so it counts *worlds*, not copies.  A player
        holding two Guards contributes one world to ``Guard``, not two -- the
        question at the table is "can they Guard me", and a second copy does
        not make that more true.  Use :meth:`expected_hand_counts` when you
        want copies.

        Derived from the world set, never tracked directly: opponents' hands
        are correlated through the deck composition, and independent marginals
        would happily claim two players both hold the Princess.
        """
        worlds = self.worlds()
        if not worlds:
            raise ValueError("no consistent world -- the constraints conflict")
        counts: dict[PlayerId, dict[Card, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        for w in worlds:
            for pid, hand in enumerate(w.hands):
                for card in set(hand):
                    counts[pid][card] += w.weight
        n = self._total_weight()
        return {
            pid: {card: k / n for card, k in sorted(c.items())}
            for pid, c in counts.items()
        }

    def expected_hand_counts(self) -> dict[PlayerId, dict[Card, float]]:
        """Expected number of copies of each card in each player's hand.

        Summed over cards this equals the player's hand size, which is the
        invariant :meth:`hand_marginals` deliberately does not satisfy.
        """
        worlds = self.worlds()
        counts: dict[PlayerId, dict[Card, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        for w in worlds:
            for pid, hand in enumerate(w.hands):
                for card in hand:
                    counts[pid][card] += w.weight
        n = self._total_weight()
        return {
            pid: {card: k / n for card, k in sorted(c.items())}
            for pid, c in counts.items()
        }

    def slot_marginals(self) -> dict[int, dict[Card, float]]:
        """P(slot holds card) for every live deck slot.

        Pinned slots are point masses.  Unpinned slots are exchangeable: given
        a world, every arrangement of that world's ``deck_pool`` is equally
        likely, so each unpinned slot has the pool's own composition.  This is
        computed, not enumerated -- see :meth:`_enumerate`.
        """
        worlds = self.worlds()
        if not worlds:
            raise ValueError("no consistent world -- the constraints conflict")
        c = self.constraints
        n = self._total_weight()
        out: dict[int, dict[Card, float]] = {}
        pinned = c.known_slot

        # Unpinned slots all share one distribution: the pool composition
        # averaged over worlds, excluding the set-aside position.
        shared: dict[Card, float] = defaultdict(float)
        for w in worlds:
            pool = Counter(w.deck_pool)
            total = sum(pool.values())
            if not total:
                continue
            for card, k in pool.items():
                shared[card] += w.weight * k / total
        shared_dist = {card: v / n for card, v in sorted(shared.items())}
        for slot in c.deck_slots:
            if slot in pinned:
                out[slot] = {pinned[slot]: 1.0}
            else:
                out[slot] = dict(shared_dist)
        return out

    def set_aside_marginal(self) -> dict[Card, float]:
        """Distribution over the facedown setup card."""
        c = self.constraints
        if c.set_aside_known is not None:
            return {c.set_aside_known: 1.0}
        worlds = self.worlds()
        shared: dict[Card, float] = defaultdict(float)
        for w in worlds:
            pool = Counter(w.deck_pool)
            total = sum(pool.values())
            if not total:
                continue
            for card, k in pool.items():
                shared[card] += w.weight * k / total
        n = self._total_weight()
        return {card: v / n for card, v in sorted(shared.items())}

    def next_draw(self) -> dict[Card, float]:
        """Distribution over the card the next draw will produce."""
        c = self.constraints
        if not c.deck_slots:
            return {}
        return self.slot_marginals().get(c.deck_slots[0], {})

    def _total_weight(self) -> float:
        """Sum of world weights, the normaliser for every marginal.

        Falls back to the world count if every weight is zero, which can only
        happen if a policy returned 0 for an action that actually occurred --
        a bug in the policy, not a real impossibility.
        """
        total = sum(w.weight for w in self.worlds())
        if total <= 0:
            return float(len(self.worlds())) or 1.0
        return total

    # --------------------------------------------------- information value

    def entropy(self) -> float:
        """Shannon entropy of the joint posterior, in bits.

        Measures how much is still unknown overall.  Zero means the hidden
        state is pinned down exactly.
        """
        total = self._total_weight()
        if total <= 0:
            return 0.0
        h = 0.0
        for w in self.worlds():
            p = w.weight / total
            if p > 0:
                h -= p * math.log2(p)
        return h

    def hand_entropy(self, pid: PlayerId) -> float:
        """Entropy of the distribution over ``pid``'s hand, in bits.

        This is what a Priest look collapses: after seeing their hand, the
        uncertainty about that player drops to zero.  So the entropy *is* the
        expected information gain from looking at them.
        """
        total = self._total_weight()
        if total <= 0:
            return 0.0
        by_hand: dict[tuple[Card, ...], float] = defaultdict(float)
        for w in self.worlds():
            by_hand[tuple(sorted(w.hands[pid]))] += w.weight
        h = 0.0
        for weight in by_hand.values():
            p = weight / total
            if p > 0:
                h -= p * math.log2(p)
        return h

    def information_gain(self, pid: PlayerId) -> float:
        """Expected bits learned by seeing ``pid``'s hand.

        Exactly :meth:`hand_entropy`: a Priest look reveals the hand with
        certainty, so the posterior over it collapses from ``H`` bits to zero.

        This exists because PIMC cannot see it.  Under a rollout policy that
        ignores what a Priest reveals, every Priest target scores *identically*
        -- not approximately, but to zero variance -- so the evaluator would
        report them as tied when they demonstrably are not.  The tool does not
        ignore the sighting: it enters the belief state and shapes the next
        real decision.  Ranking by information is the honest answer, and it is
        computable here without any rollout.

        Normalised per card held.  Raw entropy conflates two different things:
        a hand can be uncertain because we know little about it, or merely
        because it is *larger* -- a player mid-turn holds two cards and so
        scores higher whatever we know.  Dividing by hand size compares
        opponents like for like, which is the question actually being asked:
        whose card is least predictable?
        """
        size = self.constraints.hand_size(pid)
        if size <= 0:
            return 0.0
        return self.hand_entropy(pid) / size

    def effective_sample_size(self) -> float:
        """Kish effective sample size: how many worlds the weights really span.

        Weighting concentrates mass on fewer worlds. When this falls far below
        the world count, the posterior rests on a handful of assumptions about
        how opponents play, and should be read with corresponding suspicion.
        """
        worlds = self.worlds()
        s1 = sum(w.weight for w in worlds)
        s2 = sum(w.weight * w.weight for w in worlds)
        return (s1 * s1 / s2) if s2 > 0 else 0.0

    def world_count(self) -> int:
        """How many consistent worlds the posterior currently spans."""
        return len(self.worlds())

    def expected_counts(self) -> dict[Card, float]:
        """Expected copies of each card across all hidden locations.

        Summed with what is public, this must equal the variant's copy counts.
        """
        worlds = self.worlds()
        total: dict[Card, float] = defaultdict(float)
        for w in worlds:
            for hand in w.hands:
                for card in hand:
                    total[card] += w.weight
            for _, card in w.slots:
                total[card] += w.weight
            for card in w.deck_pool:
                total[card] += w.weight
            if w.set_aside is not None:
                total[w.set_aside] += w.weight
        n = self._total_weight()
        return {card: k / n for card, k in sorted(total.items())}


# ------------------------------------------------------------------ helpers


def _multiset_combinations(
    pool: Counter[Card], size: int
) -> Iterator[tuple[Card, ...]]:
    """Distinct size-``size`` selections from a multiset, sorted within each.

    A hand is unordered, so ``(Guard, King)`` and ``(King, Guard)`` are the
    same hand and must be yielded once.  Two Guards is a legal selection when
    two Guards remain.
    """
    cards = sorted(pool)

    def rec(i: int, left: int, acc: tuple[Card, ...]) -> Iterator[tuple[Card, ...]]:
        if left == 0:
            yield acc
            return
        if i >= len(cards):
            return
        card = cards[i]
        available = pool[card]
        for take in range(min(available, left) + 1):
            yield from rec(i + 1, left - take, acc + (card,) * take)

    return rec(0, size, ())


def _hand_allowed(c: Constraints, pid: PlayerId, hand: tuple[Card, ...]) -> bool:
    """Does ``hand`` violate any exclusion known for ``pid``?"""
    banned = c.excluded.get(pid)
    if banned and any(card in banned for card in hand):
        return False
    return True


def _relations_hold(c: Constraints, hands: Sequence[tuple[Card, ...]]) -> bool:
    """Check Baron facts against a candidate assignment.

    Strong forms apply to hands unchanged since the comparison, so the single
    held card is the compared card.  Weak forms apply to hands that have drawn
    since: the compared card is still in there somewhere, so it is enough that
    *some* pair of held cards satisfies the relation.
    """
    for hi, lo in c.greater_than:
        if not hands[hi] or not hands[lo]:
            continue
        if max(hands[hi]) <= max(hands[lo]):
            return False
    for a, b in c.equal_to:
        if not hands[a] or not hands[b]:
            continue
        if max(hands[a]) != max(hands[b]):
            return False
    for hi, lo in c.weak_greater:
        if not hands[hi] or not hands[lo]:
            continue
        if not any(x > y for x in hands[hi] for y in hands[lo]):
            return False
    for a, b in c.weak_equal:
        if not hands[a] or not hands[b]:
            continue
        if not any(x == y for x in hands[a] for y in hands[b]):
            return False
    for pid, value in c.beats_value:
        if not hands[pid]:
            continue
        if not any(card > value for card in hands[pid]):
            return False
    return True


def unknown_pool(c: Constraints, config: GameConfig) -> Counter[Card]:
    """Cards whose location is not yet pinned down."""
    pool = Counter(config.all_cards())
    for card in c.public:
        pool[card] -= 1
    for hand in c.known_hand.values():
        for card in hand:
            pool[card] -= 1
    # Only pins on slots still in the deck describe hidden cards. A stale pin
    # (its slot already drawn) would subtract a card that has since moved to a
    # hand or a discard pile, and the pool would go negative.
    for slot, card in c.known_slot.items():
        if slot in c.deck_slots:
            pool[card] -= 1
    if c.set_aside_known is not None:
        pool[c.set_aside_known] -= 1
    negative = {k: v for k, v in pool.items() if v < 0}
    if negative:
        raise ValueError(f"more cards accounted for than exist: {negative}")
    return Counter({k: v for k, v in pool.items() if v > 0})


# ------------------------------------------------------------------- replay


def replay(
    log: EventLog,
    viewer: PlayerId,
    n_players: int,
    *,
    config: GameConfig = STANDARD,
    faceup: Sequence[Card] = (),
    viewer_hand: Sequence[Card] = (),
    initial_slots: Sequence[int] | None = None,
) -> Constraints:
    """Fold a projected log into hard constraints.

    Pure: same log in, same constraints out.  Nothing is learned here that is
    not in ``log``, which is what makes the replay test a real check -- if the
    posterior cannot be rebuilt from the log alone, the log is missing an
    event.

    ``log`` must be the output of :func:`~loveletter.observation.observe` for
    ``viewer``.  Passing the engine's raw log makes the tracker omniscient.
    """
    c = Constraints(n_players=n_players)
    c.public.extend(faceup)
    # ``viewer_hand`` is the *dealt* card only. Every card the viewer drew
    # afterwards arrives through a Drew event, so seeding the current hand here
    # would double-count the draws. Passing the current hand is the common
    # mistake; ``dealt`` names what this actually wants.
    if viewer_hand:
        c.known_hand[viewer] = tuple(viewer_hand)

    # The deck after setup, before the first player's opening draw. Slot ids
    # are assigned 0..n-1 at setup, matching the engine.
    if initial_slots is None:
        initial_slots = range(config.undealt_deck_size(n_players))
    c.deck_slots = tuple(initial_slots)

    # Hand sizes track mid-turn state: everyone is dealt 1, the actor draws to
    # 2, and plays back down to 1. Eliminated players are forced to 0 below.
    sizes: dict[PlayerId, int] = {p: 1 for p in range(n_players)}

    for e in log.events:
        if isinstance(e, Dealt):
            if e.actor == viewer and e.card is not None:
                c.known_hand[viewer] = (e.card,)

        elif isinstance(e, Drew):
            sizes[e.actor] = sizes.get(e.actor, 1) + 1
            pinned_draw: Card | None = None
            if e.slot >= 0:
                c.deck_slots = tuple(s for s in c.deck_slots if s != e.slot)
                # A pinned slot that gets drawn stops being deck knowledge:
                # the card is in a hand now. Leaving the pin in place makes
                # ``unknown_pool`` subtract a card that is no longer hidden in
                # the deck, and the posterior goes empty a few turns later.
                pinned_draw = c.known_slot.pop(e.slot, None)
            if e.actor == viewer and e.card is not None:
                # The viewer sees their own draw, so their hand stays exactly
                # known -- but a draw still changes the hand, so relations
                # about it must weaken exactly as an opponent's would. Baron
                # facts compare the card held *at the time*; leaving them
                # strong makes them contradict the hand we can plainly see,
                # and the posterior collapses to no consistent world.
                seen = c.known_hand.get(viewer, ())
                c.note_draw(viewer)
                c.known_hand[viewer] = seen + (e.card,)
                c.must_hold.pop(viewer, None)
            else:
                # An opponent drew: they gained a card without losing one, so
                # a sighting narrows to "still holds that" instead of expiring.
                c.note_draw(e.actor)
                if pinned_draw is not None:
                    # They drew a card we had watched into a known slot, so we
                    # know they hold it -- the Chancellor's real payoff.
                    c.must_hold[e.actor] = c.must_hold.get(e.actor, ()) + (
                        pinned_draw,
                    )

        elif isinstance(e, Played):
            sizes[e.actor] = max(1, sizes.get(e.actor, 2) - 1)
            # Record the play before the discard is made public, so the context
            # describes what was on the table when the decision was taken.
            if e.actor != viewer:
                # Older plays by this player are now stale: they describe a
                # hand that has since been replaced by a draw. Scoring them
                # against the current card would count evidence about a card
                # the player no longer holds, and would multiply that error
                # once per play. Only the most recent play still describes
                # the hand we are reasoning about.
                c.observed_plays = [
                    rec for rec in c.observed_plays if rec[0] != e.actor
                ]
                c.observed_plays.append(
                    (
                        e.actor,
                        e.card,
                        PlayContext(
                            actor=e.actor,
                            discarded=tuple(c.public),
                            players_left=n_players - len(c.out),
                            deck_size=len(c.deck_slots),
                            all_others_protected=False,
                        ),
                        None,
                    )
                )
            c.public.append(e.card)
            # Baron facts describe the card that was compared. Playing a card
            # may have played that very card, so nothing need satisfy the
            # relation afterwards. Keeping them contradicts hands we can
            # plainly see and collapses the posterior to zero worlds.
            c.beats_value = [p for p in c.beats_value if p[0] != e.actor]
            c.weak_greater = [p for p in c.weak_greater if e.actor not in p]
            c.weak_equal = [p for p in c.weak_equal if e.actor not in p]
            c.greater_than = [p for p in c.greater_than if e.actor not in p]
            c.equal_to = [p for p in c.equal_to if e.actor not in p]
            held = c.must_hold.get(e.actor)
            if held is not None:
                # They played something. If it was the card we had seen, the
                # constraint is discharged; otherwise it still holds.
                if e.card in held:
                    rest = list(held)
                    rest.remove(e.card)
                    if rest:
                        c.must_hold[e.actor] = tuple(rest)
                    else:
                        c.must_hold.pop(e.actor, None)
                elif sizes[e.actor] <= len(held):
                    # They now hold exactly the cards we know about.
                    c.known_hand[e.actor] = held
                    c.must_hold.pop(e.actor, None)
            if e.actor == viewer:
                held = list(c.known_hand.get(viewer, ()))
                if e.card not in held:
                    raise ValueError(
                        f"viewer played {e.card} while the tracker had them "
                        f"holding {tuple(held)} -- the viewer's hand has "
                        f"diverged from the log, which means an event that "
                        f"changed it was not replayed"
                    )
                held.remove(e.card)
                c.known_hand[viewer] = tuple(held)

        elif isinstance(e, GuardResult):
            if e.hit:
                pass  # the Eliminated event carries the revealed card
            else:
                # Public: the target did not hold that card *at this moment*.
                c.excluded[e.target].add(e.guess)

        elif isinstance(e, PriestLook):
            # Only the viewer's own look is a hard constraint. An opponent's
            # look is logged and deliberately unread in Phase 2a.
            if e.actor == viewer and e.seen is not None:
                c.known_hand[e.target] = (e.seen,)

        elif isinstance(e, BaronCompare):
            if e.outcome == "tie":
                c.equal_to.append((e.actor, e.target))
            else:
                winner = e.actor if e.outcome == "target_out" else e.target
                loser = e.target if e.outcome == "target_out" else e.actor
                c.greater_than.append((winner, loser))
                # The loser's card is revealed by the Eliminated event that
                # follows; pin the bound to it there.
                c.pending_baron_win = (winner, loser)

        elif isinstance(e, KingTrade):
            # The two hands change places, so every per-player fact travels
            # with the cards it describes. Swapping some and dropping others
            # would leave a card recorded in two hands at once.
            a, b = e.actor, e.target
            for store in (c.known_hand, c.must_hold, c.excluded, c.at_most):
                va, vb = store.get(a), store.get(b)
                store.pop(a, None)
                store.pop(b, None)
                if vb is not None:
                    store[a] = vb
                if va is not None:
                    store[b] = va
            # Comparisons spoke about hands that no longer sit where they did.
            for rel in ("greater_than", "equal_to", "weak_greater", "weak_equal"):
                setattr(
                    c,
                    rel,
                    [p for p in getattr(c, rel) if a not in p and b not in p],
                )
            c.beats_value = [p for p in c.beats_value if p[0] not in (a, b)]
            # A trader saw the card handed to them, so their own hand stays
            # certain regardless of what was known about the other player.
            got = None
            if e.actor == viewer and e.actor_got is not None:
                got = e.actor_got
            elif e.target == viewer and e.target_got is not None:
                got = e.target_got
            if got is not None:
                # Received a card they can see, but the hand was replaced, so
                # facts about the old one must not survive.
                c.clear_player_inferences(viewer)
                c.known_hand[viewer] = (got,)

        elif isinstance(e, PrinceDiscard):
            c.public.append(e.discarded)
            if e.target == viewer:
                # The viewer's own hand is never uncertain: they discarded a
                # card they could see, and the replacement arrives as a Drew
                # whose identity is projected to them. Running the generic
                # "hand replaced, forget everything" path here would delete
                # their known hand and leave the redraw appended to nothing.
                # The viewer sees their own discard and replacement, so the
                # hand stays known -- but it was still *replaced*, so every
                # time-indexed fact about it dies exactly as it would for an
                # opponent. Skipping that leaves Baron bounds asserting things
                # about a card that is now in the discard pile.
                held = list(c.known_hand.get(viewer, ()))
                if e.discarded in held:
                    held.remove(e.discarded)
                if e.drew is not None:
                    held.append(e.drew)
                c.clear_player_inferences(viewer)
                c.known_hand[viewer] = tuple(held)
            else:
                c.clear_player_inferences(e.target)
            if e.slot >= 0:
                c.deck_slots = tuple(s for s in c.deck_slots if s != e.slot)
                drawn_pin = c.known_slot.pop(e.slot, None)
                if drawn_pin is not None and e.target != viewer:
                    c.must_hold[e.target] = (drawn_pin,)
            if e.from_set_aside:
                # The set-aside card entered a hand; it is no longer hidden
                # there, and if we knew it we now know their hand.
                if c.set_aside_known is not None:
                    c.known_hand[e.target] = (c.set_aside_known,)
                c.set_aside_known = None

        elif isinstance(e, ChancellorExchange):
            # Two slots left the deck and n new slots joined the bottom. The
            # cards are unchanged as a multiset but their slot mapping is new,
            # so drop any slot knowledge that the exchange invalidated.
            for slot in e.drew:
                c.known_slot.pop(slot, None)
            c.deck_slots = tuple(s for s in c.deck_slots if s not in e.drew)
            c.deck_slots = c.deck_slots + tuple(e.returned)
            if e.actor == viewer:
                # The viewer saw everything they drew and chose what to keep,
                # so their hand stays exactly known across the exchange.
                if e.kept is not None:
                    c.known_hand[viewer] = (e.kept,)
                # And they placed specific cards at specific bottom slots.
                # Pinning them is the entire payoff of the ordered-deck
                # representation -- the viewer will know, several turns from
                # now, exactly which card is about to be drawn. (This was
                # missed at first: the Phase 2 test verified the slot ids
                # moved but never that the identities stuck to them.)
                for slot, card in zip(e.returned, e.returned_cards):
                    c.known_slot[slot] = card
            else:
                c.clear_player_inferences(e.actor)

        elif isinstance(e, Eliminated):
            c.out.add(e.actor)
            c.known_hand.pop(e.actor, None)
            c.excluded.pop(e.actor, None)
            c.greater_than = [p for p in c.greater_than if e.actor not in p]
            c.equal_to = [p for p in c.equal_to if e.actor not in p]
            c.weak_greater = [p for p in c.weak_greater if e.actor not in p]
            c.weak_equal = [p for p in c.weak_equal if e.actor not in p]
            c.observed_plays = [
                rec for rec in c.observed_plays if rec[0] != e.actor
            ]
            if e.card is not None:
                c.public.append(e.card)
                pending = c.pending_baron_win
                if pending and pending[1] == e.actor:
                    # The loser's card is now public: convert the relation into
                    # a bound on a value, which survives the loser leaving.
                    c.beats_value.append((pending[0], e.card))
            c.pending_baron_win = None
            sizes[e.actor] = 0

        elif isinstance(e, RoundEnded):
            for pid, card in e.revealed:
                c.known_hand[pid] = (card,)

    # A fully known hand is authoritative about its own size. This matters for
    # constructed states whose log does not start at turn zero: the log cannot
    # describe cards that were never logged as drawn.
    for pid, hand in c.known_hand.items():
        sizes[pid] = len(hand)
    for pid in c.out:
        sizes[pid] = 0
    c.sizes = sizes
    return c
