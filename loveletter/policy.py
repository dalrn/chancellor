"""Opponent models: how likely was the play we actually saw?

Phase 2b reweights candidate worlds by how plausible each opponent's observed
actions were given the hand that world assigns them.  That requires an explicit
model of how opponents choose, which is a *belief about people*, not a rule of
the game.  It therefore lives behind an interface and never inside the tracker.

The contract
------------
A policy answers one question: given that a player held ``hand`` and the table
looked like ``context``, what was the probability they would play ``action``?
The tracker multiplies those probabilities across every observed play to weight
each world.

Probabilities must be strictly positive for any legal action.  A policy that
returns zero asserts an opponent *could never* make that play, which turns a
soft inference into a hard constraint and can eliminate the true world -- the
one failure mode worse than being vague.  :class:`HeuristicPolicy` clamps to
``MIN_PROB`` for exactly this reason.

The Countess is why this phase exists
-------------------------------------
The rulebook deliberately leaves an ambiguity: playing the Countess is forced
when you also hold the King or a Prince, and permitted when you hold neither.
Observers cannot tell which happened.  So a Countess play is strong evidence of
a King or Prince alongside it -- ``P(play | forced) = 1`` against a small
bluffing probability -- and that single update is worth a great deal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .config import COUNTESS_FORCERS, Card, GameConfig
from .events import PlayerId

#: Floor for any legal action's probability.  Never return 0 from a policy:
#: a zero is a hard constraint wearing soft clothing, and it can prune the
#: true world out of the posterior.
MIN_PROB = 1e-3


@dataclass(frozen=True, slots=True)
class PlayContext:
    """What was publicly true when a play was made.

    Deliberately small.  A policy that needs more should get it added here
    explicitly rather than reaching into the tracker, so the dependency stays
    one-directional: policies read context, the tracker reads policies.
    """

    actor: PlayerId
    #: Cards already public when the play was made.
    discarded: tuple[Card, ...] = ()
    #: How many players were still in the round.
    players_left: int = 2
    #: Cards remaining in the deck at that moment.
    deck_size: int = 0
    #: Whether every other live player was Handmaid-protected, which forces
    #: some plays and makes others pointless.
    all_others_protected: bool = False


class OpponentPolicy(Protocol):
    """How an opponent chooses, expressed as P(action | hand)."""

    def play_probability(
        self,
        hand: Sequence[Card],
        played: Card,
        context: PlayContext,
    ) -> float:
        """P(this player plays ``played`` | they hold ``hand``).

        ``hand`` is the 2-card hand *before* the play.  The result must be
        strictly positive whenever ``played`` is in ``hand``.
        """
        ...


class UniformPolicy:
    """The null model: every world keeps the weight it had.

    Returns a constant, deliberately.  The obvious implementation --
    ``1 / len(set(hand))`` -- is *not* a null model: it hands 1.0 to a world
    where the player held a pair and 0.5 to one where they held two different
    cards, so pairs get double weight and the posterior shifts. That is a real
    modelling claim (that pairs are twice as likely), smuggled in under the
    name "uniform".

    A constant leaves the Phase 2a posterior untouched, which is what makes
    this usable as the control when judging whether a richer policy helps.
    """

    def play_probability(
        self,
        hand: Sequence[Card],
        played: Card,
        context: PlayContext,
    ) -> float:
        if played not in hand:
            return 0.0
        return 1.0


@dataclass(frozen=True, slots=True)
class HeuristicPolicy:
    """A simple, legible model of a competent human player.

    Every number here is a guess about people, not a fact about the game.
    They are named and exposed so they can be argued with and tuned, rather
    than buried as literals in the reweighting loop.
    """

    #: Propensity to play the Countess while holding neither King nor Prince.
    #: Low, but never zero -- it is a legal and occasionally good bluff, and
    #: zeroing it would make every Countess play *prove* a King or Prince,
    #: which is exactly the overconfidence the rulebook invites.
    #:
    #: This is an unnormalised score, not a probability, and it competes with
    #: the other card's score (~1.0-1.6). Raising it never fully erases the
    #: Countess signal: with a King or Prince the play is *compulsory*, so it
    #: scores 1.0 against the other card's 0. That asymmetry is a rule of the
    #: game, not a modelling assumption, and it survives any bluff setting.
    #: To disable the inference entirely, use :class:`UniformPolicy`.
    countess_bluff: float = 0.05

    #: Weight on simply keeping the higher card, which is what wins a
    #: deck-out. Applied as a soft preference, not a rule.
    keep_high_bias: float = 0.6

    #: How much less likely a player is to play the Princess (instant loss)
    #: or to Prince themselves without reason.
    self_destruct_penalty: float = 0.02

    def play_probability(
        self,
        hand: Sequence[Card],
        played: Card,
        context: PlayContext,
    ) -> float:
        if played not in hand:
            return 0.0
        cards = tuple(hand)
        if len(set(cards)) == 1:
            return 1.0  # two of a kind: the choice is not a choice

        # Floor the scores *before* normalising, not the result after. Doing
        # it after leaves the distribution summing to more than 1, and a
        # policy whose probabilities do not sum to 1 silently biases every
        # world weight it touches.
        scores = {
            c: max(self._score(c, cards, context), MIN_PROB) for c in set(cards)
        }
        total = sum(scores.values())
        if total <= 0:
            return 1.0 / len(scores)
        return scores[played] / total

    def _score(
        self, card: Card, hand: tuple[Card, ...], context: PlayContext
    ) -> float:
        """Unnormalised propensity to play ``card`` from ``hand``."""
        other = next((c for c in hand if c is not card), card)

        # Playing the Princess loses on the spot. Almost nobody does it.
        if card is Card.PRINCESS:
            return self.self_destruct_penalty

        # The Countess is compulsory alongside the King or a Prince, and a
        # rare bluff otherwise. This is the single most informative case.
        if card is Card.COUNTESS:
            return 1.0 if other in COUNTESS_FORCERS else self.countess_bluff

        # Holding the Countess with a King or Prince, the *other* card cannot
        # legally be played at all.
        if Card.COUNTESS in hand and card in COUNTESS_FORCERS:
            return 0.0

        # Otherwise: a mild preference for keeping the higher card, so the
        # lower one gets played.
        keep_value = other / 9.0
        base = 1.0 + self.keep_high_bias * keep_value

        # A Prince aimed only at yourself is usually bad; when everyone else
        # is protected that is the only legal target.
        if card is Card.PRINCE and context.all_others_protected:
            if other is Card.PRINCESS:
                return self.self_destruct_penalty
            base *= 0.5

        # Targeted cards do nothing when everyone else is protected, so a
        # player who has a choice tends to keep them for a turn that matters.
        if context.all_others_protected and card in (
            Card.GUARD,
            Card.PRIEST,
            Card.BARON,
            Card.KING,
        ):
            base *= 0.3

        return base


DEFAULT_POLICY: OpponentPolicy = HeuristicPolicy()
