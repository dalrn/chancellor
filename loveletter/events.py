"""The public record of a round.

The log records everything that *publicly happened*, whether or not any
current consumer reads it.  It is a record of the game, not of anyone's
beliefs: an opponent Priest look tells me nothing about the cards, but it did
happen, so it is logged.  Phase 2a's tracker simply does not read that field;
Phase 2b will, with no change to the log format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from .config import Card

PlayerId: TypeAlias = int


@dataclass(frozen=True, slots=True)
class Event:
    """Base class for logged events.  ``turn`` is the 0-based turn index."""

    turn: int
    actor: PlayerId


@dataclass(frozen=True, slots=True)
class Dealt(Event):
    """``actor`` was dealt their opening card during setup.

    Logged for every player. ``card`` is filled only for the viewer the log is
    projected for -- everyone sees that a deal happened, nobody sees anyone
    else's card. This is the tracker's starting point: without it, the viewer's
    own hand has to be reconstructed from the outcome, and cards that arrived
    by King trade or Prince redraw get miscounted as dealt.
    """

    card: Card | None = None


@dataclass(frozen=True, slots=True)
class Drew(Event):
    """``actor`` drew the card occupying deck slot ``slot``.

    ``slot`` is a stable slot id, not a position: positions shift as the deck
    is drawn from and as the Chancellor appends to the bottom, but a slot id
    names the same physical card for as long as it exists.  The belief tracker
    moves probability mass from that slot into ``actor``'s hand.

    ``card`` is filled only when the identity is public to this tool's user
    (their own draw).  For opponents it stays None and each particle resolves
    it from its own deck assignment.
    """

    slot: int = -1
    card: Card | None = None


@dataclass(frozen=True, slots=True)
class Played(Event):
    """``actor`` played ``card`` faceup.  Always fully public."""

    card: Card = Card.SPY
    target: PlayerId | None = None
    guess: Card | None = None
    fizzled: bool = False


@dataclass(frozen=True, slots=True)
class GuardResult(Event):
    """Public announcement of whether ``target`` held ``guess``."""

    target: PlayerId = 0
    guess: Card = Card.SPY
    hit: bool = False


@dataclass(frozen=True, slots=True)
class PriestLook(Event):
    """``actor`` privately saw ``target``'s hand.

    ``seen`` is filled only for looks this tool's user is entitled to know.
    The event is logged either way -- that the look happened is public.
    """

    target: PlayerId = 0
    seen: Card | None = None


@dataclass(frozen=True, slots=True)
class BaronCompare(Event):
    """Public outcome of a Baron comparison.

    On an elimination, the loser's card becomes public (logged separately as
    :class:`Eliminated`) and the winner's is known only to be strictly
    greater.  On a tie, nothing is revealed but both hands are known equal.
    """

    target: PlayerId = 0
    outcome: Literal["actor_out", "target_out", "tie"] = "tie"


@dataclass(frozen=True, slots=True)
class KingTrade(Event):
    """``actor`` and ``target`` swapped hands.

    The contents stay hidden from everyone else, but the two traders each
    physically receive a card and can see it.  ``actor_got``/``target_got``
    record that, projected so each trader sees only their own side.  Without
    them a trader's own hand becomes unknowable from the log, and the log
    stops being sufficient to rebuild the posterior.
    """

    target: PlayerId = 0
    actor_got: Card | None = None
    target_got: Card | None = None


@dataclass(frozen=True, slots=True)
class PrinceDiscard(Event):
    """``target`` discarded ``discarded`` faceup without resolving it.

    ``redrew`` is False when the discard was the Princess (the player is out
    and draws nothing) and False when the deck and set-aside are exhausted.
    """

    target: PlayerId = 0
    discarded: Card = Card.SPY
    redrew: bool = False
    from_set_aside: bool = False
    #: Deck slot the replacement came from; -1 for the set-aside card or when
    #: nothing was drawn (a discarded Princess).
    slot: int = -1
    #: The replacement card. Private to the target, who physically draws it;
    #: projected away for everyone else. Without it the target's own hand
    #: cannot be rebuilt from the log after they are Princed.
    drew: Card | None = None


@dataclass(frozen=True, slots=True)
class ChancellorExchange(Event):
    """``actor`` drew ``n`` slots and returned ``n`` slots to the bottom.

    Nothing about the card identities is revealed, but the mapping of cards to
    slots changes: ``drew`` names the slots taken into hand, ``returned`` names
    the slots now at the bottom in top-to-bottom order.  The union of the two
    is not the same multiset of slot ids -- the actor kept one card and put
    back others -- so the tracker must permute within the pool rather than
    assume slots stayed put.  This is why the deck is an ordered list of slots
    and not a multiset.
    """

    n: int = 0
    drew: tuple[int, ...] = ()
    returned: tuple[int, ...] = ()
    #: The card the actor kept, and the cards they put back, in order. Both are
    #: private to the actor -- projected away for everyone else -- but without
    #: them the actor's own hand cannot be reconstructed from the log, and the
    #: log would no longer be sufficient to rebuild the posterior.
    kept: Card | None = None
    returned_cards: tuple[Card, ...] = ()


@dataclass(frozen=True, slots=True)
class Eliminated(Event):
    """``actor`` is out of the round; ``card`` is their revealed hand."""

    card: Card | None = None
    reason: Literal["guard", "baron", "prince_princess", "played_princess"] = "guard"


@dataclass(frozen=True, slots=True)
class RoundEnded(Event):
    """Round over.  ``winners`` is plural: ties are real."""

    winners: tuple[PlayerId, ...] = ()
    reason: Literal["last_standing", "deck_out"] = "deck_out"
    revealed: tuple[tuple[PlayerId, Card], ...] = ()
    spy_bonus: PlayerId | None = None


@dataclass(slots=True)
class EventLog:
    """Append-only public record, with undo for CLI mistyping."""

    events: list[Event] = field(default_factory=list)

    def append(self, event: Event) -> None:
        self.events.append(event)

    def extend(self, events: list[Event]) -> None:
        self.events.extend(events)

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self):
        return iter(self.events)

    def of_type[E: Event](self, kind: type[E]) -> list[E]:
        return [e for e in self.events if isinstance(e, kind)]
