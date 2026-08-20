"""Projecting the true event log down to what one player may know.

The engine's log records ground truth, including card identities that a given
player has no right to see.  The belief tracker must never read that log
directly: if it did, it would silently be omniscient and every probability it
produced would be wrong in a way no test of its arithmetic would catch.

``observe(log, viewer)`` is the only supported bridge.  It returns the same
event sequence with hidden identities blanked -- opponents' draws lose their
``card``, opponents' Priest looks lose their ``seen`` -- while keeping every
structural fact: who drew, from which slot, how many cards moved where.

That structural residue is the requirement the tracker is built against: the
projected log must be sufficient to reconstruct the posterior from turn zero.
If replaying it does not reproduce the current belief state, something
happened that was not logged.
"""

from __future__ import annotations

from dataclasses import replace

from .events import (
    ChancellorExchange,
    Dealt,
    Drew,
    KingTrade,
    PrinceDiscard,
    Event,
    EventLog,
    PlayerId,
    PriestLook,
    RoundEnded,
)


def observe_event(event: Event, viewer: PlayerId) -> Event:
    """Blank the parts of ``event`` that ``viewer`` is not entitled to see."""
    if isinstance(event, Dealt):
        # Everyone sees that each player was dealt a card; only the viewer
        # sees their own.
        if event.actor != viewer:
            return replace(event, card=None)
        return event

    if isinstance(event, Drew):
        # Everyone sees that a draw happened and which slot emptied; only the
        # drawer sees what it was.
        if event.actor != viewer:
            return replace(event, card=None)
        return event

    if isinstance(event, ChancellorExchange):
        # Slot movement is public; which card was kept and which went back is
        # known only to the actor.
        if event.actor != viewer:
            return replace(event, kept=None, returned_cards=())
        return event

    if isinstance(event, KingTrade):
        # Each trader sees the card they received; nobody else sees either.
        if viewer == event.actor:
            return replace(event, target_got=None)
        if viewer == event.target:
            return replace(event, actor_got=None)
        return replace(event, actor_got=None, target_got=None)

    if isinstance(event, PrinceDiscard):
        # The discard is faceup and public; the replacement is drawn into the
        # target's hand and seen only by them.
        if event.target != viewer:
            return replace(event, drew=None)
        return event

    if isinstance(event, PriestLook):
        # The look is public; the card is not. Logged either way -- that the
        # look happened is a fact about the game, and Phase 2b will use it.
        if event.actor != viewer:
            return replace(event, seen=None)
        return event

    # Played, GuardResult, BaronCompare, KingTrade, PrinceDiscard,
    # ChancellorExchange, Eliminated and RoundEnded are public as logged:
    # every identity they carry was revealed faceup at the table.
    return event


def observe(log: EventLog, viewer: PlayerId) -> EventLog:
    """Project the whole log into ``viewer``'s view of the round."""
    return EventLog([observe_event(e, viewer) for e in log.events])


def hidden_identities(log: EventLog, viewer: PlayerId) -> int:
    """Count events whose identity is withheld from ``viewer``.

    A sanity handle for tests: a projection that hides nothing is a projection
    that is not working.
    """
    n = 0
    for e in log.events:
        if isinstance(e, Drew) and e.actor != viewer and e.card is not None:
            n += 1
        elif isinstance(e, PriestLook) and e.actor != viewer and e.seen is not None:
            n += 1
    return n
