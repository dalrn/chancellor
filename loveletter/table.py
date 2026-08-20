"""Phase 4 core: the table tracker behind the CLI.

At a real table nobody holds the engine's ``GameState`` -- the hidden hands
are hidden.  What the user *can* know is exactly the projected event log that
:func:`loveletter.observation.observe` produces, and that log is the one thing
:class:`loveletter.belief.Belief` consumes.  So this class maintains that log
directly, from typed-in public facts, plus the public bookkeeping (deck slots,
protection, eliminations, tokens) needed to validate entries and drive the
display.

Correctness contract
--------------------
For any sequence of real-game events, the log built here must equal the
engine's log projected for this seat.  The test suite enforces that literally:
random engine games are re-entered through this class fact by fact, and the
logs are compared event for event.  If they ever diverge, beliefs built at the
table would silently differ from beliefs built in the engine, and every
probability shown would be wrong -- so divergence is an error, never a
tolerance.

Validation
----------
Every entry is checked against what is publicly knowable and rejected loudly
(:class:`EntryError`) when impossible: a card whose copies are all visible, a
protected or eliminated target, a Guard naming Guard, an out-of-order actor.
A silently accepted bad entry corrupts every subsequent probability, which is
worse than any interruption at the table.

Undo restores the complete previous state, including the log.  Mistyping is
assumed.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Sequence

from .config import COUNTESS_FORCERS, STANDARD, Card, GameConfig
from .events import (
    BaronCompare,
    ChancellorExchange,
    Dealt,
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


class EntryError(Exception):
    """A typed-in fact that cannot be true. Rejected loudly, never absorbed."""


@dataclass(slots=True)
class SeatState:
    """Public knowledge about one seat."""

    discards: list[Card] = field(default_factory=list)
    out: bool = False
    protected: bool = False
    tokens: int = 0


class Table:
    """Public state of one round, as seen from ``my_seat``."""

    def __init__(
        self,
        n_players: int,
        my_seat: PlayerId,
        my_dealt: Card,
        *,
        faceup: Sequence[Card] = (),
        first_player: PlayerId = 0,
        config: GameConfig = STANDARD,
    ) -> None:
        config.validate_players(n_players)
        if n_players == 2 and len(faceup) != config.faceup_count(2):
            raise EntryError(
                f"a 2-player round sets {config.faceup_count(2)} cards faceup; "
                f"got {len(faceup)}"
            )
        if n_players != 2 and faceup:
            raise EntryError("faceup setup cards exist only in 2-player rounds")
        if not 0 <= my_seat < n_players:
            raise EntryError(f"seat {my_seat} out of range for {n_players} players")

        self.config = config
        self.n_players = n_players
        self.me = my_seat
        self.faceup = tuple(faceup)
        self.seats = [SeatState() for _ in range(n_players)]
        #: My actual cards. 1 between turns, 2 mid-turn.
        self.my_hand: list[Card] = [my_dealt]
        #: Live deck slot ids, top first -- mirrors the engine exactly.
        undealt = config.undealt_deck_size(n_players)
        self.deck_slots: list[int] = list(range(undealt))
        self.next_slot: int = undealt
        self.set_aside_available = True
        self.current: PlayerId = first_player
        self.turn = 0
        self.awaiting_draw = True
        self.round_over = False
        self.winners: tuple[PlayerId, ...] = ()
        self.log = EventLog(
            [
                Dealt(turn=0, actor=p, card=my_dealt if p == my_seat else None)
                for p in range(n_players)
            ]
        )
        self._snapshots: list[dict] = []
        self.pending_reveal = False
        #: A Chancellor of mine has been played but not yet resolved. Between
        #: the play and the resolution I am physically holding cards the tool
        #: has not been told about, so every other entry is blocked.
        self.pending_chancellor = False
        #: What I reported drawing, once entered. None until then.
        self.pending_drawn: list[Card] | None = None
        # extra_mine=1: the dealt card is already counted inside my_hand.
        self._check_entry_possible(my_dealt, extra_mine=1)

    # ------------------------------------------------------------------ undo

    def snapshot(self) -> None:
        """Save the complete state. Called before every mutating entry."""
        self._snapshots.append(
            copy.deepcopy(
                {
                    "seats": self.seats,
                    "my_hand": self.my_hand,
                    "deck_slots": self.deck_slots,
                    "next_slot": self.next_slot,
                    "set_aside_available": self.set_aside_available,
                    "current": self.current,
                    "turn": self.turn,
                    "awaiting_draw": self.awaiting_draw,
                    "round_over": self.round_over,
                    "pending_reveal": self.pending_reveal,
                    "pending_chancellor": self.pending_chancellor,
                    "pending_drawn": self.pending_drawn,
                    "winners": self.winners,
                    "events": list(self.log.events),
                }
            )
        )

    def undo(self) -> None:
        """Restore the state before the last entry."""
        if not self._snapshots:
            raise EntryError("nothing to undo")
        s = self._snapshots.pop()
        self.seats = s["seats"]
        self.my_hand = s["my_hand"]
        self.deck_slots = s["deck_slots"]
        self.next_slot = s["next_slot"]
        self.set_aside_available = s["set_aside_available"]
        self.current = s["current"]
        self.turn = s["turn"]
        self.awaiting_draw = s["awaiting_draw"]
        self.round_over = s["round_over"]
        self.pending_reveal = s["pending_reveal"]
        self.pending_chancellor = s["pending_chancellor"]
        self.pending_drawn = s["pending_drawn"]
        self.winners = s["winners"]
        self.log = EventLog(s["events"])

    # ------------------------------------------------------------ validation

    def visible_count(self, card: Card) -> int:
        """Copies of ``card`` I can currently see (discards, faceup, my hand)."""
        n = sum(s.discards.count(card) for s in self.seats)
        n += self.faceup.count(card)
        n += self.my_hand.count(card)
        return n

    def _check_entry_possible(self, card: Card, *, extra_mine: int = 0) -> None:
        """Reject a card entry that exceeds the copies that exist.

        ``extra_mine`` discounts copies about to leave my hand in the same
        entry (playing a card I hold does not double-count it).
        """
        have = self.visible_count(card) - extra_mine
        if have + 1 > self.config.copies(card):
            raise EntryError(
                f"impossible: all {self.config.copies(card)} cop"
                f"{'y is' if self.config.copies(card) == 1 else 'ies are'} "
                f"of {card} already visible"
            )

    def _require_turn(self, actor: PlayerId) -> None:
        if self.round_over:
            raise EntryError("the round is over")
        if self.pending_chancellor:
            raise EntryError(
                "your Chancellor is unresolved -- enter `drew ...` and then "
                "`keep ... ret ...` first"
            )
        if actor != self.current:
            raise EntryError(
                f"it is P{self.current}'s turn, not P{actor}'s"
            )
        if self.awaiting_draw:
            raise EntryError(
                f"P{actor} must draw before playing (use the draw entry)"
            )

    def _check_target(
        self, actor: PlayerId, card: Card, target: PlayerId | None
    ) -> None:
        others = [
            p
            for p in range(self.n_players)
            if p != actor and not self.seats[p].out and not self.seats[p].protected
        ]
        if card in (Card.GUARD, Card.PRIEST, Card.BARON, Card.KING):
            if target is None:
                if others:
                    raise EntryError(
                        f"{card} fizzles only when every other live player is "
                        f"protected; {['P%d' % p for p in others]} are targetable"
                    )
                return
            if target == actor:
                raise EntryError(f"{card} cannot target its own player")
        if card is Card.PRINCE:
            if target is None:
                raise EntryError("Prince always has a target (possibly self)")
            if not others and target != actor:
                raise EntryError(
                    "everyone else is protected: the Prince must self-target"
                )
        if target is not None:
            if not 0 <= target < self.n_players:
                raise EntryError(f"no such player P{target}")
            if self.seats[target].out:
                raise EntryError(f"P{target} is out of the round")
            if target != actor and self.seats[target].protected:
                raise EntryError(f"P{target} is Handmaid-protected")

    # --------------------------------------------------------------- entries

    def draw(self, card: Card | None = None) -> None:
        """The current player draws. ``card`` is given only when it is mine."""
        if self.round_over:
            raise EntryError("the round is over")
        if self.pending_chancellor:
            raise EntryError(
                "your Chancellor is unresolved -- enter `drew ...` and then "
                "`keep ... ret ...` first"
            )
        if not self.awaiting_draw:
            raise EntryError(f"P{self.current} has already drawn")
        if not self.deck_slots:
            raise EntryError("the deck is empty -- the round should have ended")
        if self.current == self.me:
            if card is None:
                raise EntryError("enter the card you drew")
            self._check_entry_possible(card)
        elif card is not None:
            raise EntryError(
                f"P{self.current}'s draw is hidden -- enter it without a card"
            )
        self.snapshot()
        if self.current == self.me:
            self.my_hand.append(card)
        slot = self.deck_slots.pop(0)
        self.seats[self.current].protected = False
        self.log.append(
            Drew(turn=self.turn, actor=self.current, slot=slot, card=card)
        )
        self.awaiting_draw = False

    def play(
        self,
        card: Card,
        *,
        target: PlayerId | None = None,
        guess: Card | None = None,
        # Effect facts, entered with the play so one line logs one turn:
        hit: bool | None = None,  # Guard
        seen: Card | None = None,  # Priest, when I am the looker
        baron_loser: PlayerId | None = None,  # Baron: who was out (None=tie)
        baron_revealed: Card | None = None,  # Baron: the loser's card
        king_got: Card | None = None,  # King: the card I received, if involved
        prince_discarded: Card | None = None,  # Prince: target's faceup discard
        prince_drew: Card | None = None,  # Prince: replacement, if target is me
        chancellor_kept: Card | None = None,  # Chancellor: mine only
        chancellor_returned: Sequence[Card] = (),  # Chancellor: mine, in order
        revealed: Card | None = None,  # Princess self-play: the other card
    ) -> None:
        """Record one complete turn: the play and its public outcome."""
        actor = self.current
        self._require_turn(actor)
        mine = actor == self.me

        if mine:
            if card not in self.my_hand:
                raise EntryError(f"you are not holding {card}")
            # The compulsion: Countess in hand + a forcer being played means
            # the entry is illegal. (An earlier version tested the *kept*
            # card against the forcer set, which is backwards and never
            # fired -- caught by the compulsion test, not by inspection.)
            if Card.COUNTESS in self.my_hand and card in COUNTESS_FORCERS:
                raise EntryError(
                    "holding the Countess with the King or a Prince, the "
                    "Countess must be played"
                )
        else:
            self._check_entry_possible(card)

        if card is Card.GUARD and guess is Card.GUARD:
            raise EntryError("a Guard cannot name Guard")
        if card is Card.GUARD and target is not None and guess is None:
            raise EntryError("a targeted Guard needs a guess")
        self._check_target(actor, card, target)

        self.snapshot()
        try:
            self._apply_play(
                actor, mine, card, target, guess, hit, seen, baron_loser,
                baron_revealed, king_got, prince_discarded, prince_drew,
                chancellor_kept, chancellor_returned, revealed,
            )
        except EntryError:
            # Resolution may have mutated state before the rejection; restore
            # the pre-entry snapshot so a rejected entry leaves no trace, then
            # re-raise. Popping alone would leave corrupted state behind.
            self.undo()
            raise

    # ------------------------------------------------------------ resolution

    def _apply_play(
        self, actor, mine, card, target, guess, hit, seen, baron_loser,
        baron_revealed, king_got, prince_discarded, prince_drew,
        chancellor_kept, chancellor_returned, revealed,
    ) -> None:
        seat = self.seats[actor]
        if mine:
            self.my_hand.remove(card)
        seat.discards.append(card)
        seat.protected = False
        fizzled = target is None and card in (
            Card.GUARD, Card.PRIEST, Card.BARON, Card.KING
        )
        self.log.append(
            Played(
                turn=self.turn, actor=actor, card=card,
                target=target, guess=guess, fizzled=fizzled,
            )
        )

        if card is Card.PRINCESS:
            self._eliminate(actor, self._own_revealed(actor, revealed),
                            "played_princess")
        elif card is Card.HANDMAID:
            seat.protected = True
        elif card is Card.GUARD and target is not None:
            if hit is None:
                raise EntryError("enter whether the Guard hit or missed")
            self.log.append(
                GuardResult(turn=self.turn, actor=actor, target=target,
                            guess=guess, hit=hit)
            )
            if hit:
                self._eliminate(target, guess, "guard")
        elif card is Card.PRIEST and target is not None:
            if mine and seen is None:
                raise EntryError("enter the card you saw")
            if not mine and seen is not None:
                raise EntryError("you cannot see an opponent's Priest look")
            if seen is not None and target != self.me:
                self._check_entry_possible(seen)
            self.log.append(
                PriestLook(turn=self.turn, actor=actor, target=target, seen=seen)
            )
        elif card is Card.BARON and target is not None:
            self._baron(actor, target, baron_loser, baron_revealed)
        elif card is Card.KING and target is not None:
            self._king(actor, target, king_got)
        elif card is Card.PRINCE and target is not None:
            self._prince(actor, target, prince_discarded, prince_drew)
        elif card is Card.CHANCELLOR:
            self._chancellor(actor, mine, chancellor_kept, chancellor_returned)
        # Spy and Countess: no effect.

        if self.pending_chancellor:
            return  # the turn stays open until the Chancellor is resolved
        self._finish_turn()

    def _own_revealed(self, actor: PlayerId, revealed: Card | None) -> Card | None:
        """The card a self-eliminated player reveals: mine is known."""
        if actor == self.me:
            return self.my_hand[0] if self.my_hand else None
        if revealed is None:
            raise EntryError("enter the card they revealed when eliminated")
        return revealed

    def _eliminate(self, pid: PlayerId, card: Card | None, reason: str) -> None:
        if pid == self.me:
            card = self.my_hand[0] if self.my_hand else card
            self.my_hand.clear()
        if card is not None and pid != self.me:
            self._check_entry_possible(card)
        seat = self.seats[pid]
        if card is not None:
            seat.discards.append(card)
        seat.out = True
        seat.protected = False
        self.log.append(
            Eliminated(turn=self.turn, actor=pid, card=card, reason=reason)  # type: ignore[arg-type]
        )

    def _baron(self, actor, target, loser, revealed_card) -> None:
        if loser is None:
            outcome = "tie"
        elif loser == target:
            outcome = "target_out"
        elif loser == actor:
            outcome = "actor_out"
        else:
            raise EntryError("the Baron loser must be one of the two compared")
        self.log.append(
            BaronCompare(turn=self.turn, actor=actor, target=target,
                         outcome=outcome)  # type: ignore[arg-type]
        )
        if loser is not None:
            if loser == self.me:
                revealed_card = self.my_hand[0] if self.my_hand else None
            elif revealed_card is None:
                raise EntryError("enter the card the Baron loser revealed")
            self._eliminate(loser, revealed_card, "baron")

    def _king(self, actor, target, king_got) -> None:
        involved = self.me in (actor, target)
        if involved:
            if king_got is None:
                raise EntryError("enter the card you received in the trade")
            self._check_entry_possible(king_got, extra_mine=0)
            given = self.my_hand.pop() if self.my_hand else None
            self.my_hand.append(king_got)
            actor_got = king_got if actor == self.me else None
            target_got = king_got if target == self.me else None
            # The card I gave away is what the *other* side received; that is
            # their private knowledge, not mine to log.
            del given
        else:
            if king_got is not None:
                raise EntryError("you cannot see a trade you are not part of")
            actor_got = target_got = None
        self.log.append(
            KingTrade(turn=self.turn, actor=actor, target=target,
                      actor_got=actor_got, target_got=target_got)
        )

    def _prince(self, actor, target, discarded, drew) -> None:
        if target == self.me:
            discarded = self.my_hand[0] if self.my_hand else None
        if discarded is None:
            raise EntryError("enter the card the Prince made them discard")
        if target != self.me:
            self._check_entry_possible(discarded)
        seat = self.seats[target]
        if target == self.me:
            self.my_hand.clear()
        seat.discards.append(discarded)

        if discarded is Card.PRINCESS:
            seat.out = True
            seat.protected = False
            self.log.append(
                PrinceDiscard(turn=self.turn, actor=actor, target=target,
                              discarded=discarded, redrew=False, slot=-1)
            )
            self.log.append(
                Eliminated(turn=self.turn, actor=target, card=None,
                           reason="prince_princess")
            )
            return

        from_set_aside = not self.deck_slots
        if from_set_aside:
            if not self.set_aside_available:
                raise EntryError("nothing left for the Prince target to draw")
            self.set_aside_available = False
            slot = -1
        else:
            slot = self.deck_slots.pop(0)
        if target == self.me:
            if drew is None:
                raise EntryError("enter the replacement card you drew")
            self._check_entry_possible(drew)
            self.my_hand.append(drew)
        elif drew is not None:
            raise EntryError("you cannot see an opponent's replacement draw")
        self.log.append(
            PrinceDiscard(turn=self.turn, actor=actor, target=target,
                          discarded=discarded, redrew=True,
                          from_set_aside=from_set_aside, slot=slot,
                          drew=drew)
        )

    def _chancellor(self, actor, mine, kept, returned) -> None:
        k = min(2, len(self.deck_slots))
        returned = tuple(returned)

        if mine and k and kept is None and not returned:
            # Two-step entry: `p chancellor` bare. The play is logged; the
            # draw and the keep/return arrive as separate entries, because
            # that is the order things happen at the table -- the cards are
            # unknown until picked up, and the keep choice deserves its own
            # recommendation. No slots move yet: the one ChancellorExchange
            # event is emitted at resolution so the log matches the engine's
            # single-event shape exactly, whichever entry style was used.
            self.pending_chancellor = True
            self.pending_drawn = None
            return

        drew_slots = tuple(self.deck_slots[:k])
        del self.deck_slots[:k]
        ret_slots = tuple(range(self.next_slot, self.next_slot + k))
        self.next_slot += k
        self.deck_slots.extend(ret_slots)
        if mine:
            if k:
                if kept is None or len(returned) != k:
                    raise EntryError(
                        f"enter the card you kept and the {k} you returned, "
                        f"in order (top first) -- or play `p chancellor` "
                        f"bare and enter the draw and decision step by step"
                    )
                self.my_hand.clear()
                self.my_hand.append(kept)
            else:
                # Empty deck: no exchange, but the engine still records the
                # kept (= only remaining) card, and the logs must match.
                kept = self.my_hand[0] if self.my_hand else None
                returned = ()
        elif kept is not None or returned:
            raise EntryError("you cannot see an opponent's Chancellor cards")
        self.log.append(
            ChancellorExchange(
                turn=self.turn, actor=actor, n=k,
                drew=drew_slots, returned=ret_slots,
                kept=kept if mine else None,
                returned_cards=returned if mine else (),
            )
        )

    # ---------------------------------------------- two-step chancellor entry

    def chancellor_drawn(self, cards: Sequence[Card]) -> int:
        """Report the cards physically drawn for a pending Chancellor.

        Returns how many were expected, for the CLI's messaging.
        """
        if not self.pending_chancellor:
            raise EntryError("no Chancellor is waiting for its draw")
        if self.pending_drawn is not None:
            raise EntryError(
                "the draw is already entered -- `keep ... ret ...` to resolve,"
                " or `u` to re-enter it"
            )
        k = min(2, len(self.deck_slots))
        cards = list(cards)
        if len(cards) != k:
            raise EntryError(f"the deck offers {k} card(s), not {len(cards)}")
        # Cumulative check: the pair counts together, so `drew princess
        # princess` is rejected even though each alone would pass.
        for i, card in enumerate(cards):
            already = cards[:i].count(card)
            if self.visible_count(card) + already + 1 > self.config.copies(card):
                raise EntryError(
                    f"impossible: that would make more copies of {card} than "
                    f"exist"
                )
        self.snapshot()
        self.pending_drawn = cards
        return k

    def chancellor_resolve(self, kept: Card, returned: Sequence[Card]) -> None:
        """Resolve a pending Chancellor: keep one, return the rest in order."""
        if not self.pending_chancellor:
            raise EntryError("no Chancellor is waiting to be resolved")
        if self.pending_drawn is None:
            raise EntryError("enter the drawn cards first: `drew <card> ...`")
        pool = list(self.my_hand) + list(self.pending_drawn)
        chosen = [kept, *returned]
        check = list(pool)
        for card in chosen:
            if card not in check:
                raise EntryError(
                    f"{card} is not among the cards in hand "
                    f"({', '.join(str(c) for c in pool)})"
                )
            check.remove(card)
        if check:
            raise EntryError(
                f"keep 1 and return {len(pool) - 1}: "
                f"{', '.join(str(c) for c in check)} unaccounted for"
            )
        self.snapshot()
        k = len(pool) - 1
        drew_slots = tuple(self.deck_slots[:k])
        del self.deck_slots[:k]
        ret_slots = tuple(range(self.next_slot, self.next_slot + k))
        self.next_slot += k
        self.deck_slots.extend(ret_slots)
        self.my_hand = [kept]
        self.log.append(
            ChancellorExchange(
                turn=self.turn, actor=self.me, n=k,
                drew=drew_slots, returned=ret_slots,
                kept=kept, returned_cards=tuple(returned),
            )
        )
        self.pending_chancellor = False
        self.pending_drawn = None
        self._finish_turn()

    # ------------------------------------------------------------- turn flow

    def active(self) -> list[PlayerId]:
        return [p for p in range(self.n_players) if not self.seats[p].out]

    def _finish_turn(self) -> None:
        live = self.active()
        if len(live) <= 1:
            self._end_round(live, "last_standing")
            return
        if not self.deck_slots:
            # Deck out: hands are revealed at the table; the CLI collects them
            # via end_round(). Until then the round is flagged as pending.
            self.round_over = True
            self.pending_reveal = True
            return
        self.current = self._next_player()
        self.turn += 1
        self.awaiting_draw = True

    def _next_player(self) -> PlayerId:
        p = self.current
        for _ in range(self.n_players):
            p = (p + 1) % self.n_players
            if not self.seats[p].out:
                return p
        raise EntryError("no live player to act")

    def end_round(self, revealed: dict[PlayerId, Card] | None = None) -> None:
        """Finish a deck-out round with the revealed hands."""
        if not self.pending_reveal:
            raise EntryError("the round is not waiting for revealed hands")
        self.snapshot()
        live = self.active()
        revealed = dict(revealed or {})
        if self.me in live and self.my_hand:
            revealed.setdefault(self.me, self.my_hand[0])
        missing = [p for p in live if p not in revealed]
        if missing:
            raise EntryError(
                f"enter the revealed hands of {['P%d' % p for p in missing]}"
            )
        best = max(revealed[p] for p in live)
        winners = [p for p in live if revealed[p] == best]
        self._end_round(
            winners, "deck_out",
            revealed=tuple((p, revealed[p]) for p in live),
        )

    def _end_round(self, winners, reason, revealed=()) -> None:
        spy_holders = [
            p for p in self.active() if Card.SPY in self.seats[p].discards
        ]
        bonus = spy_holders[0] if len(spy_holders) == 1 else None
        for p in winners:
            self.seats[p].tokens += 1
        if bonus is not None:
            self.seats[bonus].tokens += 1
        self.round_over = True
        self.pending_reveal = False
        self.winners = tuple(winners)
        self.log.append(
            RoundEnded(turn=self.turn, actor=self.current,
                       winners=tuple(winners), reason=reason,  # type: ignore[arg-type]
                       revealed=tuple(revealed), spy_bonus=bonus)
        )

    # -------------------------------------------------------------- belief

    def belief(self, *, policy=None, rng=None):
        """The posterior over hidden state, from this table's own log."""
        from .belief import Belief

        return Belief.from_log(
            self.log, self.me, self.n_players,
            config=self.config, faceup=self.faceup,
            initial_slots=None, rng=rng, policy=policy,
        )
