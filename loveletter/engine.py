"""Phase 1: the rules engine.

Pure and deterministic. No I/O, no global randomness -- any shuffling is done
by an injected ``random.Random``. This module must not import the belief
tracker, the evaluator, or the CLI.

Deck representation
-------------------
The deck is an **ordered list of slots**, index 0 == top (next to be drawn),
index -1 == bottom. A multiset will not do: the Chancellor places specific
cards at specific bottom positions in a chosen order, and those cards are
drawn back in that order several turns later.

Chancellor accounting
---------------------
The Chancellor is already in the play area when its effect resolves, so the
hand holds 1 card at that moment, not 2. Draw ``k = min(2, deck_size)``, hold
``1 + k``, keep 1, return ``k`` to the bottom. Deck size is therefore
unchanged by a Chancellor, and hand size at end of turn is always 1.
"""

from __future__ import annotations

import random
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Sequence, TypeAlias

from .config import COUNTESS_FORCERS, STANDARD, TARGETS_OTHER, Card, GameConfig
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


#: When True, ``apply``/``apply_unchecked`` skip the per-transition invariant
#: checks (card conservation, slot integrity, hand sizes).
#:
#: Those checks caught most of the Phase 1 bugs and every state transition
#: should be running them -- but they rebuild and compare full card multisets
#: twice per turn, which profiling showed to be a large share of rollout time.
#: A PIMC search replays millions of transitions whose inputs came from
#: ``legal_actions`` and are already known good.
#:
#: Never set this globally. Use :func:`unchecked_transitions` around the
#: rollout loop so it cannot leak into real play, where a silent state
#: corruption would poison every probability the tool reports.
_SKIP_INVARIANTS = False


class RulesError(Exception):
    """An illegal action, or a violated invariant."""


@contextmanager
def unchecked_transitions():
    """Skip per-transition invariant checks inside this block.

    For determinized rollouts only, where every action came from
    ``legal_actions`` on a state the engine itself produced. Restores the
    previous setting on exit, including on exception, so a crash mid-search
    cannot leave real play running unchecked.
    """
    global _SKIP_INVARIANTS
    previous = _SKIP_INVARIANTS
    _SKIP_INVARIANTS = True
    try:
        yield
    finally:
        _SKIP_INVARIANTS = previous


@dataclass(frozen=True, slots=True)
class Action:
    """One play. Target/guess/return-order are None when inapplicable."""

    card: Card
    target: PlayerId | None = None
    guess: Card | None = None
    #: Chancellor: cards put back, in order; first goes higher in the deck.
    chancellor_return: tuple[Card, ...] | None = None

    def __str__(self) -> str:
        parts = [str(self.card)]
        if self.target is not None:
            parts.append(f"-> P{self.target}")
        if self.guess is not None:
            parts.append(f"guessing {self.guess}")
        if self.chancellor_return:
            back = ", ".join(str(c) for c in self.chancellor_return)
            parts.append(f"returning [{back}]")
        return " ".join(parts)


@dataclass(frozen=True, slots=True)
class PlayerState:
    """One player's public and private state within a round."""

    pid: PlayerId
    hand: tuple[Card, ...] = ()
    #: Cards played or force-discarded, in order. Public.
    discards: tuple[Card, ...] = ()
    out: bool = False
    protected: bool = False
    tokens: int = 0

    @property
    def in_round(self) -> bool:
        return not self.out

    def played_spy(self) -> bool:
        return Card.SPY in self.discards


@dataclass(frozen=True, slots=True)
class GameState:
    """Complete state of a round, including hidden information."""

    config: GameConfig
    players: tuple[PlayerState, ...]
    #: Ordered deck. Index 0 is the top; the last element is the bottom.
    deck: tuple[Card, ...]
    #: The facedown setup card. Real, unknown, out of play -- except that a
    #: Prince on an empty deck draws it.
    set_aside: Card | None
    #: The 3 faceup setup cards in a 2-player game. Public from turn zero.
    faceup: tuple[Card, ...] = ()
    current: PlayerId = 0
    turn: int = 0
    round_over: bool = False
    winners: tuple[PlayerId, ...] = ()
    spy_bonus: PlayerId | None = None
    log: EventLog = field(default_factory=EventLog)
    #: Every card this state was built with. The conservation reference.
    census: tuple[Card, ...] = ()
    #: Stable slot ids, parallel to :attr:`deck`. Positions shift as cards are
    #: drawn and as the Chancellor appends to the bottom; a slot id names the
    #: same physical card for as long as that card sits in the deck. The log
    #: refers to slots, never positions, so a replay can follow the movement.
    slots: tuple[int, ...] = ()
    #: Next unused slot id. Slot ids are never recycled within a round.
    next_slot: int = 0

    # ---------------------------------------------------------------- access

    @property
    def n_players(self) -> int:
        return len(self.players)

    @property
    def deck_size(self) -> int:
        return len(self.deck)

    def player(self, pid: PlayerId) -> PlayerState:
        return self.players[pid]

    def active(self) -> list[PlayerId]:
        """Players still in the round, in seat order."""
        return [p.pid for p in self.players if p.in_round]

    def targetable(self, by: PlayerId, *, include_self: bool) -> list[PlayerId]:
        """Players ``by`` may choose: in the round and not Handmaid-protected.

        A Handmaid protects you from *others*, never from your own Prince.
        """
        out: list[PlayerId] = []
        for p in self.players:
            if p.out:
                continue
            if p.pid == by:
                if include_self:
                    out.append(p.pid)
                continue
            if p.protected:
                continue
            out.append(p.pid)
        return out

    # ------------------------------------------------------------ invariants

    def all_cards(self) -> list[Card]:
        """Every card, wherever it currently is. Used for conservation."""
        cards: list[Card] = []
        for p in self.players:
            cards.extend(p.hand)
            cards.extend(p.discards)
        cards.extend(self.deck)
        cards.extend(self.faceup)
        if self.set_aside is not None:
            cards.append(self.set_aside)
        return cards

    def check_conservation(self) -> None:
        """Assert no card is created, destroyed, or duplicated.

        The reference multiset is :attr:`census`, fixed when the state was
        built.  Full games set it to the whole variant deck; constructed test
        and rollout states set it to whatever cards they contain.  Either way
        ``apply`` may never change the total.
        """
        have = Counter(self.all_cards())
        want = Counter(self.census)
        if have != want:
            diff = {
                c: have.get(c, 0) - want.get(c, 0)
                for c in set(have) | set(want)
                if have.get(c, 0) != want.get(c, 0)
            }
            raise RulesError(f"card conservation violated: {diff}")

    def check_full_deck(self) -> None:
        """Assert this state holds every card of the variant. Real games only."""
        if Counter(self.all_cards()) != Counter(self.config.all_cards()):
            raise RulesError("state does not contain the full variant deck")

    def check_slots(self) -> None:
        """Slot ids must be parallel to the deck and unique."""
        if len(self.slots) != len(self.deck):
            raise RulesError(
                f"{len(self.slots)} slot ids for {len(self.deck)} deck cards"
            )
        if len(set(self.slots)) != len(self.slots):
            raise RulesError("duplicate slot ids in the deck")
        if any(sid >= self.next_slot for sid in self.slots):
            raise RulesError("slot id beyond the allocation watermark")

    def check_hand_sizes(self, *, actor_has_drawn: bool = True) -> None:
        """Out players hold 0 cards; live players 1, except the actor.

        The invariant has two shapes depending on when it is checked.  Between
        turns -- which is what ``apply`` returns -- the next player has already
        drawn, so the actor holds 2.  Immediately after a card resolves but
        before that draw, every live player holds 1; pass
        ``actor_has_drawn=False`` there.  A finished round has no actor.
        """
        acting = (
            self.current
            if actor_has_drawn
            and not self.round_over
            and self.player(self.current).in_round
            else None
        )
        for p in self.players:
            if p.out:
                expect = 0
            else:
                expect = 2 if p.pid == acting else 1
            if len(p.hand) != expect:
                raise RulesError(
                    f"P{p.pid} holds {len(p.hand)} cards, expected {expect}"
                )

    def _replace_player(self, pid: PlayerId, **changes) -> tuple[PlayerState, ...]:
        players = list(self.players)
        players[pid] = replace(players[pid], **changes)
        return tuple(players)


# --------------------------------------------------------------------- setup


def new_round(
    n_players: int,
    rng: random.Random,
    *,
    config: GameConfig = STANDARD,
    first_player: PlayerId = 0,
    tokens: Sequence[int] | None = None,
) -> GameState:
    """Shuffle, set aside, deal. ``rng`` is injected; never global."""
    config.validate_players(n_players)
    deck = config.all_cards()
    rng.shuffle(deck)

    set_aside = deck.pop(0)
    faceup = tuple(deck.pop(0) for _ in range(config.faceup_count(n_players)))

    tok = list(tokens) if tokens is not None else [0] * n_players
    if len(tok) != n_players:
        raise ValueError("tokens length must match n_players")

    players = tuple(
        PlayerState(pid=i, hand=(deck.pop(0),), tokens=tok[i])
        for i in range(n_players)
    )
    state = GameState(
        config=config,
        players=players,
        deck=tuple(deck),
        set_aside=set_aside,
        faceup=faceup,
        current=first_player,
        turn=0,
        census=tuple(config.all_cards()),
        slots=tuple(range(len(deck))),
        next_slot=len(deck),
        log=EventLog(
            [
                Dealt(turn=0, actor=p.pid, card=p.hand[0])
                for p in players
            ]
        ),
    )
    if state.deck_size != config.undealt_deck_size(n_players):
        raise RulesError("setup produced the wrong deck size")
    state.check_conservation()
    state.check_full_deck()
    state.check_slots()
    state.check_hand_sizes(actor_has_drawn=False)
    # The first player draws to open the round, so every state handed to
    # ``apply`` has the acting player holding 2 cards.
    return _draw_for_turn(state, first_player, advance=False)


def state_from_hands(
    hands: Sequence[Sequence[Card]],
    deck: Sequence[Card],
    set_aside: Card | None,
    *,
    config: GameConfig = STANDARD,
    faceup: Sequence[Card] = (),
    current: PlayerId = 0,
    tokens: Sequence[int] | None = None,
    discards: Sequence[Sequence[Card]] | None = None,
    out: Sequence[bool] | None = None,
    protected: Sequence[bool] | None = None,
) -> GameState:
    """Build an exact state for tests and for determinized rollouts."""
    n = len(hands)
    tok = list(tokens) if tokens is not None else [0] * n
    disc = [tuple(d) for d in discards] if discards is not None else [()] * n
    outs = list(out) if out is not None else [False] * n
    prot = list(protected) if protected is not None else [False] * n
    players = tuple(
        PlayerState(
            pid=i,
            hand=tuple(hands[i]),
            discards=disc[i],
            out=outs[i],
            protected=prot[i],
            tokens=tok[i],
        )
        for i in range(n)
    )
    state = GameState(
        config=config,
        players=players,
        deck=tuple(deck),
        set_aside=set_aside,
        faceup=tuple(faceup),
        current=current,
        slots=tuple(range(len(deck))),
        next_slot=len(deck),
        # Constructed states start mid-turn, so the "deal" is whatever each
        # player is holding. The actor's second card is logged as a draw so
        # the log describes a reachable history rather than a 2-card deal.
        #
        # Pre-set discards are logged as Played events. They are public cards
        # at the table, and a belief rebuilt from the log must see them --
        # otherwise a constructed state silently loses its discard history and
        # the posterior keeps cards in play that everyone can see are gone.
        log=EventLog(
            [Dealt(turn=0, actor=p.pid, card=p.hand[0]) for p in players if p.hand]
            + [
                Played(turn=0, actor=p.pid, card=card)
                for p in players
                for card in p.discards
            ]
            + [
                Drew(turn=0, actor=current, slot=-1, card=players[current].hand[1])
                for _ in range(1)
                if len(players[current].hand) > 1
            ]
        ),
    )
    return replace(state, census=tuple(state.all_cards()))


# ------------------------------------------------------------ legal actions


def _ordered_returns(pool: Sequence[Card], k: int) -> list[tuple[Card, ...]]:
    """Distinct ordered ways to return ``k`` cards from ``pool``, keeping 1.

    Order matters -- returned cards are drawn back in that order -- so these
    are permutations, deduplicated across repeated card values.
    """
    if k == 0:
        return [()]
    seen: set[tuple[Card, ...]] = set()
    out: list[tuple[Card, ...]] = []
    if k == 1:
        for i in range(len(pool)):
            combo = (pool[i],)
            if combo not in seen:
                seen.add(combo)
                out.append(combo)
        return out
    for i in range(len(pool)):
        for j in range(len(pool)):
            if i == j:
                continue
            combo = (pool[i], pool[j])
            if combo not in seen:
                seen.add(combo)
                out.append(combo)
    return out


def legal_actions(state: GameState) -> list[Action]:
    """Every legal play for the current player, who holds 2 cards.

    Enforces both compulsions: the Countess with King/Prince co-held, and the
    Prince self-target when every other live player is protected.
    """
    if state.round_over:
        return []
    me = state.player(state.current)
    if me.out:
        raise RulesError(f"P{me.pid} is out of the round")
    if len(me.hand) != 2:
        raise RulesError(f"P{me.pid} must hold 2 cards to act, has {len(me.hand)}")

    hand = me.hand
    if Card.COUNTESS in hand and any(c in COUNTESS_FORCERS for c in hand):
        return [Action(Card.COUNTESS)]

    actions: list[Action] = []
    for i, card in enumerate(hand):
        if card in hand[:i]:  # two of a kind: one set of actions
            continue
        actions.extend(_actions_for_card(state, card, hand, i))
    if not actions:
        raise RulesError("no legal actions -- engine bug")
    return actions


def _actions_for_card(
    state: GameState, card: Card, hand: tuple[Card, ...], idx: int
) -> list[Action]:
    me = state.current

    if card in TARGETS_OTHER:
        others = state.targetable(me, include_self=False)
        if not others:
            return [Action(card)]  # all others protected: no effect
        if card is Card.GUARD:
            guessable = [c for c in state.config.counts if c is not Card.GUARD]
            return [
                Action(card, target=t, guess=g) for t in others for g in guessable
            ]
        return [Action(card, target=t) for t in others]

    if card is Card.PRINCE:
        # Compulsion 2: with no other live target, self-target only.
        return [
            Action(card, target=t)
            for t in state.targetable(me, include_self=True)
        ]

    if card is Card.CHANCELLOR:
        k = min(2, state.deck_size)
        if k == 0:
            return [Action(card)]  # empty deck: no effect
        kept = tuple(c for j, c in enumerate(hand) if j != idx)
        pool = kept + tuple(state.deck[:k])
        return [
            Action(card, chancellor_return=ret) for ret in _ordered_returns(pool, k)
        ]

    return [Action(card)]  # Spy, Handmaid, Countess, Princess


# ---------------------------------------------------------------- resolution


def _draw(deck: tuple[Card, ...]) -> tuple[Card, tuple[Card, ...]]:
    if not deck:
        raise RulesError("draw from an empty deck")
    return deck[0], deck[1:]


def _slot_at(state: GameState, index: int) -> int:
    """Slot id at deck position ``index``, or -1 if unallocated."""
    return state.slots[index] if index < len(state.slots) else -1


def apply(state: GameState, action: Action) -> GameState:
    """Validate ``action``, resolve it, and return a new state.

    Use this for anything that came from outside the engine -- above all the
    CLI, where a mistyped entry that is silently accepted would corrupt every
    subsequent probability.  Validation is O(number of legal actions), which is
    dominated by Guard guesses.

    Rollouts should call :func:`apply_unchecked` instead: they only ever feed
    back actions that came out of :func:`legal_actions`, so re-validating is
    millions of repetitions of a question already answered.
    """
    if state.round_over:
        raise RulesError("the round is over")
    if action not in legal_actions(state):
        raise RulesError(f"illegal action: {action}")
    return apply_unchecked(state, action)


def apply_unchecked(state: GameState, action: Action) -> GameState:
    """Resolve ``action`` without validating it. Never mutates ``state``.

    The caller guarantees ``action`` came from :func:`legal_actions` for this
    exact state.  Passing anything else is a programming error and may raise a
    :class:`RulesError` from an invariant, or silently corrupt the state.

    The current player holds 2 cards on entry; on exit the next player has
    drawn and holds 2.
    """
    me = state.current
    events: list[Event] = []
    players = list(state.players)
    deck = state.deck
    slots = state.slots
    next_slot = state.next_slot
    set_aside = state.set_aside

    # The played card leaves the hand *before* its effect resolves.
    hand = list(players[me].hand)
    hand.remove(action.card)
    players[me] = replace(
        players[me],
        hand=tuple(hand),
        discards=players[me].discards + (action.card,),
        protected=False,  # your own Handmaid lapses as your turn begins
    )
    fizzled = action.target is None and action.card in TARGETS_OTHER
    events.append(
        Played(
            turn=state.turn,
            actor=me,
            card=action.card,
            target=action.target,
            guess=action.guess,
            fizzled=fizzled,
        )
    )

    eliminated: list[tuple[PlayerId, Card | None, str]] = []
    card = action.card
    tgt = action.target

    if card is Card.PRINCESS:
        eliminated.append((me, None, "played_princess"))

    elif card is Card.HANDMAID:
        players[me] = replace(players[me], protected=True)

    elif card is Card.GUARD and tgt is not None:
        assert action.guess is not None
        hit = action.guess in players[tgt].hand
        events.append(
            GuardResult(
                turn=state.turn,
                actor=me,
                target=tgt,
                guess=action.guess,
                hit=hit,
            )
        )
        if hit:
            eliminated.append((tgt, players[tgt].hand[0], "guard"))

    elif card is Card.PRIEST and tgt is not None:
        # Logged as a public fact; the seen card is private to the actor.
        events.append(
            PriestLook(
                turn=state.turn, actor=me, target=tgt, seen=players[tgt].hand[0]
            )
        )

    elif card is Card.BARON and tgt is not None:
        mine, theirs = players[me].hand[0], players[tgt].hand[0]
        if mine > theirs:
            outcome = "target_out"
            eliminated.append((tgt, theirs, "baron"))
        elif theirs > mine:
            outcome = "actor_out"
            eliminated.append((me, mine, "baron"))
        else:
            outcome = "tie"
        events.append(
            BaronCompare(turn=state.turn, actor=me, target=tgt, outcome=outcome)
        )

    elif card is Card.KING and tgt is not None:
        mine, theirs = players[me].hand, players[tgt].hand
        players[me] = replace(players[me], hand=theirs)
        players[tgt] = replace(players[tgt], hand=mine)
        events.append(
            KingTrade(
                turn=state.turn,
                actor=me,
                target=tgt,
                actor_got=theirs[0] if theirs else None,
                target_got=mine[0] if mine else None,
            )
        )

    elif card is Card.PRINCE and tgt is not None:
        discarded = players[tgt].hand[0]
        players[tgt] = replace(
            players[tgt],
            hand=(),
            discards=players[tgt].discards + (discarded,),
        )
        redrew = False
        from_set_aside = False
        draw_slot = -1
        if discarded is Card.PRINCESS:
            # Out immediately, and draws nothing.
            eliminated.append((tgt, None, "prince_princess"))
        elif deck:
            draw_slot = slots[0] if slots else -1
            drawn, deck = _draw(deck)
            slots = slots[1:]
            players[tgt] = replace(players[tgt], hand=(drawn,))
            redrew = True
        elif set_aside is not None:
            players[tgt] = replace(players[tgt], hand=(set_aside,))
            set_aside = None
            redrew = True
            from_set_aside = True
        else:
            raise RulesError("Prince target has nothing to draw")
        events.append(
            PrinceDiscard(
                turn=state.turn,
                actor=me,
                target=tgt,
                discarded=discarded,
                redrew=redrew,
                from_set_aside=from_set_aside,
                slot=draw_slot,
                drew=players[tgt].hand[0] if players[tgt].hand else None,
            )
        )

    elif card is Card.CHANCELLOR:
        k = min(2, len(deck))
        drew_slots: tuple[int, ...] = ()
        ret_slots: tuple[int, ...] = ()
        if k:
            ret = action.chancellor_return or ()
            if len(ret) != k:
                raise RulesError(f"Chancellor must return {k}, got {len(ret)}")
            drew_slots = tuple(slots[:k])
            pool = list(players[me].hand) + list(deck[:k])
            deck = deck[k:]
            slots = slots[k:]
            for c in ret:  # validate the return is drawn from what was held
                if c not in pool:
                    raise RulesError(f"cannot return {c}: not in hand")
                pool.remove(c)
            if len(pool) != 1:
                raise RulesError("Chancellor must keep exactly 1 card")
            players[me] = replace(players[me], hand=(pool[0],))
            # Fresh slot ids: the returned cards are a new mapping of cards to
            # positions, not the slots they arrived in.
            ret_slots = tuple(range(next_slot, next_slot + k))
            next_slot += k
            deck = deck + tuple(ret)  # first returned sits higher
            slots = slots + ret_slots
        events.append(
            ChancellorExchange(
                turn=state.turn,
                actor=me,
                n=k,
                drew=drew_slots,
                returned=ret_slots,
                kept=players[me].hand[0] if players[me].hand else None,
                returned_cards=tuple(action.chancellor_return or ()),
            )
        )

    # Spy and Countess have no play effect.

    for pid, revealed, reason in eliminated:
        p = players[pid]
        hand_out = p.hand
        players[pid] = replace(
            players[pid],
            hand=(),
            out=True,
            protected=False,
            discards=p.discards + hand_out,
        )
        events.append(
            Eliminated(
                turn=state.turn,
                actor=pid,
                card=revealed if revealed is not None else (
                    hand_out[0] if hand_out else None
                ),
                reason=reason,  # type: ignore[arg-type]
            )
        )

    log = EventLog(list(state.log.events) + events)
    new = replace(
        state,
        players=tuple(players),
        deck=deck,
        slots=slots,
        next_slot=next_slot,
        set_aside=set_aside,
        log=log,
    )
    if not _SKIP_INVARIANTS:
        new.check_conservation()
        new.check_slots()
        new.check_hand_sizes(actor_has_drawn=False)
    return _finish_turn(new)


# ------------------------------------------------------------- turn / ending


def _finish_turn(state: GameState) -> GameState:
    """Run the end-of-turn checks, then advance to the next live player."""
    if is_round_over(state):
        return _end_round(state)

    # Deck size is >= 1 at the start of every turn: the round would have
    # ended after the previous turn otherwise.
    if state.deck_size < 1:
        raise RulesError("next turn would start with an empty deck")

    return _draw_for_turn(state, _next_player(state, state.current))


def _draw_for_turn(
    state: GameState, pid: PlayerId, *, advance: bool = True
) -> GameState:
    """Give ``pid`` their draw and make it their turn.

    Used both to open the round and to advance between turns, so the acting
    player always holds 2 cards when ``legal_actions`` is called.
    """
    drawn, deck = _draw(state.deck)
    slot = _slot_at(state, 0)
    players = list(state.players)
    players[pid] = replace(
        players[pid],
        hand=players[pid].hand + (drawn,),
        protected=False,  # your Handmaid lapses at the start of your turn
    )
    turn = state.turn + 1 if advance else state.turn
    log = EventLog(
        list(state.log.events)
        + [Drew(turn=turn, actor=pid, slot=slot, card=drawn)]
    )
    new = replace(
        state,
        players=tuple(players),
        deck=deck,
        slots=state.slots[1:],
        current=pid,
        turn=turn,
        log=log,
    )
    if not _SKIP_INVARIANTS:
        new.check_conservation()
        new.check_slots()
    return new


def _next_player(state: GameState, after: PlayerId) -> PlayerId:
    n = state.n_players
    for step in range(1, n + 1):
        pid = (after + step) % n
        if state.players[pid].in_round:
            return pid
    raise RulesError("no live player to act")


def is_round_over(state: GameState) -> bool:
    """True if the round has ended.

    Checked **after any turn**, never mid-turn: a Chancellor can empty the
    deck and refill it within one turn, and an emptiness check placed inside
    the turn would end rounds that should continue.
    """
    if state.round_over:
        return True
    if len(state.active()) <= 1:
        return True
    return state.deck_size == 0


def round_winners(state: GameState) -> list[PlayerId]:
    """Winners of the round. Plural: ties are real and every tied player wins."""
    live = state.active()
    if not live:
        raise RulesError(
            "zero players left: unreachable, since no effect eliminates two "
            "players at once and the round ends the instant one remains"
        )
    # One-left is checked before deck-out. Same winner either way; fixing the
    # order keeps tests deterministic.
    if len(live) == 1:
        return live
    best = max(state.player(p).hand[0] for p in live)
    return [p for p in live if state.player(p).hand[0] == best]


def spy_bonus_winner(state: GameState) -> PlayerId | None:
    """The lone surviving Spy player, if exactly one qualifies.

    Only players still in the round at the end count: a Spy played by someone
    later knocked out does not block the bonus. Both Spies by one player is
    still just one token.
    """
    holders = [p for p in state.active() if state.player(p).played_spy()]
    return holders[0] if len(holders) == 1 else None


def _end_round(state: GameState) -> GameState:
    live = state.active()
    reason = "last_standing" if len(live) <= 1 else "deck_out"
    winners = round_winners(state)
    bonus = spy_bonus_winner(state)

    players = list(state.players)
    for pid in winners:
        players[pid] = replace(players[pid], tokens=players[pid].tokens + 1)
    if bonus is not None:
        players[bonus] = replace(players[bonus], tokens=players[bonus].tokens + 1)

    # Hands are revealed only on a deck-out comparison. A last-standing win
    # ends with the survivor's card still hidden at a real table, and the log
    # must not know more than the table does.
    revealed = (
        tuple((p, state.player(p).hand[0]) for p in live if state.player(p).hand)
        if reason == "deck_out"
        else ()
    )
    log = EventLog(
        list(state.log.events)
        + [
            RoundEnded(
                turn=state.turn,
                actor=state.current,
                winners=tuple(winners),
                reason=reason,  # type: ignore[arg-type]
                revealed=revealed,
                spy_bonus=bonus,
            )
        ]
    )
    return replace(
        state,
        players=tuple(players),
        round_over=True,
        winners=tuple(winners),
        spy_bonus=bonus,
        log=log,
    )
