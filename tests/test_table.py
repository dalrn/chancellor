"""Phase 4 core tests: the table tracker.

The load-bearing test is the oracle: play random engine games, re-enter every
public fact through :class:`Table` exactly as a user at a real table would,
and require the resulting log to equal ``observe(engine_log, seat)`` event for
event. Beliefs are a pure function of that log, so log equality *is* belief
equality -- a divergence anywhere would mean the numbers shown at the table
silently differ from the numbers every arena measurement validated.
"""

from __future__ import annotations

import random
import unittest

from loveletter.config import Card
from loveletter.engine import apply, legal_actions, new_round
from loveletter.events import (
    BaronCompare,
    ChancellorExchange,
    Dealt,
    Drew,
    Eliminated,
    GuardResult,
    KingTrade,
    Played,
    PriestLook,
    PrinceDiscard,
    RoundEnded,
)
from loveletter.observation import observe
from loveletter.table import EntryError, Table

C = Card

EFFECTS = (
    GuardResult, PriestLook, BaronCompare, KingTrade,
    PrinceDiscard, ChancellorExchange, Eliminated,
)


def reenter(events, me, n_players, faceup) -> Table:
    """Re-enter a projected engine log through the Table, as a user would.

    Consumes only what the projection exposes -- the same information a
    player at the table has, no more.
    """
    dealt = next(e.card for e in events if isinstance(e, Dealt) and e.actor == me)
    first = next(e.actor for e in events if isinstance(e, Drew))
    table = Table(n_players, me, dealt, faceup=faceup, first_player=first)

    i = sum(1 for e in events if isinstance(e, Dealt))
    while i < len(events):
        e = events[i]
        if isinstance(e, Drew):
            table.draw(e.card)
            i += 1
        elif isinstance(e, Played):
            j = i + 1
            group = []
            while j < len(events) and isinstance(events[j], EFFECTS):
                group.append(events[j])
                j += 1
            table.play(e.card, target=e.target, guess=e.guess,
                       **_facts(e, group, me))
            i = j
        elif isinstance(e, RoundEnded):
            if table.pending_reveal:
                table.end_round(dict(e.revealed))
            i += 1
        else:
            raise AssertionError(f"unhandled event {e}")
    return table


def _facts(play: Played, group, me) -> dict:
    """Extract the public facts a user would type in for one turn."""
    kw: dict = {}
    for e in group:
        if isinstance(e, GuardResult):
            kw["hit"] = e.hit
        elif isinstance(e, PriestLook):
            if e.seen is not None:
                kw["seen"] = e.seen
        elif isinstance(e, BaronCompare):
            if e.outcome == "tie":
                kw["baron_loser"] = None
            else:
                kw["baron_loser"] = (
                    e.target if e.outcome == "target_out" else e.actor
                )
        elif isinstance(e, KingTrade):
            got = e.actor_got if e.actor == me else e.target_got
            if got is not None:
                kw["king_got"] = got
        elif isinstance(e, PrinceDiscard):
            kw["prince_discarded"] = e.discarded
            if e.drew is not None:
                kw["prince_drew"] = e.drew
        elif isinstance(e, ChancellorExchange):
            if e.kept is not None:
                kw["chancellor_kept"] = e.kept
                kw["chancellor_returned"] = e.returned_cards
        elif isinstance(e, Eliminated):
            if e.reason == "baron" and e.actor != me:
                kw["baron_revealed"] = e.card
            elif e.reason == "played_princess" and e.actor != me:
                kw["revealed"] = e.card
    return kw


class TestOracle(unittest.TestCase):
    """The table's log must equal the engine's projected log, always."""

    def test_random_games_reproduce_the_projected_log(self) -> None:
        for seed in range(30):
            for n in (2, 3, 4, 5):
                rng = random.Random(seed * 10 + n)
                state = new_round(n, rng)
                while not state.round_over:
                    state = apply(state, rng.choice(legal_actions(state)))
                for me in range(n):
                    events = observe(state.log, me).events
                    with self.subTest(seed=seed, n=n, me=me):
                        table = reenter(events, me, n, state.faceup)
                        self.assertEqual(
                            len(table.log.events), len(events),
                            "event counts differ",
                        )
                        for a, b in zip(table.log.events, events):
                            self.assertEqual(
                                a, b, f"table produced {a}, engine {b}"
                            )

    def test_beliefs_from_the_table_equal_beliefs_from_the_engine(self) -> None:
        """The point of the whole exercise, checked directly."""
        from loveletter.belief import Belief

        rng = random.Random(7)
        state = new_round(3, rng)
        for _ in range(5):
            if state.round_over:
                break
            state = apply(state, rng.choice(legal_actions(state)))
        events = observe(state.log, 0).events
        table = reenter(events, 0, 3, state.faceup)

        from_engine = Belief.from_log(
            observe(state.log, 0), 0, 3, faceup=state.faceup
        ).hand_marginals()
        from_table = table.belief().hand_marginals()
        self.assertEqual(from_engine, from_table)


class TestValidation(unittest.TestCase):
    """Impossible entries are rejected loudly and leave no trace."""

    def _table(self) -> Table:
        return Table(3, 0, C.GUARD, first_player=0)

    def test_out_of_turn_play_is_rejected(self) -> None:
        t = self._table()
        t.draw(C.SPY)
        t.play(C.SPY)
        with self.assertRaises(EntryError):
            t.play(C.BARON)  # P1 has not drawn yet

    def test_playing_before_drawing_is_rejected(self) -> None:
        t = self._table()
        with self.assertRaises(EntryError):
            t.play(C.GUARD, target=1, guess=C.BARON, hit=False)

    def test_card_i_do_not_hold_is_rejected(self) -> None:
        t = self._table()
        t.draw(C.SPY)
        with self.assertRaises(EntryError):
            t.play(C.KING, target=1, king_got=C.BARON)

    def test_overcounted_card_is_rejected(self) -> None:
        """The sixth visible Princess does not exist."""
        t = Table(3, 0, C.PRINCESS, first_player=1)
        t.draw()
        with self.assertRaises(EntryError) as ctx:
            t.play(C.PRINCESS, revealed=C.SPY)  # I hold the only Princess
        self.assertIn("impossible", str(ctx.exception))

    def test_guard_naming_guard_is_rejected(self) -> None:
        t = self._table()
        t.draw(C.SPY)
        with self.assertRaises(EntryError):
            t.play(C.GUARD, target=1, guess=C.GUARD, hit=False)

    def test_protected_target_is_rejected(self) -> None:
        t = Table(3, 1, C.GUARD, first_player=0)
        t.draw()
        t.play(C.HANDMAID)  # P0 protects themselves
        t.draw(C.SPY)
        with self.assertRaises(EntryError):
            t.play(C.GUARD, target=0, guess=C.BARON, hit=False)

    def test_countess_compulsion_is_enforced_for_me(self) -> None:
        t = Table(3, 0, C.COUNTESS, first_player=0)
        t.draw(C.KING)
        with self.assertRaises(EntryError):
            t.play(C.KING, target=1, king_got=C.SPY)
        t.play(C.COUNTESS)  # the compelled play goes through

    def test_rejected_entry_leaves_no_trace(self) -> None:
        """A mid-resolution rejection must restore the pre-entry state."""
        t = self._table()
        t.draw(C.SPY)
        before = (list(t.log.events), list(t.my_hand), t.turn)
        with self.assertRaises(EntryError):
            t.play(C.GUARD, target=1, guess=C.BARON)  # missing hit/miss
        self.assertEqual(
            (list(t.log.events), list(t.my_hand), t.turn), before,
            "a rejected entry mutated the table",
        )


class TestUndo(unittest.TestCase):
    def test_undo_restores_everything(self) -> None:
        t = Table(3, 0, C.GUARD, first_player=0)
        t.draw(C.SPY)
        snap = (
            list(t.log.events), list(t.my_hand), t.current, t.turn,
            list(t.deck_slots),
        )
        t.play(C.GUARD, target=1, guess=C.PRINCESS, hit=False)
        t.undo()
        self.assertEqual(
            (list(t.log.events), list(t.my_hand), t.current, t.turn,
             list(t.deck_slots)),
            snap,
        )

    def test_undo_a_draw(self) -> None:
        t = Table(3, 0, C.GUARD, first_player=0)
        t.draw(C.SPY)
        t.undo()
        self.assertEqual(t.my_hand, [C.GUARD])
        self.assertTrue(t.awaiting_draw)

    def test_undo_chain_replays_correctly(self) -> None:
        """Undo twice, re-enter differently, and the log stays consistent."""
        t = Table(3, 0, C.GUARD, first_player=0)
        t.draw(C.SPY)
        t.play(C.SPY)
        t.undo()
        t.undo()
        t.draw(C.BARON)
        t.play(C.BARON, target=1, baron_loser=1, baron_revealed=C.SPY)
        self.assertTrue(t.seats[1].out)
        self.assertEqual(t.my_hand, [C.GUARD])

    def test_undo_with_nothing_to_undo_is_loud(self) -> None:
        t = Table(3, 0, C.GUARD, first_player=0)
        with self.assertRaises(EntryError):
            t.undo()


class TestRoundEndAtTheTable(unittest.TestCase):
    def test_last_standing_ends_and_scores(self) -> None:
        t = Table(2, 0, C.GUARD, faceup=(C.SPY, C.SPY, C.PRIEST),
                  first_player=0)
        t.draw(C.BARON)
        t.play(C.GUARD, target=1, guess=C.PRINCESS, hit=True)
        self.assertTrue(t.round_over)
        self.assertEqual(t.winners, (0,))
        self.assertEqual(t.seats[0].tokens, 1)

    def test_deck_out_waits_for_revealed_hands(self) -> None:
        """A tiny variant makes the deck-out path short and deterministic."""
        from types import MappingProxyType

        from loveletter.config import GameConfig

        tiny = GameConfig(
            counts=MappingProxyType(
                {C.GUARD: 2, C.PRIEST: 1, C.BARON: 1, C.KING: 1, C.PRINCESS: 1}
            ),
            tokens_to_win=MappingProxyType({2: 3}),
            faceup_at_two_players=0,
            max_players=2,
        )
        t = Table(2, 0, C.GUARD, first_player=0, config=tiny)
        self.assertEqual(len(t.deck_slots), 3)
        t.draw(C.PRIEST)
        t.play(C.PRIEST, target=1, seen=C.BARON)
        t.draw()
        t.play(C.GUARD, target=0, guess=C.PRINCESS, hit=False)
        t.draw(C.KING)
        t.play(C.KING, target=1, king_got=C.BARON)
        self.assertTrue(t.pending_reveal, "deck is out; hands must be revealed")
        with self.assertRaises(EntryError):
            t.end_round({})  # P1's hand is missing
        t.end_round({1: C.GUARD})
        self.assertTrue(t.round_over)
        self.assertEqual(t.winners, (0,))  # my Baron(3) beats their Guard(1)

    def test_spy_bonus_awarded_at_the_table(self) -> None:
        t = Table(2, 0, C.SPY, faceup=(C.PRIEST, C.PRIEST, C.HANDMAID),
                  first_player=0)
        t.draw(C.GUARD)
        t.play(C.SPY)
        t.draw()
        t.play(C.GUARD, target=0, guess=C.PRINCESS, hit=False)
        t.draw(C.BARON)
        t.play(C.BARON, target=1, baron_loser=1, baron_revealed=C.SPY)
        # P1 out; I survive having played a Spy: round token + spy token.
        self.assertTrue(t.round_over)
        self.assertEqual(t.seats[0].tokens, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestTwoStepChancellor(unittest.TestCase):
    """`p chancellor` bare, then `drew`, then `keep ... ret ...`.

    The keep/return decision happens *after* the cards are picked up, so the
    entry has to allow that order -- and it must produce exactly the log the
    one-shot entry produces, or beliefs would depend on how the user typed.
    """

    def _two_step(self) -> Table:
        t = Table(4, 0, C.GUARD, first_player=0)
        t.draw(C.CHANCELLOR)
        t.play(C.CHANCELLOR)
        t.chancellor_drawn([C.KING, C.SPY])
        t.chancellor_resolve(C.KING, [C.GUARD, C.SPY])
        return t

    def _one_shot(self) -> Table:
        t = Table(4, 0, C.GUARD, first_player=0)
        t.draw(C.CHANCELLOR)
        t.play(
            C.CHANCELLOR,
            chancellor_kept=C.KING,
            chancellor_returned=(C.GUARD, C.SPY),
        )
        return t

    def test_both_entry_styles_give_the_same_log(self) -> None:
        a, b = self._two_step(), self._one_shot()
        self.assertEqual(a.log.events, b.log.events)

    def test_both_give_the_same_state(self) -> None:
        a, b = self._two_step(), self._one_shot()
        self.assertEqual(a.my_hand, b.my_hand)
        self.assertEqual(a.deck_slots, b.deck_slots)
        self.assertEqual(a.current, b.current)
        self.assertEqual(a.turn, b.turn)

    def test_turn_stays_open_until_resolved(self) -> None:
        t = Table(4, 0, C.GUARD, first_player=0)
        t.draw(C.CHANCELLOR)
        t.play(C.CHANCELLOR)
        self.assertTrue(t.pending_chancellor)
        self.assertEqual(t.current, 0, "the turn advanced mid-Chancellor")

    def test_other_entries_are_blocked_while_pending(self) -> None:
        """I am holding cards the tool has not been told about."""
        t = Table(4, 0, C.GUARD, first_player=0)
        t.draw(C.CHANCELLOR)
        t.play(C.CHANCELLOR)
        with self.assertRaises(EntryError):
            t.draw()
        with self.assertRaises(EntryError):
            t.play(C.GUARD, target=1, guess=C.SPY, hit=False)

    def test_wrong_draw_count_is_rejected(self) -> None:
        t = Table(4, 0, C.GUARD, first_player=0)
        t.draw(C.CHANCELLOR)
        t.play(C.CHANCELLOR)
        with self.assertRaises(EntryError):
            t.chancellor_drawn([C.KING])

    def test_impossible_draw_pair_is_rejected(self) -> None:
        """Two Princesses do not exist, even though one would be fine."""
        t = Table(4, 0, C.GUARD, first_player=0)
        t.draw(C.CHANCELLOR)
        t.play(C.CHANCELLOR)
        with self.assertRaises(EntryError):
            t.chancellor_drawn([C.PRINCESS, C.PRINCESS])

    def test_resolving_with_a_card_not_held_is_rejected(self) -> None:
        t = Table(4, 0, C.GUARD, first_player=0)
        t.draw(C.CHANCELLOR)
        t.play(C.CHANCELLOR)
        t.chancellor_drawn([C.KING, C.SPY])
        with self.assertRaises(EntryError):
            t.chancellor_resolve(C.COUNTESS, [C.KING, C.SPY])

    def test_resolve_before_draw_is_rejected(self) -> None:
        t = Table(4, 0, C.GUARD, first_player=0)
        t.draw(C.CHANCELLOR)
        t.play(C.CHANCELLOR)
        with self.assertRaises(EntryError):
            t.chancellor_resolve(C.GUARD, [C.KING, C.SPY])

    def test_undo_unwinds_each_step(self) -> None:
        t = Table(4, 0, C.GUARD, first_player=0)
        t.draw(C.CHANCELLOR)
        t.play(C.CHANCELLOR)
        t.chancellor_drawn([C.KING, C.SPY])
        t.undo()
        self.assertTrue(t.pending_chancellor)
        self.assertIsNone(t.pending_drawn, "the draw survived its undo")
        t.chancellor_drawn([C.PRIEST, C.BARON])
        t.chancellor_resolve(C.PRIEST, [C.GUARD, C.BARON])
        self.assertEqual(t.my_hand, [C.PRIEST])

    def test_returned_cards_are_pinned_in_the_posterior(self) -> None:
        """The whole point of the ordered deck: knowing what comes back."""
        t = self._two_step()
        constraints = t.belief().constraints
        bottom = constraints.deck_slots[-2:]
        self.assertEqual(
            [constraints.known_slot.get(s) for s in bottom],
            [C.GUARD, C.SPY],
            "returned cards were not pinned to their slots",
        )
