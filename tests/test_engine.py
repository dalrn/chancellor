"""Phase 1 tests: the rules engine.

Every case required by the brief, plus the three invariants agreed on:
deck size unchanged by Chancellor, hand size 1 at end of turn on every
branch, and deck size >= 1 at the start of every turn.
"""

from __future__ import annotations

import random
import unittest

from loveletter.config import CLASSIC, STANDARD, Card
from loveletter.engine import (
    Action,
    RulesError,
    apply,
    is_round_over,
    legal_actions,
    new_round,
    round_winners,
    spy_bonus_winner,
    state_from_hands,
)
from loveletter.events import ChancellorExchange, GuardResult, PriestLook

C = Card


class TestConfig(unittest.TestCase):
    def test_standard_is_21_cards(self) -> None:
        self.assertEqual(STANDARD.deck_size, 21)
        self.assertEqual(len(STANDARD.all_cards()), 21)

    def test_classic_is_16_cards_and_is_a_parameter_change(self) -> None:
        self.assertEqual(CLASSIC.deck_size, 16)
        self.assertEqual(CLASSIC.copies(C.CHANCELLOR), 0)
        self.assertEqual(CLASSIC.copies(C.SPY), 0)
        self.assertEqual(CLASSIC.copies(C.GUARD), 5)

    def test_setup_deck_sizes(self) -> None:
        self.assertEqual(STANDARD.undealt_deck_size(2), 15)  # 21-1-3-2
        self.assertEqual(STANDARD.undealt_deck_size(4), 16)  # 21-1-4

    def test_two_player_setup_has_three_faceup(self) -> None:
        state = new_round(2, random.Random(1))
        self.assertEqual(len(state.faceup), 3)
        # 15 undealt, less the first player's opening draw.
        self.assertEqual(state.deck_size, 14)

    def test_multiplayer_setup_has_no_faceup(self) -> None:
        state = new_round(4, random.Random(1))
        self.assertEqual(state.faceup, ())
        # 16 undealt, less the first player's opening draw.
        self.assertEqual(state.deck_size, 15)


class TestConservation(unittest.TestCase):
    def test_conservation_holds_through_random_games(self) -> None:
        """Card conservation after every apply, across many seeded games."""
        for seed in range(60):
            rng = random.Random(seed)
            for n in (2, 3, 4, 5, 6):
                state = new_round(n, rng)
                guard = 0
                while not state.round_over:
                    actions = legal_actions(state)
                    state = apply(state, rng.choice(actions))
                    state.check_conservation()
                    guard += 1
                    self.assertLess(guard, 100, "round did not terminate")
                state.check_conservation()

    def test_hand_size_one_at_end_of_every_turn(self) -> None:
        for seed in range(40):
            rng = random.Random(seed)
            state = new_round(4, rng)
            while not state.round_over:
                state = apply(state, rng.choice(legal_actions(state)))
                state.check_hand_sizes()

    def test_deck_at_least_one_at_start_of_every_turn(self) -> None:
        for seed in range(40):
            rng = random.Random(seed)
            state = new_round(3, rng)
            while not state.round_over:
                # Every state handed to apply() has the actor holding 2 cards,
                # which means the deck had >= 1 card at the start of the turn.
                self.assertEqual(len(state.player(state.current).hand), 2)
                state = apply(state, rng.choice(legal_actions(state)))


class TestCountess(unittest.TestCase):
    def test_compelled_with_king(self) -> None:
        state = state_from_hands(
            [[C.COUNTESS, C.KING], [C.GUARD]], [C.SPY, C.SPY], C.BARON
        )
        self.assertEqual(legal_actions(state), [Action(C.COUNTESS)])

    def test_compelled_with_prince(self) -> None:
        state = state_from_hands(
            [[C.COUNTESS, C.PRINCE], [C.GUARD]], [C.SPY, C.SPY], C.BARON
        )
        self.assertEqual(legal_actions(state), [Action(C.COUNTESS)])

    def test_legal_bluff_without_king_or_prince(self) -> None:
        """Countess with neither forcer: playable, but not compelled."""
        state = state_from_hands(
            [[C.COUNTESS, C.GUARD], [C.PRIEST]], [C.SPY, C.SPY], C.BARON
        )
        cards = {a.card for a in legal_actions(state)}
        self.assertEqual(cards, {C.COUNTESS, C.GUARD})

    def test_not_compelled_during_chancellor_draw(self) -> None:
        """Drawing a King via Chancellor does not force the Countess."""
        state = state_from_hands(
            [[C.CHANCELLOR, C.COUNTESS], [C.GUARD]], [C.KING, C.PRINCE], C.BARON
        )
        actions = legal_actions(state)
        self.assertTrue(any(a.card is C.CHANCELLOR for a in actions))
        chancellor = [a for a in actions if a.card is C.CHANCELLOR][0]
        after = apply(state, chancellor)
        # The Chancellor resolved; no compulsion was raised mid-draw.
        self.assertEqual(len(after.player(0).hand), 1)


class TestPrincess(unittest.TestCase):
    def test_playing_princess_eliminates(self) -> None:
        state = state_from_hands(
            [[C.PRINCESS, C.GUARD], [C.BARON]], [C.SPY, C.SPY], C.KING
        )
        after = apply(state, Action(C.PRINCESS))
        self.assertTrue(after.player(0).out)
        self.assertIn(C.PRINCESS, after.player(0).discards)

    def test_prince_forced_princess_discard_eliminates_and_draws_nothing(
        self,
    ) -> None:
        state = state_from_hands(
            [[C.PRINCE, C.GUARD], [C.PRINCESS]], [C.SPY, C.BARON], C.KING
        )
        before_deck = state.deck_size
        after = apply(state, Action(C.PRINCE, target=1))
        self.assertTrue(after.player(1).out)
        self.assertEqual(after.player(1).hand, ())
        self.assertIn(C.PRINCESS, after.player(1).discards)
        # The Princess discard drew nothing. P1 is out and only P0 remains,
        # so the round ends at once and no further draw happens.
        self.assertTrue(after.round_over)
        self.assertEqual(after.winners, (0,))
        self.assertEqual(after.deck_size, before_deck)

    def test_self_prince_princess_ends_round_with_one_survivor(self) -> None:
        """Heads-up, opponent has Handmaid, I hold Prince + Princess.

        Compelled to self-target, I discard the Princess and am out. One
        player remains, so the count reaches one and never zero.
        """
        state = state_from_hands(
            [[C.PRINCE, C.PRINCESS], [C.HANDMAID]],
            [C.SPY, C.GUARD],
            C.KING,
            protected=[False, True],
        )
        actions = legal_actions(state)
        prince = [a for a in actions if a.card is C.PRINCE]
        # The compulsion constrains the Prince's target, not which card I play:
        # playing the Princess outright is legal, and equally fatal.
        self.assertEqual(prince, [Action(C.PRINCE, target=0)])
        after = apply(state, prince[0])
        self.assertTrue(after.player(0).out)
        self.assertTrue(after.round_over)
        self.assertEqual(after.winners, (1,))


class TestPrince(unittest.TestCase):
    def test_prince_on_empty_deck_draws_set_aside(self) -> None:
        state = state_from_hands(
            [[C.PRINCE, C.GUARD], [C.BARON], [C.SPY]], [], C.KING
        )
        after = apply(state, Action(C.PRINCE, target=1))
        self.assertEqual(after.player(1).hand, (C.KING,))
        self.assertIsNone(after.set_aside)
        self.assertIn(C.BARON, after.player(1).discards)

    def test_prince_may_target_self(self) -> None:
        state = state_from_hands(
            [[C.PRINCE, C.GUARD], [C.BARON]], [C.SPY, C.SPY], C.KING
        )
        targets = {a.target for a in legal_actions(state) if a.card is C.PRINCE}
        self.assertEqual(targets, {0, 1})


class TestChancellor(unittest.TestCase):
    def test_draws_two_and_returns_two(self) -> None:
        state = state_from_hands(
            [[C.CHANCELLOR, C.GUARD], [C.BARON]], [C.KING, C.SPY, C.PRIEST], C.SPY
        )
        before = state.deck_size
        after = apply(
            state,
            Action(C.CHANCELLOR, chancellor_return=(C.GUARD, C.KING)),
        )
        self.assertEqual(after.player(0).hand, (C.SPY,))
        self.assertEqual(after.deck_size, before - 1)  # -1 for next player draw
        self.assertEqual(after.log.of_type(ChancellorExchange)[-1].n, 2)

    def test_deck_size_unchanged_by_chancellor(self) -> None:
        """Draw k, return k. The Chancellor permutes; it never shortens."""
        for deck in ([C.KING, C.SPY, C.PRIEST], [C.KING], []):
            with self.subTest(deck=deck):
                state = state_from_hands(
                    [[C.CHANCELLOR, C.GUARD], [C.BARON]], deck, C.SPY
                )
                before = state.deck_size
                action = [
                    a for a in legal_actions(state) if a.card is C.CHANCELLOR
                ][0]
                after = apply(state, action)
                # Chancellor itself is size-neutral; the only change is the
                # next player's draw, which does not happen if the round ended.
                drew_next = 0 if after.round_over else 1
                self.assertEqual(after.deck_size, before - drew_next)

    def test_one_card_in_deck_draws_one_returns_one(self) -> None:
        """Deck 2 at turn start -> draw 1 -> deck 1 -> Chancellor draws 1."""
        state = state_from_hands(
            [[C.CHANCELLOR, C.GUARD], [C.BARON]], [C.KING], C.SPY
        )
        actions = [a for a in legal_actions(state) if a.card is C.CHANCELLOR]
        for a in actions:
            self.assertIsNotNone(a.chancellor_return)
            self.assertEqual(len(a.chancellor_return or ()), 1)
        after = apply(state, Action(C.CHANCELLOR, chancellor_return=(C.GUARD,)))
        self.assertEqual(after.player(0).hand, (C.KING,))

    def test_empty_deck_chancellor_has_no_effect(self) -> None:
        """Deck 1 at turn start -> draw 1 -> deck 0 -> no effect."""
        state = state_from_hands(
            [[C.CHANCELLOR, C.GUARD], [C.BARON]], [], C.SPY
        )
        actions = [a for a in legal_actions(state) if a.card is C.CHANCELLOR]
        self.assertEqual(actions, [Action(C.CHANCELLOR)])
        after = apply(state, actions[0])
        self.assertEqual(after.player(0).hand, (C.GUARD,))
        self.assertEqual(after.log.of_type(ChancellorExchange)[-1].n, 0)

    def test_returns_land_at_bottom_in_order_and_are_drawn_back(self) -> None:
        """First returned card sits higher and is drawn back first."""
        state = state_from_hands(
            [[C.CHANCELLOR, C.GUARD], [C.HANDMAID], [C.SPY]],
            [C.KING, C.PRIEST, C.BARON],
            C.SPY,
        )
        after = apply(
            state, Action(C.CHANCELLOR, chancellor_return=(C.GUARD, C.KING))
        )
        # Deck was [King, Priest, Baron]; drew King+Priest, kept Priest?
        # Returned Guard then King to the bottom, then P1 drew the top.
        self.assertEqual(after.deck[-2:], (C.GUARD, C.KING))

    def test_emptying_then_refilling_does_not_end_round(self) -> None:
        """Chancellor empties the deck mid-turn, then refills it."""
        state = state_from_hands(
            [[C.CHANCELLOR, C.GUARD], [C.HANDMAID], [C.SPY]],
            [C.KING, C.PRIEST],
            C.BARON,
        )
        after = apply(
            state, Action(C.CHANCELLOR, chancellor_return=(C.GUARD, C.KING))
        )
        self.assertFalse(after.round_over, "round ended on a mid-turn empty deck")
        self.assertEqual(len(after.active()), 3)


class TestAllOthersProtected(unittest.TestCase):
    def _state(self, mine: list[Card]):
        return state_from_hands(
            [mine, [C.GUARD], [C.BARON]],
            [C.SPY, C.SPY],
            C.KING,
            protected=[False, True, True],
        )

    def test_guard_priest_baron_king_fizzle(self) -> None:
        for card in (C.GUARD, C.PRIEST, C.BARON, C.KING):
            with self.subTest(card=card):
                state = self._state([card, C.SPY])
                actions = [a for a in legal_actions(state) if a.card is card]
                self.assertEqual(actions, [Action(card)])
                after = apply(state, actions[0])
                self.assertEqual(len(after.active()), 3)  # nobody knocked out

    def test_prince_must_self_target(self) -> None:
        state = self._state([C.PRINCE, C.SPY])
        actions = [a for a in legal_actions(state) if a.card is C.PRINCE]
        self.assertEqual(actions, [Action(C.PRINCE, target=0)])


class TestGuardAndBaron(unittest.TestCase):
    def test_guard_cannot_name_guard(self) -> None:
        state = state_from_hands(
            [[C.GUARD, C.SPY], [C.BARON]], [C.SPY, C.KING], C.PRIEST
        )
        guesses = {a.guess for a in legal_actions(state) if a.card is C.GUARD}
        self.assertNotIn(C.GUARD, guesses)

    def test_guard_hit_eliminates_and_reveals(self) -> None:
        state = state_from_hands(
            [[C.GUARD, C.SPY], [C.BARON], [C.HANDMAID]],
            [C.SPY, C.KING],
            C.PRIEST,
        )
        after = apply(state, Action(C.GUARD, target=1, guess=C.BARON))
        self.assertTrue(after.player(1).out)
        self.assertIn(C.BARON, after.player(1).discards)
        self.assertTrue(after.log.of_type(GuardResult)[-1].hit)

    def test_guard_miss_is_publicly_announced(self) -> None:
        state = state_from_hands(
            [[C.GUARD, C.SPY], [C.BARON], [C.HANDMAID]],
            [C.SPY, C.KING],
            C.PRIEST,
        )
        after = apply(state, Action(C.GUARD, target=1, guess=C.KING))
        result = after.log.of_type(GuardResult)[-1]
        self.assertFalse(result.hit)
        self.assertEqual(result.guess, C.KING)
        self.assertFalse(after.player(1).out)

    def test_baron_lower_is_out(self) -> None:
        state = state_from_hands(
            [[C.BARON, C.KING], [C.GUARD], [C.SPY]], [C.SPY, C.PRIEST], C.HANDMAID
        )
        after = apply(state, Action(C.BARON, target=1))
        self.assertTrue(after.player(1).out)
        self.assertIn(C.GUARD, after.player(1).discards)

    def test_baron_tie_with_legal_duplicates(self) -> None:
        state = state_from_hands(
            [[C.BARON, C.PRIEST], [C.PRIEST], [C.SPY]],
            [C.SPY, C.GUARD],
            C.HANDMAID,
        )
        after = apply(state, Action(C.BARON, target=1))
        self.assertEqual(len(after.active()), 3)

    def test_baron_actor_can_lose(self) -> None:
        state = state_from_hands(
            [[C.BARON, C.SPY], [C.KING], [C.HANDMAID]],
            [C.SPY, C.GUARD],
            C.PRIEST,
        )
        after = apply(state, Action(C.BARON, target=1))
        self.assertTrue(after.player(0).out)


class TestPriestAndKing(unittest.TestCase):
    def test_priest_look_is_logged_with_seen_card(self) -> None:
        state = state_from_hands(
            [[C.PRIEST, C.SPY], [C.BARON], [C.HANDMAID]],
            [C.SPY, C.KING],
            C.GUARD,
        )
        after = apply(state, Action(C.PRIEST, target=1))
        look = after.log.of_type(PriestLook)[-1]
        self.assertEqual((look.actor, look.target, look.seen), (0, 1, C.BARON))

    def test_king_swaps_hands(self) -> None:
        state = state_from_hands(
            [[C.KING, C.SPY], [C.PRINCESS], [C.HANDMAID]],
            [C.SPY, C.GUARD],
            C.BARON,
        )
        after = apply(state, Action(C.KING, target=1))
        self.assertEqual(after.player(0).hand, (C.PRINCESS,))
        # P1 received the Spy, then drew for their own turn.
        self.assertEqual(after.current, 1)
        self.assertIn(C.SPY, after.player(1).hand)
        self.assertEqual(len(after.player(1).hand), 2)


class TestHandmaid(unittest.TestCase):
    def test_protection_lapses_at_start_of_own_turn(self) -> None:
        state = state_from_hands(
            [[C.HANDMAID, C.SPY], [C.GUARD]], [C.SPY, C.PRIEST, C.BARON], C.KING
        )
        after = apply(state, Action(C.HANDMAID))
        self.assertTrue(after.player(0).protected)
        # P1 acts, then P0's turn begins and the Handmaid lapses.
        after = apply(after, legal_actions(after)[0])
        self.assertEqual(after.current, 0)
        self.assertFalse(after.player(0).protected)


class TestRoundEnd(unittest.TestCase):
    def test_deck_out_highest_hand_wins(self) -> None:
        state = state_from_hands(
            [[C.SPY, C.GUARD], [C.KING], [C.BARON]], [], C.PRIEST
        )
        after = apply(state, Action(C.GUARD, target=2, guess=C.HANDMAID))
        self.assertTrue(after.round_over)
        self.assertEqual(after.winners, (1,))

    def test_deck_out_tie_awards_every_tied_player(self) -> None:
        state = state_from_hands(
            [[C.SPY, C.GUARD], [C.PRIEST], [C.PRIEST]], [], C.KING
        )
        after = apply(state, Action(C.GUARD, target=1, guess=C.HANDMAID))
        self.assertEqual(set(after.winners), {1, 2})
        self.assertEqual(after.player(1).tokens, 1)
        self.assertEqual(after.player(2).tokens, 1)

    def test_last_standing_wins_immediately(self) -> None:
        state = state_from_hands(
            [[C.GUARD, C.SPY], [C.BARON]], [C.SPY, C.KING, C.PRIEST], C.HANDMAID
        )
        after = apply(state, Action(C.GUARD, target=1, guess=C.BARON))
        self.assertTrue(after.round_over)
        self.assertEqual(after.winners, (0,))
        self.assertGreater(after.deck_size, 0, "ended on survivors, not deck-out")

    def test_zero_survivors_is_unreachable(self) -> None:
        state = state_from_hands([[C.SPY], [C.GUARD]], [C.KING], C.BARON,
                                 out=[True, True])
        with self.assertRaises(RulesError):
            round_winners(state)


class TestSpyBonus(unittest.TestCase):
    def _ended(self, discards, out=None):
        state = state_from_hands(
            [[C.SPY, C.GUARD], [C.PRIEST], [C.BARON]],
            [],
            C.KING,
            discards=discards,
            out=out,
        )
        return state

    def test_no_spies_no_bonus(self) -> None:
        state = self._ended([[], [], []])
        self.assertIsNone(spy_bonus_winner(state))

    def test_single_surviving_spy_gets_bonus(self) -> None:
        state = self._ended([[C.SPY], [], []])
        self.assertEqual(spy_bonus_winner(state), 0)

    def test_spy_from_eliminated_player_does_not_block(self) -> None:
        """A Spy played by someone later knocked out does not count."""
        state = state_from_hands(
            [[C.SPY, C.GUARD], [C.PRIEST], [C.BARON]],
            [],
            C.KING,
            discards=[[C.SPY], [C.SPY], []],
            out=[False, True, False],
        )
        self.assertEqual(spy_bonus_winner(state), 0)

    def test_two_surviving_spy_players_no_bonus(self) -> None:
        state = self._ended([[C.SPY], [C.SPY], []])
        self.assertIsNone(spy_bonus_winner(state))

    def test_both_spies_by_one_player_is_one_token(self) -> None:
        state = state_from_hands(
            [[C.SPY, C.GUARD], [C.PRIEST], [C.BARON]],
            [],
            C.KING,
            discards=[[C.SPY, C.SPY], [], []],
        )
        self.assertEqual(spy_bonus_winner(state), 0)
        after = apply(state, Action(C.GUARD, target=1, guess=C.HANDMAID))
        # P2 wins the round (Baron 3 > Priest 2); P0 gets exactly 1 spy token.
        self.assertEqual(after.player(0).tokens, 1)

    def test_winner_also_takes_spy_bonus(self) -> None:
        """The round winner still gains their token, even if that is also me."""
        state = state_from_hands(
            [[C.SPY, C.GUARD], [C.PRIEST], [C.BARON]],
            [],
            C.KING,
            discards=[[C.SPY], [], []],
        )
        after = apply(state, Action(C.GUARD, target=2, guess=C.BARON))
        # P2 eliminated -> P0 (Spy 0) vs P1 (Priest 2): P1 wins the round.
        self.assertEqual(after.winners, (1,))
        self.assertEqual(after.player(0).tokens, 1)  # spy bonus only
        self.assertEqual(after.player(1).tokens, 1)  # round win only


class TestClassicVariant(unittest.TestCase):
    def test_classic_round_plays_out(self) -> None:
        rng = random.Random(5)
        for n in (2, 3, 4):
            state = new_round(n, rng, config=CLASSIC)
            while not state.round_over:
                state = apply(state, rng.choice(legal_actions(state)))
                state.check_conservation()
            self.assertTrue(state.round_over)
            self.assertGreaterEqual(len(state.winners), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
