"""Phase 2a tests: the belief tracker, hard constraints only.

The load-bearing test here is :class:`TestLogReplaySufficiency`: replaying the
projected log from turn zero must reproduce the posterior exactly.  If it does
not, an event happened that was not logged, and no amount of correct
arithmetic downstream will save the numbers.
"""

from __future__ import annotations

import random
import unittest
from collections import Counter, defaultdict

from loveletter.belief import SET_ASIDE, Belief, replay, unknown_pool
from loveletter.config import STANDARD, Card
from loveletter.engine import (
    Action,
    apply,
    legal_actions,
    new_round,
    state_from_hands,
)
from loveletter.events import ChancellorExchange, Drew, EventLog, PriestLook
from loveletter.observation import hidden_identities, observe

C = Card


def dealt_card(state, viewer: int) -> tuple:
    """The card ``viewer`` was dealt at setup.

    Read from the engine's ground truth rather than reconstructed from the
    current hand: cards arrive by King trade and Prince redraw as well as by
    drawing, so "hand + discards - draws" is not the dealt card and feeding
    that to the tracker asserts a hand the player never had.

    A real CLI knows this directly -- the user types the card they were dealt.
    """
    from loveletter.events import Dealt

    for e in state.log.of_type(Dealt):
        if e.actor == viewer and e.card is not None:
            return (e.card,)
    return ()


def belief_for(state, viewer: int, **kw) -> Belief:
    """Build the viewer's posterior from the engine state's projected log."""
    return Belief.from_log(
        observe(state.log, viewer),
        viewer,
        state.n_players,
        faceup=state.faceup,
        **kw,
    )


class TestObservation(unittest.TestCase):
    def test_projection_hides_opponent_draws(self) -> None:
        rng = random.Random(3)
        state = new_round(3, rng)
        for _ in range(4):
            if state.round_over:
                break
            state = apply(state, rng.choice(legal_actions(state)))
        seen = observe(state.log, viewer=0)
        for e in seen.of_type(Drew):
            if e.actor != 0:
                self.assertIsNone(e.card, "opponent draw leaked its identity")

    def test_projection_hides_opponent_priest_looks(self) -> None:
        state = state_from_hands(
            [[C.PRIEST, C.SPY], [C.BARON], [C.HANDMAID]],
            [C.SPY, C.KING],
            C.GUARD,
        )
        after = apply(state, Action(C.PRIEST, target=1))
        # P0 looked, so P0 sees the card and P2 does not.
        self.assertEqual(observe(after.log, 0).of_type(PriestLook)[0].seen, C.BARON)
        self.assertIsNone(observe(after.log, 2).of_type(PriestLook)[0].seen)

    def test_projection_keeps_the_structural_fact(self) -> None:
        """The look is public even when its content is not."""
        state = state_from_hands(
            [[C.PRIEST, C.SPY], [C.BARON], [C.HANDMAID]],
            [C.SPY, C.KING],
            C.GUARD,
        )
        after = apply(state, Action(C.PRIEST, target=1))
        looks = observe(after.log, 2).of_type(PriestLook)
        self.assertEqual(len(looks), 1)
        self.assertEqual((looks[0].actor, looks[0].target), (0, 1))

    def test_projection_actually_hides_something(self) -> None:
        rng = random.Random(11)
        state = new_round(4, rng)
        for _ in range(6):
            if state.round_over:
                break
            state = apply(state, rng.choice(legal_actions(state)))
        self.assertGreater(
            hidden_identities(state.log, viewer=0),
            0,
            "projection hid nothing -- it is not working",
        )


class TestSlotLogging(unittest.TestCase):
    def test_every_draw_names_its_slot(self) -> None:
        rng = random.Random(5)
        state = new_round(4, rng)
        while not state.round_over:
            state = apply(state, rng.choice(legal_actions(state)))
        for e in state.log.of_type(Drew):
            self.assertGreaterEqual(e.slot, 0, "a draw was logged without a slot")

    def test_slot_ids_are_never_reused(self) -> None:
        rng = random.Random(9)
        for seed in range(20):
            rng = random.Random(seed)
            state = new_round(4, rng)
            drawn: list[int] = []
            while not state.round_over:
                state = apply(state, rng.choice(legal_actions(state)))
            drawn = [e.slot for e in state.log.of_type(Drew) if e.slot >= 0]
            self.assertEqual(
                len(drawn), len(set(drawn)), "a slot id was drawn twice"
            )

    def test_chancellor_logs_both_slot_movements(self) -> None:
        state = state_from_hands(
            [[C.CHANCELLOR, C.GUARD], [C.HANDMAID], [C.SPY]],
            [C.KING, C.PRIEST, C.BARON],
            C.SPY,
        )
        after = apply(
            state, Action(C.CHANCELLOR, chancellor_return=(C.GUARD, C.KING))
        )
        ex = after.log.of_type(ChancellorExchange)[-1]
        self.assertEqual(ex.n, 2)
        self.assertEqual(len(ex.drew), 2)
        self.assertEqual(len(ex.returned), 2)
        # Returned slots are fresh ids, not the ones drawn.
        self.assertFalse(set(ex.drew) & set(ex.returned))

    def test_returned_slots_sit_at_the_bottom(self) -> None:
        state = state_from_hands(
            [[C.CHANCELLOR, C.GUARD], [C.HANDMAID], [C.SPY]],
            [C.KING, C.PRIEST, C.BARON],
            C.SPY,
        )
        after = apply(
            state, Action(C.CHANCELLOR, chancellor_return=(C.GUARD, C.KING))
        )
        ex = after.log.of_type(ChancellorExchange)[-1]
        self.assertEqual(after.slots[-2:], ex.returned)


class TestHardConstraints(unittest.TestCase):
    def test_viewer_knows_own_hand(self) -> None:
        state = state_from_hands(
            [[C.PRINCESS, C.GUARD], [C.BARON], [C.SPY]], [C.KING, C.PRIEST], C.SPY
        )
        b = belief_for(state, 0)
        marg = b.hand_marginals()
        self.assertEqual(marg[0].get(C.PRINCESS), 1.0)
        self.assertEqual(marg[0].get(C.GUARD), 1.0)

    def test_public_discards_leave_the_unknown_pool(self) -> None:
        state = state_from_hands(
            [[C.GUARD, C.SPY], [C.BARON], [C.HANDMAID]],
            [C.KING, C.PRIEST],
            C.SPY,
        )
        after = apply(state, Action(C.GUARD, target=1, guess=C.PRINCESS))
        b = belief_for(after, 0)
        # The played Guard is public; one fewer Guard can be hidden anywhere.
        self.assertLessEqual(
            b.unknown_pool[C.GUARD], STANDARD.copies(C.GUARD) - 1
        )

    def test_guard_miss_excludes_that_card_at_that_moment(self) -> None:
        """The announcement is public and pins the hand as it was."""
        state = state_from_hands(
            [[C.GUARD, C.SPY], [C.BARON], [C.HANDMAID]],
            [C.KING, C.PRIEST],
            C.SPY,
        )
        after = apply(state, Action(C.GUARD, target=1, guess=C.PRINCESS))
        c = replay(observe(after.log, 0), 0, after.n_players,
                   viewer_hand=dealt_card(after, 0))
        # P1 drew right after the miss, so the ban has decayed to a ceiling:
        # the card they held is provably not the Princess, the drawn one might
        # be. Exactly one copy is therefore still possible.
        self.assertEqual(c.at_most.get(1, {}).get(C.PRINCESS), 1)

    def test_guard_miss_is_a_hard_ban_before_the_target_draws(self) -> None:
        """Undiluted, a miss rules the card out entirely."""
        state = state_from_hands(
            [[C.GUARD, C.SPY], [C.BARON], [C.HANDMAID], [C.KING]],
            [C.PRIEST, C.CHANCELLOR, C.PRINCE],
            C.SPY,
        )
        # P0 misses on P2, who does not act next -- P1 does.
        after = apply(state, Action(C.GUARD, target=2, guess=C.PRINCESS))
        b = belief_for(after, 0)
        self.assertEqual(b.hand_marginals()[2].get(C.PRINCESS, 0.0), 0.0)

    def test_two_player_faceup_cards_are_known_from_turn_zero(self) -> None:
        rng = random.Random(4)
        state = new_round(2, rng)
        b = belief_for(state, 0)
        # A faceup card cannot also be in the opponent's hand, unless another
        # copy exists. Count-limited cards must show this exactly.
        for card in state.faceup:
            if STANDARD.copies(card) == 1:
                self.assertEqual(b.hand_marginals()[1].get(card, 0.0), 0.0)

    def test_eliminated_player_hand_is_known_and_holds_nothing(self) -> None:
        state = state_from_hands(
            [[C.GUARD, C.SPY], [C.BARON], [C.HANDMAID]],
            [C.KING, C.PRIEST],
            C.SPY,
        )
        after = apply(state, Action(C.GUARD, target=1, guess=C.BARON))
        b = belief_for(after, 0)
        self.assertNotIn(1, {p for p, h in b.hand_marginals().items() if h})

    def test_baron_loss_reveals_loser_and_bounds_winner(self) -> None:
        """The winner's card is unrevealed but known strictly greater."""
        state = state_from_hands(
            [[C.SPY], [C.BARON, C.GUARD], [C.HANDMAID], [C.PRIEST]],
            [C.KING, C.CHANCELLOR, C.PRINCE],
            C.SPY,
            current=1,
        )
        # P1 plays Baron at P2: Baron(3) > Handmaid(4)? No -- P1 loses.
        after = apply(state, Action(C.BARON, target=2))
        b = belief_for(after, 0)
        # P1 played the Baron and compared their *other* card, the Guard, which
        # the Eliminated event reveals. So the bound is against the Guard, not
        # against the Baron -- the played card never enters the comparison.
        self.assertIn((2, C.GUARD), b.constraints.beats_value)
        for w in b.worlds():
            self.assertTrue(
                any(x > C.GUARD for x in w.hands[2]),
                f"world has P2 holding {w.hands[2]}, none beating the Guard",
            )

    def test_baron_loss_is_a_strict_bound_before_the_winner_draws(self) -> None:
        """Undiluted, every card the winner holds must beat the loser's."""
        state = state_from_hands(
            [[C.SPY], [C.BARON, C.GUARD], [C.HANDMAID], [C.PRIEST]],
            [C.KING, C.CHANCELLOR, C.PRINCE],
            C.SPY,
            current=1,
        )
        after = apply(state, Action(C.BARON, target=2))
        c = replay(observe(after.log, 0), 0, after.n_players,
                   viewer_hand=dealt_card(after, 0))
        # P1 was eliminated, so the relation is pinned to their revealed card
        # rather than to a hand that no longer exists.
        self.assertIn((2, C.GUARD), c.beats_value)

    def test_baron_tie_constrains_both_hands_to_equal_values(self) -> None:
        state = state_from_hands(
            [[C.SPY], [C.BARON, C.PRIEST], [C.PRIEST], [C.KING]],
            [C.CHANCELLOR, C.PRINCE, C.HANDMAID],
            C.SPY,
            current=1,
        )
        after = apply(state, Action(C.BARON, target=2))
        b = belief_for(after, 0)
        # Nobody is out, but everyone learned the two hands were equal. P2 has
        # since drawn, so the fact survives in weakened form: some card they
        # hold still equals P1's.
        c = b.constraints
        self.assertEqual(len(c.equal_to) + len(c.weak_equal), 1)
        self.assertIn((1, 2), c.equal_to + c.weak_equal)
        for w in b.worlds():
            self.assertTrue(
                any(x == y for x in w.hands[1] for y in w.hands[2]),
                "a world violates the Baron tie",
            )

    def test_priest_sighting_is_used_by_the_looker_only(self) -> None:
        state = state_from_hands(
            [[C.PRIEST, C.SPY], [C.BARON], [C.HANDMAID], [C.KING]],
            [C.CHANCELLOR, C.PRINCE, C.GUARD],
            C.SPY,
        )
        after = apply(state, Action(C.PRIEST, target=1))
        mine = belief_for(after, 0)
        self.assertEqual(mine.hand_marginals()[1].get(C.BARON), 1.0)
        # P2 saw the look happen but not the card, so P2 stays uncertain.
        theirs = belief_for(after, 2)
        self.assertLess(theirs.hand_marginals()[1].get(C.BARON, 0.0), 1.0)

    def test_priest_sighting_decays_when_the_target_draws(self) -> None:
        """A sighting is time-indexed: the target may have drawn since."""
        state = state_from_hands(
            [[C.PRIEST, C.SPY], [C.BARON], [C.HANDMAID], [C.KING]],
            [C.CHANCELLOR, C.PRINCE, C.GUARD],
            C.SPY,
        )
        after = apply(state, Action(C.PRIEST, target=1))
        self.assertEqual(belief_for(after, 0).hand_marginals()[1].get(C.BARON), 1.0)
        # P1 now takes their turn: they drew, so the sighting no longer pins
        # their whole hand.
        after2 = apply(after, [a for a in legal_actions(after)][0])
        c = replay(
            observe(after2.log, 0), 0, after2.n_players,
            viewer_hand=after2.player(0).hand,
        )
        self.assertNotIn(1, c.known_hand)


class TestJointDistribution(unittest.TestCase):
    def test_hands_are_correlated_not_independent(self) -> None:
        """Only one Princess exists, so two players cannot both hold it."""
        state = state_from_hands(
            [[C.GUARD, C.SPY], [C.BARON], [C.HANDMAID]],
            [C.PRINCESS, C.KING],
            C.PRIEST,
        )
        b = belief_for(state, 0)
        for w in b.worlds():
            holders = sum(1 for h in w.hands if C.PRINCESS in h)
            pinned = sum(1 for _, c in w.slots if c is C.PRINCESS)
            loose = sum(1 for c in w.deck_pool if c is C.PRINCESS)
            aside = 1 if w.set_aside is C.PRINCESS else 0
            self.assertEqual(holders + pinned + loose + aside, 1)

    def test_marginals_are_probabilities(self) -> None:
        rng = random.Random(2)
        state = new_round(3, rng)
        for _ in range(3):
            if state.round_over:
                break
            state = apply(state, rng.choice(legal_actions(state)))
        b = belief_for(state, 0)
        for pid, marg in b.hand_marginals().items():
            for card, p in marg.items():
                self.assertGreaterEqual(p, 0.0)
                self.assertLessEqual(p, 1.0)

    def test_expected_counts_sum_to_hand_size(self) -> None:
        """Expected copies, summed over cards, equal how many they hold.

        Note this is the *count* view, not hand_marginals: a probability of
        holding a card must not be summed to a hand size, since a player
        holding two Guards holds one distinct card value.
        """
        state = state_from_hands(
            [[C.GUARD, C.SPY], [C.BARON], [C.HANDMAID]],
            [C.PRINCESS, C.KING],
            C.PRIEST,
        )
        b = belief_for(state, 0)
        c = b.constraints
        for pid, counts in b.expected_hand_counts().items():
            self.assertAlmostEqual(
                sum(counts.values()),
                c.hand_size(pid),
                places=9,
                msg=f"P{pid} expected counts do not sum to their hand size",
            )

    def test_marginals_never_exceed_one(self) -> None:
        """A marginal is a probability, even when duplicates are possible."""
        state = state_from_hands(
            [[C.SPY], [C.GUARD, C.PRIEST], [C.HANDMAID]],
            [C.PRINCESS, C.KING, C.BARON],
            C.CHANCELLOR,
            current=1,
        )
        b = belief_for(state, 0)
        for pid, marg in b.hand_marginals().items():
            for card, p in marg.items():
                self.assertLessEqual(p, 1.0, f"P{pid} {card} marginal {p} > 1")

    def test_expected_counts_match_the_deck_composition(self) -> None:
        """Hidden expectation plus public count equals the true copy count."""
        state = state_from_hands(
            [[C.GUARD, C.SPY], [C.BARON], [C.HANDMAID]],
            [C.PRINCESS, C.KING],
            C.PRIEST,
        )
        b = belief_for(state, 0)
        public = Counter(b.constraints.public)
        expected = b.expected_counts()
        for card in STANDARD.counts:
            total = expected.get(card, 0.0) + public.get(card, 0)
            self.assertAlmostEqual(
                total, STANDARD.copies(card), places=9, msg=f"{card} miscounted"
            )


class TestBruteForce(unittest.TestCase):
    """Cross-check the tracker against exhaustive enumeration.

    These use a deliberately tiny config so the raw product is enumerable.
    Against the full 21-card deck the unknown pool is ~19 cards and the raw
    product is 10^17 -- which is the whole reason the tracker enumerates hands
    and treats the deck as an exchangeable multiset.
    """

    def _tiny(self):
        """A 6-card variant: small enough to enumerate by brute force."""
        from types import MappingProxyType

        from loveletter.config import GameConfig

        return GameConfig(
            counts=MappingProxyType(
                {C.GUARD: 2, C.PRIEST: 1, C.BARON: 1, C.KING: 1, C.PRINCESS: 1}
            ),
            tokens_to_win=MappingProxyType({2: 3, 3: 3}),
            faceup_at_two_players=0,
            max_players=3,
        )

    def test_matches_exhaustive_enumeration(self) -> None:
        """Every world the tracker finds, and no others, is consistent."""
        from itertools import permutations

        config = self._tiny()
        # P0 (viewer) holds Guard+Priest. P1 and P2 hold one card each; one
        # card sits in the deck and one is set aside.
        state = state_from_hands(
            [[C.GUARD, C.PRIEST], [C.BARON], [C.KING]],
            [C.PRINCESS],
            C.GUARD,
            config=config,
        )
        b = Belief.from_log(
            observe(state.log, 0),
            0,
            3,
            config=config,
            viewer_hand=(C.GUARD, C.PRIEST),
            initial_slots=state.slots,
        )
        pool = list(b.unknown_pool.elements())
        self.assertLessEqual(len(pool), 6, "fixture is too big to brute force")

        # Independently: every arrangement of the pool over (P1, P2, deck,
        # set-aside), filtered to those consistent with what P0 knows.
        want = {(perm[0],) + (perm[1],) for perm in set(permutations(pool))}
        want = {(a, b_) for a, b_ in want}
        got = {(w.hands[1], w.hands[2]) for w in b.worlds()}
        got_pairs = {(h1[0], h2[0]) for h1, h2 in got}
        self.assertEqual(got_pairs, want)

    def test_no_world_double_spends_a_card(self) -> None:
        """Across every world, card totals equal the variant's copy counts."""
        config = self._tiny()
        state = state_from_hands(
            [[C.GUARD, C.PRIEST], [C.BARON], [C.KING]],
            [C.PRINCESS],
            C.GUARD,
            config=config,
        )
        b = Belief.from_log(
            observe(state.log, 0),
            0,
            3,
            config=config,
            viewer_hand=(C.GUARD, C.PRIEST),
            initial_slots=state.slots,
        )
        for w in b.worlds():
            tally = Counter()
            for hand in w.hands:
                tally.update(hand)
            tally.update(card for _, card in w.slots)
            tally.update(w.deck_pool)
            if w.set_aside is not None:
                tally[w.set_aside] += 1
            for card, n in tally.items():
                self.assertLessEqual(
                    n, config.copies(card), f"world over-spends {card}"
                )

    def test_next_draw_is_a_normalised_distribution(self) -> None:
        """The next-draw distribution is a proper distribution over the pool.

        It is deliberately *not* the raw pool proportion: some of the pool is
        in opponents' hands, and conditioning on the hand assignments shifts
        the deck's composition. Asserting the raw proportion here would be
        asserting that hands and deck are independent, which is the exact
        error the joint representation exists to avoid.
        """
        state = state_from_hands(
            [[C.GUARD, C.SPY], [C.BARON], [C.HANDMAID]],
            [C.PRINCESS, C.KING],
            C.PRIEST,
        )
        b = belief_for(state, 0)
        draw = b.next_draw()
        self.assertAlmostEqual(sum(draw.values()), 1.0, places=9)
        for card, p in draw.items():
            self.assertGreater(p, 0.0)
            self.assertIn(card, b.unknown_pool)

    def test_next_draw_matches_the_pool_when_no_hands_are_hidden(self) -> None:
        """With every opponent hand pinned, the deck is the raw pool.

        Heads-up, after a Priest look, the only hidden positions are deck
        slots and the set-aside -- all exchangeable -- so the independence the
        previous test denies is restored.
        """
        state = state_from_hands(
            [[C.PRIEST, C.SPY], [C.BARON]],
            [C.PRINCESS, C.KING, C.GUARD],
            C.COUNTESS,
        )
        after = apply(state, Action(C.PRIEST, target=1))
        b = belief_for(after, 0)
        # P1 was seen holding the Baron and has since drawn, so one card of
        # theirs is known and one is not -- still a hidden hand.
        self.assertIn(1, b.constraints.must_hold)
        draw = b.next_draw()
        self.assertAlmostEqual(sum(draw.values()), 1.0, places=9)


class TestLogReplaySufficiency(unittest.TestCase):
    """The log must be sufficient to reconstruct the posterior from scratch.

    If replaying from turn zero does not reproduce the current belief state,
    something happened that was not logged.
    """

    def _posterior(self, state, viewer: int):
        b = belief_for(state, viewer)
        return (
            {p: dict(m) for p, m in b.hand_marginals().items()},
            {s: dict(m) for s, m in b.slot_marginals().items()},
        )

    def test_replay_reconstructs_the_viewers_own_hand(self) -> None:
        """The strongest form: replay must match ground truth, not itself.

        Comparing a replay against another replay only proves determinism. The
        property that matters is that the log carries enough to rebuild what is
        actually true -- so this checks the reconstructed hand against the
        engine's real one, for every player, after every turn.

        Every event field added during Phase 2a (Dealt, the Chancellor's kept
        card, both sides of a King trade, the Prince's replacement) was added
        because this test failed without it.
        """
        for seed in range(15):
            rng = random.Random(seed)
            for n in (2, 3, 4):
                state = new_round(n, rng)
                while not state.round_over:
                    for viewer in range(state.n_players):
                        if state.player(viewer).out:
                            continue
                        c = replay(
                            observe(state.log, viewer),
                            viewer,
                            state.n_players,
                            faceup=state.faceup,
                        )
                        self.assertEqual(
                            tuple(sorted(c.known_hand.get(viewer, ()))),
                            tuple(sorted(state.player(viewer).hand)),
                            f"seed {seed}: P{viewer}'s hand cannot be rebuilt "
                            f"from the log at turn {state.turn}",
                        )
                    state = apply(state, rng.choice(legal_actions(state)))

    def test_the_truth_is_always_among_the_candidate_worlds(self) -> None:
        """The real world must never be pruned away by a hard constraint.

        Hard constraints are logical certainties, so the actual arrangement of
        cards must satisfy all of them. If the true world is ever missing, the
        tracker has concluded something false -- far worse than being vague.
        """
        for seed in range(10):
            rng = random.Random(seed)
            state = new_round(3, rng)
            while not state.round_over:
                b = belief_for(state, 0)
                truth_hands = tuple(
                    tuple(sorted(p.hand)) for p in state.players
                )
                found = any(
                    tuple(tuple(sorted(h)) for h in w.hands) == truth_hands
                    for w in b.worlds()
                )
                self.assertTrue(
                    found,
                    f"seed {seed} turn {state.turn}: the true hands "
                    f"{truth_hands} were pruned from the posterior",
                )
                state = apply(state, rng.choice(legal_actions(state)))

    def test_replay_is_a_pure_function_of_the_projected_log(self) -> None:
        """Nothing enters the posterior except through the log."""
        rng = random.Random(21)
        state = new_round(4, rng)
        for _ in range(5):
            if state.round_over:
                break
            state = apply(state, rng.choice(legal_actions(state)))

        projected = observe(state.log, 0)
        c1 = replay(projected, 0, state.n_players,
                    viewer_hand=state.player(0).hand)
        # Rebuild from a structurally identical copy of the log.
        copied = EventLog(list(projected.events))
        c2 = replay(copied, 0, state.n_players,
                    viewer_hand=state.player(0).hand)
        self.assertEqual(c1.public, c2.public)
        self.assertEqual(c1.deck_slots, c2.deck_slots)
        self.assertEqual(c1.known_hand, c2.known_hand)
        self.assertEqual(c1.out, c2.out)

    def test_replayed_deck_slots_track_the_engine(self) -> None:
        """The tracker's live slot list must equal the engine's, always."""
        for seed in range(20):
            rng = random.Random(seed)
            state = new_round(3, rng)
            while not state.round_over:
                c = replay(
                    observe(state.log, 0),
                    0,
                    state.n_players,
                    viewer_hand=state.player(0).hand,
                )
                self.assertEqual(
                    c.deck_slots,
                    state.slots,
                    "tracker lost track of the deck; an event is missing",
                )
                state = apply(state, rng.choice(legal_actions(state)))

    def test_chancellor_sequence_predicts_a_later_draw(self) -> None:
        """Cards returned to the bottom must be predicted when drawn back."""
        state = state_from_hands(
            [[C.CHANCELLOR, C.GUARD], [C.HANDMAID], [C.SPY]],
            [C.KING, C.PRIEST, C.BARON],
            C.SPY,
        )
        after = apply(
            state, Action(C.CHANCELLOR, chancellor_return=(C.GUARD, C.KING))
        )
        c = replay(
            observe(after.log, 0),
            0,
            after.n_players,
            viewer_hand=after.player(0).hand,
        )
        # The tracker knows which slots are at the bottom, by id.
        ex = after.log.of_type(ChancellorExchange)[-1]
        self.assertEqual(c.deck_slots[-2:], ex.returned)

    def test_uncertainty_shrinks_as_the_round_progresses(self) -> None:
        """Every public discard must strictly shrink the unknown pool.

        This is the property the tool exists for: the deck becomes knowable as
        cards hit the table. A tracker whose pool never shrinks is not
        tracking anything.
        """
        rng = random.Random(8)
        state = new_round(2, rng)
        sizes = [sum(belief_for(state, 0).unknown_pool.values())]
        while not state.round_over:
            state = apply(state, rng.choice(legal_actions(state)))
            if state.round_over:
                break
            sizes.append(sum(belief_for(state, 0).unknown_pool.values()))
        self.assertGreater(sizes[0], sizes[-1], "the pool never shrank")
        for a, b_ in zip(sizes, sizes[1:]):
            self.assertLessEqual(b_, a, "the pool grew, which is impossible")

    def test_pool_equals_the_hidden_positions(self) -> None:
        """The unknown pool must be exactly the cards in hidden positions.

        Hidden = opponents' hands + the deck + the set-aside. Checking the
        identity rather than a threshold means the test says something true of
        every round, not just long ones -- a round that ends on turn four
        legitimately leaves most of the deck unknown.
        """
        for seed in (8, 13, 21, 34):
            rng = random.Random(seed)
            state = new_round(2, rng)
            last = state
            while not state.round_over:
                last = state
                state = apply(state, rng.choice(legal_actions(state)))
            b = belief_for(last, 0)
            hidden = (
                sum(len(last.player(p).hand) for p in last.active() if p != 0)
                + last.deck_size
                + (1 if last.set_aside is not None else 0)
            )
            self.assertEqual(
                sum(b.unknown_pool.values()),
                hidden,
                f"seed {seed}: pool does not match the hidden positions",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestSampling(unittest.TestCase):
    """When the world set is too large, sampling must stay unbiased."""

    def test_reservoir_sample_is_unbiased(self) -> None:
        """A small cap must not skew the marginals away from the exact ones."""
        state = state_from_hands(
            [[C.GUARD, C.SPY], [C.BARON], [C.HANDMAID], [C.PRIEST]],
            [C.PRINCESS, C.KING, C.CHANCELLOR],
            C.COUNTESS,
        )
        exact = belief_for(state, 0)
        exact_marg = exact.hand_marginals()
        self.assertTrue(exact.exact)

        # Cap far below the true count, averaged over seeds to damp noise.
        totals: dict[Card, float] = defaultdict(float)
        trials = 40
        for seed in range(trials):
            b = Belief.from_log(
                observe(state.log, 0),
                0,
                state.n_players,
                faceup=state.faceup,
                rng=random.Random(seed),
            )
            b.max_worlds = 50
            b._worlds = None
            self.assertFalse(b.exact if b._worlds else False)
            for card, p in b.hand_marginals()[1].items():
                totals[card] += p
        for card, p in exact_marg[1].items():
            self.assertAlmostEqual(
                totals[card] / trials,
                p,
                delta=0.06,
                msg=f"sampled marginal for {card} is biased",
            )

    def test_exact_flag_and_total_are_reported(self) -> None:
        state = state_from_hands(
            [[C.GUARD, C.SPY], [C.BARON], [C.HANDMAID], [C.PRIEST]],
            [C.PRINCESS, C.KING, C.CHANCELLOR],
            C.COUNTESS,
        )
        b = belief_for(state, 0)
        b.max_worlds = 10
        b._worlds = None
        b.worlds()
        self.assertFalse(b.exact)
        self.assertGreater(b.total_worlds, 10)
        self.assertEqual(len(b.worlds()), 10)


class TestNeverContradicts(unittest.TestCase):
    """The posterior must never collapse to zero consistent worlds.

    Hard constraints are logical certainties, so at least one world -- the
    true one -- always satisfies them. Zero worlds means a constraint is
    wrong. Every failure below was a real bug found by the arena, each one a
    time-indexed fact left applying to a hand that had since changed:

    * a Baron relation surviving the *viewer's own* draw;
    * a weak Baron relation surviving a play that may have played the very
      card that was compared;
    * a Baron bound surviving the viewer's own Prince redraw;
    * the same for the viewer's own King trade.

    All four hid behind a try/except in the agent, so a 2500-pair sweep
    silently measured the baseline against itself.
    """

    def test_posterior_is_never_empty(self) -> None:
        # Trimmed to keep the suite fast; the arena exercises this far
        # harder on every run, which is where the bugs were found.
        for n in (2, 3, 4):
            for seed in range(12):
                rng = random.Random(seed)
                state = new_round(n, rng)
                while not state.round_over:
                    for viewer in range(n):
                        if state.player(viewer).out:
                            continue
                        b = belief_for(state, viewer)
                        self.assertGreater(
                            len(b.worlds()),
                            0,
                            f"{n}p seed {seed} turn {state.turn}: viewer "
                            f"{viewer} has no consistent world",
                        )
                    state = apply(state, rng.choice(legal_actions(state)))

    def test_posterior_is_never_empty_with_a_policy(self) -> None:
        from loveletter.policy import HeuristicPolicy

        for seed in range(8):
            rng = random.Random(seed)
            state = new_round(3, rng)
            while not state.round_over:
                b = belief_for(state, 0, policy=HeuristicPolicy())
                self.assertGreater(len(b.worlds()), 0)
                state = apply(state, rng.choice(legal_actions(state)))

    def test_viewer_draw_weakens_baron_relations(self) -> None:
        """A draw changes the hand whoever owns it."""
        state = state_from_hands(
            [[C.PRIEST], [C.BARON, C.PRIEST], [C.HANDMAID]],
            [C.GUARD, C.KING, C.CHANCELLOR],
            C.SPY,
            current=1,
        )
        after = apply(state, Action(C.BARON, target=0))  # tie: Priest vs Priest
        # P0 draws on their turn; the strong equality must not survive it.
        while not after.round_over and after.current != 0:
            after = apply(after, legal_actions(after)[0])
        c = replay(observe(after.log, 0), 0, after.n_players)
        self.assertEqual(
            c.equal_to, [], "a strong Baron tie outlived the viewer's draw"
        )


class TestChancellorSlotPins(unittest.TestCase):
    """Cards returned to the bottom are pinned -- and unpinned when drawn.

    Pinning is the whole payoff of the ordered-deck representation: several
    turns after a Chancellor, the tool still knows what is at the bottom. But
    a pin describes a card *in the deck*. When that slot is drawn the card
    moves to a hand, and a pin left behind makes `unknown_pool` subtract a
    card that is no longer hidden -- which emptied the posterior a few turns
    later, in six different tests at once.
    """

    def test_returned_cards_are_pinned_to_their_slots(self) -> None:
        state = state_from_hands(
            [[C.CHANCELLOR, C.GUARD], [C.HANDMAID], [C.SPY]],
            [C.KING, C.PRIEST, C.BARON],
            C.SPY,
        )
        after = apply(
            state, Action(C.CHANCELLOR, chancellor_return=(C.GUARD, C.KING))
        )
        c = replay(observe(after.log, 0), 0, after.n_players)
        bottom = c.deck_slots[-2:]
        self.assertEqual(
            [c.known_slot.get(s) for s in bottom],
            [C.GUARD, C.KING],
            "the ordered deck forgot what was placed at the bottom",
        )

    def test_a_pin_dies_when_its_slot_is_drawn(self) -> None:
        for seed in range(25):
            rng = random.Random(seed)
            state = new_round(4, rng)
            while not state.round_over:
                for viewer in range(4):
                    if state.player(viewer).out:
                        continue
                    c = replay(
                        observe(state.log, viewer), viewer, 4,
                        faceup=state.faceup,
                    )
                    stale = [s for s in c.known_slot if s not in c.deck_slots]
                    self.assertFalse(
                        stale,
                        f"seed {seed}: pins {stale} outlived their slots",
                    )
                state = apply(state, rng.choice(legal_actions(state)))

    def test_pinning_actually_fires_in_real_games(self) -> None:
        """Guard against the previous test passing because nothing pins."""
        live = 0
        for seed in range(25):
            rng = random.Random(seed)
            state = new_round(4, rng)
            while not state.round_over:
                c = replay(observe(state.log, 0), 0, 4, faceup=state.faceup)
                live += sum(1 for s in c.known_slot if s in c.deck_slots)
                state = apply(state, rng.choice(legal_actions(state)))
        self.assertGreater(live, 0, "no slot was ever pinned -- test is vacuous")

    def test_drawing_a_pinned_card_reveals_the_opponent_holds_it(self) -> None:
        """The payoff: watching a known card into a known hand."""
        # Deck of 1: the Chancellor draws 1 and returns 1 (draw k, return k),
        # so the returned Guard is the only card left for P1 to draw.
        state = state_from_hands(
            [[C.CHANCELLOR, C.GUARD], [C.HANDMAID], [C.SPY]],
            [C.KING],
            C.PRIEST,
        )
        after = apply(
            state, Action(C.CHANCELLOR, chancellor_return=(C.GUARD,))
        )
        # P1's draw consumed that pinned slot, so we know what they hold.
        c = replay(observe(after.log, 0), 0, after.n_players)
        held = tuple(c.must_hold.get(1, ())) + tuple(c.known_hand.get(1, ()))
        self.assertIn(
            C.GUARD, held,
            "drawing a pinned card taught us nothing about the drawer",
        )
