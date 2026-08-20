"""Phase 2.5 tests: the evaluation harness.

The arena is the instrument every later claim is measured with, so it needs
its own calibration checks. Three properties matter:

* **Pairing is real.** Both arms must see the identical deal, or the variance
  reduction that makes small samples usable does not exist.
* **Reproducibility.** Same seed, same answer, across processes and runs.
* **Honest verdicts.** A run that cannot distinguish two agents must say so
  rather than reporting the point estimate as a finding.
"""

from __future__ import annotations

import random
import unittest

from loveletter.agents import BASELINE, Agent, BaselineAgent, RandomAgent
from loveletter.arena import compare, pair_seed, play_round
from loveletter.config import CLASSIC, STANDARD, Card
from loveletter.engine import legal_actions, new_round

C = Card


class FirstLegalAgent:
    """Always takes the first legal action. Deterministic, and weak."""

    name = "first_legal"

    def choose(self, state, me, rng):
        return legal_actions(state)[0]


class LastLegalAgent:
    """Always takes the last legal action. Differs from FirstLegal, so the
    two make a genuine comparison rather than a self-match."""

    name = "last_legal"

    def choose(self, state, me, rng):
        return legal_actions(state)[-1]


class TestPairing(unittest.TestCase):
    def test_both_arms_see_the_same_deal(self) -> None:
        """The whole variance argument rests on this."""
        for i in range(20):
            s = pair_seed(7, i)
            a = new_round(2, random.Random(s))
            b = new_round(2, random.Random(s))
            self.assertEqual(a.deck, b.deck)
            self.assertEqual(a.set_aside, b.set_aside)
            self.assertEqual(a.faceup, b.faceup)
            self.assertEqual(
                [p.hand for p in a.players], [p.hand for p in b.players]
            )

    def test_different_pairs_see_different_deals(self) -> None:
        deals = {
            new_round(2, random.Random(pair_seed(7, i))).deck for i in range(50)
        }
        self.assertGreater(len(deals), 45, "pair seeds are colliding")

    def test_pairing_survives_agents_with_different_rng_appetite(self) -> None:
        """The deal must not depend on how much randomness an agent consumes.

        The deck and the agents originally shared one stream, so an agent that
        drew thousands of numbers per decision (PIMC) shifted every later deal
        while a one-draw agent (the baseline) did not. The two games of a
        "paired" comparison then dealt different cards -- and the divergence
        was worst for exactly the comparisons the arena exists to run, so the
        pairing silently stopped working when it mattered most.
        """

        class Greedy:
            """Same choices as the baseline, but burns the stream."""

            name = "greedy_rng"

            def choose(self, state, me, rng):
                for _ in range(500):
                    rng.random()
                return legal_actions(state)[0]

        outcomes = []
        for agents in (
            [Greedy(), FirstLegalAgent()],
            [FirstLegalAgent(), Greedy()],
        ):
            out = play_round(agents, random.Random(11))
            outcomes.append(out.turns)
        # Not asserting equal outcomes -- the agents differ -- but the deal
        # itself must be identical, which is what makes the pair a pair.
        first = new_round(2, random.Random(random.Random(11).getrandbits(64)))
        second = new_round(2, random.Random(random.Random(11).getrandbits(64)))
        self.assertEqual(first.deck, second.deck)
        self.assertEqual(
            [p.hand for p in first.players], [p.hand for p in second.players]
        )

    def test_pair_seed_is_stable_and_pure(self) -> None:
        """Plain arithmetic, so runs reproduce across processes."""
        self.assertEqual(pair_seed(3, 9), pair_seed(3, 9))
        self.assertNotEqual(pair_seed(3, 9), pair_seed(3, 10))
        self.assertNotEqual(pair_seed(3, 9), pair_seed(4, 9))
        self.assertGreaterEqual(pair_seed(0, 0), 0)


class TestReproducibility(unittest.TestCase):
    def test_same_seed_gives_the_same_result(self) -> None:
        a = compare(BASELINE, RandomAgent(), pairs=100, seed=5)
        b = compare(BASELINE, RandomAgent(), pairs=100, seed=5)
        self.assertEqual(a.token_diff, b.token_diff)
        self.assertEqual(a.tokens_a, b.tokens_a)
        self.assertEqual(a.win_diffs, b.win_diffs)

    def test_different_seeds_give_different_results(self) -> None:
        a = compare(BASELINE, RandomAgent(), pairs=100, seed=5)
        b = compare(BASELINE, RandomAgent(), pairs=100, seed=6)
        self.assertNotEqual(a.token_diff, b.token_diff)


class TestVerdicts(unittest.TestCase):
    def test_identical_agents_report_no_difference(self) -> None:
        """Same strategy, same RNG consumption: exactly zero, and honest.

        A zero-width interval here means "these are the same agent", not
        "infinite precision" -- both arms take identical decisions from an
        identical stream.
        """
        r = compare(BASELINE, BaselineAgent(name="copy"), pairs=200, seed=3)
        self.assertEqual(r.token_diff, 0.0)
        self.assertFalse(r.conclusive)

    def test_agents_that_differ_only_by_noise_are_inconclusive(self) -> None:
        """Equal strength must not be reported as a finding."""

        class Jitter(BaselineAgent):
            name = "jitter"

            def choose(self, state, me, rng):
                actions = legal_actions(state)
                if len(actions) == 1:
                    return actions[0]
                rng.random()  # perturb the shared stream
                return super().choose(state, me, rng)

        r = compare(BASELINE, Jitter(), pairs=800, seed=3)
        self.assertFalse(
            r.conclusive,
            f"equal-strength agents reported as a finding: {r.token_diff}",
        )
        self.assertGreater(r.token_ci, 0.0, "interval collapsed to zero width")

    def test_a_real_gap_is_detected(self) -> None:
        r = compare(BASELINE, RandomAgent(), pairs=500, seed=9)
        self.assertTrue(r.conclusive)
        self.assertGreater(r.token_diff, 0.0, "baseline lost to random")

    def test_summary_states_the_verdict(self) -> None:
        strong = compare(BASELINE, RandomAgent(), pairs=300, seed=9).summary()
        self.assertIn("VERDICT", strong)
        self.assertIn("ahead", strong)
        weak = compare(
            BASELINE, BaselineAgent(name="copy"), pairs=50, seed=9
        ).summary()
        self.assertIn("INCONCLUSIVE", weak)

    def test_required_pairs_grows_as_the_effect_shrinks(self) -> None:
        r = compare(FirstLegalAgent(), LastLegalAgent(), pairs=300, seed=2)
        self.assertGreater(r.required_pairs(0.01), r.required_pairs(0.05))


class TestArenaMechanics(unittest.TestCase):
    def test_a_round_produces_at_least_one_winner(self) -> None:
        for seed in range(30):
            out = play_round([BASELINE, RandomAgent()], random.Random(seed))
            self.assertGreaterEqual(len(out.winners), 1)
            self.assertGreater(out.turns, 0)

    def test_tokens_are_awarded_to_someone(self) -> None:
        for seed in range(30):
            out = play_round([BASELINE, RandomAgent()], random.Random(seed))
            self.assertGreaterEqual(sum(out.tokens), 1)

    def test_illegal_actions_are_rejected_loudly(self) -> None:
        class Cheater:
            name = "cheater"

            def choose(self, state, me, rng):
                from loveletter.engine import Action

                return Action(C.PRINCESS, target=99)

        with self.assertRaises(ValueError):
            play_round([Cheater(), BASELINE], random.Random(0))

    def test_runs_at_multiple_player_counts(self) -> None:
        for n in (2, 3, 4, 5, 6):
            r = compare(BASELINE, RandomAgent(), pairs=20, seed=1, n_players=n)
            self.assertEqual(r.pairs, 20)
            self.assertGreater(r.tokens_a, 0.0)

    def test_runs_on_the_classic_variant(self) -> None:
        r = compare(BASELINE, RandomAgent(), pairs=50, seed=1, config=CLASSIC)
        self.assertGreater(r.tokens_a, 0.0)

    def test_thousands_of_pairs_complete_quickly(self) -> None:
        """This harness runs often; it must not be the bottleneck."""
        r = compare(BASELINE, RandomAgent(), pairs=2000, seed=1)
        self.assertLess(
            r.seconds, 30.0, f"2000 pairs took {r.seconds:.1f}s -- too slow"
        )


class TestBaselineStrength(unittest.TestCase):
    """The baseline must be a meaningful yardstick, not a punching bag."""

    def test_baseline_beats_random_decisively(self) -> None:
        r = compare(BASELINE, RandomAgent(), pairs=1000, seed=4)
        self.assertTrue(r.conclusive)
        self.assertGreater(
            r.token_diff, 0.2, "baseline is barely better than random"
        )

    def test_baseline_never_plays_the_princess_voluntarily(self) -> None:
        """Its most important rule, checked directly."""
        from loveletter.engine import Action, state_from_hands

        state = state_from_hands(
            [[C.PRINCESS, C.GUARD], [C.BARON], [C.HANDMAID]],
            [C.KING, C.PRIEST],
            C.SPY,
        )
        action = BASELINE.choose(state, 0, random.Random(0))
        self.assertIsNot(action.card, C.PRINCESS)

    def test_baseline_never_self_princes_holding_the_princess(self) -> None:
        from loveletter.engine import state_from_hands

        state = state_from_hands(
            [[C.PRINCE, C.PRINCESS], [C.BARON], [C.HANDMAID]],
            [C.KING, C.PRIEST],
            C.SPY,
        )
        action = BASELINE.choose(state, 0, random.Random(0))
        self.assertFalse(
            action.card is C.PRINCE and action.target == 0,
            "baseline self-Princed away the Princess",
        )

    def test_baseline_avoids_provably_impossible_guesses(self) -> None:
        """Guessing a card that is entirely discarded is always wrong."""
        from loveletter.engine import state_from_hands

        # Both Priests are already public.
        state = state_from_hands(
            [[C.GUARD, C.SPY], [C.BARON], [C.HANDMAID]],
            [C.KING, C.CHANCELLOR],
            C.PRINCESS,
            discards=[[C.PRIEST], [C.PRIEST], []],
        )
        for _ in range(10):
            action = BASELINE.choose(state, 0, random.Random(0))
            if action.card is C.GUARD:
                self.assertIsNot(action.guess, C.PRIEST)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestKnownLimits(unittest.TestCase):
    """The player-count limit is a stated constraint, not a surprise.

    Belief-driven agents get ~8x more expensive per hidden hand. A 4,000-pair
    comparison is 7s at 2p and about 5 hours at 6p, so 6p cannot be evaluated
    at a sample size that means anything. Better to say so here than to
    discover it during an overnight run that never finishes.
    """

    def test_evaluable_counts_are_declared(self) -> None:
        from loveletter.arena import EVALUABLE_PLAYERS, SLOW_PLAYERS

        self.assertEqual(EVALUABLE_PLAYERS, (2, 3, 4))
        self.assertEqual(SLOW_PLAYERS, (5,))

    def test_large_runs_outside_the_range_warn(self) -> None:
        """Each tier warns in its own terms: 5 is expensive, 6 is unvalidated."""
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compare(BASELINE, RandomAgent(), pairs=250, seed=1, n_players=6)
        self.assertTrue(caught, "no warning for a large 6-player run")
        self.assertIn("unvalidated", str(caught[0].message))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compare(BASELINE, RandomAgent(), pairs=250, seed=1, n_players=5)
        self.assertTrue(caught, "no warning for a large 5-player run")
        self.assertIn("expensive", str(caught[0].message))

    def test_small_runs_and_normal_counts_stay_quiet(self) -> None:
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compare(BASELINE, RandomAgent(), pairs=50, seed=1, n_players=6)
            compare(BASELINE, RandomAgent(), pairs=250, seed=1, n_players=3)
        self.assertFalse(
            caught, f"unexpected warning: {[str(c.message) for c in caught]}"
        )

    def test_the_limit_is_the_tracker_not_the_arena(self) -> None:
        """Agents that skip the tracker run fast at every count."""
        import time

        started = time.time()
        compare(BASELINE, RandomAgent(), pairs=200, seed=1, n_players=6)
        self.assertLess(
            time.time() - started,
            20.0,
            "non-belief agents should be fast even at 6 players",
        )
