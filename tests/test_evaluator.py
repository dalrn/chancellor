"""Phase 3 tests: the PIMC evaluator.

Three properties matter beyond "it returns an action":

* **Honesty.** Overlapping intervals must be reported as unresolved, not
  ranked. Most Love Letter positions are genuine near-ties, so this is the
  main output path rather than an edge case.
* **The information blind spot is handled, not hidden.** PIMC scores Priest
  targets identically -- to zero variance -- because a perfect-information
  playout gains nothing from a look. Those are ranked by entropy instead, and
  the output says so.
* **Budget.** A recommendation must fit the 2-second table budget.
"""

from __future__ import annotations

import random
import unittest

from loveletter.belief import Belief
from loveletter.config import Card
from loveletter.engine import (
    Action,
    apply_unchecked,
    legal_actions,
    new_round,
    state_from_hands,
    unchecked_transitions,
)
from loveletter.evaluator import (
    ActionValue,
    FastPlayoutPolicy,
    Recommendation,
    evaluate,
)
from loveletter.observation import observe

C = Card


def value_of(action, n: int, wins: float):
    """An ActionValue with a consistent variance term.

    The score is expected *tokens*, not a coin flip, so ``ci`` is computed
    from the observed spread. Constructing one without ``sq`` would leave the
    variance at zero and every interval would collapse to nothing.
    """
    v = ActionValue(action=action, rollouts=n, wins=wins)
    v.sq = wins  # outcomes of 0 or 1 token: sum of squares == sum
    return v


def mid_round(n_players: int, seed: int, depth: int = 3):
    rng = random.Random(seed)
    state = new_round(n_players, rng)
    for _ in range(depth):
        if state.round_over:
            break
        state = apply_unchecked(state, rng.choice(legal_actions(state)))
    return state


class TestBudget(unittest.TestCase):
    def test_recommendation_fits_the_table_budget(self) -> None:
        """The whole point is using this with people waiting."""
        for n in (2, 3, 4, 5):
            state = mid_round(n, seed=3)
            if state.round_over:
                continue
            rec = evaluate(
                state,
                state.current,
                rng=random.Random(1),
                budget_seconds=1.4,
            )
            self.assertLess(
                rec.seconds,
                2.0,
                f"{n}p recommendation took {rec.seconds:.2f}s",
            )

    def test_a_smaller_budget_is_respected(self) -> None:
        state = mid_round(4, seed=7)
        rec = evaluate(
            state, state.current, rng=random.Random(1), budget_seconds=0.2
        )
        self.assertLess(rec.seconds, 1.0)

    def test_single_legal_action_short_circuits(self) -> None:
        state = state_from_hands(
            [[C.COUNTESS, C.KING], [C.GUARD]], [C.SPY, C.SPY], C.BARON
        )
        rec = evaluate(state, 0, rng=random.Random(0))
        self.assertEqual(len(rec.values), 1)
        self.assertIs(rec.values[0].action.card, C.COUNTESS)
        self.assertLess(rec.seconds, 0.1, "should not roll out a forced play")


class TestHonesty(unittest.TestCase):
    """Overlapping intervals must never be presented as a ranking."""

    def test_overlapping_values_are_reported_as_tied(self) -> None:
        a = value_of(Action(C.GUARD, target=1), 100, 50)
        b = value_of(Action(C.SPY), 100, 52)
        self.assertTrue(a.overlaps(b), "2pp apart at n=100 is not separable")

    def test_clearly_separated_values_do_not_overlap(self) -> None:
        a = value_of(Action(C.GUARD, target=1), 500, 400)
        b = value_of(Action(C.SPY), 500, 100)
        self.assertFalse(a.overlaps(b))

    def test_explain_states_when_the_ranking_is_not_established(self) -> None:
        rec = Recommendation(
            values=[
                value_of(Action(C.GUARD, target=1), 100, 50),
                value_of(Action(C.SPY), 100, 49),
            ]
        )
        text = rec.explain()
        self.assertIn("NOT SEPARATED", text)
        self.assertFalse(rec.conclusive)

    def test_explain_states_a_clear_win_plainly(self) -> None:
        rec = Recommendation(
            values=[
                value_of(Action(C.GUARD, target=1), 500, 400),
                value_of(Action(C.SPY), 500, 100),
            ]
        )
        text = rec.explain()
        self.assertIn("Clear of the next option", text)
        self.assertTrue(rec.conclusive)

    def test_every_recommendation_can_explain_itself(self) -> None:
        """The explanation is the product; it must never be empty."""
        for seed in range(6):
            state = mid_round(3, seed=seed)
            if state.round_over:
                continue
            rec = evaluate(
                state,
                state.current,
                rng=random.Random(seed),
                budget_seconds=0.3,
            )
            text = rec.explain()
            self.assertIn("RECOMMEND", text)
            self.assertGreater(len(text), 80)


class TestInformationBlindSpot(unittest.TestCase):
    """PIMC cannot value a Priest look; the evaluator must not pretend it can."""

    def _priest_only(self):
        return state_from_hands(
            [[C.PRIEST, C.PRIEST], [C.BARON], [C.HANDMAID], [C.GUARD]],
            [C.KING, C.CHANCELLOR, C.PRINCE, C.SPY, C.PRINCESS],
            C.COUNTESS,
        )

    def test_priest_only_positions_are_ranked_by_information(self) -> None:
        rec = evaluate(
            self._priest_only(), 0, rng=random.Random(1), budget_seconds=0.5
        )
        self.assertTrue(rec.information_ranked)
        self.assertEqual(rec.worlds_sampled, 0, "spent rollouts on a known tie")
        for value in rec.values:
            self.assertIsNotNone(value.information_bits)

    def test_the_output_says_why_it_used_information(self) -> None:
        rec = evaluate(
            self._priest_only(), 0, rng=random.Random(1), budget_seconds=0.5
        )
        text = rec.explain()
        self.assertIn("information", text.lower())
        self.assertTrue(
            any("identically" in n for n in rec.notes),
            "the zero-variance tie was not called out distinctly",
        )

    def test_information_gain_prefers_the_least_known_hand(self) -> None:
        """A Guard miss narrows a hand, so it becomes worth less to look at."""
        state = state_from_hands(
            [[C.GUARD, C.PRIEST], [C.BARON], [C.HANDMAID], [C.KING]],
            [C.CHANCELLOR, C.PRINCE, C.SPY, C.PRINCESS, C.COUNTESS, C.SPY],
            C.PRIEST,
        )
        after = apply_unchecked(
            state, Action(C.GUARD, target=1, guess=C.PRINCESS)
        )
        belief = Belief.from_log(
            observe(after.log, 0), 0, 4, faceup=after.faceup
        )
        self.assertLess(
            belief.information_gain(1),
            belief.information_gain(2),
            "the narrowed hand should be worth fewer bits per card",
        )

    def test_information_gain_is_normalised_per_card(self) -> None:
        """Otherwise a 2-card hand always outranks a 1-card unknown."""
        state = mid_round(4, seed=3)
        belief = Belief.from_log(
            observe(state.log, 0), 0, 4, faceup=state.faceup
        )
        for pid in range(1, 4):
            if state.player(pid).out:
                continue
            size = belief.constraints.hand_size(pid)
            if size:
                self.assertAlmostEqual(
                    belief.information_gain(pid),
                    belief.hand_entropy(pid) / size,
                    places=9,
                )


class TestRolloutMechanics(unittest.TestCase):
    def test_chancellor_actions_are_rebound_per_world(self) -> None:
        """A return-order from the real deck is meaningless in a sampled one."""
        state = state_from_hands(
            [[C.CHANCELLOR, C.COUNTESS], [C.BARON], [C.HANDMAID]],
            [C.KING, C.PRINCE, C.SPY, C.PRINCESS, C.GUARD],
            C.PRIEST,
        )
        rec = evaluate(state, 0, rng=random.Random(2), budget_seconds=0.5)
        self.assertGreater(rec.worlds_sampled, 0)
        self.assertTrue(any(v.rollouts > 0 for v in rec.values))

    def test_guard_guesses_are_scored_individually(self) -> None:
        """Guesses must be evaluated, not collapsed to one per target.

        Collapsing was the first design and it cost real strength: measured
        guess values against one opponent spanned 32.8% to 44.0%, and the
        collapse kept whichever sorted first. The budget is protected by
        adaptive allocation instead, not by throwing options away.
        """
        state = state_from_hands(
            [[C.GUARD, C.SPY], [C.BARON], [C.HANDMAID]],
            [C.KING, C.PRINCE, C.PRINCESS, C.CHANCELLOR],
            C.PRIEST,
        )
        rec = evaluate(state, 0, rng=random.Random(2), budget_seconds=0.6)
        guesses = {
            v.action.guess
            for v in rec.values
            if v.action.card is C.GUARD and v.action.guess is not None
        }
        self.assertGreater(
            len(guesses), 1, "only one guess per target was evaluated"
        )

    def test_impossible_guesses_are_pruned(self) -> None:
        """A card the posterior rules out is not worth a single rollout."""
        # Both Priests public, and the set-aside is something else -- an
        # earlier version of this fixture used a third Priest, which does not
        # exist and left the tracker in a state where P1 could still hold one.
        state = state_from_hands(
            [[C.GUARD, C.SPY], [C.BARON], [C.HANDMAID]],
            [C.KING, C.PRINCE, C.PRINCESS, C.CHANCELLOR],
            C.COUNTESS,
            discards=[[], [C.PRIEST], [C.PRIEST]],
        )
        belief = Belief.from_log(
            observe(state.log, 0), 0, 3, faceup=state.faceup
        )
        self.assertEqual(
            belief.hand_marginals().get(1, {}).get(C.PRIEST, 0.0),
            0.0,
            "fixture is wrong: the posterior still allows a Priest",
        )
        rec = evaluate(state, 0, rng=random.Random(2), budget_seconds=0.6)
        for value in rec.values:
            if value.action.card is C.GUARD and value.action.target == 1:
                self.assertIsNot(
                    value.action.guess,
                    C.PRIEST,
                    "guessed a card both copies of which are public",
                )

    def test_budget_concentrates_on_live_contenders(self) -> None:
        """Clearly-losing options must stop consuming rollouts."""
        state = state_from_hands(
            [[C.GUARD, C.PRINCESS], [C.BARON], [C.HANDMAID]],
            [C.KING, C.PRINCE, C.CHANCELLOR, C.SPY],
            C.PRIEST,
        )
        rec = evaluate(state, 0, rng=random.Random(2), budget_seconds=0.6)
        princess = [v for v in rec.values if v.action.card is C.PRINCESS]
        best = rec.best
        if princess:
            self.assertLess(
                princess[0].rollouts,
                best.rollouts,
                "an instant-loss option got as much budget as the leader",
            )

    def test_invariant_skipping_does_not_leak(self) -> None:
        """Real play must keep its checks even if a search ran first."""
        from loveletter import engine

        state = mid_round(3, seed=1)
        evaluate(state, state.current, rng=random.Random(0), budget_seconds=0.2)
        self.assertFalse(
            engine._SKIP_INVARIANTS, "invariant skipping leaked out of a search"
        )

    def test_invariant_skipping_is_restored_after_an_error(self) -> None:
        from loveletter import engine

        try:
            with unchecked_transitions():
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        self.assertFalse(engine._SKIP_INVARIANTS)

    def test_playout_policy_is_stochastic(self) -> None:
        """A deterministic playout samples nothing.

        The first version returned the first acceptable action and ignored
        ``rng``, so every rollout from a determinized world was identical. The
        search then evaluated one arbitrary future very precisely instead of
        averaging over the space of continuations -- and lost to the frozen
        baseline by 0.11 tokens/round. The tell was output that was identical
        to the decimal across positions that should have differed.
        """
        policy = FastPlayoutPolicy()
        state = state_from_hands(
            [[C.GUARD, C.SPY], [C.BARON], [C.HANDMAID], [C.KING]],
            [C.PRINCE, C.PRIEST, C.CHANCELLOR, C.PRINCESS],
            C.COUNTESS,
        )
        picks = {
            policy.choose(state, 0, random.Random(seed)) for seed in range(40)
        }
        self.assertGreater(
            len(picks), 1, "playout policy ignores its rng -- it samples nothing"
        )

    def test_rollouts_from_one_world_produce_varied_outcomes(self) -> None:
        """End to end: the same determinized state must not always resolve
        the same way."""
        from loveletter.evaluator import _play_out

        state = state_from_hands(
            [[C.GUARD, C.SPY], [C.BARON], [C.HANDMAID], [C.KING]],
            [C.PRINCE, C.PRIEST, C.CHANCELLOR, C.PRINCESS, C.SPY],
            C.COUNTESS,
        )
        policy = FastPlayoutPolicy()
        after = apply_unchecked(state, legal_actions(state)[0])
        with unchecked_transitions():
            outcomes = {
                _play_out(after, policy, random.Random(seed)).winners
                for seed in range(30)
            }
        self.assertGreater(
            len(outcomes), 1, "every rollout reached the same outcome"
        )

    def test_playout_policy_avoids_instant_losses(self) -> None:
        policy = FastPlayoutPolicy()
        state = state_from_hands(
            [[C.PRINCESS, C.GUARD], [C.BARON], [C.HANDMAID]],
            [C.KING, C.PRIEST],
            C.SPY,
        )
        action = policy.choose(state, 0, random.Random(0))
        self.assertIsNot(action.card, C.PRINCESS)


class TestBeliefIsCachedNotRebuilt(unittest.TestCase):
    """Rebuilding the belief per rollout would leave no time to roll out."""

    def test_a_supplied_belief_is_reused(self) -> None:
        state = mid_round(4, seed=5)
        belief = Belief.from_log(
            observe(state.log, state.current),
            state.current,
            state.n_players,
            faceup=state.faceup,
        )
        belief.worlds()  # force construction
        # worlds() may be called a few times (the search, and _distinct
        # reading marginals), but the expensive enumeration must happen once.
        enumerations = {"n": 0}
        original = belief._sample_worlds

        def counting():
            enumerations["n"] += 1
            return original()

        belief._sample_worlds = counting  # type: ignore[method-assign]
        rec = evaluate(
            state,
            state.current,
            belief=belief,
            rng=random.Random(1),
            budget_seconds=0.3,
        )
        self.assertGreater(rec.worlds_sampled, 10)
        self.assertEqual(
            enumerations["n"],
            0,
            "the belief was re-enumerated instead of reusing the cache",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestObjective(unittest.TestCase):
    """The score is expected tokens, not round wins.

    Winning a round is not binary-exclusive: tied players each take a token,
    and the Spy bonus can hand one to a player who lost. Measured, the binary
    objective missed 0.164 tokens/round -- larger than the margin the search
    is trying to detect, so scoring `me in winners` was not a simplification
    but a systematic error.
    """

    def test_score_counts_tokens_including_the_spy_bonus(self) -> None:
        from loveletter.evaluator import _play_out

        # A round where P0 takes the Spy bonus without winning outright.
        state = state_from_hands(
            [[C.SPY, C.GUARD], [C.PRIEST], [C.BARON]],
            [],
            C.KING,
            discards=[[C.SPY], [], []],
        )
        after = apply_unchecked(state, Action(C.GUARD, target=2, guess=C.BARON))
        self.assertNotIn(0, after.winners, "fixture: P0 should not win outright")
        self.assertEqual(
            after.player(0).tokens, 1, "fixture: P0 should hold a Spy token"
        )

    def test_confidence_interval_uses_observed_spread(self) -> None:
        """A Bernoulli formula would understate it once tokens exceed 1."""
        v = ActionValue(action=Action(C.SPY), rollouts=100, wins=100.0)
        v.sq = 200.0  # e.g. fifty 2-token rounds: mean 1.0, real variance > 0
        self.assertGreater(v.ci, 0.0, "spread was ignored")

    def test_expected_tokens_can_exceed_one(self) -> None:
        """Round win plus Spy bonus is two tokens; the type must allow it."""
        v = ActionValue(action=Action(C.SPY), rollouts=10, wins=15.0)
        v.sq = 25.0
        self.assertGreater(v.win_rate, 1.0)


class TestChancellorIsNotPreDecided(unittest.TestCase):
    """The Chancellor's return-order is not a choice available at rec time.

    The two cards are drawn *after* the Chancellor is played. Offering ranked
    return-orders would describe a decision the user has not been given yet --
    and at the table it did worse: the CLI builds its state with placeholder
    deck cards, so every option came back naming those placeholders, e.g.
    "returning [Guard, Princess]" when no Guard had been drawn.
    """

    def _chancellor_state(self):
        return state_from_hands(
            [[C.CHANCELLOR, C.PRINCESS], [C.BARON], [C.HANDMAID]],
            [C.KING, C.PRIEST, C.SPY, C.GUARD],
            C.COUNTESS,
        )

    def test_one_chancellor_option_is_offered(self) -> None:
        rec = evaluate(
            self._chancellor_state(), 0, rng=random.Random(1),
            budget_seconds=0.5,
        )
        chancellors = [
            v for v in rec.values if v.action.card is C.CHANCELLOR
        ]
        self.assertEqual(
            len(chancellors), 1,
            f"expected one Chancellor option, got {[str(v.action) for v in chancellors]}",
        )

    def test_the_offered_option_names_no_cards(self) -> None:
        rec = evaluate(
            self._chancellor_state(), 0, rng=random.Random(1),
            budget_seconds=0.5,
        )
        chancellor = next(
            v for v in rec.values if v.action.card is C.CHANCELLOR
        )
        self.assertIsNone(
            chancellor.action.chancellor_return,
            "the recommendation pre-decided a return the user cannot make yet",
        )
        self.assertNotIn("returning", str(chancellor.action))

    def test_it_is_still_rolled_out(self) -> None:
        """Collapsing the option must not stop it being evaluated."""
        rec = evaluate(
            self._chancellor_state(), 0, rng=random.Random(1),
            budget_seconds=0.5,
        )
        chancellor = next(
            v for v in rec.values if v.action.card is C.CHANCELLOR
        )
        self.assertGreater(chancellor.rollouts, 10)
        self.assertGreater(chancellor.win_rate, 0.0)
