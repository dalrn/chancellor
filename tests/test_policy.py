"""Phase 2b tests: soft constraints and the opponent model.

Two properties matter more than any individual number here.

First, 2b must be *switchable off*: with no policy, or with the null policy,
the posterior has to equal Phase 2a's exactly. Otherwise there is no way to
tell whether the opponent model is helping or just moving numbers around.

Second, soft evidence must never behave like a hard constraint. A policy that
returns zero for something an opponent actually did would prune the true world
out of the posterior -- strictly worse than staying vague, because the tracker
would then be confidently wrong.
"""

from __future__ import annotations

import random
import unittest

from loveletter.belief import Belief
from loveletter.config import Card
from loveletter.engine import (
    Action,
    apply,
    legal_actions,
    new_round,
    state_from_hands,
)
from loveletter.observation import observe
from loveletter.policy import (
    MIN_PROB,
    HeuristicPolicy,
    PlayContext,
    UniformPolicy,
)

C = Card


def countess_state():
    """P1 plays the Countess while holding the King -- so, compelled."""
    state = state_from_hands(
        [[C.SPY], [C.COUNTESS, C.KING], [C.HANDMAID], [C.GUARD]],
        [C.PRIEST, C.BARON, C.CHANCELLOR],
        C.PRINCE,
        current=1,
    )
    return apply(state, Action(C.COUNTESS))


def belief(state, viewer=0, policy=None, **kw):
    return Belief.from_log(
        observe(state.log, viewer),
        viewer,
        state.n_players,
        faceup=state.faceup,
        policy=policy,
        **kw,
    )


class TestPolicyContract(unittest.TestCase):
    def test_probability_is_positive_for_any_held_card(self) -> None:
        """No legal play may be assigned probability zero."""
        pol = HeuristicPolicy()
        ctx = PlayContext(actor=1)
        cards = list(C)
        for a in cards:
            for b in cards:
                with self.subTest(hand=(a, b)):
                    p = pol.play_probability((a, b), a, ctx)
                    # MIN_PROB floors the unnormalised score, so the resulting
                    # probability lands just under it once normalised. What
                    # must hold is that no legal play is impossible.
                    self.assertGreater(
                        p, 0.0, f"holding ({a}, {b}), playing {a} got zero"
                    )
                    self.assertGreater(p, MIN_PROB / 2)

    def test_probabilities_over_a_hand_sum_to_one(self) -> None:
        pol = HeuristicPolicy()
        ctx = PlayContext(actor=1)
        for a in list(C):
            for b in list(C):
                if a is b:
                    continue
                with self.subTest(hand=(a, b)):
                    total = pol.play_probability(
                        (a, b), a, ctx
                    ) + pol.play_probability((a, b), b, ctx)
                    self.assertAlmostEqual(total, 1.0, places=6)

    def test_card_not_in_hand_is_impossible(self) -> None:
        pol = HeuristicPolicy()
        ctx = PlayContext(actor=1)
        self.assertEqual(
            pol.play_probability((C.GUARD, C.SPY), C.PRINCESS, ctx), 0.0
        )

    def test_two_of_a_kind_is_certain(self) -> None:
        """With a pair there is no choice to model."""
        pol = HeuristicPolicy()
        ctx = PlayContext(actor=1)
        self.assertEqual(
            pol.play_probability((C.GUARD, C.GUARD), C.GUARD, ctx), 1.0
        )

    def test_countess_is_near_certain_when_compelled(self) -> None:
        """Compulsion gives ~1.0, not exactly 1.0.

        The MIN_PROB floor keeps the illegal alternative barely possible, so
        the compelled play normalises to just under one. That gap is the
        policy refusing to state a certainty, which is the property that stops
        soft evidence from pruning worlds.
        """
        pol = HeuristicPolicy()
        ctx = PlayContext(actor=1)
        for forcer in (C.KING, C.PRINCE):
            with self.subTest(forcer=forcer):
                p = pol.play_probability((C.COUNTESS, forcer), C.COUNTESS, ctx)
                self.assertGreater(p, 0.99)
                self.assertLess(p, 1.0)

    def test_countess_is_unlikely_when_free(self) -> None:
        """Playing her without a forcer is a bluff: possible, not likely."""
        pol = HeuristicPolicy()
        ctx = PlayContext(actor=1)
        p = pol.play_probability((C.COUNTESS, C.GUARD), C.COUNTESS, ctx)
        self.assertGreater(p, 0.0)
        self.assertLess(p, 0.5)

    def test_princess_is_almost_never_played(self) -> None:
        pol = HeuristicPolicy()
        ctx = PlayContext(actor=1)
        p = pol.play_probability((C.PRINCESS, C.GUARD), C.PRINCESS, ctx)
        self.assertGreater(p, 0.0, "must stay possible -- it is legal")
        self.assertLess(p, 0.1)


class TestNullModel(unittest.TestCase):
    """2b must be switchable off, exactly."""

    def test_uniform_policy_reproduces_phase_2a(self) -> None:
        state = countess_state()
        hard = belief(state, policy=None).hand_marginals()
        null = belief(state, policy=UniformPolicy()).hand_marginals()
        self.assertEqual(set(hard), set(null))
        for pid in hard:
            for card in set(hard[pid]) | set(null[pid]):
                self.assertAlmostEqual(
                    hard[pid].get(card, 0.0),
                    null[pid].get(card, 0.0),
                    places=9,
                    msg=f"P{pid} {card} differs under the null model",
                )

    def test_no_policy_leaves_weights_at_one(self) -> None:
        state = countess_state()
        b = belief(state, policy=None)
        for w in b.worlds():
            self.assertEqual(w.weight, 1.0)

    def test_uniform_policy_matches_across_random_games(self) -> None:
        for seed in range(8):
            rng = random.Random(seed)
            state = new_round(3, rng)
            for _ in range(4):
                if state.round_over:
                    break
                state = apply(state, rng.choice(legal_actions(state)))
            hard = belief(state, policy=None).hand_marginals()
            null = belief(state, policy=UniformPolicy()).hand_marginals()
            for pid in hard:
                for card in hard[pid]:
                    self.assertAlmostEqual(
                        hard[pid][card],
                        null[pid].get(card, 0.0),
                        places=9,
                        msg=f"seed {seed}: P{pid} {card}",
                    )


class TestCountessInference(unittest.TestCase):
    """The update the rulebook's ambiguity is designed to allow."""

    def test_countess_play_raises_king_or_prince_probability(self) -> None:
        state = countess_state()
        hard = belief(state, policy=None).hand_marginals()[1]
        soft = belief(state, policy=HeuristicPolicy()).hand_marginals()[1]
        hard_p = hard.get(C.KING, 0.0) + hard.get(C.PRINCE, 0.0)
        soft_p = soft.get(C.KING, 0.0) + soft.get(C.PRINCE, 0.0)
        self.assertGreater(
            soft_p,
            hard_p * 2,
            f"Countess play barely moved the posterior: {hard_p} -> {soft_p}",
        )

    def test_lower_bluff_rate_means_stronger_inference(self) -> None:
        """The parameter must actually control the strength of the update."""
        state = countess_state()
        results = []
        for bluff in (0.01, 0.05, 0.2, 0.5):
            m = belief(
                state, policy=HeuristicPolicy(countess_bluff=bluff)
            ).hand_marginals()[1]
            results.append(m.get(C.KING, 0.0) + m.get(C.PRINCE, 0.0))
        for a, b in zip(results, results[1:]):
            self.assertGreater(
                a, b, "a higher bluff rate must weaken the inference"
            )

    def test_inference_never_reaches_certainty(self) -> None:
        """Even a tiny bluff rate must leave the bluff possible."""
        state = countess_state()
        m = belief(
            state, policy=HeuristicPolicy(countess_bluff=1e-6)
        ).hand_marginals()[1]
        p = m.get(C.KING, 0.0) + m.get(C.PRINCE, 0.0)
        self.assertLess(p, 1.0, "the bluff was ruled out entirely")


class TestSoftNeverPrunes(unittest.TestCase):
    """Reweighting must not act as a hard constraint."""

    def test_the_true_world_survives_reweighting(self) -> None:
        """Soft evidence may downweight the truth, never delete it."""
        for seed in range(10):
            rng = random.Random(seed)
            state = new_round(3, rng)
            while not state.round_over:
                b = belief(state, policy=HeuristicPolicy())
                truth = tuple(tuple(sorted(p.hand)) for p in state.players)
                match = [
                    w
                    for w in b.worlds()
                    if tuple(tuple(sorted(h)) for h in w.hands) == truth
                ]
                self.assertTrue(
                    match,
                    f"seed {seed} turn {state.turn}: true world was pruned",
                )
                self.assertGreater(
                    match[0].weight,
                    0.0,
                    f"seed {seed} turn {state.turn}: true world got weight 0",
                )
                state = apply(state, rng.choice(legal_actions(state)))

    def test_world_set_is_identical_with_and_without_a_policy(self) -> None:
        """A policy changes weights, never which worlds are possible."""
        state = countess_state()
        hard = belief(state, policy=None).worlds()
        soft = belief(state, policy=HeuristicPolicy()).worlds()
        self.assertEqual(len(hard), len(soft))
        self.assertEqual(
            {w.hands for w in hard},
            {w.hands for w in soft},
            "the policy changed the set of possible worlds",
        )

    def test_marginals_stay_probabilities_under_weighting(self) -> None:
        for seed in range(8):
            rng = random.Random(seed)
            state = new_round(4, rng)
            for _ in range(5):
                if state.round_over:
                    break
                state = apply(state, rng.choice(legal_actions(state)))
            b = belief(state, policy=HeuristicPolicy())
            for pid, marg in b.hand_marginals().items():
                for card, p in marg.items():
                    self.assertGreaterEqual(p, 0.0)
                    self.assertLessEqual(
                        p, 1.0 + 1e-9, f"seed {seed}: P{pid} {card} = {p}"
                    )

    def test_expected_counts_still_match_the_deck(self) -> None:
        """Weighting must not create or destroy cards."""
        from collections import Counter

        from loveletter.config import STANDARD

        state = countess_state()
        b = belief(state, policy=HeuristicPolicy())
        public = Counter(b.constraints.public)
        expected = b.expected_counts()
        for card in STANDARD.counts:
            total = expected.get(card, 0.0) + public.get(card, 0)
            self.assertAlmostEqual(
                total, STANDARD.copies(card), places=6, msg=f"{card} miscounted"
            )


class TestEffectiveSampleSize(unittest.TestCase):
    def test_ess_equals_world_count_without_a_policy(self) -> None:
        b = belief(countess_state(), policy=None)
        self.assertAlmostEqual(
            b.effective_sample_size(), b.world_count(), places=6
        )

    def test_ess_drops_when_weighting_concentrates_mass(self) -> None:
        """A sharp opponent model must report that it is leaning on itself."""
        state = countess_state()
        flat = belief(state, policy=None).effective_sample_size()
        sharp = belief(
            state, policy=HeuristicPolicy(countess_bluff=0.01)
        ).effective_sample_size()
        self.assertLess(
            sharp, flat, "ESS did not fall despite concentrated weights"
        )
        self.assertGreater(sharp, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestPairWeighting(unittest.TestCase):
    """Pairs raise a world's weight -- in the heuristic, never in the null.

    ``P(play Guard | held two Guards) = 1`` while a mixed hand splits, so a
    world implying a pair really does explain the observed play better and
    Bayes says to upweight it. That is inference, and the heuristic should do
    it.

    The identical arithmetic inside ``UniformPolicy`` was a bug: a model that
    claims to assert nothing must not quietly assert that pairs are twice as
    likely. Same numbers, opposite verdicts, so both directions are pinned
    here.
    """

    def _guard_pair_state(self):
        state = state_from_hands(
            [[C.SPY], [C.GUARD, C.GUARD], [C.HANDMAID], [C.BARON]],
            [C.PRIEST, C.KING, C.CHANCELLOR],
            C.PRINCE,
            current=1,
        )
        return apply(state, Action(C.GUARD, target=2, guess=C.PRINCESS))

    def test_heuristic_upweights_the_pair_explanation(self) -> None:
        state = self._guard_pair_state()
        hard = belief(state, policy=None).hand_marginals()[1]
        soft = belief(state, policy=HeuristicPolicy()).hand_marginals()[1]
        self.assertGreater(
            soft.get(C.GUARD, 0.0),
            hard.get(C.GUARD, 0.0),
            "a pair explains the observed play better and should gain weight",
        )

    def test_null_model_is_indifferent_to_pairs(self) -> None:
        state = self._guard_pair_state()
        hard = belief(state, policy=None).hand_marginals()[1]
        null = belief(state, policy=UniformPolicy()).hand_marginals()[1]
        self.assertAlmostEqual(
            null.get(C.GUARD, 0.0),
            hard.get(C.GUARD, 0.0),
            places=9,
            msg="UniformPolicy is asserting something about pairs",
        )

    def test_the_pair_update_stays_modest(self) -> None:
        """Legitimate evidence, not a landslide -- it must not dominate."""
        state = self._guard_pair_state()
        hard = belief(state, policy=None).hand_marginals()[1]
        soft = belief(state, policy=HeuristicPolicy()).hand_marginals()[1]
        self.assertLess(
            soft.get(C.GUARD, 0.0),
            hard.get(C.GUARD, 0.0) * 3,
            "a single play should not swamp the prior",
        )


class TestEvidenceFreshness(unittest.TestCase):
    """Only a player's most recent play may reweight their current hand.

    A play describes the hand held at that moment. Once the player draws, the
    card they kept is joined by a new one, and an older play no longer
    describes what they hold. Scoring every historical play against the
    current hand multiplies the same stale inference once per play, which
    quietly manufactures confidence the evidence does not support.
    """

    def test_only_the_latest_play_per_player_is_scored(self) -> None:
        for seed in range(6):
            rng = random.Random(seed)
            state = new_round(3, rng)
            while not state.round_over:
                b = belief(state, policy=HeuristicPolicy())
                actors = [rec[0] for rec in b.constraints.observed_plays]
                self.assertEqual(
                    len(actors),
                    len(set(actors)),
                    f"seed {seed}: a player has more than one scored play",
                )
                state = apply(state, rng.choice(legal_actions(state)))

    def test_eliminated_players_stop_contributing_evidence(self) -> None:
        """Their hand is public; their old plays say nothing about it."""
        for seed in range(6):
            rng = random.Random(seed)
            state = new_round(4, rng)
            while not state.round_over:
                b = belief(state, policy=HeuristicPolicy())
                for actor, *_ in b.constraints.observed_plays:
                    self.assertNotIn(
                        actor,
                        b.constraints.out,
                        f"seed {seed}: eliminated P{actor} still reweighting",
                    )
                state = apply(state, rng.choice(legal_actions(state)))

    def test_a_later_play_replaces_an_earlier_one(self) -> None:
        state = state_from_hands(
            [[C.SPY], [C.COUNTESS, C.KING], [C.HANDMAID], [C.GUARD]],
            [C.PRIEST, C.BARON, C.CHANCELLOR, C.PRINCESS, C.SPY],
            C.PRINCE,
            current=1,
        )
        after = apply(state, Action(C.COUNTESS))
        first = belief(after, policy=HeuristicPolicy())
        self.assertEqual(len(first.constraints.observed_plays), 1)
        # Walk to P1's next turn and have them play again.
        while not after.round_over and after.current != 1:
            after = apply(after, legal_actions(after)[0])
        if not after.round_over:
            after = apply(after, legal_actions(after)[0])
            later = belief(after, policy=HeuristicPolicy())
            p1_plays = [
                r for r in later.constraints.observed_plays if r[0] == 1
            ]
            self.assertLessEqual(len(p1_plays), 1)
