"""A PIMC agent for arena measurement."""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from loveletter.agents import BaselineAgent
from loveletter.belief import Belief
from loveletter.config import Card
from loveletter.engine import Action, GameState, PlayerId, legal_actions
from loveletter.evaluator import evaluate
from loveletter.observation import observe


@dataclass
class PIMCAgent:
    """Recommends by determinized search, with the belief built once per turn."""

    name: str = "pimc"
    budget_seconds: float = 0.05      # arena budget; the CLI uses 1.4
    policy: object | None = None
    max_worlds: int = 400

    def _pick(self, rec, state: GameState, me: PlayerId) -> Action:
        """Convert a Recommendation into one action, honestly.

        When the search separated a winner, take it.  When the top options'
        intervals overlap, the argmax is winner's curse: among options the
        search itself says it cannot rank, the one with the highest estimate
        is mostly the one whose noise came up heads, and at ~45 rollouts per
        decision that coin lands on the genuinely-worse tail often enough to
        cancel the search's real gains. So ties are broken by the frozen baseline's
        heuristic score. The search then equals the baseline where it cannot
        distinguish and deviates only on gaps it actually established.

        Information-ranked recommendations (Priest looks) are exempt: their
        ordering is exact, not estimated, and the baseline scores all Priest
        targets identically, so "tie-breaking" would erase the ranking.
        """
        if rec.information_ranked or not rec.tied:
            return rec.best.action
        contenders = [rec.best] + rec.tied
        base = BaselineAgent()
        return max(
            contenders, key=lambda v: base._score(state, me, v.action)
        ).action

    def choose(self, state: GameState, me: PlayerId, rng: random.Random) -> Action:
        actions = legal_actions(state)
        if len(actions) == 1:
            return actions[0]
        belief = Belief.from_log(
            observe(state.log, me), me, state.n_players,
            config=state.config, faceup=state.faceup, policy=self.policy,
        )
        belief.max_worlds = self.max_worlds
        rec = evaluate(
            state, me, belief=belief, rng=rng,
            budget_seconds=self.budget_seconds,
        )
        chosen = self._pick(rec, state, me)
        # _distinct collapsed Guard guesses; pick the guess from the posterior.
        if chosen.card is Card.GUARD and chosen.target is not None:
            marg = belief.hand_marginals().get(chosen.target, {})
            best, bestp = None, -1.0
            for a in actions:
                if a.card is not Card.GUARD or a.target != chosen.target:
                    continue
                p = marg.get(a.guess, 0.0)
                if p > bestp:
                    bestp, best = p, a
            if best is not None:
                return best
        for a in actions:
            if a.card is chosen.card and a.target == chosen.target:
                return a
        return actions[0]
