"""Agents: things that choose an action, so they can be measured against
each other.

An agent sees only what a real player at the table would see -- the engine
state projected to their seat -- and returns one of the legal actions. That
restriction is what makes an arena result mean anything: an agent that peeks
at hidden state would win every comparison and teach us nothing.

The baseline
------------
:class:`BaselineAgent` is the permanent yardstick.  It is deliberately frozen:
never tuned, never deleted, never "improved".  Every future version of the tool
is measured against this exact policy, so a number from today is comparable
with a number from six months from now.  If it changes, every historical
measurement silently becomes meaningless.

Tuning belongs in agents that are *not* the baseline.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol, Sequence

from .config import Card, GameConfig
from .engine import Action, GameState, PlayerId, legal_actions


class Agent(Protocol):
    """Chooses one legal action from the acting player's point of view."""

    name: str

    def choose(
        self, state: GameState, me: PlayerId, rng: random.Random
    ) -> Action:
        """Pick an action for ``me``.  Must return a member of
        ``legal_actions(state)``."""
        ...


class RandomAgent:
    """Uniform over legal actions.  The floor: anything worth shipping beats
    this comfortably, and if it does not, something is wrong."""

    name = "random"

    def choose(
        self, state: GameState, me: PlayerId, rng: random.Random
    ) -> Action:
        return rng.choice(legal_actions(state))


def _visible_counts(state: GameState, me: PlayerId) -> dict[Card, int]:
    """Cards this player can see are gone: public discards, faceup, own hand.

    Deliberately not the belief tracker.  The baseline must stay a simple,
    frozen reference point; wiring it to the tracker would make it move
    whenever the tracker does, which is exactly what a baseline must not do.
    """
    seen: dict[Card, int] = {}
    for p in state.players:
        for card in p.discards:
            seen[card] = seen.get(card, 0) + 1
    for card in state.faceup:
        seen[card] = seen.get(card, 0) + 1
    for card in state.player(me).hand:
        seen[card] = seen.get(card, 0) + 1
    return seen


@dataclass(frozen=True, slots=True)
class BaselineAgent:
    """The permanent baseline: fixed, legible heuristics.

    FROZEN.  Do not tune these rules, do not add to them, do not delete this
    class.  Its whole value is that a score measured against it is stable over
    time.  New ideas go in a new agent -- and a subclass that overrides
    ``choose`` is a new agent, so give it its own ``name`` or arena results
    will be labelled with the baseline's.

    The rules, in priority order:

    1. Never play the Princess.
    2. Play the Countess when compelled (the engine enforces this anyway).
    3. Guard-guess the most likely remaining card, counting what is visible.
    4. Prefer Handmaid protection when the deck is still deep.
    5. Play Baron only when likely ahead; never against an unknown when
       holding a low card.
    6. Never Prince yourself while holding the Princess.
    7. Otherwise play the lower-value card and keep the higher one, which is
       what wins on a deck-out.
    """

    name: str = "baseline"

    def choose(
        self, state: GameState, me: PlayerId, rng: random.Random
    ) -> Action:
        actions = legal_actions(state)
        if len(actions) == 1:
            return actions[0]
        scored = [(self._score(state, me, a), i) for i, a in enumerate(actions)]
        best = max(scored)[0]
        ties = [i for s, i in scored if s == best]
        # Deterministic given the rng, so paired seeds stay paired.
        return actions[rng.choice(ties)]

    def _score(self, state: GameState, me: PlayerId, action: Action) -> float:
        hand = state.player(me).hand
        other = next((c for c in hand if c is not action.card), action.card)
        card = action.card
        deck = state.deck_size

        if card is Card.PRINCESS:
            return -1000.0  # instant loss

        if card is Card.COUNTESS:
            # Compelled plays are the only action available; otherwise she is
            # a poor discard -- high value, and holding her wins deck-outs.
            return -5.0

        if card is Card.GUARD:
            if action.target is None:
                return -8.0  # fizzles: everyone protected
            return 6.0 + self._guess_bonus(state, me, action)

        if card is Card.PRIEST:
            return 3.0 if action.target is not None else -8.0

        if card is Card.BARON:
            if action.target is None:
                return -8.0
            # Only fight when the kept card is genuinely high.
            return 4.0 if other >= Card.PRINCE else -2.0

        if card is Card.HANDMAID:
            # Protection is worth most while the round still has turns to run.
            return 5.0 if deck > 2 else 2.0

        if card is Card.PRINCE:
            if action.target == me:
                # Self-Prince discards the kept card. Fatal with the Princess.
                if other is Card.PRINCESS:
                    return -1000.0
                return -3.0
            return 4.5

        if card is Card.KING:
            if action.target is None:
                return -8.0
            # Trading away a low card for an unknown is usually fine; trading
            # away a high one is not.
            return 3.0 if other <= Card.BARON else -4.0

        if card is Card.CHANCELLOR:
            return 2.0 if deck >= 2 else 0.5

        if card is Card.SPY:
            return 1.0  # near-free, and the bonus is real

        return 0.0

    def _guess_bonus(
        self, state: GameState, me: PlayerId, action: Action
    ) -> float:
        """Prefer guesses that are likely to be right.

        Counts only what is publicly visible plus the agent's own hand, which
        is what a competent human tracks without help.
        """
        if action.guess is None:
            return 0.0
        seen = _visible_counts(state, me)
        remaining = state.config.copies(action.guess) - seen.get(action.guess, 0)
        if remaining <= 0:
            return -5.0  # provably impossible: never guess it
        # Scaled so the most abundant unseen card wins, ties broken by value.
        return remaining * 0.5 + action.guess * 0.01


#: The frozen reference. Import this, never a tuned copy.
BASELINE = BaselineAgent()


@dataclass(frozen=True, slots=True)
class BeliefAgent:
    """Baseline heuristics, but Guard guesses come from the belief tracker.

    Not the baseline: this one is allowed to change.  It exists so the arena
    can measure whether the tracker -- and the opponent-model constants that
    feed it -- actually buy anything at the table.

    Only the Guard guess is belief-driven.  That keeps the comparison against
    :class:`BaselineAgent` a test of *one* thing: does knowing the posterior
    over an opponent's hand make guesses better?  Wiring the tracker into
    every decision at once would tell us it helped, or did not, without saying
    which part did the work.
    """

    name: str = "belief"
    policy: object | None = None
    max_worlds: int = 2000

    def choose(
        self, state: GameState, me: PlayerId, rng: random.Random
    ) -> Action:
        actions = legal_actions(state)
        if len(actions) == 1:
            return actions[0]

        guards = [a for a in actions if a.card is Card.GUARD and a.target is not None]
        if guards:
            best = self._best_guard(state, me, guards)
            if best is not None:
                # Compare the belief-chosen Guard against the other options on
                # the baseline's own scale, so only the guess differs.
                base = BaselineAgent()
                alternatives = [
                    a for a in actions if not (a.card is Card.GUARD and a.target is not None)
                ]
                best_alt = max(
                    (base._score(state, me, a) for a in alternatives),
                    default=float("-inf"),
                )
                if base._score(state, me, best) >= best_alt:
                    return best

        return BaselineAgent().choose(state, me, rng)

    def _best_guard(
        self, state: GameState, me: PlayerId, guards: list[Action]
    ) -> Action | None:
        """Pick the (target, guess) with the highest posterior probability."""
        from .belief import Belief
        from .observation import observe

        # Deliberately no try/except. An earlier version swallowed tracker
        # failures and fell back to the baseline, which meant a 1.2% rate of
        # "no consistent world" bugs went unnoticed and a whole constant sweep
        # silently measured the baseline against itself. A tracker that
        # contradicts itself is a defect to fix, not a condition to survive.
        belief = Belief.from_log(
            observe(state.log, me),
            me,
            state.n_players,
            config=state.config,
            faceup=state.faceup,
            policy=self.policy,
        )
        belief.max_worlds = self.max_worlds
        marginals = belief.hand_marginals()

        best: Action | None = None
        best_p = -1.0
        for action in guards:
            if action.target is None or action.guess is None:
                continue
            p = marginals.get(action.target, {}).get(action.guess, 0.0)
            if p > best_p:
                best_p, best = p, action
        return best
