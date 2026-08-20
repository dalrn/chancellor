"""Deck composition and token thresholds.

Everything variant-specific lives here.  The classic 16-card game is a
different ``GameConfig``, not a different code path.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from types import MappingProxyType
from typing import Mapping


class Card(IntEnum):
    """Character cards, ordered by the value used for Baron and deck-out."""

    SPY = 0
    GUARD = 1
    PRIEST = 2
    BARON = 3
    HANDMAID = 4
    PRINCE = 5
    CHANCELLOR = 6
    KING = 7
    COUNTESS = 8
    PRINCESS = 9

    def __str__(self) -> str:
        return self.name.capitalize()


#: Cards that must choose *another* player.  Prince may choose anyone.
TARGETS_OTHER: frozenset[Card] = frozenset(
    {Card.GUARD, Card.PRIEST, Card.BARON, Card.KING}
)

#: Cards that choose a target at all.
TARGETING: frozenset[Card] = TARGETS_OTHER | {Card.PRINCE}

#: Cards that force the Countess when co-held with her.
COUNTESS_FORCERS: frozenset[Card] = frozenset({Card.KING, Card.PRINCE})


@dataclass(frozen=True, slots=True)
class GameConfig:
    """A deck composition plus the token thresholds that go with it."""

    counts: Mapping[Card, int]
    tokens_to_win: Mapping[int, int]
    faceup_at_two_players: int = 3
    min_players: int = 2
    max_players: int = 6

    def __post_init__(self) -> None:
        for card, n in self.counts.items():
            if n < 0:
                raise ValueError(f"negative count for {card}: {n}")
        if self.min_players < 2:
            raise ValueError("Love Letter needs at least 2 players")
        if self.max_players < self.min_players:
            raise ValueError("max_players below min_players")
        for n in range(self.min_players, self.max_players + 1):
            if n not in self.tokens_to_win:
                raise ValueError(f"no token threshold for {n} players")

    @property
    def deck_size(self) -> int:
        """Total number of character cards in the shuffled deck."""
        return sum(self.counts.values())

    def copies(self, card: Card) -> int:
        """How many of ``card`` exist in this variant (0 if absent)."""
        return self.counts.get(card, 0)

    def all_cards(self) -> list[Card]:
        """Every physical card, sorted.  Length == :attr:`deck_size`."""
        return [c for c in sorted(self.counts) for _ in range(self.counts[c])]

    def faceup_count(self, n_players: int) -> int:
        """Cards set aside faceup during setup for ``n_players``."""
        return self.faceup_at_two_players if n_players == 2 else 0

    def undealt_deck_size(self, n_players: int) -> int:
        """Deck size once setup is complete, before the first draw."""
        return (
            self.deck_size
            - 1  # facedown set-aside card
            - self.faceup_count(n_players)
            - n_players  # dealt hands
        )

    def validate_players(self, n_players: int) -> None:
        if not self.min_players <= n_players <= self.max_players:
            raise ValueError(
                f"{n_players} players outside {self.min_players}"
                f"-{self.max_players} for this variant"
            )
        if self.undealt_deck_size(n_players) < 1:
            raise ValueError(f"deck too small to deal {n_players} players")


STANDARD = GameConfig(
    counts=MappingProxyType(
        {
            Card.SPY: 2,
            Card.GUARD: 6,
            Card.PRIEST: 2,
            Card.BARON: 2,
            Card.HANDMAID: 2,
            Card.PRINCE: 2,
            Card.CHANCELLOR: 2,
            Card.KING: 1,
            Card.COUNTESS: 1,
            Card.PRINCESS: 1,
        }
    ),
    tokens_to_win=MappingProxyType({2: 6, 3: 5, 4: 4, 5: 3, 6: 3}),
)

#: The 16-card classic: one fewer Guard, no Chancellors, no Spies, 2-4 players.
CLASSIC = GameConfig(
    counts=MappingProxyType(
        {
            Card.GUARD: 5,
            Card.PRIEST: 2,
            Card.BARON: 2,
            Card.HANDMAID: 2,
            Card.PRINCE: 2,
            Card.KING: 1,
            Card.COUNTESS: 1,
            Card.PRINCESS: 1,
        }
    ),
    tokens_to_win=MappingProxyType({2: 7, 3: 5, 4: 4}),
    max_players=4,
)

assert STANDARD.deck_size == 21, STANDARD.deck_size
assert CLASSIC.deck_size == 16, CLASSIC.deck_size
