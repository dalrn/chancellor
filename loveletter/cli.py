"""Phase 4: the command-line interface.

Typed with one hand mid-game.  Short commands, loud rejections, undo.

    start 4 seat 2 dealt guard      begin a 4-player round, I sit at seat 2
    d king                          I draw a King (opponents: just `d`)
    p 1 3 pr hit                    play Guard (value 1) at P3 guessing
                                    Priest -- it hit
    p king 1 got baron              play King at P1; I received the Baron
    p chancellor                    play my Chancellor -- then `drew k sp`,
                                    `r` for advice, `keep k ret g sp`
    u                               undo the last entry
    s                               show the table
    r                               recommendation
    end 1=king 2=prince             deck-out: enter revealed hands
    abandon                         drop this round, keep the advisor open
    q                               quit (asks once if a round is live;
                                    `q!` or `abort` quit immediately)

Cards are named by value (0-9), full name, or unique prefix.

What the recommendation shows, and why
--------------------------------------
**Heads-up the recommendation is belief-derived, not search-ranked.**  The
arena measured search adding nothing at 2 players (+0.0084 +/- 0.0195 over
8,000 paired games) while belief-guided play beat the reference heuristic
conclusively (+0.0578 +/- 0.0230).  Showing a search-ranked list there would
imply precision the measurement does not support.

At 3-5 players the recommendation is search-ranked (conclusive arena edges:
+0.076 at 3p, +0.096 at 4p), and every line carries its rollout count and 95%
interval so it is visible what the number rests on.  Overlapping intervals are
reported as NOT SEPARATED -- most Love Letter positions are genuine near-ties,
and saying so is the honest answer, not a hedge.  All validation was done at
the 0.05s arena configuration; the table budget (1.4s) is stronger but
unmeasured, and the footer says so.
"""

from __future__ import annotations

import random
from dataclasses import replace as _dc_replace
from typing import Callable, Sequence

from .agents import BaselineAgent
from .config import STANDARD, Card, GameConfig
from .engine import Action, GameState, legal_actions, state_from_hands
from .evaluator import evaluate
from .table import EntryError, Table

#: Arena-validated strength per player count, shown with recommendations.
VALIDATION_NOTE = {
    2: "heads-up: belief-derived (search added nothing measurable: "
       "+0.008 +/- 0.020 over 8,000 paired games)",
    3: "arena-validated at 3p: +0.076 +/- 0.023 tokens/round over the "
       "reference heuristic (0.05s config; table budget unmeasured)",
    4: "arena-validated at 4p: +0.096 +/- 0.021 tokens/round over the "
       "reference heuristic and +0.071 +/- 0.020 over a different "
       "belief-guided opponent (0.05s config; table budget unmeasured)",
    5: "5 players: playable, but strength is unmeasured (evaluation cost); "
       "treat rankings with extra caution",
    6: "6 players: UNVALIDATED -- over the latency budget and never "
       "arena-measured; numbers shown are best-effort only",
}


def parse_card(token: str) -> Card:
    """A card by value digit, full name, or unique prefix. Loud otherwise."""
    token = token.strip().lower()
    if token.isdigit():
        value = int(token)
        for card in Card:
            if card.value == value:
                return card
        raise EntryError(f"no card has value {value}")
    matches = [c for c in Card if c.name.lower().startswith(token)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise EntryError(f"no card called '{token}'")
    raise EntryError(
        f"'{token}' is ambiguous: {', '.join(c.name.lower() for c in matches)}"
    )


# ---------------------------------------------------------------- rendering


def render_table(t: Table) -> str:
    lines = [
        f"deck {len(t.deck_slots)}"
        + ("" if t.set_aside_available else "  (set-aside gone)")
        + f"   turn {t.turn}   "
        + ("ROUND OVER" if t.round_over else f"P{t.current} to act"),
    ]
    if t.faceup:
        lines.append("faceup: " + " ".join(str(c) for c in t.faceup))
    for p in range(t.n_players):
        seat = t.seats[p]
        tags = []
        if p == t.me:
            tags.append("me:" + "+".join(str(c) for c in t.my_hand))
        if seat.out:
            tags.append("OUT")
        if seat.protected:
            tags.append("protected")
        disc = " ".join(str(c) for c in seat.discards) or "-"
        lines.append(
            f"  P{p} [{seat.tokens} tok] {' '.join(tags):<18} {disc}"
        )
    unseen = {
        c: t.config.copies(c) - t.visible_count(c)
        for c in sorted(t.config.counts)
    }
    lines.append(
        "unseen: "
        + "  ".join(f"{c}:{n}" for c, n in unseen.items() if n > 0)
    )
    return "\n".join(lines)


def render_marginals(t: Table) -> str:
    """Per-opponent card probability table from the posterior."""
    belief = t.belief()
    marg = belief.hand_marginals()
    cards = sorted(t.config.counts)
    header = "P(x) " + " ".join(f"{c.value:>4}" for c in cards)
    lines = [header]
    for p in range(t.n_players):
        if p == t.me or t.seats[p].out:
            continue
        row = marg.get(p, {})
        lines.append(
            f"  P{p} " + " ".join(f"{row.get(c, 0.0):>4.0%}" for c in cards)
        )
    lines.append(
        f"({belief.world_count()} candidate worlds"
        + ("" if belief.exact else f", sampled from {belief.total_worlds}")
        + ")"
    )
    return "\n".join(lines)


def _pseudo_state(t: Table) -> GameState:
    """A GameState carrying the table's public facts.

    Hidden fields hold placeholders: at a real table their true values are
    unknown, and each rollout overwrites them from its sampled world via
    ``_world_to_state``.

    **The placeholders are not entirely unread**, and assuming otherwise
    caused a live bug.  ``legal_actions`` inspects the deck top to enumerate
    Chancellor return-orders, so it saw the placeholder Guards and offered
    "returning [Guard, Princess]" for cards the user had never drawn.  The
    evaluator now collapses Chancellor variants to a single option, which is
    correct for a second reason -- the return is chosen *after* the card is
    played -- but the lesson stands: check what reads these before trusting
    them.  Slot *ids* are the table's real ones, because Chancellor-pinned
    cards are addressed by id.
    """
    hands: list[list[Card]] = []
    for p in range(t.n_players):
        if p == t.me:
            hands.append(list(t.my_hand))
        elif t.seats[p].out:
            hands.append([])
        else:
            hands.append([Card.GUARD])  # placeholder; resampled per rollout
    state = state_from_hands(
        hands,
        [Card.GUARD] * len(t.deck_slots),  # placeholders; see the docstring
        Card.GUARD if t.set_aside_available else None,
        config=t.config,
        faceup=t.faceup,
        current=t.me,
        tokens=[s.tokens for s in t.seats],
        discards=[list(s.discards) for s in t.seats],
        out=[s.out for s in t.seats],
        protected=[s.protected for s in t.seats],
    )
    return _dc_replace(
        state, slots=tuple(t.deck_slots), next_slot=t.next_slot
    )


def render_recommendation(
    t: Table, *, budget: float = 1.4, rng: random.Random | None = None
) -> str:
    """The advice, built to the per-player-count validation tier."""
    if t.round_over:
        return "the round is over"
    if t.pending_chancellor:
        if t.pending_drawn is None:
            return "enter the cards you drew first: drew <card> [<card>]"
        return _chancellor_recommendation(t, budget=budget, rng=rng)
    if t.current != t.me or t.awaiting_draw:
        return "not your decision point (draw first if it is your turn)"

    note = VALIDATION_NOTE.get(t.n_players, "")
    if t.n_players == 2:
        return _belief_recommendation(t) + f"\n  [{note}]"

    state = _pseudo_state(t)
    rec = evaluate(
        state,
        t.me,
        belief=t.belief(),
        rng=rng or random.Random(),
        budget_seconds=budget,
        config=t.config,
    )
    return rec.explain() + f"\n  [{note}]"


def _chancellor_recommendation(
    t: Table, *, budget: float = 1.4, rng: random.Random | None = None
) -> str:
    """Rank the keep options of a pending Chancellor.

    The drawn cards are known here, so this is a real ranking over a real
    decision -- unlike before the play, where the return-order is
    deliberately not offered (the cards have not been seen yet).
    """
    from .evaluator import evaluate_chancellor

    drawn = list(t.pending_drawn or [])
    pool = list(t.my_hand) + drawn

    # Mid-resolution pseudo state: the Chancellor goes back into the hand so
    # legal_actions can enumerate return-orders, and the deck top carries the
    # real drawn cards rather than placeholders.
    hands: list[list[Card]] = []
    for p in range(t.n_players):
        if p == t.me:
            hands.append(list(t.my_hand) + [Card.CHANCELLOR])
        elif t.seats[p].out:
            hands.append([])
        else:
            hands.append([Card.GUARD])  # placeholder; resampled per rollout
    deck = drawn + [Card.GUARD] * (len(t.deck_slots) - len(drawn))
    state = state_from_hands(
        hands, deck,
        Card.GUARD if t.set_aside_available else None,
        config=t.config, faceup=t.faceup, current=t.me,
        tokens=[x.tokens for x in t.seats],
        discards=[list(x.discards) for x in t.seats],
        out=[x.out for x in t.seats],
        protected=[x.protected for x in t.seats],
    )
    state = _dc_replace(state, slots=tuple(t.deck_slots), next_slot=t.next_slot)

    rec = evaluate_chancellor(
        state, t.me, drawn,
        belief=t.belief(), rng=rng or random.Random(),
        budget_seconds=budget,
    )

    def label(action) -> str:
        ret = list(action.chancellor_return or ())
        held = list(pool)
        for c in ret:
            held.remove(c)
        kept = held[0] if held else "?"
        back = ", ".join(str(c) for c in ret)
        return f"keep {kept}, return [{back}]"

    lines = []
    top = rec.values[0]
    lines.append(
        f"RECOMMEND  {label(top.action)}   "
        f"{top.win_rate:.3f} +/- {top.ci:.3f} tokens"
    )
    if not rec.conclusive and rec.tied:
        names = ", ".join(label(v.action) for v in rec.tied[:3])
        lines.append(
            f"  NOT SEPARATED from: {names}. Their intervals overlap -- "
            f"treat them as equivalent."
        )
    lines.append("")
    lines.append(f"  {'option':<34} {'E[tokens]':>9}  {'95% CI':>8}  n")
    for v in rec.values[:6]:
        lines.append(
            f"  {label(v.action):<34} {v.win_rate:>9.3f}  "
            f"{v.ci:>8.3f}  {v.rollouts}"
        )
    lines.append("")
    lines.append(
        f"  {rec.worlds_sampled} rollouts over {rec.belief_worlds} worlds "
        f"in {rec.seconds:.2f}s; your drawn cards are pinned as known"
    )
    lines.append("  resolve with: keep <card> ret <card> [<card>]  (top first)")
    return "\n".join(lines)


def _belief_recommendation(t: Table) -> str:
    """Heads-up: baseline-scored move, posterior-chosen Guard guess.

    This mirrors the agent the arena validated at +0.058 exactly: the Guard
    guess is a pure posterior argmax (never the baseline's visible-count
    bonus), and the card/target choice is the baseline's. Mixing the two --
    an earlier version let the count bonus outvote the posterior -- would
    present a different, unmeasured policy under the validated label.
    """
    state = _pseudo_state(t)
    actions = legal_actions(state)
    if len(actions) == 1:
        return f"FORCED    {actions[0]}"
    base = BaselineAgent()
    marg = t.belief().hand_marginals()

    guards = [a for a in actions if a.card is Card.GUARD and a.target is not None]
    others = [a for a in actions if not (a.card is Card.GUARD and a.target is not None)]

    best_guard = None
    if guards:
        best_guard = max(
            guards,
            key=lambda a: marg.get(a.target, {}).get(a.guess, 0.0),
        )
    best_other = max(others, key=lambda a: base._score(state, t.me, a), default=None)

    if best_guard is not None and (
        best_other is None
        or base._score(state, t.me, best_guard)
        >= base._score(state, t.me, best_other)
    ):
        chosen = best_guard
    else:
        chosen = best_other if best_other is not None else actions[0]

    lines = [f"RECOMMEND  {chosen}   (belief-derived)"]
    if chosen.card is Card.GUARD and chosen.target is not None:
        row = {
            c: p
            for c, p in marg.get(chosen.target, {}).items()
            if c is not Card.GUARD and p > 0
        }
        top = sorted(row.items(), key=lambda kv: -kv[1])[:4]
        lines.append(
            "  guess evidence on P%d: " % chosen.target
            + "  ".join(f"{c} {p:.0%}" for c, p in top)
        )
    return "\n".join(lines)


# ------------------------------------------------------------------ the REPL


#: Returned by :meth:`CLI.dispatch` when the user asked to leave. The REPL
#: stops on it; tests can assert on it without driving a real terminal.
EXIT = object()


class CLI:
    """Line-in, text-out. All state lives in the Table; this only parses."""

    def __init__(self, *, config: GameConfig = STANDARD) -> None:
        self.config = config
        self.table: Table | None = None
        #: Set when a quit was refused, so an immediate repeat confirms it.
        self._quit_armed = False

    def dispatch(self, line: str):
        """Handle one line. Returns text, or :data:`EXIT` to end the session."""
        try:
            return self._dispatch(line)
        except EntryError as e:
            return f"REJECTED: {e}"

    def _need_table(self) -> Table:
        if self.table is None:
            raise EntryError("no round in progress -- use `start`")
        return self.table

    def _dispatch(self, line: str) -> str:
        parts = line.split()
        if not parts:
            return ""
        cmd, args = parts[0].lower(), parts[1:]

        # Strip a trailing bang before matching, so `q!` is the `q` command
        # with force set rather than an unknown command.
        forced = cmd.endswith("!")
        bare = cmd.rstrip("!")
        if bare in ("q", "quit", "exit", "abort"):
            return self._quit(bare, force=forced or "-f" in args)
        # Any other command clears a pending quit confirmation: the user
        # carried on playing, so a later bare `q` should ask again rather
        # than silently inheriting the earlier arming.
        self._quit_armed = False

        if cmd == "drew":
            t = self._need_table()
            k = t.chancellor_drawn([parse_card(a) for a in args])
            hand = " + ".join(str(c) for c in t.my_hand + (t.pending_drawn or []))
            return (
                f"holding: {hand}\n"
                f"`r` for a keep recommendation, then "
                f"`keep <card> ret <card>{' <card>' if k == 2 else ''}` "
                f"(top first)"
            )

        if cmd == "keep":
            t = self._need_table()
            if not args:
                raise EntryError("usage: keep <card> ret <card> [<card>]")
            kept = parse_card(args[0])
            returned: list = []
            rest = args[1:]
            if rest and rest[0].lower() == "ret":
                returned = [parse_card(a) for a in rest[1:]]
            elif rest:
                raise EntryError("usage: keep <card> ret <card> [<card>]")
            else:
                # Only one card to give back: it is whatever was not kept.
                pool = list(t.my_hand) + list(t.pending_drawn or [])
                leftovers = list(pool)
                if kept in leftovers:
                    leftovers.remove(kept)
                if len(leftovers) != 1:
                    raise EntryError(
                        "two cards go back -- give the order: "
                        "keep <card> ret <card> <card> (top first)"
                    )
                returned = leftovers
            t.chancellor_resolve(kept, returned)
            out = [f"kept {kept}; hand: "
                   + "+".join(str(c) for c in t.my_hand)]
            if t.round_over and not t.pending_reveal:
                out.append("ROUND OVER -- winners "
                           + ", ".join(f"P{w}" for w in t.winners))
            elif t.pending_reveal:
                out.append("deck is out -- enter revealed hands: "
                           "end 1=<card> 2=<card> ...")
            return "\n".join(out)

        if cmd == "abandon":
            if self.table is None:
                raise EntryError("no round in progress")
            self.table = None
            return "round abandoned -- `start` to begin another"

        if cmd == "start":
            return self._start(args)
        if cmd in ("d", "draw"):
            t = self._need_table()
            t.draw(parse_card(args[0]) if args else None)
            mine = t.current == t.me
            return (
                f"you drew; hand: {'+'.join(str(c) for c in t.my_hand)}"
                if mine
                else f"P{t.current} drew"
            )
        if cmd in ("p", "play"):
            return self._play(args)
        if cmd in ("u", "undo"):
            self._need_table().undo()
            return "undone"
        if cmd in ("s", "show"):
            return render_table(self._need_table())
        if cmd in ("m", "marg"):
            return render_marginals(self._need_table())
        if cmd in ("r", "rec"):
            return render_recommendation(self._need_table())
        if cmd == "end":
            t = self._need_table()
            revealed = {}
            for token in args:
                pid, _, card = token.partition("=")
                revealed[int(pid.lstrip("pP"))] = parse_card(card)
            t.end_round(revealed)
            return (
                f"round over -- winners {', '.join('P%d' % w for w in t.winners)}\n"
                + render_table(t)
            )
        if cmd in ("h", "help", "?"):
            return __doc__ or ""
        raise EntryError(f"unknown command '{cmd}' (try `help`)")

    def _quit(self, cmd: str, *, force: bool) -> object:
        """Leave the advisor. Confirms once if a round is still live.

        Losing a half-entered round costs the user everything they have typed
        this hand, so a bare quit mid-round asks first. ``abort`` and a bang
        (``q!``) skip the question -- someone typing those has already
        decided.
        """
        live = self.table is not None and not self.table.round_over
        if live and not force and cmd not in ("abort",):
            if not self._quit_armed:
                self._quit_armed = True
                return (
                    "a round is still in progress -- press `q` again to quit, "
                    "`q!` to quit now, or `abandon` to drop the round and "
                    "keep the advisor open"
                )
        return EXIT

    def _start(self, args: list[str]) -> str:
        if not args:
            raise EntryError(
                "usage: start <players> seat <s> dealt <card> "
                "[faceup a b c] [first <p>]"
            )
        n = int(args[0])
        seat = dealt = first = None
        faceup: list[Card] = []
        i = 1
        while i < len(args):
            key = args[i].lower()
            if key == "seat":
                seat = int(args[i + 1]); i += 2
            elif key == "dealt":
                dealt = parse_card(args[i + 1]); i += 2
            elif key == "first":
                first = int(args[i + 1]); i += 2
            elif key == "faceup":
                i += 1
                while i < len(args) and args[i].lower() not in (
                    "seat", "dealt", "first"
                ):
                    faceup.append(parse_card(args[i])); i += 1
            else:
                raise EntryError(f"unexpected '{args[i]}'")
        if seat is None or dealt is None:
            raise EntryError("both `seat` and `dealt` are required")
        self.table = Table(
            n, seat, dealt, faceup=faceup,
            first_player=first if first is not None else 0,
            config=self.config,
        )
        return f"round started\n{render_table(self.table)}"

    def _play(self, args: list[str]) -> str:
        t = self._need_table()
        if not args:
            raise EntryError("usage: p <card> [target] [guess] [facts...]")
        card = parse_card(args[0])
        i = 1
        target = guess = None
        if i < len(args) and args[i].isdigit() and card in (
            Card.GUARD, Card.PRIEST, Card.BARON, Card.KING, Card.PRINCE
        ):
            target = int(args[i]); i += 1
        if card is Card.GUARD and target is not None and i < len(args):
            guess = parse_card(args[i]); i += 1

        kw: dict = {}
        while i < len(args):
            tok = args[i].lower()
            if tok == "hit":
                kw["hit"] = True; i += 1
            elif tok == "miss":
                kw["hit"] = False; i += 1
            elif tok == "tie":
                kw["baron_loser"] = None; i += 1
            elif tok == "out":
                kw["baron_loser"] = int(args[i + 1].lstrip("pP"))
                i += 2
                if i < len(args) and args[i].lower() not in _FACT_WORDS:
                    kw["baron_revealed"] = parse_card(args[i]); i += 1
            elif tok == "saw":
                kw["seen"] = parse_card(args[i + 1]); i += 2
            elif tok == "got":
                kw["king_got"] = parse_card(args[i + 1]); i += 2
            elif tok == "disc":
                kw["prince_discarded"] = parse_card(args[i + 1]); i += 2
            elif tok == "drew":
                kw["prince_drew"] = parse_card(args[i + 1]); i += 2
            elif tok == "kept":
                kw["chancellor_kept"] = parse_card(args[i + 1]); i += 2
            elif tok == "ret":
                i += 1
                returned = []
                while i < len(args) and args[i].lower() not in _FACT_WORDS:
                    returned.append(parse_card(args[i])); i += 1
                kw["chancellor_returned"] = returned
            elif tok == "rev":
                kw["revealed"] = parse_card(args[i + 1]); i += 2
            else:
                raise EntryError(f"unknown fact '{args[i]}'")

        t.play(card, target=target, guess=guess, **kw)
        if t.pending_chancellor:
            k = min(2, len(t.deck_slots))
            return (
                f"chancellor played -- draw {k} card"
                f"{'s' if k == 2 else ''} and enter them: "
                f"drew <card>{' <card>' if k == 2 else ''}"
            )
        out = ["logged"]
        if t.round_over and not t.pending_reveal:
            out.append(
                f"ROUND OVER -- winners "
                + ", ".join(f"P{w}" for w in t.winners)
            )
        elif t.pending_reveal:
            out.append(
                "deck is out -- enter revealed hands: end 1=<card> 2=<card> ..."
            )
        return "\n".join(out)


_FACT_WORDS = {
    "hit", "miss", "tie", "out", "saw", "got", "disc", "drew",
    "kept", "ret", "rev",
}


def main() -> None:  # pragma: no cover - interactive shell
    cli = CLI()
    print("love letter advisor -- `help` for commands, `q` to quit")
    while True:
        try:
            line = input("> ")
        except EOFError:  # ctrl-D: leave immediately
            print()
            return
        except KeyboardInterrupt:  # ctrl-C: cancel the line, stay in
            print("\n(interrupted -- `q` to quit)")
            continue
        out = cli.dispatch(line)
        if out is EXIT:
            return
        if out:
            print(out)


if __name__ == "__main__":  # pragma: no cover
    main()
