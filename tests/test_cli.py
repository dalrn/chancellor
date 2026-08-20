"""Phase 4 tests: the CLI layer.

The Table underneath is oracle-tested; these tests cover what the CLI adds on
top: parsing, loud rejections that reach the user as text, the recommendation
tiers the interface must respect (belief-derived at 2p, search-ranked with
rollout counts and CIs at 3+), and undo through the command surface.
"""

from __future__ import annotations

import random
import unittest

from loveletter.cli import CLI, parse_card, render_recommendation
from loveletter.config import Card
from loveletter.table import EntryError, Table

C = Card


class TestParseCard(unittest.TestCase):
    def test_by_value(self) -> None:
        self.assertIs(parse_card("1"), C.GUARD)
        self.assertIs(parse_card("9"), C.PRINCESS)
        self.assertIs(parse_card("0"), C.SPY)

    def test_by_name_and_prefix(self) -> None:
        self.assertIs(parse_card("guard"), C.GUARD)
        self.assertIs(parse_card("co"), C.COUNTESS)
        self.assertIs(parse_card("K"), C.KING)
        self.assertIs(parse_card("b"), C.BARON)

    def test_ambiguous_prefix_is_loud_and_lists_options(self) -> None:
        with self.assertRaises(EntryError) as ctx:
            parse_card("pri")
        message = str(ctx.exception)
        self.assertIn("priest", message)
        self.assertIn("prince", message)
        self.assertIn("princess", message)

    def test_nonsense_is_loud(self) -> None:
        with self.assertRaises(EntryError):
            parse_card("zebra")
        with self.assertRaises(EntryError):
            parse_card("42")


class TestDispatch(unittest.TestCase):
    def _start_2p(self) -> CLI:
        cli = CLI()
        out = cli.dispatch(
            "start 2 seat 0 dealt guard faceup spy priest handmaid first 0"
        )
        self.assertIn("round started", out)
        return cli

    def test_full_turn_round_trip(self) -> None:
        cli = self._start_2p()
        self.assertIn("Guard+Baron", cli.dispatch("d baron"))
        self.assertEqual(cli.dispatch("p 1 1 chancellor miss"), "logged")
        self.assertIn("P1 drew", cli.dispatch("d"))
        self.assertEqual(cli.dispatch("p handmaid"), "logged")

    def test_rejections_reach_the_user_as_text(self) -> None:
        """EntryErrors become REJECTED lines, never tracebacks."""
        cli = self._start_2p()
        out = cli.dispatch("p 1 1 priest miss")  # playing before drawing
        self.assertTrue(out.startswith("REJECTED:"), out)
        out = cli.dispatch("d king")
        out = cli.dispatch("p 1 1 guard miss")  # Guard naming Guard
        self.assertTrue(out.startswith("REJECTED:"), out)
        out = cli.dispatch("nonsense")
        self.assertTrue(out.startswith("REJECTED:"), out)

    def test_rejected_entry_does_not_advance_the_table(self) -> None:
        cli = self._start_2p()
        cli.dispatch("d baron")
        before = list(cli.table.log.events)
        cli.dispatch("p 1 1 guard miss")  # rejected
        self.assertEqual(list(cli.table.log.events), before)

    def test_undo_through_the_cli(self) -> None:
        cli = self._start_2p()
        cli.dispatch("d baron")
        cli.dispatch("p baron 1 tie")
        n = len(cli.table.log.events)
        self.assertEqual(cli.dispatch("u"), "undone")
        self.assertLess(len(cli.table.log.events), n)
        # Re-enter differently: the mistyped tie was actually a loss.
        self.assertEqual(
            cli.dispatch("p baron 1 out 1 priest"), "logged\n"
            "ROUND OVER -- winners P0"
        )

    def test_no_round_in_progress_is_loud(self) -> None:
        cli = CLI()
        self.assertIn("REJECTED", cli.dispatch("d king"))
        self.assertIn("REJECTED", cli.dispatch("r"))

    def test_king_trade_records_my_side(self) -> None:
        cli = self._start_2p()
        cli.dispatch("d king")
        self.assertEqual(cli.dispatch("p king 1 got countess"), "logged")
        self.assertEqual(cli.table.my_hand, [C.COUNTESS])

    def test_chancellor_entry(self) -> None:
        cli = self._start_2p()
        cli.dispatch("d chancellor")
        out = cli.dispatch("p chancellor kept princess ret 2 5")
        self.assertEqual(out, "logged")
        self.assertEqual(cli.table.my_hand, [C.PRINCESS])

    def test_deck_out_flow(self) -> None:
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
        cli = CLI(config=tiny)
        cli.dispatch("start 2 seat 0 dealt guard first 0")
        cli.dispatch("d priest")
        cli.dispatch("p priest 1 saw baron")
        cli.dispatch("d")
        cli.dispatch("p 1 0 princess miss")
        cli.dispatch("d king")
        out = cli.dispatch("p king 1 got baron")
        self.assertIn("deck is out", out)
        out = cli.dispatch("end 1=guard")
        self.assertIn("winners P0", out)


class TestRecommendationTiers(unittest.TestCase):
    """The interface caveats, as behaviour rather than documentation."""

    def test_heads_up_is_belief_derived_never_search_ranked(self) -> None:
        table = Table(
            2, 0, C.GUARD,
            faceup=(C.SPY, C.PRIEST, C.HANDMAID), first_player=0,
        )
        table.draw(C.BARON)
        out = render_recommendation(table, rng=random.Random(1))
        self.assertIn("belief-derived", out)
        self.assertIn("+0.008", out, "the null result must be cited")
        self.assertNotIn("rollouts", out, "no search ranking at 2 players")
        self.assertNotIn("E[tokens]", out)

    def test_multiplayer_shows_rollout_counts_and_ci(self) -> None:
        table = Table(4, 0, C.GUARD, first_player=0)
        table.draw(C.BARON)
        out = render_recommendation(
            table, budget=0.3, rng=random.Random(1)
        )
        self.assertIn("rollouts", out, "rollout count must be visible")
        self.assertIn("95% CI", out, "the interval must be visible")
        self.assertIn("0.05s config", out, "validation config must be cited")

    def test_five_and_six_players_carry_their_warnings(self) -> None:
        t5 = Table(5, 0, C.GUARD, first_player=0)
        t5.draw(C.BARON)
        self.assertIn(
            "unmeasured",
            render_recommendation(t5, budget=0.2, rng=random.Random(1)),
        )
        t6 = Table(6, 0, C.GUARD, first_player=0)
        t6.draw(C.BARON)
        self.assertIn(
            "UNVALIDATED",
            render_recommendation(t6, budget=0.2, rng=random.Random(1)),
        )

    def test_not_your_turn_says_so(self) -> None:
        table = Table(3, 1, C.GUARD, first_player=0)
        self.assertIn("not your decision", render_recommendation(table))

    def test_heads_up_guess_evidence_never_lists_guard(self) -> None:
        """Guard cannot be named, so it must not appear as evidence."""
        table = Table(
            2, 0, C.GUARD,
            faceup=(C.SPY, C.PRIEST, C.HANDMAID), first_player=0,
        )
        table.draw(C.SPY)
        out = render_recommendation(table, rng=random.Random(1))
        if "guess evidence" in out:
            evidence = out.split("guess evidence", 1)[1].splitlines()[0]
            self.assertNotIn("Guard", evidence)


class TestPseudoState(unittest.TestCase):
    """The evaluator bridge must carry the table's real public facts."""

    def test_slot_ids_survive_into_the_pseudo_state(self) -> None:
        from loveletter.cli import _pseudo_state

        table = Table(3, 0, C.GUARD, first_player=0)
        table.draw(C.CHANCELLOR)
        table.play(
            C.CHANCELLOR, chancellor_kept=C.GUARD,
            chancellor_returned=(C.SPY, C.KING),
        )
        # Chancellor renumbered the bottom slots; the pseudo state must agree.
        state = _pseudo_state(table)
        self.assertEqual(tuple(table.deck_slots), state.slots)
        self.assertEqual(table.next_slot, state.next_slot)

    def test_public_flags_survive(self) -> None:
        from loveletter.cli import _pseudo_state

        table = Table(3, 2, C.GUARD, first_player=0)
        table.draw()
        table.play(C.HANDMAID)  # P0 protects themselves
        table.draw()
        # P1 must target me (P2): P0 is protected -- and the validator
        # enforces exactly that, which the first draft of this test tripped.
        table.play(C.GUARD, target=2, guess=C.PRINCESS, hit=False)
        state = _pseudo_state(table)
        self.assertTrue(state.player(0).protected)
        self.assertFalse(state.player(0).out)
        self.assertFalse(state.player(1).out)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestExit(unittest.TestCase):
    """Leaving mid-round must be possible, and must not be an accident."""

    S = "start 4 seat 0 dealt guard first 0"

    def _run(self, *lines):
        cli = CLI()
        out = None
        for line in lines:
            out = cli.dispatch(line)
        return cli, out

    def test_quit_with_no_round_leaves_at_once(self) -> None:
        from loveletter.cli import EXIT

        _, out = self._run("q")
        self.assertIs(out, EXIT)

    def test_quit_mid_round_asks_first(self) -> None:
        from loveletter.cli import EXIT

        _, out = self._run(self.S, "q")
        self.assertIsNot(out, EXIT)
        self.assertIn("still in progress", out)

    def test_second_quit_confirms(self) -> None:
        from loveletter.cli import EXIT

        _, out = self._run(self.S, "q", "q")
        self.assertIs(out, EXIT)

    def test_intervening_command_disarms_the_confirmation(self) -> None:
        """`q`, then carry on playing, then `q` -- must ask again."""
        from loveletter.cli import EXIT

        _, out = self._run(self.S, "q", "s", "q")
        self.assertIsNot(out, EXIT, "a stale confirmation quit the session")
        self.assertIn("still in progress", out)

    def test_bang_and_abort_skip_the_question(self) -> None:
        from loveletter.cli import EXIT

        for command in ("q!", "quit!", "exit!", "abort"):
            with self.subTest(command=command):
                _, out = self._run(self.S, command)
                self.assertIs(out, EXIT)

    def test_abandon_drops_the_round_but_stays_open(self) -> None:
        from loveletter.cli import EXIT

        cli, out = self._run(self.S, "abandon")
        self.assertIsNot(out, EXIT)
        self.assertIsNone(cli.table)
        self.assertIs(cli.dispatch("q"), EXIT, "quitting after abandon asks")

    def test_abandon_without_a_round_is_loud(self) -> None:
        _, out = self._run("abandon")
        self.assertIn("REJECTED", out)

    def test_a_finished_round_quits_without_asking(self) -> None:
        from loveletter.cli import EXIT

        cli = CLI()
        cli.dispatch("start 2 seat 0 dealt guard faceup spy priest handmaid")
        cli.dispatch("d baron")
        cli.dispatch("p 1 1 princess hit")  # opponent out, round over
        self.assertTrue(cli.table.round_over)
        self.assertIs(cli.dispatch("q"), EXIT)


class TestChancellorFlow(unittest.TestCase):
    """The two-step Chancellor, through the command surface.

    Before this existed the tool would recommend "play the Chancellor" and
    then reject `p chancellor` for want of cards the user had not yet drawn --
    it demanded the answer before accepting the question.
    """

    def _played(self) -> CLI:
        cli = CLI()
        cli.dispatch("start 4 seat 0 dealt king first 0")
        cli.dispatch("d chancellor")
        return cli

    def test_bare_play_is_accepted_and_asks_for_the_draw(self) -> None:
        cli = self._played()
        out = cli.dispatch("p chancellor")
        self.assertNotIn("REJECTED", out)
        self.assertIn("drew", out)
        self.assertTrue(cli.table.pending_chancellor)

    def test_recommendation_before_the_draw_asks_for_it(self) -> None:
        cli = self._played()
        cli.dispatch("p chancellor")
        self.assertIn("drew", cli.dispatch("r"))

    def test_recommendation_after_the_draw_ranks_keep_options(self) -> None:
        cli = self._played()
        cli.dispatch("p chancellor")
        cli.dispatch("drew princess guard")
        out = cli.dispatch("r")
        self.assertIn("keep", out)
        self.assertIn("E[tokens]", out)
        self.assertIn("pinned as known", out)

    def test_resolution_completes_the_turn(self) -> None:
        cli = self._played()
        cli.dispatch("p chancellor")
        cli.dispatch("drew princess guard")
        out = cli.dispatch("keep princess ret king guard")
        self.assertIn("kept Princess", out)
        self.assertEqual(cli.table.my_hand, [C.PRINCESS])
        self.assertFalse(cli.table.pending_chancellor)
        self.assertEqual(cli.table.current, 1, "the turn did not advance")

    def test_other_commands_are_blocked_mid_chancellor(self) -> None:
        cli = self._played()
        cli.dispatch("p chancellor")
        self.assertIn("REJECTED", cli.dispatch("d guard"))
        self.assertIn("REJECTED", cli.dispatch("p guard 1 spy miss"))

    def test_one_shot_entry_still_works(self) -> None:
        """The compact form stays available for a user who knows already."""
        cli = self._played()
        out = cli.dispatch("p chancellor kept king ret princess guard")
        self.assertEqual(out, "logged")
        self.assertEqual(cli.table.my_hand, [C.KING])

    def test_undo_returns_to_the_pending_state(self) -> None:
        cli = self._played()
        cli.dispatch("p chancellor")
        cli.dispatch("drew princess guard")
        cli.dispatch("u")
        self.assertTrue(cli.table.pending_chancellor)
        self.assertIsNone(cli.table.pending_drawn)
        self.assertIn("drew", cli.dispatch("r"))
