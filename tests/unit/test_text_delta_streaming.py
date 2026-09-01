"""Streaming sends the new text, not the transcript so far.

Each token used to re-broadcast the whole answer, so an N-token reply cost
O(N^2) bytes — around 800 MB for one long answer, to every connected client.
The client now appends, which only works if the server's increments reconstruct
exactly what the old whole-text events produced, step separators included.
"""

from __future__ import annotations

import random

import pytest

from rune.api.server import join_steps


def replay(deltas: list[str], step_at: set[int]) -> tuple[str, str, int]:
    """Run the increment scheme, returning (client buffer, expected, bytes sent)."""
    collected: list[str] = []
    starts: list[int] = []
    sent = ""
    client = ""
    sent_bytes = 0

    for i, delta in enumerate(deltas):
        if i in step_at:
            starts.append(len(collected))
        collected.append(delta)
        full = join_steps(collected, starts)
        if full.startswith(sent):
            chunk = full[len(sent):]
            sent = full
            if chunk:
                client += chunk
                sent_bytes += len(chunk)
        else:  # prefix changed — the server resends whole and the client replaces
            sent = full
            client = full
            sent_bytes += len(full)

    return client, join_steps(collected, starts), sent_bytes


class TestTheClientRebuildsTheSameText:
    @pytest.mark.parametrize(
        ("deltas", "step_at"),
        [
            (["Hello", " world", ". Done."], set()),
            (["Opening", " the page.", "Now", " searching."], {2}),
            (["   ", "  ", "Real text."], {2}),
            (["a", " ", " ", "b", "  ", "c"], {3}),
            (["  ", "start", " mid", " end"], set()),
            ([f"s{i} text " for i in range(12)], {3, 6, 9}),
            (["one", "  ", "three"], {1, 2}),
            (["only"], set()),
            ([], set()),
        ],
    )
    def test_known_shapes(self, deltas, step_at):
        client, expected, _ = replay(deltas, step_at)
        assert client == expected

    def test_randomised_streams(self):
        rng = random.Random(7)
        pieces = ["x", " ", "  ", "word ", "\n", "\n\n", "ab", ""]
        for _ in range(200):
            n = rng.randint(1, 60)
            deltas = [rng.choice(pieces) for _ in range(n)]
            step_at = set(rng.sample(range(1, n), k=min(4, n - 1))) if n > 1 else set()
            client, expected, _ = replay(deltas, step_at)
            assert client == expected, (deltas, step_at)

    def test_step_separators_survive(self):
        client, expected, _ = replay(["first", "second"], {1})
        assert client == "first\n\nsecond" == expected


class TestItIsActuallyCheaper:
    def test_bytes_grow_linearly_not_quadratically(self):
        deltas = ["tok " for _ in range(400)]
        _, expected, sent = replay(deltas, set())

        whole_text_cost = sum(
            len(join_steps(deltas[: i + 1], [])) for i in range(len(deltas))
        )

        assert sent == len(expected)
        assert sent * 50 < whole_text_cost


class TestStreamJoinerMatchesJoinSteps:
    """The joiner replaced join_steps on the streaming path, so its output has
    to be identical — the answer the user reads comes from it."""

    def _replay(self, deltas, step_at):
        from rune.api.server import StreamJoiner, join_steps

        collected, starts = [], []
        joiner = StreamJoiner()
        rebuilt = ""
        for i, delta in enumerate(deltas):
            if i in step_at:
                starts.append(len(collected))
                joiner.start_step()
            collected.append(delta)
            rebuilt += joiner.append(delta)
            assert joiner.text == join_steps(collected, starts)
        return rebuilt, join_steps(collected, starts)

    @pytest.mark.parametrize(
        ("deltas", "step_at"),
        [
            (["Hello", " world."], set()),
            (["Opening", " the page.", "Now", " searching."], {2}),
            (["   ", "  ", "Real text."], {2}),
            (["one", "  ", "three"], {1, 2}),
            (["  lead", " mid", " tail  "], set()),
            ([], set()),
        ],
    )
    def test_known_shapes(self, deltas, step_at):
        rebuilt, expected = self._replay(deltas, step_at)
        assert rebuilt == expected

    def test_randomised_streams(self):
        rng = random.Random(5)
        pieces = ["x", " ", "  ", "word ", "\n", "\n\n", "ab", "", "\t"]
        for _ in range(400):
            n = rng.randint(1, 40)
            deltas = [rng.choice(pieces) for _ in range(n)]
            step_at = set(rng.sample(range(n), k=min(4, n)))
            rebuilt, expected = self._replay(deltas, step_at)
            assert rebuilt == expected, (deltas, step_at)

    def test_a_whitespace_only_delta_emits_nothing(self):
        from rune.api.server import StreamJoiner

        joiner = StreamJoiner()
        joiner.append("hello")
        assert joiner.append("   ") == ""
        assert joiner.text == "hello"

    def test_held_whitespace_reappears_before_the_next_word(self):
        from rune.api.server import StreamJoiner

        joiner = StreamJoiner()
        joiner.append("hello")
        joiner.append(" ")
        assert joiner.append("world") == " world"
        assert joiner.text == "hello world"

    def test_an_empty_step_drops_out(self):
        from rune.api.server import StreamJoiner

        joiner = StreamJoiner()
        joiner.append("first")
        joiner.start_step()
        joiner.append("   ")
        joiner.start_step()
        assert joiner.append("third") == "\n\nthird"
        assert joiner.text == "first\n\nthird"

    def test_cost_does_not_grow_with_the_transcript(self):
        """The point of the rewrite: linear overall, not quadratic."""
        import time

        from rune.api.server import StreamJoiner

        def run(n):
            joiner = StreamJoiner()
            start = time.perf_counter()
            for _ in range(n):
                joiner.append("token ")
            return time.perf_counter() - start

        run(2000)  # warm
        small, large = run(2000), run(8000)
        # 4x the tokens must not cost anywhere near 16x (quadratic).
        assert large < small * 8
