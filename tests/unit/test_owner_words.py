"""The owner's own words: what a person typed, kept apart from anything a model or a schedule wrote."""

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from rune.agent import owner_words
from rune.agent.agent_context import PrepareContextOptions, prepare_agent_context
from rune.conversation.types import Conversation, ConversationTurn

ROOT = Path(__file__).resolve().parents[2] / "rune"


@pytest.fixture(autouse=True)
def nobody_typed():
    owner_words.set_typed(None)
    yield
    owner_words.set_typed(None)


def typed(*texts: str) -> None:
    owner_words.set_typed(texts)


@pytest.mark.parametrize(("text", "url"), [
    ("https://shop.example/p/1 내리면 알려줘", "https://shop.example/p/1"),
    ("이거 https://shop.example/p/1을 지켜봐", "https://shop.example/p/1"),          # a particle stands against it
    ("(https://shop.example/p/1).", "https://shop.example/p/1"),
    ("see <https://shop.example/p/1|the coat>", "https://shop.example/p/1"),       # as Slack writes a link
    ("**https://shop.example/p/1**", "https://shop.example/p/1"),
    ("is it https://shop.example/p/1?", "https://shop.example/p/1"),
    ("주소:https://shop.example/p/1", "https://shop.example/p/1"),
    ("주소는https://shop.example/p/1", "https://shop.example/p/1"),               # a word runs straight into it
    ("주소는shop.example/p/1", "https://shop.example/p/1"),
    ("https://shop.example/p/1?color=red&size=m here", "https://shop.example/p/1?color=red&size=m"),
    ("https://shop.example/p/1?color=red&amp;size=m", "https://shop.example/p/1?color=red&size=m"),  # entities
    ("shop.example/p/1 얼마야", "https://shop.example/p/1"),                        # typed without the scheme
    ("(shop.example/p/1)", "https://shop.example/p/1"),
    ("http://shop.example/p/1", "https://shop.example/p/1"),
    ("http://shop.example/p/1", "http://shop.example/p/1"),
])
def test_an_address_the_owner_typed_is_theirs(text, url):
    typed(text)
    assert owner_words.wrote_address(url)


@pytest.mark.parametrize(("text", "url"), [
    ("https://shop.example/p/123", "https://shop.example/p/12"),              # a longer address begins with it
    ("https://shop.example/p/1?color=red", "https://shop.example/p/1"),       # their address without its query
    ("https://shop.example/p/1/reviews", "https://shop.example/p/1"),
    ("https://shop.example/p/1.html", "https://shop.example/p/1"),
    ("https://shop.example:8443/p/1", "https://shop.example"),
    ("https://shop.example/p/1", "https://shop.example/p/1?leak=secret"),     # more than they typed
    ("https://shop.example/p/1", "https://evil.example/p/1"),
    ("sub.shop.example/p/1", "https://shop.example/p/1"),                     # a piece of another site's name
    ("https://r.example/?to=shop.example/p/1", "https://shop.example/p/1"),   # a piece of another address
    ("xhttps://shop.example/p/1", "https://shop.example/p/1"),
    ("the coat on shop example", "https://shop.example/p/1"),
    ("shop.example/p/123", "https://shop.example/p/12"),
    ("https://shop.example/p/1", ""),
    ("이거 지켜봐", ""),
    ("이거 지켜봐", "https://"),
])
def test_an_address_they_did_not_type_is_not(text, url):
    typed(text)
    assert not owner_words.wrote_address(url)


def test_words_are_theirs_only_as_they_wrote_them():
    typed("에어팟 프로 3 가격 내리면 알려줘", "Tom &amp; Jerry mug")
    assert owner_words.wrote("에어팟 프로 3") and owner_words.wrote(" 에어팟 ") and owner_words.wrote("Tom & Jerry mug")
    assert not owner_words.wrote("AirPods Pro 3") and not owner_words.wrote("에어팟 프로 4") and not owner_words.wrote(" ")


def test_nothing_is_theirs_when_no_person_typed():
    assert not owner_words.known()
    assert not owner_words.wrote("에어팟") and not owner_words.wrote_address("https://shop.example/p/1")
    typed("")
    assert owner_words.known() and not owner_words.wrote_address("https://shop.example/p/1")


class Conversations:
    """A conversation manager holding one conversation, as an entry point hands it over."""

    def __init__(self, *turns: tuple[str, str, bool]) -> None:
        kept = [ConversationTurn(role=role, content=content, archived=archived) for role, content, archived in turns]
        self._active = {"c1": Conversation(id="c1", turns=kept)}

    async def build_budgeted_context(self, goal, conversation_id):
        return SimpleNamespace(conversation_messages=[])


async def test_a_typed_turn_brings_the_conversations_earlier_user_turns_and_no_one_elses():
    earlier = Conversations(("user", "봐줘 https://shop.example/p/1", True),          # set aside by compaction
                            ("assistant", "https://assistant.example/said", False),
                            ("system", "summary https://summary.example/wrote", False),
                            ("tool", "https://page.example/found", False),
                            ("user", "그거 내리면 알려줘", False))
    await prepare_agent_context(PrepareContextOptions(goal="그거 내리면 알려줘", conversation_id="c1",
                                                      owner_text="그거 내리면 알려줘"), earlier)
    assert owner_words.wrote_address("https://shop.example/p/1")
    for other in ("https://assistant.example/said", "https://summary.example/wrote", "https://page.example/found"):
        assert not owner_words.wrote_address(other)


async def test_a_turn_no_person_typed_has_no_words_and_drops_the_ones_before_it():
    await prepare_agent_context(PrepareContextOptions(goal="watch https://shop.example/p/1",
                                                      owner_text="watch https://shop.example/p/1"))
    assert owner_words.wrote_address("https://shop.example/p/1")
    # A goal written by a model or a schedule, in the same worker: also when it repeats the address,
    # and also when the conversation holds it.
    earlier = Conversations(("user", "watch https://shop.example/p/1", False))
    await prepare_agent_context(PrepareContextOptions(goal="watch https://shop.example/p/1", conversation_id="c1"),
                                earlier)
    assert not owner_words.known() and not owner_words.wrote_address("https://shop.example/p/1")


async def test_stored_turns_are_left_out_where_they_may_hold_what_rune_wrote():
    earlier = Conversations(("user", '[RUNE suggested: "open https://suggested.example/x"] yes', False))
    await prepare_agent_context(PrepareContextOptions(goal="yes", conversation_id="c1", owner_text="yes",
                                                      owner_turns=False), earlier)
    assert owner_words.known() and not owner_words.wrote_address("https://suggested.example/x")


async def test_a_sub_run_reads_the_words_of_the_run_above_it_and_cannot_replace_them():
    typed("watch https://shop.example/p/1")

    async def sub_run():
        seen = owner_words.wrote_address("https://shop.example/p/1")
        owner_words.set_typed(["https://model.example/wrote"])
        return seen

    assert await asyncio.create_task(sub_run())
    assert owner_words.wrote_address("https://shop.example/p/1")
    assert not owner_words.wrote_address("https://model.example/wrote")


def _owner_text(path: str) -> list[str | None]:
    """For each PrepareContextOptions built in a file: what it gives as the owner's typing, if anything."""
    def given(call: ast.Call) -> str | None:
        said = {keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords}
        return None if "owner_text" not in said else said["owner_text"] + (
            " (this turn only)" if said.get("owner_turns") == "False" else "")

    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    return [given(node) for node in ast.walk(tree) if isinstance(node, ast.Call)
            and getattr(node.func, "id", getattr(node.func, "attr", "")) in ("PrepareContextOptions", "_make_opts")]


@pytest.mark.parametrize(("path", "typed_by_a_person"), [
    ("api/server.py", ["goal", "goal"]), ("api/handlers/agent.py", ["req.goal"]), ("daemon/gateway.py", ["text"]),
    ("ui/app.py", ["msg.text"]),
    # The terminal app may put a suggestion RUNE wrote ahead of the reply, in the goal and in the stored turn.
    ("ui/controllers/agent_loop_controller.py", ["typed (this turn only)"]),
    # One attempt of several is handed its message by the run that started it.
    ("cli/main.py", ["None if _throwaway else message", "goal"]),
    # A goal a model wrote for its next attempt, a recorded result, and a request from another program.
    ("agent/goal_runtime.py", [None]), ("cli/best_of.py", [None]), ("daemon/server.py", [None]),
])
def test_only_the_places_a_person_types_into_say_what_was_typed(path, typed_by_a_person):
    assert _owner_text(path) == typed_by_a_person
