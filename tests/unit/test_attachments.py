"""build_user_content must never send an image the model can't read, and never
drop one silently."""
import pytest

from rune.agent.attachments import (
    MAX_IMAGE_BASE64_BYTES,
    build_user_content,
    content_text,
)

IMG = "iVBORw0KGgo="


@pytest.mark.asyncio
async def test_no_attachments_returns_the_goal_unchanged():
    content, notes = await build_user_content("hello", [], vision=True)
    assert content == "hello"
    assert notes == []


@pytest.mark.asyncio
async def test_image_becomes_a_content_part_alongside_the_goal():
    content, notes = await build_user_content(
        "what is this",
        [{"name": "a.png", "mimeType": "image/png", "data": IMG}],
        vision=True,
    )
    assert notes == []
    assert [p["type"] for p in content] == ["text", "image_url"]
    assert content[0]["text"] == "what is this"
    assert content[1]["image_url"]["url"] == f"data:image/png;base64,{IMG}"


@pytest.mark.asyncio
async def test_model_without_vision_gets_text_and_the_user_is_told():
    content, notes = await build_user_content(
        "what is this",
        [{"name": "a.png", "mimeType": "image/png", "data": IMG}],
        vision=False,
    )
    assert isinstance(content, str)
    assert "can't read images" in notes[0]
    assert "a.png" in content


@pytest.mark.asyncio
async def test_non_image_is_named_but_not_sent_as_content():
    content, notes = await build_user_content(
        "summarise",
        [{"name": "d.pdf", "mimeType": "application/pdf", "data": IMG}],
        vision=True,
    )
    assert isinstance(content, str)
    assert "d.pdf" in content
    assert "not an image" in notes[0]


@pytest.mark.asyncio
async def test_oversized_image_is_refused_with_its_size():
    # Valid base64 (4-char groups of 'A' decode fine) so it reaches the size check.
    oversized = "A" * (MAX_IMAGE_BASE64_BYTES + 4 - (MAX_IMAGE_BASE64_BYTES % 4))
    content, notes = await build_user_content(
        "look",
        [{"name": "big.png", "mimeType": "image/png", "data": oversized}],
        vision=True,
    )
    assert isinstance(content, str)
    assert "big.png" in notes[0]
    assert "limit" in notes[0]


@pytest.mark.asyncio
async def test_empty_data_is_refused():
    _, notes = await build_user_content(
        "look", [{"name": "x.png", "mimeType": "image/png", "data": ""}], vision=True,
    )
    assert "empty file" in notes[0]


@pytest.mark.asyncio
async def test_good_and_bad_attachments_mix_without_losing_either():
    content, notes = await build_user_content(
        "compare",
        [
            {"name": "ok.png", "mimeType": "image/png", "data": IMG},
            {"name": "d.pdf", "mimeType": "application/pdf", "data": IMG},
        ],
        vision=True,
    )
    assert [p["type"] for p in content] == ["text", "image_url", "text"]
    # parts[0] must stay the bare goal — see test_note_does_not_change_the_goal_part.
    assert content[0]["text"] == "compare"
    assert "d.pdf" in content[2]["text"]
    assert len(notes) == 1


@pytest.mark.asyncio
async def test_content_text_reads_the_goal_back_out():
    # The adapter uses this to spot that the goal is already in history, so it
    # doesn't append a duplicate text-only copy on every step.
    content, _ = await build_user_content(
        "the goal", [{"name": "a.png", "mimeType": "image/png", "data": IMG}], vision=True,
    )
    assert content_text(content) == "the goal"
    assert content_text("plain") == "plain"
    assert content_text([]) == ""


# ── conversation history ──

class _FakeConvManager:
    def __init__(self):
        self.turns = []

    def add_turn(self, conversation_id, role, text):
        self.turns.append((conversation_id, role, text))


def test_user_turn_notes_attachments_without_storing_them():
    # History replays into every later turn, so the image itself must not be
    # kept — but a follow-up like "how many images?" has to know one was sent.
    from rune.api.conversation_wiring import record_user_turn

    cm = _FakeConvManager()
    record_user_turn(cm, "c1", "analyse this", [
        {"name": "shot.png", "mimeType": "image/png", "data": "A" * 5000},
    ])
    _, role, text = cm.turns[0]
    assert role == "user"
    assert text.startswith("analyse this")
    assert "shot.png" in text
    assert "A" * 100 not in text          # no image data in history


def test_user_turn_without_attachments_is_unchanged():
    from rune.api.conversation_wiring import record_user_turn

    cm = _FakeConvManager()
    record_user_turn(cm, "c1", "plain question")
    assert cm.turns[0][2] == "plain question"


@pytest.mark.asyncio
async def test_large_image_is_downscaled_before_sending():
    # A phone photo is far larger than any model's vision input resolution, so
    # sending it whole only buys latency and tokens.
    pytest.importorskip("PIL")
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (4000, 3000), (10, 20, 30)).save(buf, "JPEG", quality=90)
    original = base64.b64encode(buf.getvalue()).decode()

    content, notes = await build_user_content(
        "what is this",
        [{"name": "photo.jpg", "mimeType": "image/jpeg", "data": original}],
        vision=True,
    )
    assert notes == []
    sent = content[1]["image_url"]["url"].split(",", 1)[1]
    assert len(sent) < len(original) / 2, "large image should be shrunk substantially"


@pytest.mark.asyncio
async def test_corrupt_image_data_is_refused_before_the_request():
    # Forwarding undecodable bytes only buys a provider 400 mid-run; refuse it
    # here so the reason reaches the user instead.
    content, notes = await build_user_content(
        "look", [{"name": "x.png", "mimeType": "image/png", "data": "!!!not-base64!!!"}],
        vision=True,
    )
    assert isinstance(content, str)
    assert "corrupted" in notes[0]
    assert "x.png" in notes[0]


@pytest.mark.asyncio
async def test_note_does_not_change_the_goal_part():
    # The adapter decides "the goal is already in history" by comparing
    # content_text() to the goal. If the note were appended to parts[0] the
    # comparison would fail and a second, text-only copy of the goal would be
    # sent on every step.
    content, notes = await build_user_content(
        "the goal",
        [
            {"name": "ok.png", "mimeType": "image/png", "data": IMG},
            {"name": "d.pdf", "mimeType": "application/pdf", "data": IMG},
        ],
        vision=True,
    )
    assert notes
    assert content_text(content) == "the goal"


# ── context accounting ──

def test_an_image_is_not_counted_as_its_base64_length():
    # str()-ing multimodal content charges the base64 payload as text: a
    # downscaled screenshot scored >130k "tokens", blew every cap, and the
    # trimmer then dropped the image and the history around it every step.
    from rune.agent.loop import IMAGE_TOKEN_ESTIMATE, NativeAgentLoop

    big = "A" * 400_000
    msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{big}"}},
        ],
    }
    estimate = NativeAgentLoop._estimate_tokens(msg)
    assert estimate < IMAGE_TOKEN_ESTIMATE * 2
    assert estimate < 80_000, "must stay under the smallest context cap"


def test_plain_text_estimation_is_unchanged():
    from rune.agent.loop import NativeAgentLoop

    assert NativeAgentLoop._estimate_tokens({"role": "user", "content": "a" * 400}) == 100
