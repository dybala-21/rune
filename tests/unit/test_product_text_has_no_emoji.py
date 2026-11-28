"""What RUNE shows people is written in words, with a geometric mark (✓ ◐ ○ ✗) at most.

The owner called a bell on every scheduled message "too AI-looking"; emoji status badges had been
turned down for the same reason before. Model-facing text is not shown and is left out.
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import rune
from rune.channels.telegram import TelegramAdapter

# Pictographs, the clock and warning signs, and the check/cross marks drawn as emoji.
_EMOJI = re.compile("[\U0001F300-\U0001FAFF⏰-⏺⌚⌛⚠-⛿✅❌❎]")
# Lines the model reads about past runs (rune.memory.manager), never shown to the owner.
_MODEL_ONLY = {"rune/memory/manager.py"}


def test_no_product_string_carries_an_emoji():
    root = Path(rune.__file__).parent
    found = [
        f"{path.relative_to(root.parent)}:{number}: {line.strip()}"
        for path in sorted(root.rglob("*.py"))
        if str(path.relative_to(root.parent)) not in _MODEL_ONLY
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if _EMOJI.search(line) and not line.lstrip().startswith("#")
    ]
    assert found == []


async def test_a_telegram_approval_for_a_shell_command_is_sent_as_valid_html():
    """The bot sends HTML; a command's "&&" or ">" left raw is text Telegram must refuse."""
    sent: list[dict] = []

    async def send_message(**kwargs):
        sent.append(kwargs)

    adapter = TelegramAdapter("123:abc")
    adapter._bot = SimpleNamespace(send_message=send_message)
    await adapter.send_approval("777", "cd app && make > build.log", "a1")
    (message,) = sent
    assert message["text"] == "<b>Approval required</b>\n\ncd app &amp;&amp; make &gt; build.log"
    labels = [button.text for row in message["reply_markup"].inline_keyboard for button in row]
    assert labels == ["Approve", "Always approve", "Deny"]
