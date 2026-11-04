"""NOL's observed room-card and read-only detail DOM adapter.

Selectors come from live DOM probes (2026-11-01), not natural-language intent
rules. Unknown layouts fail closed. No booking or coupon-claim controls are used.
"""

import hashlib
import json
import re
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urldefrag

from rune.lodging.rates import RateCard, RateRegion, locate

CARD_SELECTOR = 'div.cursor-pointer.select-none:has(span.typography-subtitle-20-bold):has(button.underline)'
MODAL_SELECTOR = '[mode="overflow"]:visible'

# Read rendered nodes and their hierarchy in one browser turn. Preserve <s> and
# computed line-through separately from the current price, including its label.
RATE_DOM = r"""() => {
 const visible = e => !!e && e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true});
 const struck = e => {for(let n=e;n;n=n.parentElement) {
   if(n.matches('s,del') || getComputedStyle(n).textDecorationLine.includes('line-through')) return true;
 } return false;};
 return {text:document.body.innerText, rooms:[...document.querySelectorAll('[data-room-type-id]')]
 .filter(visible).map(g => ({room_id:g.dataset.roomTypeId, room:g.querySelector('h2')?.innerText || '',
 cards:[...g.querySelectorAll('div.cursor-pointer.select-none')].filter(c =>
   c.querySelector('button.underline') && c.querySelector('span.typography-subtitle-20-bold'))
 .map((c,index) => {
   const prices=[...c.querySelectorAll('span.typography-subtitle-20-bold')].filter(visible);
   const p=prices.length===1 ? prices[0] : null;
   const priceGroup=p?.parentElement?.parentElement?.parentElement;
   const clone=c.cloneNode(true); clone.querySelectorAll('[role="tooltip"]').forEach(e=>e.remove());
   return {index,text:c.innerText,signature:clone.textContent,
     price_text:p?.innerText || '', price_label:priceGroup?.innerText || '',
     struck:!p || struck(p), struck_prices:[...c.querySelectorAll('s,del')].filter(visible).map(e=>e.innerText)};
 })}))};
}"""


async def capture_rates(page: Any) -> tuple[str, list[RateCard]]:
    data = await page.evaluate(RATE_DOM)
    text, rates, cursor = data["text"], [], 0
    for group in data["rooms"]:
        if not group["room"] or not group["cards"]:
            continue
        room_lines, room_end = locate(text, group["room"], cursor)
        cursor = room_end
        for card in group["cards"]:
            lines, cursor = locate(text, card["text"], cursor)
            if card["struck"] or not re.fullmatch(r"\d{1,3}(?:,\d{3})*|\d+", card["price_text"]):
                continue
            signature = hashlib.sha256(json.dumps({"url": page.url, "room": group["room"],
                "room_id": group["room_id"], "index": card["index"], "text": card["signature"]},
                ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            rates.append(RateCard(id=signature, room_id=group["room_id"], index=card["index"],
                room=group["room"], room_lines=room_lines, lines=lines, text=card["text"],
                current_price=int(card["price_text"].replace(",", "")), price_text=card["price_text"],
                price_label=card["price_label"], struck_prices=card["struck_prices"], signature=signature))
    return text, rates


async def inspect_rate(page: Any, ledger: Any, source_id: str, rate_id: str, document_id: str) -> list[Any]:
    """Bind each new policy panel to the revalidated card that opened it."""
    source = ledger.sources[source_id]
    selected = next((r for r in source.rates if r.id == rate_id), None)
    if selected is None or urldefrag(page.url).url != urldefrag(source.url).url:
        raise ValueError("Use a rate ID from capture on the current room-list URL")
    if await page.locator(MODAL_SELECTOR).count():
        raise ValueError("Close the existing dialog before inspecting a rate")
    captures = []
    for kind in ("price", "detail"):
        required = {"price_conditions_quote"} if kind == "price" else {"cancellation_quote", "taxes_quote"}
        available = {r.kind for s in ledger.sources.values() if s.fresh(datetime.now(UTC))
                     for r in s.regions if r.rate_id == rate_id}
        if required <= available:
            continue
        # Never reuse an ordinal after a rerender without verifying its identity.
        _, current = await capture_rates(page)
        if selected.id not in {r.id for r in current}:
            raise ValueError("Selected rate changed; capture the current room list again")
        # Room names may repeat across suppliers; the issued DOM room identity is authoritative.
        group = page.locator('[data-room-type-id]').locator(f'xpath=self::*[@data-room-type-id={json.dumps(selected.room_id)}]')
        card = group.locator(CARD_SELECTOR).nth(selected.index)
        trigger = card.locator('[aria-label="요금 정보"]' if kind == "price" else 'button.underline')
        if await trigger.count() != 1:
            raise ValueError(f"Selected rate has no unambiguous read-only {kind} control (matches={await trigger.count()})")
        await trigger.click(timeout=5000)
        modal = page.locator(MODAL_SELECTOR)
        try:
            await modal.wait_for(state="visible", timeout=5000)
            if await modal.count() != 1:
                raise ValueError("Multiple dialogs make the selected rate ambiguous")
            if kind == "price":
                await modal.locator('[data-simulator-state="success"]').wait_for(timeout=12000)
            else:
                await modal.locator('#cancelPolicy').wait_for(timeout=5000)
            # Capture body and structural sections atomically; no model-supplied associations.
            data = await modal.evaluate(r"""m => ({text:document.body.innerText, modal:m.innerText,
              sections:[...m.querySelectorAll('#cancelPolicy,#importantInfo')].map(e=>({selector:'#'+e.id,text:e.innerText})),
              name:m.querySelector('.line-clamp-1.text-text-neutral-sub')?.innerText || ''})""")
            if urldefrag(page.url).url != urldefrag(source.url).url:
                raise ValueError("Navigation during rate inspection invalidated the evidence")
            offset = data["text"].rfind(data["modal"].strip())
            if offset < 0:
                raise ValueError("Dialog is absent from the rendered source")
            regions = []
            if kind == "price":
                lines, _ = locate(data["text"], data["modal"], offset)
                regions.append(RateRegion(rate_id=rate_id, kind="price_conditions_quote", lines=lines, selector='[mode="overflow"]'))
            else:
                for section in data["sections"]:
                    lines, _ = locate(data["text"], section["text"], offset)
                    regions.append(RateRegion(rate_id=rate_id, kind="cancellation_quote" if section["selector"] == '#cancelPolicy'
                                              else "taxes_quote", lines=lines, selector=section["selector"]))
                if data["name"]:
                    lines, _ = locate(data["text"], data["name"], offset)
                    regions.append(RateRegion(rate_id=rate_id, kind="name", lines=lines,
                                              selector='.line-clamp-1.text-text-neutral-sub'))
            captures.append(ledger.capture(page.url, data["text"], document_id, regions=regions))
        finally:
            close = modal.locator('button[aria-label="닫기"]')
            if await close.count() == 1:
                await close.click(timeout=5000)
                await modal.wait_for(state="hidden", timeout=5000)
    return captures
