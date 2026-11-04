"""Independent browser witnesses: collect raw card and dialog nodes around clicks.

Does not import the production extractor or accept its self-reported bindings.
"""

import re
from urllib.parse import urldefrag

WITNESS = r"""() => {
 const visible=e=>e.checkVisibility({checkVisibilityCSS:true,checkOpacity:true});
 const cards=[...document.querySelectorAll('[data-room-type-id]')].flatMap(g=> {
   const buttons=[...g.querySelectorAll('button.underline')];
   return buttons.map((b,index)=>{
     const c=b.closest('.cursor-pointer.select-none');
     if(!c || !visible(c)) return null;
     const prices=[...c.querySelectorAll('span.typography-subtitle-20-bold')].filter(visible);
     return {room_id:g.getAttribute('data-room-type-id'),index,room:g.querySelector('h2')?.innerText,
       text:c.innerText, html:c.outerHTML,
       prices:prices.map(p=>({text:p.innerText, struck:!!p.closest('s,del') ||
          [...function*(e){while(e){yield e; e=e.parentElement}}(p)].some(e=>getComputedStyle(e).textDecorationLine.includes('line-through'))}))};
   }).filter(Boolean);
 });
 const dialogs=[...document.querySelectorAll('[mode="overflow"]')].filter(visible).map(m=>({
   text:m.innerText,html:m.outerHTML,cancellation:m.querySelector('#cancelPolicy')?.innerText,
   taxes:m.querySelector('#importantInfo')?.innerText,price_ready:!!m.querySelector('[data-simulator-state="success"]')}));
 return {cards,dialogs};
}"""

TRIGGER = r"""e => {
 const c=e.closest('.cursor-pointer.select-none'), g=e.closest('[data-room-type-id]');
 if(!c || !g) return null;
 const buttons=[...g.querySelectorAll('button.underline')];
 return {room_id:g.getAttribute('data-room-type-id'), index:buttons.indexOf(c.querySelector('button.underline')),
   room:g.querySelector('h2')?.innerText, text:c.innerText,
   kind:e.matches('button.underline') ? 'detail' : e.getAttribute('aria-label') === '요금 정보' ? 'price' : 'other'};
}"""


def verify_dom_rate(offer, snapshots):
    errors = []
    room_id, index = offer.dom_rate.get("room_id"), offer.dom_rate.get("index")
    if not room_id or type(index) is not int:
        return ["missing selected DOM room/rate identity"]
    relevant = [s for s in snapshots if urldefrag(s.get("url", "")).url == urldefrag(offer.url).url]
    cards = [c for s in relevant for c in s.get("dom", {}).get("cards", [])
             if c.get("room_id") == room_id and c.get("index") == index
             and c.get("room") == offer.room and c.get("text", "").strip() == offer.price_quote]
    if not any(len(c["prices"]) == 1 and not c["prices"][0]["struck"]
               and re.fullmatch(r"\d+(?:,\d{3})*", c["prices"][0]["text"])
               and int(c["prices"][0]["text"].replace(",", "")) == offer.total_krw for c in cards):
        errors.append("current price is not witnessed in the selected DOM card (strike/old price rejected)")
    for field, kind, section in (("cancellation_quote", "detail", "cancellation"),
                                 ("taxes_quote", "detail", "taxes"),
                                 ("price_conditions_quote", "price", "text")):
        ref = offer.evidence_refs.get(field, {})
        bound = False
        for snapshot in relevant:
            trigger = snapshot.get("trigger") or {}
            if (snapshot.get("url") != ref.get("url") or trigger.get("kind") != kind
                    or trigger.get("room_id") != room_id or trigger.get("index") != index
                    or trigger.get("room") != offer.room
                    or trigger.get("text", "").strip() != offer.price_quote):
                continue
            bound |= any((d.get(section) or "").strip() == getattr(offer, field)
                         and (kind != "price" or d.get("price_ready"))
                         for d in snapshot.get("dom", {}).get("dialogs", []))
        if not bound:
            errors.append(f"{field} lacks an independently witnessed selected-card click and entire policy region")
    return errors
