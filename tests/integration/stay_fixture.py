"""Small adversarial NOL-shaped DOM, with two rates for the same room."""

HTML = r'''<meta charset="utf-8"><style>div,h2,p,section,button {display:block} .price {display:flex} [hidden]{display:none}</style>
<pre>Example Sapporo Hotel
2026.11.15 to 2026.11.16
1 room, 2 adults, 0 children
Sapporo, Example street</pre>
<div data-room-type-id="room-a"><h2>Double room</h2>
<div class="cursor-pointer select-none"><div>Free cancellation
<button class="underline" onclick="detail(0)">Details</button></div><div><div>
<s>100,000</s><div><span>Member</span><div class="price"><div><span class="typography-subtitle-20-bold">90,000</span><span>KRW</span></div>
<span aria-label="요금 정보" onclick="price(0)">Price info</span></div></div></div></div></div>
<div class="cursor-pointer select-none"><div>Non-refundable
<button class="underline" onclick="detail(1)">Details</button></div><div><div>
<s>110,000</s><div><span>Member</span><div class="price"><div><span class="typography-subtitle-20-bold">95,000</span><span>KRW</span></div>
<span aria-label="요금 정보" onclick="price(1)">Price info</span></div></div></div></div></div>
</div><button onclick="window.bookings++">Book</button>
<script>
window.bookings=0; window.reads=0;
function modal(content) {window.reads++; location.hash='dialog'; const d=document.createElement('div');
 d.setAttribute('mode','overflow'); d.innerHTML='<button aria-label="닫기" onclick="this.parentElement.remove(); history.replaceState(null,\'\',location.pathname+location.search)">Close</button>'+content;
 document.body.append(d);}
function price(i) {modal('<div data-simulator-state="success">Product price '+(i ? '110,000':'100,000')+' KRW\nTaxes included\nMember coupon required and not owned\nDisplayed rate '+(i ? '95,000':'90,000')+' KRW</div>');}
function detail(i) {modal('<span class="line-clamp-1 text-text-neutral-sub">Example Sapporo Hotel</span><div id="cancelPolicy">'+(i ? 'Non-refundable' : 'Free cancellation until 2026-11-13 00:00<br>After that: non-refundable')+'</div><div id="importantInfo">Pay local accommodation tax at the property: JPY 600</div>');}
</script>'''
