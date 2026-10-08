"""Small workspaces with independently checkable outcomes."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

MONEY = '''from decimal import Decimal, ROUND_HALF_EVEN

def line_total(price, quantity):
    return (Decimal(price) * quantity).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN)
'''

INVOICES = '''from decimal import Decimal
from money import line_total

def invoice_total(items):
    return sum((line_total(row["price"], row["quantity"]) for row in items), Decimal("0.00"))
'''

CHECKS = '''import unittest
from decimal import Decimal
from money import line_total
from invoices import invoice_total

class InvoiceTests(unittest.TestCase):
    def test_half_cent(self):
        self.assertEqual(line_total("1.005", 1), Decimal("1.01"))
    def test_negative_half_cent(self):
        self.assertEqual(line_total("-1.005", 1), Decimal("-1.01"))
    def test_regular_line(self):
        self.assertEqual(line_total("12.30", 2), Decimal("24.60"))
    def test_cancelled_line(self):
        self.assertEqual(invoice_total([{"price": "99.00", "quantity": 1, "status": "cancelled"}]), Decimal("0.00"))
    def test_round_each_line(self):
        self.assertEqual(invoice_total([{"price": "1.005", "quantity": 1, "status": "paid"}] * 2), Decimal("2.02"))
    def test_empty_invoice(self):
        self.assertEqual(invoice_total([]), Decimal("0.00"))
    def test_refund(self):
        self.assertEqual(invoice_total([{"price": "-5.20", "quantity": 1, "status": "paid"}]), Decimal("-5.20"))
'''

EXPENSES = '''id,department,amount,status
R01,engineering,1200,approved
R02,sales,850,approved
R03,engineering,-200,approved
R04,support,450,approved
R05,sales,150,approved
R05,sales,150,approved
R06,support,900,pending
R07,engineering,300,rejected
R08,support,50,approved
'''

POLICY = """# Monthly expense policy
Use only approved rows. Count each receipt id once; duplicate ids contain identical data.
Negative approved amounts are refunds and reduce the department total.
Do not change either input file. Currency is KRW.
"""

BOOKING_HTML = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Team room booking</title>
<h1>Team room booking</h1>
<nav><button role="tab" onclick="show('availability')">Availability</button>
<button role="tab" onclick="show('draft')">Booking draft</button></nav>
<section id="availability"><h2>Available rooms</h2>
<table><tr><th>Room</th><th>Capacity</th><th>Projector</th><th>Time</th></tr>
<tr><td>Birch</td><td>4</td><td>No</td><td>10:00</td></tr>
<tr><td>Cedar</td><td>8</td><td>Yes</td><td>10:00</td></tr>
<tr><td>Pine</td><td>12</td><td>Yes</td><td>14:00</td></tr></table></section>
<section id="draft" hidden><h2>Booking draft</h2>
<label>Room <select id="room"><option>Birch</option><option>Cedar</option><option>Pine</option></select></label>
<label>Time <select id="time"><option>10:00</option><option>14:00</option></select></label>
<label>Attendees <input id="people" type="number" value="2"></label>
<label>Meeting title <input id="title" value=""></label>
<button onclick="document.querySelector('#confirm').showModal()">Review booking</button>
<p id="status" role="status">Not saved</p></section>
<dialog id="confirm"><h2>Confirm draft</h2><p>This saves a local practice draft only.</p>
<button onclick="save()">Save draft</button><button onclick="document.querySelector('#confirm').close()">Cancel</button></dialog>
<script>
function show(id){ for(const name of ['availability','draft']) document.getElementById(name).hidden = name !== id; }
async function save(){
 const data = Object.fromEntries(['room','time','people','title'].map(id => [id,document.getElementById(id).value]));
 const response = await fetch('/save',{method:'POST',body:JSON.stringify(data)});
 if(response.ok){ document.getElementById('status').textContent = 'Saved: '+JSON.stringify(data); document.getElementById('confirm').close(); }
}
</script></html>'''


@pytest.fixture
def booking_site():
    state = {"loads": 0, "saves": []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def do_GET(self):
            if self.path != "/booking":
                self.send_error(404)
                return
            state["loads"] += 1
            body = BOOKING_HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != "/save":
                self.send_error(404)
                return
            state["saves"].append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/booking", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
