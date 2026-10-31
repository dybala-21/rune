"""Precision checks for generated invoice code."""

import shutil

import pytest

from tests.e2e.test_live_task_continuity import _check_invoice_precision

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is required")


@pytest.mark.parametrize("rounding", [None, 2, 10, 16])
def test_invoice_grader_rejects_fixed_scale_rounding(tmp_path, rounding):
    if rounding is not None:
        source = "export const total = rows => rows.filter(r => r.status === 'paid').reduce((s,r) => +(s+r.amount).toFixed(SCALE),0);".replace("SCALE", str(rounding))
    else:
        source = """
export function total(rows) {
  let sum = 0n, scale = 0;
  for (const row of rows.filter(row => row.status === 'paid')) {
    const [coefficient, exponent = '0'] = String(row.amount).split('e');
    const [whole, fraction = ''] = coefficient.split('.');
    let amount = BigInt(whole + fraction), digits = fraction.length - Number(exponent);
    if (digits < 0) { amount *= 10n ** BigInt(-digits); digits = 0; }
    if (digits > scale) { sum *= 10n ** BigInt(digits - scale); scale = digits; }
    sum += amount * 10n ** BigInt(scale - digits);
  }
  return Number(`${sum}e-${scale}`);
}
"""
    (tmp_path / 'invoice.mjs').write_text(source)
    if rounding is not None:
        with pytest.raises(AssertionError, match='Precision changed'):
            _check_invoice_precision(tmp_path)
    else:
        _check_invoice_precision(tmp_path)
