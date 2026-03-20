"""HTML report generator for RUNE evaluation results.

Produces a self-contained HTML dashboard with probe results, cost
summaries, and optional trend data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from rune.evaluation.types import ProbeResult


@dataclass(slots=True)
class TrendEntry:
    """A single historical trend data-point."""

    timestamp: str = ""
    passed: int = 0
    failed: int = 0
    total: int = 0
    pass_rate: float = 0.0
    duration: float = 0.0
    probes: dict[str, bool] = field(default_factory=dict)


@dataclass(slots=True)
class ReportData:
    """Input data for the HTML reporter."""

    timestamp: datetime = field(default_factory=datetime.now)
    probe_results: list[ProbeResult] = field(default_factory=list)
    cost_summary: dict[str, Any] | None = None
    grading_results: list[dict[str, Any]] | None = None
    trend_data: list[TrendEntry] | None = None
    metadata: dict[str, Any] | None = None


# Section generators

def _probe_section(results: list[ProbeResult]) -> str:
    passed = sum(1 for r in results if r.success)
    failed = len(results) - passed
    rows = ""
    for r in results:
        status_cls = "success" if r.success else "error"
        status_txt = "PASS" if r.success else "FAIL"
        rows += (
            f"<tr>"
            f"<td>{r.probe_name}</td>"
            f'<td class="{status_cls}">{status_txt}</td>'
            f"<td>{r.score:.2f}</td>"
            f"<td>{r.duration_ms:.0f}ms</td>"
            f"</tr>\n"
        )
    return (
        '<div class="section"><h2>Probe Results</h2>'
        '<div class="grid">'
        f'<div class="card"><div class="card-label">Passed</div>'
        f'<div class="card-value success">{passed}</div></div>'
        f'<div class="card"><div class="card-label">Failed</div>'
        f'<div class="card-value error">{failed}</div></div>'
        f'<div class="card"><div class="card-label">Total</div>'
        f'<div class="card-value">{len(results)}</div></div>'
        "</div>"
        "<table><thead><tr>"
        "<th>Probe</th><th>Status</th><th>Score</th><th>Duration</th>"
        "</tr></thead><tbody>"
        f"{rows}</tbody></table></div>"
    )


def _cost_section(cost: dict[str, Any]) -> str:
    items = "".join(
        f'<div class="card"><div class="card-label">{k}</div>'
        f'<div class="card-value">{v}</div></div>'
        for k, v in cost.items()
    )
    return f'<div class="section"><h2>Cost Summary</h2><div class="grid">{items}</div></div>'


# Public API

_CSS = """\
:root {
  --bg-primary: #0f172a;
  --bg-secondary: #1e293b;
  --text-primary: #f1f5f9;
  --text-secondary: #94a3b8;
  --accent: #3b82f6;
  --success: #22c55e;
  --warning: #eab308;
  --error: #ef4444;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: var(--bg-primary); color: var(--text-primary);
  line-height: 1.6; padding: 2rem;
}
.container { max-width: 1200px; margin: 0 auto; }
header { margin-bottom: 2rem; padding-bottom: 1rem; border-bottom: 1px solid var(--bg-secondary); }
h1 { font-size: 2rem; margin-bottom: 0.5rem; }
.timestamp { color: var(--text-secondary); font-size: 0.875rem; }
.section { background: var(--bg-secondary); border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem; }
.section h2 { font-size: 1.25rem; margin-bottom: 1rem; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; }
.card { background: var(--bg-primary); border-radius: 6px; padding: 1rem; }
.card-label { color: var(--text-secondary); font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }
.card-value { font-size: 1.5rem; font-weight: 600; margin-top: 0.25rem; }
.card-value.success, .success { color: var(--success); }
.card-value.warning { color: var(--warning); }
.card-value.error, .error { color: var(--error); }
table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
th, td { text-align: left; padding: 0.75rem; border-bottom: 1px solid var(--bg-primary); }
th { color: var(--text-secondary); font-weight: 500; font-size: 0.75rem; text-transform: uppercase; }
"""


def generate_html_report(data: ReportData) -> str:
    """Generate a self-contained HTML evaluation report.

    Parameters:
        data: Report input data.

    Returns:
        A complete HTML document as a string.
    """
    probe_html = _probe_section(data.probe_results) if data.probe_results else ""
    cost_html = _cost_section(data.cost_summary) if data.cost_summary else ""
    ts = data.timestamp.strftime("%Y-%m-%d %H:%M:%S")

    return (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        "<title>RUNE Evaluation Report</title>\n"
        f"<style>{_CSS}</style>\n"
        "</head>\n<body>\n"
        '<div class="container">\n'
        f"<header><h1>RUNE Evaluation Report</h1>"
        f'<div class="timestamp">{ts}</div></header>\n'
        f"{probe_html}\n{cost_html}\n"
        "</div>\n</body>\n</html>"
    )
