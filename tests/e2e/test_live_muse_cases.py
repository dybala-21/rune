"""Rune reproductions of Muse's public scenarios, not a Muse-engine A/B test.

Sources and exclusions: benchmarks/nol_comparison/rate-dom-research.json.
No purchase, message sending, login, or external calendar writes are authorized.
"""

import csv
import json
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import pytest

from tests.e2e.test_live_workflows import completed
from tests.e2e.test_live_workflows import workflow as workflow


@pytest.fixture
def public_observations(monkeypatch, request):
    from rune.capabilities.browser.session import current_session
    from rune.capabilities.registry import CapabilityRegistry

    original = CapabilityRegistry.execute
    captures = []
    directory = request.config.getoption('--live-report-dir')
    folder = Path(directory).resolve() / ('muse-cases-' + uuid4().hex) if directory else None

    async def execute(registry, name, params):
        result = await original(registry, name, params)
        if name.startswith('browser_'):
            page = current_session().page
            if page is not None and not page.is_closed() and page.url.startswith('https:'):
                capture = {'url': page.url, 'tool': name, 'success': result.success}
                try:
                    capture['text'] = await page.locator('body').inner_text(timeout=5000)
                    capture['title'] = await page.title()
                except Exception as exc:
                    capture['capture_error'] = type(exc).__name__
                captures.append(capture)
        return result

    monkeypatch.setattr(CapabilityRegistry, 'execute', execute)
    yield captures
    if folder:
        folder.mkdir(parents=True)
        (folder / 'observations.json').write_text(json.dumps(captures, ensure_ascii=False, indent=2))


@pytest.mark.parametrize('case', ['shopping', 'calendar'])
@pytest.mark.parametrize('workflow', [{'max_iterations':60, 'timeout_seconds':600, 'max_tokens':6000,
                                      'cost_budget_usd':5.0, 'token_budget_override':1_500_000}], indirect=True)
def test_muse_public_scenario(case, workflow, public_observations, request):
    from rune.config import get_config

    cfg = get_config()
    cfg.llm.cost_budget_usd = 5.0
    cfg.llm.model_request_limit = 70
    cfg.browser.default_profile = 'visible'
    if case == 'shopping':
        goal = ('브라우저로 Apple 한국 공식스토어에서 현재 판매하는 AirPods 3종을 비교해줘. '
                '상품명, 가격(원), 구매 URL, 가격 원문을 열로 airpods.csv에 저장해줘. 구매는 하지 마.')
        filename, domain = 'airpods.csv', 'apple.com'
    else:
        goal = ('브라우저로 서울대학교 공식 2026학년도 학사일정에서 3월의 첫 세 일정을 찾아줘. '
                '일정명, 시작일, 종료일, 출처 URL, 날짜 원문을 열로 calendar.csv에 저장해줘.')
        filename, domain = 'calendar.csv', 'snu.ac.kr'
    workflow.settings = {**workflow.settings, 'scenario': case, 'request_chars':len(goal), 'browser_profile':'visible',
                         'comparison': 'Rune reproduction of public Muse use case; Muse not executed'}
    run = workflow.submit(goal, wait_seconds=630)
    path = workflow.workspace / filename
    rows = list(csv.reader(path.open(encoding='utf-8-sig'))) if path.exists() else []
    failures = []
    if len(rows) != 4:
        failures.append(f'Expected header and 3 rows, got {len(rows)}')
    for row in rows[1:]:
        urls = [value for value in row if value.startswith('https://')]
        pages = [p for p in public_observations if p.get('text') and p['url'] in urls
                 and ((urlsplit(p['url']).hostname or '') == domain
                      or (urlsplit(p['url']).hostname or '').endswith('.'+domain))]
        if not pages:
            failures.append('Row has no independently visited official source: ' + str(row[:1]))
        # Preserve source quotations so later semantic review can check the
        # selected product/variant and calendar row, not a model self-rating.
        quote = row[-1] if row else ''
        if not quote or not any(' '.join(quote.split()) in ' '.join(p['text'].split()) for p in pages):
            failures.append('Exact source quotation missing: ' + str(row[:1]))
    if case == 'calendar' and len(rows) == 4:
        expected = [('제1학기 시작','2026-03-01','2026-03-01'),
                    ('제1학기 개강','2026-03-03','2026-03-03'),
                    ('제1학기 수강신청 변경','2026-03-03','2026-03-09')]
        if [tuple(r[:3]) for r in rows[1:]] != expected:
            failures.append('Calendar event names or date ranges do not match independently reviewed official rows')
    workflow.checks = {'scenario':case, 'passed':not failures, 'failures':failures, 'rows':rows,
                       'observations':public_observations, 'one_request':len(workflow.runs)==1,
                       'semantic_review_required':case=='shopping',
                       'no_user_intervention':not any(workflow.runs[0][key] for key in ('approved_fixture_edits','denied_fixture_actions','answered_fixture_questions')),
                       'scope':'Public browser research and local artifact only; no checkout or external calendar sync'}
    if path.exists() and request.config.getoption('--live-report-dir'):
        destination = Path(request.config.getoption('--live-report-dir')).resolve() / (uuid4().hex+'-'+filename)
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes(path.read_bytes())
        workflow.checks['artifact'] = str(destination)
    assert not failures, failures
    completed(run)
    assert len(workflow.runs) == 1 and workflow.checks['no_user_intervention']
    request.node.live_passed = True


@pytest.mark.parametrize('workflow', [{'max_iterations':10, 'timeout_seconds':150, 'max_tokens':1500,
                                      'cost_budget_usd':0.8, 'token_budget_override':120000}], indirect=True)
def test_muse_primary_source_audit(workflow, request):
    from rune.config import get_config

    get_config().llm.cost_budget_usd = 0.8
    run = workflow.submit('Read https://introducing.muse.ai/ and https://muse.ai/ . '
        'Identify the official shopping, restaurant, calendar and artifact scenarios. '
        'Audit whether reproducing product price comparison and public school calendar extraction in Rune '
        'tests all of those promises. State missing transaction, authentication, background and app integration '
        'coverage. Cite the official sources, separate examples from measured reliability. Under 300 words. '
        'Read-only research, no delegation or edits.', wait_seconds=180)
    workflow.checks = {'purpose':'Muse use-case primary-source audit', 'review_required':True}
    completed(run)
    request.node.live_passed = True
