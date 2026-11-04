"""One request through Rune's real web runtime and the public NOL website.

No fixture prices, search responses, website HTML, model calls or browser actions
are mocked. The observer only records what the actual execution saw.
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest

from tests.e2e.lodging_checks import Offer, StayRequest, nol_url, verify_comparison
from tests.e2e.test_live_workflows import completed
from tests.e2e.test_live_workflows import workflow as workflow


@pytest.fixture
def nol_contexts(monkeypatch, request):
    from copy import deepcopy

    from rune.agent import browser_context

    original = browser_context.compact_browser_history
    directory = request.config.getoption("--live-report-dir")
    path = Path(directory).resolve() / ("context-" + uuid4().hex + ".json") if directory else None
    observations = []

    def compact(messages, **kwargs):
        result = original(messages, **kwargs)
        progress = kwargs.get("retrieval_progress") or {}
        observations.append({"before_chars": len(json.dumps(messages, ensure_ascii=False)),
                             "after_chars": len(json.dumps(result, ensure_ascii=False)),
                             "removed_messages": len(messages) - len(result),
                             "complete_properties": progress.get("complete_properties"),
                             "has_trusted_progress": bool(progress)})
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps({"observations": observations, "last_messages": deepcopy(result)},
                                           ensure_ascii=False))
            temporary.chmod(0o600)
            temporary.replace(path)
        return result

    monkeypatch.setattr(browser_context, "compact_browser_history", compact)
    return {"observations": observations, "path": str(path) if path else None}


@pytest.fixture
def nol_observations(monkeypatch, request):
    from rune.capabilities.browser.session import current_session
    from rune.capabilities.registry import CapabilityRegistry

    captures = []
    ledger_paths = set()
    original = CapabilityRegistry.execute
    from playwright.async_api import Locator

    from tests.e2e.lodging_dom_checks import TRIGGER, WITNESS
    original_click = Locator.click
    directory = request.config.getoption("--live-report-dir")
    evidence = Path(directory).resolve() / ("nol-evidence-" + uuid4().hex) if directory else None
    if evidence:
        evidence.mkdir(parents=True)

    async def execute(registry, name, params):
        result = await original(registry, name, params)
        if path := (result.metadata or {}).get("evidence_path"):
            ledger_paths.add(path)
        if name.startswith("browser_"):
            session = current_session()
            page = session.page
            if page is not None and not page.is_closed() and nol_url(page.url):
                capture = {"tool": name, "url": page.url, "success": result.success,
                           "output_chars": len(result.output or ""),
                           "profile": session.profile,
                           "observed_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat()}
                response = session.navigation_response
                if response is not None and response.url == page.url:
                    capture["http_status"] = response.status
                try:
                    async with asyncio.timeout(5):
                        capture["text"] = (await page.locator("body").inner_text())[:80_000]
                        capture["dom"] = await page.evaluate(WITNESS)
                        if evidence:
                            shot = evidence / f"{len(captures):03d}.png"
                            await page.screenshot(path=str(shot))
                            capture["screenshot"] = str(shot)
                except Exception as exc:
                    capture["capture_error"] = type(exc).__name__
                captures.append(capture)
        return result

    async def click(locator, *args, **kwargs):
        page = locator.page
        trigger = await locator.evaluate(TRIGGER) if nol_url(page.url) else None
        result = await original_click(locator, *args, **kwargs)
        if trigger and trigger["kind"] in {"price", "detail"}:
            panel = page.locator('[mode="overflow"]:visible')
            await panel.wait_for(timeout=5000)
            if trigger["kind"] == "price":
                await panel.locator('[data-simulator-state="success"]').wait_for(timeout=12000)
            else:
                await panel.locator('#cancelPolicy').wait_for(timeout=5000)
            capture = {"tool": "independent_click_observer", "url": page.url, "trigger": trigger,
                       "text": await page.locator('body').inner_text(), "dom": await page.evaluate(WITNESS),
                       "observed_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat()}
            if evidence:
                shot = evidence / f"{len(captures):03d}.png"
                await page.screenshot(path=str(shot))
                capture["screenshot"] = str(shot)
            captures.append(capture)
        return result

    monkeypatch.setattr(Locator, "click", click)
    monkeypatch.setattr(CapabilityRegistry, "execute", execute)
    yield captures
    if evidence:
        (evidence / "observations.json").write_text(json.dumps(captures, ensure_ascii=False, indent=2))
        for index, path in enumerate(sorted(ledger_paths)):
            (evidence / f"stay-ledger-{index}.json").write_text(Path(path).read_text())


def pytest_generate_tests(metafunc):
    if "workflow" in metafunc.fixturenames:
        extended = metafunc.config.getoption("--nol-budget") == "extended"
        metafunc.parametrize("workflow", [{"max_iterations": 60, "timeout_seconds": 600, "max_tokens": 6000,
                                           "cost_budget_usd": 8.0 if extended else 5.0,
                                           "token_budget_override": 2_500_000 if extended else 1_500_000}], indirect=True)


def test_nol_comparison_from_one_request(nol_observations, nol_contexts, workflow, request):
    from rune.config import get_config

    cfg = get_config()
    cfg.llm.cost_budget_usd = workflow.settings["cost_budget_usd"]
    cfg.llm.model_request_limit = 70
    check_in = datetime.now(ZoneInfo("Asia/Seoul")).date() + timedelta(days=14)
    overseas = request.config.getoption("--nol-destination") == "sapporo"
    stay = StayRequest(city="삿포로" if overseas else "서울",
                       city_labels=["Sapporo"] if overseas else ["Seoul"],
                       market="overseas" if overseas else "domestic",
                       check_in=check_in, check_out=check_in + timedelta(days=1),
                       adults=2, max_total_krw=200_000)
    browser = request.config.getoption("--nol-browser")
    cfg.browser.default_profile = browser
    workflow.settings = {**workflow.settings, "requested_browser": browser, "browser_profile": browser}
    opening = ("새 표시 브라우저를 browser_open으로 열어서 작업해줘. " if browser == "visible"
               else "관리형 브라우저를 browser_navigate로 열어서 작업해줘. ")
    entry = "https://nol.yanolja.com/"
    if overseas:
        entry += ("stay/overseas/TDP-125731?verticalCategory=PRODUCT_CATEGORY_GLOBAL_ACCOMMODATION"
                  f"&checkInDate={stay.check_in}&checkOutDate={stay.check_out}"
                  f"&rooms%5B0%5D%5Badult%5D={stay.adults}")
    fields = ", ".join(Offer.model_fields)
    goal = (
        opening + f"NOL({entry})에서 시작해서 {stay.city} 숙소 {stay.count}곳을 비교해줘. "
        f"체크인 {stay.check_in}, 체크아웃 {stay.check_out}, 성인 {stay.adults}명, 객실 1개, "
        f"전체 숙박기간의 표시 객실료가 객실당 {stay.max_total_krw:,}원 이하야. 별도 현장 세금은 분리해서 원문으로 표시하고 최종 결제 총액으로 단정하지 마. "
        "실제 브라우저에서 날짜와 인원을 적용하고 각 숙소 상세에서 객실명, 해당 기간 가격, "
        "세금/수수료 포함 여부와 쿠폰 조건, 취소 조건을 확인해줘. 예약·결제·로그인은 하지 마. "
        "첫 화면의 날짜 미지정 최저가나 검색 요약을 해당 날짜의 객실 가격으로 쓰지 마. "
        "최종 결과는 코드블록 없이 JSON만 반환해줘: {\"offers\": [...], \"limitations\": [...]}. "
        f"각 offers 항목의 필드는 {fields}야. total_krw는 정수로, url은 실제 열어 본 상세 URL로 써줘. "
        "*_quote는 해당 숙소 화면에서 그대로 읽은 연속된 원문이며 price_quote에 객실명과 가격을 함께 넣어줘. "
        "stay_quote에는 날짜, occupancy_quote에는 인원, location_quote에는 주소, "
        "cancellation_quote에는 취소 조건, price_conditions_quote에는 할인/가격 적용 조건, taxes_quote에는 세금/수수료 조건을 담아줘. "
        "browser_stay_evidence에 숙소·객실·요금제별 증거를 저장하고, 누락 항목만 조회해. "
        "evidence_refs는 그 도구가 발급한 필드별 원문 참조를 그대로 사용해. finish가 반환한 JSON을 최종 결과로 써줘. "
        "확인할 수 없는 후보는 넣지 말고 limitations에 이유를 남겨줘. "
        "접속 오류나 인증 요구가 나오면 무리하게 반복하거나 우회하지 말고 확인하지 못한 범위를 보고해줘."
    )
    wording = request.config.getoption("--nol-request")
    if wording == "short":
        goal = (f"NOL에서 {stay.check_in}~{stay.check_out} {stay.city} 숙소 3곳 비교해줘. "
                f"성인 {stay.adults}명, 객실 1개, 숙박요금 {stay.max_total_krw // 10000}만원 이하로 "
                "객실·가격·취소조건·할인조건·별도 세금까지 확인해줘. 예약은 하지 마.")
    workflow.settings = {**workflow.settings, "nol_request": wording, "request_chars": len(goal)}
    run = workflow.submit(goal, wait_seconds=630)
    answer = run.get("answer") or run.get("text") or ""
    checks = verify_comparison(answer, stay, nol_observations)
    http_errors = [{"url": page["url"], "status": page["http_status"]} for page in nol_observations
                   if page.get("http_status", 0) >= 400]
    checks.update({"request": stay.model_dump(mode="json"), "snapshots": nol_observations,
                   "contexts": nol_contexts,
                   "http_errors": http_errors,
                   "failure_stage": ("site_access" if http_errors and not checks["offers_checked"]
                                     else "comparison") if not checks["passed"] else None,
                   "one_request": len(workflow.runs) == 1,
                   "requested_browser": browser,
                   "requested_browser_used": any(page.get("profile") == browser for page in nol_observations),
                   "no_user_intervention": not any(workflow.runs[0][key] for key in (
                       "approved_fixture_edits", "denied_fixture_actions", "answered_fixture_questions"))
                       and not run.get("approval") and not run.get("question")})
    workflow.checks = checks
    assert checks["one_request"] and checks["no_user_intervention"], checks
    assert nol_observations, "No actual NOL browser observation was recorded"
    assert checks["requested_browser_used"], "The requested execution environment was not used"
    assert checks["passed"], checks["failures"]
    completed(run)
    request.node.live_passed = True
