import pytest

from rune.capabilities.browser.capabilities import (
    BrowserActParams,
    BrowserObserveParams,
    browser_act,
    browser_observe,
)
from rune.capabilities.browser.core import BrowserNavigateParams, browser_navigate
from tests.integration.test_browser_foundation import screen as screen
from tests.integration.test_browser_readback import site as site
from tests.integration.test_browser_receipts import act, check, start


@pytest.mark.parametrize("mode,status", [("saved", "pass"), ("fake", "fail")])
async def test_repeated_record_url_keeps_the_unchecked_response(screen, site, mode, status):
    url, state = site
    state["mode"] = mode
    evidence = await start("Set people to 3 and save", [
        {"requirement": "people to 3", "expected": "3"}, {"requirement": "save", "kind": "result"}])
    evidence.observe("browser_navigate", await browser_navigate(BrowserNavigateParams(url=url + "/form")))
    await act(evidence, action="type", selector="#people", value="3")
    await act(evidence, action="click", selector="#save")
    params = {"readbackUrl": url + "/stored", "checks": [check("#missing", "people to 3", "3", "value")]}
    evidence.prepare_tool("browser_observe", params)
    first = await browser_observe(BrowserObserveParams(**params))
    evidence.observe("browser_observe", first)
    report = first.metadata["independent_readback"]
    discovery = {"readbackUrl": url + "/stored"}
    evidence.prepare_tool("browser_observe", discovery)
    repeated = await browser_observe(BrowserObserveParams(**discovery))
    evidence.observe("browser_observe", repeated)
    assert repeated.metadata["independent_readback"]["snapshot"] == report["snapshot"]
    assert state["reads"] == state["posts"] == 1
    assert evidence.acceptance()["status"] != "pass"
    params["checks"] = [check(report["fields"][0]["selector"], "people to 3", "3", "value")]
    evidence.prepare_tool("browser_observe", params)
    checked = await browser_observe(BrowserObserveParams(**params))
    evidence.observe("browser_observe", checked)
    assert checked.metadata["independent_readback"]["source"] == "readback_snapshot"
    assert evidence.acceptance()["status"] == status
    assert state["reads"] == state["posts"] == 1


@pytest.mark.parametrize("change", ["input", "document", "expiry", "url", "result"])
async def test_implicit_reuse_requires_the_same_unfinished_read(screen, site, change):
    import time

    from rune.capabilities.browser.session import current_session

    url, state = site
    await browser_navigate(BrowserNavigateParams(url=url + "/form"))
    params = {"readbackUrl": url + "/stored", "resultRequirement": "save", "checks": [
        check("#missing", "people to 3", "3", "value")]}
    await browser_observe(BrowserObserveParams(**params))
    state["saved"] = "3"
    if change == "input":
        await browser_act(BrowserActParams(action="type", selector="#people", value="3"))
    elif change == "document":
        await screen.reload()
    elif change == "expiry":
        current_session().readback_document["created"] = time.monotonic() - 61
    elif change == "url":
        params["readbackUrl"] = url + "/form"
    else:
        params["resultRequirement"] = "another save"
    params["checks"] = [check("#people" if change == "url" else "#stored", "people to 3", "3", "value")]
    result = await browser_observe(BrowserObserveParams(**params))
    report = result.metadata["independent_readback"]
    assert report.get("source") != "readback_snapshot"
    assert report["status"] == ("fail" if change == "url" else "pass")
    assert state["reads"] == (1 if change == "url" else 2)


@pytest.mark.parametrize("stored", ["1", "3"])
async def test_completed_field_comparison_does_not_mask_a_later_read(screen, site, stored):
    url, state = site
    state["saved"] = stored
    await browser_navigate(BrowserNavigateParams(url=url + "/form"))
    params = BrowserObserveParams(readbackUrl=url + "/stored", checks=[check("#stored", "people to 3", "3", "value")])
    first = await browser_observe(params)
    assert first.metadata["independent_readback"]["status"] == ("pass" if stored == "3" else "fail")
    state["saved"] = "2"
    second = await browser_observe(params)
    assert second.metadata["independent_readback"]["status"] == "fail"
    assert second.metadata["independent_readback"]["checks"][0]["actual"] == "2"
    assert state["reads"] == 2
