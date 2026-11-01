"""Browser regression checks against Vite; all Azure and save requests are mocked.

Run Vite, then set RUNE_UI_TEST_URL=http://127.0.0.1:18893 when invoking pytest.
"""

import copy
import os

import pytest

BASE = os.environ.get("RUNE_UI_TEST_URL")
pytestmark = pytest.mark.skipif(not BASE, reason="Set RUNE_UI_TEST_URL to a local Vite server")


@pytest.fixture
def ui():
    playwright = pytest.importorskip("playwright.sync_api")
    state = {"azure": {
        "endpoint": "https://fixture.services.ai.azure.com/openai/v1", "endpointKey": "AZURE_API_BASE", "endpointScope": "project",
        "hasKey": True, "keyName": "AZURE_API_KEY", "keyScope": "project", "revision": "initial", "credentialRevision": "credentials",
        "deployments": {"daily": "Kimi-K2.7-Code", "reasoner": "gpt-6-sol"}, "pricing": None,
        "suggestedModels": ["Kimi-K2.7-Code", "DeepSeek-V4-Pro"],
    }, "activeModel": {"provider": "azure", "model": "daily", "source": "active"},
        "requests": [], "hold": False, "held": [], "save_failure": False, "errors": []}
    with playwright.sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1100, "height": 1000})
        page.on("pageerror", lambda error: state["errors"].append(str(error)))

        def respond(route):
            path = route.request.url.split("/api/v1/")[1]
            body = route.request.post_data_json or {}
            state["requests"].append((path, body))
            if path == "auth/bootstrap":
                return route.fulfill(json={})
            if path == "rpc":
                assert body["method"] == "config.get", "No split env/config writes should occur"
                return route.fulfill(json={"success": True, "data": {"azure": state["azure"], "activeModel": state["activeModel"]}})
            if path == "azure/settings":
                if body["revision"] != state["azure"]["revision"]:
                    return route.fulfill(status=409, json={"detail": "Azure settings changed elsewhere. Reload before trying again."})
                if state["save_failure"]:
                    return route.fulfill(status=500, json={"detail": "Could not save Azure settings. The original files and active settings were retained."})
                state["azure"] = {**state["azure"], "endpoint": body["endpoint"], "deployments": body["deployments"],
                                  "pricing": body["pricing"], "revision": state["azure"]["revision"] + "-saved"}
                if body.get("apiKey"):
                    state["azure"]["credentialRevision"] += "-rotated"
                if body.get("useDeployment"):
                    state["activeModel"] = {"provider": "azure", "model": body["useDeployment"], "source": "active"}
                return route.fulfill(json={"saved": True, "revision": state["azure"]["revision"],
                    "credentialRevision": state["azure"]["credentialRevision"], "activeDeployment": body.get("useDeployment")})
            if state["hold"]:
                state["held"].append(route)
                return
            if path == "azure/models":
                return route.fulfill(json={"ok": True, "models": ["Kimi-K2.7-Code", "Kimi-K2.7-Code-2026-06-12", "DeepSeek-V4-Pro", "gpt-6-sol"],
                                          "detail": "Endpoint and API key verified. This is the model catalog, not a list of your deployments."})
            assert path == "azure/test"
            return route.fulfill(json={"ok": True, "detail": "Deployment connected: Azure returned a complete text response."})

        page.route("**/api/v1/**", respond)
        page.goto(f"{BASE}/tests/fixtures/azureSettings.html")
        page.locator(".azure-settings > summary").click()
        yield page, state, playwright.expect
        assert not state["errors"]
        browser.close()


def saved_requests(state):
    return [body for path, body in state["requests"] if path == "azure/settings"]


def test_search_add_and_save_as_selected_model(ui):
    page, state, expect = ui
    page.get_by_role("button", name="Connect and fetch models", exact=True).click()
    page.get_by_label("Search model catalog").fill("deepseek")
    expect(page.get_by_role("status").filter(has_text="1 of 4 models")).to_be_visible()
    page.get_by_label("Matching models").select_option("DeepSeek-V4-Pro")
    page.get_by_role("button", name="Add selected model").click()
    name = page.get_by_label("Deployment name", exact=True).nth(2)
    expect(name).to_be_focused()
    expect(name).to_have_value("")
    name.fill("my-deepseek")
    page.get_by_role("button", name="Save and use my-deepseek", exact=True).click()
    expect(page.get_by_text("Saved. my-deepseek is selected for new runs.")).to_be_visible()
    expect(page.get_by_role("button", name="Save settings", exact=True)).to_be_disabled()
    assert len(saved_requests(state)) == 1
    assert saved_requests(state)[0]["useDeployment"] == "my-deepseek"
    assert saved_requests(state)[0]["deployments"]["my-deepseek"] == "DeepSeek-V4-Pro"
    expect(page.locator(".azure-deployment").nth(2).get_by_text("In use", exact=True)).to_be_visible()


def test_new_resource_and_invalid_rows_focus_the_precise_error(ui):
    page, state, expect = ui
    page.get_by_label("Resource endpoint", exact=True).fill("https://other.services.ai.azure.com/openai/v1")
    page.get_by_role("button", name="Save settings", exact=True).click()
    expect(page.locator("#azure-key")).to_be_focused()
    expect(page.locator("#azure-key")).to_have_attribute("aria-invalid", "true")
    assert not saved_requests(state)
    page.locator("#azure-key").fill("mock-key")
    page.get_by_label("Deployment name", exact=True).nth(1).fill("daily")
    page.get_by_role("button", name="Save settings", exact=True).click()
    expect(page.get_by_label("Deployment name", exact=True).nth(0)).to_be_focused()
    expect(page.locator('[aria-invalid="true"]')).to_have_count(2)
    assert not saved_requests(state)


def test_pricing_addition_polling_and_save_keep_valid_test_results(ui):
    page, state, expect = ui
    page.get_by_role("button", name="Connect and fetch models", exact=True).click()
    page.get_by_role("button", name="Test deployment daily", exact=True).click()
    page.get_by_role("button", name="Test deployment reasoner", exact=True).click()
    expect(page.get_by_text("Deployment connected:", exact=False)).to_have_count(2)
    second = page.locator(".azure-deployment").nth(1)
    second.get_by_text("Cost estimates (optional)", exact=True).click()
    second.get_by_label("Deployment pricing").select_option("global")
    expect(page.get_by_text("Deployment connected:", exact=False)).to_have_count(2)
    page.locator("#poll").click()
    expect(page.get_by_text("Deployment connected:", exact=False)).to_have_count(2)
    page.get_by_role("button", name="Save settings", exact=True).click()
    expect(page.get_by_text("Azure settings saved.", exact=True)).to_be_visible()
    expect(page.get_by_text("Deployment connected:", exact=False)).to_have_count(2)
    expect(page.get_by_label("Search model catalog")).to_be_visible()
    page.get_by_role("button", name="Add deployment manually").click()
    expect(page.get_by_text("Deployment connected:", exact=False)).to_have_count(2)
    page.get_by_label("Underlying model", exact=True).nth(0).fill("DeepSeek-V4-Pro")
    expect(page.get_by_text("Deployment connected:", exact=False)).to_have_count(1)


def test_test_can_be_cancelled_without_locking_other_inputs(ui):
    page, state, expect = ui
    state["hold"] = True
    page.get_by_role("button", name="Test deployment daily", exact=True).click()
    expect(page.get_by_text("Waiting for this deployment", exact=False)).to_be_visible()
    expect(page.get_by_label("Resource endpoint", exact=True)).to_be_enabled()
    expect(page.get_by_label("Underlying model", exact=True).nth(1)).to_be_enabled()
    page.get_by_role("button", name="Stop waiting", exact=True).click()
    expect(page.get_by_text("Stopped waiting.", exact=False)).to_be_visible()
    expect(page.get_by_role("button", name="Test deployment daily", exact=True)).to_be_enabled()
    for route in state["held"]:
        route.fulfill(json={"ok": True, "detail": "STALE RESPONSE"})
    expect(page.get_by_text("STALE RESPONSE", exact=True)).to_have_count(0)


def test_editing_credentials_cancels_and_discards_pending_catalog(ui):
    page, state, expect = ui
    state["hold"] = True
    page.get_by_role("button", name="Connect and fetch models", exact=True).click()
    expect(page.get_by_text("Checking the key", exact=False)).to_be_visible()
    page.locator("#azure-key").fill("replacement-key")
    expect(page.get_by_role("button", name="Connect and fetch models", exact=True)).to_be_enabled()
    for route in state["held"]:
        route.fulfill(json={"ok": True, "detail": "STALE RESPONSE", "models": ["stale"]})
    expect(page.get_by_text("STALE RESPONSE", exact=True)).to_have_count(0)
    expect(page.get_by_label("Search model catalog")).to_have_count(0)


def test_conflict_keeps_draft_and_reload_fetches_current_settings(ui):
    page, state, expect = ui
    page.get_by_label("Deployment name", exact=True).nth(0).fill("local-draft")
    state["azure"]["revision"] = "changed-elsewhere"
    state["azure"]["deployments"] = {"remote-model": "DeepSeek-V4-Pro"}
    page.get_by_role("button", name="Save settings", exact=True).click()
    expect(page.get_by_role("button", name="Reload saved settings", exact=True)).to_be_visible()
    expect(page.get_by_label("Deployment name", exact=True).nth(0)).to_have_value("local-draft")
    page.get_by_role("button", name="Reload saved settings", exact=True).click()
    expect(page.get_by_label("Deployment name", exact=True)).to_have_count(1)
    expect(page.get_by_label("Deployment name", exact=True)).to_have_value("remote-model")
    expect(page.get_by_role("button", name="Reload saved settings", exact=True)).to_have_count(0)
    page.get_by_role("button", name="Save and use remote-model", exact=True).click()
    expect(page.get_by_text("Saved. remote-model is selected for new runs.")).to_be_visible()
    assert saved_requests(state)[-1]["revision"] == "changed-elsewhere"


def test_failed_save_keeps_draft_and_does_not_issue_partial_writes(ui):
    page, state, expect = ui
    original = copy.deepcopy(state["azure"])
    state["save_failure"] = True
    page.locator("#azure-key").fill("replacement-key")
    page.get_by_role("button", name="Save settings", exact=True).click()
    expect(page.locator("#azure-save-error")).to_be_focused()
    expect(page.locator("#azure-save-error")).to_contain_text("original files")
    expect(page.locator("#azure-key")).to_have_value("replacement-key")
    assert state["azure"] == original and len(saved_requests(state)) == 1
    state["save_failure"] = False
    page.get_by_role("button", name="Save settings", exact=True).click()
    expect(page.get_by_text("Azure settings saved.", exact=True)).to_be_visible()
    expect(page.locator("#azure-key")).to_have_value("")


def test_mobile_layout_and_budget_block_are_actionable(ui, tmp_path):
    page, state, expect = ui
    page.set_viewport_size({"width": 375, "height": 812})
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    state["azure"]["deploymentTestBlocked"] = "Deployment tests are disabled by the configured cost limit. Connection checks remain available."
    state["azure"]["revision"] = "budget-changed"
    page.locator("#poll").click()
    expect(page.get_by_role("button", name="Test deployment daily", exact=True)).to_be_disabled()
    expect(page.locator("#azure-test-blocked")).to_be_visible()
    expect(page.get_by_role("button", name="Connect and fetch models", exact=True)).to_be_enabled()
    page.locator('.fixture-scroll').evaluate('element => element.scrollTop = 0')
    page.screenshot(path=str(tmp_path / "azure-settings-mobile.png"), full_page=True)
