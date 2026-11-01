"""Real Chromium checks for identity-bound duplicate controls."""

import pytest

from rune.capabilities.browser.capabilities import BrowserActParams, browser_act
from rune.capabilities.browser.helpers import (
    extract_interactive_elements,
    find_element_locator,
    format_interactive_elements,
    get_element_store,
)
from tests.integration.test_browser_foundation import screen as screen


async def cards(page):
    await page.set_content('''
        <article id="alpha"><h2>Alpha room</h2><button onclick="window.chosen='alpha'">Details</button></article>
        <article id="beta"><h2>Beta room</h2><button onclick="window.chosen='beta'">Details</button></article>
        <script>window.chosen = ''</script>
    ''')
    return [meta for meta in await extract_interactive_elements(page) if meta.name == "Details"]


async def test_duplicate_controls_have_bound_nodes_and_context(screen):
    elements = await cards(screen)
    assert len(elements) == 2 and len({el.ref for el in elements}) == 2
    assert all(el.ref in get_element_store().handles for el in elements)
    rendered = format_interactive_elements(elements)
    assert "Alpha room" in rendered and "Beta room" in rendered
    beta = next(el for el in elements if "Beta room" in el.context)
    result = await browser_act(BrowserActParams(action="click", selector=beta.ref))
    assert result.success and await screen.evaluate("window.chosen") == "beta"


async def test_reordering_does_not_redirect_a_reference(screen):
    elements = await cards(screen)
    beta = next(el for el in elements if "Beta room" in el.context)
    await screen.evaluate("document.body.prepend(document.querySelector('#beta'))")
    fresh = await extract_interactive_elements(screen)
    assert next(el for el in fresh if "Beta room" in el.context).ref == beta.ref
    result = await browser_act(BrowserActParams(action="click", selector=beta.ref))
    assert result.success and await screen.evaluate("window.chosen") == "beta"


@pytest.mark.parametrize("change", ["replace", "rename", "remove", "add_duplicate"])
async def test_changed_duplicate_target_requires_fresh_observation(screen, change):
    elements = await cards(screen)
    beta = next(el for el in elements if "Beta room" in el.context)
    target = screen.locator('#beta button')
    scripts = {"replace": "el => el.outerHTML = el.outerHTML",
               "rename": "el => el.textContent = 'Book now'",
               "remove": "el => el.remove()",
               "add_duplicate": "el => el.after(el.cloneNode(true))"}
    await target.evaluate(scripts[change])
    assert await find_element_locator(screen, beta.ref) is None
    result = await browser_act(BrowserActParams(action="click", selector=beta.ref))
    assert not result.success and await screen.evaluate("window.chosen") == ""


async def test_scoped_observation_does_not_include_other_cards(screen):
    await cards(screen)
    elements = await extract_interactive_elements(screen, '#beta')
    assert len(elements) == 1 and "Beta room" in elements[0].context
    result = await browser_act(BrowserActParams(action="click", selector=elements[0].ref))
    assert result.success and await screen.evaluate("window.chosen") == "beta"
