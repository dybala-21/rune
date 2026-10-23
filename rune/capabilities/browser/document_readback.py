"""Reuse a retrieved record document without another navigation or model call."""

import asyncio
import json
from contextlib import asynccontextmanager
from urllib.parse import urldefrag

from rune.capabilities.browser.helpers import find_element_locator, is_element_ref
from rune.capabilities.browser.readback import (
    _origin,
    document_report,
    readback_result,
    remember_document,
    response_document,
)
from rune.utils.logger import get_logger

log = get_logger(__name__)


async def _field_selector(page, selector: str) -> str:
    if not is_element_ref(selector):
        return selector
    target = await find_element_locator(page, selector)
    if target is None:
        raise ValueError("The field reference is stale")
    attrs = await target.evaluate("el => ({tag: el.localName, id: el.id, name: el.getAttribute('name')})")
    for attr in ("id", "name"):
        if attrs[attr]:
            candidate = attrs['tag'] + '[' + attr + '=' + json.dumps(attrs[attr], ensure_ascii=False) + ']'
            if await page.locator(candidate).count() == 1:
                return candidate
    raise ValueError("This field has no unique ID or name in the fetched HTML; use an observed CSS selector")


async def _read_current(session, params):
    saved, store = session.result_document, session.elements
    if (not saved or not params or not getattr(params, "checks", None)
            or not getattr(params, "resultRequirement", "") or getattr(params, "readbackUrl", "")):
        return None
    page = session.page
    try:
        if (store is None or store.document_id != saved['document_id'] or page is not saved['page']
                or page.url != saved['url'] or not await store.document.evaluate("doc => doc === document")):
            session.result_document = None
            return None
        checks = []
        for check in params.checks:
            selector = await _field_selector(page, check.selector)
            checks.append(check.model_copy(update={"selector": selector}))
        report = document_report(saved['document'], saved['url'], checks, params.resultRequirement)
        report['source'] = 'navigation_response'
        remember_document(session, page, saved['document'], report)
        return readback_result(report)
    except Exception as exc:
        log.debug("browser_navigation_readback_unavailable", error=type(exc).__name__)
        return readback_result({"status": "unavailable", "scope": "http_document", "url": page.url,
                               "result": params.resultRequirement, "checks": [],
                               "detail": str(exc) if isinstance(exc, ValueError) else "The fetched record could not be checked"})


@asynccontextmanager
async def observed_document(session, tool: str):
    page, store = session.page, session.elements
    responses, known = [], set()
    source = page.url if page is not None else ""
    listening = page is not None and tool in {"browser_act", "browser_navigate", "browser_open"}
    if listening and store is not None and store.page is page and store.last_url == source:
        known = {urldefrag(meta.href)[0] for meta in store.all.values() if meta.href}
        known.add(urldefrag(source)[0])

    def received(response):
        if response.request.is_navigation_request() and response.frame == page.main_frame:
            responses[:] = [response]

    if listening:
        from rune.capabilities.browser.network import get_network_monitor
        monitor = get_network_monitor()
        await monitor.attach(page)
        sequence = monitor.document_sequence
        page.on("response", received)

    async def finish(result, params):
        if responses:
            session.result_document = None
            response = responses[-1]
            try:
                store = session.elements
                if (response.request.method == 'GET' and response.request.redirected_from is None
                        and not response.from_service_worker and urldefrag(response.url)[0] in known
                        and _origin(response.url) == _origin(source) and page.url == response.url
                        and store is not None and store.page is page):
                    if monitor.fresh_document(response.url, sequence):
                        async with asyncio.timeout(1):
                            document = await response_document(response)
                        session.result_document = {"document": document, "url": page.url, "page": page,
                                                   "document_id": store.document_id}
                    else:
                        result.output += ("\nThis document may be cached. If it is the saved record, use browser_observe "
                                          "with readbackUrl set to this URL and its field checks once; do not navigate again or repeat the write.")
            except Exception as exc:
                log.debug("browser_navigation_response_unavailable", error=type(exc).__name__)
        if tool == "browser_observe" and "independent_readback" not in result.metadata:
            fetched = await _read_current(session, params)
            if fetched is not None:
                result.metadata.update(fetched.metadata)
                result.output += "\n" + fetched.output

    try:
        yield finish
    finally:
        if listening:
            page.remove_listener("response", received)
