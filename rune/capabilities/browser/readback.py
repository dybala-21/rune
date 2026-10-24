"""Read an observed result URL without changing or resubmitting the open page."""

import json
import time
from copy import deepcopy
from email.message import Message
from urllib.parse import urldefrag, urljoin, urlsplit
from uuid import uuid4

from lxml import html
from pydantic import BaseModel, Field

from rune.capabilities.browser.checks import BrowserCheck
from rune.capabilities.browser.values import equivalent
from rune.types import CapabilityResult
from rune.utils.logger import get_logger

_MAX_BYTES = 1_000_000


class ResultReadback(BaseModel):
    url: str = Field(max_length=2000, description="Observed same-origin URL for the saved record; never guess a path.")
    checks: list[BrowserCheck] = Field(min_length=1, max_length=32, description="Requested field values in the returned HTML, using CSS selectors. Do not guess success wording.")
    result: str = Field(default="", description="Saved outcome's supplied requirement ID or exact quote; omit only if there is one saved outcome.")


def _origin(url: str) -> tuple:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Readback needs an HTTP(S) URL without embedded credentials")
    return parsed.scheme, parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)


def _read(document, check: BrowserCheck) -> str:
    targets = document.cssselect(check.selector)
    if len(targets) != 1:
        raise ValueError("The response must contain exactly one matching element")
    target = targets[0]
    tag, kind = target.tag.lower(), target.get("type", "text").lower()
    if tag in {"script", "style"} or tag == "input" and kind in {"password", "hidden", "file"}:
        raise ValueError("This field cannot be read back")
    if check.property == "text":
        if tag in {"input", "textarea", "select", "button", "a"} or target.get("role") in {"button", "link"}:
            raise ValueError("Read a result or field value, not an action label")
        if target.cssselect("input,textarea,select"):
            raise ValueError("Container text omits form values. Check each field using its selector and value/checked property")
        return target.text_content().strip()
    if check.property == "checked" and tag == "input" and kind in {"checkbox", "radio"}:
        return str("checked" in target.attrib).lower()
    if check.property == "value":
        if tag == "input":
            return target.get("value", "")
        if tag == "textarea":
            return target.text_content()
        if tag == "select" and "multiple" not in target.attrib:
            options = target.cssselect("option[selected]") or target.cssselect("option")[:1]
            return options[0].get("value", options[0].text_content()) if options else ""
    raise ValueError("The requested property is unavailable in this response")


async def response_document(response):
    if response.status != 200 or "text/html" not in response.headers.get("content-type", "").lower():
        raise ValueError("The result URL did not return a readable HTML document")
    if int(response.headers.get("content-length", "0")) > _MAX_BYTES:
        raise ValueError("The result document exceeds the readback limit")
    body = await response.body()
    if len(body) > _MAX_BYTES:
        raise ValueError("The result document exceeds the readback limit")
    content_type = Message()
    content_type['content-type'] = response.headers.get('content-type', '')
    return html.fromstring(body, parser=html.HTMLParser(no_network=True, encoding=content_type.get_content_charset()))


def document_report(document, url: str, checks: list[BrowserCheck], result: str) -> dict:
    report = {"status": "unavailable", "scope": "http_document", "url": url, "result": result, "checks": []}
    for check in checks:
        actual, error = None, ""
        try:
            actual = _read(document, check)
            if len(actual) > 1000:
                raise ValueError("The returned value exceeds the readback limit")
        except Exception as exc:
            get_logger(__name__).debug("browser_readback_field_unavailable", error=type(exc).__name__)
            error = str(exc) if isinstance(exc, ValueError) else "The field could not be read from the response"
        report["checks"].append({**check.model_dump(), "actual": actual[:1000] if actual is not None else None,
                                  "error": error, "passed": not error and equivalent(actual, check.expected, check.value_type)})
    report["status"] = ("fail" if any(not row["passed"] and not row["error"] for row in report["checks"])
                        else "unavailable" if not checks or any(row["error"] for row in report["checks"]) else "pass")
    report["fields"] = document_fields(document)
    return report


def document_fields(document) -> list[dict]:
    fields = []
    for target in document.cssselect("input,textarea,select"):
        if target.get("type", "text").lower() in {"password", "hidden", "file", "submit", "button", "reset"}:
            continue
        selector = ""
        for attribute in ("id", "name"):
            value = target.get(attribute, "")
            if value and len(value) <= 100 and not any(ord(char) < 32 for char in value):
                candidate = f"{target.tag}[{attribute}={json.dumps(value, ensure_ascii=False)}]"
                if len(document.cssselect(candidate)) == 1:
                    selector = candidate
                    break
        if selector:
            prop = "checked" if target.get("type", "").lower() in {"checkbox", "radio"} else "value"
            fields.append({"selector": selector, "property": prop})
        if len(fields) >= 16:
            break
    return fields


def remember_document(session, page, document, report):
    if report.get("status") not in {"pass", "unavailable"} or not report.get("fields"):
        session.readback_document = None
        return
    snapshot = uuid4().hex
    store = session.elements
    session.readback_document = {"snapshot": snapshot, "document": document, "url": report["url"],
                                "page": page, "page_url": page.url, "store": store,
                                "document_id": store.document_id if store else "",
                                "created": time.monotonic()}
    report["snapshot"] = snapshot
    session.readback_document["report"] = deepcopy(report)


async def _snapshot_matches(saved, session, page, url: str) -> bool:
    store = session.elements
    if (saved["url"] != url or saved["page"] is not page or saved["page_url"] != page.url
            or saved["store"] is not store or store is None or saved["document_id"] != store.document_id
            or time.monotonic() - saved["created"] > 60):
        return False
    try:
        return await store.document.evaluate("doc => doc === document")
    except Exception as exc:
        get_logger(__name__).debug("browser_readback_document_changed", error=type(exc).__name__)
        return False


def readback_result(report: dict) -> CapabilityResult:
    guidance = "This checks fetched fields, not long-term durability. Report only fields actually read. Do not repeat a write to verify it."
    if report.get("status") == "unavailable":
        guidance += " A selector/property error is unknown evidence, not an empty stored value or proof that saving failed."
        if report.get("snapshot") and report.get("fields"):
            guidance += (" Use browser_observe(readbackSnapshot=SNAPSHOT, checks=[...]) with the supplied fields "
                         "to finish checking this response once. No new GET is needed.")
    elif report.get("status") == "fail":
        guidance += " Report the stored-value mismatch; the requested result is not confirmed."
    else:
        guidance += " If all requested fields are covered, finish from these results."
    return CapabilityResult(success=True, output=(
        "Result readback: " + json.dumps(report, ensure_ascii=False) + "\n" + guidance
    ), metadata={"independent_readback": report})


async def read_result(page, url: str, checks: list[BrowserCheck], result: str = "", snapshot: str = "") -> CapabilityResult:
    from rune.capabilities.browser.helpers import find_element_locator, is_element_ref
    from rune.capabilities.browser.session import current_session

    session = current_session()
    response = None
    report = {"status": "unavailable", "scope": "http_document", "url": url, "result": result, "checks": []}
    try:
        saved = None
        if snapshot:
            saved, session.readback_document = session.readback_document, None
            if not saved or saved["snapshot"] != snapshot:
                raise ValueError("The readback snapshot expired or the page changed; inspect before requesting a new read")
            url = url or saved["url"]
        if is_element_ref(url):
            target = await find_element_locator(page, url)
            if target is None:
                raise ValueError("Use a current observed link ref")
            url = await target.evaluate("el => el.matches('a[href]') ? el.href : ''")
            if not url:
                raise ValueError("The reference is not a link")
        url = urldefrag(urljoin(page.url, url))[0]
        report["url"] = url
        if _origin(url) != _origin(page.url):
            raise ValueError("Use an observed result URL on the current origin")
        known = url == urldefrag(page.url)[0] or await page.evaluate(
            "url => [...document.querySelectorAll('a[href]')].some(a => a.href.split('#')[0] === url)", url)
        if not known or any(check.property == "url" for check in checks):
            raise ValueError("Use the current URL or an observed link, with checks for the returned fields")
        if snapshot:
            if not await _snapshot_matches(saved, session, page, url):
                raise ValueError("The readback snapshot expired or the page changed; inspect before requesting a new read")
        else:
            candidate = session.readback_document
            pending = candidate.get("report", {}) if candidate else {}
            if (pending.get("status") == "unavailable" and pending.get("result", "") == result
                    and await _snapshot_matches(candidate, session, page, url)):
                saved, session.readback_document = candidate, None
        if saved:
            document = saved["document"]
        else:
            session.readback_document = None
            response = await page.context.request.get(url, timeout=5000, max_redirects=0, max_retries=0,
                                                      headers={"Cache-Control": "no-cache"})
            document = await response_document(response)
        report = deepcopy(saved["report"]) if saved and not checks else document_report(document, url, checks, result)
        if saved:
            report["source"] = "readback_snapshot"
            if not checks:
                session.readback_document = saved
        else:
            remember_document(session, page, document, report)
    except Exception as exc:
        get_logger(__name__).debug("browser_result_readback_unavailable", error=type(exc).__name__)
        report["detail"] = str(exc) if isinstance(exc, ValueError) else "The result could not be fetched"
    finally:
        if response is not None:
            await response.dispose()
    return readback_result(report)
