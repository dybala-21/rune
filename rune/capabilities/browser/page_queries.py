"""Read-only page queries: screenshots, extracted text and text search on the open page."""

from __future__ import annotations

from pydantic import BaseModel, Field

from rune.capabilities.browser.core import _get_browser
from rune.capabilities.browser.session import browser_operation, current_session
from rune.types import CapabilityResult
from rune.utils.logger import get_logger

log = get_logger(__name__)


class BrowserScreenshotParams(BaseModel):
    path: str = Field(default="", description="Optional output file path; otherwise saved as a unique session screenshot")
    full_page: bool = Field(default=False, alias="fullPage")


class BrowserExtractParams(BaseModel):
    selector: str = Field(description="Current element ref or CSS selector. Accessibility roles such as spinbutton/status are not HTML tag names; use the observed ref or a real CSS selector.")
    attribute: str = Field(
        default="",
        description="HTML attribute such as href or src (empty for text); DOM properties such as outerHTML are not attributes. value is the original attribute, not the live form value; use browser_observe for current input values.",
    )


class BrowserFindParams(BaseModel):
    text: str = Field(description="Text to search for on the page")


@browser_operation
async def browser_screenshot(params: BrowserScreenshotParams) -> CapabilityResult:
    """Take a screenshot of the current page."""
    log.debug("browser_screenshot", path=params.path, full_page=params.full_page)

    try:
        _, page = await _get_browser()
        from pathlib import Path
        from uuid import uuid4

        from rune.safety.guardian import get_guardian
        from rune.utils.paths import rune_data

        if params.path:
            from rune.agent.isolation import enforce
            if denied := enforce(params.path):
                return CapabilityResult(success=False, error=denied)
            validation = get_guardian().validate_file_path(params.path)
            if not validation.allowed:
                return CapabilityResult(success=False, error=validation.reason)
            output_path = Path(params.path).expanduser().resolve()
        else:
            output_path = rune_data() / "screenshots" / current_session().id / f"{uuid4().hex}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        await page.screenshot(path=str(output_path), full_page=params.full_page)
        if not params.path:
            output_path.chmod(0o600)

        return CapabilityResult(
            success=True,
            output=f"Screenshot saved: {output_path}",
            metadata={
                "path": str(output_path),
                "url": page.url,
                "full_page": params.full_page,
            },
        )

    except RuntimeError as exc:
        return CapabilityResult(success=False, error=str(exc))
    except Exception as exc:
        return CapabilityResult(
            success=False,
            error=f"Screenshot failed: {exc}",
        )


@browser_operation
async def browser_extract(params: BrowserExtractParams) -> CapabilityResult:
    """Read observed elements or CSS matches without changing the page."""
    log.debug("browser_extract", selector=params.selector, attribute=params.attribute)

    try:
        _, page = await _get_browser()
        from rune.capabilities.browser.helpers import find_element_locator, is_element_ref
        if is_element_ref(params.selector):
            target = await find_element_locator(page, params.selector)
            elements = [target] if target is not None else []
        else:
            elements = await page.query_selector_all(params.selector)

        if not elements:
            return CapabilityResult(
                success=False,
                error=(f"No elements found matching: {params.selector}. "
                       "Use browser_observe for the current values and refs; do not repeat this selector. "
                       "Accessibility roles in snapshots are not HTML tag names."),
                metadata={"count": 0},
            )

        values: list[str] = []
        for el in elements:
            if params.attribute:
                val = await el.get_attribute(params.attribute)
                if params.attribute == "value":
                    live = await el.evaluate("el => 'value' in el ? String(el.value) : null")
                    if live is not None and live != val:
                        values.append(f"HTML value attribute: {val!r}; current input value: {live!r}")
                        continue
                if val is not None:
                    values.append(val)
            else:
                text = await el.inner_text()
                if text.strip():
                    values.append(text.strip())

        if not values:
            detail = (f"Matched {len(elements)} element(s), but none has the HTML attribute {params.attribute!r}. "
                      "DOM properties such as outerHTML are not HTML attributes."
                      if params.attribute else f"Matched {len(elements)} element(s), but they contain no readable text.")
            return CapabilityResult(
                success=False,
                error=detail + " Use browser_observe to inspect the page, or extract a known attribute such as iframe src.",
                metadata={"selector": params.selector, "attribute": params.attribute, "count": 0,
                          "matched_count": len(elements), "empty_content": True},
            )

        return CapabilityResult(
            success=True,
            output="\n".join(values),
            metadata={
                "selector": params.selector,
                "attribute": params.attribute,
                "count": len(values),
            },
        )

    except RuntimeError as exc:
        return CapabilityResult(success=False, error=str(exc))
    except Exception as exc:
        return CapabilityResult(
            success=False,
            error=f"Extraction failed: {exc}",
        )


@browser_operation
async def browser_find(params: BrowserFindParams) -> CapabilityResult:
    """Find elements on the page containing the specified text."""
    log.debug("browser_find", text=params.text)

    try:
        _, page = await _get_browser()

        escaped = params.text.replace("'", "\\'")
        xpath = f"//*[contains(text(), '{escaped}')]"

        elements = await page.query_selector_all(f"xpath={xpath}")

        if not elements:
            return CapabilityResult(
                success=True,
                output=f"No elements found containing text: '{params.text}'",
                metadata={"count": 0},
            )

        results: list[dict] = []
        for el in elements[:20]:
            tag = await el.evaluate("el => el.tagName.toLowerCase()")
            text = await el.inner_text()
            el_id = await el.get_attribute("id")
            el_class = await el.get_attribute("class")

            selector = tag
            if el_id:
                selector = f"#{el_id}"
            elif el_class:
                first_class = el_class.strip().split()[0]
                selector = f"{tag}.{first_class}"

            results.append({
                "tag": tag,
                "selector": selector,
                "text": text[:200],
            })

        lines: list[str] = [f"Found {len(results)} element(s) with text '{params.text}':"]
        for r in results:
            lines.append(f"  <{r['tag']}> [{r['selector']}] {r['text'][:80]}")

        return CapabilityResult(
            success=True,
            output="\n".join(lines),
            metadata={
                "count": len(results),
                "elements": results,
            },
        )

    except RuntimeError as exc:
        return CapabilityResult(success=False, error=str(exc))
    except Exception as exc:
        return CapabilityResult(
            success=False,
            error=f"Find failed: {exc}",
        )
