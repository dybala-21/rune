"""Fix browser result requirements before the executor chooses its checks."""

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rune.capabilities.browser.values import ValueType, equivalent, normalized
from rune.utils.logger import get_logger

log = get_logger(__name__)


# Shared by the routing classifier and the executor tool schema so their wording cannot drift.
KIND_GUIDE = (
    "value=field to change; preserve=field to keep unchanged; result=saved/submitted object, including its "
    "confirmation (one per object; 'save only once' limits that save and is not another result); "
    "criterion=condition for choosing an option or record, with expected=null. "
    "How to work (which tab, order of steps, retries) is not an item."
)
EXECUTOR_KINDS = ("value", "preserve", "result", "criterion")


class BrowserRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    requirement: str = Field(min_length=1, max_length=1000, description="Exact user quote for one final condition; include its requested value.")
    expected: str | None = Field(default=None, max_length=1000, description="Requested final value quoted in requirement; null for page-derived values. Actions and button labels are not expected results.")
    kind: Literal["value", "preserve", "result", "confirmation", "criterion"] = Field(
        default="value", description=KIND_GUIDE + " confirmation=check that a result saved (not another result).")
    value_type: ValueType = "text"
    canonical: str | None = Field(default=None, max_length=1000)
    confirms: str | None = Field(default=None, max_length=1000, description="For a confirmation-only item, copy the requirement quote of its result item. Otherwise null. Never link independent saves.")

    @property
    def id(self) -> str:
        source = json.dumps([self.kind, self.requirement], ensure_ascii=False)
        return "r_" + hashlib.sha256(source.encode()).hexdigest()[:10]

    @model_validator(mode="after")
    def validate_canonical(self):
        if self.kind == "criterion":
            # A selection condition constrains the choice; it is not a literal the page must hold.
            self.expected, self.canonical, self.value_type, self.confirms = None, None, "text", None
            return self
        if self.kind == "preserve" and (self.expected is None or self.expected not in self.requirement):
            self.expected, self.canonical, self.value_type = None, None, "text"
        if self.value_type == "text":
            if self.expected is None:
                # Page-derived values have no literal to normalize.
                if self.canonical is not None:
                    log.debug("browser_requirement_unused_canonical", kind=self.kind)
                self.canonical = None
            elif self.canonical == self.expected:
                self.canonical = None
            elif self.canonical is not None:
                raise ValueError("Text and identifiers must retain their literal value")
            return self
        if self.canonical is None and self.expected is not None and normalized(self.expected, self.value_type) is not None:
            self.canonical = self.expected
        if self.kind == "result":
            raise ValueError("A saved result uses value_type=text. Use kind=value for a numeric or time field.")
        if self.expected is None or self.canonical is None:
            raise ValueError("A typed value needs a literal source and a canonical value")
        if normalized(self.canonical, self.value_type) is None:
            raise ValueError("Canonical values must be decimal numbers or ISO times")
        if (normalized(self.expected, self.value_type) is not None
              and not equivalent(self.expected, self.canonical, self.value_type)):
            raise ValueError("The canonical value changes the requested value")
        return self

    def accepts(self, value: str | None) -> bool:
        return self.expected is None or equivalent(value, self.canonical or self.expected, self.value_type)


class BrowserRequirements(BaseModel):
    model_config = ConfigDict(extra="forbid")
    complete: bool = Field(description="All requested end-state conditions are listed; unknown page values or pending execution do not make this false.")
    items: list[BrowserRequirement] = Field(max_length=16)

    @model_validator(mode="after")
    def validate_confirmations(self):
        indexed = {item.requirement: item for item in self.items}
        results = [item for item in self.items if item.kind == "result" and item.confirms is None]
        for item in self.items:
            if item.kind == "confirmation" and item.confirms is None:
                if len(results) != 1:
                    raise ValueError("A confirmation needs one saved result; use confirms when there are multiple objects")
                item.confirms = results[0].requirement
            if item.confirms is None:
                continue
            target = indexed.get(item.confirms)
            if (item.kind not in {"result", "confirmation"} or item.expected is not None or item.canonical is not None
                    or target is None or target is item or target.kind != "result" or target.confirms is not None):
                raise ValueError("A confirmation must reference one result item and add no separate value constraint")
        return self

    @property
    def outcomes(self) -> list[BrowserRequirement]:
        return [item for item in self.items if item.confirms is None and item.kind != "criterion"]

    def outcome(self, item: BrowserRequirement) -> BrowserRequirement:
        return next(target for target in self.items if target.requirement == item.confirms) if item.confirms else item


INSTRUCTION = (
    "browser_requirements: null unless webpage data must change. List distinct final states, not execution steps. Opening a page is a step, not a saved result. "
    "A save and its confirmation share ONE result, quoting their combined phrase. "
    "Example: 'Save the profile. Check it saved.' -> items=[{\"requirement\":\"Save the profile. Check it saved.\",\"kind\":\"result\"}]. "
    "A separate confirmation phrase uses kind=confirmation; Rune links it to the single result. "
    "With multiple saved objects, confirms names the matching result's exact quote. "
    "A confirmation adds no new value constraint; put any required confirmation wording on the result item. "
    "Separate saved objects and fields stay separate; confirms=null otherwise. "
    "requirement=exact user quote; expected=requested literal within it, null if page-derived. "
    "kind=value/preserve/result/confirmation/criterion. A condition for choosing which option or record to use "
    "(e.g. 'a room with a projector', 'the cheapest flight') is criterion with expected=null; the chosen option's field is a separate value only if the user named it. "
    "How to work (which tab, order of steps, retrying, being careful) is not a requirement; omit it. "
    "A limit on how often to act ('save only once', 'do not save twice') belongs to the save itself: "
    "never list it as another result. Example: 'Change the date and save. Save only once.' -> one result for the save. "
    "Save/submit uses result, expected=null unless final wording is requested. "
    "Quantities/times: value_type=number/time, canonical=decimal/ISO. Text/identifiers/uncertain units/timezones: text, canonical=null. "
    "Later changes override earlier values. complete=true for all conditions, even before execution or with null expected; "
    "false only for unclear intent, omissions or over 16 items. No invented conditions."
)


def request_key(requests: tuple[str, ...]) -> str:
    return hashlib.sha256(json.dumps(requests, ensure_ascii=False).encode()).hexdigest()


def validated(value, requests: tuple[str, ...]) -> BrowserRequirements | None:
    if value is None:
        return None
    result = BrowserRequirements.model_validate(value)
    seen, errors = set(), []
    for item in result.items:
        if not any(item.requirement in request for request in requests):
            errors.append(f"requirement={item.requirement!r} is not an exact user quote. Copy the original phrase, not a label or paraphrase.")
        if item.requirement in seen:
            errors.append(f"Duplicate requirement={item.requirement!r}. Represent each outcome once.")
        if item.expected is not None and item.expected not in item.requirement:
            recovery = ("For a saved object, use kind=result and expected=null unless the user requested exact message text."
                        if item.kind == "result" else "Copy the complete original phrase containing that value.")
            errors.append(f"requirement={item.requirement!r} must include expected={item.expected!r}. {recovery}")
        seen.add(item.requirement)
    if errors:
        raise ValueError("\n".join(errors))
    return result


def quote_repair(value, requests: tuple[str, ...]) -> dict | None:
    try:
        plan = BrowserRequirements.model_validate(value)
        items = []
        for item in plan.items:
            if (item.kind == "value" and item.expected is not None and item.expected in item.requirement
                    and not any(item.requirement in request for request in requests)
                    and sum(other.expected is not None and other.accepts(item.expected) for other in plan.items) == 1):
                item = item.model_copy(update={"requirement": item.expected})
            items.append({**item.model_dump(exclude_defaults=True, exclude_none=True),
                          "requirement": item.requirement, "kind": item.kind, "expected": item.expected})
        proposal = {"complete": plan.complete, "items": items}
        return proposal if validated(proposal, requests) is not None else None
    except ValueError as exc:
        log.debug("browser_quote_repair_unavailable", error=type(exc).__name__)
        return None


def response_schema() -> dict:
    schema = BrowserRequirements.model_json_schema()
    schema["properties"]["items"]["items"] = schema.pop("$defs")["BrowserRequirement"]
    schema["properties"]["items"]["items"]["required"] = list(schema["properties"]["items"]["items"]["properties"])
    def compact(value):
        if isinstance(value, dict):
            return {key: compact(item) for key, item in value.items() if key not in {"title", "default"}}
        return [compact(item) for item in value] if isinstance(value, list) else value
    schema = compact(schema)
    return {"anyOf": [schema, {"type": "null"}]}

def matches(item: BrowserRequirement, check: dict, baseline: dict) -> bool:
    return (refers_to(item, check)
            and item.accepts(check.get("expected"))
            and (item.kind != "preserve" or baseline.get(item.id) == (check.get("property"), check.get("actual")))
            and (item.kind != "result" or check.get("kind") in {"result", "url"})
            and check.get("property") in {"text", "value", "checked", "url"})


def refers_to(item: BrowserRequirement, check: dict) -> bool:
    ids = check.get("requirement_ids")
    return item.id in ids if ids else check.get("requirement") == item.requirement
