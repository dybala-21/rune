"""Keep page messages separate from separately retrieved saved values."""

from rune.agent.browser_requirements import BrowserRequirements, refers_to
from rune.capabilities.browser.values import equivalent
from rune.utils.logger import get_logger

log = get_logger(__name__)


def saved_outcomes(evidence):
    plan = evidence.requirements
    return [item for item in plan.outcomes if item.kind == "result"] if isinstance(plan, BrowserRequirements) else []


def retrieved_coverage(evidence) -> set[str]:
    report = evidence.independent_readback or {}
    plan = evidence.requirements
    if (report.get("status") != "pass" or not isinstance(plan, BrowserRequirements)
            or not evidence.browser_actions):
        return set()
    covered, used, field_targets = set(), set(), {}
    results = {item.id: item for item in saved_outcomes(evidence)}
    for record in (report.get('records') or {'': report}).values():
        fields, targets = set(), set()
        rows = record.get('checks', [])
        for item in plan.outcomes:
            if item.kind == 'result':
                continue
            for row in rows:
                target = (record.get('url'), row.get('selector'), row.get('property'))
                if (target[1] and field_targets.get(target, item.id) == item.id
                        and row.get("passed") is True and not row.get("error") and refers_to(item, row)
                        and item.accepts(row.get("expected"))
                        and equivalent(row.get("actual"), row.get("expected"), item.value_type)
                        and (item.kind != "preserve" or evidence.preserved_values.get(item.id)
                             == (row.get("property"), row.get("actual")))):
                    fields.add(item.id)
                    targets.add(target)
                    field_targets[target] = item.id
                    break
        covered.update(fields)
        identity = frozenset(targets)
        result = results.get(record.get('result'))
        wording_confirmed = result is not None and (
            result.expected is None or result.id in evidence.covered_requirements
            or any(row.get('passed') is True and not row.get('error') and refers_to(result, row)
                   and result.accepts(row.get('expected')) and equivalent(row.get('actual'), result.expected, 'text') for row in rows))
        # Each saved object needs its own record fields, not a shared success message.
        if fields and identity not in used and wording_confirmed:
            covered.add(result.id)
            used.add(identity)
    return covered


def acknowledged(evidence) -> bool:
    """The run's only accepted write came from its latest action and sent the requested values.

    Weaker than a retrieved record: it shows what the server accepted, not what it stored.
    Each requested value must equal a parsed field the write sent, never a substring; a plan with no
    literal value to compare needs a readback.
    """
    receipt, plan = evidence.write_receipt, evidence.requirements
    # Any other accepted write in the run could be the one that counted.
    if (not receipt or evidence.accepted_writes != 1 or evidence.browser_failure or not isinstance(plan, BrowserRequirements)
            or (evidence.independent_readback or {}).get("status") == "fail"):
        return False
    results = saved_outcomes(evidence)
    if len(results) != 1 or results[0].expected is not None:
        return False
    wanted = []
    for item in plan.outcomes:
        if item.kind == "result":
            continue
        if item.kind == "preserve":
            prop, literal = evidence.preserved_values.get(item.id, (None, None))
            if prop not in {"value", "text"}:
                return False
        else:
            literal = item.canonical or item.expected
        if not isinstance(literal, str) or not literal.strip():
            return False
        wanted.append((literal, item.value_type))
    if not wanted:
        return False
    # A select sends its option's value, while the user names its label.
    options = [pair for pair in receipt.get("selected_options") or [] if isinstance(pair, list) and len(pair) == 2]

    def sent(literal: str, value_type: str, fields: list[str]) -> bool:
        literal = literal.replace("\r\n", "\n")
        if any(equivalent(field, literal, value_type) for field in fields):
            return True
        return any(label.strip() == literal.strip() and value in fields for label, value in options)

    return any(all(sent(literal, value_type, fields) for literal, value_type in wanted)
               for fields in receipt.get("sent") or [])


def pending_results(evidence) -> list[str]:
    covered = retrieved_coverage(evidence)
    return [item.id for item in saved_outcomes(evidence) if item.id not in covered]


READBACK_GUIDANCE = (
    "The requested saved result has not been read. "
    "Use browser_observe(readbackUrl=OBSERVED_RECORD_LINK) to retrieve the saved record without changing the tab. "
    "Include requested field checks if its selectors are known; otherwise the response supplies fields and a readbackSnapshot for checking them without another GET. "
    "A success message and current input values are page evidence only. "
    "If no readable result is available, report that limit; do not invent a URL or repeat the write."
)


ACKNOWLEDGED_GUIDANCE = (
    "The site accepted the save (HTTP success) and the submitted fields matched. "
    "If a link to the saved record is already visible, read it once with browser_observe(readbackUrl=...) and check the requested fields. "
    "If none is visible, do not search further: finish and report that the site accepted the save but the stored record was not read back."
)


REFUSED_GUIDANCE = (
    "The site refused the last save (see its HTTP status), so nothing was stored by it. "
    "If the user asked to retry after such a failure, repeat the same save action once with the current input; "
    "otherwise report the refusal. Do not change the input to make it pass."
)
UNKNOWN_WRITE_GUIDANCE = (
    "The site's answer to the last save does not show whether it was stored (for example a 500, a timeout or an error "
    "reported inside a success response). Do not repeat it: a second submission could save twice. "
    "Read the saved record if a link to it is visible; otherwise report that the outcome is unknown."
)


def record_link_visible() -> bool:
    """Whether the open page links to another page of the same site, where the saved record might be read."""
    try:
        from rune.capabilities.browser.session import current_session

        elements = current_session().elements
        return elements is None or bool(elements.other_page_links())
    except Exception as exc:  # Unknown page state: ask for the readback rather than skip it.
        log.debug("record_link_check_failed", error=str(exc))
        return True


def write_guidance(evidence) -> str | None:
    """Guidance chosen from the site's own answer to the latest save, when it settles what to do next."""
    return {"rejected": REFUSED_GUIDANCE, "unknown": UNKNOWN_WRITE_GUIDANCE}.get(evidence.last_write_status)


def recovery_tools(evidence) -> set[str] | None:
    if (evidence.browser_task and evidence.coverage_required
            and evidence.requirements_prepared and evidence.requirements is None
            and not evidence.browser_actions
            and not evidence.approval_denied and not evidence.browser_failure):
        if not evidence.document_id:
            return {"browser_navigate", "browser_open", "browser_observe", "ask_user", "task_blocked"}
        if evidence.plan_correction_required:
            return {"browser_observe", "ask_user", "task_blocked"}
    if (not evidence.browser_task or not evidence.browser_actions
            or evidence.approval_denied or evidence.browser_failure or evidence.requirements_unavailable):
        return None
    results = {item.id for item in saved_outcomes(evidence)}
    covered = retrieved_coverage(evidence)
    if not evidence.recovery_requested and not (results and results <= evidence.covered_requirements | covered):
        return None
    report = evidence.independent_readback or {}
    if report:
        missing = any(item.kind == "value" and item.id not in covered for item in evidence.requirements.outcomes)
        pending = bool(report.get("snapshot") and report.get("fields")
                       and (report.get("status") == "unavailable" or report.get("status") == "pass" and missing))
    else:
        pending = bool(results and results <= evidence.covered_requirements and pending_results(evidence))
    return {"browser_observe", "browser_navigate", "ask_user", "task_blocked"} if pending else None


def can_finish_from_readback(evidence, classification) -> bool:
    if (classification.goal_type != "browser" or classification.intent_categories - {"browser"}
            or classification.requires_code or classification.requires_execution
            or classification.requires_desktop_input or classification.needs_web_evidence
            or classification.output_expectation != "text" or classification.source_output != "none"
            or classification.is_multi_task or evidence.approval_denied
            or not evidence.browser_task or not evidence.coverage_required
            or evidence.requirements_unavailable or not saved_outcomes(evidence)):
        return False
    acceptance = evidence.acceptance() or {}
    return (acceptance.get("status") == "pass"
            and (evidence.independent_readback or {}).get("status") == "pass"
            and {item.id for item in evidence.requirements.outcomes} <= retrieved_coverage(evidence))


def budget_answer(evidence, locale: str = "en") -> str:
    text = ("조회한 저장 기록에서 요청한 값을 확인했습니다. 추가 설명은 모델 호출 한도에 도달해 생략했습니다."
            if locale.split("-", 1)[0] == "ko" else
            "The retrieved record matches the requested values. Further explanation stopped at the model budget limit.")
    return text + record_facts(evidence, locale)


def final_answer(evidence, trace, answer: str, locale: str = "en") -> str:
    from rune.agent.run_outcome import run_outcome

    acceptance = getattr(trace, "requirement_acceptance", None) or {}
    if not evidence.browser_actions or not saved_outcomes(evidence) or not acceptance.get("required"):
        return answer
    if acceptance.get("status") == "pass":
        if not answer.strip() and getattr(trace, "reason", "") in {"request_budget_exhausted", "cost_budget_exhausted"}:
            return budget_answer(evidence, locale) + evidence_limits(evidence, acceptance, locale)
        return answer + evidence_limits(evidence, acceptance, locale)
    outcome = run_outcome(trace)
    if outcome.success:
        return answer + evidence_limits(evidence, acceptance, locale)
    summary = unfinished_summary(evidence, locale)
    # Keep the stop reason alongside the evidence, including cancellation and budget limits.
    reason = outcome.reason.replace("\n", " ")[:160]
    return summary + (f"\n\n중단 사유: {reason}" if locale.split("-", 1)[0] == "ko" else f"\n\nStopped: {reason}")


def selection_criteria(evidence) -> list[str]:
    plan = evidence.requirements
    return [item.requirement for item in plan.items if item.kind == "criterion"] if isinstance(plan, BrowserRequirements) else []


def evidence_limits(evidence, acceptance: dict, locale: str = "en") -> str:
    """State what a successful run did not check, independently of how the model phrased its answer."""
    korean = locale.split("-", 1)[0] == "ko"
    lines = []
    if acceptance.get("status") == "acknowledged":
        lines.append("사이트가 저장 요청을 수락했지만, 저장된 기록을 다시 조회해 확인하지는 않았습니다." if korean else
                     "The site accepted the save, but the stored record was not read back.")
    if criteria := selection_criteria(evidence):
        quoted = ", ".join(f"“{criterion[:120]}”" for criterion in criteria[:6])
        lines.append(f"선택 조건은 자동으로 검증하지 않았습니다: {quoted}" if korean else
                     f"Selection conditions were not checked automatically: {quoted}")
    return "\n\n" + "\n".join(f"- {line}" for line in lines) if lines else ""


def unfinished_summary(evidence, locale: str = "en") -> str:
    report = evidence.independent_readback or {}
    korean = locale.split("-", 1)[0] == "ko"
    if report.get("status") == "fail":
        text = ("조회한 저장 기록이 요청한 값과 다릅니다. 완료된 것으로 처리할 수 없습니다."
                if korean else "The retrieved record does not match the requested values. The task is not confirmed complete.")
    else:
        text = ("브라우저에서 수행한 작업은 있지만, 요청한 저장 결과를 확인하지 못했습니다. 재시도 전에 저장 기록을 확인해야 합니다."
                if korean else "Browser actions were performed, but the requested saved result remains unconfirmed. Check the saved record before retrying.")
    return text + record_facts(evidence, locale)


def record_facts(evidence, locale: str) -> str:
    report = evidence.independent_readback or {}
    korean = locale.split("-", 1)[0] == "ko"
    rows = report.get("checks", []) or evidence.page_checks
    if not rows:
        return ""
    from html import escape
    def label(value):
        return escape(" ".join(str(value).split())[:180]).replace("`", "&#96;")
    facts = []
    for row in rows[:12]:
        actual = label(row['actual']) if row.get('actual') is not None else ('확인 불가' if korean else 'unavailable')
        expected = label(row.get('expected', ''))
        comparison = f"요청: {expected} / 조회: {actual}" if korean else f"requested: {expected}; retrieved: {actual}"
        if row.get("observed") or any(item.expected is None and refers_to(item, row) for item in saved_outcomes(evidence)):
            comparison = f"관찰: {actual}" if korean else f"observed: {actual}"
        facts.append(f"- {label(row.get('requirement', ''))} — {comparison}")
    source = ("저장 기록 조회" if korean else "Retrieved record") if report.get('checks') else ("화면에서 확인한 내용" if korean else "Observed page fields")
    return "\n\n" + source + ":\n\n" + "\n".join(facts)
