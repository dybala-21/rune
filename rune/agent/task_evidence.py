"""Completion evidence for browser interaction and calculated answers."""

import json
import re
from dataclasses import dataclass, field

from rune.agent.browser_requirements import BrowserRequirements, refers_to
from rune.agent.intent_engine import IntentContract
from rune.agent.tool_errors import ToolPreparationError
from rune.capabilities.browser.values import equivalent
from rune.types import CapabilityResult

_CALCULATORS = frozenset({"python", "python3", "node", "deno", "bun", "ruby", "perl", "php", "Rscript", "julia",
                          "awk", "gawk", "jq", "bc", "dc", "sqlite3", "duckdb", "psql", "mysql", "mlr", "csvstat", "qsv"})


def _runs_calculator(command: str) -> bool:
    """Whether a shell command runs an interpreter or data tool, judged from each segment's program."""
    import os
    import shlex

    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|")
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError:
        return False
    segments, current = [], []
    for token in tokens:
        if token and set(token) <= set(";&|"):
            segments.append(current)
            current = []
        else:
            current.append(token)
    segments.append(current)
    for words in segments:
        while words and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", words[0]):
            words = words[1:]
        if words[:1] == ["env"]:
            words = words[1:]
        if words[:2] in (["uv", "run"], ["poetry", "run"], ["pipenv", "run"]):
            words = words[2:]
        program = os.path.basename(words[0]) if words else ""
        if program in _CALCULATORS or re.sub(r"\d+(\.\d+)*$", "", program) in _CALCULATORS:
            return True
    return False


@dataclass
class TaskEvidence:
    browser_actions: int = 0
    browser_observed: bool = False
    browser_failure: str = ""
    computations: int = 0
    browser_effect: str = "change"
    browser_confirmed: bool = False
    calculation_operation: str = "other"
    operations: set[str] = field(default_factory=set)
    request_texts: tuple[str, ...] = ()
    expected_checks: set[tuple[str, str, str, str]] = field(default_factory=set)
    check_specs: dict[tuple[str, str, str, str], dict] = field(default_factory=dict)
    unread_checks: set[tuple[str, str, str, str]] = field(default_factory=set)
    property_corrections: dict[tuple[str, str, str, str], str] = field(default_factory=dict)
    read_checks: set[tuple[str, str, str, str]] = field(default_factory=set)
    approval_denied: bool = False
    browser_task: bool = False
    task_scope: str = "step"
    coverage_required: bool = False
    requirements: BrowserRequirements | dict | None = None
    requirements_key: str = ""
    requirements_prepared: bool = False
    covered_requirements: set[str] = field(default_factory=set)
    result_receipts: dict[str, tuple] = field(default_factory=dict)
    preserved_values: dict[str, tuple[str, str]] = field(default_factory=dict)
    observed_controls: dict[tuple[str, str], str] = field(default_factory=dict)
    plan_pending: bool = False
    plan_correction_required: bool = False
    recovery_requested: bool = False
    independent_readback: dict | None = None
    readback_write_block: bool = False
    page_checks: list[dict] = field(default_factory=list)
    document_id: str = ""
    # The site's own response to the latest accepted write, with the field values that write sent.
    write_receipt: dict | None = None
    accepted_writes: int = 0
    # What the site answered to the latest action's writes: accepted, rejected, unknown, or None if it sent none.
    last_write_status: str | None = None
    readback_offered: bool = False
    multiple_saves_confirmed: bool = False

    @property
    def requirements_unavailable(self) -> bool:
        return self.coverage_required and (not isinstance(self.requirements, BrowserRequirements)
                                            or not self.requirements.complete or not self.requirements.items)

    @property
    def requires_clarification(self) -> bool:
        return self.coverage_required and isinstance(self.requirements, BrowserRequirements) and not self.requirements.complete

    @classmethod
    def for_request(cls, goal: str, classification) -> "TaskEvidence":
        from rune.agent.calculation import calculation_context

        # The exact arithmetic supplied in the prompt is already computed evidence.
        computed = bool(calculation_context(goal, classification))
        return cls(computations=int(computed),
                   browser_task=classification.goal_type == "browser" and classification.browser_effect != "view",
                   task_scope=classification.task_scope,
                   requirements=classification.browser_requirements,
                   requirements_key=classification.browser_requirements_key,
                   browser_effect=classification.browser_effect,
                   calculation_operation=classification.calculation_operation,
                   operations={classification.calculation_operation} if computed else set(),
                   request_texts=(goal,))

    async def prepare_requirements(self) -> str:
        from rune.agent.browser_requirements import request_key, validated

        if not self.browser_task:
            return ""
        self.coverage_required = True
        if not self.requirements_prepared:
            self.requirements_prepared = True
            reusable_keys = {request_key(self.request_texts)}
            if self.task_scope == "step" and self.request_texts:
                reusable_keys.add(request_key(self.request_texts[-1:]))
            if self.requirements is None or self.requirements_key not in reusable_keys:
                self.requirements = None
            else:
                try:
                    self.requirements = validated(self.requirements, self.request_texts)
                except ValueError as exc:
                    from rune.utils.logger import get_logger
                    get_logger(__name__).info("browser_requirements_invalid", error=str(exc)[:160])
                    self.requirements = None
        if self.requires_clarification:
            return "The requested browser outcomes are unclear. Inspect the page and ask for the missing intent before changing data."
        if self.requirements_unavailable:
            return ("Inspect the page, then include taskRequirements with the first browser_batch (or browser_observe). "
                    "Declare all requested end states with exact quotes from the request, before input. "
                    "Navigation is a step, not a result. No separate classifier call is needed. "
                    "Use current_request for step scope and preserve earlier remaining conditions for task scope. "
                    + json.dumps({"scope": self.task_scope, "requests": self.request_texts}, ensure_ascii=False))
        return ("Browser requirements interpreted from the request: " + json.dumps(
            [{"id": item.id, "requirement": item.requirement,
              **item.model_dump(exclude_defaults=True, exclude_none=True)} for item in self.requirements.items],
            ensure_ascii=False, separators=(",", ":"))
            + ". Use one requirement ID per check; confirms links reuse the referenced result and need no extra check. Separate results need separate evidence. "
            "Check fields separately. Missing kind/value_type mean value/text. Use canonical values where given; "
            "otherwise retain literal values. Read page-derived expectations and preserved values before changing data. "
            "Check saved results without repeating writes. Report unresolved coverage; do not repeat successful checks just to bind an ID. "
            "Do not guess save-message wording; check the stored fields. Page messages do not prove storage. "
            "When a separate result link is available and confirmation is needed, browser_observe(readbackUrl=...) "
            "checks its returned fields without resubmitting or changing the open page. Unavailable readback is not a reason to repeat a write. "
            "State which evidence confirmed the result.")

    def _check_key(self, item: dict) -> tuple[str, str, str, str] | None:
        requirement, prop, expected = (item.get(key, "") for key in ("requirement", "property", "expected"))
        if not all(isinstance(value, str) for value in (requirement, prop, expected)):
            return None
        if prop not in {"text", "value", "checked", "url"}:
            return None
        ids = item.get("requirement_ids")
        if ids:
            known = {entry.id for entry in self.requirements.items} if isinstance(self.requirements, BrowserRequirements) else set()
            if not isinstance(ids, list) or not all(isinstance(ref, str) and ref in known for ref in ids):
                return None
            requirement = ",".join(sorted(set(ids)))
        literal = bool(expected) and any(expected in text for text in self.request_texts)
        if not ids and self.request_texts and not (requirement.strip() and any(requirement in text for text in self.request_texts)):
            if not literal or prop not in {"value", "text"}:
                return None
        selector = item.get("selector", "")
        if not isinstance(selector, str):
            return None
        return requirement or selector, expected, selector, prop

    def prepare_tool(self, name: str, params: dict) -> None:
        from rune.agent.browser_plan import accept_plan
        accept_plan(self, name, params)
        def validate(tool, values):
            if tool in {"browser_batch", "browser_workflow"}:
                steps = values.get("actions" if tool == "browser_batch" else "steps", [])
                for step in steps if isinstance(steps, list) else []:
                    if isinstance(step, dict) and isinstance(step.get("params"), dict):
                        validate("browser_" + str(step.get("type" if tool == "browser_batch" else "action", "")), step["params"])
            elif tool in {"browser_act", "browser_observe"} and self.coverage_required and isinstance(self.requirements, BrowserRequirements):
                readback = values.get("readback") if tool == "browser_act" else None
                if readback:
                    nested = {"readbackUrl": readback["url"], "checks": readback["checks"],
                              "resultRequirement": readback.get("result", "")}
                    validate("browser_observe", nested)
                    if nested["checks"]:
                        readback.update(checks=nested["checks"], result=nested["resultRequirement"])
                    else:
                        # An unspecified success message is page evidence, not a stored-field check.
                        values["checks"] = [*values.get("checks", []), *readback["checks"]]
                        values.pop("readback")
                        readback = None
                criteria = [item for item in self.requirements.items if item.kind == "criterion"]
                if criteria and isinstance(values.get("checks"), list):
                    ids, quotes = {item.id for item in criteria}, {item.requirement for item in criteria}
                    def quotes_criterion(text) -> bool:
                        # The model may quote part of the condition's phrase rather than all of it.
                        return isinstance(text, str) and len(text.strip()) >= 4 and any(
                            text.strip() in quote or quote in text for quote in quotes)

                    kept = [check for check in values["checks"] if not (
                        isinstance(check, dict) and (quotes_criterion(check.get("requirement"))
                        or isinstance(check.get("requirement_ids"), list) and check["requirement_ids"]
                        and set(check["requirement_ids"]) <= ids))]
                    if len(kept) != len(values["checks"]):
                        # A selection condition has no field value to check; keep the action and drop the binding.
                        from rune.utils.logger import get_logger
                        get_logger(__name__).debug("browser_criterion_checks_dropped", count=len(values["checks"]) - len(kept))
                        values["checks"] = kept
                fixed = {item.requirement: self.requirements.outcome(item) for item in self.requirements.items if item.kind != "criterion"}
                indexed = {item.id: self.requirements.outcome(item) for item in self.requirements.items if item.kind != "criterion"}
                record_read = bool(values.get("readbackUrl") or values.get("readbackSnapshot"))
                if record_read or tool == "browser_observe" and values.get("checks"):
                    from rune.agent.browser_result import saved_outcomes
                    results = saved_outcomes(self)
                    reference = values.get("resultRequirement", "")
                    target = indexed.get(reference) or fixed.get(reference)
                    if not reference and len(results) == 1:
                        target = results[0]
                    elif not reference and len(results) > 1 and record_read:
                        raise ToolPreparationError("ambiguous_result", "Choose the saved outcome's supplied ID or exact quote for this record.")
                    if target is not None and target.kind != "result" or reference and target is None:
                        raise ToolPreparationError("unknown_result", "Use the saved outcome's supplied ID or exact quote.")
                    values["resultRequirement"] = target.id if target else ""
                for check in values.get("checks", []):
                    ids = check.get("requirement_ids")
                    reference = (indexed.get(ids[0]) if isinstance(ids, list) and len(ids) == 1 else None) or fixed.get(check.get("requirement"))
                    if reference is not None and reference.kind == "result" and reference.expected is None:
                        candidates = [item for item in self.requirements.outcomes
                                      if item.kind == "value" and item.expected is not None and item.accepts(check.get("expected"))]
                        if len(candidates) == 1:
                            ids = check["requirement_ids"] = [candidates[0].id]
                        elif check.get("property") in {"value", "checked"}:
                            raise ToolPreparationError(
                                "ambiguous_requirement", "Bind this field to its value requirement, not the save outcome.",
                                requirements=[{"id": item.id, "requirement": item.requirement} for item in candidates],
                            )
                    if ids:
                        if not isinstance(ids, list) or not all(isinstance(ref, str) and ref in indexed for ref in ids):
                            candidates = ([fixed[check["requirement"]]] if check.get("requirement") in fixed else
                                          [item for item in self.requirements.outcomes if item.expected is not None and item.accepts(check.get("expected"))])
                            if isinstance(ids, list) and len(ids) == 1 and len(candidates) == 1:
                                ids = [candidates[0].id]
                            else:
                                raise ToolPreparationError("unknown_requirement", "Use the provided requirement IDs.", requirements=list(indexed))
                        bound = [indexed[ref] for ref in dict.fromkeys(ids)]
                        if len(bound) > 1:
                            raise ToolPreparationError("ambiguous_requirement", "Use one requirement ID per check and separate evidence for each result.")
                        for item in bound:
                            if not item.accepts(check.get("expected")):
                                raise ToolPreparationError("requirement_value_mismatch", "Use the fixed requested value.",
                                                           requirement=item.id, expected=item.canonical or item.expected)
                        check["requirement"] = bound[0].requirement
                    requirement = check.get("requirement")
                    if fixed and requirement not in fixed:
                        candidates = [item for item in self.requirements.outcomes if item.expected is not None and item.accepts(check.get("expected"))]
                        if not candidates and check.get("property") == "text":
                            candidates = [item for item in self.requirements.outcomes if item.kind == "result" and item.expected is None]
                        if len(candidates) == 1:
                            requirement = check["requirement"] = candidates[0].requirement
                        else:
                            raise ToolPreparationError(
                                "unknown_requirement", "Choose the exact requirement quote for this field before sending input.",
                                supplied=requirement, requirements=list(fixed),
                            )
                    if item := fixed.get(requirement):
                        if not item.accepts(check.get("expected")):
                            raise ToolPreparationError(
                                "requirement_value_mismatch", "Use the fixed value in the page's format; do not change its meaning.",
                                requirement=requirement, expected=item.canonical or item.expected,
                                supplied=check.get("expected"), value_type=item.value_type,
                            )
                        baseline = self.preserved_values.get(item.id)
                        if (record_read and item.kind == "preserve" and baseline is not None
                                and baseline != (check.get("property"), check.get("expected"))):
                            raise ToolPreparationError("preserved_value_unconfirmed", "Use the preserved field's previously observed value and property.")
                        check["value_type"] = item.value_type
                        check["requirement"] = item.requirement
                        check["requirement_ids"] = [item.id]
                        if item.canonical is not None:
                            check["expected"] = item.canonical
                if isinstance(values.get("checks"), list):
                    unique = []
                    for check in values["checks"]:
                        item = fixed.get(check.get("requirement"))
                        if record_read and item and item.kind == "result" and item.expected is None:
                            continue
                        if tool == "browser_act" and item and item.kind == "result" and item.expected is None:
                            if readback:
                                continue
                            if check.get("property") == "text":
                                # Observe unspecified message wording; stored fields establish the saved result.
                                check["expected"] = None
                        if check not in unique:
                            unique.append(check)
                    values["checks"] = unique
            if tool == "browser_act" and values.get("action") != "scroll":
                if self.readback_write_block:
                    raise ToolPreparationError(
                        "result_unconfirmed", "The previous write's stored result is unconfirmed. Read the record again or report the mismatch; do not repeat input or saving.",
                    )
                if self.requires_clarification:
                    raise ToolPreparationError("requirements_unclear", "Clarify the requested outcome before changing data.")
                if self.requirements_unavailable:
                    raise ToolPreparationError("requirements_missing", "Use browser_observe(taskRequirements=...) to bind the requested end states before input.")
        validate(name, params)
        self.prepare(name, params)

    def prepare(self, name: str, params: dict) -> None:
        if name in {"browser_batch", "browser_workflow"}:
            steps = params.get("actions" if name == "browser_batch" else "steps", [])
            for step in steps if isinstance(steps, list) else []:
                if isinstance(step, dict) and isinstance(step.get("params"), dict):
                    self.prepare("browser_" + str(step.get("type" if name == "browser_batch" else "action", "")), step["params"])
            return
        if name not in {"browser_act", "browser_observe"}:
            return
        if name == "browser_observe" and (params.get("readbackUrl") or params.get("readbackSnapshot")):
            return
        checks = params.get("checks", [])
        supplied = {self._check_key(item) for item in checks if isinstance(item, dict)} if isinstance(checks, list) else set()
        observations = {(key[0], key[2], key[3]) for item in checks if isinstance(item, dict)
                        and item.get("property") == "text" and item.get("expected") is None
                        and (key := self._check_key({**item, "expected": ""})) is not None} if isinstance(checks, list) else set()
        for item in checks if isinstance(checks, list) else []:
            if isinstance(item, dict) and (key := self._check_key(item)) is not None:
                if item.get("property") in {"value", "checked", "text", "url"}:
                    previous = [old for old in self.check_specs if old[:2] == key[:2]
                                and (old[3] == key[3] or old[2] == key[2] and self.property_corrections.get(old) == key[3])]
                    literal = bool(key[1]) and any(key[1] in text for text in self.request_texts)
                    if item.get("property") in {"text", "url"} and not literal:
                        previous = [old for old in self.expected_checks if old[0] == key[0] and old[3] == key[3]]
                        if (len(previous) == 1 and previous[0][1] != key[1]
                                and not any(previous[0][1] and previous[0][1] in text for text in self.request_texts)):
                            old = previous.pop()
                            self.expected_checks.discard(old)
                            self.check_specs.pop(old, None)
                            self.unread_checks.discard(old)
                            self.read_checks.discard(old)
                    if (key not in self.check_specs and len(previous) == 1
                            and previous[0] in self.unread_checks and previous[0] not in supplied):
                        # Replace a single unread target; successfully read fields remain separate.
                        old = previous[0]
                        self.expected_checks.discard(old)
                        self.check_specs.pop(old)
                        self.unread_checks.discard(old)
                        self.property_corrections.pop(old, None)
                    self.expected_checks.add(key)
                    if item.get("selector"):
                        self.check_specs[key] = {name: item[name] for name in
                                                 ("selector", "property", "expected", "requirement", "requirement_ids", "value_type") if name in item}
        if isinstance(checks, list) and (checks or name == "browser_observe" or params.get("action") == "click"):
            # Check saved results after input, not between field edits.
            read_results = name == "browser_observe" or params.get("action") == "click"
            params["checks"] = [*checks, *(dict(spec) for key, spec in self.check_specs.items()
                                         if key not in supplied and (key[0], key[2], key[3]) not in observations
                                         and (read_results or key[3] in {"value", "checked"}))]

    def observe(self, name: str, result: CapabilityResult, params: dict | None = None) -> None:
        meta = result.metadata or {}
        document = meta.get("document_id")
        if document and self.document_id and document != self.document_id:
            if self.independent_readback is not None and not self.readback_write_block:
                self.independent_readback = {"status": "inconclusive", "scope": "http_document",
                                             "detail": "The document changed after the saved record was read."}
            self.expected_checks.clear()
            self.check_specs.clear()
            self.unread_checks.clear()
            self.read_checks.clear()
            self.property_corrections.clear()
            self.observed_controls.clear()
            self.covered_requirements.clear()
            self.page_checks.clear()
            self.result_receipts.clear()
            self.browser_confirmed = False
        if document:
            self.document_id = document
        if self.plan_pending:
            import json
            result.metadata["browser_evidence_note"] = "Task outcomes: " + json.dumps([item.model_dump() | {"id": item.id} for item in self.requirements.items], ensure_ascii=False)
            self.plan_pending = False
        if meta.get("cached") or meta.get("replayed"):
            return
        if (meta.get("action_status") == "not_executed"
                and (meta.get("validation_error") or {}).get("code") == "result_unconfirmed"):
            return
        if (meta.get("action_status") == "not_executed" and not result.success
                and (meta.get("validation_error") or {}).get("code") in {
                    "requirements_missing", "invalid_requirements", "empty_requirements", "multiple_saves"}
                and self.browser_task and self.coverage_required and self.requirements_prepared
                and self.requirements is None and not self.browser_actions
                and meta.get("approval_decision") != "denied"):
            self.plan_correction_required = True
            return
        if name == "browser_observe" and isinstance(meta.get("independent_readback"), dict):
            self._observe_readback(result)
            return
        if result.success and not self.browser_actions and name in {"browser_observe", "browser_navigate", "browser_open"}:
            for control in meta.get("observed_controls", []):
                if (isinstance(control.get("node_id"), str) and control.get("property") in {"value", "checked"}
                        and isinstance(control.get("actual"), str) and len(self.observed_controls) < 64):
                    self.observed_controls.setdefault((control["node_id"], control["property"]), control["actual"])
        if meta.get("approval_decision") == "denied":
            self.approval_denied = True
        if name in {"browser_batch", "browser_workflow"}:
            for step in meta.get("browser_steps", []):
                self.observe(step["tool"], CapabilityResult(
                    success=step["success"], metadata=step.get("metadata", {}), error=step.get("error"),
                ))
            if not result.success:
                if not meta.get("verification_failed"):
                    self.browser_failure = result.error or "A browser step did not finish."
                self.browser_confirmed = False
        elif name in {"browser_navigate", "browser_open"}:
            self.browser_confirmed = False
            self.result_receipts.clear()
            if self.independent_readback is not None and not self.readback_write_block and not meta.get("skipped_navigation"):
                self.independent_readback = {"status": "inconclusive", "scope": "http_document",
                                             "detail": "The page changed after the separate readback."}
        elif name == "browser_act":
            self.browser_confirmed = False
            if result.success and meta.get("action_status") == "dispatched":
                if self.independent_readback is not None and meta.get("action") != "scroll":
                    self.independent_readback = {"status": "inconclusive", "scope": "http_document",
                                                 "detail": "The page changed after the separate readback."}
                self.browser_actions += 1
                self.browser_observed = not meta.get("observation_failed", False)
                if meta.get("action") != "scroll":
                    self.browser_failure = ""
                receipt = meta.get("write_receipt")
                if meta.get("action") != "scroll":
                    self.last_write_status = (receipt or {}).get("status")
                if receipt and receipt.get("status") == "accepted":
                    self.accepted_writes += 1
                    from rune.capabilities.browser.network import get_network_monitor

                    # Read what the write sent now: the monitor's buffer moves on as the page keeps talking.
                    self.write_receipt = {**receipt, "sent": get_network_monitor().submitted_fields(receipt.get("seqs") or [])}
                elif meta.get("action") != "scroll":
                    # Only the latest action's own write can stand for the save; a later click that sent
                    # nothing, or new input, leaves no accepted save behind it.
                    self.write_receipt = None
            elif not (result.success and meta.get("unchanged")):
                self.browser_failure = result.error or "The browser action did not finish."
        elif name in {"browser_observe", "browser_extract"}:
            self.browser_confirmed = False
            if self.browser_actions:
                self.browser_observed = result.success
        elif name in {"file_read", "document_read"} and result.success:
            profile = meta.get("table_profile", {})
            if profile.get("complete") and isinstance(profile.get("rows"), int):
                self.computations += 1
                self.operations.add("count")
                if profile.get("sums"):
                    self.operations.add("sum")
        elif name == "bash_execute" and result.success and _runs_calculator(str((params or {}).get("command") or "")):
            # Code that computed the answer is a computed result; a command that only lists or prints is not.
            self.computations += 1
            self.operations.add("code")
        elif name in {"web_fetch", "file_query"} and result.success:
            query = meta.get("record_query", {})
            if (query.get("scope") == ("file_snapshot" if name == "file_query" else "retrieved_response")
                    and isinstance(query.get("matched"), int)
                    and query.get("query", {}).get("operation") in {"sum", "mean", "min", "max", "count"}):
                self.computations += 1
                self.operations.add(query["query"]["operation"])
        if name in {"browser_act", "browser_observe"}:
            self.covered_requirements.clear()
            checks = meta.get("result_checks")
            if isinstance(checks, dict):
                items = checks.get("checks")
                valid = [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []
                self.page_checks = valid[:32]
                if self.coverage_required and self.requirements is not None and not self.browser_actions and name == "browser_observe":
                    for item in self.requirements.items:
                        if item.kind == "preserve":
                            for check in valid:
                                if (refers_to(item, check) and not check.get("error")
                                        and isinstance(check.get("actual"), str)):
                                    self.preserved_values.setdefault(item.id, (check["property"], check["actual"]))
                self.prepare(name, {"checks": valid})
                for item in valid:
                    if (key := self._check_key(item)) is not None:
                        if item.get("error") and key not in self.read_checks:
                            self.unread_checks.add(key)
                            if item.get("property_hint") in {"value", "checked"}:
                                self.property_corrections[key] = item["property_hint"]
                        elif not item.get("error"):
                            self.read_checks.add(key)
                            self.unread_checks.discard(key)
                self._retire_aliases(valid)
                fixed = {item.requirement: item for item in self.requirements.items} if isinstance(self.requirements, BrowserRequirements) else {}
                matched = {key for item in valid if (key := self._check_key(item)) is not None
                           and item.get("passed") is True and equivalent(item.get("actual"), item.get("expected"),
                               fixed[item.get("requirement")].value_type if item.get("requirement") in fixed else "text")}
                saved = self.browser_effect != "change" or checks.get("effect_observed") is True
                self.browser_confirmed = bool(result.success and checks.get("passed") is True
                                              and valid and len(valid) == len(items) and len(matched) > 0
                                              and all(self._check_key(item) in matched for item in valid)
                                              and self.expected_checks <= matched and saved)
                if self.requirements is not None and self.coverage_required:
                    # The baseline is a pre-action observation, so it must not wait for a visible save effect.
                    for item in self.requirements.outcomes:
                        for check in valid:
                            identity = (check.get("node_id") or check.get("selector"), check.get("property"))
                            if (item.kind == "preserve" and not check.get("error")
                                    and len(set(check.get("requirement_ids") or [])) <= 1
                                    and refers_to(item, check) and identity in self.observed_controls):
                                self.preserved_values.setdefault(item.id, (check["property"], self.observed_controls[identity]))
                if self.browser_confirmed and self.requirements is not None and self.coverage_required:
                    from rune.agent.browser_requirements import matches

                    used = set()
                    result_targets = {}
                    for item in self.requirements.outcomes:
                        for check in valid:
                            if len(set(check.get("requirement_ids") or [])) > 1:
                                continue
                            identity = (check.get("node_id") or check.get("selector"), check.get("property"))
                            receipt_key = tuple(sorted(check.get("requirement_ids") or [check.get("requirement", "")]))
                            if (item.kind == "result" and identity in result_targets
                                    and result_targets[identity] != receipt_key):
                                continue
                            if (check.get("passed") is True and matches(item, check, self.preserved_values)
                                    and (item.kind == "result" or identity not in used)):
                                if item.kind == "result" and not self._result_is_current(item.id, check, checks):
                                    continue
                                self.covered_requirements.add(item.id)
                                if item.kind == "result":
                                    result_targets[identity] = receipt_key
                                else:
                                    used.add(identity)
                                break
        if isinstance(meta.get("independent_readback"), dict):
            self._observe_readback(result)
        from rune.agent.browser_result import (
            ACKNOWLEDGED_GUIDANCE,
            READBACK_GUIDANCE,
            acknowledged,
            pending_results,
            write_guidance,
        )
        accepted = bool(self.browser_actions and pending_results(self) and not self.independent_readback and acknowledged(self))
        if (self.coverage_required and not self.requirements_unavailable and not accepted
                and name in {"browser_act", "browser_observe", "browser_batch", "browser_workflow"}
                and (meta.get("result_checks") or any(step.get("metadata", {}).get("result_checks")
                                                      for step in meta.get("browser_steps", [])))):
            from rune.agent.browser_result import retrieved_coverage, saved_outcomes
            covered = retrieved_coverage(self) if saved_outcomes(self) else self.covered_requirements
            result.metadata["remaining_browser_requirements"] = [
                {"id": item.id, "requirement": item.requirement} for item in self.requirements.outcomes if item.id not in covered]
        if accepted:
            result.metadata["browser_evidence_note"] = ACKNOWLEDGED_GUIDANCE
        elif self.browser_actions and pending_results(self) and not self.independent_readback:
            result.metadata["browser_evidence_note"] = write_guidance(self) or READBACK_GUIDANCE

    def _observe_readback(self, result: CapabilityResult) -> None:
        meta = result.metadata or {}
        report = meta["independent_readback"]
        rows = [{**row, "source_url": report.get("url")} for row in report.get("checks", [])]
        for item in self.requirements.outcomes if isinstance(self.requirements, BrowserRequirements) else []:
            if item.kind != "preserve":
                continue
            for row in rows:
                if refers_to(item, row) and self.preserved_values.get(item.id) != (row.get("property"), row.get("expected")):
                    row.update(passed=False, error="The expected value does not match an observed preservation baseline")
                    report = {**report, "status": "unavailable", "detail": row["error"]}
                    result.output += "\n" + row["error"]
        def identity(row):
            return (row.get("source_url"), tuple(row.get("requirement_ids") or [row.get("requirement")]),
                    row.get("selector"), row.get("property"), row.get("expected"))
        seen = {identity(row) for row in rows if not row.get("error") and row.get("actual") is not None}
        unresolved = [row for row in (self.independent_readback or {}).get("checks", [])
                      if not row.get("passed") and not row.get("error") and identity(row) not in seen]
        records = dict((self.independent_readback or {}).get("records", {}))
        records[report.get("result", "")] = {**report, "checks": [*rows, *unresolved]}
        self.independent_readback = {**report, "checks": [*rows, *unresolved], "records": records}
        if unresolved:
            self.independent_readback["status"] = "fail"
            self.independent_readback["detail"] = "An earlier fetched field mismatch remains unresolved. Read that field again; do not repeat the write to verify it."
        states = {record.get('status') for record in records.values()}
        if self.independent_readback['status'] != 'fail' and states - {'pass'}:
            self.independent_readback['status'] = next((state for state in ('fail', 'unavailable', 'inconclusive') if state in states), 'inconclusive')
        self.readback_write_block = bool(self.browser_actions and self.independent_readback["status"] != "pass")

    def _retire_aliases(self, checks: list[dict]) -> None:
        from rune.capabilities.browser.helpers import is_element_ref

        targets: dict[tuple, tuple] = {}
        for item in checks:
            key = self._check_key(item)
            if key is None or not item.get("node_id") or item.get("error"):
                continue
            identity = (item["node_id"], key[0], key[1], key[3])
            previous = targets.setdefault(identity, key)
            # Only retire aliases proven to read the same node and the same requirement.
            if previous != key and is_element_ref(key[2]):
                self.expected_checks.discard(key)
                self.check_specs.pop(key, None)
                self.read_checks.discard(key)
                self.unread_checks.discard(key)

    def _result_is_current(self, key: str, check: dict, checks: dict) -> bool:
        if self.browser_effect != "change":
            return True
        effect = checks.get("effect") or {}
        fingerprint = (check.get("node_id"), check.get("property"), check.get("actual"))
        previous = self.result_receipts.get(key)
        fresh = check.get("changed") is not False
        controls = checks.get("controls")
        if (previous and (previous[1] == effect.get("id") or previous[2] != effect.get("target"))
                and not self._controls_match(previous[3], controls)):
            return False
        if fresh and not self._controls_match(effect.get("controls"), controls, allow_removed=True):
            return False
        if fresh and check.get("passed") and fingerprint[0] and effect.get("id") and effect.get("target"):
            saved = effect.get("controls")
            if saved is not None and controls is not None:
                saved = {**saved, "values": {key: value for key, value in saved["values"].items() if key in controls["values"]}}
            self.result_receipts[key] = (fingerprint, effect["id"], effect["target"], saved)
        if fresh:
            return True
        # A different action can leave an already checked receipt intact.
        return bool(previous and previous[0] == fingerprint and check.get("passed")
                    and effect.get("id") and effect.get("target")
                    and self._controls_match(previous[3], controls)
                    and (previous[1] == effect["id"] or previous[2] != effect["target"]))

    @staticmethod
    def _controls_match(saved: dict | None, current: dict | None, *, allow_removed: bool = False) -> bool:
        if saved is None:
            return True
        if current is None or not saved.get("complete") or not current.get("complete"):
            return False
        return all(current["values"].get(key) == value or allow_removed and key not in current["values"]
                   for key, value in saved["values"].items())

    def acceptance(self) -> dict | None:
        from rune.agent.browser_result import pending_results, retrieved_coverage, saved_outcomes
        if not self.coverage_required:
            if self.independent_readback:
                state = self.independent_readback.get("status")
                return {"required": True, "status": state if state in {"pass", "fail", "unavailable"} else "inconclusive",
                        "method": "page_readback", "scope": "browser_result_readback",
                        "independentReadback": self.independent_readback,
                        "detail": self.independent_readback.get("detail", "")}
            return None
        retrieved = retrieved_coverage(self)
        covered = retrieved if saved_outcomes(self) else self.covered_requirements | retrieved
        pending = pending_results(self)
        missing = ([item.id for item in self.requirements.outcomes if item.id not in covered]
                   if isinstance(self.requirements, BrowserRequirements) else [])
        page_missing = {item.id for item in self.requirements.outcomes} - self.covered_requirements if isinstance(self.requirements, BrowserRequirements) else set()
        page_status = ("fail" if self.browser_failure else "unavailable" if self.requirements_unavailable
                       else "pass" if self.browser_confirmed and not page_missing else "inconclusive")
        status = ("fail" if self.browser_failure else "unavailable" if self.requirements_unavailable
                  else "pass" if (self.browser_confirmed or retrieved) and not missing and not pending else "inconclusive")
        if self.independent_readback and status != "fail":
            state = self.independent_readback.get("status")
            if state != "pass":
                status = state if state in {"fail", "unavailable", "inconclusive"} else "inconclusive"
        from rune.agent.action_receipts import redact_url
        from rune.agent.browser_result import acknowledged
        receipt = status == "inconclusive" and acknowledged(self)
        if receipt:
            status = "acknowledged"
        from rune.agent.browser_result import selection_criteria
        return {"required": True, "status": status, "scope": "browser_request_requirements",
                **({"uncheckedCriteria": criteria} if (criteria := selection_criteria(self)) else {}),
                **({"writeReceipt": {"status": self.write_receipt.get("status"), "requests": [
                    {**request, "url": redact_url(request.get("url"))} for request in self.write_receipt.get("requests") or []]}}
                   if receipt else {}),
                **({"savedWriteSeqs": list(self.write_receipt.get("seqs") or [])} if self.write_receipt else {}),
                "method": "page_readback", "missing": missing, "actions": self.browser_actions,
                "pendingResults": pending, "pageStatus": page_status,
                "requirements": [item.requirement for item in self.requirements.outcomes]
                if isinstance(self.requirements, BrowserRequirements) else [],
                "readback": {"status": "pass" if self.browser_confirmed else "inconclusive",
                             "scope": "observed_page_state"},
                **({"independentReadback": self.independent_readback} if self.independent_readback else {}),
                "detail": self.review_detail()}

    def review_detail(self) -> str:
        from rune.agent.browser_result import (
            acknowledged,
            pending_results,
            retrieved_coverage,
            saved_outcomes,
        )
        if self.browser_failure:
            return self.browser_failure
        # Only when no readback was attempted; an attempted readback explains the result itself.
        if not self.independent_readback and acknowledged(self) and not retrieved_coverage(self):
            return "The site accepted the save and the submitted fields matched; the stored record was not read back."
        if self.independent_readback and self.independent_readback.get("status") != "pass":
            return self.independent_readback.get("detail") or "Separate result readback did not confirm the requested fields. Do not repeat a write to verify it."
        if self.requirements_unavailable:
            return "The requested browser outcomes were not bound before input."
        if pending_results(self):
            return "The requested saved record has not been independently confirmed. Page messages alone do not establish storage. Do not repeat the write to check it."
        retrieved = retrieved_coverage(self)
        covered = retrieved if saved_outcomes(self) else self.covered_requirements | retrieved
        missing = [item.requirement for item in self.requirements.outcomes if item.id not in covered]
        if missing:
            return (("The observed page checks passed. " if self.browser_confirmed else "")
                    + "Request coverage remains unconfirmed: " + "; ".join(missing))
        return ""

    def stop_reason(self) -> str:
        state = (self.independent_readback or {}).get("status")
        if self.browser_failure or state == "fail":
            return "task_incomplete"
        if self.requirements_unavailable or state == "unavailable":
            return "verification_unavailable"
        return "verification_inconclusive"

    def blocker(self, contract: IntentContract, executions: int) -> str | None:
        if contract.kind == "browser_write":
            from rune.agent.browser_result import (
                READBACK_GUIDANCE,
                pending_results,
                retrieved_coverage,
            )
            if self.independent_readback:
                report = self.independent_readback
                covered = retrieved_coverage(self)
                missing = [item.requirement for item in self.requirements.outcomes if item.kind == "value" and item.id not in covered] if isinstance(self.requirements, BrowserRequirements) else []
                if ((report.get("status") == "unavailable" or report.get("status") == "pass" and missing)
                        and report.get("fields") and report.get("snapshot")
                        and not self.recovery_requested):
                    return ("The fetched record is available, but some requested fields remain unchecked. "
                            "Use browser_observe with readbackSnapshot from the last report, "
                            "and all requested field checks using the supplied selectors/properties. This reuses the same response. "
                            "Do not repeat the write or infer that saving failed. Unconfirmed fields: "
                            + json.dumps(missing, ensure_ascii=False))
                return None
            if self.browser_failure:
                return ("The requested browser interaction is unfinished: " + self.browser_failure[:500]
                        + " Inspect the page and complete the remaining action, or report the blocker. "
                        "Do not substitute a calculated answer for a requested page update.")
            if self.browser_effect in {"change", "edit"}:
                if self.coverage_required:
                    if self.requirements is None or not self.requirements.complete or not self.requirements.items:
                        return "The browser requirement checker is unavailable. Report the observed actions and unresolved verification; do not repeat input."
                    pending = pending_results(self)
                    baseline_missing = any(item.kind == "preserve" and item.id not in self.preserved_values for item in self.requirements.outcomes)
                    if self.browser_actions and pending and not (self.browser_confirmed and (len(pending) > 1 or baseline_missing)):
                        from rune.agent.browser_result import acknowledged
                        # Tool results already offered the record readback; the server's acceptance is the evidence.
                        if acknowledged(self):
                            from rune.agent.browser_result import record_link_visible

                            # Prefer reading the stored record: ask once when the page links somewhere it could be.
                            if self.readback_offered or not record_link_visible():
                                return None
                            self.readback_offered = True
                            return READBACK_GUIDANCE
                        from rune.agent.browser_result import write_guidance
                        return write_guidance(self) or READBACK_GUIDANCE
                    if retrieved_coverage(self):
                        return None
                    # Re-reading cannot recover a missing baseline or bind another saved result.
                    missing = [f"{item.id}: {item.requirement}" for item in self.requirements.outcomes
                               if item.id not in self.covered_requirements
                               and (not self.browser_confirmed or item.kind == "value"
                                    or item.kind == "preserve" and item.id in self.preserved_values)]
                    if missing:
                        return ("Browser requirements still need readback: " + "; ".join(missing)
                                + ". Check these original requirements without repeating a save or submission.")
                if self.browser_confirmed:
                    return None
                return ("The requested browser result has not been checked. Use browser_act.checks on the final action, "
                        "or browser_observe.checks to read the expected page value without repeating input. "
                        "Use the supplied requirement IDs in checks; retain earlier expected results. "
                        "For saved changes, use readbackUrl to retrieve the stored fields; current inputs and messages do not prove storage. "
                        "Scrolling, an existing button, or a dispatched click does not verify a saved change. Do not repeat a write to check it.")
            if not self.browser_actions:
                return "No browser input confirms the requested interaction. Observing the page alone does not complete it."
            if not self.browser_observed:
                return "Read the page after the dispatched action to verify its result. Do not repeat the action."
        if contract.kind == "calculation" and not self.computations:
            return ("No computed result supports this calculation yet. Read the CSV with file_read for its "
                    "whole-file counts and sums, or file_query/web_fetch recordQuery for a filtered aggregate. "
                    "A successful shell command alone is not a calculation result. Do not create an unrequested deliverable.")
        if (contract.kind == "calculation" and self.calculation_operation != "other"
                and self.calculation_operation not in self.operations and "code" not in self.operations):
            return (f"The requested {self.calculation_operation} has no matching computed result. "
                    "Use that operation on the requested data; another aggregate does not complete it.")
        return None
