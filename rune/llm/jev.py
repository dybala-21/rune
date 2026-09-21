"""Translate bounded routing questions to TypeSafe's decision API."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any

import httpx

from rune.agent.classification_response import InvalidClassification, validate_decision
from rune.agent.provenance import ArtifactRoleHints, role_hint_names

# Pin the model so an alias update cannot change the confidence distribution.
# https://docs.typesafe.ai/api
# https://docs.typesafe.ai/confidence
MODEL = "jev-1.13.0"
ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MIN_CONFIDENCE = 0.8
# File roles affect missing-input checks; borderline answers use the fallback.
MIN_ROLE_CONFIDENCE = 0.9


@dataclass(frozen=True)
class DecisionBatch:
    values: dict | None
    artifact_roles: ArtifactRoleHints | None = None
    fallback_reason: str = ""


class DecisionAbstained(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class JevUnavailable(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _choice(instructions: str, criteria: dict[str, str]) -> dict:
    return {"type": "choice", "instructions": instructions, "criteria": {
        **criteria, "unknown": "The request is ambiguous or none of the described options fits.",
    }}


def build_questions(state: dict) -> tuple[dict, dict[str, str]]:
    from rune.agent.calculation import expression_candidates

    expressions = {f"expression_{i}": value for i, value in enumerate(
        expression_candidates(state["request_to_classify"]))}
    questions = {
        "goal_type": _choice("Choose the requested outcome, following routing_rules.", {
            "chat": "Conversation or a general question, including direct arithmetic answers.",
            "web": "Online lookup or reading a URL, without interacting with webpage controls.",
            "research": "Read-only analysis of code or a project.",
            "code_modify": "Create, save, or edit files or code.",
            "execution": "Run commands, tests, builds, installs, or deployments.",
            "browser": "Interact with webpage controls, forms, seats, or bookings.",
            "full": "Work spanning several of these categories or operating native apps.",
        }),
        "desktop": _choice("What native app access does the current request require?", {
            "none": "No native app access. Creating files, Excel-compatible output, direct arithmetic, and webpage interaction alone do not require a native app.",
            "read": "Open or inspect a native app or its unsaved state, without editing or navigating inside it.",
            "input": "Edit, save, navigate, enter input, or calculate inside a native app. Honor an explicit request to use an app.",
        }),
        "requires_execution": _choice("Does correctness require running code, scripts, commands, or tests?", {
            "yes": "The user requests code/tests/commands to run, or executing them is required for correctness.",
            "no": "Reading, analysis, prose, or document work verifiable by reading. Native app input alone does not require code execution.",
        }),
        "email": _choice("Is the request work on email itself?", {
            "yes": "Work on an inbox, email message, draft, or reply.",
            "no": "The requested work is not on email itself.",
        }),
        "document": _choice("Does the user want a standalone document as a deliverable?", {
            "yes": "Produce a standalone report, proposal, or formal document, excluding code.",
            "no": "Conversation, explanations, code, or work without a standalone document deliverable.",
        }),
        "table_output": _choice("Does the user request a CSV/XLSX deliverable aggregating existing source data?", {
            "none": "No such deliverable. Markdown tables, code/test summaries, software, and blank templates use none.",
            "csv": "Save or revise a CSV aggregation of existing source data.",
            "xlsx": "Save or revise an XLSX aggregation of existing source data.",
        }),
        "calculation": _choice("Which COMPLETE literal numeric expression is the user asking to evaluate directly?", {
            "none": "No direct numeric arithmetic evaluation was requested. Use none for native app tasks, identifiers, dates, code-writing, or quoted examples not to evaluate.",
            "unavailable": "A direct numeric expression must be evaluated, but its complete verbatim text is missing from the candidates.",
            **{key: f"The complete requested expression is exactly: {value}" for key, value in expressions.items()},
        }),
    }
    if state.get("previous_request"):
        questions["is_related_to_previous"] = _choice("Does the current request continue the previous request?", {
            "yes": "A continuation that depends on the previous request.",
            "no": "An unrelated new task; do not inherit previous app or deliverable requirements.",
        })
    return questions, expressions


def _validated_choices(payload: Any, questions: dict) -> tuple[dict[str, str], dict[str, float]]:
    if not isinstance(payload, dict) or payload.get("model") != MODEL:
        raise JevUnavailable("invalid_model")
    answers = payload.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(questions):
        raise JevUnavailable("invalid_answers")
    picked = {}
    confidences = {}
    for name, question in questions.items():
        answer = answers[name]
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            raise JevUnavailable("invalid_choice")
        choice, confidence, probabilities = (answer.get(k) for k in ("choice", "confidence", "probabilities"))
        if not isinstance(choice, str) or choice not in question["criteria"]:
            raise JevUnavailable("invalid_choice")
        if (type(confidence) not in (int, float) or not math.isfinite(confidence)
                or not 0 <= confidence <= 1 or not isinstance(probabilities, dict)
                or set(probabilities) != set(question["criteria"])
                or any(type(p) not in (int, float) or not math.isfinite(p) or not 0 <= p <= 1
                       for p in probabilities.values())
                or not math.isclose(sum(probabilities.values()), 1, abs_tol=.02)
                or probabilities[choice] < max(probabilities.values())):
            raise JevUnavailable("invalid_probabilities")
        picked[name] = choice
        confidences[name] = confidence
    return picked, confidences


def _routing_values(picked: dict[str, str], confidences: dict[str, float], expressions: dict[str, str]) -> dict:
    if any(choice == "unknown" or confidences[name] < MIN_CONFIDENCE
           for name, choice in picked.items()):
        raise DecisionAbstained("uncertain")

    if picked["calculation"] == "unavailable":
        raise DecisionAbstained("extraction_needed")
    if picked["desktop"] != "none" and picked["calculation"] != "none":
        raise DecisionAbstained("inconsistent_decision")
    if (picked["desktop"] != "none" and picked["goal_type"] in {"chat", "web", "research", "browser"}
            or picked["calculation"] != "none" and picked["goal_type"] != "chat"
            or picked["table_output"] != "none" and picked["goal_type"] not in {"code_modify", "full"}):
        raise DecisionAbstained("inconsistent_decision")
    intents = [key for key in ("email", "document") if picked[key] == "yes"]
    if picked["desktop"] != "none":
        intents.append("desktop")
    if picked["table_output"] != "none":
        intents.append("table")
    values = {
        "goal_type": picked["goal_type"],
        # Distribution confidence is not comparable to the connected model's score.
        "confidence": min(confidences.values()), "reason": "Structured routing decision",
        "requires_execution": picked["requires_execution"] == "yes",
        "intent_categories": intents, "requires_desktop_input": picked["desktop"] == "input",
        "is_related_to_previous": picked.get("is_related_to_previous") == "yes",
        "table_output": picked["table_output"],
        "calculation_expression": expressions.get(picked["calculation"], ""),
    }
    try:
        return validate_decision(values)
    except InvalidClassification as exc:
        raise JevUnavailable("invalid_decision") from exc


def decode_answers(payload: Any, questions: dict, expressions: dict[str, str]) -> dict:
    picked, confidences = _validated_choices(payload, questions)
    return _routing_values(picked, confidences, expressions)


async def _request(*, model: str, state: dict, questions: dict, timeout: float, api_key: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            response = await client.post(ENDPOINT, headers={
                "Authorization": "Bearer " + api_key,
            }, json={"model": model.removeprefix("typesafe/"), "state": state, "questions": questions})
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise JevUnavailable("invalid_response")
        return payload
    except httpx.HTTPStatusError as exc:
        raise JevUnavailable(f"http_{exc.response.status_code}") from None
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        raise JevUnavailable(type(exc).__name__) from None


async def classify(system: str, content: str, *, timeout: float, api_key: str | None = None) -> DecisionBatch:
    from rune.agent.timing import timed_completion

    state = json.loads(content)
    # Send large requests straight to the existing classifier to avoid duplicate work.
    if len(content) > 16000:
        raise DecisionAbstained("large_request")
    questions, expressions = build_questions(state)
    routing_names = tuple(questions)
    request = state["request_to_classify"]
    file_questions = {f"file_role_{i}": name for i, name in enumerate(role_hint_names(request))}
    for question_id, name in file_questions.items():
        questions[question_id] = _choice(
            f"Classify {json.dumps(name, ensure_ascii=False)} from the current request. "
            "Choose input when its contents are needed even if it must also remain unchanged. "
            "Checking only a hash to prove it is unchanged uses preserve.", {
                "input": "An existing file whose contents the task needs, including code or tests to run or edit.",
                "output": "A file the request asks to create, without requiring pre-existing contents.",
                "preserve": "Mentioned only to keep unchanged; its contents are not needed for this task.",
            })
    payload = await timed_completion(_request, {
        "model": f"typesafe/{MODEL}", "state": {**state, "routing_rules": system},
        "questions": questions, "timeout": timeout,
        "api_key": api_key if api_key is not None else os.environ.get("TYPESAFE_API_KEY", ""),
    })
    # Validate the entire response before reusing any independent answer.
    picked, confidences = _validated_choices(payload, questions)
    roles = {name: picked[key] for key, name in file_questions.items()
             if picked[key] != "unknown" and confidences[key] >= MIN_ROLE_CONFIDENCE}
    hints = ArtifactRoleHints.for_request(request, roles) if roles else None
    try:
        values = _routing_values({key: picked[key] for key in routing_names},
                                 {key: confidences[key] for key in routing_names}, expressions)
    except DecisionAbstained as exc:
        return DecisionBatch(None, hints, exc.reason)
    return DecisionBatch(values, hints)
