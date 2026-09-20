"""Compare observed app state without treating new reference IDs as progress."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import deque
from io import BytesIO
from typing import Any

from PIL import Image, ImageChops

from rune.computer.protocol import DesktopAction, DesktopCondition, DesktopError


def normalized(value: Any) -> Any:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, list):
        return [normalized(item) for item in value]
    if isinstance(value, dict):
        return {key: normalized(item) for key, item in value.items()}
    return value


def control_state(control: dict) -> dict:
    return {key: control[key] for key in ("role", "name", "value", "nameTruncated", "nameSHA256",
                                        "valueTruncated", "valueSHA256", "bounds", "press", "writable") if key in control}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(normalized(value), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def present(data: dict) -> dict:
    visible = {key: value for key, value in data.items() if key != "image_base64"}
    visible["controls"] = [{**{key: value for key, value in row.items() if key not in {"nameSHA256", "valueSHA256"}},
                            "ref": f"e{index}"} for index, row in enumerate(data.get("controls", []), 1)]
    return visible


def resolve_action(action: DesktopAction, view: dict) -> DesktopAction:
    if action.observation != view.get("observation"):
        raise DesktopError("Read the app with desktop_observe before proposing input.")
    if not action.ref:
        return action
    for index, control in enumerate(view.get("controls", []), 1):
        if action.ref == f"e{index}":
            return action.model_copy(update={"ref": control["ref"]})
    raise DesktopError("The control ref is not in the current observation. Read the app again.")


def match_condition(condition: DesktopCondition, view: dict) -> dict | None:
    def matches(row: dict, field: str, text: str, *, exact: bool) -> bool:
        value = row.get(field)
        if not isinstance(value, str):
            return False
        actual, expected = normalized(value), normalized(text)
        if exact and row.get(f"{field}Truncated"):
            return row.get(f"{field}SHA256") == hashlib.sha256(expected.encode()).hexdigest()
        return actual == expected if exact else expected in actual

    if condition.kind == "title":
        return {"title": view["title"]} if matches(view, "title", condition.text, exact=condition.match == "equals") else None
    for index, control in enumerate(view.get("controls", []), 1):
        if control.get("role") != condition.role:
            continue
        if condition.name is not None and not matches(control, "name", condition.name, exact=True):
            continue
        if matches(control, condition.field, condition.text, exact=condition.match == "equals"):
            evidence = {"ref": f"e{index}", "role": control["role"], condition.field: control[condition.field]}
            if control.get(f"{condition.field}Truncated"):
                evidence["truncated"] = True
                evidence["matchedBy"] = "full_value_sha256" if condition.match == "equals" else "visible_prefix"
            return evidence
    return None


class ObservationProgress:
    def __init__(self) -> None:
        self.states: dict[str, str] = {}
        self.unchanged: dict[str, int] = {}
        self.noops: deque[tuple[str, str, str]] = deque(maxlen=16)
        self._semantics: dict[str, str] = {}
        self._frames: dict[str, Image.Image] = {}

    def observe(self, view: dict, frame: bytes) -> dict:
        app = view["app"]
        semantics = digest({"app": app, "title": view.get("title"), "width": view.get("width"),
                            "height": view.get("height"), "controls": sorted(
                                digest({k: v for k, v in control_state(c).items() if k != "bounds"})
                                 for c in view.get("controls", []))})
        with Image.open(BytesIO(frame)) as image:
            image.thumbnail((384, 384), Image.Resampling.BOX)
            sample = image.convert("RGB")
        previous = self.states.get(app)
        changed = (semantics != self._semantics.get(app)
                   or _screen_changed(self._frames.get(app), sample))
        change = "initial" if previous is None else "changed" if changed else "unchanged"
        self.unchanged[app] = self.unchanged.get(app, 0) + 1 if change == "unchanged" else 0
        if change == "changed":
            self.noops = deque((item for item in self.noops if item[0] != app), maxlen=16)
        if changed:
            self.states[app] = digest((semantics, hashlib.sha256(sample.tobytes()).hexdigest()))
            self._semantics[app] = semantics
            # Keep this frame until changes accumulate past the threshold.
            self._frames[app] = sample
        return {"change": change, "unchangedObservations": self.unchanged[app]}

    def action_key(self, action: DesktopAction, view: dict) -> tuple[str, str, str]:
        target = next((control_state(c) for c in view.get("controls", []) if c.get("ref") == action.ref), None)
        args = action.model_dump(exclude={"observation", "ref"}, exclude_defaults=True)
        return view["app"], self.states[view["app"]], digest({"input": args, "target": target})

    def blocker(self, key: tuple[str, str, str]) -> str | None:
        if self.noops.count(key) >= 2:
            return ("The same input produced no visible change twice. No further input was sent. "
                    "Use desktop_wait to check for a delayed result, inspect a different control, or ask the user for help. "
                    "A new observation ID alone does not make this a different action.")
        return None

    def acted(self, key: tuple[str, str, str]) -> dict:
        changed = self.states.get(key[0]) != key[1]
        if not changed:
            self.noops.append(key)
        return {"change": "changed" if changed else "no_visible_change", "unchangedAttempts": self.noops.count(key)}


def _screen_changed(before: Image.Image | None, after: Image.Image) -> bool:
    if before is None or before.size != after.size:
        return True
    bands = ImageChops.difference(before, after).split()
    difference = ImageChops.lighter(ImageChops.lighter(bands[0], bands[1]), bands[2])
    # Filter cursor blinks and compression noise. Accessibility handles small
    # text changes; the model still receives the original screenshot.
    changed_pixels = sum(difference.histogram()[24:])
    return changed_pixels >= max(16, after.width * after.height / 1000)
