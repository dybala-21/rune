"""Load tool schemas as the task needs them."""

import json
import re
from collections.abc import Collection

_CORE = {"think", "ask_user", "task_blocked", "bash_execute", "project_map", "table_requirements", "table_verify"}
_BROWSER = ("browser_observe", "browser_act", "browser_navigate", "browser_extract", "browser_batch")


def deferred_for_task(names: list[str], classification) -> set[str]:
    from rune.agent.goal_classifier import is_coding_task, is_simple_lookup

    if is_simple_lookup(classification):
        return set(names) - {"web_search", "web_fetch", "ask_user", "task_blocked"}
    intents = classification.intent_categories
    coding = is_coding_task(classification)
    execution = coding or classification.requires_execution or classification.goal_type == "execution" or "calculation" in intents
    return {name for name in names
            if (not coding and (name.startswith("code_") or name == "project_map"))
            or (not coding and name in {"think", "file_delete"})
            or (not execution and name == "bash_execute")
            or ("table" not in intents and name in {"table_requirements", "table_verify"})}


class ToolCatalog:
    def __init__(self, schemas: list[dict], *, deferred_tools: Collection[str] | None = None) -> None:
        self.all = {tool["function"]["name"]: tool for tool in schemas if tool.get("type") == "function"}
        # Provider-native tools stay in the request without passing through discovery.
        self.schemas = [tool for tool in schemas if tool.get("type") != "function"
                        or tool["function"]["name"] not in (deferred_tools or ())
                        and (tool["function"]["name"] in _CORE
                             or tool["function"]["name"].startswith(("file_", "code_", "web_", "desktop_")))]
        self.loaded = {tool["function"]["name"] for tool in self.schemas if tool.get("type") == "function"}
        self.groups = sorted({re.split(r"[_.]", name)[0] for name in self.all if name not in self.loaded})
        self.schemas.append({"type": "function", "function": {
            "name": "tool_search",
            "description": ("Find and load additional tools by exact name or English keywords before calling them. "
                            "Core tools are already available. Available families: " + ", ".join(self.groups)[:600]),
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                           "required": ["query"], "additionalProperties": False},
        }})
        self.loaded.add("tool_search")

    @staticmethod
    def needed(schemas: list[dict]) -> bool:
        return len(schemas) > 24 and len(json.dumps(schemas, ensure_ascii=False)) > 32000

    def with_guidance(self, messages: list[dict]) -> list[dict]:
        from rune.agent.prompts import PROMPT_BROWSER, PROMPT_TABLE

        guidance = []
        if any(name.startswith("browser_") for name in self.loaded):
            guidance.append(PROMPT_BROWSER)
        if self.loaded & {"table_requirements", "table_verify"}:
            guidance.append(PROMPT_TABLE)
        if not guidance:
            return messages

        for index, message in enumerate(messages):
            content = message.get("content")
            if message.get("role") == "system" and isinstance(content, str):
                missing = [section for section in guidance if section not in content]
                if not missing:
                    return messages
                return [*messages[:index], {**message, "content": content + "\n" + "\n".join(missing)}, *messages[index + 1:]]
        return [{"role": "system", "content": "\n".join(guidance)}, *messages]

    async def search(self, query: str) -> str:
        if not isinstance(query, str) or not query.strip() or len(query) > 300:
            return "Use a tool name or a short English description. Families: " + ", ".join(self.groups)
        words = set(re.findall(r"[a-z0-9]+", query.lower()))
        ranked = []
        for name, tool in self.all.items():
            function = tool["function"]
            name_words = set(re.findall(r"[a-z0-9]+", name.lower()))
            description = set(re.findall(r"[a-z0-9]+", function.get("description", "").lower()))
            score = (100 if name.lower() == query.lower().strip() else 0) + 8 * len(words & name_words) + len(words & description)
            if score:
                ranked.append((-score, name))
        exact = next((name for name in self.all if name.lower() == query.lower().strip()), None)
        if exact:
            names = [exact]
        elif query.lower().strip() == "browser":
            names = [name for name in _BROWSER if name in self.all]
        else:
            names = [name for _, name in sorted(ranked)[:5]]
        for name in names:
            if name not in self.loaded:
                self.schemas.append(self.all[name])
                self.loaded.add(name)
        result = {"tools": [{"name": name, "description": self.all[name]["function"].get("description", "")[:500]}
                            for name in names], "families": self.groups if not names else []}
        return json.dumps(result)
