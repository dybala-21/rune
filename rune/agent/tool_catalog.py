"""Load uncommon tool schemas on demand without changing the cached core."""

import json
import re

_CORE = {"think", "ask_user", "task_blocked", "bash_execute", "project_map", "table_requirements", "table_verify"}


class ToolCatalog:
    def __init__(self, schemas: list[dict]) -> None:
        self.all = {tool["function"]["name"]: tool for tool in schemas}
        self.schemas = [tool for tool in schemas if tool["function"]["name"] in _CORE
                        or tool["function"]["name"].startswith(("file_", "code_", "web_", "desktop_"))]
        self.loaded = {tool["function"]["name"] for tool in self.schemas}
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
        names = [exact] if exact else [name for _, name in sorted(ranked)[:5]]
        for name in names:
            if name not in self.loaded:
                self.schemas.append(self.all[name])
                self.loaded.add(name)
        return json.dumps({"tools": [{"name": name, "description": self.all[name]["function"].get("description", "")[:500]}
                                     for name in names], "families": self.groups if not names else []})
