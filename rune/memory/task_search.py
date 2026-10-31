"""Search saved task requirements and artifact names."""

from __future__ import annotations

import re
import sqlite3

_BODY = " || char(10) || ".join(
    f"coalesce(json_extract(data, '$.{key}'), '')"
    for key in ("goal", "notes", "checkpoint.requests", "checkpoint.files")
)


def ensure_index(db: sqlite3.Connection) -> None:
    if db.execute("SELECT 1 FROM sqlite_master WHERE name='task_search'").fetchone():
        return
    with db:
        db.execute("BEGIN IMMEDIATE")
        if db.execute("SELECT 1 FROM sqlite_master WHERE name='task_search'").fetchone():
            return
        db.execute(f"CREATE VIEW task_search_content AS SELECT rowid, {_BODY} AS body FROM tasks")
        old, new = (_BODY.replace("json_extract(data,", f"json_extract({prefix}.data,") for prefix in ("old", "new"))
        for name, tokenizer in (("task_search", "trigram"), ("task_words", "unicode61")):
            db.execute(f"CREATE VIRTUAL TABLE {name} USING fts5(body, content='task_search_content', "
                       f"content_rowid='rowid', tokenize='{tokenizer}')")
            db.execute(f"""CREATE TRIGGER {name}_insert AFTER INSERT ON tasks BEGIN
                INSERT INTO {name}(rowid, body) VALUES(new.rowid, {new}); END""")
            db.execute(f"""CREATE TRIGGER {name}_delete AFTER DELETE ON tasks BEGIN
                INSERT INTO {name}({name}, rowid, body) VALUES('delete', old.rowid, {old}); END""")
            db.execute(f"""CREATE TRIGGER {name}_update AFTER UPDATE OF data ON tasks
                WHEN ({old}) IS NOT ({new}) BEGIN
                INSERT INTO {name}({name}, rowid, body) VALUES('delete', old.rowid, {old});
                INSERT INTO {name}(rowid, body) VALUES(new.rowid, {new}); END""")
            db.execute(f"INSERT INTO {name}({name}) VALUES('rebuild')")
            db.execute(f"CREATE VIRTUAL TABLE {name}_vocab USING fts5vocab({name}, 'row')")


def match_queries(query: str) -> tuple[str, str]:
    words = list(dict.fromkeys("".join(char if char.isalnum() else " " for char in query[:8000]).casefold().split()))
    grams = dict.fromkeys(word[index:index + 3] for word in words for index in range(len(word) - 2))
    # Use words for short names and trigrams for partial matches.
    return (" OR ".join(f'"{gram}"' for gram in list(grams)[:128]),
            " OR ".join(f'"{word}"*' for word in words[:32]))


def selective_queries(db: sqlite3.Connection, expressions: tuple[str, str]) -> tuple[str, str]:
    total = db.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
    selected = []
    for name, expression, prefix in zip(("task_search", "task_words"), expressions, (False, True), strict=True):
        terms = [part[1:-2] if prefix else part[1:-1] for part in expression.split(" OR ")] if expression else []
        if not terms:
            selected.append("")
            continue
        placeholders = ",".join("?" for _ in terms)
        rows = db.execute(f"SELECT term FROM {name}_vocab WHERE term IN ({placeholders}) AND doc * 2 <= ? "
                          "ORDER BY doc, term LIMIT 8", [*terms, total]).fetchall()
        selected.append(" OR ".join(f'"{row[0]}"' for row in rows))
    return selected[0], selected[1]


def referenced_ids(query: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"(?<![\w])([a-f0-9]{12})(?![\w])", query[:8000])))[:4]


def search(db: sqlite3.Connection, owner: str, workspace: str, query: str, *,
           limit: int, offset: int, status: str = "", category: str = "", selective: bool = False) -> list[sqlite3.Row]:
    statements = []
    args: list[str | int] = []
    expressions = match_queries(query)
    # Skip common terms when selecting candidates for the router.
    if selective:
        expressions = selective_queries(db, expressions)
    for name, expression in zip(("task_search", "task_words"), expressions, strict=True):
        if not expression:
            continue
        statement = (f"SELECT tasks.rowid AS identity, rank AS score, "
                     f"snippet({name}, 0, '', '', ' … ', 24) AS matched_text "
                     f"FROM {name} JOIN tasks ON tasks.rowid={name}.rowid "
                     f"WHERE {name} MATCH ? AND tasks.owner=? AND tasks.workspace=?")
        args.extend((expression, owner, workspace))
        for field, value in (("status", status), ("category", category)):
            if value:
                statement += f" AND json_extract(tasks.data, '$.{field}')=?"
                args.append(value)
        statements.append(statement)
    if not statements:
        return []
    # Deduplicate tasks before paging.
    statement = ("WITH matches AS MATERIALIZED (" + " UNION ALL ".join(statements) + "), "
                 "ranked AS (SELECT identity, SUM(score) AS score, MIN(matched_text) AS matched_text "
                 "FROM matches GROUP BY identity) SELECT tasks.*, ranked.matched_text "
                 "FROM ranked JOIN tasks ON tasks.rowid=ranked.identity "
                 "ORDER BY ranked.score, tasks.updated DESC, tasks.id LIMIT ? OFFSET ?")
    args.extend((max(1, min(limit, 100)), max(0, offset)))
    return list(db.execute(statement, args))
