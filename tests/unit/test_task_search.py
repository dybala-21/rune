"""Search saved tasks within owner, workspace and prompt limits."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest

from rune.agent.task_context import from_history, routing_context
from rune.agent.task_continuity import attach_shared_tasks
from rune.conversation.types import Conversation
from rune.memory.task_search import match_queries
from rune.memory.task_store import TaskStore
from rune.utils.tokenizer import count_tokens


def record(goal, **values):
    return {"id": uuid4().hex[:12], "goal": goal, "status": "pending", "kind": "manual", **values}


@pytest.mark.parametrize("goal,query", [
    ("Fix invoice.mjs; only count paid invoices", "Continue the invoice.mjs correction"),
    ("청구 합계 버그: paid 행만 합산", "청구 합계 수정을 이어가줘"),
    ("請求書の合計計算を修正", "先週の請求書を再開して"),
    ("整理季度销售报表", "继续销售报表"),
    ("Préparer le résumé trimestriel", "Continuer le résumé trimestriel"),
    ("税", "税"),
])
def test_old_task_retrieval_is_scoped_and_bounded(tmp_path, goal, query):
    store = TaskStore()
    wanted = record(goal)
    store.create("local", str(tmp_path), [wanted])
    store.create("local", str(tmp_path), [record(f"Unrelated deck {i}") for i in range(12)])
    store.create("channel:other", str(tmp_path), [record(goal)])
    store.create("local", str(tmp_path / "other"), [record(goal)])
    history = []
    attach_shared_tasks(history, Conversation(user_id="web:session"), str(tmp_path), query)
    routed = routing_context(history)
    assert wanted["id"] in {task["id"] for task in routed}
    assert len(routed) <= 4
    assert count_tokens(json.dumps(routed, ensure_ascii=False)) <= 600
    loaded = next(task for task in from_history(history) if task["id"] == wanted["id"])
    assert loaded["requests"] == [goal]


def test_index_tracks_requirement_changes_and_rolls_back_with_records(tmp_path):
    store = TaskStore(tmp_path / "tasks.db")
    task = record("Keep the report", notes="pomegranate")
    store.create("local", "workspace", [task])
    store.update("local", "workspace", task["id"], {"notes": "quartz"})
    assert not store.search("local", "workspace", "pomegranate")
    assert store.search("local", "workspace", "quartz")[0]["id"] == task["id"]
    with pytest.raises(sqlite3.IntegrityError):
        store.create("local", "workspace", [record("rollback-marker"), task])
    assert not store.search("local", "workspace", "rollback-marker")
    stored = store.get("local", "workspace", task["id"])
    store.checkpoint("local", "workspace", {"id": task["id"], "request": "Keep the report",
                     "requests": ["Update zirconia appendix"], "_revision": stored["revision"]}, lease="")
    assert store.search("local", "workspace", "zirconia")[0]["id"] == task["id"]
    with sqlite3.connect(store.path) as db:
        db.execute("DELETE FROM tasks WHERE id=?", (task["id"],))
    assert not store.search("local", "workspace", "zirconia")


def test_existing_database_is_indexed_once_under_concurrent_access(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE tasks(owner TEXT, workspace TEXT, id TEXT, data TEXT, revision INTEGER, "
                   "updated REAL, lease TEXT DEFAULT '', pid INTEGER, PRIMARY KEY(owner,workspace,id))")
        db.execute("INSERT INTO tasks(owner,workspace,id,data,revision,updated) VALUES(?,?,?,?,1,1)",
                   ("local", "work", "abcdef123456", json.dumps(record("legacy orchid", id="abcdef123456"))))
    store = TaskStore(path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        found = list(pool.map(lambda _: store.search("local", "work", "orchid"), range(2)))
    assert all(rows[0]["id"] == "abcdef123456" for rows in found)
    with sqlite3.connect(path) as db:
        db.execute("INSERT INTO task_search(task_search, rank) VALUES('integrity-check', 1)")
        db.execute("INSERT INTO task_words(task_words, rank) VALUES('integrity-check', 1)")


def test_search_treats_query_syntax_as_text_and_has_a_fixed_query_budget(tmp_path):
    store = TaskStore(tmp_path / "tasks.db")
    store.create("local", "work", [record("orchid"), record("ordinary")])
    assert not store.search("local", "work", '"; DROP TABLE tasks; --')
    assert not store.search("local", "work", " ")
    grams, words = match_queries(" ".join(f"word{index}" for index in range(5000)))
    assert len(grams.split(" OR ")) <= 128
    assert len(words.split(" OR ")) <= 32


def test_search_pages_are_stable_and_filters_apply_before_paging(tmp_path):
    store = TaskStore(tmp_path / "tasks.db")
    rows = [record(f"Orchid chapter {index}", category="document", status="pending") for index in range(25)]
    store.create("local", "work", rows + [record("Orchid", category="code", status="completed")])
    expected = store.search("local", "work", "Orchid chapter", limit=100, category="document", status="pending")
    pages = [row for offset in (0, 10, 20) for row in store.search(
        "local", "work", "Orchid chapter", limit=10, offset=offset, category="document", status="pending")]
    assert len(pages) == 25
    assert [row["id"] for row in pages] == [row["id"] for row in expected]


def test_numbered_task_is_not_replaced_by_a_newer_prefix_match(tmp_path):
    store = TaskStore(tmp_path / "tasks.db")
    wanted = record("Prepare report 23")
    store.create("local", "work", [wanted, *[record(f"Prepare report {i}") for i in range(230, 250)]])
    found = store.candidates("local", "work", "Continue report 23")
    assert wanted["id"] in {item["id"] for item in found}
    common = store.candidates("local", "work", "Prepare report")
    assert {item["id"] for item in common} == {item["id"] for item in store.list("local", "work", limit=4, unfinished_first=True)}


def test_requirement_outside_title_is_visible_to_router(tmp_path):
    store = TaskStore()
    wanted = record("Prepare the document", notes="Keep the zirconia appendix")
    store.create("local", str(tmp_path), [wanted, *[record(f"Other work {i}") for i in range(8)]])
    history = []
    attach_shared_tasks(history, Conversation(user_id="web:session"), str(tmp_path), "Continue the zirconia appendix")
    routed = next(item for item in routing_context(history) if item["id"] == wanted["id"])
    assert routed["request"] == wanted["goal"]
    assert "zirconia" in routed["matched_context"]
    assert len(routed["matched_context"]) <= 160


@pytest.mark.parametrize("goal,query", [
    ("invoice.mjs의 청구 합계 버그를 맡아줘. paid 행만 합산하고 공개 함수 이름과 인자를 유지해야 해. "
     "지금은 파일을 읽고 수정할 사항만 파악해. 코드와 테스트 파일은 아직 변경하지 마.",
     "앞서 맡긴 청구 합계 수정을 이제 진행해. 추가로 소수 금액도 정확히 계산돼야 해. "
     "기존 테스트는 변경하지 말고 node --test로 실행해서 결과를 확인해."),
    ("Prepare the release plan; preserve the original source. Inspect it first without creating a document.",
     "Continue the release plan with the updated source and preserve the original."),
])
def test_retrieved_task_keeps_its_identity_beside_matching_constraints(tmp_path, goal, query):
    wanted = record(goal, status="completed", kind="conversation", category="research")
    wanted["checkpoint"] = {"id": wanted["id"], "request": goal, "requests": [goal],
                            "status": "completed", "goal_type": "research"}
    store = TaskStore()
    store.create("local", str(tmp_path), [wanted, *[record(f"Unrelated presentation {i}") for i in range(8)]])
    history = []
    attach_shared_tasks(history, Conversation(user_id="web:session"), str(tmp_path), query)
    routed = routing_context(history)
    selected = next(task for task in routed if task["id"] == wanted["id"])
    assert selected["request"] == goal[:240]
    assert selected["status"] == "completed"
    assert len(routed) <= 4
    assert count_tokens(json.dumps(routed, ensure_ascii=False)) <= 600
    assert next(task for task in from_history(history) if task["id"] == wanted["id"])["requests"] == [goal]


def test_matching_excerpt_is_not_repeated_when_the_request_contains_it():
    from rune.agent.task_context import HISTORY_KEY

    history = [{HISTORY_KEY: [{"id": "abcdef123456", "request": "Keep the complete source precision",
                              "_match": "complete source precision"}]}]
    assert "matched_context" not in routing_context(history)[0]


@pytest.mark.parametrize("excerpt", [None, False, 1, [], {}])
def test_invalid_search_excerpt_cannot_hide_a_valid_task(excerpt):
    from rune.agent.task_context import HISTORY_KEY

    history = [{HISTORY_KEY: [{"id": "abcdef123456", "request": "Update the report", "_match": excerpt}]}]
    task = routing_context(history)[0]
    assert task["request"] == "Update the report"
    assert "matched_context" not in task


@pytest.mark.parametrize("excerpt", ["Matching source context " * 20, "관련 검색 발췌문 " * 20])
def test_optional_excerpts_do_not_displace_task_candidates(excerpt):
    from rune.agent.task_context import HISTORY_KEY

    tasks = [{"id": f"{index:012x}", "request": f"Job {index}: " + "요구 사항 " * 30,
              "_match": excerpt} for index in range(4)]
    routed = routing_context([{HISTORY_KEY: tasks}])
    assert [task["id"] for task in routed] == [task["id"] for task in tasks]
    assert [task["request"] for task in routed] == [task["request"][:240] for task in tasks]
    assert count_tokens(json.dumps(routed, ensure_ascii=False)) <= 600
