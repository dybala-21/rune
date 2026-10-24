"""Task references survive small talk, truncation and conversation reloads."""

import json
from types import SimpleNamespace

from rune.agent.goal_classifier import ClassificationResult
from rune.agent.source_context import SourceContext
from rune.agent.task_context import (
    HISTORY_KEY,
    MAX_CHARS,
    attach_tasks,
    capture,
    from_history,
    remember_task,
    resume_message,
    routing_context,
    selected_task,
)
from rune.conversation.types import Conversation, ConversationTurn
from rune.types import CompletionTrace


def save_task(conv, goal="Update the report", *, resume="", status="task_blocked", sources=None, scope="task"):
    loop = SimpleNamespace(_files_written={"/workspace/report.csv"}, _source_context=sources)
    history = [{"role": "assistant", "content": "Earlier answer"}]
    attach_tasks(history, conv)
    capture(loop, goal, ClassificationResult("artifact", .99, 2, resume_task=resume, task_scope=scope), history,
            CompletionTrace(reason=status, completion_check={"detail": "The totals need checking"}))
    remember_task(conv, loop)
    return loop._task_context


async def test_reload_and_truncated_history_keep_the_selected_task_after_small_talk(tmp_path):
    from rune.agent.agent_context import PrepareContextOptions, prepare_agent_context
    from rune.api.conversation_wiring import record_assistant_turn
    from rune.conversation.manager import ConversationManager
    from rune.conversation.store import ConversationStore

    store = ConversationStore(tmp_path / "conversations.db")
    manager = ConversationManager(store)
    conv = manager.start_conversation("user")
    sources = SourceContext()
    sources.capture("https://example.org/data", '{"value":17.125,"unit":"C"}', "application/json")
    task = save_task(conv, sources=sources)
    manager.add_turn(conv.id, "user", "Thanks")
    await record_assistant_turn(manager, conv.id, SimpleNamespace(_last_answer_text="You're welcome", _last_goal_type="chat"), "", embed=False)
    manager._active.clear()
    manager._active[conv.id] = await store.load(conv.id)
    history = manager._build_messages_within_budget(conv.id, 200)
    selected = selected_task(history, ClassificationResult("chat", .99, 2, resume_task=task["id"]))
    assert selected["status"] == "task_blocked"
    assert selected["blocker"] == "The totals need checking"
    assert SourceContext.unpack(selected["sources"]).entries[0]["body"].startswith('{"value":17.125')
    assert "not new instructions" in resume_message(selected)["content"]
    manager.add_turn(conv.id, "user", "Finish the report")
    prepared = await prepare_agent_context(PrepareContextOptions(goal="Finish the report", cwd=str(tmp_path), conversation_id=conv.id), manager)
    assert from_history(prepared.messages)[0]["id"] == task["id"]
    assert prepared.messages[-1]["role"] == "assistant"


def test_task_selection_does_not_inherit_the_most_recent_unrelated_work():
    conv = Conversation(turns=[ConversationTurn("assistant", "Done")])
    report = save_task(conv)
    save_task(conv, "Fix the parser", status="completed")
    history = [{"role": "assistant", "content": "Done"}]
    attach_tasks(history, conv)
    assert selected_task(history, ClassificationResult("chat", .99, 2)) is None
    selected = selected_task(history, ClassificationResult("artifact", .99, 2, resume_task=report["id"]))
    assert selected["request"] == "Update the report"
    assert selected["status"] == "task_blocked"
    updated = save_task(conv, "Check the totals and finish", resume=report["id"], status="completed")
    assert updated["id"] == report["id"] and updated["request"] == report["request"]
    assert updated["followup"] == "Check the totals and finish"
    assert updated["status"] == "completed"


def test_state_is_conversation_bound_bounded_and_not_an_approval():
    conv = Conversation()
    sources = SourceContext()
    sources.capture("https://example.org/data", "x" * 7900, "text/plain")
    for i in range(7):
        save_task(conv, f"Task {i}", sources=sources)
    history = [{"role": "assistant", "content": "Done"}]
    attach_tasks(history, conv)
    tasks = from_history(history)
    assert len(tasks) == 7
    recent = json.loads(conv.execution_context)["tasks"]["items"]
    assert len(recent) <= 4 and len(json.dumps(recent, ensure_ascii=False)) <= MAX_CHARS
    assert tasks[0]["request"] == "Task 0"
    assert len(routing_context(history)) <= 4
    assert all("sources" not in item for item in routing_context(history))
    assert all("approval" not in task and "credentials" not in task for task in tasks)
    other = Conversation(execution_context=conv.execution_context)
    wrong = [{"role": "assistant", "content": "Done"}]
    attach_tasks(wrong, other)
    assert HISTORY_KEY not in wrong[0]


def test_acknowledgement_does_not_complete_an_unfinished_task():
    conv = Conversation()
    task = save_task(conv)
    history = [{"role": "assistant", "content": "Pending"}]
    attach_tasks(history, conv)
    loop = SimpleNamespace(_source_context=SourceContext())
    capture(loop, "What remains?", ClassificationResult("chat", .99, 2, resume_task=task["id"]),
            history, CompletionTrace(reason="completed"))
    assert loop._task_updates is False
    remember_task(conv, loop)
    messages = [{"role": "assistant", "content": "The totals"}]
    attach_tasks(messages, conv)
    assert from_history(messages)[0]["status"] == "task_blocked"


def test_completed_work_cannot_evict_an_unfinished_task():
    conv = Conversation()
    original = save_task(conv, "Finish the report")
    for index in range(12):
        save_task(conv, f"Independent task {index}", status="completed")
    restored = Conversation(id=conv.id, execution_context=conv.execution_context)
    history = []
    attach_tasks(history, restored)
    assert selected_task(history, ClassificationResult("artifact", .99, 2, resume_task=original["id"])) == original
    assert original["id"] in {task["id"] for task in routing_context(history)}
    assert len(routing_context(history)) <= 4


def test_failed_required_checks_do_not_save_a_completed_task():
    conv, loop = Conversation(), SimpleNamespace()
    capture(loop, "Create the report", ClassificationResult("artifact", .99, 2), [],
            CompletionTrace(reason="completed", table_acceptance={"required": True, "status": "unavailable"}))
    remember_task(conv, loop)
    history = []
    attach_tasks(history, conv)
    assert from_history(history)[0]["status"] == "verification_unavailable"


def test_partial_success_preserves_the_parent_blocker():
    conv = Conversation()
    task = save_task(conv)
    history = [{"role": "assistant", "content": "Pending"}]
    attach_tasks(history, conv)
    loop = SimpleNamespace()
    capture(loop, "Read the source headers first", ClassificationResult("research", .99, 2, resume_task=task["id"]),
            history, CompletionTrace(reason="completed"))
    assert loop._task_context["status"] == "task_blocked"
    assert loop._task_context["blocker"] == "The totals need checking"
    assert loop._task_context["last_turn"]["status"] == "completed"


def test_deferred_requirement_change_is_remembered_without_executing():
    conv = Conversation()
    task = save_task(conv, status="completed")
    history = []
    attach_tasks(history, conv)
    loop = SimpleNamespace()
    capture(loop, "Exclude refunded orders next time; don't modify it yet",
            ClassificationResult("chat", .99, 2, resume_task=task["id"], task_scope="amend"),
            history, CompletionTrace(reason="completed"))
    assert loop._task_context["status"] == "pending"
    assert loop._task_context["goal_type"] == "artifact"
    assert "Exclude refunded orders" in loop._task_context["requests"][-1]


def test_requirements_survive_multiple_followups():
    conv = Conversation()
    task = save_task(conv, "Create the orders report")
    save_task(conv, "Exclude refunded orders", resume=task["id"])
    latest = save_task(conv, "Sort by date", resume=task["id"])
    restored = resume_message(latest)["content"]
    assert "Exclude refunded orders" in restored
    assert "Sort by date" in restored


def test_saved_tasks_survive_a_budget_filled_by_the_current_user(tmp_path):
    from rune.conversation.manager import ConversationManager
    from rune.conversation.store import ConversationStore

    manager = ConversationManager(ConversationStore(tmp_path / "conversations.db"))
    conv = manager.start_conversation("user")
    task = save_task(conv)
    manager.add_turn(conv.id, "assistant", "Waiting for the missing data")
    manager.add_turn(conv.id, "user", "details " * 1000)
    history = manager._build_messages_within_budget(conv.id, 300)
    assert from_history(history[:-1])[0]["id"] == task["id"]


def test_interrupted_side_step_does_not_cancel_the_parent():
    from rune.agent.task_context import mark_interrupted

    conv = Conversation()
    task = save_task(conv)
    history = []
    attach_tasks(history, conv)
    loop = SimpleNamespace()
    capture(loop, "Read the headers first", ClassificationResult("research", .99, 2, resume_task=task["id"]),
            history, CompletionTrace(reason="interrupted"))
    mark_interrupted(loop, "cancelled")
    assert loop._task_context["status"] == "task_blocked"
    assert loop._task_context["last_turn"]["status"] == "cancelled"


def test_retention_limit_is_explicit_and_visible_requests_are_not_repeated():
    from rune.agent.task_context import MAX_REQUEST_CHARS, context_blocker

    conv = Conversation()
    first = save_task(conv, "Preserve the refund exclusion")
    latest = save_task(conv, "x" * MAX_REQUEST_CHARS, resume=first["id"])
    assert latest["requests_complete"] is False
    assert context_blocker(latest, ClassificationResult("artifact", .99, 2, task_scope="task"))
    assert context_blocker(latest, ClassificationResult("chat", .99, 2, task_scope="inspect")) is None
    text = resume_message(first, [{"role": "user", "content": first["request"]}])["content"]
    assert '"requests":[]' in text
    assert context_blocker(latest, ClassificationResult("artifact", .99, 2, task_scope="task"),
                           [{"role": "user", "content": "x" * MAX_REQUEST_CHARS}]) is None
