from taskman.tasks import add_task, complete_task, pending_tasks, format_task


def test_add_task_does_not_leak_tags_across_calls():
    tasks = []
    add_task(tasks, "Buy milk")
    add_task(tasks, "Write report")
    assert tasks[0]["tags"] == ["untagged"]
    assert tasks[1]["tags"] == ["untagged"], f"tags leaked across calls: {tasks[1]['tags']}"


def test_complete_task_marks_last_task_done():
    tasks = [{"title": "Buy milk", "tags": [], "done": False}]
    complete_task(tasks, "Buy milk")
    assert tasks[0]["done"] is True


def test_pending_tasks_excludes_done():
    tasks = [
        {"title": "Buy milk", "tags": [], "done": False},
        {"title": "Write report", "tags": [], "done": True},
    ]
    result = pending_tasks(tasks)
    assert [t["title"] for t in result] == ["Buy milk"]


def test_format_task_output_format():
    task = {"title": "Buy milk", "tags": [], "done": False}
    assert format_task(task) == "[pending] Buy milk"
