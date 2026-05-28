"""A tiny CLI todo app with JSON persistence.

Starter fixture for the JARVIS app-task harness. Intentionally has:
  - no priority support (the `feature` task adds it)
  - an O(n^2) `list_tasks` and no input validation on `complete`/`delete`
    (the `refactor` task fixes these)
It is otherwise correct and its tests pass.
"""

import json
import os
import sys

DB_PATH = os.environ.get("TODO_DB", "tasks.json")


def load_tasks():
    if not os.path.exists(DB_PATH):
        return []
    with open(DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_tasks(tasks):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2)


def add_task(text):
    tasks = load_tasks()
    new_id = (max((t["id"] for t in tasks), default=0)) + 1
    tasks.append({"id": new_id, "text": text, "done": False})
    save_tasks(tasks)
    return new_id


def list_tasks():
    """Return a list of (display_index, task) rows.

    NOTE: this is intentionally O(n^2) — for each task it rescans the whole
    list to compute its 1-based position among the not-yet-listed tasks.
    """
    tasks = load_tasks()
    rows = []
    for t in tasks:
        # O(n) rescan per task -> O(n^2) overall
        display_index = 0
        for other in tasks:
            display_index += 1
            if other["id"] == t["id"]:
                break
        rows.append((display_index, t))
    return rows


def complete_task(task_id):
    tasks = load_tasks()
    for t in tasks:
        if t["id"] == task_id:
            t["done"] = True
    save_tasks(tasks)


def delete_task(task_id):
    tasks = load_tasks()
    tasks = [t for t in tasks if t["id"] != task_id]
    save_tasks(tasks)


def _format_row(display_index, t):
    mark = "x" if t["done"] else " "
    return f"{display_index}. [{mark}] (#{t['id']}) {t['text']}"


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: todo.py {add|list|complete|delete} ...")
        return 1
    cmd = argv[0]
    if cmd == "add":
        new_id = add_task(argv[1])
        print(f"added #{new_id}")
    elif cmd == "list":
        for display_index, t in list_tasks():
            print(_format_row(display_index, t))
    elif cmd == "complete":
        complete_task(int(argv[1]))
        print("ok")
    elif cmd == "delete":
        delete_task(int(argv[1]))
        print("ok")
    else:
        print(f"unknown command: {cmd}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
