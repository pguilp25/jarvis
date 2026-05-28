"""Tests for the starter todo app. All pass on the fixture as shipped."""

import importlib
import os

import pytest


@pytest.fixture
def todo(tmp_path, monkeypatch):
    db = tmp_path / "tasks.json"
    monkeypatch.setenv("TODO_DB", str(db))
    import todo as todo_mod
    importlib.reload(todo_mod)  # re-bind DB_PATH from the patched env
    return todo_mod


def test_add_and_list(todo):
    todo.add_task("buy milk")
    todo.add_task("walk dog")
    rows = todo.list_tasks()
    assert [t["text"] for _, t in rows] == ["buy milk", "walk dog"]
    assert [idx for idx, _ in rows] == [1, 2]


def test_complete(todo):
    tid = todo.add_task("buy milk")
    todo.complete_task(tid)
    rows = todo.list_tasks()
    assert rows[0][1]["done"] is True


def test_delete(todo):
    a = todo.add_task("a")
    todo.add_task("b")
    todo.delete_task(a)
    rows = todo.list_tasks()
    assert [t["text"] for _, t in rows] == ["b"]


def test_persistence(todo):
    todo.add_task("persist me")
    again = todo.load_tasks()
    assert again[0]["text"] == "persist me"
