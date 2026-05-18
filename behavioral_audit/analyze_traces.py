#!/usr/bin/env python3
"""Analyze JARVIS session traces for behavioral signals."""
import sys, re
from pathlib import Path
from collections import defaultdict

def analyze_session(session_dir: Path) -> dict:
    """Return per-instance metrics."""
    if not session_dir.exists():
        return {}
    metrics = defaultdict(dict)
    # Read workflow log to map instances → time windows
    log_text = (session_dir / "workflow.log").read_text() if (session_dir / "workflow.log").exists() else ""
    # Count specific behavioral signals per model file
    for md in session_dir.glob("*.md"):
        if md.name in ("review.md",):  # skip
            continue
        model = md.stem
        text = md.read_text(errors="replace")
        signals = {
            "scenario_trace_mentions": len(re.findall(r"FAILING SCENARIO|SCENARIO TRACE|the failing scenario|articulate the principle", text, re.I)),
            "principle_mentions":      len(re.findall(r"the principle|principle this bug|articulate the principle", text, re.I)),
            "enumerate_mentions":      len(re.findall(r"enumerate sites|enumerate.*sites|every site", text, re.I)),
            "think_blocks":            len(re.findall(r"\[think\]", text)),
            "edit_blocks":             len(re.findall(r"=== EDIT:", text)),
            "search_calls":            len(re.findall(r"\[SEARCH:", text)),
            "code_calls":              len(re.findall(r"\[CODE:", text)),
            "view_calls":              len(re.findall(r"\[VIEW:", text)),
            "refs_calls":              len(re.findall(r"\[REFS:", text)),
            "lines":                   text.count("\n"),
        }
        metrics["__per_model__"][model] = signals
    # Combined totals
    totals = defaultdict(int)
    for m, sigs in metrics["__per_model__"].items():
        for k, v in sigs.items():
            totals[k] += v
    metrics["__totals__"] = dict(totals)
    return metrics

def main():
    if len(sys.argv) < 2:
        # find latest session
        sessions = sorted(Path.home().glob("jarvis_thinking_logs/*/"), key=lambda p: p.stat().st_mtime)
        if not sessions:
            print("No sessions found")
            return
        target = sessions[-1]
    else:
        target = Path(sys.argv[1])
    print(f"Analyzing: {target}")
    m = analyze_session(target)
    print()
    print("=== TOTALS ===")
    for k, v in sorted(m.get("__totals__", {}).items()):
        print(f"  {k}: {v}")
    print()
    print("=== PER MODEL ===")
    for model, sigs in m.get("__per_model__", {}).items():
        print(f"  {model}:")
        for k, v in sorted(sigs.items()):
            print(f"    {k}: {v}")

if __name__ == "__main__":
    main()
