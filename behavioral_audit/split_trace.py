#!/usr/bin/env python3
"""Split a JARVIS_ROUND_TRACE jsonl into per-group files for round-by-round audit.

Groups:
  prompt_round0.txt          — the full assembled coder prompt (system+user) the coder saw
  coder_step_<N>.txt         — every coder round of step N (reasoning + tool args/results)
  planner_<label>_<model>.txt — every planner round for one (label, model) draft/merge call

Each group file is human/agent-readable text (not jsonl) so a subagent can Read it directly.
Usage: python3 behavioral_audit/split_trace.py <trace.jsonl> <out_dir>
"""
import json, os, sys, re


def _safe(s):
    return re.sub(r'[^A-Za-z0-9._-]', '_', str(s))[:60]


def main():
    trace, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    rows = []
    with open(trace) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                try:
                    rows.append(json.loads(ln))
                except Exception:
                    pass

    coder_steps = {}   # step -> list of round records
    planners = {}      # (label, model) -> list of round records
    manifest = []

    for r in rows:
        phase = r.get("phase")
        if phase == "coder" and r.get("event") == "prompt":
            p = os.path.join(out, "prompt_round0.txt")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(f"=== FULL CODER PROMPT (step {r.get('step')}, model {r.get('model')}) ===\n\n")
                for m in r.get("messages", []):
                    fh.write(f"\n----- [{m.get('role')}] -----\n{m.get('content','')}\n")
            manifest.append(("prompt", p))
        elif phase == "coder":
            coder_steps.setdefault(r.get("step"), []).append(r)
        elif phase == "planner":
            planners.setdefault((r.get("label"), r.get("model")), []).append(r)

    for step, recs in sorted(coder_steps.items(), key=lambda x: (x[0] is None, x[0])):
        recs.sort(key=lambda x: x.get("round", 0))
        p = os.path.join(out, f"coder_step_{_safe(step)}.txt")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(f"=== CODER STEP {step} — {len(recs)} rounds, model {recs[0].get('model')} ===\n")
            for r in recs:
                fh.write(f"\n========== ROUND {r.get('round')} ==========\n")
                fh.write(f"--- reasoning ---\n{r.get('reasoning','')}\n")
                for i, io in enumerate(r.get("io", [])):
                    fh.write(f"\n--- tool call #{i+1}: {io.get('tool')} ---\n")
                    fh.write(f"args: {json.dumps(io.get('args'), default=str)[:1500]}\n")
                    fh.write(f"result: {io.get('result','')}\n")
        manifest.append((f"coder_step_{step}", p))

    for (label, model), recs in planners.items():
        recs.sort(key=lambda x: x.get("round", 0))
        p = os.path.join(out, f"planner_{_safe(label)}_{_safe(model)}.txt")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(f"=== PLANNER {label} / {model} — {len(recs)} rounds ===\n")
            for r in recs:
                fh.write(f"\n========== ROUND {r.get('round')} ==========\n")
                _pr = r.get("prompt", "") or ""
                if r.get("round") == 1:
                    # Round 1 = the full base prompt (system + plan brief + manifest) — show it whole.
                    fh.write(f"--- prompt the model saw (FULL, round 1) ---\n{_pr}\n")
                else:
                    # Later rounds re-dump the base prompt + accumulated tool results. The base is
                    # identical to round 1; only the NEW tool-result tail + manifest differs. Show
                    # just the tail so the file stays auditable (the base is already above).
                    # #25 (ckpt-216): 15000 was often entirely consumed by the static base, so the
                    # NEWEST tool results (the dynamic part an auditor needs) were clipped. 40000
                    # reliably reaches the fresh tool-result tail + manifest.
                    fh.write(f"--- prompt the model saw (round {r.get('round')}: base same as R1; "
                             f"showing last 40000 chars = newest tool results + manifest) ---\n")
                    fh.write((_pr[-40000:] if len(_pr) > 40000 else _pr) + "\n")
                fh.write(f"\n--- model response (reasoning + bracket tool calls) ---\n")
                fh.write((r.get("response", "") or "") + "\n")
        manifest.append((f"planner_{label}_{model}", p))

    print(f"groups written to {out}:")
    for kind, p in manifest:
        print(f"  {kind}: {p}  ({os.path.getsize(p)} bytes)")
    print(f"\nTOTAL: {len(coder_steps)} coder step(s), {len(planners)} planner draft(s), {len(rows)} trace rows")


if __name__ == "__main__":
    main()
