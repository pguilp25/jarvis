"""
System knowledge — facts that are MORE RECENT than the AIs' training data.
Injected into all prompts so models don't deny things that exist.

v8 (May 2026): full restructure. CAPS budget cut from ~12 markers to 3.
THINK-INTERLEAVED block moved here (previously duplicated 4x across the
planner prompts). Investigation reframed as iterative (was: single-batch).
Forbidden-phrases lists dropped. See core/prompts_v8.py for the actual
content; this module re-exports it as SYSTEM_KNOWLEDGE for callers that
import the v7 name.
"""

from core.prompts_v8 import SYSTEM_KNOWLEDGE_V8 as SYSTEM_KNOWLEDGE
