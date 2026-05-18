# Real-life task #1: Add a function

You're working in a fresh Python project. Add a function `parse_log_line` 
to `logparse/parser.py` that:
- Takes a log line string in format `[TIMESTAMP LEVEL] MESSAGE`
- Returns a dict {"timestamp": str, "level": str, "message": str}
- Raises ValueError if the line doesn't match the format
- Handle TIMESTAMP as ISO-format (YYYY-MM-DD HH:MM:SS) and LEVEL as one of INFO/WARN/ERROR/DEBUG

Also add a test file `logparse/tests/test_parser.py` with at least 3 test cases.
