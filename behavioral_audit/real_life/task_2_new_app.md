# Real-life task #2: Build a small tool

Build a CLI tool `wordfreq.py` that reads a text file and prints the top N most-frequent words.

Requirements:
- Takes `--file` (path) and `--top` (int, default 10) arguments
- Strips punctuation and lowercases all words
- Excludes a default stopword list (the, a, an, and, or, of, in, to, is)
- Prints output as: "WORD: COUNT" one per line
- Add a simple smoke test that runs against a sample text
- Single-file Python; no external dependencies
