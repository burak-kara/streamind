---
name: submission-validator
description: Validates all window JSON files in results/ against challenge output format requirements
---

Parse every `results/**/*.json` (skip `judge/` subdirs). For each file check:
- Required keys present: `from`, `to`, `summary`, `keywords`, `proc_time`
- `keywords` is a list of exactly 3 strings
- `proc_time` is a positive number
- No extra top-level keys

Report pass/fail per file and an aggregate summary. On any failure, report the file path and the specific violation.
