# Repository Guidelines

## Project Structure & Module Organization
- Core package: `nandmachine/`.
- Simulator logic: `nandmachine/simulator/` (`hardware/`, `runtime/`, `entry_point.py`).
- Frontend graph/passes: `nandmachine/frontend/core/` and `nandmachine/frontend/network/`.
- Command abstractions: `nandmachine/commands/` (`micro.py`, `macro.py`).
- Kernel prototypes: `nandmachine/kernels/`.
- Tests: `archive_test/` with `test_*.py` modules.
- Design notes and planning docs: `plan/`, `prompts/`, and notebooks in repo root.



## Coding Style & Naming Conventions
- Python with 4-space indentation and PEP 8-style spacing.
- Use `snake_case` for functions/variables/modules, `PascalCase` for classes, and `UPPER_CASE` for constants.
- Prefer explicit type hints for new/changed APIs (existing code uses `Optional`, `ClassVar`, typed dict/list forms).
- Keep comments/docstrings concise and in English; explain intent, not obvious mechanics.

## Testing Guidelines
- Framework: `pytest` (assert-style tests, function-based test cases).
- Naming: files `test_*.py`, functions `test_*`.
- Add tests alongside behavior changes, especially for address translation, runtime tables, and NAND timing/serialization paths.
- For regressions, add a focused failing test first, then implement the fix.

## Commit & Pull Request Guidelines
- Follow existing history style: imperative, sentence-case summaries (for example, `Add runtime resource management`, `Refactor address encoding`).
- Keep commits scoped to one logical change.
- PRs should include:
  - What changed and why.
  - A short test summary (`pytest` command + result).
  - Linked issue/task when available.
  - Notes on behavior or API changes.

## Agent-Specific Notes
- See `CLAUDE.md`: user-facing discussion should be in Chinese, while code comments and technical artifacts remain in English.
