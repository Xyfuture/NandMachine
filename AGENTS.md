# Repository Guidelines

## Project Structure & Module Organization
`nandmachine/` is the main package. Use `commands/` for macro and micro ops, `simulator/hardware/` for NAND and xPU timing models, `simulator/runtime/` for address translation and tables, `frontend/` for Torch FX graph capture and passes, and `kernels/` for macro-op codegen prototypes. `archive_test/` is the active regression suite despite its name. Root notebooks such as `hw_pipeline.ipynb` and `graph_pipeline.ipynb` are exploratory; reusable logic belongs in package modules. `plan/` and `prompts/` contain design notes, not runtime code.

## Build, Test, and Development Commands
There is no formal build pipeline yet; work from the repository root in an environment that already includes `pytest`, `torch`, `transformers`, and `Desim`.

- `python3 -m pytest archive_test -q` runs the current test suite.
- `python3 -m pytest archive_test/test_nand_sim_core.py -q` checks NAND timing and request serialization paths.
- `python3 -m compileall nandmachine` performs a quick syntax pass before committing.
- `jupyter notebook hw_pipeline.ipynb` opens a hardware-pipeline prototype for manual exploration.

## Coding Style & Naming Conventions
Use Python with 4-space indentation, PEP 8 spacing, and explicit type hints on new or changed APIs. Prefer `snake_case` for modules, functions, and variables, `PascalCase` for classes, and `UPPER_CASE` for constants. Follow local conventions when touching legacy identifiers such as `xPU` or `tRead` instead of renaming them in isolated changes. Keep comments and docstrings brief, technical, and in English.

## Testing Guidelines
Write tests with `pytest` using `test_*.py` files and `test_*` functions. Add focused regression coverage for changes in address encoding, runtime tables, graph passes, or NAND/xPU scheduling behavior. No coverage gate is configured, so every behavior change should include at least one targeted test and, when useful, a minimal failing case first.

## Commit & Pull Request Guidelines
Recent history uses short, imperative, sentence-case subjects such as `Refactor macro commands and xPU transfer engine`. Keep commits scoped to one subsystem or behavioral change. Pull requests should explain what changed, why it matters, and which validation commands were run. If a change affects timing, mapping, or graph lowering, include the relevant test target or notebook evidence in the PR description.
