## Summary
- Add `runtimes-validator`, a unified validation tool for running automated model checks across inference engines (vLLM, llama.cpp, Ollama)
- Supports managed (framework controls engine lifecycle) and external (connect to running engine) execution modes via the `granite-validate` CLI
- Includes 18 validation tests (common + engine-specific), 220 unit tests, and a plugin architecture for adding new engines and tests

## Test plan
- [ ] `cd runtimes-validator && uv sync --extra dev` installs cleanly
- [ ] `uv run pytest` passes (218/220 — 2 pre-existing failures in `test_openai_compat.py`)
- [ ] `uv run granite-validate --list-engines` returns `llamacpp`, `ollama`, `vllm`
- [ ] `uv run granite-validate --list-tests` returns all 18 validation tests
