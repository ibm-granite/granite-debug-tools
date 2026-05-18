from __future__ import annotations

from runtimes_validator.domain.models import CheckResult
from runtimes_validator.engines.vllm import VllmEngine


def check_model_listed(
    engine: VllmEngine,
    checks: list[CheckResult],
    name: str,
) -> None:
    try:
        models = engine.list_models()
    except Exception as e:
        checks.append(CheckResult(name=name, passed=False, detail=str(e)))
        return

    checks.append(
        CheckResult(
            name=name,
            passed=len(models) > 0,
            expected=">= 1 model",
            actual=len(models),
        )
    )
