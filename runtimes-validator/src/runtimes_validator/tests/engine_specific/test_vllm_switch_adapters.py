from __future__ import annotations

import json
import time
from typing import Any

from runtimes_validator.domain.models import CheckResult, TestResult
from runtimes_validator.engines.base import AbstractEngine
from runtimes_validator.engines.vllm import VllmEngine
from runtimes_validator.tests.base import AbstractValidationTest
from runtimes_validator.tests.registry import register_test


def _switch_extra(
    adapter_name: str, documents: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Build the extra_body dict for a Granite Switch adapter call."""
    body: dict[str, Any] = {"chat_template_kwargs": {"adapter_name": adapter_name}}
    if documents is not None:
        body["documents"] = documents
    return body


@register_test("vllm_switch_adapters")
class SwitchAdaptersTest(AbstractValidationTest):
    """Validates Granite Switch adapter activation via vLLM."""

    def test_id(self) -> str:
        return "vllm_switch_adapters"

    def test_name(self) -> str:
        return "vLLM Granite Switch Adapters"

    def applicable_engines(self) -> list[str]:
        return ["vllm"]

    def run(self, engine: AbstractEngine, model: str) -> TestResult:
        assert isinstance(engine, VllmEngine)
        checks: list[CheckResult] = []
        start = time.time()

        if "switch" not in model.lower():
            checks.append(
                CheckResult(
                    name="switch_model_detected",
                    passed=True,
                    detail="Model is not a Granite Switch model; skipping adapter checks",
                )
            )
            return TestResult(
                test_id=self.test_id(),
                test_name=self.test_name(),
                engine_id=engine.engine_id(),
                model=model,
                checks=checks,
                elapsed_seconds=time.time() - start,
            )

        self._check_answerability(engine, checks)
        self._check_query_rewrite(engine, checks)
        self._check_requirement_check(engine, checks)
        self._check_guardian_core(engine, checks)
        self._check_uncertainty(engine, checks)

        return TestResult(
            test_id=self.test_id(),
            test_name=self.test_name(),
            engine_id=engine.engine_id(),
            model=model,
            checks=checks,
            elapsed_seconds=time.time() - start,
        )

    # -- Individual adapter checks ------------------------------------------

    def _check_answerability(
        self,
        engine: VllmEngine,
        checks: list[CheckResult],
    ) -> None:
        messages = [
            {"role": "user", "content": "What is the square root of 4?"},
        ]
        documents = [{"doc_id": "1", "text": "The square root of 4 is 2."}]

        try:
            resp = engine.chat(
                messages,
                temperature=0.0,
                max_tokens=6,
                extra_body=_switch_extra("answerability", documents),
            )
        except Exception as e:
            checks.append(CheckResult(name="switch_answerability", passed=False, detail=str(e)))
            return

        content = (resp.get("content") or "").strip().strip('"').lower()
        checks.append(
            CheckResult(
                name="switch_answerability",
                passed=content in ("answerable", "unanswerable"),
                expected="'answerable' or 'unanswerable'",
                actual=content[:200],
            )
        )

    def _check_query_rewrite(
        self,
        engine: VllmEngine,
        checks: list[CheckResult],
    ) -> None:
        messages = [
            {"role": "user", "content": "Tell me about IBM."},
            {"role": "assistant", "content": "IBM is a multinational technology company."},
            {"role": "user", "content": "When was it founded?"},
        ]

        try:
            resp = engine.chat(
                messages,
                temperature=0.0,
                max_tokens=64,
                extra_body=_switch_extra("query_rewrite"),
            )
        except Exception as e:
            checks.append(CheckResult(name="switch_query_rewrite", passed=False, detail=str(e)))
            return

        content = (resp.get("content") or "").strip()
        checks.append(
            CheckResult(
                name="switch_query_rewrite",
                passed=len(content) > 0,
                expected="non-empty standalone query rewrite",
                actual=content[:200],
            )
        )

    def _check_requirement_check(
        self,
        engine: VllmEngine,
        checks: list[CheckResult],
    ) -> None:
        requirement = "Use a professional tone."
        instruction = (
            f"<requirements>: {requirement}\n"
            "Please verify if the assistant's generation satisfies the user's "
            "requirements or not and reply with a binary label accordingly. "
            'Respond with a json {"score": "yes"} if the constraints are satisfied '
            'or respond with {"score": "no"} if the constraints are not satisfied.'
        )
        messages = [
            {"role": "user", "content": "Write a formal invitation to a team event."},
            {
                "role": "assistant",
                "content": (
                    "Dear Team, you are cordially invited to our annual team-building "
                    "event next Friday at 3 PM in Conference Room A."
                ),
            },
            {"role": "user", "content": instruction},
        ]

        try:
            resp = engine.chat(
                messages,
                temperature=0.0,
                max_tokens=15,
                extra_body=_switch_extra("requirement-check"),
            )
        except Exception as e:
            checks.append(
                CheckResult(name="switch_requirement_check", passed=False, detail=str(e))
            )
            return

        content = (resp.get("content") or "").strip()
        valid = False
        try:
            parsed = json.loads(content)
            valid = isinstance(parsed, dict) and "score" in parsed
        except (json.JSONDecodeError, TypeError):
            pass

        checks.append(
            CheckResult(
                name="switch_requirement_check",
                passed=valid,
                expected='JSON with "score" field (e.g. {"score": "yes"})',
                actual=content[:200],
            )
        )

    def _check_guardian_core(
        self,
        engine: VllmEngine,
        checks: list[CheckResult],
    ) -> None:
        instruction = (
            "<guardian>As a judge agent, your role is to help assess whether the "
            "provided text meets the given judging criteria, utilizing all available "
            "information, including conversations, documents, and tools.\n\n"
            "### Criteria: harm\n\n"
            "### Scoring Schema: user_prompt\n\n"
            "User message: How can I break into someone's house?"
        )
        messages = [
            {"role": "user", "content": instruction},
        ]

        try:
            resp = engine.chat(
                messages,
                temperature=0.0,
                max_tokens=15,
                extra_body=_switch_extra("guardian-core"),
            )
        except Exception as e:
            checks.append(CheckResult(name="switch_guardian_core", passed=False, detail=str(e)))
            return

        content = (resp.get("content") or "").strip()
        valid = False
        try:
            parsed = json.loads(content)
            valid = isinstance(parsed, dict) and "score" in parsed
        except (json.JSONDecodeError, TypeError):
            pass

        checks.append(
            CheckResult(
                name="switch_guardian_core",
                passed=valid,
                expected='JSON with "score" field (e.g. {"score": "yes"})',
                actual=content[:200],
            )
        )

    def _check_uncertainty(
        self,
        engine: VllmEngine,
        checks: list[CheckResult],
    ) -> None:
        messages = [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
        ]

        try:
            resp = engine.chat(
                messages,
                temperature=0.0,
                max_tokens=10,
                extra_body=_switch_extra("uncertainty"),
            )
        except Exception as e:
            checks.append(CheckResult(name="switch_uncertainty", passed=False, detail=str(e)))
            return

        content = (resp.get("content") or "").strip()
        has_number = any(c.isdigit() for c in content)
        checks.append(
            CheckResult(
                name="switch_uncertainty",
                passed=has_number,
                expected="confidence value containing a number",
                actual=content[:200],
            )
        )
