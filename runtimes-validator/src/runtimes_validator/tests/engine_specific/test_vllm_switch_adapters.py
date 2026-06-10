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

        self._check_base_model(engine, checks)
        self._check_answerability(engine, checks)
        self._check_query_rewrite(engine, checks)
        self._check_clarify_query(engine, checks)
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

    def _check_base_model(
        self,
        engine: VllmEngine,
        checks: list[CheckResult],
    ) -> None:
        """Base model sanity: without any adapter the model must respond as a plain chat model."""
        messages = [
            {"role": "user", "content": "What is 2 + 2?"},
        ]

        try:
            resp = engine.chat(
                messages,
                temperature=0.0,
                max_tokens=64,
            )
        except Exception as e:
            checks.append(CheckResult(name="switch_base_model", passed=False, detail=str(e)))
            return

        content = (resp.get("content") or "").strip()

        non_empty = len(content) > 0

        # Verify the switch layer did not activate an adapter by accident
        looks_like_adapter_json = False
        try:
            parsed = json.loads(content)
            looks_like_adapter_json = isinstance(parsed, dict) and any(
                k in parsed for k in ("score", "clarification", "answerable")
            )
        except (json.JSONDecodeError, TypeError):
            pass

        contains_answer = "4" in content

        checks.append(
            CheckResult(
                name="switch_base_model",
                passed=non_empty and not looks_like_adapter_json and contains_answer,
                expected="plain natural-language response containing '4', not adapter JSON",
                actual=content[:200],
            )
        )

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

    def _check_clarify_query(
        self,
        engine: VllmEngine,
        checks: list[CheckResult],
    ) -> None:
        """Repetition detection: clarify_query must return concise JSON, not a runaway list."""
        messages = [
            {
                "role": "user",
                "content": (
                    "What are the eligibility criteria for the veterans program "
                    "that provides either health care or disability benefits?"
                ),
            },
        ]

        try:
            resp = engine.chat(
                messages,
                temperature=0.0,
                max_tokens=300,
                extra_body=_switch_extra("clarify_query"),
            )
        except Exception as e:
            checks.append(
                CheckResult(name="switch_clarify_query_repetition", passed=False, detail=str(e))
            )
            return

        content = (resp.get("content") or "").strip()
        finish_reason = resp.get("finish_reason")

        # A clarifying question should be short — hitting max_tokens means runaway generation
        if finish_reason == "length":
            checks.append(
                CheckResult(
                    name="switch_clarify_query_repetition",
                    passed=False,
                    expected="finish_reason=stop (clarification should be short)",
                    actual=f"finish_reason=length; content={content[:200]}",
                )
            )
            return

        # Strip punctuation from each token so "VA." and "(VA" both count as "va"
        words = [
            "".join(c for c in token.lower() if c.isalpha())
            for token in content.split()
        ]
        words = [w for w in words if w]

        # Check for repeated trigrams — a runaway loop produces "va outreach programs",
        # "va health care" etc. multiple times; normal text rarely repeats a 3-word phrase
        no_repetition = True
        top_trigram = ""
        if len(words) >= 6:
            trigram_counts: dict[tuple[str, str, str], int] = {}
            for i in range(len(words) - 2):
                trigram = (words[i], words[i + 1], words[i + 2])
                trigram_counts[trigram] = trigram_counts.get(trigram, 0) + 1
            top = max(trigram_counts, key=lambda t: trigram_counts[t])
            max_trigram = trigram_counts[top]
            top_trigram = f'"{top[0]} {top[1]} {top[2]}" x{max_trigram}'
            no_repetition = max_trigram <= 2

        checks.append(
            CheckResult(
                name="switch_clarify_query_repetition",
                passed=no_repetition,
                expected="no trigram repeating more than twice",
                actual=f"{top_trigram} | {content[:150]}",
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
            checks.append(CheckResult(name="switch_requirement_check", passed=False, detail=str(e)))
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
        is_short = len(content) <= 20
        checks.append(
            CheckResult(
                name="switch_uncertainty",
                passed=has_number and is_short,
                expected="short confidence value containing a number (<=20 chars)",
                actual=content[:200],
            )
        )
