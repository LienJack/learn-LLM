from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

REFUSAL_MARKERS = (
    "资料不足",
    "无法判断",
    "不能确定",
    "无法确认",
    "不能替代",
    "建议咨询",
    "人工复核",
    "seek professional",
    "not enough information",
)


@dataclass(frozen=True)
class JsonMetricResult:
    valid: bool
    parsed: dict[str, Any] | None
    missing_fields: list[str]
    error: str | None = None

    @property
    def score(self) -> float:
        return 1.0 if self.valid and not self.missing_fields else 0.0


@dataclass(frozen=True)
class CitationMetricResult:
    citation_exists: bool
    citation_supported: bool
    missing_citations: list[str]
    unsupported_citations: list[str]

    @property
    def score(self) -> float:
        return 1.0 if self.citation_exists and self.citation_supported else 0.0


@dataclass(frozen=True)
class RefusalMetricResult:
    expected_refusal: bool
    refused: bool

    @property
    def correct(self) -> bool:
        return self.expected_refusal == self.refused

    @property
    def score(self) -> float:
        return 1.0 if self.correct else 0.0


def check_json_output(raw_output: str, required_fields: list[str]) -> JsonMetricResult:
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        return JsonMetricResult(
            valid=False,
            parsed=None,
            missing_fields=required_fields,
            error=str(exc),
        )

    if not isinstance(parsed, dict):
        return JsonMetricResult(
            valid=False,
            parsed=None,
            missing_fields=required_fields,
            error="json output must be an object.",
        )

    missing = [field for field in required_fields if field not in parsed]
    return JsonMetricResult(valid=True, parsed=parsed, missing_fields=missing)


def check_citations(
    output_citations: list[str],
    required_citations: list[str],
    knowledge_base: dict[str, str],
    output_text: str = "",
) -> CitationMetricResult:
    if not required_citations and not output_citations:
        return CitationMetricResult(
            citation_exists=True,
            citation_supported=True,
            missing_citations=[],
            unsupported_citations=[],
        )

    missing = [
        citation for citation in required_citations if citation not in output_citations
    ]
    unsupported = [
        citation for citation in output_citations if citation not in knowledge_base
    ]

    for citation in output_citations:
        evidence = knowledge_base.get(citation, "")
        if evidence and output_text and not _has_shared_term(output_text, evidence):
            unsupported.append(citation)

    unique_unsupported = sorted(set(unsupported))
    return CitationMetricResult(
        citation_exists=bool(output_citations) and not missing,
        citation_supported=not unique_unsupported,
        missing_citations=missing,
        unsupported_citations=unique_unsupported,
    )


def check_refusal(raw_output: str, expected_refusal: bool) -> RefusalMetricResult:
    normalized = raw_output.lower()
    refused = any(marker.lower() in normalized for marker in REFUSAL_MARKERS)
    return RefusalMetricResult(expected_refusal=expected_refusal, refused=refused)


def _has_shared_term(output_text: str, evidence: str) -> bool:
    output_terms = _content_terms(output_text)
    evidence_terms = _content_terms(evidence)
    return bool(output_terms & evidence_terms)


def _content_terms(text: str) -> set[str]:
    separators = " \t\r\n，。；：、,.!?;:()[]{}\"'"
    tokens: set[str] = set()
    current: list[str] = []
    for char in text.lower():
        if char in separators:
            if current:
                tokens.add("".join(current))
                current.clear()
            continue
        current.append(char)
    if current:
        tokens.add("".join(current))
    return {token for token in tokens if len(token) >= 2}
