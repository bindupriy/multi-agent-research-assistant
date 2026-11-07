"""Structural citation checks; these do not verify factual support."""

from typing import Sequence

from research_assistant.models import DraftReport, Evidence


def find_citation_issues(
    draft: DraftReport, evidence: Sequence[Evidence]
) -> tuple[str, ...]:
    """Report missing or ambiguous evidence links in a draft."""
    issues: list[str] = []
    known_ids: set[str] = set()

    for item in evidence:
        if item.id in known_ids:
            issues.append(f"Duplicate evidence ID: {item.id}")
        known_ids.add(item.id)

    if not draft.claims:
        issues.append("Draft has no claims.")

    for index, claim in enumerate(draft.claims, start=1):
        if not claim.evidence_ids:
            issues.append(f"Claim {index} has no citations.")
        for evidence_id in claim.evidence_ids:
            if evidence_id not in known_ids:
                issues.append(f"Claim {index} cites unknown evidence ID: {evidence_id}")

    return tuple(issues)