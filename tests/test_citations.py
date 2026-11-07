import unittest

from research_assistant.citations import find_citation_issues
from research_assistant.models import Claim, DraftReport, Evidence


def example_evidence() -> Evidence:
    return Evidence(
        id="e1",
        task_id="q1",
        title="Example document",
        uri="data/example.pdf",
        locator="page 2",
        excerpt="An illustrative excerpt, not a verified fact.",
        kind="local",
    )


class CitationTests(unittest.TestCase):
    def test_linked_claim_has_no_structural_issues(self) -> None:
        draft = DraftReport("Test", (Claim("Illustrative claim", ("e1",)),))
        self.assertEqual(find_citation_issues(draft, (example_evidence(),)), ())

    def test_missing_and_unknown_citations_are_reported(self) -> None:
        draft = DraftReport(
            "Test", (Claim("Uncited", ()), Claim("Unknown citation", ("e2",)))
        )
        self.assertEqual(
            find_citation_issues(draft, (example_evidence(),)),
            ("Claim 1 has no citations.", "Claim 2 cites unknown evidence ID: e2"),
        )

    def test_duplicate_evidence_ids_are_ambiguous(self) -> None:
        draft = DraftReport("Test", (Claim("Illustrative claim", ("e1",)),))
        self.assertEqual(
            find_citation_issues(draft, (example_evidence(), example_evidence())),
            ("Duplicate evidence ID: e1",),
        )

    def test_empty_report_is_not_accepted(self) -> None:
        self.assertEqual(
            find_citation_issues(DraftReport("Test", ()), ()),
            ("Draft has no claims.",),
        )


if __name__ == "__main__":
    unittest.main()