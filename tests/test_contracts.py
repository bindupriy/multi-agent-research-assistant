import unittest
from dataclasses import FrozenInstanceError

from research_assistant.models import (
    Claim,
    Critique,
    DraftReport,
    Evidence,
    ResearchPlan,
    SearchTask,
)


class ContractTests(unittest.TestCase):
    def test_plan_and_evidence_preserve_source_and_task(self) -> None:
        task = SearchTask(id="q1", query="What changed?", sources=("local", "web"))
        plan = ResearchPlan(question="Summarize the changes", tasks=(task,))
        evidence = Evidence(
            id="e1",
            task_id="q1",
            title="Example document",
            uri="data/example.pdf",
            locator="page 2",
            excerpt="Example excerpt.",
            kind="local",
        )

        self.assertEqual(plan.tasks[0].id, evidence.task_id)
        self.assertEqual(evidence.locator, "page 2")
        self.assertEqual(evidence.kind, "local")

    def test_report_and_critique_are_immutable(self) -> None:
        draft = DraftReport("Example", (Claim("A claim", ("e1",)),))
        critique = Critique(approved=False, issues=("Not verified",))

        self.assertEqual(draft.claims[0].evidence_ids, ("e1",))
        self.assertFalse(critique.approved)
        with self.assertRaises(FrozenInstanceError):
            draft.title = "Changed"