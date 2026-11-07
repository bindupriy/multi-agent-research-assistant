"""Interfaces for future swappable agent implementations."""

from typing import Protocol, Sequence

from research_assistant.models import (
    Critique,
    DraftReport,
    Evidence,
    ResearchPlan,
    SearchTask,
)


class Planner(Protocol):
    def plan(self, question: str) -> ResearchPlan:
        """Break a question into research tasks."""
        ...


class Retriever(Protocol):
    def retrieve(self, task: SearchTask) -> Sequence[Evidence]:
        """Find evidence for a task, with source locations intact."""
        ...


class Writer(Protocol):
    def write(self, plan: ResearchPlan, evidence: Sequence[Evidence]) -> DraftReport:
        """Synthesize explicit claims tied to evidence IDs."""
        ...


class Critic(Protocol):
    def review(
        self, plan: ResearchPlan, evidence: Sequence[Evidence], draft: DraftReport
    ) -> Critique:
        """Judge factual support and coverage before publication."""
        ...