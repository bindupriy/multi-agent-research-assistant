"""Immutable data passed between the planned research agents."""

from dataclasses import dataclass
from typing import Literal


SourceKind = Literal["local", "web"]


@dataclass(frozen=True)
class SearchTask:
    id: str
    query: str
    sources: tuple[SourceKind, ...]


@dataclass(frozen=True)
class ResearchPlan:
    question: str
    tasks: tuple[SearchTask, ...]


@dataclass(frozen=True)
class Evidence:
    id: str
    task_id: str
    title: str
    uri: str
    locator: str
    excerpt: str
    kind: SourceKind


@dataclass(frozen=True)
class Claim:
    text: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class DraftReport:
    title: str
    claims: tuple[Claim, ...]


@dataclass(frozen=True)
class Critique:
    approved: bool
    issues: tuple[str, ...]