from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Status(StrEnum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class VerificationStatus(StrEnum):
    VALID = "valid"
    INVALID = "invalid"
    UNVERIFIED = "unverified"
    SKIPPED = "skipped"


class Finding(BaseModel):
    id: str
    file: str
    line: int
    category: str
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)
    title: str
    evidence: str
    consequence: str
    suggestion: str
    introduced_by_diff: bool
    actionable: bool
    style_only: bool
    duplicate_of: str | None = None
    reviewer: str = ""
    status: Status = Status.PROPOSED
    rejection_reason: str = ""
    verification_status: VerificationStatus | None = None
    verification_verdict: str = ""
    verification_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    verification_reason: str = ""
    verification_evidence_lines: list[int] = Field(default_factory=list)
    verification_model: str = ""
    verification_elapsed_ms: int | None = None
    verification_context_line_start: int | None = None
    verification_context_line_end: int | None = None
    verification_line_in_context: bool | None = None
    verification_prompt_chars: int | None = None
    verification_response_text: str = ""
    verification_contradiction_code: str = "none"


class FindingsPayload(BaseModel):
    findings: list[Finding] = Field(default_factory=list)


class ReviewerFailure(BaseModel):
    reviewer: str
    chunk_index: int
    error_type: str
    message: str


class ReviewerSkip(BaseModel):
    reviewer: str
    chunk_index: int
    reason: str
    message: str = ""


@dataclass(slots=True)
class DiffFile:
    status: str
    old_path: Path | None
    path: Path
    changed_lines: set[int] = field(default_factory=set)


@dataclass(slots=True)
class DiffSnapshot:
    base: str
    head: str
    changed_files: list[DiffFile]
    diff_text: str
    diff_lines: int


@dataclass(slots=True)
class ReviewContext:
    base: str
    head: str
    diff_text: str
    file_contexts: dict[str, str]


class ReviewMetadata(BaseModel):
    base: str
    head: str
    changed_files: int
    diff_lines: int
    reviewers: list[str]
    review_mode: str = "separate"
    reviewer_failures: list[ReviewerFailure] = Field(default_factory=list)
    reviewer_skips: list[ReviewerSkip] = Field(default_factory=list)
    completed_requests: int = 0
    failed_requests: int = 0
    planned_requests: int = 0
    skipped_requests: int = 0
    reviewable_chunks: int = 0
    skipped_chunks: int = 0
    chunk_count: int = 0
    total_elapsed_seconds: float = 0.0
    total_time_budget_seconds: float = 0.0
    verification_enabled: bool = False
    verification_model: str = ""
    verification_fail_policy: str = "unverified"
    verification_uncertain_policy: str = "unverified"
    verification_requests_planned: int = 0
    verification_requests_completed: int = 0
    verification_requests_failed: int = 0
    verification_requests_skipped: int = 0
    verification_valid_count: int = 0
    verification_invalid_count: int = 0
    verification_uncertain_count: int = 0
    verification_unverified_count: int = 0
    verification_skipped_count: int = 0
    verification_elapsed_seconds: float = 0.0


class ReviewResult(BaseModel):
    metadata: ReviewMetadata
    candidate_findings: list[Finding] = Field(default_factory=list)
    verified_findings: list[Finding] = Field(default_factory=list)
    verification_rejected_findings: list[Finding] = Field(default_factory=list)
    accepted_findings: list[Finding]
    rejected_findings: list[Finding]
