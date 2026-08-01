from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from reviewer.models import Finding


class ExpectedFinding(BaseModel):
    file: str
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    category: str
    severity: list[str] = Field(default_factory=list)
    concept: str
    concept_id: str | None = None


class CaseGroundTruth(BaseModel):
    case_id: str
    clean: bool
    expected_findings: list[ExpectedFinding] = Field(default_factory=list)
    forbidden_concepts: list[str] = Field(default_factory=list)


class CasePaths(BaseModel):
    case_id: str
    case_dir: Path
    base_dir: Path
    head_dir: Path
    expected_file: Path


class CandidateRunConfig(BaseModel):
    provider: str
    model: str
    review_mode: str
    llm_timeout: float
    max_review_seconds: float
    reviewers: list[str] = Field(default_factory=list)
    max_prompt_tokens: int = 3500
    test_max_prompt_tokens: int = 2500
    max_findings_per_request: int = 3
    reviewer_prompt_fingerprints: dict[str, str] = Field(default_factory=dict)
    deterministic_seed: int | None = None
    sampling_params: dict[str, object] = Field(default_factory=dict)


class ReplayCaseSource(BaseModel):
    pr: int
    commit: str
    file: str
    line: int = Field(ge=1)


class CandidateCaseRecord(BaseModel):
    case_id: str
    source_identifier: str
    expected: CaseGroundTruth
    candidate_findings: list[Finding] = Field(default_factory=list)
    source: ReplayCaseSource | None = None
    expected_verdict: Literal["valid", "invalid", "uncertain"] | None = None


class CandidateDataset(BaseModel):
    schema_version: int = 1
    run_id: str
    timestamp_utc: str
    run_config: CandidateRunConfig
    cases: list[CandidateCaseRecord] = Field(default_factory=list)


def load_ground_truth(path: Path) -> CaseGroundTruth:
    data = path.read_text(encoding="utf-8")
    return CaseGroundTruth.model_validate_json(data)


def load_candidate_dataset(path: Path) -> CandidateDataset:
    data = json.loads(path.read_text(encoding="utf-8"))
    return CandidateDataset.model_validate(data)
