"""Data models shared across the pipeline. TailoredResume arrives in Phase 3."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Source = Literal["greenhouse", "lever", "ashby", "jsearch"]
# Must match the family names in config.yaml filters.role_families (tests enforce this), plus "other".
RoleFamily = Literal["ai_engineer", "ai_solutions", "forward_deployed", "ai_consultant", "ai_eval",
                     "analyst_backup", "other"]


class Job(BaseModel):
    # Globally unique: "{source}:{board_slug}:{ats_job_id}", so IDs never collide across ATSs.
    id: str
    company: str
    title: str
    location: str
    remote: bool
    url: str
    description: str  # plain text (HTML stripped), ready for filters and LLM prompts
    source: Source
    posted_at: datetime | None = None
    # Not in the core spec, but the ATS tells us this directly and it's the most
    # reliable signal for the "no contract-only" filter.
    employment_type: str | None = None


class SponsorshipSignal(BaseModel):
    signal: Literal["yes", "no", "unclear"]
    # Verbatim sentence from the posting that supports the signal; null when the posting says nothing.
    quote: str | None = None


class ScoredJob(BaseModel):
    """What the LLM must return for each job (enforced via structured output)."""
    fit_score: int = Field(ge=0, le=100, description="0-100 fit of the candidate's resume to this job")
    role_family: RoleFamily
    matched_skills: list[str] = Field(description="Skills from the candidate's resume that the job asks for")
    gaps: list[str] = Field(description="Requirements in the posting the resume does not show")
    sponsorship_signal: SponsorshipSignal
    is_staffing_firm: bool
    seniority_ok: bool = Field(description="True if the role's required experience is 5 years or less")
    one_line_reason: str = Field(description="One sentence explaining the score")
