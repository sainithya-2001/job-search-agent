"""Data models shared across the pipeline. ScoredJob / TailoredResume arrive in Phases 2–3."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel

Source = Literal["greenhouse", "lever", "ashby", "jsearch"]


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
