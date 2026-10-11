"""Rule-based exclusions that run before any LLM call (cheap, deterministic, explainable)."""
import re
from dataclasses import dataclass

from src.models import Job

# Location heuristics. A posting is kept if ANY of its locations looks US, or it's a bare "Remote".
_US_STATES = (
    "AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ "
    "NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY DC"
).split()
_US_MARKERS = re.compile(
    r"\b(united states|usa|americas|north america|"
    r"san francisco|bay area|new york|nyc|seattle|boston|austin|chicago|los angeles|"
    r"palo alto|mountain view|menlo park|sunnyvale|san jose|san mateo|redwood city|oakland|"
    r"denver|atlanta|washington|miami|dallas|houston|pittsburgh|philadelphia|portland|"
    r"san diego|raleigh|salt lake city|foster city|santa clara)\b",
    re.I,
)
# Abbreviations are case-sensitive so ordinary words ("in", "or", "me") don't read as states.
_US_ABBREV = re.compile(r"\b(US|USA|U\.S\.|SF|NYC)\b|,\s*(" + "|".join(_US_STATES) + r")\b")
# Matched only when no US marker is present, so "London; New York" is still kept.
_NON_US_MARKERS = re.compile(
    r"\b(uk|united kingdom|london|england|ireland|dublin|canada|toronto|vancouver|montreal|"
    r"germany|berlin|munich|france|paris|netherlands|amsterdam|spain|madrid|barcelona|"
    r"switzerland|zurich|poland|warsaw|sweden|stockholm|denmark|italy|portugal|lisbon|"
    r"india|bangalore|bengaluru|hyderabad|pune|singapore|japan|tokyo|korea|seoul|china|"
    r"australia|sydney|melbourne|brazil|são paulo|sao paulo|mexico|argentina|israel|tel aviv|"
    r"emea|apac|latam|europe|asia|uae|dubai|serbia|belgrade|romania|czech|estonia)\b",
    re.I,
)


@dataclass
class FilterResult:
    keep: bool
    reason: str               # "ok" or why it was dropped — printed in the search summary
    role_hint: str | None = None


class JobFilter:
    """Compiles config patterns once; `check(job)` is then pure regex work."""

    def __init__(self, cfg: dict):
        flags = re.I
        self.families = [
            (fam["name"], [re.compile(p, flags) for p in fam["patterns"]])
            for fam in cfg["role_families"]
        ]
        self.title_exclude = [re.compile(p, flags) for p in cfg.get("title_exclude", [])]
        self.phrases = [re.compile(p, flags) for p in cfg.get("exclude_phrases", [])]
        self.bad_types = {t.lower() for t in cfg.get("exclude_employment_types", [])}
        self.us_only = cfg.get("us_only", True)

    def role_family(self, title: str) -> str | None:
        for name, patterns in self.families:
            if any(p.search(title) for p in patterns):
                return name
        return None

    @staticmethod
    def is_us_location(location: str, remote: bool) -> bool:
        loc = location.strip()
        if not loc:
            return remote  # no location at all: only trust it if the ATS says remote
        if _US_MARKERS.search(loc) or _US_ABBREV.search(loc):
            return True
        if _NON_US_MARKERS.search(loc):
            return False
        # Something like "Remote" or "Anywhere" with no country attached.
        return bool(re.search(r"\b(remote|anywhere|distributed)\b", loc, re.I))

    def check(self, job: Job) -> FilterResult:
        family = self.role_family(job.title)
        if family is None:
            return FilterResult(False, "title: not a target role")
        for p in self.title_exclude:
            if p.search(job.title):
                return FilterResult(False, "title: too senior / not IC full-time")
        if job.employment_type and any(t in job.employment_type.lower() for t in self.bad_types):
            return FilterResult(False, f"employment type: {job.employment_type}")
        if self.us_only and not self.is_us_location(job.location, job.remote):
            return FilterResult(False, "location: not US")
        for p in self.phrases:
            m = p.search(job.description)
            if m:
                return FilterResult(False, f'phrase: "{m.group(0)}"')
        return FilterResult(True, "ok", family)
