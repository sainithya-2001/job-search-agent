"""Rule-based exclusions that run before any LLM call (cheap, deterministic, explainable)."""
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from src.models import Job

_STATE_NAMES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA", "colorado": "CO",
    "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS", "kentucky": "KY", "louisiana": "LA",
    "maine": "ME", "maryland": "MD", "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY", "north carolina": "NC",
    "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
    "rhode island": "RI", "south carolina": "SC", "south dakota": "SD", "tennessee": "TN", "texas": "TX",
    "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC",
}
_ABBREVS = "|".join(sorted(set(_STATE_NAMES.values())))
_STATE_NAME_RE = re.compile(r"\b(" + "|".join(sorted(_STATE_NAMES, key=len, reverse=True)) + r")\b", re.I)
# Abbreviations only in positions where they can't be ordinary words: ", CA"  "(CA)"  "US-CA-"  "- CA"
_STATE_ABBR_RE = re.compile(r"(?:,\s*|\(|\bUS-|-\s)(" + _ABBREVS + r")\b")

_US_CITIES = re.compile(
    r"\b(san francisco|bay area|new york|nyc|seattle|boston|austin|chicago|los angeles|palo alto|mountain view|"
    r"menlo park|sunnyvale|san jose|san mateo|redwood city|oakland|denver|atlanta|miami|dallas|houston|"
    r"pittsburgh|philadelphia|portland|san diego|raleigh|salt lake city|foster city|santa clara|bellevue|"
    r"brooklyn|washington,? d\.?c\.?)\b", re.I)
# "US" is matched case-sensitively so the word "us" in a sentence doesn't count.
_US_GENERIC = re.compile(r"\b(united states|usa|u\.s\.a?|us-based|north america)\b|(?-i:\bUS\b)", re.I)
_NON_US = re.compile(
    r"\b(uk|united kingdom|london|england|ireland|dublin|canada|toronto|vancouver|montreal|ontario|"
    r"germany|berlin|munich|france|paris|netherlands|amsterdam|spain|madrid|barcelona|switzerland|zurich|"
    r"poland|warsaw|sweden|stockholm|denmark|italy|portugal|lisbon|india|bangalore|bengaluru|hyderabad|pune|"
    r"singapore|japan|tokyo|korea|seoul|china|australia|sydney|melbourne|brazil|são paulo|sao paulo|mexico|"
    r"argentina|colombia|israel|tel aviv|emea|apac|latam|europe|asia|uae|dubai|serbia|belgrade|romania|czech|"
    r"estonia|taiwan|malaysia|delhi|british columbia|alberta|eu)\b", re.I)
_REMOTE = re.compile(r"\b(remote|anywhere|distributed|work from home)\b", re.I)
# Posting text that makes a bare "Remote" location US-remote.
_DESC_US_REMOTE = re.compile(
    r"\b(remote|work from home)\b[^.\n]{0,60}(\b(united states|u\.s\.a?)|(?-i:\bUSA?\b))"
    r"|(\b(united states|u\.s\.)|(?-i:\bUS\b))[- ]based\b|\banywhere in the (us|u\.s\.|united states)\b", re.I)
# "must reside in Colorado or Utah" — remote roles restricted to particular states.
_DESC_STATE_RESTRICTION = re.compile(
    r"\b(?:must|required to|need to|needs to)\s+(?:currently\s+)?(?:reside|live|be located|be based)\s+(?:with)?in\s+"
    r"(?:the\s+states?\s+of\s+|one of the following (?:states|locations):?\s*)?([^.\n]{0,120})", re.I)


def _states_in(text: str) -> set[str]:
    found = {_STATE_NAMES[m.group(1).lower()] for m in _STATE_NAME_RE.finditer(text)}
    return found | {m.group(1) for m in _STATE_ABBR_RE.finditer(text)}


@dataclass
class FilterResult:
    keep: bool
    reason: str               # "ok" or why it was dropped — printed in the search summary
    role_hint: str | None = None


class JobFilter:
    """Compiles config patterns once; `check(job)` is then pure regex work."""

    def __init__(self, cfg: dict, us_hq_companies: set[str] | None = None, now: datetime | None = None):
        """cfg = config.yaml `filters`. us_hq_companies: names whose bare "Remote" jobs are kept."""
        flags = re.I
        # Families are listed in PRIORITY order (used for ranking). Families with match_first: true are tried
        # before the rest, so "Product Manager, Applied AI" is a program_product_manager role even though that
        # family ranks last.
        fams = sorted(cfg["role_families"], key=lambda f: not f.get("match_first", False))
        self.families = [
            (fam["name"], [re.compile(p, flags) for p in fam["patterns"]],
             [re.compile(p, flags) for p in fam.get("exclude", [])])
            for fam in fams
        ]
        self.us_hq = us_hq_companies or set()
        self.max_age = cfg.get("max_posting_age_days")
        self.now = now or datetime.now(timezone.utc)
        self.title_exclude = [re.compile(p, flags) for p in cfg.get("title_exclude", [])]
        self.unless_ai = [re.compile(p, flags) for p in cfg.get("title_exclude_unless_ai", [])]
        self.except_families = [(re.compile(r["pattern"], flags), set(r["families"]))
                                for r in cfg.get("title_exclude_except_families", [])]
        self.ai_terms = [re.compile(p, flags) for p in cfg.get("ai_title_terms", [])]
        self.phrases = [re.compile(p, flags) for p in cfg.get("exclude_phrases", [])]
        self.bad_types = {t.lower() for t in cfg.get("exclude_employment_types", [])}
        self.us_only = cfg.get("us_only", True)
        # Remote roles restricted to any US state pass (Nithya will relocate anywhere in the US). These states only
        # matter for ranking: remote roles restricted to other states rank after GA/CA/TX and unrestricted US-remote.
        self.preferred_states = set(cfg.get("remote_preferred_states", ["GA", "CA", "TX"]))

    def role_family(self, title: str) -> str | None:
        for name, patterns, excludes in self.families:
            if any(p.search(title) for p in patterns) and not any(x.search(title) for x in excludes):
                return name
        return None

    def _segment(self, seg: str) -> str:
        """Classify one location segment: ok | non_us | bare_remote | unknown."""
        states, generic = _states_in(seg), bool(_US_GENERIC.search(seg))
        city, non_us = bool(_US_CITIES.search(seg)), bool(_NON_US.search(seg))
        if _REMOTE.search(seg):
            if states:                       # "Remote - Colorado", "Remote (CA, NY)": any US state passes
                return "ok"
            if generic or city:              # "Remote - US", "Remote (United States)", "Remote / NYC"
                return "ok"
            return "non_us" if non_us else "bare_remote"
        if generic or states or city:
            return "ok"
        return "non_us" if non_us else "unknown"

    def location_reason(self, job: Job) -> str | None:
        """None if the location is acceptable, else the exclusion reason."""
        segments = [s.strip() for s in re.split(r"[;|/\n]| or ", job.location or "") if s.strip()]
        if not segments:
            segments = ["Remote"] if job.remote else []
        kinds = [self._segment(s) for s in segments]
        if "bare_remote" in kinds and (job.company in self.us_hq or _DESC_US_REMOTE.search(job.description or "")):
            kinds = ["ok" if k == "bare_remote" else k for k in kinds]
        if "ok" in kinds:
            return None
        if "bare_remote" in kinds:
            return "location: remote, country not stated"
        return "location: not US"

    def remote_restricted_elsewhere(self, location: str, remote: bool, description: str = "") -> bool:
        """True for a remote-only role limited to US states, none of which is preferred (GA/CA/TX): e.g.
        "Remote - Colorado", or "Remote - US" whose posting says "must reside in Colorado or Utah". Used for ranking
        only; such roles pass the filters."""
        segments = [s.strip() for s in re.split(r"[;|/\n]| or ", location or "") if s.strip()]
        if not (remote or any(_REMOTE.search(s) for s in segments)) or not segments \
                or not all(_REMOTE.search(s) for s in segments):
            return False
        per_segment = [_states_in(s) for s in segments]
        # A segment naming no state ("Remote - US") is open to the whole US, unless the posting text restricts it.
        states = set() if any(not st for st in per_segment) else set().union(*per_segment)
        m = _DESC_STATE_RESTRICTION.search(description or "")
        if m and not _US_GENERIC.search(m.group(1)):
            states |= _states_in(m.group(1))
        return bool(states) and not states & self.preferred_states

    def check(self, job: Job) -> FilterResult:
        family = self.role_family(job.title)
        if family is None:
            return FilterResult(False, "title: not a target role")
        for p in self.title_exclude:
            if p.search(job.title):
                return FilterResult(False, "title: excluded seniority / specialty")
        for p, allowed in self.except_families:
            if p.search(job.title) and family not in allowed:
                return FilterResult(False, "title: excluded seniority / specialty")
        if any(p.search(job.title) for p in self.unless_ai) and not any(p.search(job.title) for p in self.ai_terms):
            return FilterResult(False, "title: non-AI mobile/embedded/backend")
        if self.max_age and job.posted_at:
            posted = job.posted_at if job.posted_at.tzinfo else job.posted_at.replace(tzinfo=timezone.utc)
            age = (self.now - posted).days
            if age > self.max_age:
                return FilterResult(False, f"posted more than {self.max_age} days ago")
        if job.employment_type and any(t in job.employment_type.lower() for t in self.bad_types):
            return FilterResult(False, f"employment type: {job.employment_type}")
        if self.us_only:
            reason = self.location_reason(job)
            if reason:
                return FilterResult(False, reason)
        for p in self.phrases:
            m = p.search(job.description)
            if m:
                return FilterResult(False, f'phrase: "{m.group(0)}"')
        return FilterResult(True, "ok", family)
