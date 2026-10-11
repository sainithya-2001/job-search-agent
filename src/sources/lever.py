"""Lever public postings API: api.lever.co/v0/postings/{slug}?mode=json"""
import httpx

from src.models import Job
from src.sources.common import html_to_text, parse_datetime

URL = "https://api.lever.co/v0/postings/{slug}?mode=json"


def _description(raw: dict) -> str:
    # Lever splits a posting into an intro, titled bullet lists, and a closing section.
    parts = [raw.get("descriptionPlain") or html_to_text(raw.get("description"))]
    for section in raw.get("lists", []):
        parts.append(section.get("text", ""))
        parts.append(html_to_text(section.get("content")))
    parts.append(raw.get("additionalPlain") or html_to_text(raw.get("additional")))
    return "\n\n".join(p.strip() for p in parts if p and p.strip())


def parse(company: str, slug: str, payload: list) -> list[Job]:
    jobs = []
    for raw in payload:
        cats = raw.get("categories") or {}
        locations = cats.get("allLocations") or [cats.get("location") or ""]
        location = "; ".join(loc for loc in locations if loc)
        jobs.append(Job(
            id=f"lever:{slug}:{raw['id']}",
            company=company,
            title=raw.get("text", "").strip(),
            location=location,
            remote=raw.get("workplaceType") == "remote" or "remote" in location.lower(),
            url=raw.get("hostedUrl", ""),
            description=_description(raw),
            source="lever",
            posted_at=parse_datetime(raw.get("createdAt")),
            employment_type=cats.get("commitment"),
        ))
    return jobs


async def fetch(client: httpx.AsyncClient, company: str, slug: str) -> list[Job]:
    resp = await client.get(URL.format(slug=slug))
    resp.raise_for_status()
    return parse(company, slug, resp.json())
