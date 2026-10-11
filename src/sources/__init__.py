"""Fetch all configured boards concurrently. One failing board never sinks the whole run."""
import asyncio
from dataclasses import dataclass, field

import httpx

from src.models import Job
from src.sources import ashby, greenhouse, lever

FETCHERS = {"greenhouse": greenhouse.fetch, "lever": lever.fetch, "ashby": ashby.fetch}


@dataclass
class FetchResult:
    jobs: list[Job] = field(default_factory=list)
    per_board: dict[str, int] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)


async def fetch_all(companies: list[dict], concurrency: int = 10) -> FetchResult:
    result = FetchResult()
    sem = asyncio.Semaphore(concurrency)  # be polite to the public APIs

    async def one(client: httpx.AsyncClient, c: dict) -> None:
        key = f"{c['ats']}:{c['slug']}"
        async with sem:
            try:
                jobs = await FETCHERS[c["ats"]](client, c["name"], c["slug"])
            except Exception as e:  # noqa: BLE001 — record and continue
                result.errors[key] = f"{type(e).__name__}: {e}"
                return
        result.jobs.extend(jobs)
        result.per_board[key] = len(jobs)

    headers = {"User-Agent": "job-agent/0.1 (personal job search)"}
    async with httpx.AsyncClient(timeout=30, headers=headers, follow_redirects=True) as client:
        await asyncio.gather(*(one(client, c) for c in companies))
    return result
