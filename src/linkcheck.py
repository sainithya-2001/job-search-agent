"""Before tailoring a JSearch job: confirm its employer link still loads and still shows the same job title.

JSearch is an aggregator: its copy of a posting can outlive the employer's, or point at a page that now shows a
different role. The check is strict and code-only (no LLM):
  1. the page loads (HTTP < 400) and does not end up on a job board or aggregator after redirects;
  2. the job title, normalized like dedupe titles (case, punctuation and parentheticals ignored, e.g. a
     "(10569)" requisition number), appears as a phrase in the page text.
A plain HTTP fetch is tried first; career sites that render with JavaScript (Workday, iCIMS, ...) are then loaded
in headless Chromium. Anything else = unverified, and the job is skipped (never tailored).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import httpx

from src.sources.jsearch import _host_in, host_of
from src.store import normalize_title

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/126.0 Safari/537.36")


@dataclass
class LinkCheck:
    ok: bool
    note: str
    final_url: str | None = None


def _text_of(html: str) -> str:
    html = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", html)
    return re.sub(r"<[^>]+>", " ", html)


def _phrase(text: str) -> str:
    return " " + re.sub(r"\s+", " ", re.sub(r"[^a-z0-9+/ ]+", " ", text.lower())).strip() + " "


def title_on_page(title: str, page_text: str) -> bool:
    key = normalize_title(title)
    return bool(key) and f" {key} " in _phrase(page_text)


def _board(url: str, cfg: dict) -> bool:
    return _host_in(host_of(url), list(cfg.get("aggregator_hosts", [])) + list(cfg.get("job_board_hosts", [])))


def verify_posting(url: str, title: str, cfg: dict, timeout: float = 20.0, render: bool = True) -> LinkCheck:
    """cfg: config.yaml `jsearch` section (for the job-board host lists)."""
    if not url:
        return LinkCheck(False, "no employer link")
    try:
        r = httpx.get(url, follow_redirects=True, timeout=timeout, headers={"User-Agent": UA})
        final = str(r.url)
        if r.status_code >= 400:
            return LinkCheck(False, f"employer link returned HTTP {r.status_code}", final)
        if _board(final, cfg):
            return LinkCheck(False, f"employer link redirects to a job board ({host_of(final)})", final)
        if title_on_page(title, _text_of(r.text)):
            return LinkCheck(True, "loads and shows the same title", final)
        fetch_note = "page loads but the title isn't in its HTML"
    except httpx.HTTPError as e:
        fetch_note = f"fetch failed ({type(e).__name__})"
    if not render:
        return LinkCheck(False, fetch_note)
    return _verify_rendered(url, title, cfg, fetch_note)


def _verify_rendered(url: str, title: str, cfg: dict, fetch_note: str) -> LinkCheck:
    from playwright.sync_api import sync_playwright
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            try:
                page = browser.new_page(user_agent=UA)
                resp = page.goto(url, wait_until="networkidle", timeout=45000)
                page.wait_for_timeout(1500)          # late client-side rendering
                final = page.url
                if resp is not None and resp.status >= 400:
                    return LinkCheck(False, f"employer link returned HTTP {resp.status}", final)
                if _board(final, cfg):
                    return LinkCheck(False, f"employer link redirects to a job board ({host_of(final)})", final)
                if title_on_page(title, page.inner_text("body")):
                    return LinkCheck(True, "loads and shows the same title (rendered)", final)
                return LinkCheck(False, "page loads but doesn't show this job title (closed or changed)", final)
            finally:
                browser.close()
    except Exception as e:  # noqa: BLE001 — any failure to load = unverified
        return LinkCheck(False, f"{fetch_note}; rendered load failed ({type(e).__name__})")
