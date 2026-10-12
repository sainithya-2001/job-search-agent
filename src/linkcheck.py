"""Before tailoring a JSearch job: confirm its employer link still loads and still shows the same job title.

JSearch is an aggregator: its copy of a posting can outlive the employer's, or point at a page that now shows a
different role. The check is strict and code-only (no LLM):
  1. the page loads (HTTP < 400) and does not end up on a job board or aggregator after redirects;
  2. the job title, normalized like dedupe titles (case, punctuation and parentheticals ignored, e.g. a
     "(10569)" requisition number), appears as a phrase in the page text.
A plain HTTP fetch is tried first; career sites that render with JavaScript (Workday, iCIMS, ...) are then loaded
in headless Chromium. Anything else = unverified, and the job is skipped (never tailored).

Every job (any source) is also checked for a CLOSED posting before tailoring: "no longer accepting applications",
"position filled", "job closed", an HTTP 404/410, or Greenhouse's removed-job redirect (?error=true). Closed jobs
get status "closed" and are skipped. For non-JSearch jobs a page that can't be fetched does not block tailoring
(their link comes straight from the employer's ATS); only clear evidence of closure does.
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
    closed: bool = False          # the posting says it's closed / filled / gone: status "closed", not "unverified"


CLOSED_PATTERNS = [
    r"no longer (accepting|taking|receiving) applications",
    r"(is )?not (currently )?accepting applications",
    r"applications (are |have )?(now )?closed",
    r"(position|role|job|posting|vacancy|opening|requisition) (has been|was|is) (filled|closed)",
    r"(position|role|job|posting|vacancy|opening|requisition) (is )?no longer (available|open|active|posted)",
    r"this (job|position|posting|role|opening) (has )?(expired|closed)",
    r"job (posting )?(has )?expired",
    r"\bjob (is )?closed\b",
    r"\bposition filled\b",
    r"\bjob not found\b",
    r"the (job|position|page) you are looking for (is no longer|no longer exists|could not be found|was not found)",
]
_CLOSED = re.compile("|".join(CLOSED_PATTERNS), re.I)


def closed_reason(page_text: str) -> str | None:
    """The closure phrase found on the page, or None."""
    m = _CLOSED.search(re.sub(r"\s+", " ", page_text.replace("’", "'")))
    return m.group(0) if m else None


def _gone(status: int, final: str) -> str | None:
    if status in (404, 410):
        return f"posting returns HTTP {status}"
    if "greenhouse.io" in final and "error=true" in final:
        return "Greenhouse redirects to the job board (posting removed)"
    return None


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
        if (gone := _gone(r.status_code, final)):
            return LinkCheck(False, f"closed: {gone}", final, closed=True)
        if r.status_code >= 400:
            return LinkCheck(False, f"employer link returned HTTP {r.status_code}", final)
        if _board(final, cfg):
            return LinkCheck(False, f"employer link redirects to a job board ({host_of(final)})", final)
        if (why := closed_reason(_text_of(r.text))):
            return LinkCheck(False, f'closed: page says "{why}"', final, closed=True)
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
                if resp is not None and (gone := _gone(resp.status, final)):
                    return LinkCheck(False, f"closed: {gone}", final, closed=True)
                if resp is not None and resp.status >= 400:
                    return LinkCheck(False, f"employer link returned HTTP {resp.status}", final)
                if _board(final, cfg):
                    return LinkCheck(False, f"employer link redirects to a job board ({host_of(final)})", final)
                text = page.inner_text("body")
                if (why := closed_reason(text)):
                    return LinkCheck(False, f'closed: page says "{why}"', final, closed=True)
                if title_on_page(title, text):
                    return LinkCheck(True, "loads and shows the same title (rendered)", final)
                return LinkCheck(False, "page loads but doesn't show this job title (closed or changed)", final)
            finally:
                browser.close()
    except Exception as e:  # noqa: BLE001 — any failure to load = unverified
        return LinkCheck(False, f"{fetch_note}; rendered load failed ({type(e).__name__})")


def posting_closed(url: str, title: str, timeout: float = 20.0, render: bool = True) -> LinkCheck | None:
    """Closure check for jobs from the employer's own ATS (Greenhouse/Lever/Ashby boards). -> a closed LinkCheck,
    or None when there is no clear sign of closure (including when the page can't be fetched)."""
    if not url:
        return None
    try:
        r = httpx.get(url, follow_redirects=True, timeout=timeout, headers={"User-Agent": UA})
    except httpx.HTTPError:
        return None
    final, text = str(r.url), _text_of(r.text)
    if (gone := _gone(r.status_code, final)):
        return LinkCheck(False, f"closed: {gone}", final, closed=True)
    if (why := closed_reason(text)):
        return LinkCheck(False, f'closed: page says "{why}"', final, closed=True)
    if title_on_page(title, text) or not render:
        return None
    # The HTML is a JavaScript shell (e.g. Ashby): look at the rendered page.
    from playwright.sync_api import sync_playwright
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            try:
                page = browser.new_page(user_agent=UA)
                resp = page.goto(url, wait_until="networkidle", timeout=45000)
                page.wait_for_timeout(1500)
                if resp is not None and (gone := _gone(resp.status, page.url)):
                    return LinkCheck(False, f"closed: {gone}", page.url, closed=True)
                if (why := closed_reason(page.inner_text("body"))):
                    return LinkCheck(False, f'closed: page says "{why}"', page.url, closed=True)
            finally:
                browser.close()
    except Exception:  # noqa: BLE001 — can't render: no evidence of closure
        return None
    return None


def check_before_tailoring(job, cfg: dict) -> LinkCheck | None:
    """None = go ahead and tailor. JSearch jobs need a verified employer link; every job must not be closed."""
    if job["source"] == "jsearch":
        check = verify_posting(job["url"], job["title"], cfg.get("jsearch", {}))
        return None if check.ok else check
    return posting_closed(job["url"], job["title"])
