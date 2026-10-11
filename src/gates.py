"""Shadow-mode auto-submit gates.

There is NO submit code in this project. These gates only answer "would this application have qualified for
auto-submit?" and record why or why not, so the rules can be judged on real data before anyone considers
building a submit path. Every application goes to the review queue regardless of the result.

An application qualifies only if ALL gates pass:
  fit_score >= apply.min_fit (70) | ATS is Greenhouse or Lever | no CAPTCHA of any kind on the page |
  every question matched exactly from the answer bank (optional ones may be bank-skipped) |
  no long-answer box | fewer than apply.max_per_day qualifying today | no application to the same company in the
  last apply.company_cooldown_days (90) days, manual ones included | resume verified (PASS) with a one-page PDF |
  posting still open (the form loaded)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from src.answers import ANSWERED, SKIPPED, Match


class AutoSubmitNotSupported(RuntimeError):
    pass


def check_config(apply_cfg: dict) -> None:
    if apply_cfg.get("auto_submit"):
        raise AutoSubmitNotSupported(
            "apply.auto_submit is true, but this build has no submit code (shadow mode only). Set it back to false.")


@dataclass
class Gate:
    name: str
    passed: bool
    detail: str = ""


def evaluate(*, fit_score: int | None, ats: str, captcha: list[str], matches: list[Match], company: str,
             resume_ok: bool, form_loaded: bool, history: list[dict], qualifying_today: int, cfg: dict,
             today: date | None = None) -> list[Gate]:
    """history: rows of {company, status, date_applied} for every logged application (manual ones included)."""
    today = today or date.today()
    min_fit, allowed = cfg.get("min_fit", 70), cfg.get("auto_ats", ["greenhouse", "lever"])
    per_day, cooldown = cfg.get("max_per_day", 5), cfg.get("company_cooldown_days", 90)

    unmatched = [m.label for m in matches if not (m.status == ANSWERED or (m.status == SKIPPED and not m.required))]
    long_answers = [m.label for m in matches if m.kind == "textarea"]
    since = today - timedelta(days=cooldown)
    recent = [h for h in history if h["company"] == company and h.get("status") == "applied"
              and h.get("date_applied") and date.fromisoformat(h["date_applied"][:10]) >= since]

    return [
        Gate("fit_score", fit_score is not None and fit_score >= min_fit, f"{fit_score} (needs >= {min_fit})"),
        Gate("ats", ats in allowed, f"{ats} (allowed: {', '.join(allowed)})"),
        Gate("no_captcha", not captcha, ", ".join(captcha) or "none found"),
        Gate("all_matched", form_loaded and not unmatched,
             f"{len(unmatched)} not matched exactly: {'; '.join(unmatched[:6])}" if unmatched else "all matched"),
        Gate("no_long_answer", not long_answers, "; ".join(long_answers[:4]) or "none"),
        Gate("daily_cap", qualifying_today < per_day, f"{qualifying_today} qualified today (max {per_day})"),
        Gate("company_cooldown", not recent,
             f"applied {recent[0]['date_applied'][:10]}" if recent else f"no application in {cooldown} days"),
        Gate("resume_verified", resume_ok, "PASS, one page" if resume_ok else "no verified one-page PDF"),
        Gate("posting_open", form_loaded, "form loaded" if form_loaded else "form not found"),
    ]


def qualifies(gates: list[Gate]) -> bool:
    return all(g.passed for g in gates)


def blockers(gates: list[Gate]) -> list[str]:
    return [f"{g.name}: {g.detail}" for g in gates if not g.passed]
