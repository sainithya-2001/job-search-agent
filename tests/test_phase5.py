"""Phase 5: answer-bank matching, shadow gates, form reading/filling (local HTML, no network), submit safeguards,
review-outcome classification, Sheets upsert, and a scan proving there is no submit code."""
import json
import re
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import answers as A
from src.answers import AnswerBank, BankError, Entry, match_question
from src.apply import classify, confirm
from src.forms import FormQuestion, SubmitRefused, block_submits, fill, hand_over, read_questions, safe_click, snapshot
from src.gates import AutoSubmitNotSupported, check_config, evaluate, qualifies
from src.models import Job
from src.sheets import COLUMNS, rows_for, upsert
from src.store import Store

ROOT = Path(__file__).resolve().parent.parent


def Q(label, kind="text", required=True, options=(), key=None):
    return FormQuestion(key or label, label, kind, required, list(options), f"[name='{key or label}']")


def bank(*entries, **tokens):
    return AnswerBank(list(entries), {f"@{k}": v for k, v in tokens.items()})


# ---- answer bank matching ---------------------------------------------------------------------------------------
def test_exact_match_ignores_case_spacing_punctuation_only():
    b = bank(Entry("auth", ["Are you legally authorized to work in the US?"], "Yes"))
    assert match_question(Q("are you LEGALLY authorized to work in the U.S.  *", "yesno", options=["Yes", "No"]), b).status == A.NO_ENTRY
    m = match_question(Q("Are you legally authorized to work in the US? *", "yesno", options=["Yes", "No"]), b)
    assert m.status == A.ANSWERED and m.value == "Yes"
    # close but not identical wording is NOT matched (no fuzzy matching)
    assert match_question(Q("Are you authorized to work in the US?", "yesno", options=["Yes", "No"]), b).status == A.NO_ENTRY


def test_null_answer_and_skip_rules():
    b = bank(Entry("salary", ["Salary expectations"], None), Entry("site", ["Website"], skip=True))
    assert match_question(Q("Salary expectations"), b).status == A.NEEDS_ANSWER
    assert match_question(Q("Website", required=False), b).status == A.SKIPPED
    assert match_question(Q("Website", required=True), b).status == A.NEEDS_ANSWER   # can't skip a required one


def test_choice_answers_must_equal_an_offered_option():
    decline = ["Decline To Self Identify", "I don't wish to answer"]
    b = bank(Entry("vet", ["Veteran Status"], decline))
    m = match_question(Q("Veteran Status", "combobox", options=["I am a veteran", "I don't wish to answer"]), b)
    assert m.status == A.ANSWERED and m.value == "I don't wish to answer"
    m = match_question(Q("Veteran Status", "combobox", options=["Yes", "No"]), b)
    assert m.status == A.NO_OPTION and m.value is None
    assert match_question(Q("Veteran Status", "combobox", options=[]), b).status == A.UNVERIFIED


def test_multi_select_requires_every_checked_option():
    b = bank(Entry("heard", ["How did you hear?"], check=["LinkedIn", "Friend"]))
    ok = match_question(Q("How did you hear?", "checkbox_group", options=["Friend", "LinkedIn", "Other"]), b)
    assert ok.status == A.ANSWERED and ok.value == ["LinkedIn", "Friend"]
    assert match_question(Q("How did you hear?", "checkbox_group", options=["LinkedIn"]), b).status == A.NO_OPTION


def test_tokens_and_missing_resume():
    b = bank(Entry("email", ["Email"], "@email"), Entry("cv", ["Resume"], "@resume"), email="n@x.com", resume=None)
    assert match_question(Q("Email", "email"), b).value == "n@x.com"
    assert match_question(Q("Resume", "file"), b).status == A.NEEDS_ANSWER       # no verified PDF -> left for her


def test_text_answer_cannot_be_a_list_and_long_answers_still_fill():
    b = bank(Entry("a", ["Why us?"], "Because."), Entry("b", ["Name"], ["x", "y"]))
    assert match_question(Q("Why us?", "textarea"), b).value == "Because."       # filled in review; gate blocks auto
    assert match_question(Q("Name"), b).status == A.NO_OPTION


def test_same_question_in_two_entries_is_rejected():
    with pytest.raises(BankError):
        bank(Entry("a", ["Email"], "x"), Entry("b", ["EMAIL?"], "y"))


def test_example_bank_loads_and_declines_eeo(tmp_path):
    b = A.load_bank(ROOT / "data" / "answer_bank.example.yaml", tokens={"@full_name": "N", "@email": "e",
                                                                     "@phone": "p", "@location": "l"})
    for eid in ("eeo_gender", "eeo_hispanic", "eeo_race", "eeo_veteran", "eeo_disability"):
        e = next(x for x in b.entries if x.id == eid)
        assert all(re.search(r"decline|do(n.?t| not) (wish|want)|prefer not", a, re.I) for a in e.answer)
    assert {"work_authorization", "sponsorship", "relocation", "salary"} <= set(b.unfilled())


# ---- shadow gates -----------------------------------------------------------------------------------------------
GATE_CFG = {"min_fit": 70, "auto_ats": ["greenhouse", "lever"], "max_per_day": 5, "company_window_days": 90,
            "company_max_applications": 2}


def gates(**over):
    m = A.Match("k", "Email", "email", True, A.ANSWERED, "x")
    args = dict(fit_score=80, ats="greenhouse", captcha=[], matches=[m], company="Acme", resume_ok=True,
                form_loaded=True, history=[], qualifying_today=0, cfg=GATE_CFG, today=date(2026, 10, 11),
                role_family="ai_solutions", job_id="new")
    return {g.name: g for g in evaluate(**{**args, **over})}


def test_all_gates_pass_only_in_the_ideal_case():
    assert qualifies(gates().values())


@pytest.mark.parametrize("over,gate", [
    ({"fit_score": 69}, "fit_score"), ({"fit_score": None}, "fit_score"),
    ({"ats": "ashby"}, "ats"), ({"captcha": ["recaptcha"]}, "no_captcha"),
    ({"matches": [A.Match("k", "Q", "text", True, A.NO_ENTRY)]}, "all_matched"),
    ({"matches": [A.Match("k", "Q", "text", False, A.UNVERIFIED, "x")]}, "all_matched"),
    ({"matches": [A.Match("k", "Why?", "textarea", False, A.ANSWERED, "x")]}, "no_long_answer"),
    ({"qualifying_today": 5}, "daily_cap"), ({"resume_ok": False}, "resume_verified"),
    ({"form_loaded": False}, "posting_open"),
])
def test_each_gate_blocks(over, gate):
    g = gates(**over)
    assert not g[gate].passed and not qualifies(g.values())


def test_optional_bank_skips_are_allowed_required_are_not():
    skip_opt = A.Match("k", "Website", "text", False, A.SKIPPED)
    assert gates(matches=[skip_opt])["all_matched"].passed


def _app(day, family, status="applied", job_id="old"):
    return {"job_id": job_id, "company": "Acme", "status": status, "date_applied": day, "role_family": family}


def test_company_limit_two_per_90_days_in_different_role_families():
    # same family within 90 days (manual counts): blocked; 91 days ago: allowed
    assert not gates(history=[_app("2026-07-14", "ai_solutions")])["company_limit"].passed     # 89 days ago
    assert gates(history=[_app("2026-07-12", "ai_solutions")])["company_limit"].passed         # 91 days ago
    # e.g. Cresta: AI Strategist (ai_consultant) applied, Enterprise Solutions Engineer (ai_solutions) allowed
    assert gates(history=[_app("2026-10-01", "ai_consultant")])["company_limit"].passed
    # a third application is blocked even in a new family
    two = [_app("2026-10-01", "ai_consultant", job_id="a"), _app("2026-10-02", "ai_eval", job_id="b")]
    assert not gates(history=two)["company_limit"].passed
    # unknown role family on either side counts as a match (conservative)
    assert not gates(history=[_app("2026-10-01", None)])["company_limit"].passed
    assert not gates(history=[_app("2026-10-01", "ai_consultant")], role_family=None)["company_limit"].passed
    # skipped isn't an application; the job's own row doesn't count against itself
    assert gates(history=[_app(None, "ai_solutions", status="skipped")])["company_limit"].passed
    assert gates(history=[_app("2026-10-01", "ai_solutions", job_id="new")])["company_limit"].passed


def test_auto_submit_true_is_rejected():
    check_config({"auto_submit": False})
    with pytest.raises(AutoSubmitNotSupported):
        check_config({"auto_submit": True})


# ---- forms: reading, filling, snapshot, submit safeguards (local HTML) ------------------------------------------
GREENHOUSE_HTML = """<form id="application-form" onsubmit="window.submitted=(window.submitted||0)+1; return false">
<div class="field-wrapper"><label for="first_name">First Name<span>*</span></label>
  <input id="first_name" type="text" aria-required="true"></div>
<div class="field-wrapper"><div role="group" aria-labelledby="upload-label-resume" aria-required="true">
  <div id="upload-label-resume">Resume/CV<span>*</span></div><label for="resume">Attach</label>
  <input id="resume" type="file"></div></div>
<div class="field-wrapper"><label for="question_1">Why do you want to join?</label><textarea id="question_1"></textarea></div>
<fieldset><legend>Which offices?</legend>
  <input type="checkbox" id="o1" name="question_2[]" value="1"><label for="o1">NYC</label>
  <input type="checkbox" id="o2" name="question_2[]" value="2"><label for="o2">Remote</label></fieldset>
<textarea id="g-recaptcha-response-1" name="g-recaptcha-response"></textarea>
<input type="hidden" name="token" value="x">
<button type="submit">Submit application</button></form>"""

LEVER_HTML = """<form onsubmit="window.submitted=(window.submitted||0)+1; return false"><ul>
<li class="application-question"><label><div class="application-label">Full name<span class="required">✱</span></div>
  <div class="application-field"><input type="text" name="name" required></div></label></li>
<li class="application-question"><label><div class="application-label">Current location</div>
  <div class="application-field"><input type="text" id="location-input" name="location"></div></label></li></ul>
<div data-qa="additional-cards"><h4 data-qa="card-name">How did you hear about Acme?</h4><ul>
<li class="application-question custom-question"><div class="application-label"><div class="text">Please check all that apply:</div></div>
  <div class="application-field"><label><input type="checkbox" name="cards[c][field0]" value="Friend">Friend</label>
  <label><input type="checkbox" name="cards[c][field0]" value="LinkedIn">LinkedIn</label></div></li></ul></div>
<input type="hidden" name="h-captcha-response" id="hcaptchaResponseInput">
<button type="button" id="btn-submit">SUBMIT APPLICATION</button></form>"""

ASHBY_HTML = """<form onsubmit="window.submitted=(window.submitted||0)+1; return false">
<div class="ashby-application-form-field-entry" data-field-path="_systemfield_email">
  <label class="ashby-application-form-question-title _required_x" for="_systemfield_email">Email</label>
  <input id="_systemfield_email" name="_systemfield_email" type="email" required></div>
<div class="ashby-application-form-field-entry" data-field-path="q-yes">
  <label class="ashby-application-form-question-title _required_x">Are you legally authorized to work in the USA?</label>
  <div class="ashby-application-form-input-yesno">
    <button type="submit" aria-pressed="false" class="ashby-application-form-input-yesno-option">Yes</button>
    <button type="submit" aria-pressed="false" class="ashby-application-form-input-yesno-option">No</button>
    <input type="checkbox" name="q-yes" tabindex="-1"></div></div>
<div class="ashby-application-form-field-entry" data-field-path="q-office">
  <label class="ashby-application-form-question-title">Which office?</label>
  <div class="ashby-application-form-input-radio-group">
    <input type="radio" id="r0" name="q-office"><label for="r0">New York</label>
    <input type="radio" id="r1" name="q-office"><label for="r1">Stockholm</label></div></div>
<button type="submit" class="ashby-application-form-submit-button">Submit Application</button></form>
<script>document.querySelectorAll('.ashby-application-form-input-yesno button').forEach(b => b.addEventListener('click', () => {
  b.parentElement.querySelectorAll('button').forEach(x => x.setAttribute('aria-pressed', String(x === b)));
}));</script>"""


@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


@pytest.fixture
def page(browser):
    pg = browser.new_page()
    yield pg
    pg.close()


def test_read_greenhouse(page):
    page.set_content(GREENHOUSE_HTML)
    qs = {q.label: q for q in read_questions(page, "greenhouse", load_options=False)}
    assert set(qs) == {"First Name", "Resume/CV", "Why do you want to join?", "Which offices?"}   # captcha/hidden skipped
    assert qs["Resume/CV"].kind == "file" and qs["Resume/CV"].required
    assert qs["Why do you want to join?"].kind == "textarea"
    assert qs["Which offices?"].kind == "checkbox_group" and qs["Which offices?"].options == ["NYC", "Remote"]


def test_read_lever_prefixes_generic_card_prompts(page):
    page.set_content(LEVER_HTML)
    qs = {q.label: q for q in read_questions(page, "lever")}
    assert qs["Full name"].required and qs["Current location"].kind == "autocomplete"
    assert "How did you hear about Acme? Please check all that apply:" in qs
    assert qs["How did you hear about Acme? Please check all that apply:"].options == ["Friend", "LinkedIn"]


def test_read_fill_and_snapshot_ashby(page, tmp_path):
    page.set_content(ASHBY_HTML)
    block_submits(page)
    qs = {q.key: q for q in read_questions(page, "ashby")}
    assert qs["q-yes"].kind == "yesno" and qs["q-yes"].options == ["Yes", "No"] and qs["q-yes"].required
    assert qs["q-office"].kind == "radio" and qs["q-office"].options == ["New York", "Stockholm"]
    assert fill(page, qs["_systemfield_email"], "n@x.com") == (True, "")
    assert fill(page, qs["q-yes"], "Yes") == (True, "")          # a type=submit button: clicked, form NOT submitted
    assert fill(page, qs["q-office"], "Stockholm") == (True, "")
    assert not page.evaluate("window.submitted")
    snap = snapshot(page, list(qs.values()))
    assert snap == {"_systemfield_email": "n@x.com", "q-yes": "Yes", "q-office": "Stockholm"}


def test_safe_click_refuses_submit_controls(page):
    page.set_content(ASHBY_HTML)
    with pytest.raises(SubmitRefused):
        safe_click(page.locator(".ashby-application-form-submit-button"))
    page.set_content(LEVER_HTML)
    with pytest.raises(SubmitRefused):
        safe_click(page.locator("#btn-submit"))
    assert not page.evaluate("window.submitted")


def test_block_stops_submission_until_hand_over_then_records_her_click(page):
    page.set_content(GREENHOUSE_HTML)
    block_submits(page)
    page.evaluate("document.querySelector('button[type=submit]').click()")          # simulated stray click
    page.evaluate("document.querySelector('form').requestSubmit()")
    assert not page.evaluate("window.submitted")
    page.evaluate("window.__jaOnSubmit = () => { window.reported = true; }")
    hand_over(page)
    page.click("button[type=submit]")                                                # Nithya's own click
    assert page.evaluate("window.submitted") == 1
    assert page.evaluate("window.__jaSubmitClicks") == 1 and page.evaluate("window.reported")


def test_fill_combobox_failure_is_left_for_her_not_raised(page):
    page.set_content(GREENHOUSE_HTML)
    ok, note = fill(page, FormQuestion("nope", "Missing", "text", True, [], "#does-not-exist"), "x")
    assert not ok and note


# ---- review outcome ---------------------------------------------------------------------------------------------
def test_classify_unchanged_edited_and_filled_by_her():
    labels = {"a": "Email", "b": "Sponsorship", "c": "Why us?"}
    pre = {"a": "n@x.com", "b": "Yes", "c": None}
    same = classify(pre, {"a": "n@x.com", "b": "Yes", "c": "Because"}, {"a", "b"}, labels)
    assert same["outcome"] == "approved_unchanged" and same["filled_by_you"] == ["Why us?"]
    edited = classify(pre, {"a": "n@x.com", "b": "No", "c": None}, {"a", "b"}, labels)
    assert edited["outcome"] == "approved_edited" and edited["edited"] == ["Sponsorship"]


@pytest.fixture
def db(tmp_path):
    store = Store(tmp_path / "jobs.db")
    store.save([Job(id="j1", company="Acme", title="AI Engineer", location="Atlanta, GA", remote=False,
                    url="https://job-boards.greenhouse.io/acme/jobs/1", description="d", source="greenhouse")])
    yield store
    store.close()


def test_confirm_manual_application_counts_for_cooldown(db):
    assert confirm(db, "j1", applied=True, note="applied on their site") == "applied"
    row = db.application("j1")
    assert row["status"] == "applied" and row["review_outcome"] == "manual" and row["date_applied"]
    g = gates(history=[dict(r) for r in db.applications()], today=date.fromisoformat(row["date_applied"]))
    assert not g["company_limit"].passed           # j1 has no role family on record: counted as same family


# ---- sheets -----------------------------------------------------------------------------------------------------
class FakeWorksheet:
    def __init__(self, values=None):
        self.values = values or []
        self.batches, self.appended = [], []

    def get_all_values(self):
        return self.values

    def update(self, values, rng):
        self.values = values + self.values[1:]

    def batch_update(self, updates):
        self.batches += updates

    def append_rows(self, rows, value_input_option=None):
        self.appended += rows


def test_sheets_upsert_by_job_id(db):
    confirm(db, "j1", applied=True)
    db.upsert_application({"job_id": "j1", "status": "applied", "edited_fields": json.dumps(["Sponsorship"])})
    rows = rows_for(db.applications())
    assert rows[0][COLUMNS.index("notes")].startswith("edited: Sponsorship")
    ws = FakeWorksheet()
    assert upsert(ws, rows) == (0, 1) and ws.values[0] == COLUMNS
    ws = FakeWorksheet([COLUMNS, ["j0"] + [""] * 14, ["j1"] + [""] * 14])
    assert upsert(ws, rows) == (1, 0) and ws.batches[0]["range"] == "A3:O3"


# ---- no submit code ---------------------------------------------------------------------------------------------
def test_no_submit_code_anywhere():
    src = {p.name: p.read_text(encoding="utf-8") for p in (ROOT / "src").rglob("*.py")}
    for name, text in src.items():
        assert not re.search(r"\.submit\(|requestSubmit|\.press\(\s*['\"]Enter", text), name
    forms = src["forms.py"] + src["apply.py"]
    clicks = [m.start() for m in re.finditer(r"\.click\(", forms)]
    # the only raw click is the one inside safe_click (after its submit-word check)
    assert len(clicks) == 1 and re.findall(r"\ndef (\w+)", forms[:clicks[0]])[-1] == "safe_click"
