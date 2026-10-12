"""ATS keyword matching: only master facts count, approved wordings only, verification accepts them one-way."""
from types import SimpleNamespace

import pytest

from src.keywords import (Keyword, KeywordList, cached, equivalents, get_keywords, match, report_section,
                          resume_coverage)
from src.models import Job
from src.store import Store

MASTER = {"skills": [
    {"id": "sk_snowflake", "name": "Snowflake", "covers": ["data warehouse"]},
    {"id": "sk_etl", "name": "ETL/ELT", "aliases": ["ELT"]},
    {"id": "sk_python", "name": "Python"},
    {"id": "sk_rest", "name": "REST APIs"},
]}
TEXTS = {"exp_a_b01": "Built dashboards and KPI reporting for executives.",
         "exp_a_b02": "Automated ETL workflows using SQL."}
TECH = {"proj_x": ["Docker", "Gemini"]}


def kw(text, imp=3, kind="tool"):
    return Keyword(text=text, kind=kind, importance=imp)


def test_match_uses_only_master_facts():
    cov = match([kw("Python"), kw("ELT"), kw("data warehouse"), kw("KPI reporting", 2, "method"), kw("Docker"),
                 kw("Kubernetes"), kw("Python scripting", 1)], MASTER, TEXTS, TECH)
    by = {m.keyword.text: m for m in cov.matches}
    assert by["Python"].have and by["Python"].how == "skill name"
    assert by["ELT"].have and by["ELT"].skill_id == "sk_etl" and by["ELT"].display == "ELT"
    assert by["data warehouse"].have and by["data warehouse"].display == "Snowflake (data warehouse)"
    assert by["KPI reporting"].have and by["KPI reporting"].source_id == "exp_a_b01"
    assert by["Docker"].have and by["Docker"].how == "project tech"
    assert not by["Kubernetes"].have and not by["Kubernetes"].hint
    assert not by["Python scripting"].have and "Python" in by["Python scripting"].hint   # suggested, never claimed
    assert cov.percent == round(100 * 5 / 7)


def test_wording_prefers_most_important_keyword_and_rest_case():
    cov = match([kw("REST APIs", 3), kw("rest apis", 1)], MASTER, TEXTS, TECH)
    assert cov.wording() == {"sk_rest": "REST APIs"}


def test_resume_coverage_buckets():
    cov = match([kw("Python"), kw("data warehouse"), kw("ELT"), kw("Kubernetes")], MASTER, TEXTS, TECH)
    rc = resume_coverage("Skills: Python, Snowflake. Built pipelines.", cov, MASTER)
    assert rc.exact == ["Python"]
    assert rc.synonym == ["data warehouse (shown as Snowflake)"]
    assert rc.not_shown == ["ELT"] and rc.missing == ["Kubernetes"]
    assert "3 of 4 keywords covered (75%)" in report_section(cov)


def test_covers_is_one_way():
    (trig, allowed), (trig2, allowed2) = equivalents(MASTER)
    assert "data warehouse" in allowed and "data warehouse" not in trig
    assert {"ETL/ELT", "ELT"} == trig2 == allowed2


def test_verifier_accepts_approved_wording_only_when_source_names_the_skill(tmp_path):
    from src.verify import Verifier, VerificationResult
    master = {**MASTER, "experience": [], "projects": [], "same_work_groups": [],
              "education": [{"coursework": {"core": []}}]}
    cfg = {"tailoring": {"headline_role_labels": {}, "headline_forbidden_words": []}}
    v = Verifier(master, {"b1": "Loaded data into Snowflake with ETL/ELT jobs.", "b2": "Built a data warehouse."},
                 {}, cfg, "ai_engineer", "AI Engineer")
    res = VerificationResult()
    assert v._check_text("x", "Loaded data into a data warehouse with ELT jobs.", ["b1"], None, res).terms_ok
    res = VerificationResult()
    assert not v._check_text("y", "Built a Snowflake data warehouse.", ["b2"], None, res).terms_ok  # not the reverse


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def generate_json(self, prompt, schema, *, tier, system):
        self.calls += 1
        data = KeywordList(keywords=[kw("Python", 2), kw("python", 1), kw("SQL", 3)])
        return SimpleNamespace(data=data, model="m", input_tokens=10, output_tokens=5)


def test_extraction_is_cached_and_deduped(tmp_path):
    store = Store(tmp_path / "jobs.db")
    store.save([Job(id="j1", company="Acme", title="AI Engineer", location="NYC", remote=False, url="u",
                    description="Requirements: Python, SQL", source="greenhouse")])
    job = store.conn.execute("SELECT * FROM jobs WHERE id='j1'").fetchone()
    llm = FakeLLM()
    kws, resp = get_keywords(store, job, llm)
    assert [k.text for k in kws] == ["SQL", "Python"] and resp is not None      # importance order, one "python"
    kws2, resp2 = get_keywords(store, job, llm)
    assert resp2 is None and llm.calls == 1 and [k.text for k in cached(store, "j1")] == ["SQL", "Python"]
    store.close()


def test_prompt_lists_have_and_dont_have():
    from src.tailor import keyword_lines
    cov = match([kw("ELT"), kw("Kubernetes")], MASTER, TEXTS, TECH)
    lines = "\n".join(keyword_lines(cov, SimpleNamespace(equiv_notes=["ETL/ELT may also be written as ELT"])))
    assert "skill id sk_etl" in lines and "DON'T HAVE (never claim): 'Kubernetes'" in lines


def test_soft_skills_ignored_for_coverage():
    from main import load_config
    pats = load_config()["tailoring"]["keyword_soft_skill_patterns"]
    cov = match([kw("Python"), kw("communication skills", 2, "method"), kw("problem solving", 2, "method"),
                 kw("cross-functional teams", 2, "method"), kw("customer-obsessed", 1, "method"),
                 kw("stakeholder collaboration", 2, "method")], MASTER, TEXTS, TECH, pats)
    assert [m.keyword.text for m in cov.matches] == ["Python", "stakeholder collaboration"]
    assert set(cov.ignored) == {"communication skills", "problem solving", "cross-functional teams", "customer-obsessed"}
    assert "Ignored as soft skills" in report_section(cov)


def test_approved_synonyms_in_master_shape():
    # aliases/covers are lists on the skill; ML-style aliases work as same-skill wording
    m = {"skills": [{"id": "sk_ml", "name": "Machine learning", "aliases": ["ML"], "covers": ["AI/ML"]}]}
    cov = match([kw("ML"), kw("AI/ML")], m, {}, {})
    assert [x.display for x in cov.matches] == ["ML", "Machine learning (AI/ML)"]


def test_strip_coverage_keeps_other_sections():
    from src.keywords import strip_coverage
    rep = ("# Report\n\n## Every claim, checked\nrows\n\n## Keyword coverage (ATS)\nold\n\n## One-page fitting (content cut)\n"
           "- cut x\n\n## Keyword coverage (ATS) — read-only check of this resume\nold2\n\n## Post-verification edit\n- y\n")
    out = strip_coverage(rep)
    assert "old" not in out and "## One-page fitting" in out and "- cut x" in out and "## Post-verification edit" in out


def test_resume_coverage_counts_alias_display_as_shown():
    m = {"skills": [{"id": "sk_eval", "name": "Evaluation frameworks", "aliases": ["evals", "evaluations"]}]}
    cov = match([kw("evals"), kw("evaluations")], m, {}, {})
    rc = resume_coverage("Skills: evals, Python", cov, m)
    assert rc.exact == ["evals"] and rc.synonym == ["evaluations (shown as evals)"] and rc.not_shown == []
