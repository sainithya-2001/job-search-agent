import copy

import pytest

from main import load_config
from src.llm import LLMProvider, LLMResponse
from src.models import BulletVerdict, CitedText, Headline, SkillGroup, TailoredResume, TailoredSection, Verification
from src.render import render_html
from src.resume import citable_texts, project_tech, tailoring_source
from src.tailor import _remove_failing, tailor_job
from src.verify import Verifier, check_approved_summaries, numbers_in, select_summary

MASTER = {
    "contact": {"name": {"value": "Test Person"}, "email": {"value": "t@example.com"},
                "phone": {"value": "(555) 000-0000"}, "location": {"value": "Atlanta, GA"}},
    "years_experience": {"value": "4+ years"},
    "experience": [
        {"id": "exp_a", "company": "Acme", "title": "Data Analyst", "dates": {"start": "2024-01", "end": "present"},
         "bullets": [{"id": "exp_a_b01", "text": "Built SQL pipelines across 40+ regions, cutting error 30% to 6%."},
                     {"id": "exp_a_b02", "text": "Partnered with stakeholders on requirements."},
                     {"id": "exp_a_b03", "text": "Built SQL pipelines for 40+ regions with Power BI."}]},
        {"id": "exp_b", "company": "Beta", "title": "Business Analyst", "dates": {"start": "2020-05", "end": "2024-05"},
         "bullets": [{"id": "exp_b_b01", "text": "Wrote Python scripts to validate data."},
                     {"id": "exp_b_b02", "text": "Improved team communication."},
                     {"id": "exp_b_b03", "text": "Old claim.", "outdated": "no longer true"}]},
        {"id": "exp_c", "company": "Gamma", "title": "Intern Analyst", "dates": {"start": "2019-01", "end": "2019-04"},
         "bullets": [{"id": "exp_c_b01", "text": "Prepared reports in SQL."}]},
    ],
    "projects": [
        {"id": "proj_x", "name": "Text-to-SQL", "tech": [{"name": "Python"}, {"name": "Snowflake"}],
         "bullets": [{"id": "proj_x_b01", "text": "Built a Text-to-SQL assistant with RAG."}]},
        {"id": "proj_agent", "name": "Job Agent", "tech": [{"name": "Python"}],
         "bullets": [{"id": "proj_agent_b01", "text": "Built an agentic pipeline with structured outputs."}]},
    ],
    "skills": [{"id": "sk_python", "name": "Python"}, {"id": "sk_sql", "name": "SQL"},
               {"id": "sk_rag", "name": "RAG"}, {"id": "sk_snowflake", "name": "Snowflake"},
               {"id": "sk_powerbi", "name": "Power BI"}],
    "education": [{"id": "edu_u", "institution": "State University", "degree": {"value": "M.S. Data Science"},
                   "dates": {"end": {"value": "2026-05"}}, "gpa": {"value": "3.7"},
                   "coursework": {"core": [{"id": f"cw_{i}", "code": f"C{i}", "name": f"Course {i}"} for i in range(6)]}}],
    "approved_summaries": [
        {"id": "sum_ai", "use_for": {"role_families": ["ai_engineer"]}, "text": "Data and AI professional with 4+ years of experience. Builds RAG apps."},
        {"id": "sum_analytics", "use_for": {"role_families": ["analyst_backup"], "title_not": "business analyst"},
         "text": "Analyst with 4+ years of experience using SQL and Power BI."},
        {"id": "sum_ba", "use_for": {"role_families": ["analyst_backup"], "title": "business analyst"},
         "text": "Business analyst with 4+ years of experience in business analysis."},
    ],
    "same_work_groups": [["exp_a_b01", "exp_a_b03"]],
}


@pytest.fixture(scope="module")
def cfg():
    c = copy.deepcopy(load_config())
    c["tailoring"].update({
        "optional_experience": ["exp_c"],
        "required_bullets": [{"ids": ["exp_a_b01"], "families": ["analyst_backup", "ai_engineer"], "unless_title": "business analyst"}],
        "required_projects": [{"id": "proj_agent", "families": ["ai_engineer"]}],
        "preferences": [{"prefer": "exp_b_b01", "over": "exp_b_b02"}],
    })
    return c


def make_verifier(cfg, family="analyst_backup", title="Data Analyst"):
    return Verifier(MASTER, citable_texts(MASTER), project_tech(MASTER), cfg, family, title)


@pytest.fixture
def verifier(cfg):
    return make_verifier(cfg)


def good() -> TailoredResume:
    return TailoredResume(
        headline=Headline(role_label="Data Analyst", skill_ids=["sk_python", "sk_sql", "sk_rag"]),
        experience=[TailoredSection(id="exp_a", bullets=[CitedText(text="Built SQL pipelines across 40+ regions, reducing error from 30% to 6%.", source_ids=["exp_a_b01"])]),
                    TailoredSection(id="exp_b", bullets=[CitedText(text="Validated data with Python scripts.", source_ids=["exp_b_b01"])]),
                    TailoredSection(id="exp_c", bullets=[CitedText(text="Prepared reports in SQL.", source_ids=["exp_c_b01"])])],
        projects=[TailoredSection(id="proj_x", bullets=[CitedText(text="Built a Text-to-SQL assistant in Python using RAG.", source_ids=["proj_x_b01"])])],
        skill_groups=[SkillGroup(label="Data", skill_ids=["sk_python", "sk_sql"])],
        coursework_ids=["cw_0", "cw_1", "cw_2", "cw_3"],
    )


def test_numbers_in():
    assert numbers_in("40+ regions, error 30% → 6.5%, 200K+ rows, 3.2K MAE, 1,500 users") == {"40", "30", "6.5", "200k", "3.2k", "1500"}


def test_good_draft_passes_string_checks(verifier):
    assert verifier.check_structure(good()).issues == []


@pytest.mark.parametrize("mutate,expect", [
    (lambda t: setattr(t.experience[0].bullets[0], "text", "Built SQL pipelines across 90+ regions, error 30% to 6%."), "number"),
    (lambda t: setattr(t.experience[1].bullets[0], "text", "Validated data with Python and Kubernetes."), "tool/skill"),
    (lambda t: setattr(t.experience[1].bullets[0], "source_ids", ["exp_a_b02"]), "different entry"),
    (lambda t: setattr(t.experience[1].bullets[0], "source_ids", ["exp_b_b03"]), "unknown or unusable"),   # outdated
    (lambda t: setattr(t.headline, "role_label", "AI Engineer"), "not allowed"),
    (lambda t: setattr(t.headline, "role_label", "Senior Data Analyst"), "not allowed"),
    (lambda t: setattr(t.headline, "skill_ids", ["sk_python", "sk_sql", "sk_rag", "sk_snowflake", "sk_powerbi"]), "3-4 skills"),
    (lambda t: setattr(t.experience[0].bullets[0], "text", "Analyst with 5+ years of experience."), "5+ years"),
    (lambda t: t.experience.pop(1), "must list"),                                    # required entry missing
    (lambda t: setattr(t, "coursework_ids", ["cw_0", "cw_1"]), "4-6 courses"),
    (lambda t: setattr(t.headline, "skill_ids", ["sk_python", "sk_made_up", "sk_sql"]), "unknown skill"),
    # tailoring rules
    (lambda t: t.experience[0].bullets.append(CitedText(text="Built SQL pipelines for 40+ regions with Power BI.", source_ids=["exp_a_b03"])), "same work"),
    (lambda t: t.experience[0].bullets.append(CitedText(text="Built SQL pipelines across 40+ regions.", source_ids=["exp_a_b01"])), "same source"),
    (lambda t: setattr(t.experience[0], "bullets", [CitedText(text="Partnered with stakeholders on requirements.", source_ids=["exp_a_b02"])]), "must include a bullet citing exp_a_b01"),
    (lambda t: setattr(t.experience[1], "bullets", [CitedText(text="Improved team communication.", source_ids=["exp_b_b02"])]), "without exp_b_b01"),
])
def test_string_checks_catch(verifier, mutate, expect):
    t = good()
    mutate(t)
    issues = [str(i) for i in verifier.check_structure(t).issues]
    assert any(expect in i for i in issues), issues


def test_optional_entry_may_be_dropped(verifier):
    t = good()
    t.experience.pop()                      # exp_c is optional
    assert verifier.check_structure(t).passed


def test_required_project_for_ai_roles_only(cfg):
    ai = make_verifier(cfg, "ai_engineer", "AI Engineer")
    t = good()
    t.headline.role_label = "AI Engineer"
    assert any("proj_agent" in str(i) for i in ai.check_structure(t).issues)
    t.projects.append(TailoredSection(id="proj_agent", bullets=[CitedText(text="Built an agentic pipeline with structured outputs.", source_ids=["proj_agent_b01"])]))
    assert ai.check_structure(t).passed


def test_business_analyst_title_skips_metric_rule(cfg):
    ba = make_verifier(cfg, "analyst_backup", "Business Analyst")
    t = good()
    t.headline.role_label = "Business Analyst"
    t.experience[0].bullets = [CitedText(text="Partnered with stakeholders on requirements.", source_ids=["exp_a_b02"])]
    assert ba.check_structure(t).passed


def test_select_summary_by_family_and_title():
    assert select_summary(MASTER, "ai_engineer", "Applied AI Engineer")["id"] == "sum_ai"
    assert select_summary(MASTER, "analyst_backup", "Analytics Engineer")["id"] == "sum_analytics"
    assert select_summary(MASTER, "analyst_backup", "Senior Business Analyst")["id"] == "sum_ba"


def test_check_approved_summaries_reports_overclaims():
    findings = check_approved_summaries(MASTER, citable_texts(MASTER), project_tech(MASTER))
    assert any(f.startswith("sum_analytics: ties '4+ years'") for f in findings)       # "4+ years ... using SQL"
    assert not any(f.startswith("sum_ai") for f in findings)
    bad = copy.deepcopy(MASTER)
    bad["approved_summaries"][0]["text"] = "Analyst with 4+ years of experience and Kubernetes across 99 teams."
    f2 = check_approved_summaries(bad, citable_texts(bad), project_tech(bad))
    assert any("Kubernetes" in f for f in f2) and any("99" in f for f in f2)


def test_remove_failing_drops_later_duplicate(verifier):
    t = good()
    t.experience[0].bullets.append(CitedText(text="Built SQL pipelines for 40+ regions with Power BI.", source_ids=["exp_a_b03"]))
    removed = _remove_failing(t, verifier.check_structure(t), verifier)
    assert len(removed) == 1 and "same work" in removed[0]
    assert [b.source_ids for b in t.experience[0].bullets] == [["exp_a_b01"]]


class FakeLLM(LLMProvider):
    """Returns queued drafts for TailoredResume, and a verdict list for Verification."""
    name = "fake"

    def __init__(self, drafts, judge=None):
        super().__init__({"fast": "claude-haiku-5-5", "strong": "claude-sonnet-5-5"})
        self.drafts, self.judge, self.calls = list(drafts), judge or (lambda n: [True] * n), []

    def generate_json(self, prompt, schema, *, tier="fast", system=None):
        self.calls.append(schema.__name__)
        if schema is TailoredResume:
            data = self.drafts.pop(0)
        else:
            n = prompt.count("\nTEXT: ")
            data = Verification(verdicts=[BulletVerdict(index=i, supported=ok, problem=None if ok else "added claim")
                                          for i, ok in enumerate(self.judge(n))])
        return LLMResponse(data, self.models[tier], 4000, 1000, 0.1, 1)

    def _call(self, *a):  # pragma: no cover
        raise NotImplementedError

    def _retry_hint(self, exc):  # pragma: no cover
        return None


JOB = {"id": "j1", "company": "Acme", "title": "Data Analyst", "location": "Atlanta, GA", "description": "SQL role"}


def test_retry_then_pass(cfg, verifier):
    bad = good()
    bad.experience[0].bullets[0].text = "Built SQL pipelines across 90+ regions, error 30% to 6%."
    llm = FakeLLM([bad, good()])
    res = tailor_job(JOB, None, MASTER, tailoring_source(MASTER), verifier, llm, cfg, budget_left=1.0)
    assert res.status == "PASS" and res.attempts == 2
    assert llm.calls == ["TailoredResume", "TailoredResume", "Verification"]   # no judge call on a failing draft


def test_unsupported_sole_bullet_fails_job(cfg, verifier):
    # The judge always rejects exp_b's only bullet (item index 1). Removing it empties a required entry -> FAIL.
    llm = FakeLLM([good() for _ in range(3)], judge=lambda n: [i != 1 for i in range(n)])
    res = tailor_job(JOB, None, MASTER, tailoring_source(MASTER), verifier, llm, cfg, budget_left=1.0)
    assert res.attempts == 3 and any("Validated data" in r for r in res.removed)
    assert res.status == "FAIL"


def test_fit_one_page_drops_optional_then_cuts(tmp_path):
    pytest.importorskip("playwright")
    from src.render import fit_one_page
    t = good()
    long = "Built SQL pipelines across 40+ regions, reducing error from 30% to 6%. " * 6
    t.experience[1].bullets = [CitedText(text=long, source_ids=["exp_b_b01"]) for _ in range(14)]
    try:
        html, pages, cuts = fit_one_page(t, MASTER, "Summary.", tmp_path / "r.pdf", {"exp_c"}, {"exp_a_b01"})
    except Exception as e:  # Chromium not installed in this environment
        pytest.skip(f"playwright browser unavailable: {e}")
    assert pages == 1
    assert cuts[0] == "dropped optional entry exp_c to fit one page"
    assert all(c.startswith("cut exp_b bullet") for c in cuts[1:])
    assert t.experience[0].bullets[0].source_ids == ["exp_a_b01"]          # required bullet never cut


def test_render_uses_master_facts_and_given_summary():
    html = render_html(good(), MASTER, "Approved summary text.")
    assert "Data Analyst" in html and "Business Analyst" in html and "Jan 2024 – Present" in html
    assert "Course 0" in html and "Course 5" not in html          # only the chosen coursework
    assert "t@example.com" in html and "Approved summary text." in html
