import pytest

from main import load_config
from src.llm import LLMProvider, LLMResponse
from src.models import BulletVerdict, CitedText, Headline, SkillGroup, TailoredResume, TailoredSection, Verification
from src.render import render_html
from src.resume import citable_texts, project_tech, tailoring_source
from src.tailor import tailor_job
from src.verify import Verifier, numbers_in

MASTER = {
    "contact": {"name": {"value": "Test Person"}, "email": {"value": "t@example.com"},
                "phone": {"value": "(555) 000-0000"}, "location": {"value": "Atlanta, GA"}},
    "years_experience": {"value": "4+ years"},
    "experience": [
        {"id": "exp_a", "company": "Acme", "title": "Data Analyst", "dates": {"start": "2024-01", "end": "present"},
         "bullets": [{"id": "exp_a_b01", "text": "Built SQL pipelines across 40+ regions, cutting error 30% to 6%."},
                     {"id": "exp_a_b02", "text": "Partnered with stakeholders on requirements."}]},
        {"id": "exp_b", "company": "Beta", "title": "Business Analyst", "dates": {"start": "2020-05", "end": "2024-05"},
         "bullets": [{"id": "exp_b_b01", "text": "Wrote Python scripts to validate data."},
                     {"id": "exp_b_b02", "text": "Old claim.", "outdated": "no longer true"}]},
    ],
    "projects": [
        {"id": "proj_x", "name": "Text-to-SQL", "tech": [{"name": "Python"}, {"name": "Snowflake"}],
         "bullets": [{"id": "proj_x_b01", "text": "Built a Text-to-SQL assistant with RAG."}]},
        {"id": "proj_y", "name": "Forecasting", "tech": [{"name": "PyTorch"}],
         "bullets": [{"id": "proj_y_b01", "text": "Forecast demand with ~3% error."}]},
    ],
    "skills": [{"id": "sk_python", "name": "Python"}, {"id": "sk_sql", "name": "SQL"},
               {"id": "sk_rag", "name": "RAG"}, {"id": "sk_snowflake", "name": "Snowflake"}],
    "education": [{"id": "edu_u", "institution": "State University", "degree": {"value": "M.S. Data Science"},
                   "dates": {"end": {"value": "2026-05"}}, "gpa": {"value": "3.7"},
                   "coursework": {"core": [{"id": f"cw_{i}", "code": f"C{i}", "name": f"Course {i}"} for i in range(6)]}}],
}


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture
def verifier(cfg):
    return Verifier(MASTER, citable_texts(MASTER), project_tech(MASTER), cfg, "analyst_backup")


def good() -> TailoredResume:
    return TailoredResume(
        headline=Headline(role_label="Data Analyst", skill_ids=["sk_python", "sk_sql", "sk_rag"]),
        summary=CitedText(text="Analyst with 4+ years of experience across business analysis and data roles, building SQL pipelines.",
                          source_ids=["years_experience", "exp_a_b01"]),
        experience=[TailoredSection(id="exp_a", bullets=[CitedText(text="Built SQL pipelines across 40+ regions, reducing error from 30% to 6%.", source_ids=["exp_a_b01"])]),
                    TailoredSection(id="exp_b", bullets=[CitedText(text="Validated data with Python scripts.", source_ids=["exp_b_b01"])])],
        projects=[TailoredSection(id="proj_x", bullets=[CitedText(text="Built a Text-to-SQL assistant in Python using RAG.", source_ids=["proj_x_b01"])]),
                  TailoredSection(id="proj_y", bullets=[CitedText(text="Forecast demand with ~3% error.", source_ids=["proj_y_b01"])])],
        skill_groups=[SkillGroup(label="Data", skill_ids=["sk_python", "sk_sql"])],
        coursework_ids=["cw_0", "cw_1", "cw_2", "cw_3"],
    )


def test_numbers_in():
    assert numbers_in("40+ regions, error 30% → 6.5%, 200K+ rows, 3.2K MAE, 1,500 users") == {"40", "30", "6.5", "200k", "3.2k", "1500"}


def test_good_draft_passes_string_checks(verifier):
    assert verifier.check_structure(good()).issues == []


def test_project_tech_counts_as_source(verifier):
    # "Python" isn't in proj_x_b01's text but is in the project's tech list -> allowed.
    assert verifier.check_structure(good()).passed


@pytest.mark.parametrize("mutate,expect", [
    (lambda t: setattr(t.experience[0].bullets[0], "text", "Built SQL pipelines across 90+ regions."), "number"),
    (lambda t: setattr(t.experience[1].bullets[0], "text", "Validated data with Python and Kubernetes."), "tool/skill"),
    (lambda t: setattr(t.experience[1].bullets[0], "source_ids", ["exp_a_b01"]), "different entry"),
    (lambda t: setattr(t.experience[1].bullets[0], "source_ids", ["exp_b_b02"]), "unknown or unusable"),   # outdated
    (lambda t: setattr(t.headline, "role_label", "AI Engineer"), "not allowed"),
    (lambda t: setattr(t.headline, "role_label", "Senior Data Analyst"), "not allowed"),
    (lambda t: setattr(t.summary, "text", "Analyst with 5+ years of experience."), "5+ years"),
    (lambda t: setattr(t.summary, "text", "Analyst with 4+ years of experience building SQL pipelines."), "ties '4+ years'"),
    (lambda t: setattr(t.summary, "text", "AI engineer with 4+ years of experience."), "title she has not held"),
    (lambda t: setattr(t.summary, "text", "Data engineer building SQL pipelines."), "title she has not held"),
    (lambda t: t.experience.pop(), "every entry"),
    (lambda t: setattr(t, "coursework_ids", ["cw_0", "cw_1"]), "4-6 courses"),
    (lambda t: setattr(t.headline, "skill_ids", ["sk_python", "sk_made_up", "sk_sql"]), "unknown skill"),
])
def test_string_checks_catch(verifier, mutate, expect):
    t = good()
    mutate(t)
    issues = [str(i) for i in verifier.check_structure(t).issues]
    assert any(expect in i for i in issues), issues


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


JOB = {"id": "j1", "company": "Acme", "title": "Analyst", "location": "Atlanta, GA", "description": "SQL role"}


def test_retry_then_pass(cfg, verifier):
    bad = good()
    bad.experience[0].bullets[0].text = "Built SQL pipelines across 90+ regions."
    llm = FakeLLM([bad, good()])
    res = tailor_job(JOB, None, MASTER, tailoring_source(MASTER), verifier, llm, cfg, budget_left=1.0)
    assert res.status == "PASS" and res.attempts == 2
    assert llm.calls == ["TailoredResume", "TailoredResume", "Verification"]   # no judge call on a failing draft


def test_unsupported_bullet_removed_after_max_attempts(cfg, verifier):
    # The LLM judge always rejects the Beta bullet (item index 2: summary=0, exp_a=1, exp_b=2).
    llm = FakeLLM([good() for _ in range(3)], judge=lambda n: [i != 2 for i in range(n)])
    res = tailor_job(JOB, None, MASTER, tailoring_source(MASTER), verifier, llm, cfg, budget_left=1.0)
    assert res.attempts == 3 and any("Validated data" in r for r in res.removed)
    assert all("Validated data" not in b.text for s in res.resume.experience for b in s.bullets)
    # Removing exp_b's only bullet leaves an empty required entry -> the job FAILS rather than ship a gap silently.
    assert res.status == "FAIL"


def test_fallback_summary_passes_the_same_checks(verifier):
    from src.render import fallback_summary
    t = good()
    t.summary = CitedText(text=fallback_summary(MASTER), source_ids=["years_experience", "edu_u"])
    assert verifier.check_structure(t).issues == []
    t.summary = CitedText(text="", source_ids=[])
    assert fallback_summary(MASTER) in render_html(t, MASTER)


def test_render_uses_master_titles_not_model_text(verifier):
    html = render_html(good(), MASTER)
    assert "Data Analyst" in html and "Business Analyst" in html and "Jan 2024 – Present" in html
    assert "Course 0" in html and "Course 5" not in html          # only the chosen coursework
    assert "t@example.com" in html and "Atlanta, GA" in html
