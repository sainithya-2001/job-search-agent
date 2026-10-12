"""Phase 6 evals: blind sample selection and the metrics in run_evals.py (synthetic data, no API, no data/)."""
import csv
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evals"))

import export_labels as ex  # noqa: E402
import run_evals as ev  # noqa: E402

FAMILIES = ["ai_engineer", "ai_solutions", "forward_deployed", "analyst_backup", "program_product_manager", "other"]


def fake_rows(n=110):
    return [{"id": f"j{i}", "fit_score": (i * 37) % 90 + 5, "role_family": FAMILIES[i % len(FAMILIES)]}
            for i in range(n)]


def test_selection_mixes_bands_and_families_deterministically():
    rows = fake_rows()
    picked = ex.select(rows)
    assert len(picked) == 30 and len({r["id"] for r in picked}) == 30
    bands = Counter(ex.band_of(r["fit_score"]) for r in picked)
    assert set(bands) == {"<40", "40-54", "55-64", "65+"} and max(bands.values()) - min(bands.values()) <= 1
    assert len({r["role_family"] for r in picked}) == len(FAMILIES)
    assert [r["id"] for r in ex.select(rows)] == [r["id"] for r in picked]          # seeded
    assert [r["fit_score"] for r in picked] != sorted(r["fit_score"] for r in picked)  # shuffled, not by score


def test_selection_fills_from_other_bands_when_one_is_short():
    rows = [r for r in fake_rows() if r["fit_score"] < 65] + [{"id": "hi", "fit_score": 80, "role_family": "other"}]
    picked = ex.select(rows)
    assert len(picked) == 30 and sum(r["fit_score"] >= 65 for r in picked) == 1


def test_band_and_metrics():
    assert [ev.band(s, 65, 50) for s in (80, 65, 64, 50, 49)] == ["apply", "apply", "maybe", "maybe", "skip"]
    labels = {"a": "apply", "b": "apply", "c": "maybe", "d": "skip", "e": "skip"}
    scores = {"a": 70, "b": 55, "c": 52, "d": 30, "e": 66}
    r = ev.evaluate(scores, labels, 65, 50)
    assert r.n == 5 and r.agreement == pytest.approx(3 / 5)
    assert r.mae_steps == pytest.approx((0 + 1 + 0 + 0 + 2) / 5)
    assert r.confusion["apply"] == {"skip": 0, "maybe": 1, "apply": 1}
    assert r.confusion["skip"]["apply"] == 1


def test_spearman():
    assert ev.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert ev.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert ev.spearman([1, 1, 1], [1, 2, 3]) is None


def test_cutoff_recommendation_prefers_lower_on_ties():
    labels = {"a": "apply", "b": "apply", "c": "skip", "d": "skip", "e": "maybe"}
    scores = {"a": 72, "b": 61, "c": 58, "d": 40, "e": 50}
    best = ev.recommend(ev.cutoff_table(scores, labels))
    assert best.threshold == 59 and best.recall == 1.0 and best.precision == 1.0     # 59..61 tie -> 59
    wide = ev.recommend(ev.cutoff_table(scores, labels, ("apply", "maybe")))
    assert wide.threshold == 59 or wide.threshold <= 50


def test_pipeline_scores_escalate_like_the_pipeline():
    assert ev.pipeline_scores({"a": 70, "b": 40}, {"a": 62, "b": 90}, 60) == {"a": 62, "b": 40}


def test_read_labels_reports_blank_and_bad(tmp_path):
    p = tmp_path / "l.csv"
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["n", "job_id", "company", "title", "link", "summary", "label (apply/maybe/skip)", "notes"])
        w.writerow([1, "j1", "A", "t", "u", "s", "Apply", ""])
        w.writerow([2, "j2", "B", "t", "u", "s", "", ""])
        w.writerow([3, "j3", "C", "t", "u", "s", "yes", ""])
    labels, problems = ev.read_labels(p)
    assert labels == {"j1": "apply"} and len(problems) == 2 and "not labeled yet" in problems[0]
