"""Compare the agent's fit scores to Nithya's hand labels (evals/label_me.csv: apply / maybe / skip).

    python evals/run_evals.py                 # report from cached eval scores (free); says what's missing
    python evals/run_evals.py --score         # score the labeled jobs with BOTH models first (estimate + cap)

Scoring here is separate from the pipeline's stored scores: every labeled job is scored by the fast model (Haiku)
AND the strong model (Sonnet) with the CURRENT scoring prompt, so the two models are compared on the same jobs and
a prompt change is measured before/after. Scores are cached in evals/.eval_scores.json keyed by model + a hash of
the prompt, so re-running the report is free and only a prompt change triggers new calls. Nothing is written to
jobs.db.

Reported for Haiku, Sonnet, and the pipeline as it runs (Haiku, re-scored by Sonnet at >= escalate_threshold):
  - agreement: % of jobs where the score's band matches the label. Bands: apply >= agent.tailor_min_score,
    maybe >= digest.tier_b_min_score, else skip (the cutoffs the pipeline actually acts on)
  - mean absolute error in label steps (skip=0, maybe=1, apply=2), and Spearman rank correlation
  - confusion matrix (rows = your label, columns = the model's band)
  - recommended tailoring cutoff: the score threshold that best separates your "apply" jobs (highest F1;
    ties go to the lower cutoff, since missing a good job costs more than one extra tailored resume)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EVALS = ROOT / "evals"
LABELS = EVALS / "label_me.csv"
CACHE = EVALS / ".eval_scores.json"
RESULTS = EVALS / "results.md"
CLASSES = ["skip", "maybe", "apply"]
DEFAULT_CAP = 1.00


# ---- labels ---------------------------------------------------------------------------------------------------
def read_labels(path: Path = LABELS) -> tuple[dict[str, str], list[str]]:
    """-> ({job_id: label}, problems). Blank labels are skipped (reported); anything else must be a class."""
    labels, problems = {}, []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            col = next(k for k in row if k.startswith("label"))
            v = (row[col] or "").strip().lower()
            if not v:
                problems.append(f"row {row['n']} ({row['company']}): not labeled yet")
            elif v not in CLASSES:
                problems.append(f"row {row['n']} ({row['company']}): '{v}' is not apply / maybe / skip")
            else:
                labels[row["job_id"]] = v
    return labels, problems


# ---- metrics (pure: unit-tested) --------------------------------------------------------------------------------
def band(score: int, apply_at: int, maybe_at: int) -> str:
    return "apply" if score >= apply_at else "maybe" if score >= maybe_at else "skip"


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1          # average rank for ties
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx, vy = sum((a - mx) ** 2 for a in rx), sum((b - my) ** 2 for b in ry)
    return None if not vx or not vy else cov / (vx * vy) ** 0.5


@dataclass
class Report:
    n: int
    agreement: float
    mae_steps: float
    spearman: float | None
    confusion: dict[str, dict[str, int]]          # label -> model band -> count


def evaluate(scores: dict[str, int], labels: dict[str, str], apply_at: int, maybe_at: int) -> Report:
    ids = [j for j in labels if j in scores]
    conf = {l: {b: 0 for b in CLASSES} for l in CLASSES}
    hits, err = 0, 0
    for j in ids:
        b, l = band(scores[j], apply_at, maybe_at), labels[j]
        conf[l][b] += 1
        hits += b == l
        err += abs(CLASSES.index(b) - CLASSES.index(l))
    n = len(ids)
    return Report(n, hits / n if n else 0.0, err / n if n else 0.0,
                  spearman([scores[j] for j in ids], [CLASSES.index(labels[j]) for j in ids]), conf)


@dataclass
class Cutoff:
    threshold: int
    precision: float
    recall: float
    f1: float
    tailored: int          # jobs at or above the threshold


def cutoff_table(scores: dict[str, int], labels: dict[str, str], positive=("apply",),
                 thresholds=range(30, 91)) -> list[Cutoff]:
    ids = [j for j in labels if j in scores]
    pos = {j for j in ids if labels[j] in positive}
    out = []
    for t in thresholds:
        sel = {j for j in ids if scores[j] >= t}
        tp = len(sel & pos)
        p = tp / len(sel) if sel else 0.0
        r = tp / len(pos) if pos else 0.0
        out.append(Cutoff(t, p, r, 2 * p * r / (p + r) if p + r else 0.0, len(sel)))
    return out


def recommend(table: list[Cutoff]) -> Cutoff | None:
    best = max((c.f1 for c in table), default=0.0)
    if best == 0.0:
        return None
    return min((c for c in table if c.f1 == best), key=lambda c: c.threshold)   # tie -> lower cutoff


def pipeline_scores(fast: dict[str, int], strong: dict[str, int], escalate_at: int) -> dict[str, int]:
    """What the pipeline would store: the fast score, replaced by the strong score at >= escalate_at."""
    return {j: (strong[j] if s >= escalate_at and j in strong else s) for j, s in fast.items()}


# ---- scoring (paid; cached) -------------------------------------------------------------------------------------
def prompt_hash(system: str, prompt: str) -> str:
    return hashlib.sha256((system + "\n\n" + prompt).encode("utf-8")).hexdigest()[:16]


def cached_scores(cache: dict, model: str, jobs: dict, system: str) -> dict[str, int]:
    from src.scorer import job_prompt
    entries = cache.get(model, {})
    return {j: e["fit_score"] for j, e in entries.items()
            if j in jobs and e["prompt"] == prompt_hash(system, job_prompt(jobs[j]))}


def score_missing(cfg, llm, jobs: dict, system: str, cache: dict, cap: float, run: bool) -> float:
    from src.llm import cost_usd, estimate_tokens
    from src.models import ScoredJob
    from src.scorer import SCHEMA_OVERHEAD_TOKENS, job_prompt

    pricing, out = cfg["llm"].get("pricing", {}), cfg["scoring"]["est_output_tokens"]
    todo = []
    for tier in ("fast", "strong"):
        model = llm.models[tier]
        have = cached_scores(cache, model, jobs, system)
        todo += [(tier, model, j) for j in jobs if j not in have]
    est = sum(cost_usd(m, estimate_tokens(system + job_prompt(jobs[j])) + SCHEMA_OVERHEAD_TOKENS, out[t], pricing)
              or 0 for t, m, j in todo)
    print(f"Eval scoring: {len(todo)} call(s) needed ({sum(t == 'fast' for t, _, _ in todo)} {llm.models['fast']}, "
          f"{sum(t == 'strong' for t, _, _ in todo)} {llm.models['strong']}): estimated ${est:.3f}, cap ${cap:.2f}")
    if not todo or not run:
        return 0.0
    if est > cap:
        sys.exit("Refusing: the estimate exceeds the cap (use --cap to raise it).")
    spent, worst = 0.0, 0.0
    for tier, model, j in todo:
        if spent + max(worst, 0.0) > cap:
            print(f"Stopped at the ${cap:.2f} cap; cached scores are kept, re-run to resume.")
            break
        prompt = job_prompt(jobs[j])
        r = llm.generate_json(prompt, ScoredJob, tier=tier, system=system)
        c = cost_usd(r.model, r.input_tokens, r.output_tokens, pricing) or 0.0
        spent, worst = spent + c, max(worst, c)
        cache.setdefault(model, {})[j] = {"fit_score": r.data.fit_score, "prompt": prompt_hash(system, prompt)}
        CACHE.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    print(f"Actual eval scoring cost: ${spent:.4f}")
    return spent


# ---- report ---------------------------------------------------------------------------------------------------
def render(results: dict[str, Report], cut: dict[str, tuple[Cutoff | None, Cutoff | None]], current: int,
           maybe_at: int, n_labels: dict[str, int], problems: list[str]) -> str:
    L = ["# Eval results", "",
         f"Labels: {sum(n_labels.values())} jobs ({', '.join(f'{n_labels[c]} {c}' for c in reversed(CLASSES))}). "
         f"Bands: apply >= {current}, maybe >= {maybe_at}, else skip.", ""]
    if problems:
        L += [f"Not used ({len(problems)}): " + "; ".join(problems), ""]
    L += ["| Model | Jobs | Agreement | MAE (label steps) | Spearman | Recommended cutoff (apply) | P / R / F1 |",
          "|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        best = cut[name][0]
        rc = (f"{best.threshold} | {best.precision:.0%} / {best.recall:.0%} / {best.f1:.2f}" if best else "n/a | n/a")
        sp = f"{r.spearman:.2f}" if r.spearman is not None else "n/a"
        L.append(f"| {name} | {r.n} | {r.agreement:.0%} | {r.mae_steps:.2f} | {sp} | {rc} |")
    for name, r in results.items():
        L += ["", f"### Confusion: {name} (rows = your label, columns = model band)", "",
              "| | skip | maybe | apply |", "|---|---|---|---|"]
        L += [f"| **{l}** | " + " | ".join(str(r.confusion[l][b]) for b in CLASSES) + " |" for l in reversed(CLASSES)]
        a = r.confusion
        L.append(f"\nApply vs skip: {a['apply']['apply']} of your apply jobs reach apply, {a['apply']['skip']} land in "
                 f"skip; {a['skip']['apply']} of your skip jobs reach apply.")
    pipe = cut.get("pipeline")
    if pipe and pipe[0]:
        best, wide = pipe
        L += ["", "## Tailoring cutoff", "",
              f"Current: tailor at fit >= {current}. Based on your labels, the pipeline's scores separate your "
              f"\"apply\" jobs best at **>= {best.threshold}** (precision {best.precision:.0%}, recall "
              f"{best.recall:.0%}, {best.tailored} of {results['pipeline'].n} jobs tailored)."]
        if wide:
            L.append(f"Counting \"maybe\" as worth tailoring too, the best cutoff is >= {wide.threshold} "
                     f"(precision {wide.precision:.0%}, recall {wide.recall:.0%}).")
        L.append("Small sample: treat this as a direction, and re-check after the next batch of labels.")
    return "\n".join(L) + "\n"


def main() -> None:
    from main import load_config
    from src.llm import get_llm, load_env
    from src.resume import load_master, profile_text
    from src.scorer import system_prompt
    from src.store import Store

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--score", action="store_true", help="score labeled jobs missing from the cache (paid)")
    ap.add_argument("--cap", type=float, default=DEFAULT_CAP, help=f"cost cap for --score (default ${DEFAULT_CAP:.2f})")
    args = ap.parse_args()

    cfg = load_config()
    load_env(ROOT / ".env")
    if not LABELS.exists():
        sys.exit("evals/label_me.csv not found: run python evals/export_labels.py --run first.")
    labels, problems = read_labels()
    if not labels:
        sys.exit("No labels yet: fill in the label column (apply / maybe / skip) in evals/label_me.csv.")
    store = Store(ROOT / "data" / "jobs.db")
    jobs = {r["id"]: r for r in (store.conn.execute("SELECT * FROM jobs WHERE id=?", (j,)).fetchone()
                                 for j in labels) if r is not None}
    llm = get_llm(cfg)
    system = system_prompt(cfg, load_master(ROOT / "data" / "master_resume.yaml"))
    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    score_missing(cfg, llm, jobs, system, cache, args.cap, args.score)

    fast = cached_scores(cache, llm.models["fast"], jobs, system)
    strong = cached_scores(cache, llm.models["strong"], jobs, system)
    if len(fast) < len(jobs) or len(strong) < len(jobs):
        sys.exit(f"Missing eval scores ({len(fast)}/{len(jobs)} fast, {len(strong)}/{len(jobs)} strong). "
                 f"Run with --score (see the estimate above).")
    apply_at, maybe_at = cfg["agent"]["tailor_min_score"], cfg["digest"]["tier_b_min_score"]
    sets = {llm.models["fast"]: fast, llm.models["strong"]: strong,
            "pipeline": pipeline_scores(fast, strong, cfg["scoring"]["escalate_threshold"])}
    results = {k: evaluate(v, labels, apply_at, maybe_at) for k, v in sets.items()}
    cut = {k: (recommend(cutoff_table(v, labels)), recommend(cutoff_table(v, labels, ("apply", "maybe"))))
           for k, v in sets.items()}
    n_labels = {c: sum(v == c for v in labels.values()) for c in CLASSES}
    md = render(results, cut, apply_at, maybe_at, n_labels, problems)
    RESULTS.write_text(md, encoding="utf-8")
    print(md)
    print(f"Saved {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
