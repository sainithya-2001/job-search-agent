# Job Search Agent

A Python agent that finds AI / data jobs, scores them against a master resume, tailors a one-page resume per job
without inventing facts, logs applications to Notion, and pre-fills application forms for human review. It never
submits an application.

*Architecture, how each phase works, guardrails, and cost per run: coming in the full README (Phase 6).*

## Evals

`evals/run_evals.py` compares the agent's fit scores with hand labels (apply / maybe / skip) on the same jobs. It
is run before and after any scoring-prompt change. Scores are cached per model and prompt version, so re-running
is free until the prompt changes.

**Sample:** 30 already-scored jobs, labeled blind (company, title, link and a neutral two-line summary; no model
score). They were sampled evenly across fit bands (<40, 40–54, 55–64, 65+) and role families. Labels: 16 apply,
3 maybe, 11 skip. This is a small sample, and it holds more low-scoring jobs than the real pool does, so treat it
as a direction. It will be re-run as more jobs are labeled.

**Results (first run, 2026-10-12):**

- **Precision 100% at the old 65 cutoff:** no job labeled skip (or maybe) was recommended for tailoring. The
  scorer's misses were all in the other direction: it was too strict, and only 7 of the 16 "apply" jobs reached 65.
- **Rank correlation 0.68** (Spearman, pipeline score vs label): the model orders jobs much like the hand labels,
  on a scale that sits lower.
- Haiku alone and Sonnet alone scored about the same (rank correlation 0.69 and 0.68).

**Tailoring cutoff** (pipeline scores; precision and recall of "apply"):

| Cutoff | Precision | Recall | F1 | Jobs tailored (of 30) |
|---|---|---|---|---|
| 43 | 78% | 88% | 0.82 | 18 |
| 50 | 81% | 81% | 0.81 | 16 |
| 55 | 80% | 75% | 0.77 | 15 |
| 60 | 100% | 56% | 0.72 | 9 |
| 65 (old) | 100% | 44% | 0.61 | 7 |

Measured with the pipeline as it ran then: Haiku first, re-scored by Sonnet at 60+. **Changes made from these
results:** tailor at fit ≥ 50 (was 65), ≥ 45 for jobs posted in the last 3 days (was 58), and Sonnet re-scores
at 45+ (was 60), so borderline jobs get a second look. With the new escalation threshold, the same 30 jobs score
precision 86% / recall 75% at 50, and still 100% precision at 60 and 65. Full output: `evals/results.md`.
