# Job Search Agent — Project Spec

You are helping Nithya build a job-search agent in Python. Build it phase by phase, in order. After each phase: run it, show the output, and explain the key design decisions in 3–5 bullets so she understands the architecture. Do not move to the next phase until the current one runs end to end.

## Goal
Find relevant jobs that sponsor visas, score them against her resume, tailor a resume per job (never inventing facts), log everything to Google Sheets, and pre-fill applications for her approval.

## Target roles (in priority order)
1. AI Engineer / Applied AI Engineer / GenAI Engineer (LLM applications)
2. AI Solutions Engineer / Solutions Architect (AI)
3. Forward Deployed Engineer (associate / junior / mid level)
4. AI Consultant / AI Strategist (in-house at product companies, or McKinsey QuantumBlack, BCG X)
5. AI Evaluation / AI Quality roles
6. Backup: Data Analyst, Business Analyst, Analytics Engineer

Seniority: 0–5 years experience. Location: anywhere in the US, relocation OK, remote OK.

## Hard filters (exclude)
- Staffing agencies, body shops, C2C/contract-only, "W2 through vendor" postings
- Postings that say "no sponsorship", "must be a US citizen", "security clearance required"
- Duplicates (same company + normalized title)

## Stack
- Python 3.11+, venv
- httpx (HTTP), pydantic (models), SQLite via sqlite3 (storage)
- LLM calls go through the provider interface in src/llm.py (structured JSON output, retry with backoff on rate limits/5xx)
  - Active provider: anthropic SDK (Claude), key ANTHROPIC_API_KEY
  - Alternate: Google Gemini via google-genai SDK, key GEMINI_API_KEY (switched away 2026-10-11: Google returned 403 "project denied access" on generation)
  - Switch by changing the one line `llm.provider` in config.yaml. No other code may import an LLM SDK directly.
- pyyaml (master resume), jinja2 + weasyprint (resume PDF)
- gspread + Google service account (Sheets logging)
- playwright (form pre-fill)
- pytest (tests)
- Config in config.yaml; secrets in .env (never committed; add to .gitignore)

## Folder structure
```
job-agent/
  CLAUDE.md
  config.yaml            # llm provider + models, roles, filters, score threshold, company board list
  .env.example           # GEMINI_API_KEY, ANTHROPIC_API_KEY (optional), JSEARCH_API_KEY, GOOGLE_SHEET_ID
  data/
    master_resume.yaml   # single source of truth for every fact
    jobs.db
  src/
    models.py            # Job, ScoredJob, TailoredResume
    llm.py               # provider interface (Gemini default, Claude alternate), tiers, retry/backoff
    sources/
      greenhouse.py      # boards-api.greenhouse.io/v1/boards/{company}/jobs?content=true
      lever.py           # api.lever.co/v0/postings/{company}?mode=json
      ashby.py           # api.ashbyhq.com/posting-api/job-board/{company}
      jsearch.py         # optional aggregator (RapidAPI)
    store.py             # SQLite save / dedupe / status
    filters.py           # rule-based exclusions before any LLM call
    scorer.py            # LLM fit scoring -> structured JSON
    sponsorship.py       # check company against USCIS H-1B Employer Data Hub CSV
    tailor.py            # select + rewrite bullets from master resume
    verify.py            # guardrail: every claim must trace to master_resume.yaml
    render.py            # YAML -> HTML -> PDF
    sheets.py            # log to Google Sheets
    apply.py             # Playwright pre-fill for Greenhouse/Lever, stop before submit
    agent.py             # tool-use loop wiring the above as tools
  templates/resume.html.j2
  output/resumes/
  evals/
    labeled_jobs.csv     # 50 jobs Nithya scored by hand
    run_evals.py
  tests/
  main.py                # CLI: search | score | tailor <job_id> | apply <job_id> | run
```

## Phases

### Phase 1 — Ingest
- Pull jobs from Greenhouse/Lever/Ashby for the company list in config.yaml (start with ~40 AI-native and enterprise-AI companies that have sponsored H-1Bs).
- Normalize to the Job model: id, company, title, location, remote, url, description, source, posted_at.
- Apply filters.py (title keywords, exclusion phrases) before saving. Save to SQLite with dedupe.
- Done when: python main.py search prints a count and the top 10 new jobs.

### Phase 2 — Score
- For each new job, call the LLM with the master resume and job description. Force JSON output matching ScoredJob: fit_score (0-100), role_family, matched_skills[], gaps[], sponsorship_signal (yes/no/unclear + quote), is_staffing_firm (bool), seniority_ok (bool), one_line_reason.
- Use the "fast" tier (Claude Haiku 5.5; Gemini: 3.5 Flash-Lite) for the first pass and the "strong" tier (Claude Opus 5.5; Gemini: 3.8 Flash) only for jobs scoring >= scoring.escalate_threshold (60). Models per tier are set in config.yaml under llm.models.
- Cross-check sponsorship with sponsorship.py (USCIS Employer Data Hub CSV; match on normalized employer name).
- Done when: python main.py score ranks jobs and stores scores.

### Phase 3 — Tailor (with guardrail)
- master_resume.yaml holds every bullet with an id. Tailoring may only select, reorder, and reword existing bullets and skills. No new tools, metrics, titles, dates, or employers.
- verify.py: second LLM call + string checks. For each output bullet, cite the source bullet id. Reject and retry if any number, tool, or claim is not in the source.
- Render to output/resumes/{company}_{role}.pdf.
- Done when: python main.py tailor <job_id> produces a PDF and a verification report.

### Phase 4 — Agent loop
- Expose search_jobs, score_job, tailor_resume, log_application as tools. Write a plain tool-use loop (no framework) so the loop is readable. Add tool-calling to the src/llm.py interface so the loop stays provider-neutral.
- Log every tool call (name, args, latency, tokens, cost) to logs/.
- Done when: python main.py run does a full daily cycle autonomously up to (not including) applying.

### Phase 5 — Log + apply
- sheets.py: columns = date_found, company, role, location, url, fit_score, sponsorship, matched_skills, gaps, resume_file, status, date_applied, notes. Upsert by job id.
- apply.py: Playwright opens Greenhouse/Lever forms, fills name/email/phone/resume upload/standard questions from config.yaml, then PAUSES and waits for Nithya to review and click submit herself. Never auto-submit. Never touch LinkedIn/Indeed.
- Done when: one real application is pre-filled and the sheet row updates to "applied" after she confirms.

### Phase 6 — Evals + deploy
- evals/run_evals.py: compare agent fit scores to her hand labels (agreement %, mean absolute error, confusion on apply/skip). Run before/after any prompt change.
- Dockerfile + GitHub Actions workflow on a weekday schedule running search + score + log.
- README: architecture diagram, eval results, demo GIF.

## Rules for Claude Code
- Small commits per phase with clear messages.
- Ask before adding any dependency not listed above.
- Explain non-obvious code briefly when you write it.
- Never commit .env, jobs.db, or generated resumes.
