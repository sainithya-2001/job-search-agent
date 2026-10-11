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

Seniority: 0–5 years experience. Location: anywhere in the US, relocation OK, remote OK. Preference order (config.yaml preferences.location_priority): Atlanta GA = US-Remote > California > Texas > anywhere in US.
Remote roles count only if US-remote; remote roles limited to other countries/regions, or to states other than GA/CA/TX, are excluded.

Companies: prioritize startups and lesser-known strong AI companies (Series A–D or profitable, ~20–1,000 employees, known investors, shipping AI products), plus a few Atlanta-based ones. Each company in config.yaml has tier (startup | growth | big_tech) and e_verify (yes | no | unknown, filled by hand).

## Hard filters (exclude)
- Staffing agencies, body shops, C2C/contract-only, "W2 through vendor" postings
- Postings that say "no sponsorship", "must be a US citizen", "security clearance required"
- Export-control / US-person requirements: "export controlled", ITAR, EAR, "U.S. person", "permanent resident", "green card" (except positive mentions like "green card sponsorship")
- Titles: frontend, full-stack, infrastructure, platform, security/AppSec, systems engineer, SRE, DevOps (even with AI in the title); Staff, Principal, Lead, Director. Senior is kept.
- Duplicates (same company + normalized title)

## Stack
- Python 3.11+, venv
- httpx (HTTP), pydantic (models), SQLite via sqlite3 (storage)
- LLM calls go through the provider interface in src/llm.py (structured JSON output, retry with backoff on rate limits/5xx)
  - Active provider: anthropic SDK (Claude), key ANTHROPIC_API_KEY
  - Alternate: Google Gemini via google-genai SDK, key GEMINI_API_KEY (switched away 2026-10-11: Google returned 403 "project denied access" on generation)
  - Switch by changing the one line `llm.provider` in config.yaml. No other code may import an LLM SDK directly.
- pyyaml (master resume), jinja2 + playwright (resume PDF via headless Chromium; WeasyPrint dropped because it needs GTK system libraries on Windows)
- gspread + Google service account (Sheets logging)
- playwright (form pre-fill, review mode only)
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
    answers.py           # answer bank + exact question matching
    forms.py             # read/fill/snapshot Greenhouse, Lever, Ashby forms; submit safeguards
    gates.py             # shadow-mode auto-submit gates (no submit code)
    apply.py             # queue + review (pre-fill, hand over, log outcome) + confirm
    agent.py             # tool-use loop wiring the above as tools
  templates/resume.html.j2
  output/resumes/
  evals/
    labeled_jobs.csv     # 50 jobs Nithya scored by hand
    run_evals.py
  tests/
  main.py                # CLI: search | score | tailor | run | queue | review | apply <job_id> | confirm | sheets-sync
```

## Phases

### Phase 1 — Ingest
- Pull jobs from Greenhouse/Lever/Ashby for the company list in config.yaml (start with ~40 AI-native and enterprise-AI companies that have sponsored H-1Bs).
- Normalize to the Job model: id, company, title, location, remote, url, description, source, posted_at.
- Apply filters.py (title keywords, exclusion phrases) before saving. Save to SQLite with dedupe.
- Done when: python main.py search prints a count and the top 10 new jobs.

### Phase 2 — Score
- For each new job, call the LLM with the master resume and job description. Force JSON output matching ScoredJob: fit_score (0-100), role_family, matched_skills[], gaps[], sponsorship_signal (yes/no/unclear + quote), is_staffing_firm (bool), seniority_ok (bool), one_line_reason.
- Use the "fast" tier (Claude Haiku 5.5; Gemini: 3.5 Flash-Lite) for the first pass and the "strong" tier (Claude Sonnet 5.5; Gemini: 3.8 Flash) only for jobs scoring >= scoring.escalate_threshold (60). Models per tier are set in config.yaml under llm.models.
- Cross-check sponsorship with sponsorship.py (USCIS Employer Data Hub CSV; match on normalized employer name).
- Ranking of results: fit_score, then company tier (startup > growth > big_tech), then location priority.
- Cost control: print an estimate before each run and the actual cost after; never exceed scoring.max_run_cost_usd.
- Done when: python main.py score ranks jobs and stores scores.

### Phase 3 — Tailor (with guardrail)
- master_resume.yaml holds every bullet with an id. Tailoring may only select, reorder, and reword existing bullets and skills. No new tools, metrics, titles, dates, or employers.
- verify.py: second LLM call + string checks. For each output bullet, cite the source bullet id. Reject and retry if any number, tool, or claim is not in the source.
- Summary: never generated. Use the pre-approved summary in master_resume.yaml for the job's role family (sum_ai for AI roles, sum_analytics for Data Analyst / Analytics Engineer, sum_ba for Business Analyst).
- One page max (enforced by rendering the PDF and counting pages; cut content only, never add). No two bullets about the same work (master_resume.yaml same_work_groups). For AI and analytics roles always include the required metric bullet (config.local.yaml tailoring.required_bullets). Apply the bullet preferences in tailoring.preferences. Optional experience (tailoring.optional_experience) may be dropped to fit one page. Include the Job-Search Agent project for all AI roles. Headline: at most 4 skills.
- Coursework: pick the 4–6 courses most relevant to the job from master_resume.yaml; never list all of them.
- Headline: may change per job. It is an optional role label plus skills from master_resume.yaml. The role label must come from config.yaml tailoring.headline_role_labels for the job's role family (AI Engineer, Applied AI, AI Solutions Engineer, Forward Deployed Engineer, AI Consultant, AI Evaluation, Data Analyst, Business Analyst, Analytics Engineer); no other labels. No seniority words anywhere in the headline (Senior/Lead/Principal/Staff). Never change experience-section titles; print them exactly as stored.
- Years of experience: always "4+ years". Never write "5+".
- Never use a master_resume.yaml field whose conflict is unresolved, or anything marked pending/outdated.
- Skills: every skill in master_resume.yaml is confirmed (2026-10-11), including Anthropic API, AWS, Azure, CI/CD, and the AI-evaluation skills. Tailoring selects only the skills relevant to each job, never the full list.
- Render to output/resumes/{company}_{role}.pdf.
- Done when: python main.py tailor <job_id> produces a PDF and a verification report.

### Phase 4 — Agent loop
- Expose search_jobs, score_job, tailor_resume, log_application as tools. Write a plain tool-use loop (no framework) so the loop is readable. Add tool-calling to the src/llm.py interface so the loop stays provider-neutral.
- Log every tool call (name, args, latency, tokens, cost) to logs/.
- Limits (config.yaml `agent`, enforced in code, not by the prompt): hard cost cap $0.50 per run (model turns + scoring + tailoring), max 30 tool calls, stop and report if any tool fails twice in a row, tailor only fit_score >= 65 and at most 5 per run, no apply tool (log_application can't set "applied"). `python main.py run --dry-run` plans and estimates with no LLM calls and no DB writes.
- Done when: python main.py run does a full daily cycle autonomously up to (not including) applying.

### Phase 5 — Log + apply (review mode)
- Answer bank: data/answer_bank.yaml (gitignored; Nithya fills it in; template data/answer_bank.example.yaml is committed). Standard questions: work authorization, sponsorship, location, relocation, salary range, links; EEO = decline to answer. A form question matches an entry only on exact text (case/spacing/punctuation ignored, no fuzzy or LLM matching); choice answers must equal an option the form offers. Unmatched or unanswered questions are left for Nithya.
- Review mode is the only mode. `queue` reads each tailored job's form headlessly (Greenhouse, Lever, Ashby) and records how much the bank covers. `review` opens the form in a visible browser, pre-fills exact matches, attaches the verified tailored PDF, outlines what's left, and hands over: Nithya reviews and clicks Submit herself. `confirm <job_id> --applied|--skipped` records the result (also for applications she submitted by hand).
- Never submit. No code clicks a submit control (safe_click refuses submit-like buttons) and submissions are blocked while the agent fills. Never touch LinkedIn/Indeed. Never attempt to solve or bypass a CAPTCHA.
- Log for each application whether Nithya approved unchanged or edited (form snapshot after pre-fill vs at her Submit click; edited questions listed; edits to bank answers flagged).
- Shadow-mode auto-submit gates (config.yaml `apply`, auto_submit: false; there is no submit code and true is rejected): record whether an application WOULD qualify: fit >= 70, Greenhouse or Lever, no CAPTCHA of any kind, every question matched exactly from the bank, no long-answer box (short text fields are fine if matched exactly), at most 5 per day, no application to the same company in 90 days (manual ones included), verified one-page resume, posting open. Everything goes to the review queue regardless.
- sheets.py: columns = job_id, date_found, company, role, location, url, fit_score, sponsorship, matched_skills, gaps, resume_file, status, date_applied, review_outcome, notes. Upsert by job id. SQLite `applications` is the source of truth; the sheet is synced from it.
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
