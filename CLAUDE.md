# Job Search Agent — Project Spec

You are helping Nithya build a job-search agent in Python. Build it phase by phase, in order. After each phase: run it, show the output, and explain the key design decisions in 3–5 bullets so she understands the architecture. Do not move to the next phase until the current one runs end to end.

## Goal
Find relevant jobs she is eligible for, score them against her resume, tailor a resume per job (never inventing facts), log everything to Notion, and pre-fill applications for her approval.

## Target roles (in priority order)
1. AI Engineer / Applied AI Engineer / GenAI Engineer (LLM applications)
2. AI Solutions Engineer / Solutions Architect (AI)
3. Forward Deployed Engineer (associate / junior / mid level)
4. AI Consultant / AI Strategist (in-house at product companies, or McKinsey QuantumBlack, BCG X)
5. AI Evaluation / AI Quality roles
6. Backup: Data Analyst, Business Analyst, Analytics Engineer
7. Program / Product Manager: Program Manager, Technical Program Manager, Product Manager, Analytics Manager, AI / Data / Solutions Manager (role family program_product_manager, ranked last)

Seniority: 0–5 years experience. Location: anywhere in the US, relocation OK, remote OK. Preference order (config.yaml preferences.location_priority): Atlanta GA = US-Remote > California > Texas > anywhere in US.
Remote roles count only if US-remote; remote roles limited to other countries/regions are excluded. Remote roles restricted to any US state pass (relocation anywhere in the US is fine) but rank after GA/CA/TX and unrestricted US-remote.

Companies: prioritize startups and lesser-known strong AI companies (Series A–D or profitable, ~20–1,000 employees, known investors, shipping AI products), plus a few Atlanta-based ones. Each company in config.yaml has tier (startup | growth | big_tech) and e_verify (yes | no | unknown, filled by hand).

## Hard filters (exclude)
- Staffing agencies, body shops, C2C / corp-to-corp, "W2 through vendor" postings. Contract, contractor, contract-to-hire, and 1099 roles posted directly by the employer are NOT excluded (2026-10-12); the digest labels them [contract] / [contract-to-hire] and they are not ranked lower. Temporary and intern roles are still excluded.
- Postings that require US citizenship ("must be a US citizen", "US citizens only") or a security clearance
- Sponsorship statements ("no sponsorship", "must be authorized to work without sponsorship") are NOT excluded (2026-10-11)
- Export-control / US-person requirements: "export controlled", ITAR, EAR, "U.S. person", "permanent resident", "green card" (except positive mentions like "green card sponsorship")
- Titles: frontend, full-stack, infrastructure, platform, security/AppSec, systems engineer, SRE, DevOps, network operations, chip design, hardware, mobile, iOS, Android, electrical, and non-US regions in the title (EMEA, APAC, LATAM, EU, UK) (even with AI in the title); Staff, Principal, Lead, Director, VP, Head of, Engineering Manager. Senior is kept. Other "Manager" titles pass only if they match a role family (program_product_manager).
- Duplicates (same company + normalized title)
- Postings older than 14 days (filters.max_posting_age_days). Jobs with no reliable date are kept and treated as 4-14 days old (not fresh). Jobs already in the database are re-checked against this at display time: once a scored job's posting is older than 14 days it is hidden from ranking, tailoring candidates, and the digest's Tier B (scorer.hide_stale). Jobs that are tailored, queued, or applied are never hidden or removed; the review queue keeps them regardless of age.

## Stack
- Python 3.11+, venv
- httpx (HTTP), pydantic (models), SQLite via sqlite3 (storage)
- LLM calls go through the provider interface in src/llm.py (structured JSON output, retry with backoff on rate limits/5xx)
  - Active provider: anthropic SDK (Claude), key ANTHROPIC_API_KEY
  - Alternate: Google Gemini via google-genai SDK, key GEMINI_API_KEY (switched away 2026-10-11: Google returned 403 "project denied access" on generation)
  - Switch by changing the one line `llm.provider` in config.yaml. No other code may import an LLM SDK directly.
- pyyaml (master resume), jinja2 + playwright (resume PDF via headless Chromium; WeasyPrint dropped because it needs GTK system libraries on Windows)
- Notion API via plain httpx (application log; no Notion SDK, chosen 2026-10-12), key NOTION_TOKEN + NOTION_PARENT_PAGE_ID in .env. Replaced Google Sheets / gspread (2026-10-12).
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
    notion.py            # sync the applications table to Notion (Status pulled back)
    backup.py            # dated jobs.db backups in data/backups/
    answers.py           # answer bank + exact question matching
    forms.py             # read/fill/snapshot Greenhouse, Lever, Ashby forms; submit safeguards
    gates.py             # shadow-mode auto-submit gates (no submit code)
    apply.py             # queue + review (pre-fill, hand over, log outcome) + confirm
    agent.py             # tool-use loop wiring the above as tools
    digest.py            # daily digest (Tier A / Tier B) after each run
  templates/resume.html.j2
  output/resumes/
  evals/
    label_me.csv         # 30 jobs Nithya labeled by hand (apply/maybe/skip); label.py labels in the terminal
    run_evals.py
  tests/
  main.py                # CLI: search | score | tailor | run | digest | close | queue | review | apply <job_id> | confirm | status | notion-sync
```

## Phases

### Phase 1 — Ingest
- Pull jobs from Greenhouse/Lever/Ashby for the company list in config.yaml (AI-native and enterprise-AI companies).
- Normalize to the Job model: id, company, title, location, remote, url, description, source, posted_at.
- posted_at is always an ORIGINAL posting date, never a last-updated date: Greenhouse `first_published` (never `updated_at`), Lever `createdAt`, Ashby `publishedAt` (the only date Ashby exposes). Missing = no date.
- Apply filters.py (title keywords, exclusion phrases) before saving. Save to SQLite with dedupe.
- JSearch (optional aggregator; free plan = 200 requests/month, HARD limit): 3 broad OR queries (AI engineer / solutions / forward deployed; data / analytics / business analyst; AI product / technical program manager) x 2 locations (United States, US remote), date_posted=3days, one page per request. Budget (config.yaml `jsearch`): at most 7 requests per UTC day (6 first pages, then page 2+ of queries whose previous page was full, only while budget remains), stop for the calendar month at 160, never exceed 200. Every request is recorded in the jsearch_requests table before it is sent; RapidAPI's own remaining count is also respected. Dry runs never call JSearch. Repost/aggregator sites (Talent.com, Dice, BeBee, JobLeads, Jooble, Lensa, ...: config jsearch.exclude_publishers / aggregator_hosts) are excluded in the request and again when parsing. Strict source rule: a JSearch job is kept only if its apply link is the employer's own (src/sources/jsearch.direct_link): an ATS board under the employer's name (Greenhouse/Lever/Ashby, or Workday, SmartRecruiters, iCIMS, ... in jsearch.employer_ats_hosts) or a page on the employer's own website domain. Job boards, aggregators, and JSearch's is_direct flag alone never count. Jobs without one are dropped entirely (never saved; already-saved ones are excluded unless tailored, queued, or applied). Before tailoring a JSearch job, src/linkcheck.py fetches the employer link (plain HTTP, then headless Chromium for JavaScript career sites) and requires that it loads, doesn't redirect to a job board, and still shows the same job title; otherwise the job is marked "unverified" (status), skipped without counting as a tool failure, and listed in Tier B with the reason. Direct-link jobs that review mode can't pre-fill are tailored and shown in the digest as "apply manually on employer site".
- Done when: python main.py search prints a count and the top 10 new jobs.

### Phase 2 — Score
- For each new job, call the LLM with the master resume and job description. Force JSON output matching ScoredJob: fit_score (0-100), role_family, matched_skills[], gaps[], sponsorship_signal (yes/no/unclear + quote), is_staffing_firm (bool), seniority_ok (bool), one_line_reason.
- Use the "fast" tier (Claude Haiku 5.5; Gemini: 3.5 Flash-Lite) for the first pass and the "strong" tier (Claude Sonnet 5.5; Gemini: 3.8 Flash) only for jobs scoring >= scoring.escalate_threshold (45; was 60 before the 2026-10-12 eval). Models per tier are set in config.yaml under llm.models.
- The scoring prompt stays neutral on sponsorship: sponsorship_signal is reported from the posting but never affects fit_score. It is used only for ordering (ranking above, tailoring order in Phase 4). The USCIS cross-check (sponsorship.py) is informational only.
- Ranking of results (scorer.rank_results): fit_score, then freshness (posted 0-3 days ago first), then sponsorship_signal (yes > unclear > no), then company tier (startup > growth > big_tech), then location priority. On top of that, a fresh (0-3 day) job goes ahead of an older job scoring up to 5 points higher (config.yaml ranking.freshness_preference_points). Sponsorship is a preference, never a filter.
- Cost control: print an estimate before each run and the actual cost after; never exceed scoring.max_run_cost_usd.
- Done when: python main.py score ranks jobs and stores scores.

### Phase 3 — Tailor (with guardrail)
- master_resume.yaml holds every bullet with an id. Tailoring may only select, reorder, and reword existing bullets and skills. No new tools, metrics, titles, dates, or employers.
- verify.py: second LLM call + string checks. For each output bullet, cite the source bullet id. Reject and retry if any number, tool, or claim is not in the source.
- Summary: never generated. Use the pre-approved summary in master_resume.yaml for the job's role family (sum_ai for AI roles, sum_analytics for Data Analyst / Analytics Engineer, sum_ba for Business Analyst). A role family with no approved summary (currently program_product_manager) is scored and ranked but never tailored until Nithya adds one.
- One page max (enforced by rendering the PDF and counting pages; cut content only, never add). No two bullets about the same work (master_resume.yaml same_work_groups). For AI and analytics roles always include the required metric bullet (config.local.yaml tailoring.required_bullets). Apply the bullet preferences in tailoring.preferences. Optional experience (tailoring.optional_experience) may be dropped to fit one page. Include the Job-Search Agent project for all AI roles. Headline: at most 4 skills.
- Coursework: pick the 4–6 courses most relevant to the job from master_resume.yaml; never list all of them.
- ATS keywords (src/keywords.py): before drafting, a fast-tier call extracts the posting's top 15-20 keywords in its exact wording (cached per job). CODE decides which ones Nithya has: a skill name, an approved `aliases` entry (same skill, other words) or `covers` entry (broader term the skill evidences, one-way) on that skill in master_resume.yaml, or the phrase in a bullet / project tech. Matched skills are shown in the posting's wording ("ELT", "Snowflake (data warehouse)"); bullets may use an approved wording only when the cited source names the skill. Anything else is never claimed. Only Nithya adds aliases/covers. Every verification report gets a keyword coverage section; `python main.py keywords [--run]` checks existing resumes read-only.
- Headline: may change per job. It is an optional role label plus skills from master_resume.yaml. The role label must come from config.yaml tailoring.headline_role_labels for the job's role family (AI Engineer, Applied AI, AI Solutions Engineer, Forward Deployed Engineer, AI Consultant, AI Evaluation, Data Analyst, Business Analyst, Analytics Engineer); no other labels. No seniority words anywhere in the headline (Senior/Lead/Principal/Staff). Never change experience-section titles; print them exactly as stored.
- Years of experience: always "4+ years". Never write "5+".
- Never use a master_resume.yaml field whose conflict is unresolved, or anything marked pending/outdated.
- Skills: every skill in master_resume.yaml is confirmed (2026-10-11), including Anthropic API, AWS, Azure, CI/CD, and the AI-evaluation skills. Tailoring selects only the skills relevant to each job, never the full list.
- Render to output/resumes/{company}_{role}.pdf.
- Done when: python main.py tailor <job_id> produces a PDF and a verification report.

### Phase 4 — Agent loop
- Expose search_jobs, score_job, tailor_resume, log_application as tools. Write a plain tool-use loop (no framework) so the loop is readable. Add tool-calling to the src/llm.py interface so the loop stays provider-neutral.
- Log every tool call (name, args, latency, tokens, cost) to logs/.
- Tailoring order (scorer.tailor_order, enforced in tailor_resume): the result ranking with its freshness preference, plus a sponsorship-"yes" job goes ahead of a non-"yes" job scoring up to 3 points higher (ranking.sponsorship_preference_points), except it never moves an older job ahead of a fresh one.
- Limits (config.yaml `agent`, enforced in code, not by the prompt): hard cost cap $0.50 per run (model turns + scoring + tailoring), max 30 tool calls, stop and report if any tool fails twice in a row, tailor only fit_score >= 50 (>= 45 for jobs posted 0-3 days ago: agent.tailor_min_score_fresh; lowered from 65/58 after the 2026-10-12 eval) and at most 5 per run, no apply tool (log_application can't set "applied"). `python main.py run --dry-run` plans and estimates with no LLM calls and no DB writes.
- Before tailoring ANY job (agent and `tailor` command), src/linkcheck.py checks the posting page for closure ("no longer accepting applications", "position filled", "job closed", HTTP 404/410, Greenhouse removed-job redirect). A closed posting gets status "closed" (job and its open application row), is skipped with no spend, and leaves ranking, the review queue, and the digest. `python main.py close <job_id> [--note]` marks one closed by hand. A page that just fails to load does not block an ATS-board job.
- After each run: queue the jobs tailored today (headless form read, review mode) and write output/digest_YYYY-MM-DD.md (src/digest.py; `python main.py digest [--date]` rebuilds it): run stats for the day from logs/ (searched, filtered out, passed, new, scored, tailored, JSearch requests, run cost); Tier A = up to 5 jobs tailored today, ranked (fit, freshness, company tier, one-line reason, queue status, link; more than 5 = an overflow line only); Tier B = the next 10 ranked untailored jobs with fit >= 50 (not applied/skipped) with fit, freshness, reason, link, the approved summary that fits (or none), and why it wasn't tailored. Limits in config.yaml `digest`.
- Done when: python main.py run does a full daily cycle autonomously up to (not including) applying.

### Phase 5 — Log + apply (review mode)
- Answer bank: data/answer_bank.yaml (gitignored; Nithya fills it in; template data/answer_bank.example.yaml is committed). Standard questions: work authorization, sponsorship, location, relocation, salary range, links; EEO = decline to answer. A form question matches an entry only on exact text (case/spacing/punctuation ignored, no fuzzy or LLM matching); choice answers must equal an option the form offers. Unmatched or unanswered questions are left for Nithya.
- Review mode is the only mode. `queue` reads each tailored job's form headlessly (Greenhouse, Lever, Ashby) and records how much the bank covers. `review` opens the form in a visible browser, pre-fills exact matches, attaches the verified tailored PDF, outlines what's left, and hands over: Nithya reviews and clicks Submit herself. `confirm <job_id> --applied|--skipped` records the result (also for applications she submitted by hand).
- Never submit. No code clicks a submit control (safe_click refuses submit-like buttons) and submissions are blocked while the agent fills. Never touch LinkedIn/Indeed. Never attempt to solve or bypass a CAPTCHA.
- Log for each application whether Nithya approved unchanged or edited (form snapshot after pre-fill vs at her Submit click; edited questions listed; edits to bank answers flagged).
- Shadow-mode auto-submit gates (config.yaml `apply`, auto_submit: false; there is no submit code and true is rejected): record whether an application WOULD qualify: fit >= 70, Greenhouse or Lever, no CAPTCHA of any kind, every question matched exactly from the bank, no long-answer box (short text fields are fine if matched exactly), at most 5 per day, at most 2 applications per company in 90 days and only in different role families (manual ones included; an unknown role family counts as the same), verified one-page resume, posting open. Everything goes to the review queue regardless.
- notion.py (replaced Google Sheets 2026-10-12): database "Applications" under the parent page (NOTION_PARENT_PAGE_ID), columns Company (title), Role, Family, Fit, Status (select: queued, applied, interviewing, offer, rejected, skipped, closed), Date applied, Days since applied (Notion formula), Resume file, Job link, Notes, plus Job ID (the upsert key). jobs.db is the source of truth; synced at the start and end of each `run` and after queue/review/confirm/close/status (`notion-sync` by hand). Only Status flows back: a Status changed in Notion is applied to jobs.db on the next sync (via apply.change_status, same as the CLI); if it changed in both places, jobs.db wins and the conflict is reported. `python main.py status <job_id> <status>` sets interviewing/offer/rejected (and the others) by hand.
- Backups: each `run` first writes data/backups/jobs_YYYY-MM-DD.db (SQLite backup API, gitignored), keeping the last 14 days (config.yaml backup.keep_days).
- Done when: one real application is pre-filled and the log row updates to "applied" after she confirms.

### Phase 6 — Evals + deploy
- evals/run_evals.py: compare agent fit scores to her hand labels (agreement %, mean absolute error, confusion on apply/skip). Run before/after any prompt change.
- CI (done 2026-10-12): .github/workflows/tests.yml runs pytest on every push and PR (ubuntu, Python 3.11, Chromium for the form/PDF tests). No secrets, no personal data: the tests must pass without data/ or .env, and a guard step fails if .env, jobs.db, the master resume, the answer bank, backups, output/ or logs/ are ever tracked. Tests never write to the real logs/ (tests/conftest.py). The daily run stays on the local Task Scheduler (it needs data/ and keys), not on Actions.
- Dockerfile (done 2026-10-12): python:3.11.13-slim-bookworm + Chromium, non-root, headless commands only (review runs on the host). data/ and .env are mounted at runtime (-v data, --env-file .env), never copied: the Dockerfile copies only named code paths and .dockerignore excludes them too. CI job "docker" builds the image on every push/PR (never runs or pushes it, no secrets) and fails if the image contains .env, data/ files (except the example answer bank), output/, logs/ or tests/.
- Local schedule (done 2026-10-12): Windows Task Scheduler task "JobAgentDaily" runs run_daily.ps1 (`main.py run` with .venv Python) weekdays 8:00 AM as Nithya (interactive logon), runs as soon as possible after a missed start, 1-hour limit; output in logs\scheduled_*.log. `run_daily.ps1 -DryRun` checks the setup at no cost.
- README: architecture diagram, eval results, demo GIF.

## Rules for Claude Code
- Small commits per phase with clear messages.
- Ask before adding any dependency not listed above.
- Explain non-obvious code briefly when you write it.
- Never commit .env, jobs.db, or generated resumes.
