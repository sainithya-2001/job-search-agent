# Architecture: one job, end to end

This walks one real posting through the code: an **"AI Strategist"** role at an AI startup, on Greenhouse.
The company is anonymized here as "Company X" (board slug `companyx`). Each step names the file and function,
what happened to this job, the design decision, and the alternative that was rejected. The numbers are the
stored values from `jobs.db` and the run logs.

| Step | Where | This job |
|---|---|---|
| 1. Fetch | `src/sources/__init__.py: fetch_all` → `greenhouse.fetch` | one of ~6,000 postings that day |
| 2. Normalize | `src/sources/greenhouse.py: parse` → `src/models.py: Job` | id `greenhouse:companyx:<job id>`, remote, dated |
| 3. Filter | `src/filters.py: JobFilter.check` | kept, role family `ai_consultant` |
| 4. Store | `src/store.py: Store.save` | new row, dedupe key `companyx\|ai strategist` |
| 5. Score | `src/scorer.py: score_jobs` | Haiku 60 → escalated → Sonnet 66, $0.018 |
| 6. Rank | `src/scorer.py: rank_results`, `tailor_order` | above the tailoring cutoff of the time |
| 7. Link check | `src/linkcheck.py: check_before_tailoring` | posting open |
| 8. Keywords | `src/keywords.py: get_keywords`, `match` | 20 keywords, 5 supported, 15 never claimed |
| 9. Tailor | `src/pipeline.py: tailor_and_publish` → `src/tailor.py: tailor_job` | approved summary `sum_ai`, one draft |
| 10. Verify | `src/verify.py: Verifier.check_structure`, `check_with_llm` | passed on attempt 1, 2 API calls, $0.045 |
| 11. Render | `src/render.py: fit_one_page` | one-page PDF + verification report |
| 12. Queue | `src/apply.py: queue_jobs` | 10 of 11 questions answered from the bank; reCAPTCHA |
| 13. Review | `src/apply.py: review` | waiting for the human: the agent never submits |
| 14. Notion | `src/notion.py: sync` | page created, Status `queued` |

All LLM calls in steps 5, 8, 9 and 10 go through `src/llm.py` (`LLMProvider.generate_json`). The daily run
drives steps 1–12 through a tool-use loop in `src/agent.py`; each step can also be run by hand from `main.py`.

---

## 1. Fetch

**Where:** `src/pipeline.py: run_search` calls `src/sources/__init__.py: fetch_all`. That function fans out over
the 119 boards in `config.yaml` (`companies:`), with at most 10 requests in flight. For this company it calls
`src/sources/greenhouse.py: fetch`, which gets `boards-api.greenhouse.io/v1/boards/companyx/jobs?content=true`.

**This job:** one of about 6,000 postings fetched that day (about 8,200 a day now, after more boards were added). A board that fails (404, timeout)
is recorded in `FetchResult.errors` and the run continues.

**Decision:** read the ATS vendors' public JSON board APIs (Greenhouse, Lever, Ashby). They are stable, need no
key, and return the full description in one request per company.
**Rejected:** scraping LinkedIn or Indeed. Their terms forbid it, the HTML breaks often, and the same job shows
up many times as reposts. Aggregators are used only through JSearch, and only for postings whose apply link
goes to the employer's own site (step 2).

## 2. Normalize

**Where:** `src/sources/greenhouse.py: parse` builds a `src/models.py: Job` (pydantic). Lever and Ashby have
their own `parse`, and all three produce the same model.

**This job:**
- `id = "greenhouse:companyx:<job id>"` (source, board slug, the vendor's id).
- `location = "United States (Remote)"`, so `remote = True`. Greenhouse has no remote flag; the location
  string is the only signal.
- The HTML description is converted to plain text (`src/sources/common.py: html_to_text`).
- `posted_at` = an early-October 2026 date, taken from `first_published`.

**Decision:** `posted_at` is always the *original* posting date: Greenhouse `first_published`, Lever
`createdAt`, Ashby `publishedAt`. A job with no such field has no date. Freshness drives ranking and the
14-day cutoff, so a wrong date silently promotes old jobs.
**Rejected:** falling back to `updated_at`. An old posting edited yesterday would look brand new.

## 3. Filter

**Where:** `run_search` passes every fetched job through `src/filters.py: JobFilter.check`. The rules come
from `config.yaml` `filters:`.

**This job:** `role_family("AI Strategist")` matched the `ai_consultant` family. The title isn't excluded (no
Staff, Principal or Lead; no frontend, platform or region words). It was 5 days old (limit 14). The location is
US remote. No exclusion phrase (citizenship, clearance, export control, staffing agency) appears in the
description. Result: `FilterResult(keep=True, role_hint="ai_consultant")`.

**Decision:** cheap, deterministic rules run before any LLM call. The checks are ordered from cheapest to
dearest: title regex, then date, location, and finally a description phrase scan. About 99% of postings stop
here, so the model only sees ~60 jobs a day instead of ~8,200. Every rejection is counted by reason, so a rule
that is too broad shows up in the run stats.
**Rejected:** letting the LLM judge eligibility. At this volume it would mean about 100x more calls, and it would be
non-deterministic about hard rules like "must be a US citizen", which a regex gets right every time.

## 4. Store

**Where:** `src/store.py: Store.save`, inside one transaction for the batch. After saving,
`src/pipeline.py: recheck_existing` re-applies the current filters to jobs already in the database.

**This job:** inserted with `dedupe_key = "companyx|ai strategist"` (`dedupe_key` + `normalize_title`, which drop
parentheticals, normalize "Sr." and strip punctuation). Status `new`.

**Decision:**
- SQLite is the single source of truth, holding jobs, scores, applications, the form queue, the JSearch
  request budget and the Notion sync state.
- `INSERT OR IGNORE` on the id plus a separate dedupe key means re-fetching every day is idempotent.
- Re-checking stored jobs means a new filter rule also applies to jobs saved before the rule existed. This is
  how a "Fullstack" job that slipped in early was excluded later. Jobs already tailored, queued or applied are
  never hidden.

**Rejected:**
- Postgres or an ORM: a single-user local tool doesn't need a server.
- JSON files: no transactions, no queries.

## 5. Score

**Where:**
- `src/scorer.py: score_jobs` builds the prompt from `system_prompt(cfg, master)` (the candidate profile from
  `src/resume.py: profile_text`, which never includes contact details) and `job_prompt(job)`.
- It calls `llm.generate_json(prompt, ScoredJob, tier="fast")`. In `src/llm.py: AnthropicProvider._call` this is
  `client.messages.parse(..., output_format=ScoredJob)`, so the response is validated against the pydantic
  schema.

**This job:**
- Claude Haiku returned `fit_score = 60`. That met the escalation threshold, which was 60 at the time and is 45
  now.
- Claude Sonnet re-scored it at **66**, with role family `ai_consultant` and 8 matched skills.
- It listed 7 gaps, such as domain analytics and ROI business cases, plus `sponsorship_signal =
  unclear`, `is_staffing_firm = false`, `seniority_ok = true` and a one-line reason.
- 14,570 input tokens and 893 output tokens: **$0.018**.

**Decision:**
- Two tiers. The cheap model scores everything, and the strong model re-scores only jobs near or above the
  cutoff, where a few points decide whether a resume gets tailored.
- The output is schema-enforced, so downstream code never parses free text.
- Before each call, `score_jobs` checks the projected spend (the larger of 2x the estimate and the most
  expensive call seen so far) against the cap. An underestimate can't push the run past it.
- The scoring prompt is neutral on sponsorship: the signal is reported with a quote but never changes the fit
  score.

**Rejected:**
- Sonnet on every job: its tokens cost about 20x more, and the eval later showed Haiku and Sonnet rank jobs
  about equally well (Spearman 0.69 vs 0.68).
- Asking for JSON in the prompt and parsing it: malformed output becomes a runtime error instead of a schema
  violation the SDK catches.

## 6. Rank

**Where:** `src/scorer.py: rank_results` (display and digest) and `tailor_order` (what gets tailored first).
Both sort with `result_sort_key`: fit, then freshness, then sponsorship signal, company tier and location
priority. `preference_order` then lets a fresh job (posted in the last 3 days) pass an older job scoring up to
5 points higher. In `tailor_order` only, a sponsorship "yes" job can pass a job scoring up to 3 points higher,
but never an older one past a fresh one. `hide_stale` drops untailored jobs older than 14 days.

**This job:** 66 was above the tailoring cutoff in force that day (65; it is 50 now, 45 for fresh jobs).

**Decision:** a lexicographic sort key with small, named point bands. Each rule can be explained in one
sentence and tested on its own.
**Rejected:** one weighted score that blends fit, freshness and tier. The weights would be arbitrary, and a
blend hides *why* one job ranks above another.

## 7. Link check (before any tailoring spend)

**Where:** `src/agent.py: Toolbox.tailor_resume` refuses anything below `tailor_threshold`, anything already
tailored, or anything out of order. It then calls `src/linkcheck.py: check_before_tailoring`. For a board job
that is `posting_closed`: a fetch looking for "no longer accepting applications", HTTP 404/410 or the
Greenhouse removed-job redirect. A JSearch job instead needs `verify_posting`: the employer page must load, must
not redirect to a job board, and must still show the same title.

**This job:** the posting was open, so tailoring went ahead.

**Decision:** tailoring is the most expensive step, so check the cheap thing first. A closed posting gets
status `closed` and disappears from the ranking, the queue and the digest.
**Rejected:** trusting the board feed. A posting can be filled before the board removes it, and the check costs
one page load against a tailoring run of about $0.05.

## 8. ATS keyword extraction

**Where:** `src/pipeline.py: job_coverage` → `src/keywords.py: get_keywords`. A Haiku call returns the
posting's top 15–20 keywords in its exact wording; the result is cached per job in `job_keywords`. Then
`keywords.match` runs. This is code, not a model, and it decides which keywords the candidate actually has: a
skill name, an approved alias or `covers` entry in `master_resume.yaml`, or the phrase in a bullet or project.

**This job:** 20 keywords. Five were supported. The other 15, among them "AI strategy", "AI roadmap", "ROI",
"voice of customer" and "A/B tests", are listed in the report as *never claimed*. This job was tailored the day
before this step existed, so its keywords were extracted afterwards by the read-only `python main.py keywords
--run`. New jobs get the keywords before drafting, and the supported ones steer the draft and the skills
wording.

**Decision:** the model only *extracts*; code *matches*. Matching is exact, against synonyms the candidate
approved. The coverage number is honest: missing keywords are reported, never added.
**Rejected:** asking the LLM to "optimize the resume for these keywords". That is how keyword-stuffed,
fabricated resumes get made.

## 9. Tailor

**Where:** `src/pipeline.py: tailor_and_publish`:
1. Before any LLM call, `src/verify.py: select_summary` picks the pre-approved summary for the role family.
   No summary means no tailoring.
2. A `Verifier` is built with this job's rules.
3. `src/tailor.py: tailor_job` sends `build_prompt` (the job, its score, the rules, the supported keywords, and
   `src/resume.py: tailoring_source`, an id-labelled list of every citable fact) to the strong tier, with the
   `TailoredResume` schema.

**This job:**
- Summary `sum_ai`, which covers the AI families including `ai_consultant`.
- The draft selected and reworded existing bullets. Each bullet carries the `source_ids` it came from.
- Headline: the label "AI Consultant" (the only allowed label for `ai_consultant` in `config.yaml
  tailoring.headline_role_labels`) plus 4 skills.

**Decision:** the model can only *select, reorder and reword*. The schema has no field for free text without
sources, and the summary isn't generated at all.
**Rejected:** generating the resume freely and fact-checking it afterwards. A grounded generator plus a checker
beats an unconstrained generator plus a checker, because there is less to catch.

## 10. Verify

**Where:**
- `src/verify.py: Verifier.check_structure` runs string checks on every bullet:
  - every number and every known tool name must appear in the cited sources;
  - every cited id must exist;
  - at most one bullet per `same_work_groups` group;
  - the required bullets and projects must be there;
  - headline rules (label list, at most 4 skills, no seniority words);
  - coursework count;
  - years-of-experience wording must match the master resume.
- Only a draft that passes all of these gets `check_with_llm`: a second call, a strict fact-checker prompt
  that judges every bullet against its sources.
- On failure, `tailor_job` retries with the issues as feedback, up to 3 attempts. If drafts still fail,
  `_remove_failing` drops what failed, and the result passes only if everything left passes both checks.

**This job:** passed on attempt 1. 2 API calls (draft + fact-check), **$0.045**.

**Decision:** deterministic checks first (free, exact), then the LLM judge (costly, catches meaning).
**Rejected:**
- An LLM-only judge: numbers and tool names can be checked exactly, so leaving them to a model adds risk for no
  gain.
- Regex only: it can't see "led" replacing "supported".
- Silently "fixing" bad bullets: removed content is listed in the report instead.

## 11. Render

**Where:** `src/pipeline.py: publish` → `src/render.py: fit_one_page`.
- `render_html` fills `templates/resume.html.j2`. Contact details, titles, dates and the summary come from
  `master_resume.yaml` by id, not from the model.
- `write_pdf` prints it with headless Chromium (Playwright) and returns the page count.
- While the page count is over 1, the loop cuts: optional experience first, then the last bullet of the longest
  section. Required bullets are never cut, and nothing is ever added.
- Outputs: `output/resumes/companyx_ai_strategist.{pdf,html,json}` plus `_verification.md`. The report shows
  every claim with its sources, the cuts, the cost and keyword coverage.

**This job:** one page, verified.

**Decision:** measure the real PDF instead of guessing length, and fit by cutting only.
**Rejected:**
- WeasyPrint: it needs GTK system libraries, which don't install cleanly on Windows.
- Estimating length from character counts: wrong whenever the wrapping changes.
- `.docx` output: ATS parsing of a simple HTML-to-PDF layout is just as good, and the HTML is easy to test.

## 12. Queue

**Where:** `src/apply.py: queue_jobs` → `_queue_one`, which:
- opens the form in headless Chromium and finds the ATS (`src/forms.py: job_form`);
- runs `block_submits` first, so nothing can submit while it reads;
- reads the questions (`wait_for_form`, `read_questions`) and runs `detect_captcha`, which looks at the page
  and at the network requests;
- matches every question against the answer bank (`src/answers.py: match_form`): exact text after normalizing
  case and punctuation, and a choice answer must be one of the form's options;
- runs the shadow auto-submit gates (`src/gates.py: evaluate`).

The results go to `apply_queue`, and the application row becomes `queued`.

**This job:**
- 11 questions, 10 answered from the bank. "Location (City)" had no exactly matching entry, so it is left for
  the human.
- reCAPTCHA detected.
- Gates: `fit_score 66 (needs >= 70)`, `no_captcha: recaptcha` and `all_matched: 1 not matched` all failed. It
  would not qualify for auto-submit, and auto-submit doesn't exist anyway.

**Decision:** exact matching only. A wrong answer on an application is worse than a blank one.
**Rejected:**
- Fuzzy or LLM question matching: "Are you authorized to work in the US?" and "Will you require sponsorship?"
  are close in wording but opposite in meaning.
- Real auto-submit: the gates run in shadow mode to *measure* how often it would be safe. There is no submit
  code, and `auto_submit: true` is rejected at startup (`gates.check_config`).

## 13. Review (human in the loop)

**Where:** `src/apply.py: review` runs these steps in order:
1. Opens the form in a **visible** browser and runs `block_submits`.
2. Fills each exact match (`src/forms.py: fill`) and attaches the verified PDF.
3. Highlights what's left and takes a snapshot.
4. Calls `hand_over`, which lifts the block.
5. The human checks everything and clicks Submit. The click is detected, and a second snapshot is taken at that
   moment.
6. `classify` compares the two snapshots, so the log records "approved unchanged" or exactly which answers were
   edited, and flags edits to bank answers.
7. The human confirms they saw the employer's confirmation, and the status becomes `applied`.

`src/forms.py: safe_click` refuses to click anything that looks like Submit or Apply.

**This job:** queued and waiting for review. This is where the agent stops: it does not submit.

**Decision:** automate the typing, keep the decision. The edit log shows how much the pre-fill can be
trusted.
**Rejected:**
- CAPTCHA solving: never. A form with a CAPTCHA goes to the human like every other form.
- A browser extension: Playwright with a visible window gives the same hand-over with no install, and the same
  code is used headless in step 12.

## 14. Notion sync

**Where:** `src/notion.py: try_sync` → `sync`, called at the start and end of every `run` and after `queue`,
`review`, `confirm`, `close` and `status`. It talks to the Notion REST API over plain httpx (`NotionClient`),
with retry on 429/5xx.
1. **Pull.** For each page whose Status differs from the last synced value, apply it through
   `src/apply.py: change_status`, the same path as the CLI. If jobs.db changed too, jobs.db wins and the
   conflict is reported.
2. **Push.** Each application row becomes Notion properties. A hash of those properties is stored in
   `notion_sync`, so only changed rows are sent.

**This job:** a page in the "Applications" database (Company, Role, Family, Fit 66, Status `queued`, Resume
file, Job link). When the status moves to interviewing or rejected in Notion, the next run pulls it back.

**Decision:**
- SQLite stays the source of truth; Notion is a view the human can edit in exactly one field.
- Upsert by job id, and send only changed rows.

**Rejected:**
- Notion as the database: slow, rate-limited, and not testable offline.
- The `notion-client` SDK: five endpoints didn't justify a dependency.
- Google Sheets: the first version used it; Notion replaced it on 2026-10-12.

---

## Around the pipeline

- **Agent loop** (`src/agent.py: run_agent`). Claude Haiku picks the next tool: `search_jobs`, `score_jobs`,
  `tailor_resume` or `log_application`. Every limit is in `Toolbox`, not in the prompt:
  - a $0.50 cost cap per run;
  - 30 tool calls;
  - stop after a tool fails twice in a row;
  - at most 5 resumes;
  - the tailoring threshold and order.

  `log_application` cannot set `applied`. Every tool call is logged to `logs/run_*.jsonl` with arguments,
  latency, tokens and cost.
- **Digest** (`src/digest.py: write`). It writes `output/digest_YYYY-MM-DD.md` after each run: run stats, Tier A
  (what was tailored, with its queue status) and Tier B (the next 10 ranked jobs and why each wasn't tailored).
- **Provider interface** (`src/llm.py`). `generate_json` (structured output) and `chat` (tool calling) for
  Anthropic and Gemini. Retry with backoff on 429/5xx, and cost from a pricing table. Switching providers is one
  line in `config.yaml`.
- **Evals** (`evals/run_evals.py`). Compares fit scores with hand labels, and is how the thresholds above were
  chosen. See the README.
