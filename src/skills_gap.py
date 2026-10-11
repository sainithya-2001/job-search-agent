"""Skills-gap report over every scored job.

Counting is deterministic: a fixed vocabulary of skills, each with regex patterns, matched against the
requirements/qualifications part of each posting (company blurbs and benefits are cut off where headers allow).
The same patterns are run over the master-resume profile to decide "in resume: yes/no". A second section lists
the gap phrases the scoring model wrote most often, as a cross-check on what the vocabulary might miss.
"""
import json
import re
from collections import Counter
from datetime import date

# (canonical name, patterns). Case-insensitive unless a pattern uses (?-i:...).
VOCAB: list[tuple[str, list[str]]] = [
    # languages
    ("Python", [r"\bpython\b"]), ("SQL", [r"\bsql\b"]), ("TypeScript", [r"\btypescript\b"]),
    ("JavaScript", [r"\bjavascript\b", r"\bnode\.?js\b"]), ("Java", [r"\bjava\b(?!script)"]),
    ("Go", [r"\bgolang\b", r"(?-i:\bGo\b)(?=\s*[,/)]|\s+(and|or)\s)"]), ("Rust", [r"\brust\b"]),
    ("R", [r"(?-i:\bR\b)(?=\s*[,/)]|\s+(and|or)\s)"]), ("Scala", [r"\bscala\b"]),
    # data stack
    ("dbt", [r"\bdbt\b"]), ("Snowflake", [r"\bsnowflake\b"]), ("BigQuery", [r"\bbigquery\b"]),
    ("Redshift", [r"\bredshift\b"]), ("Databricks", [r"\bdatabricks\b"]), ("Spark / PySpark", [r"\b(py)?spark\b"]),
    ("Airflow", [r"\bairflow\b"]), ("Kafka", [r"\bkafka\b"]), ("Fivetran", [r"\bfivetran\b"]),
    ("Looker", [r"\blooker\b"]), ("Tableau", [r"\btableau\b"]), ("Power BI", [r"\bpower ?bi\b"]),
    ("Excel", [r"\bexcel\b"]), ("Pandas", [r"\bpandas\b"]), ("PostgreSQL", [r"\bpostgres(ql)?\b"]),
    ("Data modeling", [r"\bdata model(ing|ling|s)?\b", r"\bdimensional model"]),
    ("Semantic layer / metrics layer", [r"\bsemantic layer\b", r"\bmetrics layer\b"]),
    ("ETL / data pipelines", [r"\betl\b", r"\belt\b", r"\bdata pipelines?\b"]),
    ("A/B testing / experimentation", [r"\ba/b test", r"\bexperimentation\b"]),
    ("Statistics", [r"\bstatistic(s|al)\b"]), ("Dashboards / BI", [r"\bdashboards?\b", r"\bbusiness intelligence\b"]),
    # AI / ML
    ("LLMs", [r"\bllms?\b", r"\blarge language models?\b"]), ("RAG", [r"\brag\b", r"\bretrieval[- ]augmented"]),
    ("Prompt engineering", [r"\bprompt(ing| engineering| design)\b"]),
    ("Fine-tuning", [r"\bfine[- ]?tun(e|ing)\b"]), ("Embeddings", [r"\bembeddings?\b"]),
    ("Vector databases", [r"\bvector (db|database|store|search)s?\b", r"\bpinecone\b", r"\bpgvector\b", r"\bfaiss\b", r"\bweaviate\b"]),
    ("AI agents / agentic workflows", [r"\bagents?\b", r"\bagentic\b"]),
    ("Tool / function calling", [r"\btool[- ]calling\b", r"\bfunction[- ]calling\b", r"\btool use\b"]),
    ("MCP (Model Context Protocol)", [r"\bmcp\b", r"\bmodel context protocol\b"]),
    ("LLM evaluation / evals", [r"\bevals?\b", r"\bevaluation (framework|harness|pipeline)s?\b", r"\bmodel evaluation\b"]),
    ("LangChain / LangGraph", [r"\blang(chain|graph|smith)\b"]), ("LlamaIndex", [r"\bllama ?index\b"]),
    ("OpenAI API", [r"\bopenai\b(?! codex)"]), ("Anthropic / Claude API", [r"\banthropic\b", r"\bclaude\b(?! code)"]),
    ("AI coding tools (Cursor, Claude Code, Copilot)", [r"\bclaude code\b", r"\bcursor\b", r"\bcopilot\b", r"\bcodex\b"]),
    ("Hugging Face", [r"\bhugging ?face\b"]), ("PyTorch", [r"\bpytorch\b"]), ("TensorFlow", [r"\btensorflow\b"]),
    ("scikit-learn", [r"\bscikit[- ]learn\b", r"\bsklearn\b"]), ("XGBoost", [r"\bxgboost\b"]),
    ("Machine learning", [r"\bmachine learning\b", r"\b(?-i:ML)\b"]), ("Deep learning", [r"\bdeep learning\b"]),
    ("NLP", [r"\bnlp\b", r"\bnatural language processing\b"]), ("Speech / voice AI", [r"\bspeech\b", r"\bvoice ai\b", r"\btelephony\b"]),
    ("Computer vision", [r"\bcomputer vision\b"]), ("RLHF / reinforcement learning", [r"\brlhf\b", r"\breinforcement learning\b"]),
    ("MLOps / model deployment", [r"\bmlops\b", r"\bmodel (deployment|serving)\b"]),
    ("Observability / tracing", [r"\bobservability\b", r"\btracing\b"]), ("Guardrails / AI safety", [r"\bguardrails?\b"]),
    # engineering / cloud
    ("AWS", [r"\baws\b", r"\bamazon web services\b"]), ("GCP", [r"\bgcp\b", r"\bgoogle cloud\b"]), ("Azure", [r"\bazure\b"]),
    ("Docker", [r"\bdocker\b", r"\bcontaineri[sz]"]), ("Kubernetes", [r"\bkubernetes\b", r"\bk8s\b"]),
    ("Terraform", [r"\bterraform\b"]), ("CI/CD", [r"\bci/cd\b", r"\bcontinuous (integration|deployment)\b"]),
    ("Git", [r"\bgit(hub)?\b"]), ("REST APIs", [r"\brest(ful)?\b", r"\bapis?\b"]), ("GraphQL", [r"\bgraphql\b"]),
    ("Webhooks / integrations", [r"\bwebhooks?\b"]), ("SDKs", [r"\bsdks?\b"]),
    ("FastAPI / Flask / Django", [r"\bfastapi\b", r"\bflask\b", r"\bdjango\b"]), ("React", [r"\breact\b"]),
    ("Distributed systems", [r"\bdistributed systems?\b"]), ("Production software engineering", [r"\bproduction (systems|code|software|environments?)\b"]),
    ("Linux", [r"\blinux\b"]),
    # customer-facing / business
    ("Customer-facing / client-facing", [r"\b(customer|client)[- ]facing\b"]),
    ("Pre-sales / sales engineering", [r"\bpre[- ]?sales\b", r"\bsales engineer", r"\bsales cycles?\b"]),
    ("Demos / POCs", [r"\bdemos?\b", r"\bproofs? of concept\b", r"\bpocs?\b"]),
    ("Solution architecture", [r"\bsolutions? architect", r"\barchitect(ing)? solutions\b"]),
    ("Consulting", [r"\bconsult(ing|ant)\b"]), ("Stakeholder management", [r"\bstakeholders?\b"]),
    ("Requirements gathering", [r"\brequirements\b(?= gathering| analysis)"]), ("Project management", [r"\bproject management\b"]),
    ("Product management / product sense", [r"\bproduct (management|sense|manager)\b"]), ("Agile / Scrum", [r"\bagile\b", r"\bscrum\b"]),
    ("Salesforce / CRM", [r"\bsalesforce\b", r"\bcrm\b"]), ("SaaS", [r"\bsaas\b"]),
    ("Technical writing / documentation", [r"\bdocumentation\b", r"\btechnical writing\b"]),
    ("Healthcare domain", [r"\bhealthcare\b", r"\bclinical\b", r"\behr\b"]), ("Finance / FP&A domain", [r"\bfp&a\b", r"\bfinancial (services|modeling)\b"]),
    ("Legal domain", [r"\blegal (industry|tech|domain|operations|professionals|work|practice|teams)\b", r"\blaw firms?\b",
                      r"\battorneys?\b", r"\blawyers?\b"]),
    ("Startup experience", [r"\bstartups?\b", r"\bearly[- ]stage\b"]),
]

_REQ_START = re.compile(
    r"(requirements|qualifications|what you('ll| will)? bring|what we('re| are) looking for|you (might |should |will )?have|"
    r"about you|who you are|you may be a (good )?fit|skills|nice to have|preferred|bonus|you are|we('re| are) looking for)", re.I)
_REQ_END = re.compile(r"\n\s*(benefits|compensation|what we offer|perks|our benefits|salary|pay range|equal opportunity|"
                      r"why join|life at)\b|to protect yourself|recruiters (will )?only contact|privacy notice", re.I)
_URL = re.compile(r"https?://\S+|\b[\w-]+\.(com|ai|io|co)/\S*", re.I)


def requirements_text(description: str, company: str = "") -> str:
    """The part of a posting most likely to hold requirements; the whole text if no header is found.
    URLs and the hiring company's own name are removed so they can't count as skills."""
    m = _REQ_START.search(description or "")
    text = description[m.start():] if m else (description or "")
    end = _REQ_END.search(text)
    text = text[: end.start()] if end else text
    text = _URL.sub(" ", text)
    if company:
        text = re.sub(re.escape(company), " ", text, flags=re.I)
    return text


def _compiled():
    return [(name, [re.compile(p, re.I) for p in pats]) for name, pats in VOCAB]


def build_report(scored_rows, resume_text: str, top_n: int = 20) -> tuple[str, list[dict]]:
    vocab = _compiled()
    counts, examples = Counter(), {}
    for r in scored_rows:
        text = requirements_text(r["description"], r["company"])
        for name, pats in vocab:
            if any(p.search(text) for p in pats):
                counts[name] += 1
                examples.setdefault(name, []).append(r["company"])
    in_resume = {name: any(p.search(resume_text) for p in pats) for name, pats in vocab}

    total = len(scored_rows)
    rows = [{"skill": name, "jobs": n, "pct": round(100 * n / total) if total else 0, "in_resume": in_resume[name],
             "examples": sorted(set(examples[name]))[:4]} for name, n in counts.most_common(top_n)]

    gap_counts = Counter()
    for r in scored_rows:
        for g in json.loads(r["gaps"] or "[]"):
            gap_counts[re.sub(r"\s+", " ", g.strip().lower())] += 1

    lines = [f"# Skills-gap report", "",
             f"Generated {date.today().isoformat()} from **{total} scored jobs**. A job counts once per skill if the skill",
             "appears in its requirements/qualifications section (or the whole posting when no such header exists).",
             "\"In resume\" means the same pattern matches the master-resume profile the scorer sees.", "",
             f"## Top {len(rows)} skills asked for", "",
             "| # | Skill | Jobs asking | % of jobs | In master_resume | Example companies |",
             "|---|---|---|---|---|---|"]
    for i, row in enumerate(rows, 1):
        lines.append(f"| {i} | {row['skill']} | {row['jobs']} | {row['pct']}% | {'yes' if row['in_resume'] else '**no**'} | "
                     f"{', '.join(row['examples'])} |")
    missing = [r["skill"] for r in rows if not r["in_resume"]]
    lines += ["", f"**Not in master_resume ({len(missing)}):** " + (", ".join(missing) if missing else "none"), "",
              "## Gaps the scoring model named most often", "",
              "Free-text phrases from the scorer's `gaps` field (lightly normalized), as a cross-check on the vocabulary above.", "",
              "| Gap (as written by the model) | Jobs |", "|---|---|"]
    lines += [f"| {g} | {n} |" for g, n in gap_counts.most_common(15)]
    return "\n".join(lines) + "\n", rows
