"""Answer bank: Nithya's own answers to standard application questions, matched EXACTLY to form questions.

data/answer_bank.yaml (gitignored; template: data/answer_bank.example.yaml) holds entries like
    - id: sponsorship
      questions: ["Will you now or in the future require sponsorship for employment?"]
      answer: "Yes"
A form question matches an entry only when its text equals one of the entry's `questions` after normalization
(case, spacing, and punctuation ignored; nothing else — no fuzzy or semantic matching). For choice questions the
answer must also equal one of the options the form offers. Anything that doesn't match exactly is left for Nithya.

Answers:
  answer: "text"                      short text, or the option to pick
  answer: ["opt A", "opt B"]          choice questions: the first of these the form offers (e.g. EEO decline wordings)
  check: ["opt A", "opt B"]           multi-select: check all of these; every one must be offered
  answer: "@email"                    @tokens come from master_resume.yaml / the job: @full_name @email @phone
                                      @location @resume (the job's tailored PDF)
  skip: true                          leave blank on purpose (only allowed for optional questions)
  answer: null                        not filled in yet: always left for Nithya
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
BANK_PATH = ROOT / "data" / "answer_bank.yaml"

# Question kinds (from src/forms.py)
TEXT_KINDS = {"text", "email", "tel", "number", "url"}
LONG_KINDS = {"textarea"}
CHOICE_KINDS = {"select", "combobox", "radio", "yesno"}

# Match statuses
ANSWERED = "answered"            # exact match; value ready to fill
SKIPPED = "skipped"              # bank says leave blank (optional question)
NO_ENTRY = "no_entry"            # no bank entry with this exact question text
NEEDS_ANSWER = "needs_answer"    # entry exists but answer is null, or skip on a required question
NO_OPTION = "no_matching_option" # bank answer isn't one of the options this form offers
UNVERIFIED = "unverified"        # answer exists but the options can't be checked in advance (live lookups)


class BankError(ValueError):
    pass


def normalize(text: str) -> str:
    """'Are you legally authorized to work in the US? *' -> 'are you legally authorized to work in the us'."""
    t = (text or "").replace("✱", " ").lower()
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


@dataclass
class Entry:
    id: str
    questions: list[str]
    answer: object = None
    check: list[str] | None = None
    skip: bool = False


@dataclass
class AnswerBank:
    entries: list[Entry]
    tokens: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self.by_question: dict[str, Entry] = {}
        for e in self.entries:
            for q in e.questions:
                key = normalize(q)
                if key in self.by_question and self.by_question[key].id != e.id:
                    raise BankError(f"question {q!r} is in both {self.by_question[key].id} and {e.id}")
                self.by_question[key] = e

    def find(self, label: str) -> Entry | None:
        return self.by_question.get(normalize(label))

    def unfilled(self) -> list[str]:
        return [e.id for e in self.entries if e.answer is None and not e.check and not e.skip]


def load_bank(path: Path = BANK_PATH, tokens: dict[str, str] | None = None) -> AnswerBank:
    if not path.exists():
        raise BankError(f"{path.name} not found. Copy data/answer_bank.example.yaml to data/answer_bank.yaml "
                        f"and fill in your answers.")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = []
    for i, e in enumerate(raw.get("entries", [])):
        if not e.get("id") or not e.get("questions"):
            raise BankError(f"entry #{i + 1} needs an id and at least one question")
        entries.append(Entry(e["id"], [str(q) for q in e["questions"]], e.get("answer"), e.get("check"),
                             bool(e.get("skip"))))
    return AnswerBank(entries, tokens or {})


def profile_tokens(master: dict) -> dict[str, str]:
    """@tokens from master_resume.yaml contact info. @resume is per job (added by apply.py)."""
    from src.resume import _v
    c = master["contact"]
    return {"@full_name": _v(c["name"]), "@email": _v(c["email"]), "@phone": _v(c["phone"]),
            "@location": _v(c["location"])}


@dataclass
class Match:
    key: str
    label: str
    kind: str
    required: bool
    status: str
    value: object = None            # str, list[str] (multi-select), or file path
    entry_id: str | None = None
    detail: str = ""

    @property
    def fillable(self) -> bool:
        return self.value not in (None, "", [])


def _resolve(value, tokens: dict[str, str]):
    if isinstance(value, str) and value.startswith("@"):
        if value not in tokens:
            raise BankError(f"unknown token {value}")
        return tokens[value]
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return value


def match_question(q, bank: AnswerBank) -> Match:
    """q: a forms.FormQuestion (key, label, kind, required, options)."""
    m = Match(q.key, q.label, q.kind, q.required, NO_ENTRY)
    entry = bank.find(q.label)
    if entry is None:
        m.detail = "no answer-bank entry with this exact question"
        return m
    m.entry_id = entry.id
    if entry.skip:
        m.status = NEEDS_ANSWER if q.required else SKIPPED
        m.detail = "bank says skip, but the question is required" if q.required else "left blank (bank: skip)"
        return m
    if entry.answer is None and not entry.check:
        m.status, m.detail = NEEDS_ANSWER, f"bank entry {entry.id} has no answer yet"
        return m

    if q.kind == "checkbox_group":
        wanted = entry.check or ([entry.answer] if isinstance(entry.answer, str) else entry.answer)
        offered = {normalize(o): o for o in q.options}
        missing = [w for w in wanted if normalize(w) not in offered]
        if missing:
            m.status, m.detail = NO_OPTION, f"not offered: {missing}"
        else:
            m.status, m.value = ANSWERED, [offered[normalize(w)] for w in wanted]
        return m

    answer = _resolve(entry.answer, bank.tokens)
    if answer is None:                 # a token with no value for this job, e.g. @resume without a verified PDF
        m.status, m.detail = NEEDS_ANSWER, f"{entry.answer} has no value for this job"
        return m
    if q.kind in CHOICE_KINDS or q.kind == "checkbox":
        alternatives = answer if isinstance(answer, list) else [answer]
        options = q.options or (["Yes", "No"] if q.kind == "checkbox" else [])
        if not options:
            m.status, m.value = UNVERIFIED, alternatives[0]
            m.detail = "options not readable in advance"
            return m
        offered = {normalize(o): o for o in options}
        hit = next((offered[normalize(a)] for a in alternatives if normalize(str(a)) in offered), None)
        if hit is None:
            m.status, m.detail = NO_OPTION, f"bank answer {alternatives} not among options {options}"
        else:
            m.status, m.value = ANSWERED, hit
        return m
    if q.kind == "autocomplete":       # live lookup (e.g. city): the exact option only appears while typing
        m.status, m.value, m.detail = UNVERIFIED, answer, "live lookup: checked when filling"
        return m
    if isinstance(answer, list):
        m.status, m.detail = NO_OPTION, "bank gives a list, but this is a text question"
        return m
    m.status, m.value = ANSWERED, str(answer)
    return m


def match_form(questions, bank: AnswerBank) -> list[Match]:
    return [match_question(q, bank) for q in questions]
