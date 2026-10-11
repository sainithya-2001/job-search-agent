"""Cross-check employers against USCIS H-1B Employer Data Hub exports.

USCIS blocks scripted downloads, so files are downloaded by hand from
https://www.uscis.gov/tools/reports-and-studies/h-1b-employer-data-hub into data/uscis/ (any number of fiscal years).

Exports have varied over the years (comma or tab separated, UTF-8 or UTF-16, different column names), so
columns are found by name: the employer column contains "employer", approval columns contain "approval",
and the year column contains "fiscal year".

Matching is exact on a normalized name (lowercase, punctuation and legal suffixes removed). Prefix/fuzzy matching
is deliberately avoided: "Hex" would match "Hexaware Technologies". Add `uscis_names` to a company in config.yaml
when its legal name differs (e.g. Scale AI -> "Scale AI Inc" normalizes fine, but a rename would not).
"""
import csv
import io
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

_SUFFIXES = {"inc", "incorporated", "llc", "l", "c", "corp", "corporation", "co", "company", "ltd", "limited",
             "pbc", "lp", "llp", "plc", "pc", "na"}


def normalize_employer(name: str) -> str:
    t = name.lower().replace("&", " and ")
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    tokens = t.split()
    while tokens and tokens[-1] in _SUFFIXES:  # strip trailing legal suffixes: "Anthropic, PBC" -> "anthropic"
        tokens.pop()
    return " ".join(tokens)


@dataclass
class EmployerRecord:
    approvals: int = 0
    years: set[str] = field(default_factory=set)
    names: set[str] = field(default_factory=set)


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    return raw.decode("utf-8-sig", errors="replace")


def _to_int(s: str) -> int:
    s = (s or "").replace(",", "").strip()
    return int(float(s)) if re.fullmatch(r"-?\d+(\.\d+)?", s) else 0


class SponsorshipDB:
    def __init__(self, folder: str | Path = "data/uscis"):
        self.folder = Path(folder)
        self.records: dict[str, EmployerRecord] = defaultdict(EmployerRecord)
        self.files: list[str] = []
        for path in sorted(self.folder.glob("*.csv")):
            self._load(path)

    @property
    def loaded(self) -> bool:
        return bool(self.files)

    def _load(self, path: Path) -> None:
        text = _read_text(path)
        delimiter = "\t" if text.split("\n", 1)[0].count("\t") > text.split("\n", 1)[0].count(",") else ","
        reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
        headers = reader.fieldnames or []
        emp_col = next((h for h in headers if "employer" in h.lower()), None)
        approval_cols = [h for h in headers if "approval" in h.lower()]
        year_col = next((h for h in headers if "fiscal year" in h.lower()), None)
        if not emp_col or not approval_cols:
            raise ValueError(f"{path.name}: can't find employer/approval columns in {headers[:12]}")
        for row in reader:
            name = (row.get(emp_col) or "").strip()
            if not name:
                continue
            rec = self.records[normalize_employer(name)]
            rec.approvals += sum(_to_int(row.get(c)) for c in approval_cols)
            rec.names.add(name)
            if year_col and row.get(year_col):
                rec.years.add(row[year_col].strip())
        self.files.append(path.name)

    def lookup(self, company: str, aliases: list[str] | None = None, e_verify: str | None = None) -> dict:
        """-> {"status": "no_data" | "not_found" | "sponsor" | "zero_approvals", "e_verify": ..., ...}

        e_verify (yes / no / unknown) is not in the USCIS export; it comes from the company's `e_verify` field in
        config.yaml, filled in by hand (e.g. from the E-Verify employer search). It matters for STEM OPT.
        """
        ev = {"e_verify": e_verify or "unknown"}
        if not self.loaded:
            return {"status": "no_data", **ev}
        for name in [company, *(aliases or [])]:
            rec = self.records.get(normalize_employer(name))
            if rec:
                return {"status": "sponsor" if rec.approvals > 0 else "zero_approvals", **ev,
                        "approvals": rec.approvals, "years": sorted(rec.years), "matched": sorted(rec.names)[:3]}
        return {"status": "not_found", **ev}
