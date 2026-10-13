"""Label evals/label_me.csv in the terminal, one job at a time.

    python evals/label.py

Shows company, title and the 2-line summary (never the model's score). Keys (then Enter):
    a = apply   m = maybe   s = skip   o = open link in browser   b = back   q = quit
After a label you can type a short note (Enter for none). The CSV is saved after every answer, so quitting or
closing the window loses nothing; re-running resumes at the first unlabeled job.
"""
from __future__ import annotations

import csv
import os
import sys
import time
import webbrowser
from pathlib import Path

PATH = Path(__file__).resolve().parent / "label_me.csv"
LABEL, NOTES = "label (apply/maybe/skip)", "notes"
KEYS = {"a": "apply", "m": "maybe", "s": "skip"}


def load(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        return list(r.fieldnames), list(r)


def save(path: Path, header: list[str], rows: list[dict]) -> None:
    """Write to a temp file then replace, so a crash never leaves a half-written CSV. UTF-8 BOM for Excel."""
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    for attempt in range(10):           # Windows: antivirus/indexer can hold the file for a moment after a write
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.1)


def done(rows: list[dict]) -> int:
    return sum(r[LABEL].strip().lower() in KEYS.values() for r in rows)


def first_unlabeled(rows: list[dict]) -> int:
    return next((i for i, r in enumerate(rows) if r[LABEL].strip().lower() not in KEYS.values()), len(rows))


def show(i: int, rows: list[dict], out) -> None:
    r = rows[i]
    out(f"\n{'-' * 72}\n[{i + 1}/{len(rows)}]  ({done(rows)} labeled)\n")
    out(f"{r['company']}\n{r['title']}\n")
    for line in r["summary"].splitlines():
        out(f"  {line}")
    if r[LABEL].strip():
        out(f"\n  current label: {r[LABEL]}" + (f"   note: {r[NOTES]}" if r[NOTES].strip() else ""))
    out("")


def run(path: Path = PATH, ask=input, out=print, open_url=webbrowser.open) -> None:
    header, rows = load(path)
    i = first_unlabeled(rows)
    if i == len(rows):
        out(f"All {len(rows)} jobs are labeled. Starting from job 1 so you can change any (q to quit).")
        i = 0
    while i < len(rows):
        show(i, rows, out)
        key = ask("a=apply  m=maybe  s=skip  o=open link  b=back  q=quit > ").strip().lower()
        if key == "q":
            break
        if key == "o":
            open_url(rows[i]["link"])
            out(f"  opened {rows[i]['link']}")
            continue
        if key == "b":
            if i == 0:
                out("  already at the first job")
            i = max(i - 1, 0)
            continue
        if key not in KEYS:
            out("  type a, m, s, o, b or q")
            continue
        rows[i][LABEL] = KEYS[key]
        rows[i][NOTES] = ask("note (Enter for none) > ").strip() or rows[i][NOTES]
        while True:
            try:
                save(path, header, rows)
                break
            except PermissionError:          # Excel locks a CSV it has open
                if ask("  can't save: is label_me.csv open in Excel? Close it, then press Enter (q to quit) > ")\
                        .strip().lower() == "q":
                    out("  NOT saved. Labels from this session are lost for this job.")
                    return
        out(f"  saved: {rows[i][LABEL]}  ({done(rows)}/{len(rows)})")
        i += 1
    n = done(rows)
    out(f"\n{n}/{len(rows)} labeled, saved to {path.name}."
        + (" Next: python evals/run_evals.py" if n == len(rows) else " Run again to continue."))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        run()
    except (KeyboardInterrupt, EOFError):
        print("\nStopped. Everything answered so far is saved.")
