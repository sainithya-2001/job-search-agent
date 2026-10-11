"""Read, fill, and snapshot application forms on Greenhouse, Lever, and Ashby with Playwright.

Never submits. Two independent safeguards:
  1. Nothing here clicks a submit control: every click goes through safe_click(), which refuses buttons whose text
     looks like submit/apply/send (Ashby's Yes/No answer buttons are type=submit, so the type alone can't tell).
  2. While the agent is filling, a capture-phase listener cancels every form submit event and every click on a
     submit-like button. It is lifted only when the form is handed to Nithya.

Question kinds: text email tel number url textarea select combobox autocomplete radio yesno checkbox
checkbox_group file. Each question gets a stable `key` and a CSS `selector` for its main element or container.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from urllib.parse import urlparse

CAPTCHA_MARKERS = ("recaptcha", "hcaptcha", "turnstile", "challenges.cloudflare.com", "arkoselabs", "funcaptcha",
                   "friendlycaptcha", "geetest")
SUBMIT_WORDS = re.compile(r"\b(submit|apply|send|finish|complete application)\b", re.I)


@dataclass
class FormQuestion:
    key: str
    label: str
    kind: str
    required: bool
    options: list[str] = field(default_factory=list)
    selector: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def ats_of(url: str) -> str | None:
    host = urlparse(url).netloc
    if "greenhouse.io" in host:
        return "greenhouse"
    if "lever.co" in host:
        return "lever"
    if "ashbyhq.com" in host:
        return "ashby"
    return None


def form_url(job_url: str) -> str:
    """The page that holds the application form itself."""
    ats = ats_of(job_url)
    u = job_url.rstrip("/")
    if ats == "lever" and not u.endswith("/apply"):
        return u + "/apply"
    if ats == "ashby" and not u.endswith("/application"):
        return u + "/application"
    return u


# ---- reading ----------------------------------------------------------------------------------------------------
# Shared helpers injected into every extractor.
_JS_HELPERS = r"""
const SKIP_TYPES = new Set(['hidden', 'submit', 'button', 'image', 'reset']);
const clean = t => (t || '').replace(/[✱*]/g, ' ').replace(/\s+/g, ' ').trim();
const skip = el => SKIP_TYPES.has(el.type) || /captcha/i.test((el.name || '') + ' ' + (el.id || ''))
    || (el.id || '').startsWith('iti-') || el.getAttribute('aria-hidden') === 'true'
    || el.closest('[class*="autofill"]');
const kindOf = el => {
    if (el.tagName === 'TEXTAREA') return 'textarea';
    if (el.tagName === 'SELECT') return 'select';
    if (el.type === 'file') return 'file';
    if (el.type === 'checkbox') return 'checkbox';
    if (el.type === 'radio') return 'radio';
    if (['email', 'tel', 'number', 'url'].includes(el.type)) return el.type;
    return 'text';
};
const sel = el => el.id ? `[id="${el.id}"]` : (el.name ? `[name="${el.name}"]` : null);
const optionText = el => {
    const l = (el.id && document.querySelector(`label[for="${CSS.escape(el.id)}"]`)) || el.closest('label');
    return clean(l ? l.innerText : el.value);
};
"""

_JS_GREENHOUSE = _JS_HELPERS + r"""
const form = document.querySelector('form#application-form') || document.querySelector('form');
const out = [], groups = new Set();
const labelOf = el => {
    const ids = (el.getAttribute('aria-labelledby') || '').split(/\s+/).filter(Boolean);
    let t = ids.map(i => document.getElementById(i)).filter(Boolean).map(x => x.innerText).join(' ');
    if (!t && el.id) { const l = document.querySelector(`label[for="${CSS.escape(el.id)}"]`); if (l) t = l.innerText; }
    return t;
};
for (const el of form.querySelectorAll('input, textarea, select')) {
    if (skip(el)) continue;
    let kind = kindOf(el), label = labelOf(el), key = el.id || el.name, options = [], s = sel(el);
    let required = el.required || el.getAttribute('aria-required') === 'true';
    if (kind === 'file') {
        const g = el.closest('[role=group][aria-labelledby]');
        if (g) { label = labelOf(g); required = g.getAttribute('aria-required') === 'true'; }
    } else if (el.getAttribute('role') === 'combobox') {
        kind = el.id === 'candidate-location' ? 'autocomplete' : 'combobox';
    } else if (kind === 'checkbox' || kind === 'radio') {
        const fs = el.closest('fieldset');
        const name = el.name;
        const peers = name ? form.querySelectorAll(`input[name="${CSS.escape(name)}"]`) : [el];
        if (fs && peers.length > 1) {
            if (groups.has(name)) continue;
            groups.add(name);
            kind = kind === 'radio' ? 'radio' : 'checkbox_group';
            label = (fs.querySelector('legend') || {}).innerText || label;
            options = Array.from(peers).map(optionText);
            key = name; s = `[name="${name}"]`;
            required = required || !!fs.querySelector('[aria-required=true], [required]');
        }
    }
    if (!s) continue;
    out.push({key, label: clean(label) || key, kind, required: !!required || /\*\s*$/.test(label), options, selector: s});
}
return out;
"""

_JS_LEVER = _JS_HELPERS + r"""
const form = document.querySelector('form');
const out = [], used = new Set();
for (const q of form.querySelectorAll('.application-question')) {
    const inputs = Array.from(q.querySelectorAll('input, textarea, select')).filter(e => !skip(e));
    if (!inputs.length) continue;
    const lab = q.querySelector('.application-label');
    let label = clean(lab ? (lab.querySelector('.text') || lab).innerText : '');
    // Generic card prompts ("Please check all that apply:") only make sense with the card's title, so the title
    // becomes part of the text that must match exactly.
    const card = q.closest('[data-qa="additional-cards"]');
    const cardName = card && card.querySelector('[data-qa="card-name"]');
    if (cardName && /^(please\s+)?(check|select|choose)\b/i.test(label)) label = clean(cardName.innerText) + ' ' + label;
    const required = inputs.some(e => e.required) || !!q.querySelector('.required');
    const first = inputs[0];
    let kind = kindOf(first), options = [], key = first.name || first.id, s = sel(first);
    if (first.id === 'location-input') kind = 'autocomplete';
    if (kind === 'select') options = Array.from(first.options).filter(o => o.value).map(o => clean(o.text));
    if (kind === 'checkbox' && inputs.length > 1) { kind = 'checkbox_group'; options = inputs.map(optionText); }
    if (kind === 'radio') options = inputs.filter(e => e.type === 'radio').map(optionText);
    if (kind === 'checkbox_group' || kind === 'radio') s = `[name="${first.name}"]`;
    inputs.forEach(e => used.add(e));
    if (s) out.push({key, label: label || key, kind, required, options, selector: s});
}
// Anything not inside a recognised question is still reported, so no field is silently ignored.
for (const el of form.querySelectorAll('input, textarea, select')) {
    if (skip(el) || used.has(el) || el.closest('.application-question')) continue;
    const s = sel(el); if (!s) continue;
    out.push({key: el.name || el.id, label: clean(el.getAttribute('aria-label') || el.placeholder || el.name),
              kind: kindOf(el), required: el.required, options: [], selector: s});
}
return out;
"""

_JS_ASHBY = _JS_HELPERS + r"""
const out = [];
for (const q of document.querySelectorAll('.ashby-application-form-field-entry')) {
    const path = q.getAttribute('data-field-path');
    const title = q.querySelector('.ashby-application-form-question-title');
    const label = clean(title ? title.innerText : path);
    const inputs = Array.from(q.querySelectorAll('input, textarea, select')).filter(e => !skip(e));
    let required = inputs.some(e => e.required) || !!(title && /required/i.test(title.className));
    const s = `[data-field-path="${path}"]`;
    let kind, options = [];
    const yesno = q.querySelector('.ashby-application-form-input-yesno');
    if (yesno) { kind = 'yesno'; options = Array.from(yesno.querySelectorAll('button')).map(b => clean(b.innerText)); }
    else if (q.querySelector('input[type=radio]')) {
        kind = 'radio'; options = Array.from(q.querySelectorAll('input[type=radio]')).map(optionText);
    } else if (q.querySelectorAll('input[type=checkbox]').length > 1) {
        kind = 'checkbox_group'; options = Array.from(q.querySelectorAll('input[type=checkbox]')).map(optionText);
    } else if (q.querySelector('[role=combobox]')) {
        kind = path === '_systemfield_location' ? 'autocomplete' : 'combobox';
    } else if (inputs.length) { kind = kindOf(inputs[0]); if (kind === 'select') options = Array.from(inputs[0].options).filter(o => o.value).map(o => clean(o.text)); }
    else continue;
    out.push({key: path, label, kind, required, options, selector: s});
}
return out;
"""

_EXTRACTORS = {"greenhouse": _JS_GREENHOUSE, "lever": _JS_LEVER, "ashby": _JS_ASHBY}


def read_questions(page, ats: str, load_options: bool = True) -> list[FormQuestion]:
    raw = page.evaluate("() => {" + _EXTRACTORS[ats] + "}")
    qs = [FormQuestion(**r) for r in raw]
    if load_options:
        for q in qs:
            if q.kind == "combobox" and not q.options:
                q.options = combobox_options(page, q)
    return qs


def combobox_options(page, q: FormQuestion) -> list[str]:
    """Open a searchable dropdown just long enough to read its options, then close it (nothing is selected)."""
    try:
        target = _combobox_input(page, q)
        safe_click(target)
        page.wait_for_timeout(250)
        opts = page.evaluate(r"""(key) => {
            let os = document.querySelectorAll(`[id^="react-select-${key}-option-"]`);
            if (!os.length) os = Array.from(document.querySelectorAll('[role=listbox] [role=option]'))
                                      .filter(o => !(o.id || '').startsWith('iti-'));
            return Array.from(os).map(o => o.innerText.trim());
        }""", q.key)
        page.keyboard.press("Escape")
        return opts
    except Exception:          # noqa: BLE001 — unreadable options just mean the answer can't be verified
        return []


def _combobox_input(page, q: FormQuestion):
    loc = page.locator(q.selector).first
    if loc.evaluate("e => e.matches('input')"):
        return loc
    return loc.locator("[role=combobox]").first


def detect_captcha(page, requested_urls: list[str]) -> list[str]:
    """Every CAPTCHA marker seen in the page HTML, its frames, or its network requests — visible or invisible."""
    html = page.content().lower()
    frames = " ".join(f.url.lower() for f in page.frames)
    reqs = " ".join(u.lower() for u in requested_urls)
    return sorted({m for m in CAPTCHA_MARKERS if m in html or m in frames or m in reqs})


# ---- safety -----------------------------------------------------------------------------------------------------
class SubmitRefused(RuntimeError):
    pass


def safe_click(locator) -> None:
    """Click, unless the element is (or is inside) a submit-like control. The agent never submits."""
    info = locator.evaluate("""e => { const b = e.closest('button, [role=button], input[type=submit]') || e;
                                      return (b.innerText || b.value || '') + ' ' + (b.getAttribute('aria-label') || ''); }""")
    if SUBMIT_WORDS.search(info or ""):
        raise SubmitRefused(f"refusing to click a submit-like control: {info.strip()!r}")
    locator.click()


_BLOCK_JS = r"""() => {
  if (window.__jaInstalled) { window.__jaBlock = true; return; }
  window.__jaInstalled = true; window.__jaBlock = true; window.__jaSubmitClicks = 0;
  const words = /\b(submit|apply|send|finish|complete application)\b/i;
  const submitLike = t => { const b = t && t.closest && t.closest('button, [role=button], input[type=submit]');
                            return b && words.test((b.innerText || b.value || '') + ' ' + (b.getAttribute('aria-label') || '')); };
  window.addEventListener('submit', e => { if (window.__jaBlock) { e.preventDefault(); e.stopImmediatePropagation(); } }, true);
  window.addEventListener('click', e => {
    if (!submitLike(e.target)) return;
    if (window.__jaBlock) { e.preventDefault(); e.stopImmediatePropagation(); return; }
    window.__jaSubmitClicks += 1;
    if (window.__jaOnSubmit) window.__jaOnSubmit();       // reported to Python before the page navigates away
  }, true);
}"""


def block_submits(page) -> None:
    page.evaluate(_BLOCK_JS)


def hand_over(page) -> None:
    """Lift the submit block: from here on only Nithya acts on the form."""
    page.evaluate("() => { window.__jaBlock = false; }")


# ---- filling ----------------------------------------------------------------------------------------------------
def fill(page, q: FormQuestion, value) -> tuple[bool, str]:
    """Fill one question. Returns (filled, note). Never raises for a field that can't be filled — that field is
    simply left for Nithya and highlighted."""
    try:
        loc = page.locator(q.selector).first
        k = q.kind
        if k in ("text", "email", "tel", "number", "url", "textarea"):
            target = loc if loc.evaluate("e => e.matches('input, textarea')") else loc.locator("input, textarea").first
            target.fill(str(value))
        elif k == "file":
            target = loc if loc.evaluate("e => e.matches('input')") else loc.locator("input[type=file]").first
            target.set_input_files(str(value))
        elif k == "select":
            target = loc if loc.evaluate("e => e.matches('select')") else loc.locator("select").first
            target.select_option(label=str(value))
        elif k in ("combobox", "autocomplete"):
            return _fill_combobox(page, q, str(value))
        elif k == "yesno":
            safe_click(loc.locator("button", has_text=re.compile(rf"^\s*{re.escape(str(value))}\s*$")).first)
        elif k in ("radio", "checkbox_group"):
            wanted = value if isinstance(value, list) else [value]
            boxes = page.locator(q.selector) if q.selector.startswith("[name=") else loc.locator(
                "input[type=radio], input[type=checkbox]")
            hits = 0
            for i in range(boxes.count()):
                box = boxes.nth(i)
                text = box.evaluate(r"""e => { const l = (e.id && document.querySelector(`label[for="${CSS.escape(e.id)}"]`))
                                                || e.closest('label'); return (l ? l.innerText : e.value).trim(); }""")
                if any(_same(text, w) for w in wanted):
                    box.check(force=True); hits += 1
            if hits != len(wanted):
                return False, f"matched {hits} of {len(wanted)} options"
        elif k == "checkbox":
            target = loc if loc.evaluate("e => e.matches('input')") else loc.locator("input[type=checkbox]").first
            target.set_checked(str(value).lower() in ("yes", "true"), force=True)
        else:
            return False, f"unsupported kind {k}"
        return True, ""
    except SubmitRefused:
        raise
    except Exception as e:     # noqa: BLE001
        return False, f"{type(e).__name__}: {str(e).splitlines()[0][:120]}"


def _same(a: str, b: str) -> bool:
    from src.answers import normalize
    return normalize(a) == normalize(b)


def _fill_combobox(page, q: FormQuestion, value: str) -> tuple[bool, str]:
    """Type into a searchable dropdown and pick the option whose text equals the answer exactly."""
    target = _combobox_input(page, q)
    safe_click(target)
    target.fill("")
    typed = value.split(",")[0] if q.kind == "autocomplete" else value
    target.type(typed, delay=30)
    page.wait_for_timeout(1500 if q.kind == "autocomplete" else 300)
    options = page.locator(f'[id^="react-select-{q.key}-option-"], [role=listbox] [role=option]')
    for i in range(options.count()):
        opt = options.nth(i)
        if (opt.get_attribute("id") or "").startswith("iti-"):
            continue
        if _same(opt.inner_text(), value):
            safe_click(opt)
            return True, ""
    page.keyboard.press("Escape")
    target.fill("")
    return False, f"no option exactly equal to {value!r}"


# ---- snapshot + highlight ---------------------------------------------------------------------------------------
_SNAPSHOT_JS = r"""(qs) => {
  const out = {};
  const textOf = e => e ? (e.innerText || '').trim() : '';
  for (const q of qs) {
    const nodes = document.querySelectorAll(q.selector);
    const el = nodes[0];
    if (!el) { out[q.key] = null; continue; }
    let v = null;
    const find = s => el.matches(s) ? el : el.querySelector(s);
    if (q.kind === 'file') { const f = find('input[type=file]'); v = f && f.files.length ? f.files[0].name : (el.closest('.field-wrapper, .ashby-application-form-field-entry, .application-question') || el).innerText.match(/\S+\.(pdf|docx?)/i)?.[0] || null; }
    else if (q.kind === 'yesno') { const b = el.querySelector('button[aria-pressed=true]'); v = b ? textOf(b) : null; }
    else if (q.kind === 'combobox' || q.kind === 'autocomplete') {
      const sv = el.closest('.select-shell, .select__container, [data-field-path]')?.querySelector('[class*="single-value"], [class*="singleValue"]');
      const input = find('input');
      v = sv ? textOf(sv) : (input ? input.value : null) || null;
    }
    else if (q.kind === 'radio' || q.kind === 'checkbox_group') {
      const boxes = q.selector.startsWith('[name=') ? Array.from(nodes) : Array.from(el.querySelectorAll('input[type=radio], input[type=checkbox]'));
      const on = boxes.filter(b => b.checked).map(b => { const l = (b.id && document.querySelector(`label[for="${CSS.escape(b.id)}"]`)) || b.closest('label'); return (l ? l.innerText : b.value).trim(); });
      v = on.length ? on.sort().join(' | ') : null;
    }
    else if (q.kind === 'checkbox') { const c = find('input[type=checkbox]'); v = c ? (c.checked ? 'Yes' : 'No') : null; }
    else if (q.kind === 'select') { const s = find('select'); v = s && s.value ? s.options[s.selectedIndex].text.trim() : null; }
    else { const i = find('input, textarea'); v = i && i.value ? i.value : null; }
    out[q.key] = v;
  }
  return out;
}"""


def snapshot(page, questions: list[FormQuestion]) -> dict:
    return page.evaluate(_SNAPSHOT_JS, [q.to_dict() for q in questions])


def highlight(page, questions: list[FormQuestion], keys: set[str], banner: str) -> None:
    """Outline the questions Nithya must handle and pin a short instruction banner to the top of the page."""
    page.evaluate(r"""([qs, keys, banner]) => {
      for (const q of qs) {
        if (!keys.includes(q.key)) continue;
        const el = document.querySelector(q.selector);
        const box = el && (el.closest('.field-wrapper, .application-question, .ashby-application-form-field-entry, .select__container') || el);
        if (box) { box.style.outline = '3px solid #d93025'; box.style.outlineOffset = '4px'; }
      }
      const b = document.createElement('div');
      b.textContent = banner;
      b.style.cssText = 'position:fixed;top:0;left:0;right:0;z-index:2147483647;background:#202124;color:#fff;'
                      + 'font:14px/1.4 system-ui,sans-serif;padding:10px 16px;text-align:center';
      document.body.appendChild(b);
    }""", [[q.to_dict() for q in questions], sorted(keys), banner])
