"""যেকোনো লিংক পড়া — ওয়েব পেজ, JSON / API endpoint, টেক্সট / Markdown / CSV, PDF।

`read_link(url)` একটা লিংক থেকে তিন রকম জিনিস বের করে:
  • docs  — ভাষা-মডেল শেখার ও খোঁজার জন্য লেখা (বড় হলে কয়েক খণ্ডে ভাগ করা)
  • qa    — প্রশ্ন-উত্তর জোড়া (JSON-এর q/a, "প্রশ্ন: … উত্তর: …" লাইন, FAQ পেজ ইত্যাদি)
  • links — পেজের ভেতরের লিংক (মাকড়সার জন্য)
"""
from __future__ import annotations

import csv
import io
import json
import re
from urllib.parse import urlparse

from bs4 import BeautifulSoup

import crawler
from config import LOW_MEM, MAX_DOC_CHARS

MAX_PARTS = 25 if LOW_MEM else 80          # এক লিংক থেকে সর্বোচ্চ কত খণ্ড (প্রতিটি ≤ ১২,০০০ অক্ষর)
MAX_QA_PER_LINK = 5000
MAX_PDF_PAGES = 60 if LOW_MEM else 300
MAX_Q_CHARS, MAX_A_CHARS = 400, 2000

_Q_KEYS = ("q", "question", "ques", "prompt", "query", "instruction", "প্রশ্ন", "প্রশ্নঃ")
_A_KEYS = ("a", "answer", "answers", "ans", "response", "completion", "output", "reply",
           "উত্তর", "উত্তরঃ")
_TITLE_KEYS = ("title", "name", "heading", "headline", "শিরোনাম")
_BN_LETTERS = re.compile(r"[\u0985-\u09B9\u09CE\u09DC-\u09DF]")
_LETTERS = re.compile(r"[A-Za-z\u0985-\u09B9\u09CE\u09DC-\u09DF]")


# --------------------------------------------------------------------------- #
# ছোট সহায়ক
# --------------------------------------------------------------------------- #
def detect_lang(text: str) -> str:
    sample = text[:1500]
    letters = len(_LETTERS.findall(sample)) or 1
    return "bn" if len(_BN_LETTERS.findall(sample)) / letters >= 0.3 else "en"


def host_of(url: str) -> str:
    return (urlparse(url).hostname or "").lower().removeprefix("www.")


def _clean_q(s) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def _add_pair(pairs: list[dict], q, a) -> None:
    q, a = _clean_q(q), crawler.clean_text(str(a))
    if 3 <= len(q) <= MAX_Q_CHARS and 1 <= len(a) and len(pairs) < MAX_QA_PER_LINK:
        pairs.append({"q": q, "a": a[:MAX_A_CHARS]})


def chunk_text(text: str, size: int = MAX_DOC_CHARS, max_parts: int = MAX_PARTS) -> list[str]:
    """লম্বা লেখা অনুচ্ছেদ/বাক্যের সীমানায় ≤ size অক্ষরের খণ্ডে ভাগ করে।"""
    text = crawler.clean_text(text)
    if len(text) <= size:
        return [text] if text else []
    parts: list[str] = []
    cur = ""
    for para in re.split(r"\n{2,}", text):
        while len(para) > size:                       # একটাই অনুচ্ছেদ খুব লম্বা → বাক্যে/লাইনে কাটা
            cut = max(para.rfind("। ", 0, size), para.rfind(". ", 0, size),
                      para.rfind("\n", 0, size), para.rfind(" ", 0, size))
            cut = cut + 1 if cut > size // 3 else size
            head, para = para[:cut].strip(), para[cut:].strip()
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(head)
        if len(cur) + len(para) + 2 > size and cur:
            parts.append(cur)
            cur = ""
        cur = (cur + "\n\n" + para) if cur else para
    if cur:
        parts.append(cur)
    return [p for p in parts if p.strip()][:max_parts]


def _docs(url: str, title: str, text: str, min_chars: int = 120) -> list[dict]:
    text = crawler.clean_text(text)
    if len(text) < min_chars:
        return []
    lang, host = detect_lang(text), host_of(url)
    chunks = chunk_text(text)
    title = re.sub(r"\s+", " ", title or "").strip()[:280] or url
    out = []
    for i, ch in enumerate(chunks):
        out.append({"url": url if i == 0 else f"{url}#part-{i + 1}",
                    "title": title if len(chunks) == 1 else f"{title} ({i + 1}/{len(chunks)})",
                    "content": ch, "lang": lang, "site_domain": host})
    return out


# --------------------------------------------------------------------------- #
# সাধারণ লেখার ভেতরের "প্রশ্ন: … উত্তর: …" ধরা
# --------------------------------------------------------------------------- #
# "প্রশ্ন ১: …", "Question: …", "Q. …"  /  "উত্তর: …", "Answer: …", "A. …"
# (এক অক্ষরের Q/A-র পরে শুধু ":" বা "." — নইলে "a) বিকল্প" ধরনের লাইন উত্তর বলে ভুল হতো)
_Q_LINE = re.compile(r"^\s*(?:[#>*\-\d০-৯.)\s]*)(?:(?:প্রশ্ন|প্রঃ|question|ques)\s*[\d০-৯]*\s*[:：ঃ.)\-–—]"
                     r"|q\s*\d*\s*[:：.])\s*(.+)$", re.I)
_A_LINE = re.compile(r"^\s*(?:[>*\-\s]*)(?:(?:উত্তর|উঃ|answer|ans)\s*[:：ঃ.)\-–—]|a\s*[:：.])\s*(.*)$", re.I)
_QA_INLINE = re.compile(r"(?:প্রশ্ন|question)\s*[\d০-৯]*\s*[:：ঃ]\s*(.+?)\s*(?:উত্তর|answer)\s*[:：ঃ]\s*(.+)$", re.I)


def qa_from_text(text: str) -> list[dict]:
    pairs: list[dict] = []
    q, a_lines = None, None

    def flush() -> None:
        nonlocal q, a_lines
        if q and a_lines:
            _add_pair(pairs, q, " ".join(a_lines))
        q, a_lines = None, None

    for raw in text.split("\n"):
        line = raw.strip().strip("*_ ")
        if not line:
            if a_lines:                       # খালি লাইন = উত্তরের শেষ
                flush()
            continue
        m = _QA_INLINE.search(line)
        if m and (_Q_LINE.match(line) or line.lower().startswith(("প্রশ্ন", "question"))):
            flush()
            _add_pair(pairs, m.group(1), m.group(2))
            continue
        m = _Q_LINE.match(line)
        if m:
            flush()
            q = m.group(1)
            continue
        m = _A_LINE.match(line)
        if m and q is not None and a_lines is None:
            a_lines = [m.group(1)] if m.group(1) else []
            continue
        if a_lines is not None and len(" ".join(a_lines)) < MAX_A_CHARS:
            a_lines.append(line)              # উত্তরের পরের লাইন
        elif q is not None and a_lines is None:
            q = None                          # প্রশ্নের পরে উত্তর আসেনি
    flush()
    return pairs


# --------------------------------------------------------------------------- #
# JSON / API
# --------------------------------------------------------------------------- #
def _as_text(v) -> str | None:
    """উত্তরের মান: স্ট্রিং, সংখ্যা, স্ট্রিংয়ের তালিকা বা {"text": …} (SQuAD ধরন)।"""
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return v.strip() or None
    if isinstance(v, list):
        for x in v:
            t = _as_text(x)
            if t:
                return t
        return None
    if isinstance(v, dict):
        for k in ("text", "value", "content", "answer"):
            if k in v:
                return _as_text(v[k])
    return None


def _walk_json(obj, pairs: list[dict], texts: list[str], depth: int = 0) -> None:
    if depth > 12 or (len(pairs) >= MAX_QA_PER_LINK and len(texts) > 4000):
        return
    if isinstance(obj, list):
        # চ্যাট ধরন: [{"role": "user", "content": …}, {"role": "assistant", "content": …}]
        if obj and all(isinstance(x, dict) and "role" in x and "content" in x for x in obj):
            last_user = None
            for m in obj:
                role, content = str(m.get("role", "")).lower(), _as_text(m.get("content"))
                if role in ("user", "human") and content:
                    last_user = content
                elif role in ("assistant", "bot", "gpt", "model") and content and last_user:
                    _add_pair(pairs, last_user, content)
                    last_user = None
            return
        for x in obj:
            _walk_json(x, pairs, texts, depth + 1)
        return
    if isinstance(obj, str):
        s = obj.strip()
        if len(s) >= 40 and " " in s and not s.startswith(("http://", "https://", "data:")):
            texts.append(s)
        return
    if not isinstance(obj, dict):
        return

    low = {str(k).strip().lower(): k for k in obj}
    qk = next((low[k] for k in _Q_KEYS if k in low), None)
    ak = next((low[k] for k in _A_KEYS if k in low), None)
    used: set = set()
    if qk is not None and ak is not None:
        q, a = _as_text(obj[qk]), _as_text(obj[ak])
        if q and a:
            extra = _as_text(obj.get(low["input"])) if "input" in low else None   # Alpaca: instruction+input
            _add_pair(pairs, f"{q} {extra}" if extra and extra != q else q, a)
            used = {qk, ak, low.get("input")}
    elif len(obj) >= 2 and all(isinstance(k, str) and isinstance(v, str) for k, v in obj.items()) \
            and sum(1 for k in obj if k.strip().endswith(("?", "？"))) >= max(2, len(obj) * 0.6):
        for k, v in obj.items():                          # {"প্রশ্ন?": "উত্তর", …}
            _add_pair(pairs, k, v)
        return

    title = next((_as_text(obj[low[k]]) for k in _TITLE_KEYS if k in low and isinstance(obj[low[k]], str)), None)
    before = len(texts)
    for k, v in obj.items():
        if k in used:
            continue
        _walk_json(v, pairs, texts, depth + 1)
    if title and len(texts) > before and len(title) < 200 and title not in texts[before]:
        texts[before] = f"{title}\n{texts[before]}"


def parse_json(text: str):
    """JSON; না হলে JSON Lines (প্রতি লাইনে একটা অবজেক্ট)। দুটোই না হলে ValueError।"""
    text = text.lstrip("\ufeff").strip()
    try:
        return json.loads(text)
    except ValueError:
        rows = []
        for ln in text.splitlines():
            ln = ln.strip().rstrip(",")
            if not ln:
                continue
            rows.append(json.loads(ln))          # কোনো লাইন JSON না হলে ValueError
        if not rows:
            raise
        return rows


def read_json(text: str, url: str) -> tuple[list[dict], list[dict]]:
    data = parse_json(text)
    pairs: list[dict] = []
    texts: list[str] = []
    _walk_json(data, pairs, texts)
    title = ""
    if isinstance(data, dict):
        title = next((str(data[k]) for k in data if str(k).lower() in _TITLE_KEYS
                      and isinstance(data[k], str)), "")
    body = "\n\n".join(dict.fromkeys(texts))
    if pairs:                                   # প্রশ্নোত্তরও লেখা হিসেবে — ভাষা-মডেল শেখে, খোঁজাতেও আসে
        qa_text = "\n\n".join(f"প্রশ্ন: {p['q']}\nউত্তর: {p['a']}" for p in pairs)
        body = (body + "\n\n" + qa_text) if body else qa_text
    name = title or urlparse(url).path.rsplit("/", 1)[-1] or host_of(url)
    return _docs(url, name, body, min_chars=40), pairs


# --------------------------------------------------------------------------- #
# টেক্সট / Markdown / CSV
# --------------------------------------------------------------------------- #
def _strip_markdown(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)               # ছবি
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)            # [লেখা](লিংক) → লেখা
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.M)       # শিরোনামের #
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.M)            # তালিকার চিহ্ন
    text = re.sub(r"[*_`]{1,3}", "", text)
    return re.sub(r"^\s*\|?[-: |]{3,}\|?\s*$", "", text, flags=re.M)


def _read_csv(text: str) -> list[dict]:
    """প্রথম সারিতে question/answer (বা q/a, প্রশ্ন/উত্তর) কলাম থাকলে প্রতিটি সারি একটা জোড়া।"""
    first = text.lstrip("\ufeff").split("\n", 1)[0]
    delim = "\t" if first.count("\t") > first.count(",") else ","
    head = [h.strip().strip('"').lower() for h in first.split(delim)]
    qi = next((i for i, h in enumerate(head) if h in _Q_KEYS), None)
    ai = next((i for i, h in enumerate(head) if h in _A_KEYS), None)
    if qi is None or ai is None:
        return []
    pairs: list[dict] = []
    try:
        for n, r in enumerate(csv.reader(io.StringIO(text.lstrip("\ufeff")), delimiter=delim)):
            if n and len(r) > max(qi, ai):
                _add_pair(pairs, r[qi], r[ai])
    except csv.Error:
        pass
    return pairs


def read_text(text: str, url: str) -> tuple[list[dict], list[dict]]:
    path = urlparse(url).path.lower()
    pairs = _read_csv(text)
    if path.endswith((".md", ".markdown", ".mdx", ".rst")) or re.search(r"^#{1,6} \S", text, re.M):
        text = _strip_markdown(text)
    if not pairs:
        pairs = qa_from_text(text)
    first = next((ln.strip() for ln in text.splitlines() if len(ln.strip()) > 3), "")
    title = first[:120] if len(first) <= 120 else (path.rsplit("/", 1)[-1] or host_of(url))
    return _docs(url, title, text), pairs


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #
def read_pdf(raw: bytes, url: str) -> tuple[list[dict], list[dict], str | None]:
    try:
        from pypdf import PdfReader
    except ImportError:
        return [], [], "PDF পড়ার লাইব্রেরি (pypdf) ইনস্টল নেই"
    try:
        reader = PdfReader(io.BytesIO(raw))
        if reader.is_encrypted:
            try:
                if not reader.decrypt(""):
                    return [], [], "PDF পাসওয়ার্ড দিয়ে বন্ধ"
            except Exception:  # noqa: BLE001
                return [], [], "PDF পাসওয়ার্ড দিয়ে বন্ধ"
        pages: list[str] = []
        total = 0
        for i, page in enumerate(reader.pages):
            if i >= MAX_PDF_PAGES or total > MAX_PARTS * MAX_DOC_CHARS:
                break
            try:
                t = page.extract_text() or ""
            except Exception:  # noqa: BLE001
                t = ""
            total += len(t)
            pages.append(t)
        title = ""
        try:
            title = str((reader.metadata or {}).get("/Title") or "").strip()
        except Exception:  # noqa: BLE001
            pass
    except Exception as e:  # noqa: BLE001
        return [], [], f"PDF খোলা যায়নি: {type(e).__name__}"
    text = "\n\n".join(pages)
    text = re.sub(r"-\n(?=[a-z])", "", text)                 # লাইনের শেষে ভাঙা ইংরেজি শব্দ জোড়া
    text = re.sub(r"(?<![.!?।:\n])\n(?!\n|\s*(?:প্রশ্ন|উত্তর|question|answer|[QA]\s*[:.])\b)", " ", text,
                  flags=re.I)                                  # বাক্যের মাঝের লাইন-ভাঙা জোড়া
    text = crawler.clean_text(text)
    readable = len(re.findall(r"[A-Za-z\u0980-\u09FF]", text))
    if len(text) < 120 or readable < 0.5 * len(re.sub(r"\s", "", text)):
        return [], [], "এই PDF থেকে লেখা বের হয়নি — সম্ভবত স্ক্যান করা ছবি বা পুরোনো (নন-ইউনিকোড) ফন্ট"
    if not title:
        title = text.split("\n", 1)[0][:120] or urlparse(url).path.rsplit("/", 1)[-1]
    return _docs(url, title, text), qa_from_text(text), None


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
def _html_qa(html: str) -> list[dict]:
    """FAQ: JSON-LD FAQPage, <details><summary>, <dt>/<dd>।"""
    pairs: list[dict] = []
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(tag.string or "")
        except (ValueError, TypeError):
            continue
        stack = [data]
        while stack:
            cur = stack.pop()
            if isinstance(cur, list):
                stack.extend(cur)
            elif isinstance(cur, dict):
                if str(cur.get("@type", "")).lower() == "question" and cur.get("name"):
                    ans = cur.get("acceptedAnswer") or cur.get("suggestedAnswer")
                    ans = ans[0] if isinstance(ans, list) and ans else ans
                    text = ans.get("text") if isinstance(ans, dict) else None
                    if isinstance(text, str):
                        _add_pair(pairs, cur["name"], BeautifulSoup(text, "html.parser").get_text(" ", strip=True))
                stack.extend(v for v in cur.values() if isinstance(v, (dict, list)))
    for d in soup.find_all("details"):
        sm = d.find("summary")
        if sm:
            q = sm.get_text(" ", strip=True)
            sm.extract()
            _add_pair(pairs, q, d.get_text(" ", strip=True))
    for dt in soup.find_all("dt"):
        dd = dt.find_next_sibling("dd")
        q = dt.get_text(" ", strip=True)
        if dd and q.endswith(("?", "？")):
            _add_pair(pairs, q, dd.get_text(" ", strip=True))
    return pairs


def read_html(html: str, url: str, want_links: bool = False) -> tuple[list[dict], list[dict], list[str]]:
    art = crawler.extract_article(html, url) or crawler.extract_loose(html, url)
    docs = _docs(url, art["title"], art["content"]) if art else []
    pairs = _html_qa(html)
    if art:
        pairs += qa_from_text(art["content"])
    links = crawler.extract_links(html, url) if want_links else []
    return docs, pairs, links


# --------------------------------------------------------------------------- #
def read_link(url: str, want_links: bool = False) -> dict:
    """একটা লিংক পড়ে {"ok", "kind", "url", "docs", "qa", "links", "error"} ফেরায়।"""
    out = {"ok": False, "kind": "", "url": url, "docs": [], "qa": [], "links": [], "error": None}
    got = crawler.fetch_any(url)
    if "error" in got:
        out["error"] = got["error"]
        return out
    kind, raw, final = got["kind"], got["raw"], got["url"]
    out.update({"kind": kind, "url": final})
    try:
        if kind == "pdf":
            out["docs"], out["qa"], out["error"] = read_pdf(raw, final)
        else:
            text = crawler._decode(raw, got["ctype"])
            if kind == "json":
                try:
                    out["docs"], out["qa"] = read_json(text, final)
                except ValueError:
                    out["kind"] = "text"          # JSON-এর মতো দেখতে হলেও নয় → সাধারণ লেখা
                    out["docs"], out["qa"] = read_text(text, final)
            elif kind == "html":
                out["docs"], out["qa"], out["links"] = read_html(text, final, want_links)
            elif kind == "xml":
                items = crawler.parse_feed(raw, limit=200)       # RSS/Atom → সারাংশগুলো লেখা
                if want_links:
                    out["links"] = [it["url"] for it in items]
                body = "\n\n".join(f"{it['title']}\n{it['summary']}" for it in items if it.get("summary"))
                out["docs"] = _docs(final, host_of(final) + " feed", body)
                if not items:
                    out["links"] = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", text)[:2000] if want_links else []
            else:
                out["docs"], out["qa"] = read_text(text, final)
    except Exception as e:  # noqa: BLE001
        out["error"] = f"পড়তে সমস্যা: {type(e).__name__}: {e}"
    seen: set[str] = set()                       # একই প্রশ্ন একবারই
    out["qa"] = [p for p in out["qa"] if not (p["q"] in seen or seen.add(p["q"]))]
    out["ok"] = bool(out["docs"] or out["qa"] or out["links"])
    if not out["ok"] and not out["error"]:
        out["error"] = "এই লিংকে পড়ার মতো লেখা পাওয়া যায়নি"
    return out
