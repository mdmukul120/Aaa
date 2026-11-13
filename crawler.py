"""ইন্টারনেট থেকে পেজ আনা + মূল লেখা আলাদা করা (crawler + extractor)."""
from __future__ import annotations

import ipaddress
import json
import re
import socket
import time
import xml.etree.ElementTree as ET
from urllib.parse import quote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from config import MAX_DOC_CHARS, USER_AGENT

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "bn,en;q=0.9",
}

BOILER = (
    "cookie", "subscribe to our newsletter", "sign up for", "all rights reserved",
    "সাবস্ক্রাইব", "আমাদের অ্যাপ", "বিজ্ঞাপন", "সম্পাদকীয়", "যোগাযোগ করুন",
    "follow us on", "download the app",
)


# --------------------------------------------------------------------------- #
def clean_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n\n", s)
    return s.strip()


def _jsonld_body(soup: BeautifulSoup) -> str | None:
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(tag.string or "")
        except (ValueError, TypeError):
            continue
        stack = [data]
        while stack:
            cur = stack.pop()
            if isinstance(cur, dict):
                body = cur.get("articleBody")
                if isinstance(body, str) and len(body) > 500:
                    return clean_text(body)
                stack.extend(cur.values())
            elif isinstance(cur, list):
                stack.extend(cur)
    return None


def extract_article(html: str, url: str) -> dict | None:
    """HTML থেকে title + মূল লেখা বের করে। ব্যর্থ হলে None।"""
    soup = BeautifulSoup(html, "html.parser")

    title = ""
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        title = og["content"].strip()
    elif soup.title:
        title = soup.title.get_text(" ", strip=True)
    title = re.sub(r"\s+", " ", title)[:300]

    for tag in soup(["script", "style", "noscript", "nav", "footer", "header",
                     "aside", "form", "svg", "iframe", "button"]):
        tag.decompose()

    body_text = _jsonld_body(soup)

    if not body_text:
        root = soup.find("article") or soup.find("main") or soup.body or soup
        paras: list[str] = []
        for p in root.find_all(["p", "h2", "h3", "blockquote"]):
            t = re.sub(r"\s+", " ", p.get_text(" ", strip=True))
            if len(t) < 60:
                continue
            low = t.lower()
            if len(t) < 220 and any(b in low for b in BOILER):
                continue
            paras.append(t)
        # ডুপ্লিকেট বাদ, ক্রম রাখা
        paras = list(dict.fromkeys(paras))
        body_text = clean_text("\n\n".join(paras))

    if not body_text or len(body_text) < 300:
        return None
    return {"title": title or url, "content": body_text[:MAX_DOC_CHARS], "url": url}


MAX_BYTES = 2_000_000
MAX_REDIRECTS = 5


def is_public_url(url: str) -> bool:
    """শুধু পাবলিক ইন্টারনেটের http/https ঠিকানা অনুমোদিত।

    সার্ভারকে দিয়ে ভেতরের ঠিকানা (127.0.0.1, 10.x, 169.254.169.254 ইত্যাদি) পড়ানো
    ঠেকায় — যেকোনো ব্যবহারকারী URL দিতে পারে বলে এটা জরুরি (SSRF)।
    """
    try:
        u = urlparse(url)
        if u.scheme not in ("http", "https") or not u.hostname:
            return False
        if u.port not in (None, 80, 443, 8080, 8443):
            return False
        infos = socket.getaddrinfo(u.hostname, u.port or (443 if u.scheme == "https" else 80),
                                   type=socket.SOCK_STREAM)
        addrs = {i[4][0] for i in infos}
        return bool(addrs) and all(ipaddress.ip_address(a.split("%")[0]).is_global for a in addrs)
    except (OSError, ValueError):
        return False


def _decode(raw: bytes, ctype: str) -> str:
    """সঠিক এনকোডিংয়ে ডিকোড: হেডার → পেজের ভেতরের ঘোষণা → UTF-8।

    (requests হেডারে charset না পেলে text/* কে ISO-8859-1 ধরে নেয় — তাতে বাংলা লেখা
    নষ্ট হয়ে যায়, তাই নিজেরা ঠিক করি।)
    """
    m = re.search(r"charset=([\w-]+)", ctype, re.I)
    enc = m.group(1) if m else None
    if not enc:
        head = raw[:2048].decode("ascii", "ignore")
        m = re.search(r"(?:charset|encoding)=[\"']?([\w-]+)", head, re.I)
        enc = m.group(1) if m else "utf-8"
    try:
        return raw.decode(enc, errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")


def fetch_raw(url: str, timeout: int = 20) -> tuple[bytes, str] | None:
    """নিরাপদ GET: প্রতিটি redirect ধাপে ঠিকানা যাচাই, সর্বোচ্চ ২ MB।"""
    try:
        for _ in range(MAX_REDIRECTS + 1):
            if not is_public_url(url):
                return None
            with requests.get(url, headers=HEADERS, timeout=timeout,
                              allow_redirects=False, stream=True) as r:
                if r.status_code in (301, 302, 303, 307, 308):
                    loc = r.headers.get("location")
                    if not loc:
                        return None
                    url = urljoin(url, loc)
                    continue
                if r.status_code != 200:
                    return None
                ctype = r.headers.get("content-type", "")
                if "html" not in ctype and "xml" not in ctype and "text" not in ctype:
                    return None
                raw = b""
                for chunk in r.iter_content(65536):
                    raw += chunk
                    if len(raw) >= MAX_BYTES:
                        break
                return raw[:MAX_BYTES], ctype
        return None
    except Exception:  # noqa: BLE001
        return None


def fetch(url: str, timeout: int = 20) -> str | None:
    got = fetch_raw(url, timeout)
    return _decode(*got) if got else None


# --------------------------------------------------------------------------- #
# Wikipedia
# --------------------------------------------------------------------------- #
# জনপ্রিয় ও গুরুত্বপূর্ণ বিষয় — ভালো মানের ভূমিকা পাওয়া যায়
SEED_TOPICS = {
    "bn": [
        "বাংলাদেশ", "ঢাকা", "বাংলা ভাষা", "পদ্মা সেতু", "সুন্দরবন", "রবীন্দ্রনাথ ঠাকুর",
        "কাজী নজরুল ইসলাম", "বাংলাদেশের স্বাধীনতা যুদ্ধ", "ভারত", "পাকিস্তান", "চট্টগ্রাম",
        "কৃষি", "ইন্টারনেট", "কৃত্রিম বুদ্ধিমত্তা", "গণিত", "বিজ্ঞান", "সৌরজগৎ",
        "পৃথিবী", "চন্দ্র", "সূর্য", "নদী", "বঙ্গোপসাগর", "অর্থনীতি", "শিক্ষা", "স্বাস্থ্য",
        "ক্রিকেট", "ফুটবল", "চলচ্চিত্র", "সংগীত", "ইতিহাস", "সভ্যতা", "ভাষা", "কম্পিউটার",
        "প্রোগ্রামিং ভাষা", "রোবট", "জলবায়ু পরিবর্তন", "বন্যা", "চা", "মাছ", "পর্যটন",
        "মুক্তিযুদ্ধ", "জাতীয় সংসদ", "বাংলা সাহিত্য", "চিকিৎসাবিজ্ঞান", "মহাবিশ্ব",
        "কোয়ান্টাম বলবিদ্যা", "মোবাইল ফোন", "সামাজিক মাধ্যম", "বিশ্ববিদ্যালয়", "পাখি",
        "গণতন্ত্র", "নির্বাচন", "সংবিধান", "বাংলা চলচ্চিত্র", "নোবেল পুরস্কার", "ওয়েবসাইট",
        "সিলেট",
        "খুলনা",
        "রাজশাহী",
        "বরিশাল",
        "রংপুর",
        "ময়মনসিংহ",
        "ঢাকা বিশ্ববিদ্যালয়",
        "জাতীয় স্মৃতিসৌধ",
        "বাংলা ভাষা আন্দোলন",
        "একুশে বইমেলা",
        "বাংলাদেশ জাতীয় ক্রিকেট দল",
        "সাকিব আল হাসান",
        "মুশফিকুর রহিম",
        "মঙ্গল গ্রহ",
        "বৃহস্পতি গ্রহ",
        "ছায়াপথ",
        "আলবার্ট আইনস্টাইন",
        "আইজ্যাক নিউটন",
        "চার্লস ডারউইন",
        "মহাত্মা গান্ধী",
        "সুভাষচন্দ্র বসু",
        "সত্যজিৎ রায়",
        "অমর্ত্য সেন",
        "তথ্যপ্রযুক্তি",
        "স্মার্টফোন",
        "ফেসবুক",
        "গুগল",
        "ইউটিউব",
        "উইকিপিডিয়া",
        "অ্যালগরিদম",
        "সফটওয়্যার",
        "হার্ডওয়্যার",
        "আবহাওয়া",
        "ভূগোল",
        "সমুদ্র",
        "পর্বত",
        "বন",
        "প্রাণী",
        "উদ্ভিদ",
        "মানবদেহ",
        "মস্তিষ্ক",
        "রক্ত",
        "ডিএনএ",
        "কোষ",
        "পরমাণু",
        "মৌল",
        "শক্তি",
        "বিদ্যুৎ",
        "আলো",
        "শব্দ",
        "মহাকর্ষ",
        "সময়",
        "বাংলা ব্যাকরণ",
        "পহেলা বৈশাখ",
        "ঈদ",
        "দুর্গাপূজা",
        "বাঙালি জাতি",
        "কবিতা",
        "উপন্যাস",
        "স্থাপত্য",
        "রান্না",
        "ভ্রমণ",
        "মহাকাশ অনুসন্ধান",
        "কৃষ্ণগহ্বর",
        "মহাবিস্ফোরণ",
        "পর্যায় সারণি",
        "পরিসংখ্যান",
        "সম্ভাবনা তত্ত্ব",
        "ক্যালকুলাস",
        "দর্শন",
        "মনোবিজ্ঞান",
        "সমাজবিজ্ঞান",
    ],
    "en": [
        "Bangladesh", "Dhaka", "Bengali language", "Padma Bridge", "Sundarbans",
        "Rabindranath Tagore", "Kazi Nazrul Islam", "Bangladesh Liberation War", "India",
        "Pakistan", "Chittagong", "Agriculture", "Internet", "Artificial intelligence",
        "Machine learning", "Language model", "Natural language processing", "Mathematics",
        "Physics", "Chemistry", "Biology", "Solar System", "Earth", "Moon", "Sun", "River",
        "Bay of Bengal", "Economics", "Education", "Health", "Cricket", "Football", "Film",
        "Music", "History", "Civilization", "Computer", "Programming language", "Robot",
        "Climate change", "Flood", "Tea", "Fish", "Tourism", "Astronomy", "Evolution",
        "Democracy", "Election", "Constitution", "World Wide Web", "Smartphone", "Universe",
        "Quantum mechanics", "Nobel Prize", "Photography", "Ocean", "Mountain",
        "Bangladesh national cricket team",
        "Shakib Al Hasan",
        "Mushfiqur Rahim",
        "University of Dhaka",
        "Bengali language movement",
        "Sheikh Mujibur Rahman",
        "Bengali cuisine",
        "Pohela Boishakh",
        "Albert Einstein",
        "Isaac Newton",
        "Charles Darwin",
        "Mahatma Gandhi",
        "Subhas Chandra Bose",
        "Satyajit Ray",
        "Amartya Sen",
        "Facebook",
        "Google",
        "YouTube",
        "Wikipedia",
        "Algorithm",
        "Data science",
        "Software",
        "Computer hardware",
        "Internet of things",
        "Cloud computing",
        "Computer security",
        "Weather",
        "Geography",
        "Forest",
        "Animal",
        "Plant",
        "Human body",
        "Brain",
        "Blood",
        "DNA",
        "Cell (biology)",
        "Atom",
        "Chemical element",
        "Energy",
        "Electricity",
        "Light",
        "Sound",
        "Gravity",
        "Time",
        "Bengali grammar",
        "Bengali culture",
        "Eid al-Fitr",
        "Durga Puja",
        "Bengali people",
        "Poetry",
        "Novel",
        "Architecture",
        "Cooking",
        "Travel",
        "Space exploration",
        "Mars",
        "Jupiter",
        "Galaxy",
        "Milky Way",
        "Black hole",
        "Big Bang",
        "Periodic table",
        "Statistics",
        "Probability",
        "Linear algebra",
        "Calculus",
        "Philosophy",
        "Psychology",
        "Sociology",
        "Photograph",
        "Painting",
        "Sculpture",
        "Dance",
        "Theatre",
        "Newspaper",
        "Television",
        "Radio",
    ],
}


def wiki_seeds(lang: str, count: int, offset: int = 0) -> list[dict]:
    """বিষয়ভিত্তিক নির্বাচিত নিবন্ধের ভূমিকা — দীর্ঘ ও মানসম্মত লেখা।"""
    topics = SEED_TOPICS.get(lang, SEED_TOPICS["en"])
    start = offset % max(1, len(topics))
    picks = [topics[(start + i) % len(topics)] for i in range(min(count, len(topics)))]
    out: list[dict] = []
    for i in range(0, len(picks), 8):
        batch = picks[i:i + 8]
        pages = {}
        for attempt in range(4):
            try:
                r = requests.get(f"https://{lang}.wikipedia.org/w/api.php", params={
                    "action": "query", "format": "json", "titles": "|".join(batch),
                    "prop": "extracts|info", "exintro": 0, "explaintext": 1,
                    "exlimit": len(batch), "inprop": "url", "redirects": 1},
                    headers=HEADERS, timeout=30)
                if r.status_code == 429:
                    time.sleep(2.0 + attempt * 2.5)
                    continue
                pages = (r.json().get("query") or {}).get("pages") or {}
                break
            except Exception:  # noqa: BLE001
                time.sleep(1.2 + attempt)
        for p in pages.values():
            text = clean_text(p.get("extract") or "")
            if len(text) < 400:
                continue
            url = p.get("fullurl") or f"https://{lang}.wikipedia.org/wiki/{quote(str(p.get('title', '')))}"
            out.append({"url": url, "title": p.get("title") or url, "lang": lang,
                        "site_domain": f"{lang}.wikipedia.org",
                        "content": text[:MAX_DOC_CHARS]})
        time.sleep(0.45)
    return out


def wiki_random(lang: str, count: int) -> list[dict]:
    """র‍্যান্ডম নিবন্ধের ভূমিকা-অংশ আনে (প্রতি রিকোয়েস্টে ১০টি)।"""
    out: list[dict] = []
    tries = 0
    while len(out) < count and tries < 8:
        tries += 1
        want = min(10, count - len(out))
        params = {
            "action": "query", "format": "json", "generator": "random",
            "grnnamespace": 0, "grnlimit": want,
            "prop": "extracts|info", "exintro": 1, "explaintext": 1,
            "exlimit": want, "inprop": "url",
        }
        try:
            r = requests.get(f"https://{lang}.wikipedia.org/w/api.php",
                             params=params, headers=HEADERS, timeout=25)
            pages = (r.json().get("query") or {}).get("pages") or {}
        except Exception:  # noqa: BLE001
            time.sleep(1.0)
            continue
        for p in pages.values():
            text = clean_text(p.get("extract") or "")
            if len(text) < 500:
                continue
            url = p.get("fullurl") or f"https://{lang}.wikipedia.org/wiki/{quote(str(p.get('title', '')))}"
            out.append({
                "url": url, "title": p.get("title") or url, "lang": lang,
                "site_domain": f"{lang}.wikipedia.org",
                "content": text[:MAX_DOC_CHARS],
            })
        time.sleep(0.2)
    return out[:count]


def wiki_search(query: str, lang: str, limit: int = 3) -> list[dict]:
    """প্রশ্নের জন্য প্রাসঙ্গিক উইকি নিবন্ধ + ভূমিকা।"""
    try:
        s = requests.get(f"https://{lang}.wikipedia.org/w/api.php", params={
            "action": "query", "format": "json", "list": "search",
            "srsearch": query, "srlimit": limit, "srprop": "snippet"},
            headers=HEADERS, timeout=20)
        hits = ((s.json().get("query") or {}).get("search")) or []
    except Exception:  # noqa: BLE001
        return []
    titles = [h["title"] for h in hits]
    if not titles:
        return []
    try:
        r = requests.get(f"https://{lang}.wikipedia.org/w/api.php", params={
            "action": "query", "format": "json", "titles": "|".join(titles),
            "prop": "extracts|info", "exintro": 1, "explaintext": 1,
            "exlimit": len(titles), "inprop": "url", "redirects": 1},
            headers=HEADERS, timeout=25)
        pages = (r.json().get("query") or {}).get("pages") or {}
    except Exception:  # noqa: BLE001
        return []
    out = []
    for p in pages.values():
        text = clean_text(p.get("extract") or "")
        if len(text) < 200:
            continue
        out.append({
            "url": p.get("fullurl") or f"https://{lang}.wikipedia.org/wiki/{quote(str(p.get('title', '')))}",
            "title": p.get("title") or "", "lang": lang,
            "site_domain": f"{lang}.wikipedia.org", "content": text[:MAX_DOC_CHARS],
        })
    return out


# --------------------------------------------------------------------------- #
# RSS / Atom feeds
# --------------------------------------------------------------------------- #
def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower() if isinstance(tag, str) else ""


def _strip_html(s: str) -> str:
    return clean_text(BeautifulSoup(s or "", "html.parser").get_text(" ", strip=True))


def parse_feed(xml: str | bytes, limit: int = 20) -> list[dict]:
    """RSS 2.0 / Atom / RDF ফিড থেকে (title, url, summary)।

    শুধু স্ট্যান্ডার্ড লাইব্রেরি (xml.etree) — আগে BeautifulSoup-এর "xml" পার্সার
    ব্যবহার হতো, যার জন্য lxml লাগে; lxml না থাকলে সব ফিড নীরবে খালি আসত।
    """
    raw = xml.encode("utf-8") if isinstance(xml, str) else xml
    if b"<!ENTITY" in raw[:4096]:
        return []
    if isinstance(xml, str):       # str হলে ভেতরের encoding ঘোষণা আর প্রযোজ্য নয়
        raw = re.sub(rb"^\s*<\?xml[^>]*\?>", b"", raw, count=1)
    try:
        root = ET.fromstring(raw.lstrip())
    except ET.ParseError:
        return _parse_feed_loose(raw.decode("utf-8", "replace"), limit)
    items: list[dict] = []
    for it in root.iter():
        if _local(it.tag) not in ("item", "entry"):
            continue
        title = link = guid = ""
        summary = ""
        for ch in it:
            name = _local(ch.tag)
            text = "".join(ch.itertext()).strip()
            if name == "title" and not title:
                title = _strip_html(text)
            elif name == "link":
                href = ch.get("href")
                if href:                                   # Atom
                    if ch.get("rel") in (None, "alternate") or not link:
                        link = href.strip()
                elif text and not link:                    # RSS
                    link = text
            elif name in ("guid", "id") and not guid:
                guid = text
            elif name in ("description", "summary", "encoded", "content"):
                t = _strip_html(text)
                if len(t) > len(summary):
                    summary = t
        url = link if link.startswith("http") else guid if guid.startswith("http") else ""
        if url:
            items.append({"title": title, "url": url, "summary": summary})
            if len(items) >= limit:
                break
    return items


def _parse_feed_loose(text: str, limit: int) -> list[dict]:
    """ভাঙা XML-এর জন্য ঢিলেঢালা বিকল্প (regex)।"""
    items: list[dict] = []
    for m in re.finditer(r"<(item|entry)\b.*?</\1>", text, re.S | re.I):
        block = m.group(0)

        def grab(tag: str) -> str:
            g = re.search(rf"<{tag}\b[^>]*>(.*?)</{tag}>", block, re.S | re.I)
            v = g.group(1) if g else ""
            return re.sub(r"^<!\[CDATA\[|\]\]>$", "", v.strip()).strip()

        link = grab("link")
        if not link.startswith("http"):
            g = re.search(r"<link\b[^>]*href=[\"']([^\"']+)", block, re.I)
            link = g.group(1) if g else grab("guid")
        if link.startswith("http"):
            items.append({"title": _strip_html(grab("title")), "url": link.strip(),
                          "summary": _strip_html(grab("description") or grab("summary"))})
            if len(items) >= limit:
                break
    return items


def feed_links(feed_url: str, limit: int = 20) -> list[dict]:
    got = fetch_raw(feed_url)
    if not got:
        return []
    try:
        return parse_feed(got[0], limit)
    except Exception as e:  # noqa: BLE001
        print(f"[feed] {feed_url}: {e}")
        return []


def learn_feed(feed_url: str, domain: str, lang: str, count: int = 15) -> list[dict]:
    docs: list[dict] = []
    for item in feed_links(feed_url, limit=count * 2):
        if len(docs) >= count:
            break
        html = fetch(item["url"])
        art = extract_article(html, item["url"]) if html else None
        if not art:
            sm = item.get("summary") or ""
            if len(sm) < 150:
                continue
            art = {"url": item["url"], "title": item["title"], "content": sm[:MAX_DOC_CHARS]}
        art.update({"lang": lang, "site_domain": domain, "title": art.get("title") or item["title"]})
        docs.append(art)
        time.sleep(0.15)
    return docs


def learn_url(url: str) -> dict | None:
    html = fetch(url)
    if not html:
        return None
    art = extract_article(html, url)
    if not art:
        return None
    host = re.sub(r"^https?://(www\.)?", "", url).split("/")[0]
    lang = "bn" if any(c in art["content"][:800] for c in "অআইউএওকখগঘচছজঝটঠডঢতথদধনপফবভমযরলশষসহ") else "en"
    art.update({"lang": lang, "site_domain": host})
    return art
