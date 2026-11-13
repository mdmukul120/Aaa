"""স্পাইডার — একটা লিংক থেকে শুরু করে একই সাইটের ভেতরের লিংক ধরে ধরে পেজ পড়া।

নিয়ম (সাইটের ক্ষতি না করে, নিজের সার্ভারও না ডুবিয়ে):
  • শুধু একই সাইট (একই হোস্ট; www থাকা-না-থাকা এক ধরা হয়) — বাইরের লিংকে যায় না
  • robots.txt মানে; সাইট Crawl-delay বললে সেটাও (সর্বোচ্চ ১০ সেকেন্ড)
  • প্রতি পেজের মাঝে বিরতি; প্রতি রাউন্ডে নির্দিষ্ট সংখ্যক পেজ; গভীরতার সীমা
  • সারি ডেটাবেসে থাকে — থামলে/রিস্টার্ট হলে পরের বার সেখান থেকেই চলে, একই URL দু'বার পড়ে না
"""
from __future__ import annotations

import time
from typing import Callable, Iterator
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import crawler
import db
import linkreader
from config import LOW_MEM, USER_AGENT

MAX_PAGES_PER_ROUND = 60 if LOW_MEM else 200
MAX_DEPTH = 6
MAX_QUEUE = 3000 if LOW_MEM else 10000       # এক সাইটে এর বেশি লিংক জমানো হয় না
DEFAULT_DELAY = 1.0

_SKIP_EXT = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico", ".bmp", ".css", ".js", ".mjs",
             ".zip", ".rar", ".7z", ".gz", ".tar", ".exe", ".apk", ".dmg", ".iso", ".mp3", ".mp4",
             ".m4a", ".avi", ".mov", ".mkv", ".webm", ".wav", ".woff", ".woff2", ".ttf", ".eot",
             ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx")
_TRACKING = ("utm_", "fbclid", "gclid", "mc_cid", "mc_eid", "ref_src", "igshid")
_SKIP_PATH = ("/login", "/logout", "/signin", "/signup", "/register", "/cart", "/checkout",
              "/wp-admin", "/wp-login", "/account", "/share", "/print")


def norm_url(url: str) -> str | None:
    """একই পেজের নানা রূপকে এক ঠিকানায় আনা (# অংশ, ট্র্যাকিং প্যারামিটার, ডিফল্ট পোর্ট বাদ)।"""
    try:
        u = urlsplit((url or "").strip())
        port = u.port
    except ValueError:
        return None
    if u.scheme not in ("http", "https") or not u.hostname:
        return None
    host = u.hostname.lower()
    if port and port != (443 if u.scheme == "https" else 80):
        host += f":{port}"
    query = urlencode([(k, v) for k, v in parse_qsl(u.query, keep_blank_values=True)
                       if not k.lower().startswith(_TRACKING)])
    out = urlunsplit((u.scheme, host, u.path or "/", query, ""))
    return out if len(out) <= 500 else None


def site_root(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def on_origin(url: str, origin: tuple[str, str]) -> str:
    """একই সাইটের লিংককে এক রূপে আনা — http/https বা www থাকা-না-থাকায় একই পেজ যেন দু'বার না পড়া হয়।"""
    u = urlsplit(url)
    return urlunsplit((origin[0], origin[1], u.path, u.query, ""))


def wanted(url: str, root: str) -> bool:
    """এই লিংক কি সারিতে নেওয়ার মতো? (একই সাইট, পড়ার মতো ফাইল, লগইন/কার্ট নয়)"""
    if site_root(url) != root:
        return False
    u = urlsplit(url)
    path = u.path.lower()
    if path.endswith(_SKIP_EXT) or any(x in path for x in _SKIP_PATH):
        return False
    return len(u.query) <= 120 and path.count("/") <= 12


class Robots:
    """একটা সাইটের robots.txt — পড়া না গেলে নিরাপদ দিকটাই ধরা হয়।"""

    def __init__(self, any_url: str, fetch: Callable[..., dict] = crawler.fetch_any) -> None:
        u = urlsplit(any_url)
        self.url = f"{u.scheme}://{u.netloc}/robots.txt"
        self.rp = RobotFileParser()
        self.blocked_all = False
        self.note = ""
        got = fetch(self.url, timeout=10)
        if "raw" in got:
            text = got["raw"][:500_000].decode("utf-8", "replace")
            # কিছু সাইট robots.txt-র জায়গায় HTML পেজ দেয় — সেটা নিয়ম নয়
            self.rp.parse([] if got.get("kind") == "html" else text.splitlines())
        elif got.get("status") in (401, 403):
            self.blocked_all, self.note = True, "সাইট robots.txt দেখতে দেয়নি — স্পাইডার চালানো হলো না"
        elif got.get("status") and 400 <= got["status"] < 500:
            self.rp.parse([])                    # robots.txt নেই → কোনো বাধা নেই
        else:
            self.blocked_all = True
            self.note = f"robots.txt পড়া যায়নি ({got.get('error', '?')}) — পরে আবার চেষ্টা করুন"

    def allowed(self, url: str) -> bool:
        if self.blocked_all:
            return False
        try:
            return self.rp.can_fetch(USER_AGENT, url)
        except Exception:  # noqa: BLE001
            return True

    def delay(self, default: float) -> float:
        try:
            d = self.rp.crawl_delay(USER_AGENT)
        except Exception:  # noqa: BLE001
            d = None
        return max(default, min(float(d), 10.0)) if d else default


def crawl(start_url: str, pages: int = 40, max_depth: int = 3, delay: float = DEFAULT_DELAY,
          restart: bool = False,
          read: Callable[..., dict] = linkreader.read_link,
          robots_factory: Callable[[str], Robots] = Robots,
          sleep: Callable[[float], None] = time.sleep,
          should_stop: Callable[[], bool] = lambda: False) -> Iterator[dict]:
    """এক রাউন্ড চালায়; প্রতিটি পড়া পেজের জন্য একটা dict দেয়:

      {"event": "page", "url", "depth", "status": done|empty|failed, "docs", "qa", "error", "queued"}
      {"event": "skip", "url", "reason"}        — robots.txt-এ নিষেধ / অন্য সাইটে চলে গেছে
      {"event": "stop", "reason"}               — শুরুই করা যায়নি
    সেভ করা/শেখানো ডাকনেওয়ালার কাজ; সারির হিসাব এখানেই।
    """
    start = norm_url(start_url)
    if not start or not crawler.is_public_url(start):
        yield {"event": "stop", "reason": "ঠিকানা অনুমোদিত নয় — শুধু পাবলিক http/https লিংক"}
        return
    root = site_root(start)
    origin = (urlsplit(start).scheme, urlsplit(start).netloc)
    pages = max(1, min(int(pages), MAX_PAGES_PER_ROUND))
    max_depth = max(0, min(int(max_depth), MAX_DEPTH))
    robots = robots_factory(start)
    if robots.blocked_all:
        yield {"event": "stop", "reason": robots.note}
        return
    delay = robots.delay(delay)
    if restart:
        db.queue_reset(root)
    db.queue_add(root, [(start, 0)])
    queued = db.queue_stats(root)["total"]

    done = 0
    while done < pages and not should_stop():
        batch = db.queue_next(root, min(10, pages - done))
        if not batch:
            break
        for row in batch:
            if done >= pages or should_stop():
                break
            url, depth = row["url"], int(row.get("depth") or 0)
            if not robots.allowed(url):
                db.queue_mark(row, "blocked", "robots.txt")
                yield {"event": "skip", "url": url, "reason": "robots.txt-এ নিষেধ"}
                continue
            if done:
                sleep(delay)
            done += 1
            try:
                res = read(url, want_links=depth < max_depth)
            except Exception as e:  # noqa: BLE001
                res = {"ok": False, "docs": [], "qa": [], "links": [], "error": f"পড়া যায়নি: {e}", "url": url}
            final = norm_url(res.get("url") or url) or url
            if site_root(final) != root:
                db.queue_mark(row, "blocked", "অন্য সাইটে redirect")
                yield {"event": "skip", "url": url, "reason": "অন্য সাইটে চলে গেছে"}
                continue
            added = 0
            if depth < max_depth and queued < MAX_QUEUE:
                links: list[tuple[str, int]] = []
                seen = {url, on_origin(final, origin)}
                for raw in res.get("links") or []:
                    link = norm_url(raw)
                    if not link or not wanted(link, root):
                        continue
                    link = on_origin(link, origin)
                    if link not in seen:
                        seen.add(link)
                        links.append((link, depth + 1))
                links = links[:max(0, MAX_QUEUE - queued)]
                if links:
                    db.queue_add(root, links)
                    before, queued = queued, db.queue_stats(root)["total"]   # আগে থেকে থাকা লিংক গোনা হয় না
                    added = max(0, queued - before)
            docs, qa = res.get("docs") or [], res.get("qa") or []
            status = "done" if (docs or qa) else "empty" if res.get("ok") else "failed"
            db.queue_mark(row, status, res.get("error") or "")
            yield {"event": "page", "url": url, "depth": depth, "status": status, "kind": res.get("kind"),
                   "docs": docs, "qa": qa, "error": res.get("error"), "queued": added, "n": done}
