"""Supabase REST access + local JSONL fallback.

Uses only the project URL + service key (server side).
Every document is also mirrored to data/corpus.jsonl so training keeps
working even if Supabase is unreachable.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any

import requests

from config import CORPUS_PATH, SUPABASE_KEY, SUPABASE_URL

_lock = threading.Lock()
QA_PATH = CORPUS_PATH.with_name("qa.jsonl")                 # Supabase না থাকলে প্রশ্ন-উত্তর এখানে
QUEUE_PATH = CORPUS_PATH.with_name("crawl_queue.json")      # … আর স্পাইডারের সারি এখানে


def configured() -> bool:
    return bool(SUPABASE_URL and SUPABASE_KEY)


def _headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    h = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    if extra:
        h.update(extra)
    return h


def _req(method: str, path: str, *, params: dict | None = None,
         body: Any = None, extra_headers: dict | None = None,
         timeout: int = 25) -> Any:
    if not configured():
        raise RuntimeError("Supabase is not configured")
    url = f"{SUPABASE_URL}/rest/v1/{path.lstrip('/')}"
    r = requests.request(method, url, headers=_headers(extra_headers),
                         params=params, json=body, timeout=timeout)
    if r.status_code >= 300:
        raise RuntimeError(f"Supabase {method} {path} -> {r.status_code}: {r.text[:400]}")
    if not r.text:
        return None
    try:
        return r.json()
    except ValueError:
        return r.text


# --------------------------------------------------------------------------- #
# sites
# --------------------------------------------------------------------------- #
def upsert_sites(sites: list[dict]) -> int:
    if not configured() or not sites:
        return 0
    rows = [{
        "domain": s["domain"], "name": s["name"], "homepage": s["homepage"],
        "category": s.get("category", "news"), "lang": s.get("lang", "en"),
        "feed_url": s.get("feed"), "rank": s.get("rank"),
    } for s in sites]
    _req("POST", "sites", params={"on_conflict": "domain,name"}, body=rows,
         extra_headers={"Prefer": "resolution=merge-duplicates,return=minimal"})
    return len(rows)


def list_sites() -> list[dict]:
    return _req("GET", "sites", params={
        "select": "domain,name,homepage,category,lang,feed_url,rank",
        "order": "rank.asc.nullslast", "limit": 100}) or []


# --------------------------------------------------------------------------- #
# documents
# --------------------------------------------------------------------------- #
def insert_documents(docs: list[dict]) -> set[str] | None:
    """নতুন ডকুমেন্ট সেভ করে নতুন URL-গুলোর সেট ফেরায়।

    None = Supabase-এ লেখা যায়নি (তখন সব ডকুমেন্টই 'নতুন' ধরে নিতে হবে)।
    """
    if not docs:
        return set()
    new: set[str] = set()
    failed = False
    if configured():
        for i in range(0, len(docs), 40):
            batch = docs[i:i + 40]
            rows = [{
                "site_domain": d.get("site_domain"), "url": d["url"],
                "title": d.get("title", ""), "lang": d.get("lang", "en"),
                "content": d["content"],
                "char_count": len(d["content"]),
                "token_count": d.get("token_count") or len(d["content"].split()),
            } for d in batch]
            try:
                res = _req("POST", "documents", params={"on_conflict": "url"}, body=rows,
                           extra_headers={"Prefer": "resolution=ignore-duplicates,return=representation"})
                for r in (res or []):
                    if isinstance(r, dict) and r.get("url"):
                        new.add(r["url"])
            except Exception as e:  # noqa: BLE001
                print("[db] insert failed:", e)
                failed = True

    keep = docs if (failed or not configured()) else [d for d in docs if d["url"] in new]
    with _lock, CORPUS_PATH.open("a", encoding="utf-8") as f:
        for d in keep:
            f.write(json.dumps({
                "url": d["url"], "title": d.get("title", ""),
                "lang": d.get("lang", "en"), "site_domain": d.get("site_domain"),
                "content": d["content"], "token_count": d.get("token_count", 0),
            }, ensure_ascii=False) + "\n")
    return None if (failed or not configured()) else new


def download_corpus(page_size: int = 400, max_docs: int = 60000) -> int:
    """Supabase থেকে সব ডকুমেন্ট এনে data/corpus.jsonl বানায়/বাড়ায়।

    হোস্টের ফাইল সিস্টেম অস্থায়ী হলেও (Render/Fly/HF Spaces) কিছু হারায় না —
    সার্ভার চালু হওয়ার সময় এটাই নিজে নিজে চলে।
    """
    if not configured():
        return 0
    seen: set[str] = set()
    if CORPUS_PATH.exists():
        with CORPUS_PATH.open(encoding="utf-8") as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["url"])
                except (ValueError, KeyError):
                    continue
    total = offset = 0
    while offset < max_docs:
        try:
            rows = _req("GET", "documents", params={
                "select": "url,title,lang,site_domain,content,token_count",
                "order": "id.asc", "limit": page_size, "offset": offset})
        except Exception as e:  # noqa: BLE001
            print("[db] download_corpus failed:", e)
            break
        if not rows:
            break
        with _lock, CORPUS_PATH.open("a", encoding="utf-8") as f:
            for r in rows:
                if not r.get("url") or r["url"] in seen:
                    continue
                seen.add(r["url"])
                f.write(json.dumps({
                    "url": r["url"], "title": r.get("title", ""),
                    "lang": r.get("lang", "en"), "site_domain": r.get("site_domain"),
                    "content": r.get("content", ""),
                    "token_count": r.get("token_count", 0),
                }, ensure_ascii=False) + "\n")
                total += 1
        offset += len(rows)
        if len(rows) < page_size:
            break
    return total


# --------------------------------------------------------------------------- #
# Supabase Storage — মডেল ফাইল ক্যাশ (দ্রুত কোল্ড-স্টার্ট)
# --------------------------------------------------------------------------- #
BUCKET = "models"


def _storage(path: str) -> str:
    return f"{SUPABASE_URL}/storage/v1/{path.lstrip('/')}"


def ensure_bucket(name: str = BUCKET) -> bool:
    if not configured():
        return False
    try:
        r = requests.get(_storage(f"bucket/{name}"), headers=_headers(), timeout=20)
        if r.status_code == 200:
            return True
        r = requests.post(_storage("bucket"), headers=_headers(),
                          json={"id": name, "name": name, "public": False}, timeout=20)
        return r.status_code in (200, 201, 409)
    except Exception as e:  # noqa: BLE001
        print("[storage] ensure_bucket failed:", e)
        return False


def upload_model(local_path, name: str = "shobdo-model.pkl.gz") -> bool:
    """ফাইল Supabase Storage-এ রাখে (গিটে কমিট না করেই সব হোস্টে পাওয়া যাবে)।"""
    if not configured() or not local_path.exists():
        return False
    if not ensure_bucket():
        return False
    ctype = "application/json" if name.endswith(".json") else "application/octet-stream"
    try:
        with local_path.open("rb") as f:
            r = requests.post(_storage(f"object/{BUCKET}/{name}"),
                              headers={**_headers(), "content-type": ctype,
                                       "x-upsert": "true", "cache-control": "max-age=300"},
                              data=f, timeout=300)
        ok = r.status_code in (200, 201)
        print(f"[storage] upload {name}: {'OK' if ok else r.status_code}")
        return ok
    except Exception as e:  # noqa: BLE001
        print("[storage] upload failed:", e)
        return False


def download_model(local_path, name: str = "shobdo-model.pkl.gz", min_size: int = 1000) -> bool:
    """হোস্ট চালু হওয়ার সময় Storage থেকে ফাইল নামায় (কয়েক সেকেন্ডে)।"""
    if not configured():
        return False
    try:
        r = requests.get(_storage(f"object/{BUCKET}/{name}"), headers=_headers(), timeout=180)
        if r.status_code != 200 or len(r.content) < min_size:
            return False
        local_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = local_path.with_suffix(local_path.suffix + ".tmp")
        tmp.write_bytes(r.content)
        tmp.replace(local_path)
        print(f"[storage] download {name}: {len(r.content)/1e6:.1f} MB")
        return True
    except Exception as e:  # noqa: BLE001
        print("[storage] download failed:", e)
        return False


def fetch_json(name: str) -> dict | None:
    """Storage-এর ছোট JSON ফাইল (যেমন নিউরাল মডেলের মেটা) পড়ে; না থাকলে None।"""
    if not configured():
        return None
    try:
        r = requests.get(_storage(f"object/{BUCKET}/{name}"), headers=_headers(), timeout=20)
        if r.status_code != 200:
            return None
        data = r.json()
        return data if isinstance(data, dict) else None
    except Exception:  # noqa: BLE001
        return None


def recent_documents(limit: int = 20) -> list[dict]:
    if configured():
        try:
            return _req("GET", "documents", params={
                "select": "id,title,url,site_domain,lang,char_count,token_count,fetched_at",
                "order": "id.desc", "limit": limit}) or []
        except Exception:  # noqa: BLE001
            pass
    out: list[dict] = []
    if CORPUS_PATH.exists():
        lines = CORPUS_PATH.read_text(encoding="utf-8").strip().splitlines()[-limit:]
        for ln in reversed(lines):
            try:
                d = json.loads(ln)
            except ValueError:
                continue
            out.append({"id": len(out), "title": d.get("title"), "url": d.get("url"),
                        "site_domain": d.get("site_domain"), "lang": d.get("lang"),
                        "char_count": len(d.get("content", "")),
                        "token_count": d.get("token_count", 0)})
    return out


def doc_count() -> int:
    if configured():
        try:
            r = requests.get(f"{SUPABASE_URL}/rest/v1/documents",
                             headers=_headers({"Prefer": "count=exact", "Range": "0-0"}),
                             params={"select": "id"}, timeout=20)
            cr = r.headers.get("content-range", "")
            if "/" in cr:
                return int(cr.split("/")[-1])
        except Exception:  # noqa: BLE001
            pass
    if CORPUS_PATH.exists():
        return sum(1 for _ in CORPUS_PATH.open(encoding="utf-8"))
    return 0


def search_documents(q: str, limit: int = 8) -> list[dict]:
    q = (q or "").strip()
    if not q:
        return []
    if configured():
        try:
            return _req("POST", "rpc/search_documents", body={"q": q, "lim": limit}) or []
        except Exception as e:  # noqa: BLE001
            print("[db] search rpc failed:", e)
    # local fallback: token overlap
    terms = [w.lower() for w in q.split() if len(w) > 1]
    scored: list[tuple[int, dict]] = []
    if CORPUS_PATH.exists():
        for ln in CORPUS_PATH.open(encoding="utf-8"):
            try:
                d = json.loads(ln)
            except ValueError:
                continue
            text = (d.get("title", "") + " " + d.get("content", "")).lower()
            score = sum(text.count(t) for t in terms)
            if score:
                snippet = d.get("content", "")[:280]
                scored.append((score, {"id": 0, "title": d.get("title"),
                                       "url": d.get("url"),
                                       "site_domain": d.get("site_domain"),
                                       "lang": d.get("lang"), "snippet": snippet,
                                       "rank": score}))
    scored.sort(key=lambda x: -x[0])
    return [s for _, s in scored[:limit]]


# --------------------------------------------------------------------------- #
# runs + model stats + usage
# --------------------------------------------------------------------------- #
def log_run(source: str, pages: int, tokens: int, docs_added: int, notes: str = "") -> None:
    if not configured():
        return
    try:
        _req("POST", "learning_runs", body=[{
            "source": source, "pages": pages, "tokens": tokens,
            "docs_added": docs_added, "notes": notes[:500]}],
            extra_headers={"Prefer": "return=minimal"})
    except Exception as e:  # noqa: BLE001
        print("[db] log_run failed:", e)


def save_model_stats(stats: dict) -> None:
    if not configured():
        return
    try:
        _req("POST", "model_stats", params={"on_conflict": "id"}, body=[{
            "id": 1, "vocab": stats.get("vocab", 0),
            "unigrams": stats.get("unigrams", 0), "bigrams": stats.get("bigrams", 0),
            "trigrams": stats.get("trigrams", 0), "tokens": stats.get("tokens", 0),
        }], extra_headers={"Prefer": "resolution=merge-duplicates,return=minimal"})
    except Exception as e:  # noqa: BLE001
        print("[db] save_model_stats failed:", e)


def log_predictions(rows: list[dict]) -> None:
    if not configured() or not rows:
        return
    try:
        _req("POST", "predictions", body=rows[-20:],
             extra_headers={"Prefer": "return=minimal"})
    except Exception as e:  # noqa: BLE001
        print("[db] log_predictions failed:", e)


def list_runs(limit: int = 6) -> list[dict]:
    if not configured():
        return []
    try:
        return _req("GET", "learning_runs", params={
            "select": "source,pages,tokens,docs_added,notes,created_at",
            "order": "id.desc", "limit": limit}) or []
    except Exception:  # noqa: BLE001
        return []


def db_usage() -> dict:
    if configured():
        try:
            rows = _req("POST", "rpc/db_usage", body={})
            if rows:
                r = rows[0]
                return {"db_total": r.get("db_total"), "doc_count": r.get("doc_count"),
                        "doc_bytes": r.get("doc_bytes"), "site_count": r.get("site_count")}
        except Exception as e:  # noqa: BLE001
            print("[db] db_usage failed:", e)
    return {"db_total": "?", "doc_count": doc_count(), "doc_bytes": "?", "site_count": 0}


def ping() -> bool:
    if not configured():
        return False
    try:
        requests.get(f"{SUPABASE_URL}/rest/v1/sites",
                     headers=_headers(), params={"select": "domain", "limit": 1}, timeout=10)
        return True
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# প্রশ্ন-উত্তর জোড়া (qa_pairs) — "জিজ্ঞাসা"-র প্রথম উৎস
# --------------------------------------------------------------------------- #
def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def insert_qa(pairs: list[dict], source_url: str = "", site_domain: str = "") -> int:
    """[{"q", "a"}] সেভ করে; একই প্রশ্ন আবার এলে উত্তরটা নতুনটা দিয়ে বদলে যায় (সংশোধন করা যায়)।

    ফেরত: কয়টা জোড়া সেভ হলো।
    """
    from answer import norm_question

    rows: dict[str, dict] = {}
    for p in pairs:
        q, a = str(p.get("q", "")).strip(), str(p.get("a", "")).strip()
        qn = norm_question(q)
        if not qn or not a:
            continue
        rows[qn] = {"question": q[:400], "answer": a[:2000], "q_norm": qn,
                    "lang": "bn" if any("\u0980" <= ch <= "\u09FF" for ch in q) else "en",
                    "source_url": (p.get("url") or source_url or "")[:500] or None,
                    "site_domain": site_domain or None, "updated_at": _now()}
    items = list(rows.values())
    if not items:
        return 0
    if configured():
        try:
            for i in range(0, len(items), 200):
                _req("POST", "qa_pairs", params={"on_conflict": "q_norm"}, body=items[i:i + 200],
                     extra_headers={"Prefer": "resolution=merge-duplicates,return=minimal"})
            return len(items)
        except Exception as e:  # noqa: BLE001
            print("[db] qa insert failed (deploy/supabase-qa-spider.sql চালানো হয়েছে?):", e)
    with _lock, QA_PATH.open("a", encoding="utf-8") as f:
        for r in items:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(items)


def _local_qa() -> list[dict]:
    rows: dict[str, dict] = {}
    if QA_PATH.exists():
        with QA_PATH.open(encoding="utf-8") as f:
            for ln in f:
                try:
                    r = json.loads(ln)
                    rows[r["q_norm"]] = r        # পরের লাইন আগেরটাকে বদলায়
                except (ValueError, KeyError, TypeError):
                    continue
    return list(rows.values())


def match_qa(q_norm: str, limit: int = 12) -> list[dict]:
    """প্রশ্নের কাছাকাছি জমানো প্রশ্নগুলো (চূড়ান্ত বাছাই answer.Matcher করে)।"""
    if not q_norm:
        return []
    if configured():
        try:
            return _req("POST", "rpc/match_qa", body={"q": q_norm, "lim": limit}, timeout=12) or []
        except Exception as e:  # noqa: BLE001
            print("[db] match_qa failed:", e)
    want = set(q_norm.split())
    scored = [(len(want & set(r["q_norm"].split())), r) for r in _local_qa()]
    scored = [x for x in scored if x[0]]
    scored.sort(key=lambda x: -x[0])
    return [r for _, r in scored[:max(limit, 30)]]


def qa_count() -> int:
    if configured():
        try:
            r = requests.get(f"{SUPABASE_URL}/rest/v1/qa_pairs",
                             headers=_headers({"Prefer": "count=exact", "Range": "0-0"}),
                             params={"select": "id"}, timeout=12)
            cr = r.headers.get("content-range", "")
            if r.status_code < 300 and "/" in cr:
                return int(cr.split("/")[-1])
        except Exception:  # noqa: BLE001
            pass
    return len(_local_qa())


def match_documents(terms: list[str], limit: int = 8) -> list[dict]:
    """যে ডকুমেন্টগুলোতে প্রশ্নের শব্দ সবচেয়ে বেশি আছে — পুরো content সহ।"""
    terms = [t for t in dict.fromkeys(terms) if 2 <= len(t) <= 60][:12]
    if not terms:
        return []
    if configured():
        try:
            return _req("POST", "rpc/match_documents", body={"terms": terms, "lim": limit},
                        timeout=15) or []
        except Exception as e:  # noqa: BLE001
            print("[db] match_documents failed:", e)
    scored: list[tuple[int, int, dict]] = []
    if CORPUS_PATH.exists():
        for ln in CORPUS_PATH.open(encoding="utf-8"):
            try:
                d = json.loads(ln)
            except ValueError:
                continue
            text = (d.get("title", "") + " " + d.get("content", "")).lower()
            hits = sum(1 for t in terms if t in text)
            if hits:
                scored.append((hits, sum(text.count(t) for t in terms), d))
    scored.sort(key=lambda x: (-x[0], -x[1]))
    return [{"id": 0, "title": d.get("title"), "url": d.get("url"), "site_domain": d.get("site_domain"),
             "lang": d.get("lang"), "content": d.get("content", ""), "hits": h, "rank": float(c)}
            for h, c, d in scored[:limit]]


# --------------------------------------------------------------------------- #
# স্পাইডারের সারি (crawl_queue) — কোন লিংক বাকি/শেষ; রিস্টার্টের পরেও মনে থাকে
# --------------------------------------------------------------------------- #
_queue_mem: dict[str, dict] | None = None
_queue_remote = {"retry_at": 0.0}  # Supabase-এর সারি ব্যর্থ হলে ১০ মিনিট লোকাল ফাইলে চলে, পরে আবার চেষ্টা


def _use_remote_queue() -> bool:
    return configured() and time.time() >= _queue_remote["retry_at"]


def _queue_fail(e: Exception) -> None:
    print("[db] crawl_queue ব্যর্থ — আপাতত লোকাল সারি (deploy/supabase-qa-spider.sql চালানো আছে?):", e)
    _queue_remote["retry_at"] = time.time() + 600


def _queue_load() -> dict[str, dict]:
    global _queue_mem
    if _queue_mem is None:
        try:
            _queue_mem = json.loads(QUEUE_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _queue_mem = {}
    return _queue_mem


def _queue_save() -> None:
    try:
        QUEUE_PATH.write_text(json.dumps(_queue_mem, ensure_ascii=False), encoding="utf-8")
    except OSError as e:
        print("[db] queue save failed:", e)


def queue_add(root: str, items: list[tuple[str, int]]) -> None:
    """নতুন লিংক সারিতে; আগে থেকে থাকা URL ছোঁয়া হয় না (একই পেজ দু'বার পড়া হয় না)।"""
    if not items:
        return
    if _use_remote_queue():
        try:
            rows = [{"root": root, "url": u, "depth": d} for u, d in items]
            for i in range(0, len(rows), 200):
                _req("POST", "crawl_queue", params={"on_conflict": "url"}, body=rows[i:i + 200],
                     extra_headers={"Prefer": "resolution=ignore-duplicates,return=minimal"})
            return
        except Exception as e:  # noqa: BLE001
            _queue_fail(e)
    with _lock:
        q = _queue_load()
        for u, d in items:
            q.setdefault(u, {"root": root, "depth": d, "status": "pending", "n": len(q)})
        _queue_save()


def queue_next(root: str, n: int = 10) -> list[dict]:
    """পরের n-টি বাকি লিংক — কম গভীরতার (শুরুর কাছের) পেজ আগে।"""
    if _use_remote_queue():
        try:
            return _req("GET", "crawl_queue", params={
                "select": "id,url,depth", "root": f"eq.{root}", "status": "eq.pending",
                "order": "depth.asc,id.asc", "limit": n}) or []
        except Exception as e:  # noqa: BLE001
            _queue_fail(e)
    with _lock:
        rows = [{"id": 0, "url": u, "depth": r["depth"], "n": r.get("n", 0)}
                for u, r in _queue_load().items() if r["root"] == root and r["status"] == "pending"]
    rows.sort(key=lambda r: (r["depth"], r["n"]))
    return rows[:n]


def queue_mark(row: dict, status: str, note: str = "") -> None:
    if _use_remote_queue() and row.get("id"):
        try:
            _req("PATCH", "crawl_queue", params={"id": f"eq.{row['id']}"},
                 body={"status": status, "note": note[:300] or None, "fetched_at": _now()},
                 extra_headers={"Prefer": "return=minimal"})
            return
        except Exception as e:  # noqa: BLE001
            _queue_fail(e)
    with _lock:
        r = _queue_load().get(row["url"])
        if r:
            r["status"] = status
            _queue_save()


def queue_stats(root: str) -> dict:
    if _use_remote_queue():
        try:
            rows = _req("POST", "rpc/crawl_stats", body={"p_root": root})
            if rows:
                return {k: int(rows[0].get(k) or 0) for k in ("pending", "done", "failed", "total")}
        except Exception as e:  # noqa: BLE001
            _queue_fail(e)
    out = {"pending": 0, "done": 0, "failed": 0, "total": 0}
    with _lock:
        for r in _queue_load().values():
            if r["root"] == root:
                out["total"] += 1
                key = ("pending" if r["status"] == "pending"
                       else "done" if r["status"] in ("done", "empty") else "failed")
                out[key] += 1
    return out


def queue_reset(root: str) -> None:
    """একটা সাইটের সারি মুছে নতুন করে শুরু (জমানো ডকুমেন্ট/প্রশ্ন-উত্তর থেকে যায়)।"""
    if _use_remote_queue():
        try:
            _req("DELETE", "crawl_queue", params={"root": f"eq.{root}"},
                 extra_headers={"Prefer": "return=minimal"})
            return
        except Exception as e:  # noqa: BLE001
            _queue_fail(e)
    with _lock:
        q = _queue_load()
        for u in [u for u, r in q.items() if r["root"] == root]:
            del q[u]
        _queue_save()
