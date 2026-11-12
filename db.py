"""Supabase REST access + local JSONL fallback.

Uses only the project URL + service key (server side).
Every document is also mirrored to data/corpus.jsonl so training keeps
working even if Supabase is unreachable.
"""
from __future__ import annotations

import json
import threading
from typing import Any

import requests

from config import CORPUS_PATH, SUPABASE_KEY, SUPABASE_URL

_lock = threading.Lock()


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
