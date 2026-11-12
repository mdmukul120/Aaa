"""বাল্ক-লার্নার: HuggingFace-এর উইকিপিডিয়া ডেটাসেট থেকে হাজারো নিবন্ধ।

উৎস: wikimedia/wikipedia (CC BY-SA) — datasets-server API দিয়ে টুকরো টুকরো আনা হয়।
ব্যবহার:  python crawl_hf.py --bn 300 --en 200 [--offset random]
"""
from __future__ import annotations

import argparse
import random
import time

import requests

import crawler
import db
from config import MAX_DOC_CHARS

API = "https://datasets-server.huggingface.co/rows"
TOTAL = {"20231101.bn": 140_000, "20231101.en": 6_400_000}  # আনুমানিক মোট সারি


def fetch_batch(config: str, offset: int, length: int = 100) -> list[dict]:
    for attempt in range(4):
        try:
            r = requests.get(API, params={
                "dataset": "wikimedia/wikipedia", "config": config,
                "split": "train", "offset": offset, "length": length}, timeout=45)
            if r.status_code == 200:
                rows = (r.json() or {}).get("rows") or []
                return [x.get("row") or {} for x in rows]
            if r.status_code in (429, 500, 502, 503):
                time.sleep(2 + attempt * 3)
                continue
            return []
        except Exception:  # noqa: BLE001
            time.sleep(2 + attempt * 2)
    return []


def harvest(config: str, lang: str, want: int) -> list[dict]:
    docs: list[dict] = []
    got = 0
    while len(docs) < want:
        # ক্রমিক + র্যান্ডম মিশিয়ে ভিন্ন ভিন্ন নিবন্ধ
        offset = random.randint(0, max(0, TOTAL.get(config, 100_000) - 120)) if got else 0
        rows = fetch_batch(config, offset, 100)
        got += 1
        if not rows:
            break
        for row in rows:
            text = crawler.clean_text(row.get("text") or "")
            if len(text) < 600:
                continue
            docs.append({
                "url": row.get("url") or f"hfwiki://{config}/{row.get('id')}",
                "title": (row.get("title") or "")[:250],
                "lang": lang,
                "site_domain": f"{lang}.wikipedia.org",
                "content": text[:MAX_DOC_CHARS],
            })
            if len(docs) >= want:
                break
        print(f"  [{config}] batch {got}: {len(docs)}/{want} নিবন্ধ", flush=True)
        time.sleep(0.4)
    return docs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bn", type=int, default=300)
    ap.add_argument("--en", type=int, default=200)
    args = ap.parse_args()

    docs: list[dict] = []
    if args.bn:
        docs += harvest("20231101.bn", "bn", args.bn)
    if args.en:
        docs += harvest("20231101.en", "en", args.en)
    print(f"মোট আনা হয়েছে: {len(docs)} নিবন্ধ")
    new = db.insert_documents(docs)
    print(f"নতুন সেভ হয়েছে: {'সব (offline)' if new is None else len(new)}")
    db.log_run("bulk-wiki", len(docs), sum(len(d['content'].split()) for d in docs),
               len(docs) if new is None else len(new), "HuggingFace wikimedia/wikipedia")


if __name__ == "__main__":
    main()
