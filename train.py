"""CLI: ইন্টারনেট থেকে শেখা + মডেল ট্রেইনিং।

ব্যবহার:
    python train.py --crawl 200        # ইন্টারনেট থেকে ~২০০ পেজ এনে শেখাবে
    python train.py --train            # data/corpus.jsonl থেকে মডেল বানাবে
    python train.py --ask "বাংলাদেশ"    # দ্রুত পরীক্ষা
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import crawler
import db
from config import CORPUS_PATH, MODEL_PATH
from ngmodel import NGramModel, perplexity
from sources import BIG_SITES, SOURCE_PLAN


def seed_sites() -> None:
    n = db.upsert_sites(BIG_SITES)
    print(f"[sites] {n} টি বড় সাইট Supabase-এ সিঙ্ক হয়েছে")


def crawl(total: int = 300, with_random: bool = False) -> list[dict]:
    """উইকিপিডিয়ার সম্পূর্ণ টপিক-তালিকা + সংবাদ ফিড থেকে লেখা আনে।"""
    docs: list[dict] = []

    # ১) উইকিপিডিয়া — সব গুরুত্বপূর্ণ টপিক (বাংলা + ইংরেজি)
    for lang in ("bn", "en"):
        topics = crawler.SEED_TOPICS[lang]
        t0 = time.time()
        got: list[dict] = []
        try:
            got = crawler.wiki_seeds(lang, len(topics), offset=0)
        except Exception as e:  # noqa: BLE001
            print(f"[crawl] wiki_{lang} failed: {e}")
        docs.extend(got)
        print(f"[crawl] wiki_{lang:2s} {len(got):3d}/{len(topics)} পেজ  ({time.time()-t0:.1f}s)")

    # ২) সংবাদ ফিড (সবশেষ খবর)
    news_budget = max(60, total - len(docs))
    feeds = ["bbc_bn", "prothomalo", "jagonews24", "bbc_en", "guardian",
             "aljazeera", "thedailystar", "techcrunch", "theverge",
             "nasa", "sciencedaily", "cricinfo"]
    per = max(8, news_budget // len(feeds))
    for key in feeds:
        cfg = SOURCE_PLAN[key]
        t0 = time.time()
        try:
            got = crawler.learn_feed(cfg["feed"], cfg["domain"], cfg["lang"], per)
        except Exception as e:  # noqa: BLE001
            print(f"[crawl] {key} failed: {e}")
            got = []
        docs.extend(got)
        print(f"[crawl] {key:13s} {len(got):3d}/{per:<3d} পেজ  ({time.time()-t0:.1f}s)")

    # ৩) ঐচ্ছিক: র‍্যান্ডম উইকি পেজ (বৈচিত্র্য)
    if with_random:
        for lang in ("bn", "en"):
            docs.extend(crawler.wiki_random(lang, 25))

    seen: set[str] = set()
    uniq = []
    for d in docs:
        if d["url"] in seen:
            continue
        seen.add(d["url"])
        uniq.append(d)
    return uniq


def train_from_corpus(log=print) -> dict:
    """data/corpus.jsonl থেকে মডেল বানিয়ে সেভ করে (সার্ভারের boot-এও ব্যবহৃত)।"""
    by_url: dict[str, str] = {}
    if CORPUS_PATH.exists():
        with CORPUS_PATH.open(encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    by_url[d["url"]] = d["content"]
                except (ValueError, KeyError):
                    continue
    if not by_url:
        return {}
    log(f"{len(by_url)} ডকুমেন্ট থেকে শিখছি…")
    m = NGramModel()
    for t in by_url.values():
        m.update(t)
    m.prune()
    m.save()
    st = m.stats()
    db.save_model_stats(st)
    return st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crawl", type=int, default=0)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--ask", type=str, default="")
    args = ap.parse_args()

    if args.crawl:
        seed_sites()
        t0 = time.time()
        docs = crawl(args.crawl)
        new = db.insert_documents(docs)
        fresh = docs if new is None else [d for d in docs if d["url"] in new]
        total_tokens = sum(len(d["content"].split()) for d in docs)
        db.log_run("initial-crawl", len(docs), total_tokens, len(fresh),
                   notes=f"Shobdo AI bootstrap crawl ({args.crawl} pages)")
        print(f"\n[crawl] মোট {len(docs)} পেজ, ~{total_tokens:,} শব্দ, "
              f"{len(fresh)} নতুন ডকুমেন্ট Supabase-এ সেভ — {time.time()-t0:.1f}s")

    if args.train or args.crawl:
        by_url: dict[str, str] = {}
        if CORPUS_PATH.exists():
            for line in CORPUS_PATH.open(encoding="utf-8"):
                try:
                    d = json.loads(line)
                    by_url[d["url"]] = d["content"]
                except (ValueError, KeyError):
                    continue
        texts = list(by_url.values())
        print(f"[train] {len(texts)} ডকুমেন্ট থেকে শিখছি…")
        t0 = time.time()
        m = NGramModel()
        for i, t in enumerate(texts, 1):
            m.update(t)
            if i % 50 == 0:
                print(f"  ... {i}/{len(texts)}", end="\r")
        m.prune()
        m.save()
        st = m.stats()
        db.save_model_stats(st)
        print(f"\n[train] টোকেন {st['tokens']:,} | শব্দভান্ডার {st['vocab']:,} | "
              f"bigram {st['bigrams']:,} | trigram {st['trigrams']:,} — {time.time()-t0:.1f}s")

        # সৎ হেল্ড-আউট মূল্যায়ন: প্রথম ৯০% দিয়ে মডেল, শেষ ১০% অদেখা ডকুমেন্টে পরীক্ষা
        if len(texts) >= 20:
            from ngmodel import tokenize as _tok
            cut = int(len(texts) * 0.9)
            me = NGramModel()
            for t in texts[:cut]:
                me.update(t)
            hit1 = hit5 = tot = 0
            for t in texts[cut:cut + 20]:
                tk = _tok(t)
                for i in range(2, len(tk) - 1, 4):
                    ctx = " ".join(tk[max(0, i - 6):i])
                    preds = [p["word"] for p in me.predict(ctx, k=5)]
                    tot += 1
                    if preds[:1] == [tk[i]]:
                        hit1 += 1
                    if tk[i] in preds:
                        hit5 += 1
            if tot:
                print(f"[train] হেল্ড-আউট নির্ভুলতা: top-1 {100*hit1/tot:.1f}% · "
                      f"top-5 {100*hit5/tot:.1f}%  ({tot} কনটেক্সট)")

    if args.ask:
        m = NGramModel.load()
        print("টোকেন:", m.tokens, "| প্রেডিকশন:")
        for p in m.predict(args.ask, k=8):
            print(f"  {p['word']:20s} {p['p']*100:5.2f}%")
        print("জেনারেট:", m.generate(args.ask, max_tokens=24))


if __name__ == "__main__":
    sys.exit(main())
