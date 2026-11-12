"""CLI: ইন্টারনেট থেকে শেখা + মডেল ট্রেইনিং।

ব্যবহার:
    python train.py --crawl 200        # ইন্টারনেট থেকে ~২০০ পেজ এনে শেখাবে
    python train.py --train            # data/corpus.jsonl থেকে মডেল বানাবে
    python train.py --ask "বাংলাদেশ"    # দ্রুত পরীক্ষা
    python train.py --sync --upload    # Supabase-এর সব ডেটা থেকে মডেল নতুন করে বানিয়ে Storage-এ তোলো

নিউরাল মডেলের ট্রেইনিং আলাদা: train_neural.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import crawler
import db
from config import CORPUS_PATH, LOW_MEM, MAX_ENTRIES, MODEL_PATH
from ngmodel import NGramModel
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


def iter_corpus():
    """data/corpus.jsonl থেকে একটা একটা করে লেখা (একই URL একবারই) — পুরো করপাস
    মেমরিতে তোলা হয় না, তাই ৫১২ MB হোস্টেও বড় করপাস থেকে মডেল গড়া যায়।"""
    seen: set[int] = set()
    if not CORPUS_PATH.exists():
        return
    with CORPUS_PATH.open(encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                url, content = d["url"], d["content"]
            except (ValueError, KeyError):
                continue
            h = hash(url)
            if h in seen or not content:
                continue
            seen.add(h)
            yield content


def load_corpus() -> list[str]:
    return list(iter_corpus())


def build_model(texts, low_mem: bool = LOW_MEM, log=None) -> NGramModel:
    """লেখা (list বা iterator) থেকে n-gram মডেল। low_mem হলে গড়ার সময়ই বাজেটের
    মধ্যে রাখা হয় (আগে পুরো মডেল মেমরিতে গড়ে পরে ছাঁটা হতো — ৫১২ MB হোস্টে ঝুঁকিপূর্ণ)।"""
    m = NGramModel()
    budget = MAX_ENTRIES or 400_000
    if low_mem:
        m.use_skip = False
    n = 0
    for n, t in enumerate(texts, 1):
        m.update(t)
        if low_mem and n % 150 == 0:
            m.fit_budget(budget * 2)
        if log and n % 500 == 0:
            log(f"  ... {n} ডকুমেন্ট")
    if not n:
        return m
    if low_mem:
        m.compact(max_vocab=30000, min_ctx=2, max_per_ctx=6, keep_skip=False)
        m.fit_budget(budget)
    else:
        m.prune()
    return m


def train_from_corpus(log=print) -> dict:
    """data/corpus.jsonl থেকে মডেল বানিয়ে সেভ করে (সার্ভারের boot-এও ব্যবহৃত)।"""
    log("করপাস থেকে শিখছি…")
    m = build_model(iter_corpus(), log=log)
    if not m.tokens:
        return {}
    m.save()
    st = m.stats()
    db.save_model_stats(st)
    return st


def rebuild_and_upload(log=print) -> bool:
    """পুরো করপাস থেকে দুই রকম মডেলই (পূর্ণ + কম-মেমরি) নতুন করে বানিয়ে Storage-এ তোলে।

    সার্ভার চালু হওয়ার সময় এই ফাইলই নামায় — তাই এটা চালালে লাইভ সাইট
    সর্বশেষ সব ডকুমেন্ট সহ (ও ঠিক করা টোকেনাইজার দিয়ে) শুরু হয়।
    """
    ok = True
    for low, name in ((True, "shobdo-model-low.pkl.gz"), (False, "shobdo-model.pkl.gz")):
        m = build_model(iter_corpus(), low_mem=low)
        if not m.tokens:
            log("করপাস খালি")
            return False
        path = MODEL_PATH.with_name(name)
        m.save(path)
        st = m.stats()
        log(f"{name}: টোকেন {st['tokens']:,} · bigram {st['bigrams']:,} · trigram {st['trigrams']:,} "
            f"· {path.stat().st_size / 1e6:.1f} MB")
        ok = db.upload_model(path, name) and ok
        del m
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crawl", type=int, default=0)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--ask", type=str, default="")
    ap.add_argument("--sync", action="store_true", help="আগে Supabase থেকে সব ডকুমেন্ট নামাও")
    ap.add_argument("--upload", action="store_true",
                    help="পূর্ণ + কম-মেমরি দুই মডেলই বানিয়ে Supabase Storage-এ তোলো")
    args = ap.parse_args()

    if args.sync:
        print(f"[sync] Supabase থেকে {db.download_corpus()} নতুন ডকুমেন্ট")

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

    if args.upload:
        ok = rebuild_and_upload(log=lambda m: print("[upload]", m))
        print("[upload] " + ("সফল ✅" if ok else "ব্যর্থ ❌ (Supabase কনফিগ দেখুন)"))
        if not ok:
            sys.exit(1)

    if args.train or args.crawl:
        texts = load_corpus()
        print(f"[train] {len(texts)} ডকুমেন্ট থেকে শিখছি…")
        t0 = time.time()
        m = build_model(texts)
        m.save()
        st = m.stats()
        db.save_model_stats(st)
        print(f"[train] টোকেন {st['tokens']:,} | শব্দভান্ডার {st['vocab']:,} | "
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
