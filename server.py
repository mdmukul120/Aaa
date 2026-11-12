"""Shobdo AI — ওয়েব সার্ভার (শুধু স্ট্যান্ডার্ড লাইব্রেরি)।

শুরু:  python server.py     →  http://localhost:8000
"""
from __future__ import annotations

import hmac
import json
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import crawler
import db
from config import (ADMIN_TOKEN, AUTOLEARN, CORPUS_PATH, LOW_MEM, MAX_DOCS, MAX_ENTRIES,
                    MODEL_PATH, NEURAL, NEURAL_BUDGET_MS, NEURAL_BUNDLED, NEURAL_META_NAME, NEURAL_NAME,
                    NEURAL_PATH, PORT, STATIC)
from ngmodel import NGramModel, rank_words
from sources import BIG_SITES, DOMAIN_FEEDS, SOURCE_PLAN

LOCK = threading.RLock()
_CORPUS_PENDING = False
REBUILT = {"yes": False}     # boot-এ মডেল নতুন করে গড়া হলে সেটা Storage-এ তুলে রাখা হয়


def _bootstrap() -> None:
    """মডেল/করপাস ফাইল না থাকলে Supabase থেকে রিস্টোর করে নিজেই ট্রেইন করে নেয়।

    ফ্রি হোস্টিংয়ে ফাইল সিস্টেম অস্থায়ী — কিন্তু সব ডেটা Supabase-এ আছে,
    তাই সার্ভার চালু হলেই নিজেকে আগের অবস্থায় ফিরিয়ে নেয়।
    """
    global _CORPUS_PENDING
    try:
        need_corpus = (not CORPUS_PATH.exists()) or CORPUS_PATH.stat().st_size < 500
        need_model = not MODEL_PATH.exists()
        if not need_model and not NGramModel.load().is_current():
            print("[boot] লোকাল মডেল পুরোনো টোকেনাইজারের — নতুন করে গড়া হবে")
            MODEL_PATH.unlink()
            need_model = True
        if not (need_corpus or need_model):
            return
        if not db.configured():
            print("[boot] Supabase কনফিগার নেই — লোকাল ডেটা দিয়েই চলবে")
            if need_model and not need_corpus:      # লোকাল করপাস আছে → তা থেকেই মডেল
                import train as _train
                st = _train.train_from_corpus(log=lambda m: print("[boot]", m))
                print(f"[boot] মডেল তৈরি: {st.get('tokens', 0):,} টোকেন")
            return
        print(f"[boot] রিস্টোর দরকার (corpus={need_corpus}, model={need_model}, low_mem={LOW_MEM})")
        if need_model:
            # ১) Supabase Storage-এ আগে থেকে বানানো মডেল থাকলে সোজা নামাও (দ্রুততম)
            fname = "shobdo-model-low.pkl.gz" if LOW_MEM else "shobdo-model.pkl.gz"
            if db.download_model(MODEL_PATH, fname):
                if NGramModel.load().is_current():
                    need_model = False
                    print("[boot] মডেল Storage থেকে লোড হয়েছে — ট্রেইনিং লাগবে না")
                else:
                    # পুরোনো টোকেনাইজারে বানানো ফাইল — করপাস থেকে নতুন করে গড়ে পরে Storage-এ তোলা হবে
                    print("[boot] Storage-এর মডেল পুরোনো টোকেনাইজারের — করপাস থেকে নতুন করে গড়া হচ্ছে")
                    MODEL_PATH.unlink()
                    REBUILT["yes"] = True
        if need_corpus and need_model:
            # মডেলও নেই — তখন ট্রেইন করার জন্য করপাস দরকার, তাই ব্লক করে নামাই
            n = db.download_corpus()
            print(f"[boot] Supabase থেকে {n} ডকুমেন্ট রিস্টোর হয়েছে")
            _CORPUS_PENDING = False
        elif need_corpus:
            # মডেল Storage থেকে পাওয়া গেছে → করপাস পরে ব্যাকগ্রাউন্ডে নামবে
            _CORPUS_PENDING = True
        if need_model and CORPUS_PATH.exists():
            import train as _train
            st = _train.train_from_corpus(log=lambda m: print("[boot]", m))
            print(f"[boot] মডেল তৈরি: {st.get('tokens', 0):,} টোকেন")
            REBUILT["yes"] = bool(st)
    except Exception as e:  # noqa: BLE001
        print(f"[boot] রিস্টোর ব্যর্থ: {e}")


_bootstrap()
MODEL = NGramModel.load()
STORAGE_MODEL_NAME = "shobdo-model-low.pkl.gz" if LOW_MEM else "shobdo-model.pkl.gz"

# --------------------------------------------------------------------------- #
# নিউরাল মডেল (ছোট ট্রান্সফরমার) — থাকলে n-gram-এর সাথে মিশিয়ে ব্যবহার হয়
# --------------------------------------------------------------------------- #
NEURAL_LM = None            # neural.NeuralLM | None
NEURAL_INFO = {"file": None, "error": None}


def _neural_candidates() -> list:
    """সবচেয়ে নতুন ট্রেইন করা ফাইল আগে (Storage থেকে নামানো বনাম রিপোতে রাখা)।"""
    from neural import NeuralLM
    found = []
    for path in (NEURAL_PATH, NEURAL_BUNDLED):
        if path.exists():
            try:
                found.append((NeuralLM.read_meta(path).get("trained_at") or "", path))
            except Exception as e:  # noqa: BLE001
                print(f"[neural] {path.name} পড়া যায়নি: {e}")
    return [p for _, p in sorted(found, key=lambda x: x[0], reverse=True)]


def _load_neural() -> bool:
    global NEURAL_LM
    if not NEURAL:
        return False
    try:
        from neural import NeuralLM
        for path in _neural_candidates():
            try:
                lm = NeuralLM.load(path)
                lm.next_words("পরীক্ষা test")          # একবার চালিয়ে দেখা — ভাঙা ফাইল ধরা পড়ে
            except Exception as e:  # noqa: BLE001
                print(f"[neural] {path.name} লোড ব্যর্থ: {e}")
                continue
            NEURAL_LM = lm
            NEURAL_INFO.update({"file": path.name if path == NEURAL_BUNDLED else "storage:" + path.name,
                                "error": None})
            st = lm.stats()
            print(f"[neural] লোড হয়েছে: {path.name} · {st['params']:,} প্যারামিটার · "
                  f"ট্রেইন {st.get('trained_at')}")
            return True
        NEURAL_INFO["error"] = "ওজন ফাইল নেই"
    except Exception as e:  # noqa: BLE001  (numpy/sentencepiece না থাকলেও সার্ভার চলবে)
        NEURAL_INFO["error"] = str(e)
        print(f"[neural] বন্ধ — n-gram দিয়ে চলছে: {e}")
    return False


def _neural_sync_loop() -> None:
    """Storage-এ নতুনতর নিউরাল মডেল এলে (GitHub Actions ট্রেইন করে তোলে) নামিয়ে বদলে নেয়।"""
    while True:
        try:
            if NEURAL and db.configured():
                remote = db.fetch_json(NEURAL_META_NAME) or {}
                have = (NEURAL_LM.meta.get("trained_at") if NEURAL_LM else "") or ""
                if (remote.get("trained_at") or "") > have:
                    print(f"[neural] Storage-এ নতুন মডেল ({remote.get('trained_at')}) — নামানো হচ্ছে")
                    if db.download_model(NEURAL_PATH, NEURAL_NAME, min_size=100_000):
                        _load_neural()
        except Exception as e:  # noqa: BLE001
            print("[neural] sync:", e)
        time.sleep(1800)


# ছোট CPU-তে (Render ফ্রি = ০.১ CPU) নিউরাল হিসাব জমে গিয়ে সার্ভার আটকে না যায়:
#   • একসাথে সর্বোচ্চ ২টা নিউরাল হিসাব; ব্যস্ত থাকলে সেই অনুরোধ n-gram দিয়ে উত্তর পায়
#   • গড় সময় NEURAL_BUDGET_MS ছাড়ালে beam ছোট করা হয় (কম খরচ, সামান্য কম নির্ভুল)
NEURAL_SEM = threading.BoundedSemaphore(2)
NEURAL_SPEED = {"ema": 0.0, "lite": False}


def _neural_words(lm, text: str) -> dict[str, float] | None:
    if not NEURAL_SEM.acquire(timeout=0.25):
        return None
    try:
        t0 = time.perf_counter()
        if NEURAL_SPEED["lite"]:
            out = lm.next_words(text, beam=4, max_pieces=4, first=12)
        else:
            out = lm.next_words(text)
        ms = (time.perf_counter() - t0) * 1000
        ema = NEURAL_SPEED["ema"] = 0.85 * NEURAL_SPEED["ema"] + 0.15 * ms
        if ema > NEURAL_BUDGET_MS:
            NEURAL_SPEED["lite"] = True
        elif ema < NEURAL_BUDGET_MS / 3:
            NEURAL_SPEED["lite"] = False
        return out
    finally:
        NEURAL_SEM.release()


def _suggest(text: str, k: int = 6) -> tuple[list[dict], str]:
    """পরের শব্দ: নিউরাল + n-gram মিশ্রণ (নিউরাল না থাকলে/ব্যস্ত থাকলে শুধু n-gram)।"""
    ng = MODEL.word_probs(text)
    lm = NEURAL_LM
    if lm is not None:
        try:
            from neural import blend
            nw = _neural_words(lm, text)
            if nw is not None:
                # n-gram প্রায় খালি হলে (নতুন ইনস্টল) পুরোটাই নিউরাল
                w = lm.blend_w if ng else 1.0
                return rank_words(blend(nw, ng, w), k), "neural+ngram" if ng else "neural"
        except Exception as e:  # noqa: BLE001
            print("[neural] predict ব্যর্থ:", e)
    return rank_words(ng, k), "ngram"

STATUS = {
    "running": False, "mode": "", "target": 0, "done": 0,
    "log": [], "started": 0.0, "last_result": None, "error": None,
}

PRED_BUFFER: list[dict] = []
PRED_LOCK = threading.Lock()
AUTO = {"on": AUTOLEARN, "interval": 120, "last": 0.0}


# --------------------------------------------------------------------------- #
def _log(msg: str) -> None:
    with LOCK:
        STATUS["log"] = (STATUS["log"] + [f"{time.strftime('%H:%M:%S')}  {msg}"])[-40:]
    print("[learn]", msg)


# boot-এ নতুন করে গড়া মডেল ~১ মিনিটের মধ্যেই Storage-এ ওঠে (পরের বার আর গড়তে হয় না)
UPLOAD = {"dirty": REBUILT["yes"], "last": 0.0 if REBUILT["yes"] else time.time()}
UPLOAD_EVERY = 600 if LOW_MEM else 1800
SEEN_URLS: set[str] = set()      # এই প্রসেসে ইতিমধ্যে শেখা URL (একই লেখা দু'বার না শেখার জন্য)


def _train_docs(docs: list[dict]) -> int:
    added_tokens = 0
    with LOCK:
        for d in docs:
            added_tokens += MODEL.update(d["content"])
            SEEN_URLS.add(d["url"])
        MODEL.prune()
        if MAX_ENTRIES:
            # বাজেট ছাড়ালে তবেই ছাঁটা — নতুন শেখা শব্দজোড়া আর সঙ্গে সঙ্গে মুছে যায় না
            MODEL.fit_budget(MAX_ENTRIES)
        MODEL.save()
        UPLOAD["dirty"] = True
    return added_tokens


def _upload_loop() -> None:
    """শেখা মডেল Supabase Storage-এ তুলে রাখে — হোস্ট ঘুমিয়ে/রিস্টার্ট হলেও শেখা হারায় না।"""
    while True:
        time.sleep(60)
        try:
            if (UPLOAD["dirty"] and db.configured() and MODEL_PATH.exists()
                    and time.time() - UPLOAD["last"] > UPLOAD_EVERY):
                UPLOAD["dirty"] = False
                UPLOAD["last"] = time.time()
                if not db.upload_model(MODEL_PATH, STORAGE_MODEL_NAME):
                    UPLOAD["dirty"] = True
        except Exception as e:  # noqa: BLE001
            print("[upload]", e)


def _learn_job(mode: str, count: int, url: str | None) -> None:
    status_result = {"pages": 0, "tokens": 0, "docs_added": 0, "samples": []}
    try:
        with LOCK:
            STATUS.update({"running": True, "mode": mode, "target": count, "done": 0,
                           "log": [], "started": time.time(), "error": None,
                           "last_result": None})
        if mode not in ("test", "supabase") and _doc_cap_reached():
            raise RuntimeError(f"ডেটাবেসে {MAX_DOCS:,} ডকুমেন্টের সীমা পূর্ণ — নতুন শেখা বন্ধ "
                               "(MAX_DOCS বাড়ান বা পুরোনো ডেটা মুছুন)")
        _log("শুরু: ইন্টারনেট থেকে ডেটা আনা হচ্ছে…")
        docs: list[dict] = []

        if mode == "url" and url:
            d = crawler.learn_url(url)
            docs = [d] if d else []
            _log(f"URL পড়া হয়েছে: {url[:70]}")
        elif mode == "site" and url:
            host = urlparse(url).netloc.replace("www.", "")
            feeds = DOMAIN_FEEDS.get(host) or []
            if feeds:
                per = max(3, count // len(feeds))
                for feed, flang in feeds:
                    got = crawler.learn_feed(feed, host, flang, per)
                    docs += got
                    _log(f"{host} ({flang}): {len(got)} পেজ — ফিড থেকে")
            else:
                d = crawler.learn_url(url)
                docs = [d] if d else []
                _log(f"{host}: সরাসরি পেজ পড়া হয়েছে ({len(docs)})")
        elif mode == "auto":
            plan = [("wiki_bn", max(3, count // 4)), ("wiki_en", max(3, count // 4)),
                    ("bbc_bn", max(1, count // 10)), ("prothomalo", max(1, count // 10)),
                    ("dw_bn", max(1, count // 12)),
                    ("bbc_en", max(1, count // 10)), ("guardian", max(1, count // 10)),
                    ("techcrunch", max(1, count // 12))]
            for key, want in plan:
                cfg = SOURCE_PLAN[key]
                if cfg["kind"] == "wiki":
                    got = crawler.wiki_seeds(cfg["lang"], want, offset=int(time.time()) % 40)
                    if len(got) < want:
                        got += crawler.wiki_random(cfg["lang"], want - len(got))
                else:
                    got = crawler.learn_feed(cfg["feed"], cfg["domain"], cfg["lang"], want)
                docs.extend(got)
                _log(f"{key}: {len(got)} পেজ")
        elif mode.startswith("wiki"):
            lang = "bn" if mode.endswith("bn") else "en"
            topics_len = len(crawler.SEED_TOPICS[lang])
            docs = crawler.wiki_seeds(lang, min(max(3, count // 2), topics_len),
                                      offset=int(time.time()) % topics_len)
            docs += crawler.wiki_random(lang, max(3, count // 2))
            _log(f"উইকিপিডিয়া ({lang}) থেকে {len(docs)} নিবন্ধ")
        elif mode.startswith("news"):
            lang = "bn" if mode.endswith("bn") else "en"
            feeds = (["bbc_bn", "prothomalo", "jagonews24", "dw_bn"] if lang == "bn"
                     else ["bbc_en", "guardian", "aljazeera", "thedailystar", "techcrunch"])
            per = max(2, count // len(feeds))
            for key in feeds:
                cfg = SOURCE_PLAN[key]
                got = crawler.learn_feed(cfg["feed"], cfg["domain"], cfg["lang"], per)
                docs.extend(got)
                _log(f"{key}: {len(got)} পেজ")
        elif mode == "test":
            docs = _test_docs(count)
            _log(f"অফলাইন ডেমো কর্পাস: {len(docs)} পেজ")
        elif mode == "supabase":
            docs = _docs_from_supabase(count)
            _log(f"Supabase থেকে {len(docs)} ডকুমেন্ট")
        else:
            raise ValueError("অজানা mode")

        if not docs:
            raise RuntimeError("কোনো পেজ পাওয়া যায়নি — ফিড/নেটওয়ার্ক চেক করুন")

        if mode in ("supabase", "test"):
            fresh, added = docs, 0
        else:
            new = db.insert_documents(docs)
            fresh = ([d for d in docs if d["url"] not in SEEN_URLS] if new is None
                     else [d for d in docs if d["url"] in new])
            added = len(fresh)
            _log(f"{len(docs)} পেজ পাওয়া গেছে · {added} নতুন (বাকিগুলো আগেই শেখা)")

        if not fresh:
            # ফিডে নতুন কিছু নেই → উইকিপিডিয়া থেকে নতুন লেখা এনে শেখানো
            extra = crawler.wiki_random("bn", 6) + crawler.wiki_random("en", 6)
            new2 = db.insert_documents(extra)
            extra_fresh = ([d for d in extra if d["url"] not in SEEN_URLS] if new2 is None
                           else [d for d in extra if d["url"] in new2])
            if extra_fresh:
                docs += extra_fresh
                fresh = extra_fresh
                added += len(extra_fresh)
                _log("ফিডে নতুন কিছু ছিল না — উইকিপিডিয়া থেকে নতুন লেখা যোগ হলো")

        tokens = _train_docs(fresh) if fresh else 0
        db.log_run(mode, len(docs), tokens, added, f"UI learn · {mode}")
        st = MODEL.stats()
        db.save_model_stats(st)

        status_result = {
            "pages": len(docs), "tokens": tokens, "docs_added": added,
            "dup": len(docs) - added,
            "samples": [{"title": d.get("title") or d["url"][:60],
                         "url": d["url"], "lang": d.get("lang", "?"),
                         "chars": len(d["content"])} for d in docs[:12]],
        }
        with LOCK:
            STATUS["last_result"] = status_result
        if fresh:
            _log(f"শেষ: {len(fresh)} নতুন পেজ · ~{tokens:,} টোকেন · মডেল আপডেট হয়েছে ✅")
        else:
            _log("শেষ: নতুন পেজ পাওয়া যায়নি (সব আগেই শেখা ছিল) — ‘মিশ্র’ বা অন্য উৎস চেষ্টা করুন")
    except Exception as e:  # noqa: BLE001
        with LOCK:
            STATUS["error"] = str(e)
        _log(f"ত্রুটি: {e}")
    finally:
        with LOCK:
            STATUS["running"] = False
            STATUS["done"] = STATUS["target"]


TEST_DOCS = [
    ("বাংলাদেশ দক্ষিণ এশিয়ার একটি জনবহুল রাষ্ট্র। এর রাজধানী ঢাকা এবং সরকারি ভাষা বাংলা। "
     "পদ্মা, মেঘনা ও যমুনা বাংলাদেশের প্রধান নদী। কৃষি, তৈরি পোশাক শিল্প ও প্রবাসী আয় দেশের অর্থনীতির মূল ভিত্তি। "
     "সুন্দরবন বিশ্বের বৃহত্তম ম্যানগ্রোভ বন এবং রয়েল বেঙ্গল টাইগারের আবাসস্থল। "
     "চট্টগ্রাম দেশের প্রধান বন্দর শহর। ঢাকা মেগাসিটি প্রতিদিন নতুন মানুষের ভিড়ে বেড়ে চলেছে। "
     "বাংলা সাহিত্যের ইতিহাস হাজার বছরের পুরোনো। রবীন্দ্রনাথ ঠাকুর ও কাজী নজরুল ইসলাম বাংলা ভাষার শ্রেষ্ঠ কবি। "
     "মুক্তিযুদ্ধের মাধ্যমে ১৯৭১ সালে বাংলাদেশ স্বাধীন হয়। জাতীয় সংসদ ভবন রাজধানী ঢাকায় অবস্থিত। "
     "প্রতি বছর পদ্মা সেতু দিয়ে হাজার হাজার যানবাহন চলাচল করে। "
     "শিক্ষার হার বাড়ার সাথে সাথে তথ্যপ্রযুক্তি খাতে নতুন কর্মসংস্থান তৈরি হচ্ছে। "
     "মাছ, ভাত ও ডাল বাংলাদেশের মানুষের প্রতিদিনের খাবারের প্রধান অংশ। "
     "আবহাওয়া গ্রীষ্মমন্ডলীয়, বর্ষাকালে ভারী বৃষ্টি হয় এবং নদীতে বন্যা দেখা দেয়।", "bn"),
    ("The river is a natural flowing watercourse that usually flows into an ocean, sea, lake or another river. "
     "Rivers are part of the water cycle and carry water and nutrients across the landscape. "
     "The Amazon is the largest river in the world by discharge. The Nile has historically been considered the longest river. "
     "Rivers provide drinking water, irrigation, transport and hydroelectric power for millions of people. "
     "Scientists study river ecosystems because they support rich biodiversity. "
     "A delta forms where a river deposits sediment as it enters a slower body of water. "
     "The Ganges and the Brahmaputra form the largest delta in the world. "
     "Pollution and climate change threaten many rivers around the globe. "
     "Cities were often founded on river banks because of trade and fresh water. "
     "Engineers build dams and canals to control floods and to store water for the dry season.", "en"),
    ("Artificial intelligence is the intelligence demonstrated by machines, in contrast to intelligence of humans and animals. "
     "Machine learning algorithms build models based on training data and improve with experience. "
     "A language model predicts the next word or token given the preceding context. "
     "Large language models are trained on enormous text corpora collected from the internet. "
     "Natural language processing helps computers understand and generate human language. "
     "Deep learning uses neural networks with many layers of representation. "
     "Researchers measure progress with benchmarks such as perplexity and accuracy. "
     "Data quality matters as much as model size for strong performance. "
     "Artificial intelligence is used in search engines, translation, healthcare and robotics. "
     "Ethical questions about bias, privacy and safety are widely debated today.", "en"),
    ("ইন্টারনেট হলো পরস্পর সংযুক্ত কম্পিউটার নেটওয়ার্কের একটি বিশ্বব্যাপী ব্যবস্থা। "
     "ওয়েবসাইট, ইমেইল, ভিডিও এবং সামাজিক যোগাযোগ মাধ্যম ইন্টারনেটের সেবা। "
     "ডেটা ছোট ছোট প্যাকেটে ভাগ হয়ে রাউটারের মাধ্যমে গন্তব্যে পৌঁছায়। "
     "প্রোটোকল হলো এমন নিয়ম যা কম্পিউটারগুলোকে কথা বলতে সাহায্য করে। "
     "সার্চ ইঞ্জিন ওয়েব পেজ সংগ্রহ করে এবং প্রশ্নের উত্তর খুঁজে দেয়। "
     "ক্লাউড কম্পিউটিং ইন্টারনেটের মাধ্যমে স্টোরেজ ও কম্পিউটিং শক্তি দেয়। "
     "সাইবার নিরাপত্তা ব্যক্তিগত তথ্যকে চুরি থেকে রক্ষা করে। "
     "মোবাইল ইন্টারনেটের বিস্তারে গ্রামেও ডিজিটাল সেবা পৌঁছে গেছে। "
     "বাংলা ভাষায়ও এখন হাজারো অনলাইন সংবাদপত্র ও ব্লগ পাওয়া যায়। "
     "কৃত্রিম বুদ্ধিমত্তা ইন্টারনেটের বিপুল লেখা থেকে ভাষা শিখে নিতে পারে।", "bn"),
]


def _test_docs(count: int) -> list[dict]:
    out = []
    for i in range(count):
        text, lang = TEST_DOCS[i % len(TEST_DOCS)]
        out.append({"url": f"offline://demo/{i}", "title": f"অফলাইন নমুনা {i+1}",
                    "lang": lang, "site_domain": "offline.demo",
                    "content": text + "\n\n" + text})
    return out


def _docs_from_supabase(count: int) -> list[dict]:
    docs = []
    for row in db.recent_documents(limit=min(count, 50)):
        if row.get("url", "").startswith("http"):
            d = crawler.learn_url(row["url"])
            if d:
                docs.append(d)
    return docs


# --------------------------------------------------------------------------- #
_CAP = {"t": 0.0, "hit": False}


def _doc_cap_reached() -> bool:
    """ফ্রি ১ GB ডেটাবেস রক্ষা: ডকুমেন্ট সংখ্যা MAX_DOCS ছাড়ালে আর আনা হয় না।"""
    if time.time() - _CAP["t"] > 300:
        _CAP["t"] = time.time()
        try:
            _CAP["hit"] = db.doc_count() >= MAX_DOCS
        except Exception:  # noqa: BLE001
            pass
    return _CAP["hit"]


def _auto_loop() -> None:
    while True:
        time.sleep(20)
        try:
            if AUTO["on"] and not STATUS["running"] and time.time() - AUTO["last"] > AUTO["interval"]:
                AUTO["last"] = time.time()
                if _doc_cap_reached():
                    AUTO["on"] = False
                    print(f"[auto] {MAX_DOCS:,} ডকুমেন্টের সীমা পূর্ণ — অটো-লার্ন বন্ধ করা হলো")
                    continue
                _learn_job("auto", 12, None)
        except Exception as e:  # noqa: BLE001
            print("[auto]", e)


def _flush_predictions() -> None:
    while True:
        time.sleep(45)
        with PRED_LOCK:
            rows, PRED_BUFFER[:] = PRED_BUFFER[-20:], []
        if rows:
            db.log_predictions(rows)


# --------------------------------------------------------------------------- #
_TTL: dict[str, tuple[float, object]] = {}


def _cached(key: str, ttl: float, fn):
    """ছোট TTL ক্যাশ — প্রতিটি রিকোয়েস্টে Supabase-এ না গিয়ে।"""
    hit = _TTL.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    val = fn()
    _TTL[key] = (time.time(), val)
    return val


def api_stats() -> dict:
    st = MODEL.stats()
    lm = NEURAL_LM
    return {
        "model": st,
        "neural": ({**lm.stats(), "file": NEURAL_INFO["file"]} if lm is not None else None),
        "neural_error": None if lm is not None else NEURAL_INFO["error"],
        "engine": "neural+ngram" if lm is not None else "ngram",
        "db": _cached("usage", 20, db.db_usage),
        "supabase": db.configured(),
        "sites": len(BIG_SITES),
        "sites_registry": BIG_SITES,
        "runs": _cached("runs", 20, lambda: db.list_runs(5)),
        "learning": {k: STATUS[k] for k in ("running", "mode", "target", "done", "log", "error", "last_result")},
        "autolearn": AUTO["on"],
        "admin_required": bool(ADMIN_TOKEN),
        "max_docs": MAX_DOCS,
        "model_file_mb": round(MODEL_PATH.stat().st_size / 1e6, 2) if MODEL_PATH.exists() else 0,
    }


def api_predict(text: str) -> dict:
    t0 = time.perf_counter()
    preds, engine = _suggest(text, k=6)
    ms = round((time.perf_counter() - t0) * 1000, 2)
    if preds:
        with PRED_LOCK:
            PRED_BUFFER.append({"input": text[-200:], "suggestion": preds[0]["word"],
                                "probability": preds[0]["p"]})
    return {"suggestions": preds, "ms": ms, "engine": engine, "doc_ctx": MODEL.tokens}


def api_generate(text: str, temperature: float = 0.85, max_tokens: int = 28,
                 engine: str = "auto") -> dict:
    t0 = time.perf_counter()
    max_tokens = max(1, min(int(max_tokens), 60))
    temperature = max(0.0, min(float(temperature), 2.0))
    out, used = "", "ngram"
    lm = NEURAL_LM
    if lm is not None and engine != "ngram" and NEURAL_SEM.acquire(timeout=3):
        try:
            out, used = lm.generate(text, max_words=max_tokens, temperature=temperature), "neural"
        except Exception as e:  # noqa: BLE001
            print("[neural] generate ব্যর্থ:", e)
        finally:
            NEURAL_SEM.release()
    if not out:
        out, used = MODEL.generate(text, max_tokens=max_tokens, temperature=temperature), "ngram"
    return {"continuation": out, "engine": used,
            "ms": round((time.perf_counter() - t0) * 1000, 1)}


def _learn_from_ask(q: str, fresh: list[dict], pages: int) -> None:
    try:
        tokens = _train_docs(fresh)
        db.log_run("ask", pages, tokens, len(fresh), f"ask: {q}")
    except Exception as e:  # noqa: BLE001
        print("[ask] learn failed:", e)


def api_ask(q: str) -> dict:
    from ngmodel import tokenize as _tok

    q = q.strip()[:300]
    local = db.search_documents(q, limit=6)
    web: list[dict] = []
    learned = 0
    is_bn = any("\u0980" <= ch <= "\u09FF" for ch in q)
    for lang in (("bn", "en") if is_bn else ("en", "bn")):
        try:
            hits = crawler.wiki_search(q, lang, limit=2)
        except Exception:  # noqa: BLE001
            hits = []
        for h in hits:
            if h["url"] not in {w["url"] for w in web}:
                web.append(h)

    # প্রশ্নের সাথে সবচেয়ে মিলে যাওয়া সোর্স আগে
    qwords = {w.lower() for w in _tok(q) if len(w) > 1}
    def _rel(h: dict) -> float:
        tw = {w.lower() for w in _tok((h.get("title") or "") + " " + h.get("content", "")[:800])}
        return (len(qwords & tw) / (len(qwords) or 1)) + (0.5 if h.get("lang") == ("bn" if is_bn else "en") else 0)
    web.sort(key=_rel, reverse=True)
    to_learn = [h for h in web if len(h["content"]) > 300]
    if to_learn:
        # আগে সেভ, তারপর শুধু সত্যিই-নতুন পেজ থেকে শেখা — একই প্রশ্ন বারবার করলে
        # একই লেখা বারবার গুনে মডেল বিকৃত হতো।
        new = db.insert_documents(to_learn)
        fresh = [h for h in to_learn
                 if (h["url"] in new if new is not None else h["url"] not in SEEN_URLS)]
        SEEN_URLS.update(h["url"] for h in to_learn)
        learned = len(fresh)
        if fresh:      # ট্রেইনিং + সেভ ব্যাকগ্রাউন্ডে — উত্তর দেরি হয় না
            threading.Thread(target=_learn_from_ask, args=(q, fresh, len(to_learn)),
                             daemon=True).start()
    summary = ""
    if web:
        summary = web[0]["content"][:520].strip()
        if not summary.endswith(("।", ".", "!", "?")):
            summary += "…"
    return {"query": q, "summary": summary, "sources": web[:4],
            "local": local[:6], "learned": learned}


def api_search(q: str) -> dict:
    return {"query": q, "results": db.search_documents(q, limit=10)}


# --------------------------------------------------------------------------- #
# নিরাপত্তা: রেট-লিমিট + ঐচ্ছিক অ্যাডমিন টোকেন
# --------------------------------------------------------------------------- #
MAX_BODY = 64 * 1024

# রুট → (প্রতি IP প্রতি মিনিটে, সবাই মিলে প্রতি মিনিটে)
LIMITS = {
    "/api/predict": (300, 3000),
    "/api/generate": (40, 400),
    "/api/search": (60, 600),
    "/api/ask": (12, 40),
    "/api/learn": (6, 12),
    "/api/autolearn": (6, 12),
}


class RateLimiter:
    def __init__(self) -> None:
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()

    def allow(self, key: str, limit: int, window: float = 60.0) -> bool:
        now = time.time()
        with self._lock:
            if len(self._hits) > 20000:          # মেমরি সীমিত রাখা
                self._hits.clear()
            q = self._hits.setdefault(key, deque())
            while q and now - q[0] > window:
                q.popleft()
            if len(q) >= limit:
                return False
            q.append(now)
            return True


RATE = RateLimiter()
ADMIN_ROUTES = ("/api/learn", "/api/autolearn")
LEARN_MODES = ("auto", "url", "site", "wiki_bn", "wiki_en", "news_bn", "news_en", "test", "supabase")
_PING = {"t": 0.0, "ok": False}


def _db_alive() -> bool:
    """/healthz প্রতি কয়েক সেকেন্ডে ডাকা হয় — প্রতিবার Supabase-এ না গিয়ে ৬০ সেকেন্ড ক্যাশ।"""
    if time.time() - _PING["t"] > 60:
        _PING["t"] = time.time()
        _PING["ok"] = db.ping()
    return _PING["ok"]


class Handler(BaseHTTPRequestHandler):
    server_version = "ShobdoAI/2.0"

    def log_message(self, fmt: str, *args) -> None:  # noqa: D401
        if "/healthz" not in (args[0] if args else ""):
            print("[http]", self.address_string(), fmt % args)

    # ---------------- helpers ----------------
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        # Vercel/GitHub Pages-এর মতো আলাদা ডোমেইনের UI থেকে কল করার জন্য
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "content-type, x-admin-token")
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _body(self) -> dict | None:
        """JSON বডি; খুব বড় হলে None (৪১৩)।"""
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        if n > MAX_BODY:
            return None
        try:
            raw = self.rfile.read(n) if n > 0 else b"{}"
            data = json.loads(raw.decode("utf-8")) if raw else {}
            return data if isinstance(data, dict) else {}
        except (ValueError, UnicodeDecodeError):
            return {}

    def _client(self) -> str:
        fwd = (self.headers.get("X-Forwarded-For") or "").split(",")[0].strip()
        return fwd or self.client_address[0]

    def _limited(self, path: str, scope: str = "both") -> bool:
        """scope: "ip" = শুধু প্রতি-IP সীমা, "all" = শুধু সামগ্রিক সীমা, "both" = দুটোই।

        অ্যাডমিন রুটে আগে প্রতি-IP সীমা → টোকেন যাচাই → তারপর সামগ্রিক সীমা, যাতে
        টোকেন ছাড়া কেউ অনুরোধ পাঠিয়ে সামগ্রিক কোটা শেষ করে অ্যাডমিনকে আটকাতে না পারে।
        """
        per_ip, overall = LIMITS.get(path, (0, 0))
        if not per_ip:
            return False
        ok = True
        if scope in ("ip", "both"):
            ok = RATE.allow(f"{path}|{self._client()}", per_ip)
        if ok and scope in ("all", "both"):
            ok = RATE.allow(f"{path}|*", overall)
        if ok:
            return False
        self._json({"error": "খুব ঘন ঘন অনুরোধ — একটু পরে আবার চেষ্টা করুন"}, 429)
        return True

    def _admin_ok(self) -> bool:
        if not ADMIN_TOKEN:
            return True
        got = self.headers.get("X-Admin-Token") or ""
        if hmac.compare_digest(got.encode("utf-8"), ADMIN_TOKEN.encode("utf-8")):
            return True
        self._json({"error": "অ্যাডমিন টোকেন লাগবে", "admin_required": True}, 401)
        return False

    # ---------------- routes ----------------
    def do_GET(self) -> None:  # noqa: N802
        u = urlparse(self.path)
        path = u.path
        if path in ("/", "/index.html"):
            f = STATIC / "index.html"
            if f.exists():
                self._send(200, f.read_bytes(), "text/html; charset=utf-8")
            else:
                self._send(500, b"index.html missing", "text/plain")
            return
        if path == "/healthz":
            self._json({"ok": True, "tokens": MODEL.tokens, "db": _db_alive(),
                        "engine": "neural+ngram" if NEURAL_LM is not None else "ngram"})
            return
        if path == "/api/stats":
            self._json(api_stats())
            return
        if path == "/api/pages":
            self._json({"pages": _cached("pages", 10, lambda: db.recent_documents(limit=15))})
            return
        if path == "/api/learn/status":
            snap = {k: STATUS[k] for k in
                    ("running", "mode", "target", "done", "log", "error", "last_result")}
            snap["log"] = list(snap["log"])
            self._json(snap)
            return
        if path == "/api/sites":
            self._json({"sites": BIG_SITES})
            return
        if path == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
            return
        qs = parse_qs(u.query)
        if path == "/api/search":
            if self._limited(path):
                return
            self._json(api_search((qs.get("q") or [""])[0][:300]))
            return
        self._json({"error": "not found"}, 404)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send(204, b"", "text/plain")

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        body = self._body()
        if body is None:
            self._json({"error": "অনুরোধ খুব বড়"}, 413)
            return
        admin_route = path in ADMIN_ROUTES
        if self._limited(path, "ip" if admin_route else "both"):
            return
        if admin_route and (not self._admin_ok() or self._limited(path, "all")):
            return
        try:
            if path == "/api/predict":
                self._json(api_predict(str(body.get("text", ""))[-2000:]))
            elif path == "/api/generate":
                self._json(api_generate(
                    str(body.get("text", ""))[-1000:],
                    float(body.get("temperature", 0.85)),
                    int(body.get("max_tokens", 28)),
                    str(body.get("engine", "auto"))))
            elif path == "/api/ask":
                self._json(api_ask(str(body.get("q", ""))))
            elif path == "/api/search":
                self._json(api_search(str(body.get("q", ""))[:300]))
            elif path == "/api/learn":
                mode = str(body.get("mode", "auto"))
                if mode not in LEARN_MODES:
                    self._json({"ok": False, "error": "অজানা mode"}, 400)
                    return
                count = max(1, min(int(body.get("count", 15)),
                                   8 if LOW_MEM else 40))
                url = str(body.get("url") or "")[:500] or None
                if url and not crawler.is_public_url(url):
                    self._json({"ok": False, "error": "এই ঠিকানা অনুমোদিত নয় — শুধু পাবলিক http/https লিংক"}, 400)
                    return
                with LOCK:
                    if STATUS["running"]:
                        self._json({"ok": False, "error": "ইতিমধ্যে একটা লার্নিং চলছে"}, 409)
                        return
                    STATUS["running"] = True
                threading.Thread(target=_learn_job, args=(mode, count, url), daemon=True).start()
                self._json({"ok": True, "started": mode, "count": count})
            elif path == "/api/autolearn":
                AUTO["on"] = bool(body.get("on"))
                self._json({"ok": True, "autolearn": AUTO["on"]})
            else:
                self._json({"error": "not found"}, 404)
        except (TypeError, ValueError) as e:
            self._json({"error": f"ভুল ইনপুট: {e}"}, 400)
        except Exception as e:  # noqa: BLE001
            print("[http] error:", repr(e))
            self._json({"error": str(e)}, 500)


def main() -> None:
    print(f"শব্দ AI — টোকেন লোড: {MODEL.tokens:,} | Supabase: {'ON' if db.configured() else 'OFF'}"
          f" | LOW_MEM: {LOW_MEM} | অ্যাডমিন টোকেন: {'আছে' if ADMIN_TOKEN else 'নেই (খোলা)'}")
    _load_neural()
    try:     # Supabase সাময়িক বন্ধ থাকলেও সার্ভার চালু হবে
        db.upsert_sites(BIG_SITES)
    except Exception as e:  # noqa: BLE001
        print("[boot] sites সিঙ্ক ব্যর্থ (পরে আবার হবে):", e)
    for target in (_auto_loop, _flush_predictions, _upload_loop, _neural_sync_loop):
        threading.Thread(target=target, daemon=True).start()
    if _CORPUS_PENDING:
        def _bg_corpus():
            n = db.download_corpus()
            print(f"[boot] (ব্যাকগ্রাউন্ড) {n} ডকুমেন্ট সিঙ্ক হয়েছে")
        threading.Thread(target=_bg_corpus, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"সার্ভার চালু: http://0.0.0.0:{PORT}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
