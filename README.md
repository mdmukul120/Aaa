# শব্দ AI — ইন্টারনেট থেকে শেখা ভাষা-মডেল

একটা সম্পূর্ণ, চলমান সিস্টেম: **বিশ্বের বড় সাইটের লেখা → Supabase (১ GB ফ্রি) → বাংলা+ইংরেজি ভাষা-মডেল → পরবর্তী শব্দের অনুমান**, সাথে ওয়েব UI, সার্চ আর "প্রশ্ন করলে গুছিয়ে উত্তর" — সব একই প্রজেক্টে।

> **লাইভ প্রিভিউ:** এই ওয়ার্কস্পেসে `python server.py` চালু আছে — পোর্ট **8000**-এর প্রিভিউ খুললেই অ্যাপ দেখা যাবে।

---

## ১. কী কী আছে

| অংশ | ফাইল | কাজ |
|---|---|---|
| ওয়েব সার্ভার + API | `server.py` | শুধু Python স্ট্যান্ডার্ড লাইব্রেরি; UI সার্ভ করে, JSON API দেয় |
| ভাষা-মডেল | `ngmodel.py` | Interpolated n-gram LM (trigram + bigram + skip-gram + unigram), gzip+pickle সেভ |
| ক্রলার / এক্সট্র্যাক্টর | `crawler.py` | উইকি API, RSS/Atom ফিড, যেকোনো URL থেকে মূল লেখা আলাদা করা |
| বাল্ক লার্নার | `crawl_hf.py` | HuggingFace-এর `wikimedia/wikipedia` ডেটাসেট থেকে হাজারো নিবন্ধ |
| CLI ট্রেইনার | `train.py` | `--crawl N`, `--train`, `--ask "…"` |
| ডেটাবেস লেয়ার | `db.py` | Supabase REST (service key সার্ভার-সাইড), লোকাল JSONL ফলব্যাক |
| সাইট রেজিস্ট্রি | `sources.py` | ২৩+ বড় সাইট, ক্যাটাগরি, ভাষা, ফিড লিংক |
| UI | `static/index.html` | এক ফাইল, কোনো CDN নেই — বাংলা ড্যাশবোর্ড |

---

## ২. আর্কিটেকচার

```
🌐 ইন্টারনেট                                🗄 Supabase (Postgres, ১ GB ফ্রি)
 ┌──────────────────────────┐   insert   ┌────────────────────────────────┐
 │ Wikipedia bn/en (API/HF) │ ─────────► │ documents (url UNIQUE, content)│
 │ BBC বাংলা · প্রথম আলো     │            │ sites     (২৩টি বড় সাইট+ফিড)  │
 │ Guardian · Al Jazeera …  │            │ learning_runs · model_stats    │
 │ ২৩+ সাইটের RSS ফিড        │            │ predictions (লগ)               │
 └────────────┬─────────────┘            └───────────────┬────────────────┘
              │ বাংলা+ইংরেজি টোকেনাইজার                  │ FTS (tsvector)
              ▼                                          ▼
      ┌────────────────┐                       ┌──────────────────┐
      │  n-gram LM     │  P(w|a,b)=λ3P3+λ2P2+  │ search_documents │
      │ (তাত্ক্ষণিক)    │  λsPs+λ1P1            │ (Postgres FTS)   │
      └───────┬────────┘                       └──────────────────┘
              ▼
   ⚡ পরবর্তী শব্দ · ✍️ লেখা চালিয়ে যাওয়া · 🔎 জিজ্ঞাসা/উত্তর
```

**কেন এমন:** ব্রাউজারে মডেল চালানোর দরকার নেই — সার্ভারে ১ মিলিসেকেন্ডেরও কমে প্রেডিকশন হয় (পরিমাপ: **0.39 ms**)। ডেটা Supabase-এ থাকায় সার্ভার বন্ধ হলেও কিছু হারায় না; মডেল ফাইল শুধু ক্যাশ (`data/model.pkl.gz`), যেকোনো সময় `train.py --train` দিয়ে করপাস থেকে আবার বানানো যায়।

---

## ৩. ডেটাবেস (Supabase · প্রজেক্ট `Shobdo AI`)

| টেবিল | কী থাকে |
|---|---|
| `documents` | url (unique), title, lang, content, char_count — **স্ক্রল করা সব লেখা** |
| `sites` | domain, name, homepage, category, lang, feed_url — বড় সাইটের লিংক |
| `learning_runs` | কখন কোন উৎস থেকে কত পেজ/টোকেন শেখা হলো |
| `model_stats` | vocab, uni/bi/tri counts |
| `predictions` | ইনপুট, শীর্ষ পরামর্শ, সম্ভাবনা (ফিডব্যাক লগ) |

ফাংশন: `search_documents(q, lim)` — Postgres ফুল-টেক্সট সার্চ + `ts_headline` স্নিপেট; `db_usage()` — সাইজ হিসাব। RLS চালু, ডিফল্টে কোনো পাবলিক পলিসি নেই — শুধু সার্ভার-সাইড **service key** দিয়ে অ্যাক্সেস।

**বর্তমান ব্যবহার:** ডেটাবেস **২৩ MB** / ১,০২৪ MB · `documents` টেবিল **১২ MB** · **১,০৭২** ডকুমেন্ট — অর্থাৎ ফ্রি প্ল্যানের প্রায় ২% খরচে।

---

## ৪. মডেল

- টোকেনাইজার: ইংরেজি শব্দ, বাংলা শব্দ, সংখ্যা, যুক্তিচিহ্ন — একই নিয়মে দুই ভাষা
- সিগন্যাল: trigram `c(a,b,w)` + bigram `c(b,w)` + skip-gram `c(a,w)` + unigram `c(w)`
- λ শর্তসাপেক্ষে বসে: কনটেক্সট যত বেশি দেখা হয়েছে, trigram-এর ওজন তত বেশি
- অজানা শব্দে prefix-fallback (একই ৩ অক্ষরের শুরু মিলিয়ে পরামর্শ)
- জেনারেশন: top-k + temperature + repetition penalty

**বর্তমান মাপ:** ৭৫৫,৩৩৪ টোকেন · ৬৯,৭৮৩ শব্দভান্ডার · ৩,৯৪,৫৯২ bigram · ৬,১১,০৬৯ trigram · ফাইল ১২ MB
**সৎ হেল্ড-আউট মূল্যায়ন** (শেষ ১০% ডকুমেন্ট মডেল দেখেনি): top-1 **১৩.৬%**, top-5 **২৭.২%** — ছোট n-gram মডেলের জন্য স্বাভাবিক।

> এটা ChatGPT নয় — এটা পরিসংখ্যানভিত্তিক "পরের শব্দ" মডেল; যা শেখে তা-ই বলে। প্যাটার্ন চেনে, কিন্তু নতুন তথ্য বানায় না। `data/` ও DB-তে পেজ বাড়ালে প্রেডিকশন ভালো হয় (`python crawl_hf.py --bn 1000 --en 800`)।

---

## ৫. চালানোর নিয়ম

```bash
pip install -r requirements.txt
cp .env.example .env          # Supabase URL + service key বসান

python server.py              # http://localhost:8000
python train.py --crawl 300   # ইন্টারনেট থেকে ~৩০০ পেজ এনে শেখাও
python crawl_hf.py --bn 300 --en 200   # বাল্ক: হাজারো উইকি নিবন্ধ
python train.py --ask "বাংলা ভাষা"      # CLI-তে টেস্ট
```

## ৬. API

| পদ্ধতি | রুট | কাজ |
|---|---|---|
| GET | `/api/stats` | মডেল+DB+লার্নিং স্ট্যাটাস |
| POST | `/api/predict` `{text}` | পরবর্তী ৬ শব্দ + সম্ভাবনা (ms সহ) |
| POST | `/api/generate` `{text, temperature}` | লেখা চালিয়ে যাওয়া |
| POST | `/api/learn` `{mode, count, url}` | `wiki_bn\|wiki_en\|news_bn\|news_en\|auto\|site\|url\|test` |
| GET | `/api/learn/status` | প্রোগ্রেস + লগ |
| POST | `/api/ask` `{q}` | DB→ইন্টারনেট খুঁজে গুছানো উত্তর, শেখে ও বটে |
| POST | `/api/search` `{q}` | Supabase FTS সার্চ |
| GET | `/api/pages` | সর্বশেষ শেখা পেজ |
| POST | `/api/autolearn` `{on}` | প্রতি ২ মিনিটে নিজে শেখা চালু/বন্ধ |

## ৭. উৎস ও লাইসেন্স

Wikipedia (bn/en) — CC BY-SA; BBC/Al Jazeera/Guardian/প্রথম আলো ইত্যাদি রাষ্ট্রীয় RSS ফিড; `wikimedia/wikipedia` ডেটাসেট (HuggingFace)। শুধু পাবলিক লেখা, কোনো ব্যক্তিগত ডেটা নয়।

## ৮. হোস্টিং

সব ধাপের বিস্তারিত: **[`deploy/DEPLOY.md`](deploy/DEPLOY.md)**

| হোস্ট | খরচ | উপযুক্ততা |
|---|---|---|
| **Hugging Face Spaces** (Docker) | ⚠️ এখন PRO লাগে | শুধু Static ফ্রি — Python চলবে না |
| **Fly.io** | ~$৫–৭/মাস, ১ GB + ভলিউম | ✅ প্রোডাকশন-গ্রেড, অটো-স্টপ |
| **Render** | ফ্রি ৫১২ MB | ✅ কার্ড ছাড়াই ফ্রি · `LOW_MEM=1`-এ মডেল ৭১ MB র‍্যামে |
| **VPS + docker compose** | ~$৪–৬/মাস | ✅ সস্তা দীর্ঘমেয়াদে |
| **Vercel** | ফ্রি | ✅ শুধু UI (`vercel/` ফোল্ডার) — ইতিমধ্যে ডিপ্লয়ড · ❌ Python মডেল সার্ভার চলবে না |

ফাইল প্রস্তুত: `Dockerfile`, `fly.toml`, `render.yaml`, `Procfile`, `docker-compose.yml`, `.dockerignore`।

**গুরুত্বপূর্ণ:** ফ্রি হোস্টের ডিস্ক অস্থায়ী — তাই সার্ভার চালু হওয়ার সময় `_bootstrap()` নিজেই Supabase থেকে সব ডকুমেন্ট এনে মডেল রিবিল্ড করে (পরীক্ষিত: খালি ফোল্ডার → ~৪০ সেকেন্ডে ৭৫৪k টোকেনের মডেল)।
service key কখনো ব্রাউজারে বা গিটে যাবে না — `.env` শুধু সার্ভারের জন্য, হোস্টে env var/secret হিসেবে বসান।

## ৯. পরবর্তী উন্নতির পথ

1. **আরও ডেটা** (১০k–৫০k পেজ) → top-1 ৩০%+ সহজে সম্ভব
2. Kneser-Ney smoothing / char-level fallback
3. Supabase `pgvector` + embedding-ভিত্তিক সেমান্টিক সার্চ
4. গল্প/আর্টিকেল জেনারেশনের জন্য ছোট ট্রান্সফরমার (GPU) — এখনকার UI অপরিবর্তিত থাকবে
