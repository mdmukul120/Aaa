# হোস্টিং গাইড — শব্দ AI

> **TL;DR:** এই অ্যাপটা *stateful* (মেমরিতে মডেল + ডিস্কে করপাস রাখে), তাই সাধারণ serverless-এ চলবে না।
> - **ফ্রি + সবচেয়ে সহজ:** Hugging Face Spaces (Docker) — ১৬ GB RAM, AI ডেমোর জন্য আদর্শ
> - **সবচেয়ে ভালো প্রোডাকশন:** Fly.io (১ GB, ভলিউম সহ, ~$৫/মাস)
> - **সহজ + পরিচিত:** Render (ফ্রি ৫১২ MB — training-এ চাপ পড়তে পারে)
> - **নিজের VPS:** `docker compose up -d` — সবচেয়ে সস্তা দীর্ঘমেয়াদে
> - **Vercel:** শুধু স্ট্যাটিক ফ্রন্টএন্ডের জন্য; এই Python সার্ভার+মডেল Vercel-এ চলবে না

সব প্ল্যাটফর্মেই ৩টি জিনিস দরকার: `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` (গোপন), আর `PORT` বাইন্ডিং (`0.0.0.0`, কোডে আগেই আছে)।

---

## ০. আগে যা করতেই হবে — সিক্রেট সরানো

`.env` ফাইলে service key আছে। **এটা কখনো গিটে পুশ করবেন না** (`.gitignore`-এ আছে)। হোস্টের ড্যাশবোর্ডে আলাদা env var / secret হিসেবে বসাবেন।

```bash
cd shobdo-ai
git init && git add . && git commit -m "Shobdo AI"
# চাইলে: gh repo create shobdo-ai --public --source=. --push
```

---

## ১. Hugging Face Spaces — ⚠️ এখন Docker-এ PRO লাগে

HF-এর নতুন নিয়মে **Docker ও Gradio Space চালাতে PRO (মাসিক)** লাগে; ফ্রিতে শুধু **Static**।
Static-এ Python সার্ভার চলে না — তাই ব্যাকএন্ডের জন্য **২ (Render) / ৩ (Koyeb) / Fly.io** বেছে নিন।
(আপনার স্ক্রিনশটে ঠিক এটাই দেখাচ্ছে — তাই HF বাদ।)

## ২. Render — ✅ ফ্রি, কার্ড লাগে না (রেকমেন্ডেড)

https://render.com → **Sign up with GitHub**

1. **New +** → **Blueprint** → repo **shobdo-ai** বাছুন → `render.yaml` নিজেই সব সেট করবে
   *(`LOW_MEM=1` দেওয়া আছে — নিউরাল + n-gram মডেল মিলিয়ে ৫১২ MB-র অনেক নিচে থাকে)*
2. **Environment**-এ বসান:
   ```
   SUPABASE_URL         = https://yknfbloljepwrqthcvpd.supabase.co
   SUPABASE_SERVICE_KEY = <service_role key>
   ADMIN_TOKEN          = <লম্বা এলোমেলো গোপন শব্দ>   # "শেখাও" ও "অটো-লার্ন" বোতামের পাসওয়ার্ড
   ```
   `ADMIN_TOKEN` না দিলে যে-কেউ আপনার সার্ভারকে দিয়ে ইন্টারনেট থেকে শেখাতে/ডেটাবেস ভরাতে পারবে।
   (Blueprint দিয়ে বানালে Render নিজেই একটা মান বানিয়ে দেয় — Environment ট্যাবে দেখে নিন।)
   UI-তে প্রথমবার "শেখাও" চাপলে টোকেন চাইবে; একবার দিলে ব্রাউজার মনে রাখে।
3. **Deploy** → ১–২ মিনিটে: `https://shobdo-ai.onrender.com`
4. `.../healthz` খুলে `{"ok":true,...}` দেখলেই ব্যাকএন্ড চালু ✅
5. Vercel UI-র হলুদ বার-এ এই URL বসিয়ে **সংযোগ করো**

**ফ্রি প্ল্যান:** ১৫ মিনিট নিষ্ক্রিয় থাকলে ঘুমায়; পরের ভিজিটে ১০–৩০ সেকেন্ডে জাগে
(n-gram মডেল Supabase Storage থেকে নামে; নিউরাল ওজন রিপোর `models/` ফোল্ডারেই আছে)।
শেখা n-gram মডেল ১০ মিনিট পরপর Storage-এ ওঠে — ঘুমিয়ে পড়লেও শেখা হারায় না।

**নিউরাল মডেল নিজে নিজে আপডেট:** GitHub repo → Settings → Secrets and variables → Actions-এ
`SUPABASE_URL` ও `SUPABASE_SERVICE_KEY` বসান। তারপর *Actions → Train neural model* প্রতি সপ্তাহে
Supabase-এর সব লেখা দিয়ে মডেল ফাইন-টিউন করে Storage-এ তোলে; সার্ভার ৩০ মিনিটের মধ্যে নিজেই নেয়।

## ৩. Koyeb — ফ্রি, Dockerfile দিয়ে

https://app.koyeb.com → GitHub কানেক্ট → **Builder: Dockerfile** → Port `8000` → Instance: **Free**
→ একই দুইটা Supabase ভ্যারিয়েবল → Deploy → `https://<app>.koyeb.app`

## ৪. Fly.io — প্রোডাকশনের জন্য সেরা (~$৫/মাস)

ভলিউম থাকে, তাই মডেল ও করপাস ডিস্কে টেকে; অটো-স্টপে খরচ কমে।

```bash
curl -L https://fly.io/install.sh | sh
fly auth signup          # বা fly auth login
cd shobdo-ai
fly launch --no-deploy   # fly.toml আগেই দেওয়া আছে, নাম বদলাতে চাইলে বদলান
fly volumes create shobdo_data --size 1 --region sin
fly secrets set SUPABASE_URL=https://yknfbloljepwrqthcvpd.supabase.co \
                SUPABASE_SERVICE_KEY=<service_role_key>
fly deploy
fly open                 # https://shobdo-ai.fly.dev
```

মেমরি ১ GB দেওয়া আছে (`fly.toml`); মডেল ~৩০০ MB নেয়, তাই ঠিক আছে।
খরচ কমাতে `shared-cpu-1x`, ১ GB ≈ **$৬.৭/মাস** (অটো-স্টপে ব্যবহার কমলে কম)।

---

## ৫. নিজের VPS (সবচেয়ে সস্তা দীর্ঘমেয়াদে — ~$৪–৬/মাস)

যেকোনো ১ GB VPS (Hetzner CX22, Contabo, DigitalOcean, Oracle Free Tier):

```bash
ssh root@<ip>
apt update && apt install -y docker.io docker-compose-plugin git
git clone <your-repo> /opt/shobdo-ai && cd /opt/shobdo-ai
nano .env            # Supabase keys বসান
docker compose up -d --build
# চাইলে Nginx + Certbot দিয়ে ডোমেইন + HTTPS
```

`./data` ভলিউমে থাকায় মডেল টিকে থাকবে, আর `restart: unless-stopped` মানে সার্ভার রিবুটেও চালু হবে।

---

## ৬. Vercel সম্পর্কে সৎ কথা

Vercel serverless — ফাংশন কয়েক সেকেন্ড পর বন্ধ হয়ে যায়, ডিস্ক নেই, ইন-মেমরি মডেল রাখা যায় না।
- ❌ পুরো অ্যাপ (এই Python সার্ভার + ট্রেইনিং) Vercel-এ চলবে না
- ✅ চাইলে শুধু `static/index.html` Vercel-এ রেখে API অন্য জায়গায় রাখা যায় — কিন্তু তখন CORS + `API_BASE` আলাদা করতে হবে, লাভ কম
- ✅ ডোমেইন কিনলে Vercel-এ DNS রেখে সাবডোমেইন (যেমন `ai.yourdomain.com`) Fly/Render-এ পয়েন্ট করা যায়

---

## ৭. হোস্টিংয়ের পর যা করবেন

| কাজ | কীভাবে |
|---|---|
| DB ব্যাকআপ | Supabase Dashboard → Database → Backups |
| খরচ/কোটা দেখা | Dashboard → Usage (১ GB সীমা) |
| নিজে নিজে শেখা | UI-তে **অটো-লার্ন** চালু করুন (প্রতি ২ মিনিটে নতুন পেজ) |
| আরও ডেটা | `python crawl_hf.py --bn 1000 --en 800` → `python train.py --train` |
| জন্মদিনের সার্টিফিকেট | `GET /healthz` দেখলেই বোঝা যাবে সার্ভার বেঁচে আছে |

---

## ⭐ পছন্দের আর্কিটেকচার: GitHub (কোড) + Vercel (UI) + Supabase (ডেটা)

```
GitHub ─── কোড (python + vercel/ ফোল্ডার)
   │
   ├─► Vercel ─── শুধু static UI (vercel/ ফোল্ডার) ← এখানে আপনার UI ডিপ্লয় হয়েছে
   │
   └─► HF Spaces / Fly.io ─── Python মডেল সার্ভার (API + ট্রেইনিং)
                                   │
                                   └─► Supabase ─── সব ডেটা (১,০৭২ ডকুমেন্ট, ২৩ MB)
```

> **গুরুত্বপূর্ণ:** ডেটা GitHub-এ রাখা উচিত নয় — GitHub ফাইলের সাইজ সীমা ১০০ MB, আর প্রতিবার শেখার পর ৭–১২ MB মডেল কমিট করলে ইতিহাস ফুলে যাবে।
> ডেটার সঠিক জায়গা **Supabase** (ইতিমধ্যে সেখানে আছে), আর GitHub-এ থাকবে শুধু **কোড**।

### ধাপ ১ — GitHub-এ পুশ
```bash
cd shobdo-ai
git remote add origin https://github.com/<username>/shobdo-ai.git
git branch -M main
git push -u origin main
```
`.gitignore`-এ `.env`, `data/corpus.jsonl`, `data/model.pkl.gz` বাদ আছে — নিরাপদ।

### ধাপ ২ — Vercel (UI) — ইতিমধ্যে ডিপ্লয় হয়ে গেছে!
প্রজেক্ট: **shobdo-ai** · ঠিকানা: `https://shobdo-ai-mdmukul120s-projects.vercel.app`

**একটা কাজ বাকি — পাবলিক করা:**
Vercel → প্রজেক্ট **shobdo-ai** → **Settings** → **Deployment Protection** → *Vercel Authentication* → **Disabled** → Save
(নতুন প্রজেক্টে এটা ডিফল্টে চালু থাকে, তাই এখন লিংকে গেলে Vercel লগইন চায়)

**GitHub থেকে অটো-ডিপ্লয় চালু করতে (ফুল ৩১ KB ড্যাশবোর্ড বসবে):**
Vercel → shobdo-ai → Settings → Git → Connect Repository → `shobdo-ai` বাছুন
→ **Root Directory = `vercel`** → Framework Preset = **Other** → Build Command = খালি → Deploy

এরপর `vercel/` ফোল্ডারে push করলেই UI অটো-আপডেট হবে।

### ধাপ ৩ — মডেল সার্ভার হোস্ট করুন (HF Spaces, ফ্রি)
**২ নম্বর সেকশন (Render)** দেখুন। ঠিকানা পাবেন যেমন `https://shobdo-ai.onrender.com`।

### ধাপ ৪ — দুইটাকে জোড়া দিন
Vercel UI খুলে উপরের হলুদ বার-এ ব্যাকএন্ড URL বসিয়ে **সংযোগ করো** চাপুন — ব্যস।
(অথবা `vercel/config.js`-এ `window.SHOBDO_API_BASE = 'https://...'` লিখে আবার ডিপ্লয় করুন।)

CORS আগেই খোলা (`Access-Control-Allow-Origin: *`), তাই Vercel UI থেকে সরাসরি API কল কাজ করে।

## ৮. ট্রাবলশুটিং

- **`Port already in use`** → `PORT` env বদলান
- **মডেল নেই / ০ টোকেন** → লগে `[boot]` লাইন দেখুন; Supabase secret ঠিক আছে কি না যাচাই করুন
- **ফ্রি টিয়ারে ধীর প্রথম রেসপন্স** → স্বাভাবিক (কোল্ড স্টার্ট); `data/model.pkl.gz` ইমেজে রাখলে ~২ সেকেন্ড
- **৫১২ MB-তে OOM** → `config.py`-তে `MAX_TRIGRAM_CTX` কমান বা বড় প্ল্যানে যান
