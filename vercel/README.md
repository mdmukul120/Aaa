# শব্দ AI — UI (Vercel)

এই ফোল্ডারটাই Vercel-এ ডিপ্লয় হয় (স্ট্যাটিক HTML/CSS/JS — কোনো বিল্ড দরকার নেই)।

## পাতা
| ঠিকানা | ফাইল | কী |
|---|---|---|
| `/` | `index.html` | চ্যাট — ChatGPT-ধাঁচের ইন্টারফেস (`/api/ask`, "লেখক" মোডে `/api/generate`) |
| `/studio` | `studio.html` | শেখাও, স্পাইডার, পরের-শব্দ অনুমান, পরিসংখ্যান |

চ্যাটের ইতিহাস ব্রাউজারের localStorage-এ থাকে (`shobdo_chats_v1`), সার্ভারে নয়।
`static/` ফোল্ডারের একই নামের ফাইলের সাথে এগুলো হুবহু এক — পার্থক্য শুধু এখানে `config.js`-এর ট্যাগ আছে
(`tests/test_ui.py` সেটা পরীক্ষা করে)। একটা বদলালে অন্যটাও বদলান।

## ডিপ্লয়
```bash
# অপশন ১ — Vercel CLI
cd vercel && npx vercel --prod

# অপশন ২ — GitHub দিয়ে (রেকমেন্ডেড)
# repo-তে পুশ করুন → Vercel → Add New Project → repo সিলেক্ট →
# Root Directory = vercel → Framework Preset = Other → Deploy
```

## ব্যাকএন্ড যুক্ত করা
1. মডেল-সার্ভারটা কোথাও চালু করুন (`deploy/DEPLOY.md`) — যেমন Hugging Face Space বা Fly.io
2. `config.js`-এ URL বসিয়ে আবার ডিপ্লয় করুন:
   ```js
   window.SHOBDO_API_BASE = "https://shobdo-ai.fly.dev";
   ```
3. অথবা কোড না বদলেই — UI-তে উপরের হলুদ বার-এ URL লিখে **সংযোগ করো** চাপুন (localStorage-এ সেভ থাকবে)

## কেন ব্যাকএন্ড আলাদা?
Vercel serverless — মেমরিতে ভাষা-মডেল ধরে রাখা যায় না, ডিস্কও থাকে না।
তাই: **UI = Vercel**, **মডেল সার্ভার = long-running হোস্ট**, **ডেটা = Supabase**।
