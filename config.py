"""Shobdo AI — config loader."""
import os
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
STATIC = BASE / "static"
DATA.mkdir(exist_ok=True)
STATIC.mkdir(exist_ok=True)

MODEL_PATH = DATA / "model.pkl.gz"
CORPUS_PATH = DATA / "corpus.jsonl"
META_PATH = DATA / "meta.json"

# নিউরাল মডেল (ছোট ট্রান্সফরমার) — ওজন ফাইল
NEURAL_BUNDLED = BASE / "models" / "shobdo-neural.npz"   # রিপোতে রাখা (সবসময় পাওয়া যায়)
NEURAL_PATH = DATA / "shobdo-neural.npz"                 # Supabase Storage থেকে নামানো (নতুনতর)
NEURAL_NAME = "shobdo-neural.npz"                        # Storage-এ ফাইলের নাম
NEURAL_META_NAME = "shobdo-neural.json"                  # ছোট মেটা ফাইল (নতুন সংস্করণ এলো কি না)


def _load_env() -> None:
    p = BASE / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_load_env()

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
USER_AGENT = os.environ.get("CRAWLER_UA", "ShobdoAI/1.0 (+language-model demo)")
PORT = int(os.environ.get("PORT", "8000"))
AUTOLEARN = os.environ.get("AUTOLEARN", "0") == "1"

# ফ্রি হোস্ট (Render/Koyeb: ৫১২ MB) — ছোট মডেল, কম ট্রেইনিং
LOW_MEM = os.environ.get("LOW_MEM", "0") == "1"

# নিউরাল মডেল চালু/বন্ধ (ডিফল্ট: চালু; ওজন ফাইল না থাকলে নিজেই n-gram-এ চলে)
NEURAL = os.environ.get("NEURAL", "1") != "0"
# নিউরাল প্রেডিকশনের গড় সময় এর বেশি হলে সার্ভার নিজে হালকা মোডে (ছোট beam) যায়
NEURAL_BUDGET_MS = float(os.environ.get("NEURAL_BUDGET_MS", "350"))

# ---- নিরাপত্তা -------------------------------------------------------------
# সেট করা থাকলে /api/learn ও /api/autolearn-এ এই টোকেন লাগবে (হেডার: x-admin-token)।
# সেট না থাকলে আগের মতোই খোলা — কিন্তু রেট-লিমিট ও সীমা সবসময় কার্যকর।
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "").strip()
# ডেটাবেসে এর বেশি ডকুমেন্ট হলে অটো-লার্ন নিজেই থেমে যায় (১ GB ফ্রি সীমা বাঁচাতে)
MAX_DOCS = int(os.environ.get("MAX_DOCS", "40000"))

# storage / model limits (keep the free 1 GB project healthy)
MAX_DOC_CHARS = 12000
if LOW_MEM:
    MAX_TRIGRAM_CTX = 110_000
    MAX_BIGRAM_CTX = 70_000
    MAX_SKIP_CTX = 90_000
    # n-gram মডেলের মোট এন্ট্রির বাজেট; ছাড়ালে তবেই বিরল জোড়া ছাঁটা হয়
    MAX_ENTRIES = int(os.environ.get("MAX_ENTRIES", "400000"))
else:
    MAX_TRIGRAM_CTX = 2_000_000
    MAX_BIGRAM_CTX = 1_200_000
    MAX_SKIP_CTX = 1_500_000
    MAX_ENTRIES = 0          # ০ = সীমা নেই (prune() যথেষ্ট)
