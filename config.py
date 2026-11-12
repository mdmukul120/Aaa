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

# storage / model limits (keep the free 1 GB project healthy)
MAX_DOC_CHARS = 12000
if LOW_MEM:
    MAX_TRIGRAM_CTX = 110_000
    MAX_BIGRAM_CTX = 70_000
    MAX_SKIP_CTX = 90_000
else:
    MAX_TRIGRAM_CTX = 2_000_000
    MAX_BIGRAM_CTX = 1_200_000
    MAX_SKIP_CTX = 1_500_000
