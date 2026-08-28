import os
from pathlib import Path

# --- প্রজেক্টের গ্লোবাল পাথ কনফিগারেশন ---
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
IMAGE_OUTPUT_DIR = OUTPUT_DIR / "thumbnails"
JSON_OUTPUT_FILE = OUTPUT_DIR / "mhdtv_matches.json"

# ফোল্ডার নিশ্চিত করা
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- ওয়েব স্ক্র্যাপার কনফিগারেশন ---
TARGET_BASE_URL = "https://live.mhdtv.online/"
REQUEST_TIMEOUT = 15
MAX_RETRIES = 3

# প্রিমিয়াম ব্রাউজার হেডার (Anti-Hotlinking বাইপাস করার জন্য)
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,bn;q=0.8",
    "Referer": TARGET_BASE_URL,
    "Origin": "https://live.mhdtv.online",
}

# --- অটো-ইমেজ জেনারেটর কনফিগারেশন ---
BANNER_WIDTH = 1200
BANNER_HEIGHT = 630
BG_PRIMARY_COLOR = (18, 24, 38)     # ডার্ক প্রিমিয়াম থিম
BG_SECONDARY_COLOR = (30, 41, 59)   # কন্টেন্ট কার্ড কালার
ACCENT_COLOR = (16, 185, 129)       # প্রিমিয়াম গ্রিন কালার
TEXT_WHITE = (255, 255, 255)        # প্রধান টেক্সট
TEXT_MUTED = (148, 163, 184)       # সাব-টেক্সট
