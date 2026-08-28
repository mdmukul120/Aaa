import re
import json
import time
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import config

class MHDTVScraper:
    """
    Playwright Network Interceptor ব্যবহার করে live.mhdtv.online-এর 
    বাটন ক্লিক এবং আসল .m3u8 স্ট্রিমিং লিঙ্ক অটোমেটিক বের করার স্ক্র্যাপার।
    """

    def __init__(self):
        self.base_url = config.TARGET_BASE_URL

    def _extract_m3u8_with_playwright(self, page_url: str) -> List[Dict[str, str]]:
        """
        Headless Browser দিয়ে পেজ লোড করে নেটওয়ার্ক ট্রাফিক থেকে
        আসল .m3u8 এবং প্রতি সার্ভার টাইটেল এক্সট্র্যাক্ট করে।
        """
        servers = []
        captured_m3u8s = []

        with sync_playwright() as p:
            # ব্রাউজার লঞ্চ (Headless mode)
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                extra_http_headers={
                    "Referer": self.base_url,
                    "Origin": self.base_url
                }
            )
            page = context.new_page()

            # নেটওয়ার্ক রিকোয়েস্ট ট্র্যাকিং করে m3u8 ক্যাচ করা
            def handle_request(request):
                url = request.url
                if ".m3u8" in url and url not in captured_m3u8s:
                    captured_m3u8s.append(url)

            page.on("request", handle_request)

            try:
                page.goto(page_url, timeout=25000, wait_until="domcontentloaded")
                time.sleep(3) # প্লেয়ার ও জাভাস্ক্রিপ্ট লোড হওয়ার সময় দেওয়া

                # সার্ভার বাটনগুলো খুঁজে বের করা (যেমন: Willow FHD, TAPMAD HD)
                server_buttons = page.query_selector_all('a.server-btn, button.server-btn, .servers-list a, .stream-buttons a, div[class*="server"] a, div[class*="stream"] a, button, .btn')

                if server_buttons:
                    for idx, btn in enumerate(server_buttons):
                        try:
                            btn_text = btn.inner_text().strip()
                            if not btn_text or len(btn_text) > 25:
                                continue

                            # বাটনে ক্লিক করে সংশ্লিষ্ট M3U8 ট্রিগার করা
                            prev_count = len(captured_m3u8s)
                            btn.click(timeout=2000)
                            time.sleep(2)

                            # নতুন M3U8 পাওয়া গেলে তা ওই বাটনের নামে সেভ করা
                            found_url = captured_m3u8s[-1] if len(captured_m3u8s) > prev_count else ""
                            
                            servers.append({
                                "server_name": btn_text,
                                "m3u8_url": found_url,
                                "page_url": page_url,
                                "stream_type": "hls" if found_url else "iframe"
                            })
                        except Exception:
                            continue

                # যদি কোনো বাটন ক্লিক না হলেও ব্যাকগ্রাউন্ডে m3u8 পাওয়া যায়
                if not servers and captured_m3u8s:
                    servers.append({
                        "server_name": "Main Stream",
                        "m3u8_url": captured_m3u8s[0],
                        "page_url": page_url,
                        "stream_type": "hls"
                    })

            except Exception as e:
                print(f"[!] Playwright Execution Warning for {page_url}: {e}")
            finally:
                browser.close()

        # যদি M3U8 না পাওয়া যায়, তবে fallback হিসেবে আইফ্রেম ইউআরএল ব্যবহার
        if not servers:
            servers.append({
                "server_name": "Default Server",
                "m3u8_url": "",
                "page_url": page_url,
                "stream_type": "iframe"
            })

        return servers

    def get_all_matches(self) -> List[Dict[str, Any]]:
        """ মূল পেজ থেকে সকল লাইভ ও আপকামিং ম্যাচের তথ্য কালেকশন করে """
        print(f"[+] Advanced Network Scraping শুরু হচ্ছে: {self.base_url}")
        matches_data = []

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            try:
                page.goto(self.base_url, timeout=25000)
                time.sleep(3)
                html_content = page.content()
            except Exception:
                html_content = ""
            finally:
                browser.close()

        if not html_content:
            print("[-] পেজ কন্টেন্ট লোড করা সম্ভব হয়নি।")
            return matches_data

        soup = BeautifulSoup(html_content, 'html.parser')
        cards = soup.select('.post, .article, .match-card, .entry-title a, article a, div.content a, .event-item')
        seen_urls = set()

        for card in cards:
            link_tag = card if card.name == 'a' else card.find('a')
            if not link_tag or not link_tag.get('href'):
                continue

            href = link_tag['href']
            if href in seen_urls or not href.startswith('http'):
                continue

            seen_urls.add(href)
            title = link_tag.get('title') or card.text.strip()
            title = re.sub(r'\s+', ' ', title)

            if len(title) < 4:
                continue

            # থিম থাম্বনেইল পিক করা
            img_tag = card.find('img') if card.name != 'a' else (card.find_parent().find('img') if card.find_parent() else None)
            existing_image_url = ""
            if img_tag:
                existing_image_url = img_tag.get('src') or img_tag.get('data-src') or ""

            print(f"  └─ প্রসেস হচ্ছে ও M3U8 ধরা হচ্ছে: {title[:35]}...")

            # প্লেরাইট দিয়ে আসল M3U8 ধরা
            servers = self._extract_m3u8_with_playwright(href)

            matches_data.append({
                "title": title,
                "slug_url": href,
                "original_image": existing_image_url,
                "servers": servers
            })

        return matches_data
