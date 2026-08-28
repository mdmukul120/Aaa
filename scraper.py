import re
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any
import config

class MHDTVScraper:
    """
    live.mhdtv.online থেকে লাইভ ও আপকামিং ম্যাচ, প্লেয়ার আইফ্রেম এবং
    স্ট্রিমিং লিংক স্ক্র্যাপ করার ক্লাস
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(config.HTTP_HEADERS)

    def fetch_page(self, url: str) -> BeautifulSoup:
        """ পেজ ডাউনলোডের জন্য সেফ রিকোয়েস্ট হ্যান্ডলার """
        for attempt in range(config.MAX_RETRIES):
            try:
                response = self.session.get(url, timeout=config.REQUEST_TIMEOUT)
                if response.status_code == 200:
                    return BeautifulSoup(response.content, 'html.parser')
            except requests.RequestException:
                pass
        return None

    def extract_stream_sources(self, match_url: str) -> Dict[str, str]:
        """
        ম্যাচ স্লাগ পেজে ঢুকে আইফ্রেম ও m3u8 স্ট্রিম লিংক বের করে।
        সাইটের অপ্রয়োজনীয় এলিমেন্ট কেটে দিয়ে শুধু ভিডিও প্লেয়ার রাখার কোডও তৈরি করে।
        """
        soup = self.fetch_page(match_url)
        result = {
            "iframe_url": match_url,
            "m3u8_url": "",
            "embed_code": f'<iframe src="{match_url}" width="100%" height="100%" frameborder="0" allowfullscreen="true"></iframe>'
        }

        if not soup:
            return result

        page_html = str(soup)

        # ১. Iframe খোঁজা (Ads ও Google Tag Manager ফিল্টার করে)
        iframes = soup.find_all('iframe')
        ignored_domains = ['googletagmanager.com', 'facebook.com', 'analytics', 'disqus', 'ads', 'gtm']

        for iframe in iframes:
            src = iframe.get('src') or iframe.get('data-src') or ''
            if src and not any(domain in src.lower() for domain in ignored_domains):
                final_iframe = src if src.startswith('http') else f"https:{src}" if src.startswith('//') else f"{config.TARGET_BASE_URL.rstrip('/')}/{src.lstrip('/')}"
                result["iframe_url"] = final_iframe
                
                # সম্পূর্ণ ফুলস্ক্রিন প্লেয়ার embed কোড
                result["embed_code"] = f'<iframe src="{final_iframe}" width="100%" height="100%" frameborder="0" allowfullscreen="true" scrolling="no" allow="autoplay; encrypted-media"></iframe>'
                break

        # ২. m3u8 লিংক ফিল্টার করা (যদি সরাসরি ব্যাকএন্ডে থাকে)
        m3u8_matches = re.findall(r'(https?://[^\s"\'<>]+\.m3u8[^\s"\'<>]*)', page_html)
        if m3u8_matches:
            result["m3u8_url"] = m3u8_matches[0]

        return result

    def get_all_matches(self) -> List[Dict[str, Any]]:
        """ মূল পেজ থেকে সকল ম্যাচ ইনফরমেশন কালেকশন করে """
        print(f"[+] স্ক্র্যাপ করা হচ্ছে: {config.TARGET_BASE_URL}")
        soup = self.fetch_page(config.TARGET_BASE_URL)
        matches_data = []

        if not soup:
            print("[-] পেজ লোড করা সম্ভব হয়নি।")
            return matches_data

        # ম্যাচ কার্ড এবং লিংক নির্বাচন
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

            # অরিজিনাল থাম্বনেইল পিক করা
            img_tag = card.find('img') if card.name != 'a' else (card.find_parent().find('img') if card.find_parent() else None)
            existing_image_url = ""
            
            if img_tag:
                existing_image_url = img_tag.get('src') or img_tag.get('data-src') or ""
                if existing_image_url and not existing_image_url.startswith('http'):
                    existing_image_url = f"{config.TARGET_BASE_URL.rstrip('/')}/{existing_image_url.lstrip('/')}"

            print(f"  └─ ম্যাচ পাওয়া গেছে: {title[:40]}...")

            # স্লাগ পেজ থেকে ভিডিও স্ট্রিম এক্সট্র্যাক্ট
            stream_info = self.extract_stream_sources(href)

            matches_data.append({
                "title": title,
                "slug_url": href,
                "original_image": existing_image_url,
                "iframe_url": stream_info["iframe_url"],
                "m3u8_url": stream_info["m3u8_url"],
                "embed_code": stream_info["embed_code"]
            })

        return matches_data
