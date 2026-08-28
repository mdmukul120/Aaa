import re
import base64
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any
import config

class MHDTVScraper:
    """
    live.mhdtv.online থেকে লাইভ ম্যাচ, অরিজিনাল মেনিফেস্ট/m3u8 সরাসরি স্ট্রিমিং লিঙ্ক 
    এবং প্লেয়ার সোর্স এক্সট্র্যাক্ট করার প্রিমিয়াম স্ক্র্যাপার
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(config.HTTP_HEADERS)

    def fetch_page(self, url: str) -> BeautifulSoup:
        """ সেফ HTTP রিকোয়েস্ট সেভার """
        for attempt in range(config.MAX_RETRIES):
            try:
                response = self.session.get(url, timeout=config.REQUEST_TIMEOUT)
                if response.status_code == 200:
                    return BeautifulSoup(response.content, 'html.parser')
            except requests.RequestException:
                pass
        return None

    def extract_direct_stream(self, match_url: str) -> Dict[str, str]:
        """
        ম্যাচ পেজ এবং প্লেয়ার স্ক্রিপ্ট এনালাইসিস করে সরাসরি m3u8 স্ট্রিমিং লিঙ্ক খুঁজে বের করে।
        """
        soup = self.fetch_page(match_url)
        result = {
            "stream_url": "",
            "stream_type": "unknown",
            "page_url": match_url
        }

        if not soup:
            return result

        page_html = str(soup)

        # ১. সরাসরি HTML5 <source> এবং <video> ট্যাগ চেক করা
        video_tags = soup.find_all(['video', 'source'])
        for v in video_tags:
            src = v.get('src') or v.get('data-src') or ''
            if '.m3u8' in src or '.mp4' in src:
                result["stream_url"] = src if src.startswith('http') else f"https:{src}" if src.startswith('//') else f"{config.TARGET_BASE_URL.rstrip('/')}/{src.lstrip('/')}"
                result["stream_type"] = "hls" if ".m3u8" in src else "mp4"
                return result

        # ২. JavaScript স্ক্রিপ্ট ফাইল ও Clappr/JWPlayer কনফিগারেশন থেকে m3u8 ডিটেক্ট করা
        script_m3u8 = re.findall(r'(https?://[^\s"\'<>]+\.m3u8[^\s"\'<>]*)', page_html)
        if script_m3u8:
            result["stream_url"] = script_m3u8[0]
            result["stream_type"] = "hls"
            return result

        # ৩. বেস৬৪ (Base64) এনকোডেড m3u8 লিঙ্ক ফিল্টারিং
        b64_matches = re.findall(r'aHR0cHM6Ly[A0Za-z0-9+/=]+', page_html)
        for b64 in b64_matches:
            try:
                decoded = base64.b64decode(b64).decode('utf-8')
                if '.m3u8' in decoded:
                    result["stream_url"] = decoded
                    result["stream_type"] = "hls"
                    return result
            except Exception:
                pass

        # ৪. আইফ্রেম পেজে প্রবেশ করে সাব-লেভেলে স্ট্রিমিং লিঙ্ক অনুসন্ধান করা
        iframes = soup.find_all('iframe')
        ignored_domains = ['googletagmanager.com', 'facebook.com', 'analytics', 'disqus', 'ads', 'gtm']

        for iframe in iframes:
            iframe_src = iframe.get('src') or iframe.get('data-src') or ''
            if iframe_src and not any(d in iframe_src.lower() for d in ignored_domains):
                full_iframe_url = iframe_src if iframe_src.startswith('http') else f"https:{iframe_src}" if iframe_src.startswith('//') else f"{config.TARGET_BASE_URL.rstrip('/')}/{iframe_src.lstrip('/')}"
                
                # আইফ্রেম পেজ স্ক্র্যাপ করা
                sub_soup = self.fetch_page(full_iframe_url)
                if sub_soup:
                    sub_html = str(sub_soup)
                    sub_m3u8 = re.findall(r'(https?://[^\s"\'<>]+\.m3u8[^\s"\'<>]*)', sub_html)
                    if sub_m3u8:
                        result["stream_url"] = sub_m3u8[0]
                        result["stream_type"] = "hls"
                        return result

                # ডাইরেক্ট লিঙ্ক না পাওয়া গেলে fallback আইফ্রেম ইউআরএল সেভ রাখা
                if not result["stream_url"]:
                    result["stream_url"] = full_iframe_url
                    result["stream_type"] = "iframe"

        return result

    def get_all_matches(self) -> List[Dict[str, Any]]:
        """ সাইট থেকে বর্তমানে সক্রিয় সকল লাইভ ও আপকামিং ম্যাচ এক্সট্র্যাক্ট করে """
        print(f"[+] স্ক্র্যাপ করা হচ্ছে: {config.TARGET_BASE_URL}")
        soup = self.fetch_page(config.TARGET_BASE_URL)
        matches_data = []

        if not soup:
            print("[-] মূল পেজ লোড করা সম্ভব হয়নি।")
            return matches_data

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
                if existing_image_url and not existing_image_url.startswith('http'):
                    existing_image_url = f"{config.TARGET_BASE_URL.rstrip('/')}/{existing_image_url.lstrip('/')}"

            print(f"  └─ লাইভ ম্যাচ প্রসেস হচ্ছে: {title[:40]}...")

            # সরাসরি স্ট্রিমিং লিঙ্ক সোর্স এক্সট্র্যাক্ট করা
            stream_info = self.extract_direct_stream(href)

            matches_data.append({
                "title": title,
                "slug_url": href,
                "original_image": existing_image_url,
                "stream_url": stream_info["stream_url"],
                "stream_type": stream_info["stream_type"]
            })

        return matches_data
