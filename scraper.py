import re
import base64
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any
import config

class MHDTVScraper:
    """
    live.mhdtv.online থেকে লাইভ ম্যাচ, সকল সার্ভার অপশন (যেমন: Willow FHD, TAPMAD HD ইত্যাদি) 
    এবং প্লেয়ার সোর্স স্ক্র্যাপ করার অ্যাডভান্সড স্ক্র্যাপার
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(config.HTTP_HEADERS)

    def fetch_page(self, url: str) -> BeautifulSoup:
        """ এইচটিটিপি রিকোয়েস্ট সেফলি হ্যান্ডেল করার মেথড """
        for attempt in range(config.MAX_RETRIES):
            try:
                response = self.session.get(url, timeout=config.REQUEST_TIMEOUT)
                if response.status_code == 200:
                    return BeautifulSoup(response.content, 'html.parser')
            except requests.RequestException:
                pass
        return None

    def _parse_stream_source_from_html(self, html_content: str, base_page_url: str) -> Dict[str, str]:
        """ HTML কন্টেন্ট বিশ্লেষণ করে সরাসরি m3u8 বা আইফ্রেম স্ট্রিম লিঙ্ক বের করে """
        result = {"stream_url": base_page_url, "stream_type": "iframe"}

        # ১. সরাসরি m3u8 লিঙ্ক খোঁজা
        m3u8_matches = re.findall(r'(https?://[^\s"\'<>]+\.m3u8[^\s"\'<>]*)', html_content)
        if m3u8_matches:
            result["stream_url"] = m3u8_matches[0]
            result["stream_type"] = "hls"
            return result

        # ২. Base64 এনকোডেড m3u8 চেক করা
        b64_matches = re.findall(r'aHR0cHM6Ly[A0Za-z0-9+/=]+', html_content)
        for b64 in b64_matches:
            try:
                decoded = base64.b64decode(b64).decode('utf-8')
                if '.m3u8' in decoded:
                    result["stream_url"] = decoded
                    result["stream_type"] = "hls"
                    return result
            except Exception:
                pass

        # ৩. আইফ্রেম ফিল্টারিং (Ads ছাড়া মূল প্লেয়ার খোঁজা)
        soup = BeautifulSoup(html_content, 'html.parser')
        iframes = soup.find_all('iframe')
        ignored_domains = ['googletagmanager', 'facebook', 'analytics', 'disqus', 'ads', 'gtm']

        for iframe in iframes:
            src = iframe.get('src') or iframe.get('data-src') or ''
            if src and not any(d in src.lower() for d in ignored_domains):
                full_src = src if src.startswith('http') else f"https:{src}" if src.startswith('//') else f"{config.TARGET_BASE_URL.rstrip('/')}/{src.lstrip('/')}"
                result["stream_url"] = full_src
                result["stream_type"] = "iframe"
                break

        return result

    def extract_all_servers(self, match_url: str) -> List[Dict[str, str]]:
        """
        ম্যাচ পেজে থাকা সকল সার্ভার বাটনের (যেমন: Willow FHD, TAPMAD HD, Cricbuzz Fast)
        লিঙ্ক এবং স্ট্রিমিং সোর্স সংগ্রহ করে
        """
        soup = self.fetch_page(match_url)
        servers_list = []

        if not soup:
            return servers_list

        page_html = str(soup)

        # সার্ভার বাটন স্লেক্টর চিহ্নিত করা
        server_elements = soup.select('a.server-btn, button.server-btn, .servers-list a, .stream-buttons a, .server_btn, a[href*="server"], a[data-url], .btn-primary, div[class*="server"] a, div[class*="stream"] a')

        seen_servers = set()

        for elem in server_elements:
            server_name = elem.text.strip()
            if not server_name or len(server_name) > 30:
                continue

            target_url = elem.get('href') or elem.get('data-url') or elem.get('data-embed') or ''
            
            if server_name in seen_servers:
                continue
            
            seen_servers.add(server_name)

            if target_url and target_url != '#' and not target_url.startswith('javascript:'):
                full_target_url = target_url if target_url.startswith('http') else f"{config.TARGET_BASE_URL.rstrip('/')}/{target_url.lstrip('/')}"
                
                # নতুন সার্ভার পেজ স্ক্র্যাপ করে স্ট্রিম বের করা
                sub_soup = self.fetch_page(full_target_url)
                if sub_soup:
                    stream_data = self._parse_stream_source_from_html(str(sub_soup), full_target_url)
                else:
                    stream_data = {"stream_url": full_target_url, "stream_type": "iframe"}
            else:
                # পেজের ভেতরেই যদি ট্যাবের মাধ্যমে প্লেয়ার পরিবর্তন হয়
                stream_data = self._parse_stream_source_from_html(page_html, match_url)

            servers_list.append({
                "server_name": server_name,
                "stream_url": stream_data["stream_url"],
                "stream_type": stream_data["stream_type"],
                "embed_code": f'<iframe src="{stream_data["stream_url"]}" width="100%" height="100%" frameborder="0" allowfullscreen="true" scrolling="no" allow="autoplay; encrypted-media"></iframe>'
            })

        # যদি কোনো আলাদা বাটন না পাওয়া যায় তবে ডিফল্ট সার্ভার তৈরি
        if not servers_list:
            default_stream = self._parse_stream_source_from_html(page_html, match_url)
            servers_list.append({
                "server_name": "Main Server",
                "stream_url": default_stream["stream_url"],
                "stream_type": default_stream["stream_type"],
                "embed_code": f'<iframe src="{default_stream["stream_url"]}" width="100%" height="100%" frameborder="0" allowfullscreen="true" scrolling="no" allow="autoplay; encrypted-media"></iframe>'
            })

        return servers_list

    def get_all_matches(self) -> List[Dict[str, Any]]:
        """ মূল পেজ থেকে সকল সক্রিয় ম্যাচ ও তাদের সার্ভার কালেকশন করে """
        print(f"[+] স্ক্র্যাপ করা হচ্ছে: {config.TARGET_BASE_URL}")
        soup = self.fetch_page(config.TARGET_BASE_URL)
        matches_data = []

        if not soup:
            print("[-] পেজ লোড করা সম্ভব হয়নি।")
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

            # থাম্বনেইল এক্সট্র্যাক্ট করা
            img_tag = card.find('img') if card.name != 'a' else (card.find_parent().find('img') if card.find_parent() else None)
            existing_image_url = ""
            
            if img_tag:
                existing_image_url = img_tag.get('src') or img_tag.get('data-src') or ""
                if existing_image_url and not existing_image_url.startswith('http'):
                    existing_image_url = f"{config.TARGET_BASE_URL.rstrip('/')}/{existing_image_url.lstrip('/')}"

            print(f"  └─ লাইভ ম্যাচ প্রসেস হচ্ছে: {title[:40]}...")

            # ঐ ম্যাচের সব সার্ভার বের করা
            servers = self.extract_all_servers(href)

            matches_data.append({
                "title": title,
                "slug_url": href,
                "original_image": existing_image_url,
                "servers": servers
            })

        return matches_data
