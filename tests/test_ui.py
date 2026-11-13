"""UI ফাইলগুলোর দ্রুত পরীক্ষা (ব্রাউজার লাগে না)।

static/ = মডেল-সার্ভার নিজে যে পাতা দেয়; vercel/ = Vercel-এ যে পাতা ওঠে।
দুই জায়গার পাতা একই থাকতে হবে — পার্থক্য শুধু Vercel-এর পাতায় config.js-এর ট্যাগ।
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_TAG = '<script src="/config.js"></script>'
PAGES = ("index.html", "studio.html")


def read(folder: str, name: str) -> str:
    return (ROOT / folder / name).read_text(encoding="utf-8")


class UiFiles(unittest.TestCase):
    def test_static_and_vercel_pages_match(self):
        for name in PAGES:
            with self.subTest(page=name):
                v = read("vercel", name)
                self.assertEqual(v.count(CONFIG_TAG), 1)
                self.assertNotIn(CONFIG_TAG, read("static", name))
                strip = lambda s: re.sub(r"\n\s*\n+", "\n", s.replace(CONFIG_TAG, ""))   # noqa: E731
                self.assertEqual(strip(v), strip(read("static", name)))

    def test_config_is_loaded_before_the_app_script(self):
        for name in PAGES:
            v = read("vercel", name)
            self.assertLess(v.index(CONFIG_TAG), v.rindex("<script>"), name)

    def test_chat_page_is_self_contained(self):
        """কোনো CDN/বাইরের ফাইল নয় — এক ফাইলেই সব।"""
        s = read("static", "index.html")
        self.assertEqual(re.findall(r'<(?:script|link|img)[^>]+(?:src|href)="https?://', s), [])

    def test_chat_page_has_the_parts_the_script_needs(self):
        s = read("static", "index.html")
        for part in ('id="input"', 'id="send"', 'id="msgs"', 'id="chatList"', 'id="newChat"',
                     'id="modelMenu"', 'id="settings"', 'href="/studio"',
                     '"/api/ask"', '"/api/generate"', '"/healthz"'):
            self.assertIn(part, s, part)

    def test_studio_keeps_learning_tools_and_links_back(self):
        s = read("static", "studio.html")
        for part in ('id="spiderBtn"', 'id="urlBtn"', 'id="askBtn"', 'id="predictInput"', '<a class="pill" href="/"'):
            self.assertIn(part, s, part)

    def test_server_serves_both_pages(self):
        src = (ROOT / "server.py").read_text(encoding="utf-8")
        m = re.search(r"^PAGES = \{(.*?)\}", src, re.S | re.M)
        self.assertIsNotNone(m)
        routes = dict(re.findall(r'"(/[^"]*)":\s*"([^"]+)"', m.group(1)))
        self.assertEqual(routes.get("/"), "index.html")
        self.assertEqual(routes.get("/studio"), "studio.html")
        for name in set(routes.values()):
            self.assertTrue((ROOT / "static" / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
