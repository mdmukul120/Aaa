"""মূল অংশগুলোর দ্রুত পরীক্ষা (নেটওয়ার্ক/Supabase লাগে না)।

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import gzip
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import crawler  # noqa: E402
from config import NEURAL_BUNDLED  # noqa: E402
from ngmodel import NGramModel, detokenize, rank_words, tokenize  # noqa: E402


class TokenizerTest(unittest.TestCase):
    def test_case_and_joiners(self):
        self.assertEqual(tokenize("The the THE"), ["the", "the", "the"])
        # ZWNJ/ZWJ শব্দ ভাঙে না
        self.assertEqual(len(tokenize("র\u200d্যাব")), 1)
        self.assertEqual(len(tokenize("হক\u200cসাহেব")), 1)
        # নুকতা: আলাদা কোডপয়েন্ট (ড + ়) আর একক অক্ষর (ড়) একই শব্দ
        self.assertEqual(tokenize("বড\u09bc"), tokenize("বড়"))

    def test_roundtrip_readable(self):
        text = detokenize(tokenize("ঢাকা বাংলাদেশের রাজধানী। It is big, really."))
        self.assertIn("রাজধানী।", text)
        self.assertIn("big,", text)


class NGramTest(unittest.TestCase):
    def setUp(self):
        self.m = NGramModel()
        for _ in range(3):
            self.m.update("ঢাকা বাংলাদেশের রাজধানী। ঢাকা বাংলাদেশের বড় শহর। the cat sat on the mat.")

    def test_predict(self):
        words = [p["word"] for p in self.m.predict("ঢাকা", k=3)]
        self.assertEqual(words[0], "বাংলাদেশের")
        self.assertTrue(all(0 < p["p"] <= 1 for p in self.m.predict("the", k=5)))

    def test_budget_keeps_new_learning(self):
        """নতুন শেখা শব্দজোড়া বাজেটের মধ্যে থাকলে মুছে যায় না (আগের LOW_MEM বাগ)।"""
        self.m.update("নতুনশব্দ অদ্ভুতশব্দ দেখা গেল।")
        self.m.fit_budget(10_000)
        self.assertIn("অদ্ভুতশব্দ", [p["word"] for p in self.m.predict("নতুনশব্দ", k=5)])

    def test_budget_shrinks(self):
        before = self.m.size()
        self.m.fit_budget(max(4, before // 3))
        self.assertLess(self.m.size(), before)
        self.assertTrue(self.m.predict("ঢাকা", k=3))       # ঘন ঘন দেখা জোড়া টিকে থাকে

    def test_save_load_and_safe_unpickle(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.pkl.gz"
            self.m.save(path)
            m2 = NGramModel.load(path)
            self.assertEqual(m2.tokens, self.m.tokens)
            self.assertTrue(m2.is_current())

            marker = Path(d) / "pwned"

            class Evil:
                def __reduce__(self):
                    return (os.system, (f"echo x > {marker}",))

            with gzip.open(path, "wb") as f:
                pickle.dump({"version": 3, "tokens": 9, "uni": Evil()}, f)
            self.assertEqual(NGramModel.load(path).tokens, 0)   # প্রত্যাখ্যাত, খালি মডেল
            self.assertFalse(marker.exists())

            with gzip.open(path, "wb") as f:                    # পুরোনো টোকেনাইজারের ফাইল
                pickle.dump({"version": 3, "tokens": 9, "uni": {"The": 9}}, f)
            self.assertFalse(NGramModel.load(path).is_current())

    def test_rank_words(self):
        top = rank_words({"a": 0.5, "b": 0.3, "c": 0.2}, 2)
        self.assertEqual([t["word"] for t in top], ["a", "b"])
        # যুক্তিচিহ্ন সর্বোচ্চ একটা জায়গা নেয় — শব্দ থাকলে শব্দই আগে
        top = rank_words({",": 0.4, ".": 0.3, "a": 0.2, "b": 0.1}, 3)
        self.assertEqual(sorted(t["word"] for t in top), [",", "a", "b"])


class CrawlerSafetyTest(unittest.TestCase):
    def test_private_addresses_rejected(self):
        for url in ("http://127.0.0.1/", "http://localhost:8000/x", "http://10.1.2.3/",
                    "http://192.168.1.1/", "http://169.254.169.254/latest/meta-data/",
                    "http://[::1]/", "http://[::ffff:127.0.0.1]/", "ftp://example.com/",
                    "file:///etc/passwd", "http://93.184.216.34:22/", "notaurl"):
            self.assertFalse(crawler.is_public_url(url), url)

    def test_public_ip_allowed(self):
        self.assertTrue(crawler.is_public_url("http://93.184.216.34/page"))
        self.assertTrue(crawler.is_public_url("https://93.184.216.34:443/"))


class FeedParserTest(unittest.TestCase):
    RSS = """<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel><title>t</title>
    <item><title>এক</title><link>https://example.com/a</link></item>
    <item><title>দুই</title><link> https://example.com/b </link></item></channel></rss>"""
    ATOM = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
    <entry><title>x</title><link rel="alternate" href="https://example.com/c"/></entry>
    <entry><title>y</title><link href="https://example.com/d"/></entry></feed>"""

    @staticmethod
    def urls(feed) -> list[str]:
        return [e["url"] for e in crawler.parse_feed(feed)]

    def test_rss(self):
        self.assertEqual(self.urls(self.RSS), ["https://example.com/a", "https://example.com/b"])
        self.assertEqual(self.urls(self.RSS.encode("utf-8")), ["https://example.com/a", "https://example.com/b"])
        self.assertEqual(crawler.parse_feed(self.RSS)[0]["title"], "এক")

    def test_atom(self):
        self.assertEqual(self.urls(self.ATOM), ["https://example.com/c", "https://example.com/d"])

    def test_broken_xml_falls_back(self):
        broken = "<rss><channel><item><link>https://example.com/e</link></item><item>"
        self.assertEqual(self.urls(broken), ["https://example.com/e"])

    def test_entity_bomb_ignored(self):
        bomb = ('<?xml version="1.0"?><!DOCTYPE r [<!ENTITY a "aaaa"><!ENTITY b "&a;&a;&a;">]>'
                "<rss><channel><item><link>&b;</link></item></channel></rss>")
        self.assertEqual(crawler.parse_feed(bomb), [])


@unittest.skipUnless(NEURAL_BUNDLED.exists(), "models/shobdo-neural.npz নেই")
class NeuralTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from neural import NeuralLM
        except ImportError as e:            # numpy/sentencepiece ইনস্টল নেই
            raise unittest.SkipTest(str(e))
        cls.lm = NeuralLM.load(NEURAL_BUNDLED)

    def test_next_words_is_distribution(self):
        for text in ("বাংলাদেশের রাজধানী", "the capital of", ""):
            probs = self.lm.next_words(text)
            self.assertTrue(probs)
            self.assertTrue(all(p >= 0 for p in probs.values()))
            self.assertLessEqual(sum(probs.values()), 1.0001)

    def test_cache_gives_same_answer(self):
        """একটু একটু করে টাইপ করলে (KV-cache) আর একবারে দিলে একই উত্তর।"""
        from neural import NeuralLM
        text = "বাংলাদেশ দক্ষিণ এশিয়ার একটি"
        for i in range(4, len(text)):
            self.lm.next_words(text[:i])
        typed = self.lm.next_words(text)
        fresh = NeuralLM.load(NEURAL_BUNDLED).next_words(text)
        self.assertEqual(set(typed), set(fresh))
        for w in fresh:
            self.assertAlmostEqual(typed[w], fresh[w], places=4)

    def test_generate_and_blend(self):
        from neural import blend
        out = self.lm.generate("বাংলা ভাষা", max_words=8, temperature=0.7)
        self.assertIsInstance(out, str)
        mixed = blend({"a": 0.6, "b": 0.2}, {"b": 0.5, "c": 0.5}, 0.5)
        self.assertAlmostEqual(mixed["b"], 0.35, places=6)
        self.assertEqual(blend({"a": 1.0}, {}, 1.0)["a"], 1.0)


if __name__ == "__main__":
    unittest.main()
