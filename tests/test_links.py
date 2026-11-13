"""লিংক-পাঠক, স্পাইডার আর "সবচেয়ে মিলে যাওয়া উত্তর"-এর পরীক্ষা (নেটওয়ার্ক/Supabase লাগে না)।

    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import crawler  # noqa: E402
import db  # noqa: E402
import linkreader  # noqa: E402
import spider  # noqa: E402
from answer import Matcher, content_terms, norm_question, stem  # noqa: E402

HAVE_PDF = importlib.util.find_spec("pypdf") is not None


def make_pdf(lines: list[str]) -> bytes:
    """ছোট বৈধ এক-পাতার PDF (Helvetica)।"""
    esc = lambda t: t.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")  # noqa: E731
    content = ("BT /F1 12 Tf 50 780 Td 14 TL\n" + "\n".join(f"({esc(t)}) Tj T*" for t in lines) + "\nET").encode("latin-1")
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R "
            b"/Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offs = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % off for off in offs)
    return out + b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)


U = "https://example.com/data"


class SniffTest(unittest.TestCase):
    def test_kind_from_content_not_just_header(self):
        k = crawler.sniff_kind
        self.assertEqual(k(b'{"a": 1}', "text/plain"), "json")           # API অনেক সময় ভুল হেডার দেয়
        self.assertEqual(k(b"[1, 2]", "application/json"), "json")
        self.assertEqual(k(b"%PDF-1.7 ...", "application/octet-stream"), "pdf")
        self.assertEqual(k(b"<!DOCTYPE html><html>", "text/html"), "html")
        self.assertEqual(k(b"<?xml version='1.0'?><rss>", "application/xml"), "xml")
        self.assertEqual(k(b"hello world", "text/plain"), "text")
        self.assertEqual(k(b"\x89PNG\x00\x00\x01", "application/octet-stream"), "binary")


class JsonReaderTest(unittest.TestCase):
    def qa(self, obj):
        text = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
        return [(p["q"], p["a"]) for p in linkreader.read_json(text, U)[1]]

    def test_simple_pairs_and_key_variants(self):
        got = self.qa([{"q": "রাজধানী কোথায়?", "a": "ঢাকা"},
                       {"question": "Who wrote Gitanjali?", "answer": "Tagore"},
                       {"প্রশ্ন": "জাতীয় ফুল কী?", "উত্তর": "শাপলা"}])
        self.assertEqual(got, [("রাজধানী কোথায়?", "ঢাকা"), ("Who wrote Gitanjali?", "Tagore"),
                               ("জাতীয় ফুল কী?", "শাপলা")])

    def test_nested_api_alpaca_squad_chat_and_map(self):
        got = dict(self.qa({
            "data": [{"paragraphs": [{"context": "Dhaka is the capital of Bangladesh. " * 3,
                                      "qas": [{"question": "Capital of Bangladesh?",
                                               "answers": [{"text": "Dhaka"}]}]}]}],
            "items": [{"instruction": "Translate", "input": "water", "output": "পানি"}],
            "messages": [{"role": "user", "content": "তুমি কে?"},
                         {"role": "assistant", "content": "আমি শব্দ AI।"}],
            "faq": {"ঢাকা কোথায়?": "বাংলাদেশে", "সুন্দরবন কী?": "ম্যানগ্রোভ বন"}}))
        self.assertEqual(got["Capital of Bangladesh?"], "Dhaka")
        self.assertEqual(got["Translate water"], "পানি")
        self.assertEqual(got["তুমি কে?"], "আমি শব্দ AI।")
        self.assertEqual(got["ঢাকা কোথায়?"], "বাংলাদেশে")

    def test_json_lines(self):
        got = self.qa('{"prompt": "2+2?", "completion": "4"}\n{"prompt": "3+3 কত?", "completion": 6}\n')
        self.assertEqual(got, [("2+2?", "4"), ("3+3 কত?", "6")])

    def test_plain_api_text_becomes_document(self):
        docs, qa = linkreader.read_json(json.dumps(
            {"title": "বাংলাদেশ", "pageid": 7,
             "extract": "বাংলাদেশ দক্ষিণ এশিয়ার একটি সার্বভৌম রাষ্ট্র। এর রাজধানী ঢাকা। " * 4}, ensure_ascii=False), U)
        self.assertEqual(qa, [])
        self.assertEqual(docs[0]["title"], "বাংলাদেশ")
        self.assertEqual(docs[0]["lang"], "bn")
        self.assertIn("সার্বভৌম", docs[0]["content"])

    def test_qa_json_is_also_stored_as_text(self):
        pairs = [{"q": f"বাংলাদেশের {n} নম্বর বিভাগের নাম কী?", "a": f"এটি {n} নম্বর বিভাগ, রাজধানী ঢাকা থেকে দূরে।"}
                 for n in range(1, 5)]
        docs, qa = linkreader.read_json(json.dumps(pairs, ensure_ascii=False), U)
        self.assertEqual(len(qa), 4)
        self.assertIn("ঢাকা", docs[0]["content"])        # শব্দ-মডেলও এখান থেকে শেখে


class TextReaderTest(unittest.TestCase):
    def test_question_answer_lines(self):
        text = ("# প্রশ্নোত্তর\n\nপ্রশ্ন ১: জাতীয় ফুল কী?\nউত্তর: শাপলা।\n\n"
                "Q: National fruit?\nA: Jackfruit,\nwhich is large.\n\n"
                "প্রশ্ন: জাতীয় পাখি কী? উত্তর: দোয়েল।\n"
                "Q: An MCQ?\na) one\nb) two\n")
        docs, qa = linkreader.read_text(text, "https://example.com/a.md")
        got = dict((p["q"], p["a"]) for p in qa)
        self.assertEqual(got["জাতীয় ফুল কী?"], "শাপলা।")
        self.assertEqual(got["National fruit?"], "Jackfruit, which is large.")
        self.assertEqual(got["জাতীয় পাখি কী?"], "দোয়েল।")
        self.assertNotIn("An MCQ?", got)                  # "a) one" উত্তর নয়
        self.assertEqual(docs[0]["title"], "প্রশ্নোত্তর")

    def test_csv_columns(self):
        _, qa = linkreader.read_text('question,answer\n"রাজধানী কোথায়?","ঢাকা"\n"How many, total?","64"\n',
                                     "https://example.com/a.csv")
        self.assertEqual([(p["q"], p["a"]) for p in qa], [("রাজধানী কোথায়?", "ঢাকা"), ("How many, total?", "64")])

    def test_long_text_is_split_into_parts(self):
        big = ("এটি একটি লম্বা অনুচ্ছেদ যার মধ্যে অনেক বাক্য আছে। " * 40 + "\n\n") * 30
        docs, _ = linkreader.read_text(big, "https://example.com/big.txt")
        self.assertGreater(len(docs), 3)
        self.assertEqual(len({d["url"] for d in docs}), len(docs))         # প্রতিটি অংশের আলাদা ঠিকানা
        self.assertTrue(all(len(d["content"]) <= 12000 for d in docs))
        self.assertEqual(docs[0]["url"], "https://example.com/big.txt")


class HtmlReaderTest(unittest.TestCase):
    def test_faq_pairs_and_links(self):
        ld = {"@type": "FAQPage", "mainEntity": [{"@type": "Question", "name": "ফি কত?",
              "acceptedAnswer": {"@type": "Answer", "text": "<p>৫০০ টাকা</p>"}}]}
        html = ("<html><head><title>FAQ</title><script type='application/ld+json'>"
                + json.dumps(ld, ensure_ascii=False) + "</script></head><body>"
                "<details><summary>কখন খোলা?</summary>সকাল ৯টা</details>"
                "<a href='/x#frag'>x</a><a href='mailto:a@b.c'>m</a><a href='https://other.org/z'>z</a>"
                "</body></html>")
        docs, qa, links = linkreader.read_html(html, "https://example.com/faq", True)
        self.assertEqual([(p["q"], p["a"]) for p in qa], [("ফি কত?", "৫০০ টাকা"), ("কখন খোলা?", "সকাল ৯টা")])
        self.assertEqual(links, ["https://example.com/x", "https://other.org/z"])

    def test_short_line_page_is_not_dropped(self):
        html = "<html><body>" + "".join(
            f"<p>প্রশ্ন: রাজধানী কোথায়? উত্তর: ঢাকা। ({i})</p>" for i in range(12)) + "</body></html>"
        docs, qa, _ = linkreader.read_html(html, "https://example.com/q")
        self.assertEqual(len(docs), 1)
        self.assertEqual(len(qa), 12)


@unittest.skipUnless(HAVE_PDF, "pypdf ইনস্টল নেই")
class PdfReaderTest(unittest.TestCase):
    def test_text_pdf(self):
        pdf = make_pdf(["Shobdo handbook", "The Padma Bridge is a road-rail bridge across the Padma River in Bangladesh.",
                        "It connects Munshiganj to Shariatpur and Madaripur districts.",
                        "Question: When was the Padma Bridge opened?", "Answer: It was inaugurated on 25 June 2022."])
        self.assertEqual(crawler.sniff_kind(pdf, "application/octet-stream"), "pdf")
        docs, qa, err = linkreader.read_pdf(pdf, "https://example.com/a.pdf")
        self.assertIsNone(err)
        self.assertIn("Padma Bridge", docs[0]["content"])
        self.assertEqual(qa, [{"q": "When was the Padma Bridge opened?", "a": "It was inaugurated on 25 June 2022."}])

    def test_empty_or_broken_pdf_gives_a_reason(self):
        self.assertIn("স্ক্যান", linkreader.read_pdf(make_pdf(["x"]), U)[2])
        self.assertIn("খোলা যায়নি", linkreader.read_pdf(b"%PDF-1.4 garbage", U)[2])


class ReadLinkTest(unittest.TestCase):
    def test_dispatch_by_kind(self):
        raw = json.dumps([{"q": "রাজধানী কোথায়?", "a": "ঢাকা"}, {"q": "রাজধানী কোথায়?", "a": "ঢাকা"}],
                         ensure_ascii=False).encode()
        with mock.patch.object(crawler, "fetch_any", return_value={
                "raw": raw, "ctype": "text/plain", "kind": "json", "url": "https://example.com/qa"}):
            res = linkreader.read_link("https://example.com/qa")
        self.assertTrue(res["ok"])
        self.assertEqual(res["kind"], "json")
        self.assertEqual(len(res["qa"]), 1)                  # একই প্রশ্ন একবারই

    def test_error_is_passed_through(self):
        with mock.patch.object(crawler, "fetch_any", return_value={"error": "সার্ভার 403 দিয়েছে", "status": 403}):
            res = linkreader.read_link("https://example.com/x")
        self.assertFalse(res["ok"])
        self.assertIn("403", res["error"])


class MatcherTest(unittest.TestCase):
    STORED = ["বাংলাদেশের রাজধানী কোথায়?", "পদ্মা সেতু কত সালে চালু হয়?", "পদ্মা সেতু কোথায় অবস্থিত?",
              "বাংলাদেশের জাতীয় ফুল কী?", "What is the capital of Bangladesh?", "তুমি কে?"]

    def best(self, q):
        m = Matcher()
        ranked = m.best_qa(q, [{"question": s, "answer": "x"} for s in self.STORED])
        return ranked[0]["question"], ranked[0]["score"]

    def test_same_question_in_other_words(self):
        for q, want in [("বাংলাদেশের রাজধানীর নাম কী", self.STORED[0]),
                        ("বাংলাদেশের রাজধানি কোথায়", self.STORED[0]),          # বানান ভুল
                        ("রাজধানী কোথায় বাংলাদেশের?", self.STORED[0]),         # শব্দের ক্রম বদল
                        ("পদ্মা সেতু কবে চালু হয়েছে", self.STORED[1]),
                        ("পদ্মা সেতু কোথায়", self.STORED[2]),
                        ("জাতীয় ফুলের নাম কি", self.STORED[3]),
                        ("capital of bangladesh", self.STORED[4]),
                        ("তুমি কে", self.STORED[5])]:
            got, score = self.best(q)
            self.assertEqual(got, want, q)
            self.assertGreaterEqual(score, 0.62, q)

    def test_different_question_scores_low(self):
        for q in ["আজকের আবহাওয়া কেমন", "ভারতের রাজধানী কোথায়?", "জাতীয় ফল কী", "চাঁদে প্রথম কে গিয়েছিল?"]:
            self.assertLess(self.best(q)[1], 0.6, q)

    def test_where_vs_when_is_not_confused(self):
        m = Matcher()
        self.assertGreater(m.qa_score("পদ্মা সেতু কবে", "পদ্মা সেতু কত সালে চালু হয়?"),
                           m.qa_score("পদ্মা সেতু কবে", "পদ্মা সেতু কোথায় অবস্থিত?"))

    def test_best_passage_picks_the_right_sentence(self):
        text = ("বাংলাদেশ দক্ষিণ এশিয়ার একটি দেশ। এর রাজধানী ঢাকা এবং সরকারি ভাষা বাংলা। "
                "পদ্মা সেতু ২০২২ সালের ২৫ জুন উদ্বোধন করা হয়। সুন্দরবন বিশ্বের বৃহত্তম ম্যানগ্রোভ বন।")
        m = Matcher()
        score, passage = m.best_passage("পদ্মা সেতু কবে উদ্বোধন হয়?", text)
        self.assertEqual(passage, "পদ্মা সেতু ২০২২ সালের ২৫ জুন উদ্বোধন করা হয়।")
        self.assertGreaterEqual(score, 0.8)
        self.assertIn("ম্যানগ্রোভ", m.best_passage("বৃহত্তম ম্যানগ্রোভ বন কোনটি", text)[1])
        self.assertEqual(m.best_passage("চাঁদে কে প্রথম গিয়েছিল?", text), (0.0, ""))

    def test_helpers(self):
        self.assertEqual(norm_question("  বাংলাদেশের রাজধানী  কোথায়?? "), norm_question("বাংলাদেশের রাজধানী কোথায়"))
        self.assertEqual(stem("ঢাকার"), stem("ঢাকা"))
        self.assertEqual(stem("cities"), "city")
        self.assertEqual(content_terms("What is the capital of Bangladesh?"), ["capital", "bangladesh"])


class _FakeRobots:
    blocked_all, note = False, ""

    def __init__(self, _url):
        pass

    def allowed(self, url):
        return "/private/" not in url

    def delay(self, default):
        return default


class SpiderTest(unittest.TestCase):
    SITE = {
        "https://example.com/": ["/a?utm_source=x#top", "/b", "/private/x", "https://other.org/", "/pic.jpg",
                                 "/login", "https://www.example.com/a"],
        "https://example.com/a": ["/a/1", "/b"],
        "https://example.com/b": ["/"],
        "https://example.com/a/1": ["/a/1/deep"],
        "https://example.com/a/1/deep": [],
    }

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patches = [
            mock.patch.object(db, "QUEUE_PATH", Path(self.tmp.name) / "queue.json"),
            mock.patch.object(db, "_queue_mem", None),
            mock.patch.object(db, "configured", lambda: False),
            mock.patch.object(crawler, "is_public_url", lambda u: True),
        ]
        for p in self.patches:
            p.start()
        self.read_log: list[str] = []

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.tmp.cleanup()

    def read(self, url, want_links=False):
        self.read_log.append(url)
        links = ["https://example.com" + x if x.startswith("/") else x for x in self.SITE.get(url, [])]
        return {"ok": True, "kind": "html", "url": url, "error": None, "qa": [],
                "docs": [{"url": url, "title": url, "content": "x" * 300, "lang": "en"}],
                "links": links if want_links else []}

    def run_round(self, pages, depth=2, **kw):
        return list(spider.crawl("https://example.com/", pages=pages, max_depth=depth, delay=0,
                                 read=self.read, robots_factory=_FakeRobots, sleep=lambda s: None, **kw))

    def test_norm_url(self):
        n = spider.norm_url
        self.assertEqual(n("HTTPS://Example.com:443/a?utm_source=x&id=2#frag"), "https://example.com/a?id=2")
        self.assertEqual(n("https://example.com"), "https://example.com/")
        self.assertIsNone(n("ftp://example.com/x"))
        self.assertIsNone(n("javascript:void(0)"))

    def test_stays_on_site_obeys_robots_and_resumes(self):
        first = self.run_round(2)
        self.assertEqual([e["url"] for e in first if e["event"] == "page"],
                         ["https://example.com/", "https://example.com/a"])
        second = self.run_round(10)
        pages = [e["url"] for e in second if e["event"] == "page"]
        self.assertEqual(pages, ["https://example.com/b", "https://example.com/a/1"])   # যেখানে থেমেছিল সেখান থেকে
        self.assertEqual([e["url"] for e in second if e["event"] == "skip"], ["https://example.com/private/x"])
        self.assertEqual(len(self.read_log), len(set(self.read_log)))                    # কোনো URL দু'বার নয়
        self.assertFalse([u for u in self.read_log if "other.org" in u or u.endswith((".jpg", "/login"))])
        self.assertNotIn("https://example.com/a/1/deep", self.read_log)                  # গভীরতা ২-এর বাইরে
        self.assertEqual(self.run_round(10), [])                                         # আর কিছু বাকি নেই
        self.assertEqual(db.queue_stats("example.com"), {"pending": 0, "done": 4, "failed": 1, "total": 5})

    def test_restart_reads_again(self):
        self.run_round(10)
        again = self.run_round(1, restart=True)
        self.assertEqual([e["url"] for e in again if e["event"] == "page"], ["https://example.com/"])

    def test_redirect_to_other_site_is_not_stored(self):
        def read(url, want_links=False):
            return {"ok": True, "kind": "html", "url": "https://evil.org/x", "error": None, "qa": [],
                    "docs": [{"url": "https://evil.org/x", "title": "", "content": "y" * 300}], "links": []}
        got = list(spider.crawl("https://example.com/", pages=3, delay=0, read=read,
                                robots_factory=_FakeRobots, sleep=lambda s: None))
        self.assertEqual([e["event"] for e in got], ["skip"])

    def test_private_address_is_refused(self):
        with mock.patch.object(crawler, "is_public_url", lambda u: False):
            got = list(spider.crawl("http://127.0.0.1/", read=self.read, robots_factory=_FakeRobots))
        self.assertEqual(got[0]["event"], "stop")
        self.assertEqual(self.read_log, [])


class RobotsTest(unittest.TestCase):
    def make(self, result):
        return spider.Robots("https://example.com/page", fetch=lambda url, timeout=10: result)

    def test_rules_and_delay(self):
        r = self.make({"raw": b"User-agent: *\nDisallow: /private/\nCrawl-delay: 4\n", "kind": "text"})
        self.assertTrue(r.allowed("https://example.com/open"))
        self.assertFalse(r.allowed("https://example.com/private/x"))
        self.assertEqual(r.delay(1.0), 4.0)
        self.assertEqual(self.make({"raw": b"User-agent: *\nCrawl-delay: 999\n", "kind": "text"}).delay(1.0), 10.0)

    def test_missing_robots_allows_but_errors_block(self):
        self.assertTrue(self.make({"error": "x", "status": 404}).allowed("https://example.com/a"))
        self.assertTrue(self.make({"raw": b"<html>not robots</html>", "kind": "html"}).allowed("https://example.com/a"))
        self.assertTrue(self.make({"error": "x", "status": 403}).blocked_all)
        self.assertTrue(self.make({"error": "সংযোগ ব্যর্থ"}).blocked_all)
        self.assertTrue(self.make({"error": "x", "status": 503}).blocked_all)


class QaStoreTest(unittest.TestCase):
    def test_local_store_upserts_and_matches(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(db, "QA_PATH", Path(tmp) / "qa.jsonl"), \
                mock.patch.object(db, "configured", lambda: False):
            self.assertEqual(db.insert_qa([{"q": "রাজধানী কোথায়?", "a": "চট্টগ্রাম"}, {"q": "", "a": "x"}], U), 1)
            db.insert_qa([{"q": "রাজধানী  কোথায়", "a": "ঢাকা"}, {"q": "জাতীয় ফুল কী?", "a": "শাপলা"}], U)
            self.assertEqual(db.qa_count(), 2)                       # একই প্রশ্ন → একটাই সারি
            hit = db.match_qa(norm_question("রাজধানী কোথায়"))[0]
            self.assertEqual(hit["answer"], "ঢাকা")                   # পরের উত্তরটাই থাকে (সংশোধন)


if __name__ == "__main__":
    unittest.main()
