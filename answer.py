"""প্রশ্নের সাথে সবচেয়ে মিলে যাওয়া উত্তর খোঁজা (কোনো বড় মডেল ছাড়া — শব্দ-মিল দিয়ে)।

দুই রকম উৎস:
  • প্রশ্ন-উত্তর জোড়া  → জমানো প্রশ্নের সাথে ব্যবহারকারীর প্রশ্ন কতটা মেলে (qa_score)
  • সাধারণ লেখা        → লেখার কোন ১–৩ বাক্যে প্রশ্নের শব্দগুলো সবচেয়ে বেশি আছে (best_passage)

মিলের হিসাব: গুরুত্বপূর্ণ শব্দ (বিরল শব্দের ওজন বেশি) + বাংলা বিভক্তি ছেঁটে তুলনা
("ঢাকার" = "ঢাকা") + বানান-ভুলের জন্য অক্ষর-জোড়ার মিল। স্কোর ০ থেকে ১।
"""
from __future__ import annotations

import math
import re
from typing import Callable

from ngmodel import SENT_SPLIT, WORD_RE, normalize, tokenize

# প্রশ্নের ধরন — "কোথায়" আর "কবে" গুলিয়ে ফেললে ভুল উত্তর আসে, তাই আলাদা করে মেলানো হয়
_QTYPES = {
    "where": ("কোথায়", "কোথা", "কোথাকার", "কোনখানে", "where"),
    "when": ("কবে", "কখন", "সালে", "তারিখে", "when"),
    "who": ("কে", "কার", "কাকে", "কারা", "কাদের", "who", "whom", "whose"),
    "howmany": ("কত", "কতটি", "কতটা", "কয়টি", "কয়টা", "কতজন", "কয়জন", "many", "much"),
    "why": ("কেন", "why"),
    "how": ("কীভাবে", "কিভাবে", "কেমন", "how"),
    "what": ("কী", "কি", "কোন", "কোনটি", "what", "which"),
}
_QWORD = {normalize(w): t for t, ws in _QTYPES.items() for w in ws}

STOP = {normalize(w) for w in """
কী কি কে কার কাকে কারা কোন কোনটি কোথায় কোথা কবে কখন কেন কীভাবে কিভাবে কেমন কত কতটি কয়টি কতজন
এবং ও আর বা কিন্তু তবে যে যা যার তা তার এই ওই সেই এটি এটা সেটি সেটা একটি একটা হয় হয়েছে হয়েছিল হলো হল হবে
ছিল ছিলো আছে নেই করে করা করেন করেছে থেকে দিয়ে জন্য সাথে সঙ্গে মধ্যে উপর নিয়ে নাম বলে বলা হলে তো না নয়
আমি তুমি আপনি সে তিনি তারা আমরা আমার তোমার আপনার
the a an of in on at to for from by with and or is are was were be been being do does did has have had
it its this that these those as what which who whom whose when where why how many much there their
i you he she we they me my your his her our about into than then so if not no can could would should will
""".split()}

_BN_SUFFIX = ("গুলোকে", "গুলোর", "গুলোতে", "দেরকে", "গুলো", "গুলি", "দের", "েরা", "টিকে", "টির", "টাকে",
              "য়ের", "ের", "কে", "তে", "রা", "টি", "টা", "য়", "র", "ে", "ই", "ও")
_BN_SUFFIX = tuple(normalize(x) for x in _BN_SUFFIX)
_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
_HAS_NUM = re.compile(r"[0-9০-৯]")


def stem(w: str) -> str:
    """বাংলা বিভক্তি/ইংরেজি বহুবচন ছাঁটা — শুধু তুলনার জন্য (ছাঁটার পরে অন্তত ৩ অক্ষর থাকতে হবে)।"""
    if "a" <= w[:1] <= "z":
        if len(w) > 4 and w.endswith("ies"):
            return w[:-3] + "y"
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            return w[:-1]
        return w
    w = w.translate(_BN_DIGITS)
    for suf in _BN_SUFFIX:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[:-len(suf)]
    return w


def words(text: str) -> list[str]:
    return [t for t in tokenize(text) if WORD_RE.match(t)]


def norm_question(text: str) -> str:
    """তুলনা ও ডুপ্লিকেট ধরার জন্য প্রশ্নের এক রূপ (ছোট হাতের, চিহ্ন ছাড়া)।"""
    return " ".join(words(text))[:500]


def qtypes(tokens: list[str]) -> set[str]:
    """প্রশ্নের ধরন(গুলো) — "কত সালে" একসাথে howmany ও when।"""
    return {_QWORD[t] for t in tokens if t in _QWORD}


def _clash(a: set[str], b: set[str]) -> bool:
    """দুই প্রশ্ন স্পষ্টতই আলাদা কিছু জানতে চাইছে? ("কী/কোন" সবার সাথে মানানসই।)"""
    a, b = a - {"what"}, b - {"what"}
    return bool(a and b and not (a & b))


def content_terms(text: str) -> list[str]:
    """গুরুত্বপূর্ণ শব্দের stem (প্রশ্নবাচক/সাহায্যকারী শব্দ বাদ); সব বাদ পড়লে সব শব্দই।"""
    toks = words(text)
    out = [stem(t) for t in toks if t not in STOP]
    return list(dict.fromkeys(out or [stem(t) for t in toks]))


def _bigrams(w: str) -> set[str]:
    w = f" {w} "
    return {w[i:i + 2] for i in range(len(w) - 1)}


def term_match(a: str, b: str) -> float:
    """দুটো stem কতটা এক: হুবহু ১.০ · একটা আরেকটার শুরু ০.৮৫ · বানান কাছাকাছি ০.৬ · নইলে ০।"""
    if a == b:
        return 1.0
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    if len(short) >= 4 and long_.startswith(short):
        return 0.85
    if len(short) >= 4:
        x, y = _bigrams(a), _bigrams(b)
        if 2 * len(x & y) / (len(x) + len(y)) >= 0.75:
            return 0.6
    return 0.0


def _cover(terms: list[str], pool: set[str], weight: Callable[[str], float]) -> tuple[float, int]:
    """terms-এর কত অংশ (ওজনসহ) pool-এ আছে; সাথে কয়টা শব্দ মিলেছে।"""
    total = got = 0.0
    hits = 0
    for t in terms:
        w = weight(t)
        total += w
        m = 1.0 if t in pool else max((term_match(t, p) for p in pool), default=0.0)
        if m:
            hits += 1
            got += w * m
    return (got / total if total else 0.0), hits


class Matcher:
    """idf(word) দিলে বিরল শব্দ বেশি গুরুত্ব পায় (না দিলে সব শব্দ সমান)।"""

    def __init__(self, count: Callable[[str], int] | None = None, total: int = 0) -> None:
        self._count, self._total = count, max(total, 1)

    def weight(self, term: str) -> float:
        if self._count is None:
            return 1.0
        c = self._count(term)
        return 0.3 + math.log(1.0 + self._total / (1.0 + c)) / math.log(1.0 + self._total)

    # ---- প্রশ্ন বনাম জমানো প্রশ্ন ----
    def qa_score(self, query: str, question: str) -> float:
        qa, qb = words(query), words(question)
        if not qa or not qb:
            return 0.0
        if qa == qb:
            return 1.0
        ta, tb = content_terms(query), content_terms(question)
        rec, _ = _cover(ta, set(tb), self.weight)
        prec, _ = _cover(tb, set(ta), self.weight)
        f1 = 2 * rec * prec / (rec + prec) if rec + prec else 0.0
        sa, sb = "".join(qa), "".join(qb)
        ga = {sa[i:i + 3] for i in range(max(1, len(sa) - 2))}
        gb = {sb[i:i + 3] for i in range(max(1, len(sb) - 2))}
        dice = 2 * len(ga & gb) / (len(ga) + len(gb))
        score = 0.78 * f1 + 0.22 * dice
        if _clash(qtypes(qa), qtypes(qb)):
            score *= 0.6                      # "কোথায়" বনাম "কবে" — আলাদা প্রশ্ন
        return round(min(score, 0.99), 4)

    def best_qa(self, query: str, pairs: list[dict]) -> list[dict]:
        """pairs: [{"question", "answer", …}] → স্কোরসহ, বড় থেকে ছোট।"""
        out = [{**p, "score": self.qa_score(query, p.get("question", ""))} for p in pairs]
        out.sort(key=lambda p: -p["score"])
        return out

    def title_cover(self, query: str, title: str) -> float:
        """প্রশ্নের গুরুত্বপূর্ণ শব্দের কত অংশ শিরোনামে আছে (০–১)।"""
        return _cover(content_terms(query), {stem(t) for t in words(title)}, self.weight)[0]

    # ---- প্রশ্ন বনাম লেখার অংশ ----
    def best_passage(self, query: str, text: str, max_chars: int = 420) -> tuple[float, str]:
        """লেখার যে ১–৩টি পরপর বাক্যে প্রশ্নের শব্দ সবচেয়ে বেশি আছে — (স্কোর, অংশ)।

        অন্তত দুটো গুরুত্বপূর্ণ শব্দ মিলতে হবে (প্রশ্নে একটাই থাকলে সেটা), নইলে (0, "")।
        """
        terms = content_terms(query)[:12]
        if not terms or not text:
            return 0.0, ""
        kinds = qtypes(words(query))
        sents = [s.strip() for s in SENT_SPLIT.split(text) if len(s.strip()) > 8][:400]
        if not sents:
            return 0.0, ""
        weights = [self.weight(t) for t in terms]
        total = sum(weights)
        # প্রতি বাক্যে প্রতি শব্দের মিল একবারই হিসাব (হুবহু বা উপসর্গ — লেখায় বানান-ভুল ধরা লাগে না)
        rows: list[list[float]] = []
        firsts: list[str] = []
        for s in sents:
            ws = [stem(t) for t in words(s)]
            pool = set(ws)
            firsts.append(ws[0] if ws else "")
            rows.append([1.0 if t in pool else
                         0.85 if len(t) >= 4 and any(p.startswith(t) or (len(p) >= 4 and t.startswith(p))
                                                     for p in pool) else 0.0
                         for t in terms])
        need = min(2, len(terms))
        best = (0.0, "")
        for i in range(len(sents)):
            if not any(rows[i]):
                continue                      # অংশটা এমন বাক্য দিয়ে শুরু হোক যাতে অন্তত একটা শব্দ আছে
            chunk = ""
            got = [0.0] * len(terms)
            for j in range(i, min(i + 3, len(sents))):
                nxt = (chunk + " " + sents[j]).strip()
                if chunk and len(nxt) > max_chars:
                    break
                chunk = nxt
                got = [max(a, b) for a, b in zip(got, rows[j])]
                if sum(1 for g in got if g) < need:
                    continue
                score = sum(w * g for w, g in zip(weights, got)) / total
                score *= 0.9 if j == i else 0.86 if j == i + 1 else 0.82
                if kinds & {"when", "howmany"} and _HAS_NUM.search(chunk):
                    score += 0.06             # "কবে/কত" প্রশ্নের উত্তরে সাধারণত সংখ্যা থাকে
                if j == i and firsts[i] == terms[0]:
                    score += 0.03             # বাক্যটা প্রশ্নের বিষয় দিয়েই শুরু ("ঢাকা হলো …")
                if score > best[0] + 1e-9:
                    best = (score, chunk[:max_chars + 200])
        return round(min(best[0], 0.9), 4), best[1]


def lead(text: str, max_chars: int = 420) -> str:
    """লেখার শুরুর কয়েকটা পুরো বাক্য (সংক্ষিপ্ত পরিচয় হিসেবে)।"""
    out = ""
    for s in SENT_SPLIT.split(text.strip()):
        s = s.strip()
        if not s:
            continue
        if out and len(out) + len(s) + 1 > max_chars:
            break
        out = (out + " " + s).strip()
    return out[:max_chars + 200]


def confidence(score: float) -> str:
    return "high" if score >= 0.8 else "medium" if score >= 0.62 else "low" if score >= 0.45 else "none"
