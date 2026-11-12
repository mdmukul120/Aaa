"""পরবর্তী-শব্দ অনুমানের ভাষা-মডেল (Interpolated n-gram LM + skip-gram)।

P(w | a,b) = λ3·P3(a,b,w) + λ2·P2(b,w) + λs·Ps(a,w) + λ1·P1(w)

  P3 = c(a,b,w)/c(a,b)   trigram — ঠিক আগের দুই শব্দের ধারা
  P2 = c(b,w)/c(b)       bigram  — ঠিক আগের শব্দের ধারা
  Ps = c(a,w)/c(a)       skip-gram — আগের দুই শব্দের প্রথমটি (মাঝের শব্দ বাদ)
  P1 = c(w)/N            unigram — সাধারণ জনপ্রিয়তা

বাংলা ও ইংরেজি — একই টোকেনাইজার। মডেল ফাইল gzip করা pickle।
নিউরাল মডেল (neural.py) একই টোকেনাইজার ব্যবহার করে — তাই দুই মডেলের "শব্দ" একই।
"""
from __future__ import annotations

import gzip
import json
import math
import pickle
import random
import re
import unicodedata
from collections import Counter

from config import LOW_MEM, MAX_BIGRAM_CTX, MAX_SKIP_CTX, MAX_TRIGRAM_CTX, MODEL_PATH

SEP = "\u0001"
PUNCT = set("।,.;:!?…'\"“”‘’()[]{}<>—–-/%&*+=|\\@#$^_~`")
SENT_END = ("।", ".", "!", "?")

# ইংরেজি শব্দ | বাংলা শব্দ (ভেতরের ZWJ/ZWNJ সহ — যেমন "র‍্যাব") | সংখ্যা | একক চিহ্ন
_BN = r"[\u0980-\u09FF]+(?:[\u200c\u200d]+[\u0980-\u09FF]+)*"
_EN = r"[A-Za-z]+(?:'[A-Za-z]+)*"
_NUM = r"\d+(?:[.,]\d+)*"
TOKEN_RE = re.compile(rf"{_EN}|{_BN}|{_NUM}|[^\s]")
WORD_RE = re.compile(rf"^(?:{_EN}|{_BN}|{_NUM})$")
SENT_SPLIT = re.compile(r"(?<=[.!?।])\s+|\n+")

# অদৃশ্য অক্ষর যেগুলো শব্দের অংশ নয় (ZWSP, BOM, soft hyphen, word joiner, LRM/RLM)
_INVISIBLE = re.compile("[\u200b\u200e\u200f\u2060\ufeff\u00ad]")
# শব্দের ভেতরে নয় এমন জায়গার ZWJ/ZWNJ (শুরুতে, শেষে, বা অ-বাংলা অক্ষরের পাশে)
_STRAY_ZW = re.compile(r"(?<![\u0980-\u09FF])[\u200c\u200d]+|[\u200c\u200d]+(?![\u0980-\u09FF])")


# টোকেনাইজার বদলালে এই সংখ্যা বাড়ান — পুরোনো সেভ করা মডেল তখন নিজে থেকেই নতুন করে গড়া হয়
TOK_VERSION = 2


def normalize(text: str) -> str:
    """ইউনিকোড এক রূপে আনা — একই শব্দ যেন দুই রকম বানানে আলাদা না গোনা হয়।

    NFC বাংলার য়/ড়/ঢ়-কে সবসময় (য/ড/ঢ + ়) রূপে আনে, ো/ৌ-কে এক অক্ষরে জোড়ে।
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00a0", " ").replace("’", "'").replace("‘", "'")
    text = _INVISIBLE.sub("", text)
    return _STRAY_ZW.sub("", text)


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for m in TOKEN_RE.finditer(normalize(text)):
        t = m.group(0)
        out.append(t.lower() if "a" <= t[0].lower() <= "z" else t)
    return out


def detokenize(tokens: list[str]) -> str:
    out = ""
    for t in tokens:
        if not out:
            out = t
        elif t in PUNCT:
            out += t
        elif out.endswith(("(", "[", "{", "“", '"', "'")):
            out += t
        else:
            out += " " + t
    return out


def rank_words(scores: dict[str, float], k: int = 6) -> list[dict]:
    """স্কোর থেকে সেরা k পরামর্শ: শব্দ আগে, যুক্তিচিহ্ন সর্বোচ্চ একটি।"""
    if not scores:
        return []
    total = sum(scores.values()) or 1.0
    top = sorted(scores.items(), key=lambda kv: -kv[1])[:k + 6]
    words = [(w, s) for w, s in top if WORD_RE.match(w)]
    puncts = [(w, s) for w, s in top if not WORD_RE.match(w)]
    top = words[:k - 1] + puncts[:1]
    top += words[k - 1:] + puncts[1:]            # জায়গা ফাঁকা থাকলে বাকিগুলো দিয়ে ভরা
    top = sorted(top[:k], key=lambda kv: -kv[1])
    return [{"word": w, "p": round(s / total, 4)} for w, s in top]


class _SafeUnpickler(pickle.Unpickler):
    """মডেল ফাইলে শুধু সাধারণ dict/Counter থাকে — অন্য কোনো ক্লাস লোড করা হয় না,
    তাই বদলে দেওয়া/নষ্ট ফাইল দিয়ে সার্ভারে কোড চালানো যায় না।"""

    _ALLOWED = {("collections", "Counter"), ("collections", "defaultdict"),
                ("collections", "OrderedDict"), ("builtins", "dict"), ("builtins", "list"),
                ("builtins", "set"), ("builtins", "tuple"), ("builtins", "int"),
                ("builtins", "float"), ("builtins", "str")}

    def find_class(self, module: str, name: str):
        if (module, name) in self._ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"অনুমোদিত নয়: {module}.{name}")


class NGramModel:
    def __init__(self) -> None:
        self.uni: Counter = Counter()
        self.bi: dict[str, dict[str, int]] = {}
        self.tri: dict[str, dict[str, int]] = {}
        self.skip: dict[str, dict[str, int]] = {}
        self.bi_tot: dict[str, int] = {}
        self.tri_tot: dict[str, int] = {}
        self.skip_tot: dict[str, int] = {}
        self.tokens = 0
        self.sentences = 0
        self.use_skip = not LOW_MEM   # কম-মেমরি হোস্টে skip-gram রাখা হয় না

    # ------------------------------------------------------------------ #
    def update(self, text: str) -> int:
        n = 0
        for sent in SENT_SPLIT.split(text):
            toks = tokenize(sent)
            if len(toks) < 2:
                continue
            n += len(toks)
            self.tokens += len(toks)
            self.sentences += 1
            prev2, prev1 = "<s>", "<s>"
            for w in toks + ["</s>"]:
                self.uni[w] += 1
                self._add(self.bi, self.bi_tot, prev1, w)
                self._add(self.tri, self.tri_tot, prev2 + SEP + prev1, w)
                if self.use_skip:
                    self._add(self.skip, self.skip_tot, prev2, w)
                prev2, prev1 = prev1, w
        return n

    @staticmethod
    def _add(store: dict[str, dict[str, int]], tot: dict[str, int], ctx: str, w: str) -> None:
        d = store.get(ctx)
        if d is None:
            d = store[ctx] = {}
        d[w] = d.get(w, 0) + 1
        tot[ctx] = tot.get(ctx, 0) + 1

    def prune(self) -> None:
        if len(self.tri) > MAX_TRIGRAM_CTX:
            keep = sorted(self.tri_tot.items(), key=lambda kv: -kv[1])[:MAX_TRIGRAM_CTX]
            self.tri = {k: self.tri[k] for k, _ in keep}
            self.tri_tot = dict(keep)
        if len(self.bi) > MAX_BIGRAM_CTX:
            keep = sorted(self.bi_tot.items(), key=lambda kv: -kv[1])[:MAX_BIGRAM_CTX]
            self.bi = {k: self.bi[k] for k, _ in keep}
            self.bi_tot = dict(keep)
        if len(self.skip) > MAX_SKIP_CTX:
            keep = sorted(self.skip_tot.items(), key=lambda kv: -kv[1])[:MAX_SKIP_CTX]
            self.skip = {k: self.skip[k] for k, _ in keep}
            self.skip_tot = dict(keep)

    # ------------------------------------------------------------------ #
    def size(self) -> int:
        """মেমরির মোটামুটি মাপ: মোট (কনটেক্সট → শব্দ) এন্ট্রি।"""
        return (sum(len(v) for v in self.bi.values())
                + sum(len(v) for v in self.tri.values())
                + sum(len(v) for v in self.skip.values()))

    def compact(self, max_vocab: int = 30000, min_ctx: int = 2,
                max_per_ctx: int = 6, keep_skip: bool = False) -> None:
        """কম-মেমরি হোস্টের জন্য মডেল ছোট করা।

        `min_ctx`-এর কম বার দেখা জোড়া বাদ যায়। কনটেক্সটের আসল মোট গণনা (tot)
        অপরিবর্তিত থাকে — তাই বাকি শব্দগুলোর সম্ভাবনা বিকৃত হয় না, আর বারবার
        compact করলেও সংখ্যা ক্ষয়ে যায় না।
        """
        keep_words = {w for w, _ in self.uni.most_common(max_vocab)} | {"</s>", "<s>"}

        def _cut(store, tot, per_ctx):
            out, otot = {}, {}
            for k, d in store.items():
                items = [(w, c) for w, c in d.items() if c >= min_ctx and w in keep_words]
                if not items:
                    continue
                if len(items) > per_ctx:
                    items.sort(key=lambda kv: -kv[1])
                    items = items[:per_ctx]
                out[k] = dict(items)
                otot[k] = tot.get(k, 0) or sum(c for _, c in items)
            return out, otot

        self.uni = Counter({w: c for w, c in self.uni.items() if w in keep_words})
        self.bi, self.bi_tot = _cut(self.bi, self.bi_tot, max_per_ctx)
        self.tri, self.tri_tot = _cut(self.tri, self.tri_tot, max(max_per_ctx - 2, 3))
        if keep_skip:
            self.skip, self.skip_tot = _cut(self.skip, self.skip_tot, 4)
        else:
            self.skip, self.skip_tot = {}, {}

    def fit_budget(self, max_entries: int, max_vocab: int = 30000) -> bool:
        """মডেল `max_entries`-এর বেশি বড় হলে তবেই ছোট করে (কম-মেমরি হোস্ট)।

        আগে প্রতিবার শেখার পরই compact() চলত — তাতে একবার দেখা নতুন সব শব্দজোড়া
        সঙ্গে সঙ্গে মুছে যেত, অর্থাৎ মডেল কিছুই শিখত না। এখন নতুন জোড়া জমা থাকে
        (পরে আবার দেখা গেলে গণনা বাড়ে); শুধু বাজেট ছাড়ালে বিরল জোড়া ছাঁটা হয়।
        """
        if self.size() <= max_entries:
            if len(self.uni) > max_vocab * 2:
                keep = dict(self.uni.most_common(max_vocab))
                keep.setdefault("</s>", self.uni.get("</s>", 0))
                self.uni = Counter(keep)
            return False
        self.skip, self.skip_tot = {}, {}
        for min_ctx, per_ctx in ((2, 8), (2, 6), (3, 6), (4, 5), (6, 4)):
            self.compact(max_vocab=max_vocab, min_ctx=min_ctx, max_per_ctx=per_ctx)
            if self.size() <= max_entries * 0.8:
                break
        return True

    # ------------------------------------------------------------------ #
    @staticmethod
    def context(toks: list[str]) -> tuple[str, str]:
        """শেষ দুই শব্দ; বাক্য শেষ হয়ে থাকলে নতুন বাক্যের শুরু (<s>, <s>)।"""
        if not toks or toks[-1] in SENT_END:
            return "<s>", "<s>"
        b = toks[-1]
        a = toks[-2] if len(toks) >= 2 and toks[-2] not in SENT_END else "<s>"
        return a, b

    def _score(self, a: str, b: str) -> dict[str, float]:
        tri = self.tri.get(a + SEP + b) or {}
        bi = self.bi.get(b) or {}
        sk = self.skip.get(a) or {}
        t_tot = self.tri_tot.get(a + SEP + b, 0)
        b_tot = self.bi_tot.get(b, 0)
        s_tot = self.skip_tot.get(a, 0) if sk else 0

        lam3 = t_tot / (t_tot + 1.4) if t_tot else 0.0
        lam2 = (1 - lam3) * (b_tot / (b_tot + 1.4)) if b_tot else 0.0
        rest = 1 - lam3 - lam2
        lam_s = rest * 0.65 if s_tot else 0.0
        lam1 = rest - lam_s
        N = self.tokens or 1

        cands = set(tri) | set(bi)
        if len(cands) < 8:
            cands |= set(sk)
        if len(cands) < 6:
            cands |= {w for w, _ in self.uni.most_common(400) if w != "</s>"}

        out: dict[str, float] = {}
        for w in cands:
            if w == "</s>":
                continue
            p3 = tri[w] / t_tot if t_tot and w in tri else 0.0
            p2 = bi[w] / b_tot if b_tot and w in bi else 0.0
            ps = sk[w] / s_tot if s_tot and w in sk else 0.0
            p1 = self.uni.get(w, 0) / N
            p = lam3 * p3 + lam2 * p2 + lam_s * ps + lam1 * p1
            if p > 0:
                out[w] = p
        return out

    def _prefix_fallback(self, word: str) -> dict[str, float]:
        """শব্দভান্ডারে নেই এমন শব্দের জন্য: একই শুরু-অংশের শব্দ খোঁজা।"""
        if len(word) < 3:
            return {}
        pre = word[:3]
        N = self.tokens or 1
        hits = [(w, c) for w, c in self.uni.items() if w != "</s>" and w.startswith(pre)]
        hits.sort(key=lambda kv: -kv[1])
        return {w: c / N for w, c in hits[:12]}

    def word_probs(self, text: str) -> dict[str, float]:
        """পরের শব্দের (স্বাভাবিকীকরণ-ছাড়া) সম্ভাবনা — নিউরাল মডেলের সাথে মেশানোর জন্য।

        শেখা চলাকালীন অন্য থ্রেড ডিকশনারি বদলালে একবার-দু'বার আবার চেষ্টা করে।
        """
        for _ in range(4):
            try:
                return self._word_probs(text)
            except RuntimeError:  # dictionary changed size during iteration
                continue
        return {}

    def _word_probs(self, text: str) -> dict[str, float]:
        a, b = self.context(tokenize(text))
        scores = self._score(a, b)
        if not scores and b != "<s>":
            scores = self._prefix_fallback(b)
        if not scores:
            # কনটেক্সট অজানা → সাধারণ জনপ্রিয়তা
            scores = {w: c / (self.tokens or 1)
                      for w, c in self.uni.most_common(60) if w != "</s>"}

        # ন্যূনতম ৪টি শব্দ-পরামর্শ নিশ্চিত করা
        if sum(1 for w in scores if WORD_RE.match(w)) < 4:
            N = self.tokens or 1
            for w, c in self.uni.most_common(1200):
                if w != "</s>" and w not in scores:
                    scores[w] = 0.4 * c / N
                if sum(1 for x in scores if WORD_RE.match(x)) >= 8:
                    break
        return scores

    def predict(self, text: str, k: int = 6) -> list[dict]:
        return rank_words(self.word_probs(text), k)

    # ------------------------------------------------------------------ #
    def generate(self, prompt: str, max_tokens: int = 30,
                 temperature: float = 0.85, top_k: int = 60) -> str:
        for _ in range(4):
            try:
                return self._generate(prompt, max_tokens, temperature, top_k)
            except RuntimeError:
                continue
        return ""

    def _generate(self, prompt: str, max_tokens: int, temperature: float, top_k: int) -> str:
        toks = tokenize(prompt)
        gen: list[str] = []
        used: Counter = Counter()
        for _ in range(max_tokens):
            a, b = self.context(toks + gen)
            scores = self._score(a, b)
            if not scores:
                break
            items = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
            items = [(w, s / (1.0 + 2.2 * used[w])) for w, s in items]
            if temperature > 0:
                items = [(w, s ** (1.0 / max(0.15, temperature))) for w, s in items]
            tot = sum(s for _, s in items)
            if tot <= 0:
                break
            r = random.random() * tot
            acc, pick = 0.0, items[0][0]
            for w, s in items:
                acc += s
                if r <= acc:
                    pick = w
                    break
            gen.append(pick)
            used[pick] += 1
            if pick in SENT_END and len(gen) >= 6:
                break
        return detokenize(gen)

    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        return {
            "tokens": self.tokens,
            "vocab": len(self.uni),
            "unigrams": len(self.uni),
            "bigrams": sum(len(v) for v in self.bi.values()),
            "trigrams": sum(len(v) for v in self.tri.values()),
            "skipgrams": sum(len(v) for v in self.skip.values()),
            "sentences": self.sentences,
        }

    tok_version = 0          # load() সেট করে; নতুন মডেলে বর্তমান সংস্করণ

    def is_current(self) -> bool:
        """ফাইলটা বর্তমান টোকেনাইজার দিয়ে বানানো? (পুরোনো হলে নতুন করে গড়তে হবে —
        নইলে মডেলের শব্দ আর ইনপুটের শব্দ মেলে না)।"""
        return self.tokens == 0 or (self.tok_version or TOK_VERSION) >= TOK_VERSION

    def save(self, path=MODEL_PATH) -> None:
        """দ্রুত বাইনারি সেভ (pickle + gzip level 1) — বড় মডেলেও কয়েক সেকেন্ড।"""
        payload = {
            "version": 3, "tok_version": TOK_VERSION,
            "tokens": self.tokens, "sentences": self.sentences,
            "uni": dict(self.uni),
            "bi": self.bi, "tri": self.tri, "skip": self.skip,
            "bi_tot": self.bi_tot, "tri_tot": self.tri_tot, "skip_tot": self.skip_tot,
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        with gzip.open(tmp, "wb", compresslevel=1) as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)

    @classmethod
    def load(cls, path=MODEL_PATH) -> "NGramModel":
        m = cls()
        if not path.exists():
            return m
        try:
            with gzip.open(path, "rb") as f:
                p = _SafeUnpickler(f).load()
        except (OSError, EOFError, pickle.UnpicklingError, AttributeError, ImportError, IndexError):
            try:  # পুরোনো JSON ফরম্যাট
                with gzip.open(path, "rt", encoding="utf-8") as f:
                    p = json.load(f)
            except (OSError, ValueError):
                return m
        if not isinstance(p, dict):
            return m
        m.tok_version = int(p.get("tok_version") or 1)
        m.tokens = p.get("tokens", 0)
        m.sentences = p.get("sentences", p.get("docs", 0))
        m.uni = p.get("uni") if isinstance(p.get("uni"), Counter) else Counter(p.get("uni") or {})
        m.bi = p.get("bi") or {}
        m.tri = p.get("tri") or {}
        m.skip = p.get("skip") or {}
        m.bi_tot = p.get("bi_tot") or {}
        m.tri_tot = p.get("tri_tot") or {}
        m.skip_tot = p.get("skip_tot") or {}
        return m


def perplexity(model: NGramModel, text: str) -> float:
    """মডেল কতটা 'অবাক' হয় (কম = ভালো)। অজানা শব্দে unigram-floor ব্যবহার।"""
    toks = tokenize(text)
    if len(toks) < 3:
        return float("nan")
    floor = 1.0 / max(1, model.tokens + 1)
    logp = 0.0
    for i in range(2, len(toks)):
        a, b = model.context(toks[:i])
        p = model._score(a, b).get(toks[i]) or floor
        logp += math.log(max(p, floor))
    return math.exp(-logp / (len(toks) - 2))
