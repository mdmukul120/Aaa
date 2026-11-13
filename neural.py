"""নিউরাল ভাষা-মডেল — ছোট GPT-ধাঁচের ট্রান্সফরমার (শুধু NumPy দিয়ে চালানো)।

ট্রেইনিং হয় `train_neural.py` দিয়ে (JAX; GitHub Actions বা নিজের কম্পিউটারে)।
সার্ভারে শুধু এই ফাইল লাগে: numpy + sentencepiece — PyTorch/GPU ছাড়াই,
৫১২ MB র‍্যামের ফ্রি হোস্টেও চলে (ওজন ফাইল কয়েক MB)।

    টেক্সট → ngmodel.tokenize() → " ".join → SentencePiece subword → ট্রান্সফরমার
    → পরের subword-এর সম্ভাবনা → beam search দিয়ে পুরো "পরের শব্দ"

ওজন ফাইল (.npz):  wte, wpe, l{i}.*, lnf.*, meta (JSON), spm (SentencePiece মডেল)
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

# ছোট ম্যাট্রিক্সে BLAS-এর বহু থ্রেড উল্টো ধীর করে (বিশেষত ০.১ CPU-র ফ্রি হোস্টে) — একটাই যথেষ্ট
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

from ngmodel import SENT_END, detokenize, tokenize

WORD_START = "\u2581"   # SentencePiece-এর শব্দ-শুরুর চিহ্ন "▁"


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


def _ln(x: np.ndarray, g: np.ndarray, b: np.ndarray) -> np.ndarray:
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + 1e-5) * g + b


def _log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def normalize_text(text: str) -> str:
    """মডেলের ইনপুট রূপ: n-gram মডেলের একই টোকেন, ফাঁকা দিয়ে জোড়া।"""
    return " ".join(tokenize(text))


class NeuralLM:
    MEMO_SLOTS = 8      # একসাথে কতগুলো লেখার prefix-cache মনে রাখা হবে (~০.৫ MB প্রতিটি)

    def __init__(self, weights: dict[str, np.ndarray], meta: dict, spm_bytes: bytes) -> None:
        import sentencepiece as spm

        self.meta = meta
        self.w = {k: np.asarray(v, dtype=np.float32) for k, v in weights.items()}
        self.n_layer = int(meta["n_layer"])
        self.n_head = int(meta["n_head"])
        self.ctx = int(meta["ctx"])
        self.d = int(self.w["wte"].shape[1])
        self.vocab = int(self.w["wte"].shape[0])
        self.dh = self.d // self.n_head
        self.blend_w = float(meta.get("blend_w", 0.6))
        self.wte_T = np.ascontiguousarray(self.w["wte"].T)

        self.sp = spm.SentencePieceProcessor(model_proto=spm_bytes)
        self.bos, self.eos, self.unk = self.sp.bos_id(), self.sp.eos_id(), self.sp.unk_id()
        self.pieces = [self.sp.id_to_piece(i) for i in range(self.vocab)]
        special = {i for i in (self.bos, self.eos, self.unk, self.sp.pad_id()) if i >= 0}
        special |= {i for i in range(self.vocab) if self.sp.is_control(i)}
        start = np.array([p.startswith(WORD_START) for p in self.pieces])
        ok = np.ones(self.vocab, dtype=bool)
        ok[list(special)] = False
        self.start_mask = start & ok              # শব্দ শুরু করে এমন টুকরো
        self.cont_mask = (~start) & ok            # শব্দের ভেতরের টুকরো
        self.end_mask = self.start_mask.copy()    # এগুলো এলে আগের শব্দ শেষ
        if self.eos >= 0:
            self.end_mask[self.eos] = True
        self._sent_end_ids = {i for i, p in enumerate(self.pieces)
                              if p in {WORD_START + s for s in SENT_END}}

        self._lock = threading.Lock()
        self._memos: list[tuple[tuple[int, ...], list, list, np.ndarray]] = []

    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, path: str | Path) -> "NeuralLM":
        with np.load(str(path), allow_pickle=False) as z:
            meta = json.loads(bytes(z["meta"]).decode("utf-8"))
            spm_bytes = bytes(z["spm"])
            weights = {k: z[k] for k in z.files if k not in ("meta", "spm")}
        return cls(weights, meta, spm_bytes)

    @staticmethod
    def read_meta(path: str | Path) -> dict:
        with np.load(str(path), allow_pickle=False) as z:
            return json.loads(bytes(z["meta"]).decode("utf-8"))

    def stats(self) -> dict:
        m = self.meta
        return {
            "params": int(sum(v.size for v in self.w.values())),
            "layers": self.n_layer, "heads": self.n_head, "width": self.d,
            "context": self.ctx, "vocab": self.vocab,
            "val_ppl": m.get("val_ppl"), "trained_at": m.get("trained_at"),
            "train_tokens": m.get("train_tokens"), "train_docs": m.get("train_docs"),
            "source": m.get("source"), "accuracy": m.get("accuracy"),
            "blend_w": self.blend_w,
        }

    # ------------------------------------------------------------------ #
    # ট্রান্সফরমার (KV-cache সহ, যাতে প্রতি কি-স্ট্রোকে পুরো লেখা আবার না হিসাব হয়)
    # ------------------------------------------------------------------ #
    def _split(self, x: np.ndarray) -> np.ndarray:
        """(m, d) → (heads, m, dh)"""
        return x.reshape(x.shape[0], self.n_head, self.dh).transpose(1, 0, 2)

    def _extend(self, K: list, V: list, ids: list[int]) -> tuple[list, list, np.ndarray]:
        """আগের n টোকেনের cache-এ নতুন m টোকেন যোগ করে; নতুনগুলোর hidden ফেরায় (m, d)।"""
        w, m = self.w, len(ids)
        n = K[0].shape[1] if K else 0
        if n + m > self.ctx:
            raise ValueError("context overflow")
        x = w["wte"][ids] + w["wpe"][n:n + m]
        mask = np.triu(np.full((m, m), -1e9, dtype=np.float32), k=1)
        mask = np.concatenate([np.zeros((m, n), dtype=np.float32), mask], axis=1)
        scale = 1.0 / np.sqrt(self.dh)
        K2, V2 = [], []
        for i in range(self.n_layer):
            p = f"l{i}."
            h = _ln(x, w[p + "ln1.g"], w[p + "ln1.b"])
            qkv = h @ w[p + "attn.w"] + w[p + "attn.b"]
            q, k, v = (self._split(t) for t in np.split(qkv, 3, axis=-1))
            if K:
                k = np.concatenate([K[i], k], axis=1)
                v = np.concatenate([V[i], v], axis=1)
            K2.append(k)
            V2.append(v)
            att = (q @ k.transpose(0, 2, 1)) * scale + mask
            att = np.exp(att - att.max(-1, keepdims=True))
            att /= att.sum(-1, keepdims=True)
            y = (att @ v).transpose(1, 0, 2).reshape(m, self.d)
            x = x + y @ w[p + "proj.w"] + w[p + "proj.b"]
            h = _ln(x, w[p + "ln2.g"], w[p + "ln2.b"])
            x = x + _gelu(h @ w[p + "fc.w"] + w[p + "fc.b"]) @ w[p + "out.w"] + w[p + "out.b"]
        return K2, V2, _ln(x, w["lnf.g"], w["lnf.b"])

    def _step_batch(self, K: list, V: list, Ke: list, Ve: list,
                    toks: np.ndarray, pos: int) -> tuple[list, list, np.ndarray]:
        """B টি beam-এর প্রতিটিতে একটি করে নতুন টোকেন। সবার prefix (K, V) একই;
        প্রতিটি beam-এর নিজস্ব অংশ Ke/Ve: (B, heads, t, dh)।"""
        w, B = self.w, len(toks)
        x = w["wte"][toks] + w["wpe"][pos]
        scale = 1.0 / np.sqrt(self.dh)
        Ke2, Ve2 = [], []
        for i in range(self.n_layer):
            p = f"l{i}."
            h = _ln(x, w[p + "ln1.g"], w[p + "ln1.b"])
            qkv = h @ w[p + "attn.w"] + w[p + "attn.b"]
            q, k, v = (t.reshape(B, self.n_head, self.dh) for t in np.split(qkv, 3, axis=-1))
            ke = k[:, :, None, :] if Ke is None else np.concatenate([Ke[i], k[:, :, None, :]], axis=2)
            ve = v[:, :, None, :] if Ve is None else np.concatenate([Ve[i], v[:, :, None, :]], axis=2)
            Ke2.append(ke)
            Ve2.append(ve)
            sp = np.einsum("bhd,hnd->bhn", q, K[i]) * scale
            se = np.einsum("bhd,bhtd->bht", q, ke) * scale
            s = np.concatenate([sp, se], axis=-1)
            s = np.exp(s - s.max(-1, keepdims=True))
            s /= s.sum(-1, keepdims=True)
            n = K[i].shape[1]
            y = np.einsum("bhn,hnd->bhd", s[..., :n], V[i]) + np.einsum("bht,bhtd->bhd", s[..., n:], ve)
            x = x + y.reshape(B, self.d) @ w[p + "proj.w"] + w[p + "proj.b"]
            h = _ln(x, w[p + "ln2.g"], w[p + "ln2.b"])
            x = x + _gelu(h @ w[p + "fc.w"] + w[p + "fc.b"]) @ w[p + "out.w"] + w[p + "out.b"]
        return Ke2, Ve2, _ln(x, w["lnf.g"], w["lnf.b"])

    def _logp(self, hidden: np.ndarray) -> np.ndarray:
        return _log_softmax(hidden @ self.wte_T)

    # ------------------------------------------------------------------ #
    def encode(self, text: str) -> list[int]:
        return [self.bos] + self.sp.encode(normalize_text(text))

    def _window(self, ids: list[int], reserve: int) -> list[int]:
        """কনটেক্সট জানালায় আঁটানো। শুরুর বিন্দু ধাপে ধাপে সরে — যাতে টাইপ করার
        সময় cache বারবার নষ্ট না হয় — এবং সবসময় শব্দের শুরুতে পড়ে।"""
        room = self.ctx - reserve
        if len(ids) <= room:
            return ids
        hop = max(8, room // 3)
        start = ((len(ids) - room + hop - 1) // hop) * hop
        while start < len(ids) - 1 and not self.pieces[ids[start]].startswith(WORD_START):
            start += 1
        return ids[start:]

    def _prefix(self, ids: list[int]) -> tuple[list, list, np.ndarray]:
        """prefix-এর cache + শেষ টোকেনের hidden। আগের কল-এর cache থাকলে শুধু নতুন অংশ চলে।

        সর্বশেষ কয়েকটা লেখার cache রাখা হয় (LRU) — একসাথে কয়েকজন টাইপ করলেও
        একে অন্যের cache নষ্ট করে না।
        """
        key = tuple(ids)
        with self._lock:
            best = None
            for memo in self._memos:
                mk = memo[0]
                if len(mk) <= len(key) and key[:len(mk)] == mk and (best is None or len(mk) > len(best[0])):
                    best = memo
        K: list = []
        V: list = []
        done = 0
        if best is not None:
            if len(best[0]) == len(key):
                return best[1], best[2], best[3]
            K, V, done = best[1], best[2], len(best[0])
        K, V, h = self._extend(K, V, list(key[done:]))
        last = h[-1]
        with self._lock:
            # যে cache থেকে বাড়ানো হলো সেটা বদলে নতুনটা সামনে
            self._memos = [(key, K, V, last)] + [m for m in self._memos if m is not best][:self.MEMO_SLOTS - 1]
        return K, V, last

    # ------------------------------------------------------------------ #
    def next_words(self, text: str, beam: int = 8, max_pieces: int = 6,
                   first: int = 24) -> dict[str, float]:
        """পরের সম্ভাব্য শব্দগুলো ও তাদের সম্ভাবনা।

        P(শব্দ) = Π P(টুকরো) × P(এরপর নতুন শব্দ শুরু)।  একাধিক টুকরোর শব্দ
        beam search দিয়ে গড়া হয়, তাই শব্দভান্ডারে নেই এমন শব্দও বানাতে পারে।
        """
        ids = self._window(self.encode(text), reserve=max_pieces + 1)
        K, V, last = self._prefix(ids)
        n = len(ids)
        lp = self._logp(last)
        cand = np.where(self.start_mask)[0]
        top = cand[np.argsort(-lp[cand])[:first]]
        beams = [(float(lp[i]), [int(i)]) for i in top]
        out: dict[str, float] = {}
        Ke = Ve = None
        for step in range(max_pieces):
            if not beams:
                break
            toks = np.array([b[1][-1] for b in beams])
            Ke, Ve, h = self._step_batch(K, V, Ke, Ve, toks, n + step)
            lps = self._logp(h)
            ends = lps[:, self.end_mask]
            mx = ends.max(-1, keepdims=True)
            p_end = (mx[:, 0] + np.log(np.exp(ends - mx).sum(-1)))
            nxt: list[tuple[float, int, int]] = []
            for bi, (score, pcs) in enumerate(beams):
                word = "".join(self.pieces[i] for i in pcs).replace(WORD_START, "")
                if word:
                    out[word] = out.get(word, 0.0) + float(np.exp(score + p_end[bi]))
                if step + 1 < max_pieces:
                    row = np.where(self.cont_mask, lps[bi], -np.inf)
                    for j in np.argpartition(-row, 3)[:3]:
                        nxt.append((score + float(row[j]), bi, int(j)))
            # যেসব অসমাপ্ত শব্দের সম্ভাবনা ইতিমধ্যে নগণ্য, সেগুলো বাদ
            floor = sorted(out.values(), reverse=True)[:8]
            cut = np.log(max(floor[-1] * 0.05, 1e-7)) if floor else -16.0
            nxt = sorted((c for c in nxt if c[0] > cut), reverse=True)[:beam]
            if not nxt:
                break
            parents = [c[1] for c in nxt]
            Ke = [k[parents] for k in Ke]
            Ve = [v[parents] for v in Ve]
            beams = [(c[0], beams[c[1]][1] + [c[2]]) for c in nxt]
        return out

    # ------------------------------------------------------------------ #
    def generate(self, prompt: str, max_words: int = 30, temperature: float = 0.85,
                 top_k: int = 40, rng: np.random.Generator | None = None) -> str:
        """লেখা চালিয়ে যাওয়া: subword ধরে ধরে নমুনা নেওয়া (temperature + top-k)।"""
        rng = rng or np.random.default_rng()
        ids = self._window(self.encode(prompt), reserve=min(self.ctx // 2, max_words * 3))
        K, V, last = self._prefix(ids)
        K, V = list(K), list(V)
        gen: list[int] = []
        words = 0
        recent: dict[int, int] = {}
        temperature = max(0.15, float(temperature))
        while len(ids) + len(gen) < self.ctx:
            lp = self._logp(last).astype(np.float64)
            lp[self.unk] = -np.inf
            if self.bos >= 0:
                lp[self.bos] = -np.inf
            if not gen:
                lp[~self.start_mask] = -np.inf      # নতুন শব্দ দিয়ে শুরু
            for t, c in recent.items():              # পুনরাবৃত্তি কমানো
                lp[t] -= 1.2 * c
            idx = np.argpartition(-lp, top_k)[:top_k]
            z = lp[idx] / temperature
            z = np.exp(z - z.max())
            pick = int(idx[rng.choice(len(idx), p=z / z.sum())])
            if pick == self.eos:
                break
            is_start = bool(self.start_mask[pick])
            if is_start and words >= max_words:
                break
            gen.append(pick)
            words += 1 if is_start else 0
            recent[pick] = recent.get(pick, 0) + 1
            if pick in self._sent_end_ids and words >= 6:
                break
            K, V, h = self._extend(K, V, [pick])
            last = h[-1]
        return detokenize(self.sp.decode(gen).split())

    # ------------------------------------------------------------------ #
    def loss(self, ids: np.ndarray) -> float:
        """একটি টোকেন-ধারার গড় cross-entropy (যাচাই/পরীক্ষার জন্য)।"""
        ids = [int(i) for i in ids[: self.ctx + 1]]
        _, _, h = self._extend([], [], ids[:-1])
        lp = self._logp(h)
        return float(-np.mean(lp[np.arange(len(ids) - 1), ids[1:]]))


def blend(neural: dict[str, float], ngram: dict[str, float], w: float) -> dict[str, float]:
    """দুই মডেলের সম্ভাবনা মেশানো: p = w·নিউরাল + (1−w)·n-gram।"""
    out = {k: w * v for k, v in neural.items()}
    for k, v in ngram.items():
        out[k] = out.get(k, 0.0) + (1.0 - w) * v
    return out
