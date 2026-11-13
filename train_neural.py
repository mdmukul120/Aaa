"""নিউরাল ভাষা-মডেল ট্রেইনিং (JAX, শুধু CPU-তেই চলে)।

    pip install -r requirements-train.txt

    # Supabase-এর সব ডকুমেন্ট নামিয়ে, রিপোর মডেল থেকে শুরু করে ফাইন-টিউন, তারপর আপলোড
    python train_neural.py --from-supabase --init models/shobdo-neural.npz --upload

    # লোকাল করপাস (data/corpus.jsonl) থেকে একদম নতুন মডেল
    python train_neural.py --minutes 60 --out models/shobdo-neural.npz

ফলাফল একটাই .npz ফাইল (ওজন + SentencePiece টোকেনাইজার + মেটা) — সার্ভারে
`neural.py` শুধু NumPy দিয়ে এটা চালায়।

মডেল: decoder-only ট্রান্সফরমার (pre-LayerNorm, GELU, tied embedding), AdamW,
warmup + cosine learning rate, dropout, gradient clipping; সেরা validation loss-এর
ওজনই সেভ হয়। শেষে অদেখা ডকুমেন্টে n-gram, নিউরাল ও মিশ্র — তিনটারই
পরের-শব্দ নির্ভুলতা মেপে মেটায় লেখা হয়।
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from config import CORPUS_PATH, NEURAL_BUNDLED, NEURAL_META_NAME, NEURAL_NAME
from ngmodel import SENT_SPLIT, NGramModel, rank_words, tokenize

BOS, EOS, UNK = 1, 2, 0


# --------------------------------------------------------------------------- #
# ডেটা
# --------------------------------------------------------------------------- #
def load_docs(path: Path) -> list[str]:
    by_url: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                if len(d["content"]) >= 200:
                    by_url[d["url"]] = d["content"]
            except (ValueError, KeyError):
                continue
    return list(by_url.values())


def norm_doc(text: str) -> str:
    return " ".join(tokenize(text))


def train_tokenizer(docs: list[str], vocab: int, log=print) -> bytes:
    import sentencepiece as spm

    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "sents.txt"
        with src.open("w", encoding="utf-8") as f:
            for d in docs:
                for sent in SENT_SPLIT.split(d):
                    s = norm_doc(sent)
                    if s:
                        f.write(s[:4000] + "\n")
        buf = io.BytesIO()
        spm.SentencePieceTrainer.train(
            input=str(src), model_writer=buf, vocab_size=vocab, model_type="unigram",
            character_coverage=0.9995, normalization_rule_name="identity",
            unk_id=UNK, bos_id=BOS, eos_id=EOS, pad_id=-1,
            hard_vocab_limit=False, input_sentence_size=1_500_000,
            shuffle_input_sentence=True, num_threads=os.cpu_count() or 2,
            max_sentence_length=20000,
            minloglevel=2)
        log(f"টোকেনাইজার তৈরি: SentencePiece unigram, লক্ষ্য শব্দভান্ডার {vocab}")
        return buf.getvalue()


def encode_docs(sp, docs: list[str]) -> np.ndarray:
    out: list[np.ndarray] = []
    for i in range(0, len(docs), 256):
        for ids in sp.encode([norm_doc(d) for d in docs[i:i + 256]]):
            out.append(np.asarray([BOS] + ids + [EOS], dtype=np.int32))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int32)


# --------------------------------------------------------------------------- #
# মডেল (JAX)
# --------------------------------------------------------------------------- #
def init_params(rng: np.random.Generator, vocab: int, d: int, n_layer: int, ctx: int) -> dict:
    def nrm(*shape, std=0.02):
        return (rng.standard_normal(shape) * std).astype(np.float32)

    p = {"wte": nrm(vocab, d), "wpe": nrm(ctx, d, std=0.01),
         "lnf.g": np.ones(d, np.float32), "lnf.b": np.zeros(d, np.float32)}
    res = 0.02 / math.sqrt(2 * n_layer)
    for i in range(n_layer):
        k = f"l{i}."
        p[k + "ln1.g"], p[k + "ln1.b"] = np.ones(d, np.float32), np.zeros(d, np.float32)
        p[k + "ln2.g"], p[k + "ln2.b"] = np.ones(d, np.float32), np.zeros(d, np.float32)
        p[k + "attn.w"], p[k + "attn.b"] = nrm(d, 3 * d), np.zeros(3 * d, np.float32)
        p[k + "proj.w"], p[k + "proj.b"] = nrm(d, d, std=res), np.zeros(d, np.float32)
        p[k + "fc.w"], p[k + "fc.b"] = nrm(d, 4 * d), np.zeros(4 * d, np.float32)
        p[k + "out.w"], p[k + "out.b"] = nrm(4 * d, d, std=res), np.zeros(d, np.float32)
    return p


def build(n_layer: int, n_head: int, dropout: float, wd: float):
    import jax
    import jax.numpy as jnp

    def ln(x, g, b):
        mu = x.mean(-1, keepdims=True)
        var = ((x - mu) ** 2).mean(-1, keepdims=True)
        return (x - mu) / jnp.sqrt(var + 1e-5) * g + b

    def gelu(x):
        return 0.5 * x * (1.0 + jnp.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))

    def drop(x, key, train):
        if not train or dropout <= 0:
            return x
        keep = jax.random.bernoulli(key, 1.0 - dropout, x.shape)
        return jnp.where(keep, x / (1.0 - dropout), 0.0)

    def forward(p, ids, key, train):
        B, T = ids.shape
        d = p["wte"].shape[1]
        dh = d // n_head
        keys = jax.random.split(key, 2 * n_layer + 1)
        x = drop(p["wte"][ids] + p["wpe"][:T], keys[0], train)
        mask = jnp.triu(jnp.full((T, T), -1e9, dtype=x.dtype), k=1)
        for i in range(n_layer):
            k = f"l{i}."
            h = ln(x, p[k + "ln1.g"], p[k + "ln1.b"])
            qkv = h @ p[k + "attn.w"] + p[k + "attn.b"]
            q, kk, v = (t.reshape(B, T, n_head, dh).transpose(0, 2, 1, 3)
                        for t in jnp.split(qkv, 3, axis=-1))
            att = jax.nn.softmax(q @ kk.transpose(0, 1, 3, 2) / math.sqrt(dh) + mask, axis=-1)
            y = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, d)
            x = x + drop(y @ p[k + "proj.w"] + p[k + "proj.b"], keys[2 * i + 1], train)
            h = ln(x, p[k + "ln2.g"], p[k + "ln2.b"])
            y = gelu(h @ p[k + "fc.w"] + p[k + "fc.b"]) @ p[k + "out.w"] + p[k + "out.b"]
            x = x + drop(y, keys[2 * i + 2], train)
        x = ln(x, p["lnf.g"], p["lnf.b"])
        return x @ p["wte"].T

    def loss_fn(p, batch, key, train):
        logits = forward(p, batch[:, :-1], key, train)
        logp = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.take_along_axis(logp, batch[:, 1:, None], axis=-1).mean()

    @jax.jit
    def train_step(p, m, v, batch, key, lr, t):
        loss, g = jax.value_and_grad(loss_fn)(p, batch, key, True)
        gnorm = jnp.sqrt(sum(jnp.sum(x * x) for x in g.values()))
        clip = jnp.minimum(1.0, 1.0 / (gnorm + 1e-6))
        b1, b2 = 0.9, 0.98
        np_, nm, nv = {}, {}, {}
        for k in p:
            gk = g[k] * clip
            nm[k] = b1 * m[k] + (1 - b1) * gk
            nv[k] = b2 * v[k] + (1 - b2) * gk * gk
            upd = (nm[k] / (1 - b1 ** t)) / (jnp.sqrt(nv[k] / (1 - b2 ** t)) + 1e-8)
            decay = wd if p[k].ndim >= 2 else 0.0      # bias/LayerNorm-এ weight decay নয়
            np_[k] = p[k] - lr * (upd + decay * p[k])
        return np_, nm, nv, loss

    @jax.jit
    def eval_step(p, batch):
        return loss_fn(p, batch, jax.random.PRNGKey(0), False)

    return train_step, eval_step


# --------------------------------------------------------------------------- #
# সেভ / মূল্যায়ন
# --------------------------------------------------------------------------- #
def save_npz(path: Path, params: dict, meta: dict, spm_bytes: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray(v).astype(np.float16) for k, v in params.items()}
    arrays["meta"] = np.frombuffer(json.dumps(meta, ensure_ascii=False).encode("utf-8"), dtype=np.uint8)
    arrays["spm"] = np.frombuffer(spm_bytes, dtype=np.uint8)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def word_accuracy(lm, ng: NGramModel, docs: list[str], max_ctx: int = 4000,
                  every: int = 3, log=print) -> dict:
    """অদেখা ডকুমেন্টে পরের-শব্দ নির্ভুলতা: n-gram বনাম নিউরাল বনাম মিশ্র।

    প্রথম অর্ধেক নমুনায় মিশ্রণের ওজন (w) বাছা হয়, দ্বিতীয় অর্ধেকে ফল মাপা হয় —
    যাতে সংখ্যাটা সৎ থাকে।
    """
    from neural import blend

    samples: list[tuple[dict, dict, str]] = []
    per_doc = max(8, max_ctx // max(1, len(docs)) + 1)    # সব ডকুমেন্ট থেকে সমানভাবে
    for d in docs:
        toks = tokenize(d)[:240]
        for i in list(range(4, len(toks), every))[:per_doc]:
            ctx = " ".join(toks[:i])       # শুরু থেকে — prefix cache কাজে লাগে
            samples.append((lm.next_words(ctx), ng.word_probs(ctx), toks[i]))
        if len(samples) >= max_ctx:
            break
    if len(samples) < 40:
        return {}
    random.Random(0).shuffle(samples)
    half = len(samples) // 2
    tune, test = samples[:half], samples[half:]

    def acc(rows, w):
        h1 = h5 = 0
        for nn, gg, gold in rows:
            sc = gg if w == 0 else nn if w == 1 else blend(nn, gg, w)
            top = [x["word"] for x in rank_words(sc, 5)]
            h1 += top[:1] == [gold]
            h5 += gold in top
        return 100 * h1 / len(rows), 100 * h5 / len(rows)

    grid = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    best_w = max(grid, key=lambda w: (lambda a: a[0] + 0.5 * a[1])(acc(tune, w)))
    res = {"contexts": len(test)}
    for name, w in (("ngram", 0.0), ("neural", 1.0), ("blend", best_w)):
        t1, t5 = acc(test, w)
        res[name] = {"top1": round(t1, 1), "top5": round(t5, 1)}
        log(f"  {name:7s} top-1 {t1:5.1f}%  top-5 {t5:5.1f}%")
    res["blend_w"] = best_w
    return res


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="শব্দ AI — নিউরাল মডেল ট্রেইনিং")
    ap.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    ap.add_argument("--from-supabase", action="store_true", help="আগে Supabase থেকে সব ডকুমেন্ট নামাও")
    ap.add_argument("--init", type=Path, default=None, help="এই মডেল (ও তার টোকেনাইজার) থেকে শুরু — ফাইন-টিউন")
    ap.add_argument("--out", type=Path, default=NEURAL_BUNDLED)
    ap.add_argument("--upload", action="store_true", help="শেষে Supabase Storage-এ আপলোড")
    ap.add_argument("--vocab", type=int, default=8000)
    ap.add_argument("--width", type=int, default=192)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=96)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=float, default=6.0)
    ap.add_argument("--minutes", type=float, default=120.0, help="সর্বোচ্চ সময়; আগে শেষ হলে আগেই থামে")
    ap.add_argument("--lr", type=float, default=0.0, help="০ = নিজে ঠিক করবে (নতুন: 1.5e-3, ফাইন-টিউন: 4e-4)")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source", type=str, default="", help="মেটায় লেখার জন্য ডেটার বর্ণনা")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--eval-only", action="store_true",
                    help="ট্রেইন না করে --out মডেলের পরের-শব্দ নির্ভুলতা মেপে মেটায় লেখো")
    args = ap.parse_args()

    def log(msg: str) -> None:
        print(f"[neural {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    import db

    if args.from_supabase:
        if not db.configured():
            log("SUPABASE_URL / SUPABASE_SERVICE_KEY সেট নেই — থামছি।")
            return 2
        log(f"Supabase থেকে {db.download_corpus()} নতুন ডকুমেন্ট নামানো হয়েছে")
    if not args.corpus.exists():
        log(f"করপাস নেই: {args.corpus}")
        return 2

    import sentencepiece as spm

    from neural import NeuralLM

    docs = load_docs(args.corpus)
    random.Random(args.seed).shuffle(docs)
    n_val = min(400, max(8, int(len(docs) * args.val_frac)))
    if len(docs) < 30:
        log(f"মাত্র {len(docs)} ডকুমেন্ট — ট্রেইনিংয়ের জন্য খুব কম।")
        return 2
    val_docs, train_docs = docs[:n_val], docs[n_val:]
    log(f"{len(train_docs)} ট্রেইন + {len(val_docs)} যাচাই ডকুমেন্ট")

    val_batches: list = []
    if not args.eval_only:
        # ---- টোকেনাইজার + শুরুর ওজন ----
        init_meta: dict = {}
        if args.init and args.init.exists():
            base = NeuralLM.load(args.init)
            spm_bytes = base.sp.serialized_model_proto()
            params = {k: np.array(v) for k, v in base.w.items()}
            init_meta = base.meta
            args.width, args.layers, args.heads, args.ctx = base.d, base.n_layer, base.n_head, base.ctx
            log(f"শুরুর মডেল: {args.init} (ফাইন-টিউন; টোকেনাইজার অপরিবর্তিত)")
            del base
        else:
            spm_bytes = train_tokenizer(train_docs, args.vocab, log)
            params = None
        sp = spm.SentencePieceProcessor(model_proto=spm_bytes)
        vocab = sp.get_piece_size()

        t0 = time.time()
        train_ids, val_ids = encode_docs(sp, train_docs), encode_docs(sp, val_docs)
        unk = float((train_ids == UNK).mean()) * 100
        log(f"টোকেন: ট্রেইন {len(train_ids):,} · যাচাই {len(val_ids):,} · শব্দভান্ডার {vocab} "
            f"· অজানা {unk:.2f}% ({time.time() - t0:.0f}s)")

        rng = np.random.default_rng(args.seed)
        if params is None:
            params = init_params(rng, vocab, args.width, args.layers, args.ctx)
        n_params = int(sum(v.size for v in params.values()))
        T, B = args.ctx, args.batch
        if len(train_ids) < T * B * 4 or len(val_ids) < T + 2:
            log("ডেটা খুব কম — ট্রেইনিং বাদ।")
            return 2

        import jax

        train_step, eval_step = build(args.layers, args.heads, args.dropout, args.wd)
        lr_peak = args.lr or (4e-4 if init_meta else 1.5e-3)
        total = max(50, int(args.epochs * len(train_ids) / (B * T)))
        warm = min(300, total // 10)
        budget = args.minutes * 60
        log(f"মডেল: {n_params / 1e6:.2f}M প্যারামিটার · {args.layers} স্তর × {args.width} প্রস্থ × "
            f"{args.heads} হেড · কনটেক্সট {T} · সর্বোচ্চ {total} ধাপ / {args.minutes:.0f} মিনিট · lr {lr_peak}")

        # স্থির যাচাই-সেট
        n_vb = min(40, max(1, (len(val_ids) - 1) // T // B)) or 1
        vr = np.random.default_rng(123)
        val_batches = [np.stack([val_ids[s:s + T + 1] for s in
                                 vr.integers(0, len(val_ids) - T - 1, B)]) for _ in range(n_vb)]

        def validate(p) -> float:
            return float(np.mean([float(eval_step(p, b)) for b in val_batches]))

        def meta_for(val_loss: float, step: int) -> dict:
            return {
                "format": 1, "n_layer": args.layers, "n_head": args.heads, "ctx": T,
                "width": args.width, "vocab": vocab, "params": n_params,
                "val_loss": round(val_loss, 4), "val_ppl": round(math.exp(val_loss), 1),
                "steps": step, "train_tokens": int(len(train_ids)), "train_docs": len(train_docs),
                "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source": args.source or ("Supabase documents" if args.from_supabase else str(args.corpus)),
                "base": {k: init_meta.get(k) for k in ("trained_at", "source", "val_ppl")} if init_meta else None,
                "blend_w": init_meta.get("blend_w", 0.6),
            }

        m = {k: np.zeros_like(v) for k, v in params.items()}
        v = {k: np.zeros_like(v) for k, v in params.items()}
        key = jax.random.PRNGKey(args.seed)
        best = validate(params) if init_meta else float("inf")
        if init_meta:
            log(f"শুরুতে যাচাই loss {best:.4f} (ppl {math.exp(best):.1f})")
            save_npz(args.out, params, meta_for(best, 0), spm_bytes)
        best_step, step, run_loss, t0 = 0, 0, None, time.time()
        eval_every = max(100, min(500, total // 20))
        last_eval = t0
        while True:
            prog = max(step / total, (time.time() - t0) / budget)
            if prog >= 1.0:
                break
            lr = lr_peak * (step + 1) / warm if step < warm else \
                lr_peak * (0.08 + 0.92 * 0.5 * (1 + math.cos(math.pi * prog)))
            idx = rng.integers(0, len(train_ids) - T - 1, B)
            batch = np.stack([train_ids[s:s + T + 1] for s in idx])
            key, sub = jax.random.split(key)
            step += 1
            params, m, v, loss = train_step(params, m, v, batch, sub, lr, float(step))
            if step % 20 == 0:
                lv = float(loss)
                run_loss = lv if run_loss is None else 0.9 * run_loss + 0.1 * lv
                if not math.isfinite(lv):
                    log("loss অসীম হয়ে গেছে — থামছি (সেরা সংরক্ষিত ওজন থাকছে)")
                    break
            if step % eval_every == 0 or time.time() - last_eval > 600:
                last_eval = time.time()
                vl = validate(params)
                el = time.time() - t0
                mark = ""
                if vl < best:
                    best, best_step, mark = vl, step, "  ← সেরা, সেভ হলো"
                    save_npz(args.out, params, meta_for(vl, step), spm_bytes)
                log(f"ধাপ {step}/{total} · train {run_loss or 0:.3f} · val {vl:.4f} (ppl {math.exp(vl):.1f}) "
                    f"· {step * B * T / el:,.0f} টোকেন/সে · {el / 60:.1f} মিনিট{mark}")
                if step - best_step > 6 * eval_every and step > total // 4:
                    log("যাচাই loss আর কমছে না — আগেই থামছি (overfitting এড়াতে)")
                    break
        vl = validate(params)
        if vl < best:
            best, best_step = vl, step
            save_npz(args.out, params, meta_for(vl, step), spm_bytes)
        if not args.out.exists():
            log("কোনো মডেল সেভ হয়নি।")
            return 1
        log(f"ট্রেইনিং শেষ: সেরা যাচাই loss {best:.4f} (ppl {math.exp(best):.1f}) ধাপ {best_step}-এ")
        del params, m, v

    # ---- সার্ভারের NumPy কোড দিয়েই যাচাই + পরের-শব্দ নির্ভুলতা ----
    if not args.out.exists():
        log(f"মডেল নেই: {args.out}")
        return 2
    lm = NeuralLM.load(args.out)
    spm_bytes = lm.sp.serialized_model_proto()
    if val_batches:
        chk = float(np.mean([lm.loss(b[0]) for b in val_batches[:8]]))
        log(f"NumPy ইনফারেন্স যাচাই: loss {chk:.4f} (float16 ওজন)")
    meta = dict(lm.meta)
    if not args.no_eval:
        log("পরের-শব্দ নির্ভুলতা (অদেখা ডকুমেন্ট):")
        ng = NGramModel()
        ng.use_skip = False
        for i, d in enumerate(train_docs):
            ng.update(d)
            if i % 2000 == 1999:
                ng.fit_budget(12_000_000, max_vocab=400_000)
        accuracy = word_accuracy(lm, ng, val_docs, log=log)
        if accuracy:
            meta["accuracy"] = accuracy
            meta["blend_w"] = accuracy["blend_w"]
        del ng
    save_npz(args.out, lm.w, meta, spm_bytes)
    size = args.out.stat().st_size / 1e6
    log(f"সেভ: {args.out} ({size:.1f} MB)")
    for prompt in ("বাংলাদেশের রাজধানী", "the history of the"):
        log(f"নমুনা: {prompt} → {lm.generate(prompt, max_words=14, temperature=0.7)}")

    if args.upload:
        meta_file = args.out.with_suffix(".json")
        meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        ok = db.upload_model(args.out, NEURAL_NAME) and db.upload_model(meta_file, NEURAL_META_NAME)
        log("Supabase Storage-এ আপলোড: " + ("সফল ✅" if ok else "ব্যর্থ ❌"))
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
