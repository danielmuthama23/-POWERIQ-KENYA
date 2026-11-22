"""Local hybrid retrieval (sparse vectors + BM25-style keyword score) with metadata filters and citations.

This is the LOCAL FALLBACK retriever. It does not use neural embeddings (word and character TF-IDF vectors are used)
and is not Foundry IQ or Azure AI Search. Those integrations are reported as 'not configured' until a working
connection is tested.
"""
from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"
_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass
class Chunk:
    doc_id: str
    title: str
    category: str
    source: str
    chunk_id: int
    text: str


def parse_document(raw: str, fallback_title: str) -> tuple[dict, str]:
    meta = {"title": fallback_title, "category": "general", "source": "user-supplied document"}
    body = raw
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            for line in parts[1].strip().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
            body = parts[2]
    return meta, body.strip()


def chunk_text(text: str, max_words: int = 110, overlap: int = 20) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur = [], []
    count = 0
    for p in paras:
        w = p.split()
        if count + len(w) > max_words and cur:
            chunks.append("\n".join(cur))
            tail = " ".join(" ".join(cur).split()[-overlap:])
            cur, count = ([tail] if overlap else []), (overlap if overlap else 0)
        cur.append(p)
        count += len(w)
    if cur:
        chunks.append("\n".join(cur))
    return chunks


class KnowledgeBase:
    def __init__(self, directory: Path = KNOWLEDGE_DIR):
        self.dir = directory
        self.chunks: list[Chunk] = []
        self._lock = threading.Lock()
        self._dirty = True
        self.word_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self._load_dir()

    def _load_dir(self) -> None:
        if self.dir.exists():
            for p in sorted(self.dir.glob("*.md")) + sorted(self.dir.glob("*.txt")):
                self.add_document(p.stem, p.read_text(encoding="utf-8"), rebuild=False)
        self._rebuild()

    def add_document(self, doc_id: str, raw: str, rebuild: bool = True) -> int:
        meta, body = parse_document(raw, doc_id)
        parts = chunk_text(body)
        with self._lock:
            self.chunks = [c for c in self.chunks if c.doc_id != doc_id]
            for i, t in enumerate(parts):
                self.chunks.append(Chunk(doc_id, meta["title"], meta.get("category", "general"),
                                         meta.get("source", "unknown"), i, t))
            self._dirty = True
        if rebuild:
            self._rebuild()
        return len(parts)

    def _rebuild(self) -> None:
        with self._lock:
            if not self.chunks:
                self.W = self.C = None
                self._dirty = False
                return
            texts = [c.title + ". " + c.text for c in self.chunks]
            self.W = self.word_vec.fit_transform(texts)
            self.C = self.char_vec.fit_transform(texts)
            self.tokens = [_TOKEN.findall(t.lower()) for t in texts]
            self.avgdl = float(np.mean([len(t) for t in self.tokens])) or 1.0
            df: dict[str, int] = {}
            for toks in self.tokens:
                for tok in set(toks):
                    df[tok] = df.get(tok, 0) + 1
            self.df = df
            self._dirty = False

    def _bm25(self, q_tokens: list[str]) -> np.ndarray:
        n = len(self.tokens)
        s = np.zeros(n)
        k1, b = 1.5, 0.75
        for i, toks in enumerate(self.tokens):
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            dl = len(toks)
            for q in q_tokens:
                if q in tf:
                    idf = math.log(1 + (n - self.df.get(q, 0) + 0.5) / (self.df.get(q, 0) + 0.5))
                    s[i] += idf * tf[q] * (k1 + 1) / (tf[q] + k1 * (1 - b + b * dl / self.avgdl))
        return s

    def documents(self) -> list[dict]:
        seen: dict[str, dict] = {}
        for c in self.chunks:
            d = seen.setdefault(c.doc_id, {"doc_id": c.doc_id, "title": c.title, "category": c.category, "source": c.source, "chunks": 0})
            d["chunks"] += 1
        return list(seen.values())

    def query(self, question: str, k: int = 4, category: str | None = None, min_score: float = 0.18) -> dict:
        if self._dirty:
            self._rebuild()
        if not self.chunks or self.W is None:
            return {"answerable": False, "results": [], "reason": "knowledge base is empty"}
        q_tokens = _TOKEN.findall(question.lower())
        qw, qc = self.word_vec.transform([question]), self.char_vec.transform([question])
        sw = (self.W @ qw.T).toarray().ravel()
        sc = (self.C @ qc.T).toarray().ravel()
        bm = self._bm25(q_tokens)
        bm = bm / (bm.max() + 1e-9)
        score = 0.45 * sw + 0.20 * sc + 0.35 * bm
        mask = np.ones(len(self.chunks), bool)
        if category:
            mask = np.array([c.category == category for c in self.chunks])
        score = np.where(mask, score, -1)
        order = np.argsort(-score)[:k]
        res = []
        content = [t for t in q_tokens if t not in ENGLISH_STOP_WORDS and len(t) > 2]
        for i in order:
            if score[i] < min_score:
                continue
            # lexical coverage gate: enough of the question's content words must appear in the passage
            toks = set(self.tokens[i])
            stems = {t[:5] for t in toks}
            cov = (sum(1 for t in content if t in toks or t[:5] in stems) / len(content)) if content else 0.0
            if cov < 0.5:
                continue
            c = self.chunks[i]
            res.append({"citation": f"[{c.doc_id}#{c.chunk_id}]", "doc_id": c.doc_id, "title": c.title, "category": c.category,
                        "source": c.source, "chunk_id": c.chunk_id, "score": round(float(score[i]), 3), "coverage": round(cov, 2), "text": c.text})
        return {"answerable": bool(res), "results": res,
                "retriever": "local hybrid (word+char TF-IDF cosine, BM25) - not neural embeddings, not Foundry IQ",
                "reason": None if res else "no passage exceeded the relevance threshold; the knowledge base has no supporting evidence"}


_kb: KnowledgeBase | None = None


def get_kb() -> KnowledgeBase:
    global _kb
    if _kb is None:
        _kb = KnowledgeBase()
    return _kb
