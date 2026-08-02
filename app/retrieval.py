import hashlib
import math
import re
from collections import Counter
from typing import Iterable, List, Sequence

from app.models import Chunk, SearchHit

TOKEN = re.compile(r"[a-z0-9]+")


def terms(text: str):
    return TOKEN.findall(text.lower())


class HashEmbedder:
    """Deterministic offline embedding for local tests and graceful degradation."""

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.dimensions
            for term in terms(text):
                digest = hashlib.sha256(term.encode()).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimensions
                vector[index] += 1 if digest[4] % 2 else -1
            norm = math.sqrt(sum(x * x for x in vector)) or 1
            vectors.append([x / norm for x in vector])
        return vectors


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def bm25_scores(query: str, documents: Sequence[str]) -> List[float]:
    query_terms = terms(query)
    tokenized = [terms(doc) for doc in documents]
    average = sum(map(len, tokenized)) / max(1, len(tokenized))
    scores = [0.0] * len(documents)
    for term in set(query_terms):
        frequency = sum(term in doc for doc in tokenized)
        idf = math.log(1 + (len(documents) - frequency + 0.5) / (frequency + 0.5))
        for index, doc in enumerate(tokenized):
            tf = doc.count(term)
            denominator = tf + 1.5 * (1 - 0.75 + 0.75 * len(doc) / max(1, average))
            scores[index] += idf * (tf * 2.5 / denominator if denominator else 0)
    return scores


def hybrid_search(query: str, chunks: Sequence[Chunk], embedder, top_k: int) -> List[SearchHit]:
    if not chunks:
        return []
    query_vector = embedder.embed([query])[0]
    lexical = bm25_scores(query, [chunk.text for chunk in chunks])
    semantic = [cosine(query_vector, chunk.embedding) for chunk in chunks]
    lexical_max = max(lexical) or 1
    query_terms = set(terms(query))
    ranked = []
    for chunk, bm25, dense in zip(chunks, lexical, semantic):
        overlap = len(query_terms & set(terms(chunk.text))) / max(1, len(query_terms))
        # Hybrid retrieval followed by a lightweight query-overlap reranker.
        score = 0.45 * (bm25 / lexical_max) + 0.35 * max(0, dense) + 0.20 * overlap
        ranked.append(SearchHit(
            document=chunk.document, page=chunk.page, text=chunk.text, score=round(score, 6)
        ))
    return sorted(ranked, key=lambda hit: hit.score, reverse=True)[:top_k]
