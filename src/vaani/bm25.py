"""In-process BM25 over script-aware tokens."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from vaani.text import tokenize


@dataclass
class BM25:
    k1: float = 1.5
    b: float = 0.75
    df: dict[str, int] = field(default_factory=dict)
    tf: list[dict[str, int]] = field(default_factory=list)
    dl: list[int] = field(default_factory=list)
    avgdl: float = 0.0
    n: int = 0
    inverted: dict[str, list[int]] = field(default_factory=dict)

    def fit(self, docs: list[str]) -> "BM25":
        self.tf = []
        self.dl = []
        self.df = {}
        self.inverted = {}
        for i, doc in enumerate(docs):
            toks = tokenize(doc)
            bag: dict[str, int] = {}
            for t in toks:
                bag[t] = bag.get(t, 0) + 1
            self.tf.append(bag)
            self.dl.append(len(toks))
            for t in bag:
                self.df[t] = self.df.get(t, 0) + 1
                self.inverted.setdefault(t, []).append(i)
        self.n = len(docs)
        self.avgdl = (sum(self.dl) / self.n) if self.n else 0.0
        return self

    def _idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        # Lucene-style idf, floored at epsilon so unseen terms don't explode
        return math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        q_terms = tokenize(query)
        if not q_terms or self.n == 0:
            return []
        scores: dict[int, float] = {}
        seen: set[str] = set()

        # Stop words set for token weighting
        from vaani.guardrails import _STOP

        for term in q_terms:
            if term in seen:
                continue
            seen.add(term)
            postings = self.inverted.get(term)
            if not postings:
                continue
            idf = self._idf(term)
            # Boost non-stop content tokens
            weight = 1.6 if term not in _STOP and len(term) > 1 else 0.85
            for i in postings:
                freq = self.tf[i][term]
                dl = self.dl[i]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
                scores[i] = scores.get(i, 0.0) + (idf * weight) * (freq * (self.k1 + 1) / denom)

        # Contiguous bigram co-occurrence boost
        if len(q_terms) >= 2:
            for j in range(len(q_terms) - 1):
                t1, t2 = q_terms[j], q_terms[j + 1]
                if t1 in _STOP and t2 in _STOP:
                    continue
                p1 = self.inverted.get(t1)
                p2 = self.inverted.get(t2)
                if p1 and p2:
                    common = set(p1) & set(p2)
                    for doc_idx in common:
                        if doc_idx in scores:
                            scores[doc_idx] += 0.85 * (self._idf(t1) + self._idf(t2))

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def to_state(self) -> dict:
        return {
            "k1": self.k1,
            "b": self.b,
            "df": self.df,
            "tf": self.tf,
            "dl": self.dl,
            "avgdl": self.avgdl,
            "n": self.n,
            "inverted": self.inverted,
        }

    @classmethod
    def from_state(cls, state: dict) -> "BM25":
        obj = cls(k1=state["k1"], b=state["b"])
        obj.df = {str(k): int(v) for k, v in state["df"].items()}
        obj.tf = [{str(k): int(v) for k, v in bag.items()} for bag in state["tf"]]
        obj.dl = [int(x) for x in state["dl"]]
        obj.avgdl = float(state["avgdl"])
        obj.n = int(state["n"])
        obj.inverted = {str(k): [int(i) for i in v] for k, v in state["inverted"].items()}
        return obj
