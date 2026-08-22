"""Query-focused relevance: coverage + genitive attachment.

Bag-of-words coverage treats "भारत के महाराष्ट्र राज्य की राजधानी" as a
hit for "भारत की राजधानी". The owner of राजधानी in that span is
महाराष्ट्र/राज्य, not भारत. We detect that mismatch instead of
special-casing Delhi.
"""

from __future__ import annotations

from vaani.guardrails import _STOP, content_tokens, query_coverage
from vaani.text import tokenize


_GENITIVE = frozenset({"की", "का", "के", "चा", "ची", "चे", "च्या", "च्यांचे", "of"})
_PRONOUN_OWNERS = frozenset(
    {
        "इसकी",
        "इसका",
        "इसके",
        "उसकी",
        "उसका",
        "उसके",
        "उनकी",
        "त्याची",
        "त्याचा",
        "त्याचे",
        "त्यांची",
        "याची",
        "याचा",
        "याचे",
        "यांची",
        "माझी",
        "माझा",
        "माझे",
        "their",
        "its",
        "his",
        "her",
    }
)


def genitive_pairs(text: str) -> list[tuple[str, str]]:
    """(owner, prop) from 'owner की prop' / 'owner ची prop' / 'prop of owner' / 'ownerची prop'."""
    toks = tokenize(text)
    pairs: list[tuple[str, str]] = []
    marathi_suffixes = ("च्यांचे", "च्या", "ची", "चा", "चे")
    for i, t in enumerate(toks):
        if t in _GENITIVE and t != "of" and i > 0 and i + 1 < len(toks):
            pairs.append((toks[i - 1], toks[i + 1]))
        elif t == "of" and i > 0 and i + 1 < len(toks):
            pairs.append((toks[i + 1], toks[i - 1]))
        elif i + 1 < len(toks):
            for sfx in marathi_suffixes:
                if t.endswith(sfx) and len(t) > len(sfx) + 1:
                    base_owner = t[: -len(sfx)]
                    pairs.append((base_owner, toks[i + 1]))
                    break
    return pairs


def query_properties(query: str) -> dict[str, str]:
    """Map property → expected owner for genitive phrases in the query."""
    out: dict[str, str] = {}
    for owner, prop in genitive_pairs(query):
        if owner in _STOP or prop in _STOP:
            continue
        out[prop] = owner
    return out


def attachment_conflict(query: str, text: str) -> bool:
    expected = query_properties(query)
    if not expected:
        return False
    for owner, prop in genitive_pairs(text):
        if prop not in expected:
            continue
        if owner in _PRONOUN_OWNERS or owner in _STOP:
            continue
        if owner != expected[prop]:
            return True
    return False


def relevance_tuple(query: str, text: str, fused: float, dense: float) -> tuple:
    """Sort key, higher is better. Used to rerank the hybrid pool."""
    conflict = attachment_conflict(query, text)
    cover = query_coverage(query, text)
    q_norm = query.strip().rstrip("?").strip().lower()
    exact_match = 1.0 if q_norm and q_norm in text.lower() else 0.0
    return (not conflict, exact_match, cover, dense, fused)


def rerank_hits(query: str, hits: list) -> list:
    return sorted(
        hits,
        key=lambda h: relevance_tuple(query, h[0].parent_text, h[1], h[2]),
        reverse=True,
    )
