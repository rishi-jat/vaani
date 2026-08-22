"""LLM generation & query reformulation for MSMARCO-XI.

Uses Groq (fastest), xAI (Grok), or OpenAI if configured.
Falls back gracefully to extractive grounding if unconfigured or on error.
"""

from __future__ import annotations

import re
from vaani.config import Settings, get_settings
from vaani.index import StoredChunk


class GenerateError(RuntimeError):
    pass


SYSTEM_POLISH = (
    "You rewrite a grounded extractive answer for a spoken assistant. "
    "Reply with ONLY the direct, concise answer to the question — "
    "the entity, number, name, or short defining phrase. "
    "Do not restate background, lists, or extra sentences from the passages. "
    "Use ONLY facts present in the extractive answer and passages. "
    "Keep the same language as the question. "
    "One phrase, or at most one short sentence. No preamble."
)

SYSTEM_TRANSLATE = (
    "You are an expert search query translator for an Indic RAG system over the MSMARCO-XI Hindi dataset. "
    "Translate the user's English or Hinglish query into a direct, search-optimized Hindi query (Devanagari script) "
    "that matches the vocabulary of MSMARCO-XI Hindi passages. "
    "Output ONLY the Hindi query text. No preamble, no quotes, no explanations."
)

# Offline dictionary mapping for common English search terms to Devanagari Hindi
_TERM_MAP: list[tuple[str, str]] = [
    ("economic capital of india", "भारत की आर्थिक राजधानी"),
    ("financial capital of india", "भारत की वित्तीय राजधानी"),
    ("capital of india", "भारत की राजधानी"),
    ("capital of maharashtra", "महाराष्ट्र की राजधानी"),
    ("capital of goa", "गोवा की राजधानी"),
    ("what is the economic capital", "आर्थिक राजधानी क्या है"),
    ("what is the financial capital", "वित्तीय राजधानी क्या है"),
    ("what is a corporation", "कॉर्पोरेशन क्या है"),
    ("what is corporation", "कॉर्पोरेशन क्या है"),
    ("what is machine learning", "मशीन लर्निंग क्या है"),
    ("what is artificial intelligence", "आर्टिफिशियल इंटेलिजेंस क्या है"),
    ("what is computer", "कम्प्यूटर क्या है"),
    ("what is the internet", "इंटरनेट क्या है"),
    ("what is weather", "मौसम क्या है"),
    ("what is solar system", "सौर मंडल क्या है"),
    ("economic capital", "आर्थिक राजधानी"),
    ("financial capital", "वित्तीय राजधानी"),
    ("capital", "राजधानी"),
    ("india", "भारत"),
    ("weather", "मौसम"),
    ("corporation", "कॉर्पोरेशन"),
]


def _rule_based_translate(query: str) -> str:
    """Fast offline fallback mapping for English queries to Hindi."""
    q_lower = query.lower().strip().rstrip("?").strip()
    for eng, hin in _TERM_MAP:
        if eng in q_lower:
            return hin + (" क्या है?" if not hin.endswith("है") and "क्या" not in hin else "")
    # General conversion: replace what is X -> X क्या है
    m = re.match(r"^(?:what\s+is|what\s+are|who\s+is|tell\s+me\s+about)\s+(?:the\s+|a\s+|an\s+)?(.+)$", q_lower)
    if m:
        term = m.group(1).strip()
        return f"{term} क्या है?"
    return query


def translate_query_for_msmarco(
    query: str,
    settings: Settings | None = None,
) -> str:
    """Reformulate/translate an English/Hinglish query to Hindi for MSMARCO-XI retrieval."""
    if not query or not re.search(r"[a-zA-Z]", query):
        return query

    settings = settings or get_settings()
    api_key = settings.active_llm_api_key
    if not api_key:
        return _rule_based_translate(query)

    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=settings.active_llm_base_url)
    try:
        resp = client.chat.completions.create(
            model=settings.active_llm_model,
            temperature=0.0,
            max_tokens=60,
            messages=[
                {"role": "system", "content": SYSTEM_TRANSLATE},
                {"role": "user", "content": f"Query: {query}"},
            ],
            timeout=min(1.5, settings.generate_timeout_s),
        )
        translated = (resp.choices[0].message.content or "").strip().strip('"').strip("'")
        if translated and re.search(r"[\u0900-\u097F]", translated):
            return translated
    except Exception:
        pass

    return _rule_based_translate(query)


def polish(
    question: str,
    extractive: str,
    passages: list[StoredChunk],
    settings: Settings | None = None,
) -> str:
    settings = settings or get_settings()
    api_key = settings.active_llm_api_key
    if not api_key:
        raise GenerateError("No LLM API key configured (GROQ_API_KEY / XAI_API_KEY / OPENAI_API_KEY)")
    from openai import OpenAI

    ctx = "\n\n".join(
        f"[{c.parent_id} | {c.lang} | {c.query_type}]\n{c.parent_text}"
        for c in passages[:6]
    )
    user = (
        f"Question: {question}\n\n"
        f"Extractive answer (keep these facts): {extractive}\n\n"
        f"Passages:\n{ctx}"
    )
    client = OpenAI(api_key=api_key, base_url=settings.active_llm_base_url)
    try:
        resp = client.chat.completions.create(
            model=settings.active_llm_model,
            temperature=0.0,
            max_tokens=80,
            messages=[
                {"role": "system", "content": SYSTEM_POLISH},
                {"role": "user", "content": user},
            ],
            timeout=settings.generate_timeout_s,
        )
    except Exception as e:  # noqa: BLE001 — harness treats any failure as fallback
        raise GenerateError(str(e)) from e
    text = (resp.choices[0].message.content or "").strip()
    if not text:
        raise GenerateError("empty generation")
    return text
