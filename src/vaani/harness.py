"""The only entry the API and the bench call.

Retries, timeouts, typed I/O, and a fallback that is a *second answer
already computed*, not an empty except-block.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from vaani.config import Settings, get_settings
from vaani.embeddings import Encoder
from vaani.extract import extract
from vaani.generate import GenerateError, polish, translate_query_for_msmarco
from vaani.guardrails import (
    GuardDecision,
    abstain_message,
    clip_query,
    coverage_gate,
    generated_is_grounded,
    grounding,
    input_guard,
    off_topic,
    refuse_message,
    support_score,
)
from vaani.relevance import attachment_conflict, rerank_hits
from vaani.index import HybridIndex
from vaani.schema import AskResponse, Citation, Timings
from vaani.stt import STTError, Transcript, transcribe_with_retry
from vaani.text import fold_stt_transcript


def _now() -> float:
    return time.perf_counter()


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _build_citations(hits: list, limit: int = 5) -> list[Citation]:
    seen_parents: set[str] = set()
    citations: list[Citation] = []
    for h in hits:
        chunk = h[0]
        pid = chunk.parent_id
        if pid not in seen_parents:
            seen_parents.add(pid)
            citations.append(
                Citation(
                    passage_id=pid,
                    chunk_id=chunk.chunk_id,
                    lang=chunk.lang,
                    query_type=chunk.query_type,
                    text=chunk.parent_text,
                    score=h[1],
                    strategy=chunk.strategy,
                    rank=len(citations) + 1,
                )
            )
            if len(citations) >= limit:
                break
    return citations


@dataclass
class Pipeline:
    index: HybridIndex
    encoder: Encoder
    settings: Settings

    @classmethod
    def load(cls, settings: Settings | None = None) -> "Pipeline":
        settings = settings or get_settings()
        index = HybridIndex.load(settings=settings)
        enc = Encoder(settings) if index.faiss_index is not None else None
        return cls(index=index, encoder=enc, settings=settings)

    def ask_text(
        self,
        text: str,
        *,
        polish_answer: bool = False,
        language: str = "",
        transcript: str = "",
        stt_ms: float | None = None,
    ) -> AskResponse:
        wall = _now()
        timings = Timings(stt_ms=stt_ms)

        t0 = _now()
        decision = input_guard(text, max_chars=self.settings.max_query_chars)
        timings.guard_in_ms = _ms(t0)
        if not decision.ok:
            timings.total_rag_ms = _ms(wall if stt_ms is None else wall)
            timings.total_ms = timings.total_rag_ms + (stt_ms or 0.0)
            return AskResponse(
                status="refuse",
                answer=refuse_message(text or language),
                transcript=transcript or text,
                language=language,
                timings=timings,
                reason=decision.reason,
                strategy=self.index.strategy,
            )

        query = clip_query(fold_stt_transcript(text), self.settings.max_query_chars)
        search_query = translate_query_for_msmarco(query, self.settings)

        t0 = _now()
        if self.index.faiss_index is not None and self.encoder is not None:
            qvec = self.encoder.encode_query(search_query)
        else:
            import numpy as np

            qvec = np.zeros(0, dtype=np.float32)
        timings.embed_ms = _ms(t0)

        t0 = _now()
        hits = rerank_hits(search_query, self.index.search(search_query, qvec, top_k=self.settings.top_k))
        timings.retrieve_ms = _ms(t0)

        if self.index.faiss_index is not None:
            topic_score = max((h[2] for h in hits), default=0.0)
            topic_cut = self.settings.retrieve_threshold
        else:
            topic_score = max((h[3] for h in hits), default=0.0)
            topic_cut = 1.8
        citations = _build_citations(hits, limit=5)
        ot = off_topic(topic_score, topic_cut)
        if ot.ok:
            ot = coverage_gate(search_query, [h[0].parent_text for h in hits[:3]], threshold=0.6)
        if not ot.ok:
            timings.total_rag_ms = _ms(wall)
            timings.total_ms = timings.total_rag_ms + (stt_ms or 0.0)
            timings.guard_out_ms = 0.0
            return AskResponse(
                status="abstain",
                answer=abstain_message(query or language),
                transcript=transcript or query,
                language=language,
                support=topic_score,
                citations=citations,
                timings=timings,
                reason=ot.reason,
                strategy=self.index.strategy,
                within_budget=timings.total_rag_ms < self.settings.budget_ms,
            )

        t0 = _now()
        ext = extract(search_query, hits)
        timings.extract_ms = _ms(t0)

        t0 = _now()
        contexts = [h[0].parent_text for h in hits]
        if ext is None:
            g = grounding("", contexts, self.settings.support_threshold)
        elif attachment_conflict(search_query, ext.answer):
            g = GuardDecision(False, "abstain", "extract attaches the property to a different owner")
        else:
            g = grounding(ext.answer, contexts, self.settings.support_threshold)
        support = support_score(ext.answer, contexts) if ext else 0.0
        timings.guard_out_ms = _ms(t0)

        citations = _build_citations(hits, limit=5)

        if ext is None or not g.ok:
            timings.total_rag_ms = _ms(wall)
            timings.total_ms = timings.total_rag_ms + (stt_ms or 0.0)
            return AskResponse(
                status="abstain",
                answer=abstain_message(query or language),
                transcript=transcript or query,
                language=language,
                support=support,
                citations=citations,
                timings=timings,
                reason=g.reason if ext else "no extract",
                strategy=self.index.strategy,
                within_budget=timings.total_rag_ms < self.settings.budget_ms,
            )

        answer = ext.answer
        polished = False
        if polish_answer:
            t0 = _now()
            try:
                gen = polish(query, ext.answer, [h[0] for h in hits], self.settings)
                if generated_is_grounded(gen, contexts):
                    answer = gen
                    polished = True
            except GenerateError:
                pass
            timings.generate_ms = _ms(t0)

        timings.total_rag_ms = (
            timings.guard_in_ms
            + timings.embed_ms
            + timings.retrieve_ms
            + timings.extract_ms
            + timings.guard_out_ms
        )
        timings.total_ms = timings.total_rag_ms + (stt_ms or 0.0) + (timings.generate_ms or 0.0)
        return AskResponse(
            status="grounded",
            answer=answer,
            transcript=transcript or query,
            language=language,
            support=support,
            citations=citations,
            timings=timings,
            polished=polished,
            strategy=self.index.strategy,
            within_budget=timings.total_rag_ms < self.settings.budget_ms,
        )

    def ask_audio(
        self,
        audio: bytes,
        *,
        filename: str = "audio.webm",
        mime: str = "audio/webm",
        language: str | None = None,
        polish_answer: bool = False,
    ) -> AskResponse:
        t0 = _now()
        try:
            tr: Transcript = transcribe_with_retry(
                audio,
                filename=filename,
                mime=mime,
                language=language,
                settings=self.settings,
            )
        except STTError as e:
            return AskResponse(
                status="refuse",
                answer="Speech-to-text failed.",
                timings=Timings(stt_ms=_ms(t0)),
                reason=str(e),
                strategy=self.index.strategy,
            )
        stt_ms = _ms(t0)
        return self.ask_text(
            tr.text,
            polish_answer=polish_answer,
            language=tr.language or language or "",
            transcript=tr.text,
            stt_ms=stt_ms,
        )

