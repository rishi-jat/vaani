# Vaani — architecture

Voice RAG for **HH Goa 2026 Task 2**, grounded in
[ai4bharat/MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI)
and the official brief (Google Doc linked from [hhgoa.com](https://hhgoa.com/)).

This is the design we implement. Numbers in `data/reports/` are measured,
never invented.

## What the brief actually requires

| # | Requirement | How we meet it |
|---|-------------|----------------|
| 1 | STT via **Sarvam or ElevenLabs** (pick one) | Sarvam Saaras v3 primary, ElevenLabs Scribe as a drop-in |
| 2 | Chunking that is **vast**, not one naive split | Six strategies, ablation on real MSMARCO-XI queries, ship the winner |
| 3 | **< 200ms** for chunking + retrieval + final output | Extractive answer is the in-budget output. STT and optional LLM polish are outside the window and reported separately |
| 4 | **P50 / P70 / P100** on many queries | `scripts/bench.py` + live `/api/benchmark` |
| 5 | A real **harness** | Typed I/O, timeouts, retries, fallback — not a raw prompt |
| 6 | **Guardrails** that know when not to answer | Input refuse + off-topic + grounding abstain + generated-text verify |

## Why the 200ms window cannot include STT or an LLM

Sarvam/ElevenLabs STT is a network call (typically 0.5–2s). Any hosted LLM
is similar. Claiming those inside 200ms would be a lie.

The brief's wording is *"chunking + vector DB retrieval + everything through
to final output"*. The output of the RAG path is a **grounded extractive
answer with citations**, computed locally. That is a complete answer, not a
placeholder. Optional Grok polish runs after, and if it times out or fails
the grounding check we keep the extractive answer.

```
microphone
    │
    ▼
┌──────────────┐   0.5–2s, outside budget
│ Sarvam STT   │
└──────┬───────┘
       │ transcript
       ▼
╔══════════════════════════════════════════════╗
║  200ms BUDGET                                ║
║  input guard → embed → dense+BM25 → RRF      ║
║           → extractive answer → ground gate  ║
╚══════════════════════════════════════════════╝
       │
       ├── refuse / abstain
       └── grounded answer + citations
              │
              ▼
       optional Grok polish (outside budget)
              │
              └── verify or keep extractive
```

## Dataset facts we design around

MSMARCO-XI is the MS MARCO QnA set translated into 14 Indic languages
(IndicRAGSuite, arXiv:2506.01615). Full dump is **55.6 GB / 11.4M rows**.

The Hub card still documents per-language configs (`load_dataset(..., "hi")`)
and jsonl filenames. That is stale. As of 2026-08-17 the repo is parquet
only (`validation/hinval.parquet` = 462 MB, `train/hintrain.parquet` =
3.72 GB), the public config is `default`, and `datasets` will not run
`ms_marco_translations.py`. We download those parquet files directly.

Each row is:

- `query`, `Answer`, `query_id`, `query_type`
- `passages.is_selected[]`, `passages.English_passages[]`, `passages.Translated_passages[]`
- `Eng_Query`, `Eng_Answer`

Measured on the first 300 Hindi validation rows (`data/reports/dataset_inspect.json`):
passage p50 = **58 tokens / 296 chars**, p90 = 100 tokens / 512 chars. A
512-token splitter is a no-op. `window_2` on 12k selected passages produced
**2.77 chunks/parent** because Hindi translations use `।` as well as `.`.
Query p50 is 34 chars; max is 7,783 (a translation loop) — we cap at 512.
Only ~56% of val rows have a selected passage. Hindi val is 97,941 rows.

We do **not** download all 14 languages. Disk and honesty: index unique
selected passages for **Hindi + Marathi** (Devanagari pair, official STT
languages) plus the English originals so bilingual queries work. Gold
selected-passage labels stay in a sidecar for retrieval eval.

## Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Language | Python 3.11 | Wheels for FAISS / torch on macOS + Linux |
| API | FastAPI | One process, typed routes, easy Docker |
| UI | Static HTML/JS | No frontend build. Mic → POST multipart |
| STT | Sarvam `saaras:v3` (`mode=transcribe`) | Indic + code-mix; task-legal. ElevenLabs Scribe is the other adapter |
| Embeddings | `intfloat/multilingual-e5-small` (384-d) | Local, Indic-capable, ~2–5ms / query on CPU |
| Dense index | FAISS HNSW, inner product on L2-normed vectors | No extra service. Sub-5ms at 100k |
| Sparse index | In-process BM25, script-aware tokens | `\w+` shatters Hindi (`दिल्ली` → `द ल ल`). We split on separators |
| Fusion | Reciprocal rank fusion (k=60) | Rank-based, no score calibration |
| Answer (in budget) | Sentence-level extractive | Grounded by construction |
| Polish (optional) | xAI Grok via `https://api.x.ai/v1` | SpaceXAI default; never required for a correct answer |
| Eval | Held-out MSMARCO-XI queries + latency harness | MRR/Recall + P50/P70/P100 |

No Qdrant, no Redis, no GPU requirement. One container.

## Chunking strategies

All six are implemented in `src/vaani/chunking.py`. Ablation lives in
`scripts/ablate.py`. We ship one; the rest stay queryable via `/api/compare`.

1. **whole** — one chunk = one MSMARCO passage. The honest baseline.
2. **fixed** — character windows (128 / 256 / 384) with overlap.
3. **sentence** — split on `। . ? !`, merge crumbs, cap length.
4. **window** — sliding 2-sentence windows, 1-sentence overlap.
5. **semantic** — grow a chunk while consecutive-sentence token Jaccard stays high; cut on topic shift.
6. **parent_child** — retrieve on sentence children, answer from the parent passage.
7. **metadata** — embed `query_type` + `lang` as a prefix. We report the
   caveat: `query_type` is a property of the *owning query*, so gold
   passages carry a label the test query also has.

## Harness

`src/vaani/harness.py` is the only entry the API calls.

```
AskRequest { text? | audio?, lang? }
    → STT if audio (retry ×2, timeout)
    → InputGuard (empty / oversize / unsafe intent)
    → Retriever (dense + sparse + RRF, hard timeout)
    → OffTopicGate (max retrieval score)
    → Extractor (best supported span)
    → GroundingGate (support threshold)
    → optional Generate (timeout) → VerifyGenerated
    → AskResponse { status, answer, citations, timings_ms, support }
```

Statuses: `grounded` | `abstain` | `refuse`. Generation failure is not an
error — the extractive answer already exists.

## Guardrails

- **Input intent** — refuse credential theft, weapons, self-harm, etc.
  *before* retrieval. Retrieval score is not an intent classifier: the
  corpus contains bank-security passages, so "what is my password?" can
  retrieve well and still be wrong to answer.
- **Off-topic** — abstain when the best fused hit is below a measured
  threshold. Threshold is set from a small in-corpus vs. out-of-corpus
  score study, not a guess we never check.
- **Grounding** — extractive answers must be substrings of retrieved
  text. Support = lexical overlap of answer tokens with the source span.
- **Generated-text verify** — if Grok adds tokens that are not supported
  by retrieved passages, discard the polish.

## Latency measurement (no cheating)

Window: last UTF-8 byte of the transcript → serialized `AskResponse`
for the extractive path. We:

- drop a short warmup
- run queries **one at a time**, no batching
- report P50 / P70 / P100 and the fraction under 200ms
- print STT and LLM timings in a separate table
- write the raw per-query JSON so anyone can recompute the percentiles
