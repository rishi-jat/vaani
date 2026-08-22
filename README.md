# Vaani

Voice-enabled RAG over [ai4bharat/MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI) for **HH Goa 2026 Task 2**.

**Live:** [https://vaani-production-d1eb.up.railway.app](https://vaani-production-d1eb.up.railway.app)

Speak a question in Hindi or English (with Hinglish code-mixing). The system transcribes it (Sarvam Saaras v3 or ElevenLabs Scribe), retrieves from the MSMARCO-XI Hindi-val index, and returns a **grounded extractive answer with citations**. If the corpus does not support an answer, it abstains.

Two measured surfaces exist. Do not mix their numbers:

| Surface | Retrieval | RAM | What was measured |
|---------|-----------|-----|-------------------|
| Local Compose / in-process harness | hybrid e5-small + FAISS + BM25 + RRF | ~1–2 GB | bench P50/P70/P100, Delhi extract, Sarvam HTTP |
| Public Railway (this URL) | BM25-only (`dense: false`) | Trial **1 GB cap** | live Sarvam audio, password refuse, restart persist |

Railway refused `memoryGB: 2` (`The maximum allowed memory for this service is 1 GB`). Loading FAISS + e5-small + the 219 MB sidecar OOMs there, so the public process is `VAANI_LOW_MEM=true`.

## Submission (verified only)

| Item | Value |
|------|--------|
| Live | https://vaani-production-d1eb.up.railway.app |
| GitHub | https://github.com/rishi-jat/vaani |
| Dataset | `ai4bharat/MSMARCO-XI` Hindi validation, 57,331 unique selected passages |
| STT | Sarvam Saaras v3 (public audio POSTs succeeded) |
| RAG P50 / P70 / P100 | **46.9 / 56.7 / 128.8 ms** — local hybrid, 200 val queries, transcript→extract only (`data/reports/bench.json`) |
| Public audio wall | **~1.3–1.9 s** — STT 1156–1330 ms + RAG 78–111 ms. Not under 200 ms |
| Form | https://forms.gle/MNvCjcv23Hn2Eeu58 |
| Deadline | 22 Aug 2026, 23:59. No resubmissions |

**Demo (Video 2):** open the Railway URL → allow mic → speak `कॉर्पोरेशन क्या है?` or `भारत की राजधानी क्या है?` → show Sarvam transcript + grounded extract. Then type a password question and show `refuse`. Public retrieval is BM25-only (`dense: false`). Local Compose is hybrid and is the source of the P50/P70/P100 table. Say both; do not mix them.

**Video 1:** 90 s process, not the product. Post both videos on Instagram, X, and LinkedIn, every teammate, with `#RAGInGoa`.

**Do not say:** full voice pipeline &lt; 200 ms; Railway is hybrid/vector; public capital answer contains दिल्ली; 14-language or train-split coverage.

Design: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

## Architecture

```
voice ──► Sarvam / ElevenLabs STT          (outside 200ms)
              │
              ▼
     input guard → [local: embed + FAISS + BM25 + RRF]
                   [Railway public: BM25 only]
              │
              ▼
     extractive span → grounding gate      ◄── measured window, target <200ms
              │
              ├── refuse / abstain
              └── optional Grok polish     (outside 200ms; falls back to extract)
```

The extractive answer **is** the final RAG output. It is an exact substring of retrieved passages. An LLM timeout cannot take the answer with it.

## Stack & Components

| Piece | Choice |
|-------|--------|
| STT | Sarvam `saaras:v3` (default) or ElevenLabs `scribe_v2` with dynamic language codes (`hi-IN`, `en-IN`, `auto`) and VAD |
| Query Reformulation | Cross-lingual translation to Hindi search representation via Groq (`llama-3.1-8b-instant`), xAI, or rule-based offline mapper |
| Embeddings | local `intfloat/multilingual-e5-small` (384-d, quantized) |
| Dense Index | FAISS HNSW (`efSearch=64`, `efConstruction=80`) |
| Sparse Index | Script-aware BM25 with **content-token weighting** (1.6x) & **contiguous bigram co-occurrence boost** |
| Fusion | Reciprocal Rank Fusion ($k=60$) |
| Reranking | Multi-criteria reranker with genitive property attachment validation & exact phrase boost |
| Answer | Sentence-level extractive span isolation with citation deduplication |
| Guardrails | Input intent refusal, off-topic detection, query coverage gate, and grounding verification |
| Polish | Groq / xAI Grok (`grok-4.5`) — optional & outside the 200ms budget |
| Concurrency | FastAPI with `CORSMiddleware` + `starlette.concurrency.run_in_threadpool` |
| Web UI | Interactive UI with visual latency waterfall bar, live benchmark modal, strategy ablation comparison, and audio readout |
| Dataset | MSMARCO-XI Hindi val, **all 57,331 unique selected passages** |

## API Surface

| Method | Route | Description |
|--------|-------|-------------|
| `POST` | `/api/ask` | Multipart audio or text input $\rightarrow$ grounded extractive answer with timings and citations |
| `POST` | `/api/ask.json` | JSON payload endpoint (`{ "text": "...", "polish": false }`) |
| `POST` | `/api/compare` | Dynamic strategy ablation comparing `whole`, `fixed_256`, `sentence`, `window_2`, `semantic`, `parent_child`, `metadata` |
| `POST` | `/api/transcribe` | Standalone audio STT transcription endpoint with telemetry |
| `GET` | `/api/benchmark` | In-process latency benchmark over validation queries with P50/P70/P90/P100 percentiles |
| `GET` | `/api/health` | Service health, index size, active retrieval mode (hybrid vs BM25), and provider status |

## Quick start

Python 3.11. From this directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
cp .env.example .env   # add SARVAM_API_KEY (or ELEVENLABS_API_KEY)
```

Inspect the dataset (streams; does not download 55 GB):

```bash
python scripts/inspect_dataset.py --langs hi --split validation --limit 400
```

Build an index and a held-out eval set:

```bash
python scripts/ingest.py --strategy whole --max-passages 60000 --eval-queries 500
```

Unit & integration tests:

```bash
pytest -v
```

Latency + retrieval on real queries:

```bash
python scripts/bench.py --n 200
```

Serve:

```bash
python -m vaani.api
# open http://127.0.0.1:8080
```

Live P50/P70/P100 against the running process:

```bash
curl -s "http://127.0.0.1:8080/api/benchmark?n=80"
```

Local persistent run is **Docker Compose** (hybrid, ~4 GB RAM). Public run is **Railway** at the URL above (BM25-only, 1 GB).

```bash
docker compose up --build
python scripts/deploy_smoke.py --base http://127.0.0.1:8080
```

## What is in the 200ms number

**In:** input guard, query embed, dense+sparse retrieve, RRF, extractive answer, output guard.

**Out:** speech-to-text, optional Grok polish.

`scripts/bench.py` writes `data/reports/bench.json` with raw per-query timings. Percentiles are computed from those rows.

**In-process harness, 2026-08-17** (Apple Silicon, hybrid e5-small + BM25 + RRF, `whole`, **57,331** unique Hindi-val selected passages — every unique selected passage in `hinval.parquet` — 200 held-out val queries, one at a time, 15 warmup dropped):

| | P50 | P70 | P99 | P100 | <200ms | Recall@k |
|---|---:|---:|---:|---:|---:|---:|
| transcript → extractive output | **46.9ms** | **56.7ms** | 117.7ms | **128.8ms** | **200 / 200** | **0.71** |

That window is **not** audio→answer and is **not** the Railway public process (Railway is BM25-only). See `data/reports/bench.json` and `data/reports/corpus_coverage.json`.

Same 4,000-passage / 80-query BM25 ablation (`data/reports/ablation.json`):

| Strategy | Chunks | Recall@k | P50 |
|----------|-------:|---------:|----:|
| **whole** (shipped) | 4000 | **0.763** | 2.8ms |
| fixed_256 | 7833 | **0.763** | 4.0ms |
| metadata | 4000 | **0.763** | 2.8ms |
| sentence | 12192 | 0.688 | 4.4ms |
| window_2 | 11104 | 0.675 | 4.4ms |
| semantic | 12897 | 0.663 | 4.0ms |
| parent_child | 14681 | 0.650 | 4.3ms |

Splitting already-short MSMARCO passages **hurts** sparse retrieval. `whole` and `fixed_256` tie on recall; we ship `whole`. Hybrid e5+BM25 numbers are in the table above. STT and optional Grok polish are outside the 200ms window.

## Guardrails

- **Refuse** credential / weapon / self-harm asks *before* retrieval. Corpus passages about banks will retrieve for “what is my password?” — that is not permission to answer.
- **Abstain** when the best dense score is below a threshold (off-topic) or the extract is not supported by retrieved text.
- **Verify** any Grok rewrite; unsupported polish is dropped.

## Dataset notes

MSMARCO-XI is MS MARCO QnA translated into 14 Indic languages (IndicRAGSuite, arXiv:2506.01615). Full dump is 55.6 GB. The shipped index is **every unique selected passage in Hindi validation** (57,331 of 57,331). Gold labels live in `eval.jsonl` next to the index. See `data/reports/corpus_coverage.json`.

Passages are already short (~50–80 words). If a chunker emits ~1.0 chunks/passage, that is a measurement, and it belongs in the ablation report.

## Env

See `.env.example`. `STT_PROVIDER=sarvam` or `elevenlabs`. Polish needs `XAI_API_KEY`.
