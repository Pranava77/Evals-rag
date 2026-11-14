
# RAG Evaluation Harness

A self-contained evaluation harness for a Retrieval-Augmented Generation (RAG) pipeline. It measures **retrieval quality** (recall@k) and **answer faithfulness** (LLM-as-judge), then gates a build pass/fail against configurable thresholds.

## Architecture

```
data/docs/          5 Markdown knowledge-base articles
data/golden.jsonl   10 golden test cases (easy, exact-term, hard, adversarial, out-of-scope)
src/
├── chunkers.py     Chunking strategies (paragraph, fixed-size, parent-child)
├── embedder.py     Embedders (FakeEmbedder for offline, OllamaEmbedder for local)
├── store.py        Vector stores (InMemory, Chroma, pgvector) — same 2-method interface
├── retrieval.py    Dense, keyword (BM25-lite), and hybrid retrieval with rank fusion
├── rag_seed.py     Bundles chunker + embedder + store + LLM into a working RAG pipeline
├── llm_client.py   Unified LLM interface (Claude, Gemini, Ollama, Fake)
├── metrics.py      recall@k (set overlap) and faithfulness (LLM judge)
├── eval_runner.py  Runs the golden set, prints a results table, and gates the build
└── prompt_ab.py    A/B-test two prompt variants on the same golden set
tests/
├── test_metrics.py Unit tests for recall@k and faithfulness
└── test_gate.py    Integration tests for the full eval pipeline + gate logic
```

## Quick Start

```bash
pip install -r requirements.txt
```


### Run the tests (offline — no API key needed)


```bash
python -m pytest -q
```

All 8 tests use `FakeLLMClient` — deterministic, zero network calls.


### Run the evaluation pipeline (requires an LLM backend)

Create a `.env` file with **one** of these:

```env
# Option 1: Anthropic (recommended)
ANTHROPIC_API_KEY=sk-ant-...

# Option 2: Gemini (free tier)
GEMINI_API_KEY=...

# Option 3: Ollama (local, free — needs `ollama serve` running)
OLLAMA_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434
```

Then run the harness:

```bash
python -m src.eval_runner                   # healthy pipeline → GATE: PASS
python -m src.eval_runner --break-chunking  # sabotaged chunks → GATE: FAIL
```

### Prompt A/B testing

Compare two prompt variants on the same golden set:

```bash
python -m src.prompt_ab
```

## Metrics

| Metric           | How it works                                                                                 | Cost                |
| ---------------- | -------------------------------------------------------------------------------------------- | ------------------- |
| **recall@k**     | \|expected ∩ retrieved\| / \|expected\| — pure set overlap, no LLM                           | Free, microseconds  |
| **faithfulness** | LLM-as-judge: "Is every claim in the answer supported by the context?" → binary JSON verdict | 1 LLM call per case |

## Gate Thresholds

The build passes only if **both** mean scores clear their bars (defined in `eval_runner.py`):

| Metric            | Threshold |
| ----------------- | --------- |
| Mean recall@k     | ≥ 0.8     |
| Mean faithfulness | ≥ 0.6     |

## Golden Set

The 10 cases in `data/golden.jsonl` cover five difficulty tiers:

| Kind           | Cases | Purpose                                                  |
| -------------- | ----- | -------------------------------------------------------- |
| `easy`         | 4     | Basic fact retrieval                                     |
| `exact-term`   | 2     | Tests keyword/hybrid retrieval (error codes)             |
| `hard`         | 1     | Requires deeper context matching                         |
| `adversarial`  | 1     | Prompt injection — the pipeline should refuse            |
| `out-of-scope` | 1     | Question not in the docs — scored but skipped in metrics |

## Docker

```bash
docker build -t rag-eval .
docker run --env-file .env rag-eval python -m src.eval_runner
```

## LLM Backends

The pipeline uses a single `LLMClient.generate(prompt)` interface. Swap backends with an env var — no code changes:

| Backend            | Key                 | Model default           |
| ------------------ | ------------------- | ----------------------- |
| Claude (Anthropic) | `ANTHROPIC_API_KEY` | `claude-sonnet-5`       |
| Gemini (Google)    | `GEMINI_API_KEY`    | `gemini-3.5-flash-lite` |
| Ollama (local)     | —                   | `llama3.2`              |

## Vector Store Tiers

Same two-method interface (`add` / `top_k`), swap by config:

| Tier             | Class           | Dependency                               |
| ---------------- | --------------- | ---------------------------------------- |
| Teaching / tests | `InMemoryStore` | None                                     |
| On-disk          | `ChromaStore`   | `pip install chromadb`                   |
| Production       | `PgVectorStore` | `pip install psycopg[binary]` + pgvector |
