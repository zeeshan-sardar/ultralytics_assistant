# Ultralytics Code Assistant

A RAG-based chat assistant that answers questions about the [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) codebase by retrieving relevant source code and generating grounded answers.

---

## Setup

**Prerequisites:** Python 3.10+, [`uv`](https://docs.astral.sh/uv/), a free [MongoDB Atlas](https://www.mongodb.com/atlas) cluster (M0), and a free [OpenRouter](https://openrouter.ai) API key.
```bash
git clone https://github.com/yourusername/ultralytics-assistant
cd ultralytics-assistant
uv sync
cp .env.example .env   # fill in MONGODB_URI and OPENROUTER_API_KEY
```

**Get your MongoDB URI:** Atlas dashboard → your cluster → Connect → Drivers. Replace `<password>` with your database user password.

**Get your OpenRouter key:** openrouter.ai → Keys → Create Key.

---

## Usage

**Index the codebase** (run once, or again after Ultralytics updates):
```bash
uv run python indexer.py
```

Wait ~2 minutes after indexing for the Atlas vector search index to become active.

**Launch the app:**
```bash
uv run streamlit run app.py
```

---

## Example Questions

- How does YOLO handle non-maximum suppression (NMS)?
- How do I train a custom YOLOv8 model on my dataset?
- What does the BasePredictor class do?
- How does the data augmentation pipeline work?
- How can I export a YOLO model to ONNX format?

---

## Design

### Chunking

Files are parsed with Python's `ast` module rather than split by line count. This produces semantically complete chunks where every chunk is a whole class, method, or function with no cuts mid-logic. Long functions (over 80 lines) are split into overlapping windows with a 10-line overlap so context is not lost at boundaries.

**Metadata extracted per chunk:**

| Field | Example |
|---|---|
| `module` | `ultralytics.engine.predictor` |
| `chunk_type` | `method` |
| `name` | `ultralytics.engine.predictor.BasePredictor.predict` |
| `parent_class` | `BasePredictor` |
| `docstring` | `"Perform inference on image(s)."` |
| `decorators` | `["staticmethod"]` |
| `lineno_start / lineno_end` | `142 / 187` |

The text that gets embedded combines all of these fields with the raw source rather than embedding the source alone. Prepending the module path, name, and docstring alongside the code closes the vocabulary gap between natural language questions and code tokens, which significantly improves retrieval quality.

### Models

**Embeddings: `all-MiniLM-L6-v2`** runs locally with no API cost, produces 384-dimensional vectors, and handles mixed text and code well enough for this use case. A code-specific model like `jinaai/jina-embeddings-v2-base-code` would improve precision on syntax-heavy queries.

**LLM: `openrouter/free`** is an OpenRouter meta-model that automatically routes to whichever free model is currently available, so it never returns 404 from a retired model name. The pool falls back to specific named models (Gemini, Llama, Mistral) if the router itself is rate-limited.

**Vector DB: MongoDB Atlas** keeps metadata and vectors in the same document, which simplifies the stack. Chunks are upserted by content hash so re-running the indexer is always safe.

### Trade-offs

Cross-encoder re-ranking was skipped to keep latency low. Query classification and reformulation using chat history were skipped for simplicity. These can improve answer quality and are reasonable next steps.

---

## Future Work

### Retrieval Quality
- Re-rank the top-20 retrieved chunks with a cross-encoder before passing them to the LLM
- Add hybrid search combining vector similarity with BM25 keyword matching for exact class and method name lookups
- Reformulate the query using recent chat history before embedding to handle multi-turn conversations where follow-up questions use pronouns or implicit references

### Critical Missing Features
- CLI interface for querying without the Streamlit UI, useful for scripting and automation
- Unit and integration tests covering the AST chunker, retriever, and generator
- Citation linking that maps each sentence in the answer back to a specific source file and line number
- Version awareness so answers reflect the Ultralytics version the user is actually running
- Guardrails for prompt injection, safety, toxicity, PII scrubbing, and response format enforcement

### Production Readiness
- Replace the local `ultralytics_repo/` directory with a scheduled job that pulls the latest Ultralytics release tag and re-indexes only changed files by comparing file hashes
- Add a Redis layer to cache embeddings for repeated queries and reduce MongoDB round-trips under load
- Switch from the free MongoDB Atlas tier to a dedicated cluster with replica sets for availability
- Experiment tracking using MLflow to track all the changes (models, prompt template, chunk-size, parsing method) in each component of the pipeline.
- RAGAS evaluation to calculate the faitfullness, correctness, precision and recall.
- Rate limiting and request queuing to handle usage spikes without dropping requests or degrading response quality

## Working Screenshots


## Demo
![Alt Text](./docs/demo.gif)