# SearchForge

SearchForge is a compact information-retrieval service and learning project. It normalizes Unicode text, builds an inverted index, ranks lexical matches with Okapi BM25, persists documents in SQLite, and exposes CRUD and search through FastAPI.

## Features

- Unicode-aware normalization and tokenization (NFKC, case folding, punctuation cleanup).
- Inverted index with posting lists for candidate generation.
- Okapi BM25 ranking using corpus-wide document frequency and average document length.
- SQLite persistence with deterministic index recovery and synchronized create, update, and delete operations.
- Deterministic ranking ties (ascending document ID).
- Precision@k, DCG@k, and nDCG@k evaluation utilities with a labeled query set.
- FastAPI endpoints with Pydantic validation, pagination, consistent errors, and automated integration tests.
- Reproducible indexed-versus-full-scan benchmarks and a BM25-versus-term-overlap quality evaluation.

## Architecture

The query pipeline is: normalize and tokenize the query, union the postings for its known terms, score those candidate documents with corpus-wide BM25 statistics, then sort by descending score and ascending document ID. Empty and unknown-only queries return no hits.

The indexing pipeline stores original document IDs and text in SQLite. On startup, documents are loaded in ID order to rebuild the in-memory inverted index and BM25 ranker. Batch ingestion validates all entries before mutating the live index, and BM25 corpus fitting stages its replacement before publishing it. BM25 rejects non-finite tuning parameters. File databases use SQLite's native WAL mode, `synchronous=FULL`, `BEGIN IMMEDIATE`, and a 5-second busy timeout. New database parent directories are created owner-only and new database files use owner-only permissions; existing file permissions are not changed. Each create, update, or delete builds a replacement engine from the transaction-local document set, commits, and then publishes the replacement. A SQL or rebuild failure before commit rolls back. If an exception occurs after commit but before publication, the service marks its index dirty and rebuilds it from SQLite on the next index-facing read. A restarted process always reconstructs its index from stored rows.

Each service has its own in-memory index and a process-local `RLock`. Search and document-list responses use one SQLite read transaction snapshot for index refresh, ranking/result materialization, and the total/page query. A request whose snapshot starts before a concurrent commit may finish with the prior document state; a later request observes completed commits through SQLite `PRAGMA data_version`. Separate processes do not share an index, and each refresh is a full corpus rebuild. This is per-request snapshot consistency, not a guarantee that a response started before a delete will be suppressed after that delete commits.

`storage.py` owns SQLite transactions, `service.py` coordinates persistence and engine replacement, and `api.py` handles HTTP models, routes, and errors. The index and BM25 algorithms remain independent of the HTTP and persistence layers.

## Setup and Run

Python 3.10 or newer is required. The project uses setuptools metadata in `pyproject.toml`, pip for installation, and pytest for tests. Runtime dependencies are listed in `requirements.txt`; SQLite uses Python's standard library.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install -r requirements.txt
python -m pytest -q
export SEARCHFORGE_DB_PATH="$PWD/searchforge.sqlite3"
uvicorn searchforge.api:app --app-dir src --reload
```

The API listens on `http://127.0.0.1:8000`; interactive OpenAPI docs are at `/docs`. The database defaults to `~/.searchforge/searchforge.sqlite3`. Set `SEARCHFORGE_DB_PATH` to choose another file. A new database starts empty.

## API

| Method | Path | Behavior |
|---|---|---|
| `GET` | `/health` | Status and current document/vocabulary counts |
| `POST` | `/documents` | Create; duplicate ID returns `409` |
| `GET` | `/documents` | List by ID with `limit` and `offset` |
| `GET` | `/documents/{doc_id}` | Fetch one document; missing ID returns `404` |
| `PUT` | `/documents/{doc_id}` | Update an existing document; missing ID returns `404` |
| `DELETE` | `/documents/{doc_id}` | Delete; missing ID returns `404` |
| `GET` | `/search?q=...` | Ranked search with `limit` and `offset` |

Search requires `q` (up to 2,000 characters); `limit` is 1-100 and `offset` is 0-10,000. Document listing uses the same page and offset bounds and executes `COUNT(*)` plus `LIMIT`/`OFFSET` in SQLite rather than loading the entire collection. Document IDs are 1-256 characters and text is capped at 1,000,000 characters. Validation and HTTP errors use an `error` object with a stable code and message; unexpected failures return a generic message without exception details.

```bash
curl -X POST http://127.0.0.1:8000/documents \
  -H 'Content-Type: application/json' \
  -d '{"id":"doc-1","text":"BM25 ranks documents using term frequency"}'

curl 'http://127.0.0.1:8000/search?q=term%20frequency&limit=10&offset=0'
curl -X PUT http://127.0.0.1:8000/documents/doc-1 \
  -H 'Content-Type: application/json' \
  -d '{"id":"doc-1","text":"Updated searchable text"}'
curl -X DELETE http://127.0.0.1:8000/documents/doc-1
```

## Tests and Validation

Run the full suite with:

```bash
python -m pytest -q
```

Last verified in this workspace on 2026-10-26: `132 passed`, with no test failures. Pytest reported nine upstream Starlette/httpx deprecation warnings. Coverage percentage was not measured.

The tests cover normalization, indexing, ranking and deterministic ties, empty and unknown queries, CRUD and stale-posting removal, atomic rejection of invalid batches and BM25 refits, non-finite BM25 parameter rejection, SQLite restart recovery, SQL-triggered write failures for each mutation, index-rebuild failures, exceptions after commit, process exit immediately before and after commit, HTTP validation and safe error responses, bounded SQL pagination, snapshot-consistent search interleavings, and concurrent requests. Threaded `TestClient` checks exercise simultaneous searches and mutations. Spawned-process tests use separate SQLite connections to contend on creates, updates, and deletes; verify one-winner duplicate creation; verify a still-running observer refreshes; and verify recovery after restart. WAL corruption tests use disposable database copies with a truncated tail, damaged checksum, and malformed header: SQLite retained the checkpointed rows and passed `integrity_check`, while the later WAL-only transaction was unavailable. This characterizes SQLite's native behavior; SearchForge does not parse or repair WAL records. These deterministic tests cover selected schedules and local process termination only, not general crash-safety or behavior on every filesystem.

`python -m pip check` reported no broken requirements. `python -m pip_audit -r requirements.txt` reported no known vulnerabilities on 2026-10-26. The audit tool is installed only in the ignored virtual environment and is not a project dependency. A source scan found no checked uses of dynamic `eval`/`exec`, pickle, shell execution, unsafe YAML loading, or common credential patterns; no `.env` or credential-named files were found. These automated checks do not constitute a comprehensive security audit.

## Benchmarks

Run the deterministic synthetic benchmark after installation:

```bash
python benchmarks/benchmark_search.py --sizes 100 500 1000 --seed 20261026 --runs 5 --warmups 2 --output benchmark-results.json
```

For each corpus size the script reports index/BM25 build time and BM25-baseline build time, per-query nearest-rank p50/p95 latency, sequential query throughput, and P@10/nDCG@10. Every query is checked for exact ranking parity before timing. Both methods use the same documents, normalized OR-match semantics, corpus-wide BM25 implementation, and tie-break behavior. The full-scan method tokenizes/checks every document per query; its timed path includes that scan plus scoring and sorting. Build and warmup are excluded from query latency. Throughput is single-process sequential and is not a concurrent capacity measurement.

The benchmark uses a fixed seed, three shared terms, 5-45 filler terms, and five planted topic documents with six fixed query/judgment cases. Results below were measured on 2026-10-26 in this workspace with Python 3.12.3 on Linux 6.8.0-124-generic, Intel Pentium B940 @ 2.00GHz, and 2 logical CPUs. Each query had 5 timed repetitions after 2 warmups. Latency columns are the median across the six queries of each query's p50 or p95; throughput is the mean of the six per-query measured throughputs. Build times are one observed run and should not be interpreted as stable estimates.

| Documents | Method | Build (s) | Median query p50 (ms) | Median query p95 (ms) | Mean query throughput (q/s) |
|---:|---|---:|---:|---:|---:|
| 100 | Inverted index + BM25 | 0.013947 | 0.023589 | 0.025286 | 48,120.1 |
| 100 | Full scan + BM25 | 0.005981 | 3.377853 | 3.557733 | 290.5 |
| 500 | Inverted index + BM25 | 0.047102 | 0.023852 | 0.024009 | 48,344.0 |
| 500 | Full scan + BM25 | 0.025525 | 17.349266 | 17.462747 | 56.2 |
| 1,000 | Inverted index + BM25 | 0.093935 | 0.023697 | 0.023998 | 47,934.0 |
| 1,000 | Full scan + BM25 | 0.050497 | 34.480634 | 37.376697 | 28.4 |

In this synthetic run, the indexed query path had lower measured latency and higher measured sequential throughput at all three sizes, while its observed build times were longer at each size. These figures describe one machine, generated corpus, six-query workload, and five timed repetitions; they are not production performance claims. Build timings are single observations and need repeated independent process runs for stronger conclusions.

## Search Quality

Run the judged comparison with:

```bash
python benchmarks/evaluate_quality.py
```

The existing `tests/data/bm25_eval.json` contains 20 hand-authored documents and six representative queries with graded judgments. Both BM25 and the baseline rank the same documents containing at least one normalized query term. The baseline counts distinct query terms present in each document and breaks ties by ascending ID. P@10 uses a fixed denominator of 10; nDCG@10 uses graded gain.

Measured macro averages from this collection:

| Method | Precision@10 | nDCG@10 |
|---|---:|---:|
| BM25 | 0.2333 | 0.9088 |
| Distinct-term overlap | 0.2500 | 0.9048 |

The methods are close and trade small differences across two metrics. Six hand-judged queries are too few to establish which ranker is generally better. The judgments are development fixtures, not an independently collected or representative evaluation set.

## Limitations and Future Work

- Candidate generation uses the union (OR) of postings for query terms; this is not Boolean AND retrieval.
- BM25 is rebuilt after every mutation, so write cost grows with corpus size.
- The in-memory index remains process-local. `PRAGMA data_version` detects commits by other connections before indexed reads, but rebuilding is O(corpus size), and a search overlapping an external commit is not a globally atomic snapshot.
- SQLite write transactions use a 5-second busy timeout; heavier contention can still return a database lock error. Multi-process tests cover a small number of spawned local processes and selected schedules only.
- Process termination was injected before commit and after commit on the local test filesystem. Power loss, filesystem/device failure, recovery from corrupted committed WAL-only frames, network filesystems, and long-running contention are not guaranteed; SQLite may ignore an invalid WAL tail and expose only checkpointed data. `synchronous=FULL` requests SQLite's full WAL sync for commits but is not proof against hardware, filesystem, or power failures.
- There is no authentication, authorization, request rate limiting, or production deployment hardening.
- Search query length is limited to 2,000 characters, document IDs to 256 characters, document text to 1,000,000 characters, response page size to 100, and offsets to 10,000. Search still ranks all matching candidates before applying the page, so large queries/corpora can consume substantial CPU and memory.
- The SQLite path is supplied by trusted application configuration and is not exposed as a request parameter or constrained to an application directory. File permissions, symlink policy, and deployment directory ownership are not hardened by the application.
- Benchmark and quality datasets are small and synthetic/hand-judged. Results are not representative of real-world search traffic.
- Useful next steps include broader independently judged queries, repeated benchmark processes, larger and varied corpora, multi-process consistency tests, and a production-oriented update strategy that avoids full BM25 rebuilds.

## Project layout

```text
src/searchforge/
  tokenizer.py   # normalization and tokenization
  index.py       # inverted index and document store
  bm25.py        # BM25 statistics and scoring
  search.py      # candidate retrieval and ranked search
  storage.py     # SQLite document persistence
  service.py     # persistence/index consistency boundary
  metrics.py     # retrieval evaluation metrics
  api.py         # FastAPI HTTP layer
benchmarks/
  benchmark_search.py  # indexed/full-scan benchmark
  evaluate_quality.py # BM25/term-overlap judged evaluation
tests/
  test_api.py
  test_service.py
  test_tokenizer.py
  test_index.py
  test_bm25.py
  test_search.py
  test_metrics.py
  test_bm25_eval.py
```
