# Design Decisions

## Chunk Size and Overlap

The default chunk size is 500 approximate tokens with 80 approximate tokens of overlap. The chunker estimates tokens as whitespace words times 0.75, which is simple, deterministic, and good enough for this synthetic corpus. The overlap keeps important clinical context near chunk boundaries without creating a large number of duplicate vectors.

## BGE M3 for German

The default embedding model is `BAAI/bge-m3` because it has multilingual coverage and handles German retrieval better than English-only models. German compounds and mixed clinical phrasing benefit from a model trained across languages. Embeddings are normalised dense vectors so cosine search in Qdrant is stable and easy to reason about.

## Score Threshold Refusal

The RAG pipeline refuses to answer when the best retrieved score is below `SCORE_THRESHOLD`. This is a hallucination guard: the system should say the corpus lacks information instead of inventing clinical detail. The refusal path runs before any LLM call, which also keeps load testing predictable with `generate=false`.

## Schema-Validated Extraction

Clinical note extraction uses the `ClinicalNote` Pydantic schema as the contract. The LLM is asked for JSON only, then validation either succeeds or returns concrete errors to the retry prompt. After the final retry, callers receive a typed failure object with raw output and errors, not an exception.

## Local-First Architecture

The demo keeps ASR, embeddings, vector search, and LLM generation local. This mirrors hospital data-residency constraints where clinical audio and notes should not leave controlled infrastructure. External paid APIs are intentionally excluded from the core pipeline.

## Future Work

Useful next steps are hybrid sparse plus dense retrieval, a cross-encoder reranker, streaming ASR, and federated training across hospital sites. Those additions would improve retrieval precision, responsiveness, and governance for larger deployments.
