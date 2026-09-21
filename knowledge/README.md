This directory is for **legally obtained** business documents that you choose to ingest, plus small **synthetic** samples used in tests and demos.

## What this repository does not contain

Copyrighted books, playbooks, transcripts, or paid course materials are **not** included and must **not** be committed.

This project is **not affiliated with** Alex Hormozi or Acquisition.com. His published works are a **target knowledge domain** for users who already have a lawful copy. They are not a redistributable dataset.

## What you may add locally

- Your own notes and frameworks
- Public-domain or openly licensed strategy material
- Documents you have a license or permission to process
- Synthetic examples that illustrate ontology types (Problem, Metric, Offer, and so on) without copying protected text

Keep local corpora out of git unless they are clearly free to redistribute. `.gitignore` already ignores generic `data/` directories. Ingest from a path you pass to the CLI or API; do not copy paid books into this repo.

A tiny synthetic example lives in [samples/low-close-rate.md](samples/low-close-rate.md). Retrieval eval cases are in [eval/dataset.json](eval/dataset.json).

## How ingestion is intended to work

The pipeline is domain-agnostic:

1. Place Markdown, TXT, or PDF files in a path you control.
2. Call the ingest API or `IngestionService.ingest_path`.
3. Documents are parsed, normalized, chunked, extracted against the **business ontology**, embedded, and stored in Qdrant and Neo4j.

Extraction does not require Hormozi-specific vocabulary. Swap the corpus to change the knowledge source.

## Provenance

Every indexed chunk should retain source, document, and section metadata so answers can cite evidence instead of implying the model memorized a book.
