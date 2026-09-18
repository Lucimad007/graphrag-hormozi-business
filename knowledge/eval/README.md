# Evaluation dataset

Golden questions over **synthetic** notes in `knowledge/samples/`. Use this to check hybrid retrieval without a live LLM or copyrighted books.

```bash
uv run pytest apps/agent/tests/test_eval_dataset.py
```

Each case lists a query plus expected documents, entity ids, and relationship types. Extraction in the eval harness is scripted so the graph is stable; embeddings are lexical so ranking is deterministic.
