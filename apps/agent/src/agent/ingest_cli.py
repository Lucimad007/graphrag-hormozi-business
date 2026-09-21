from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.config.logging import configure_logging
from agent.config.settings import get_settings
from agent.ingestion.service import IngestionService


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Ingest a file or directory of legally obtained documents."
    )
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        help="File or directory to ingest",
    )
    parser.add_argument(
        "--suffix",
        action="append",
        default=[],
        help="Only ingest these extensions (repeatable), e.g. --suffix .pdf",
    )
    parser.add_argument(
        "--retry-extractions",
        action="store_true",
        help="Re-run LLM extraction for stored chunks that have no graph entities.",
    )
    parser.add_argument(
        "--ids-file",
        type=Path,
        help="Optional newline-separated chunk ids to retry (with --retry-extractions).",
    )
    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level)
    service = IngestionService.from_settings(settings)
    if args.retry_extractions:
        chunk_ids = None
        if args.ids_file is not None:
            chunk_ids = {
                line.strip()
                for line in args.ids_file.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.strip().startswith("#")
            }
        result = service.retry_missing_extractions(chunk_ids=chunk_ids)
        print(json.dumps(result.model_dump(), indent=2))
        return
    if args.path is None:
        raise SystemExit("path is required unless --retry-extractions is set")
    target = args.path.expanduser()
    suffixes = {
        item.lower() if item.startswith(".") else f".{item.lower()}"
        for item in args.suffix
    }
    if target.is_dir():
        result = service.ingest_tree(target, suffixes=suffixes or None)
    elif target.is_file():
        result = service.ingest_path(target)
    else:
        raise SystemExit(f"Not found: {target}")
    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    main()
