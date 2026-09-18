import hashlib
import re
from pathlib import Path

from ontology import slug


def document_id_for(source: str, raw: bytes | None = None) -> str:
    stem = slug(Path(source).stem or "document")
    if raw is None:
        return stem
    digest = hashlib.sha256(raw).hexdigest()[:12]
    return f"{stem}:{digest}"


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\ufeff", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [" ".join(line.split()) if line.strip() else "" for line in text.split("\n")]
    collapsed = "\n".join(lines).strip()
    return collapsed
