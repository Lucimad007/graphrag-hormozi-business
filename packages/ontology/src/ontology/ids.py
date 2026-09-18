from __future__ import annotations

import re

_SLUG = re.compile(r"[^a-z0-9]+")


def slug(value: str) -> str:
    compact = _SLUG.sub("-", value.strip().lower()).strip("-")
    return compact or "unnamed"


def entity_id(entity_type: str, name: str) -> str:
    return f"{slug(entity_type)}:{slug(name)}"


def relationship_id(rel_type: str, source_id: str, target_id: str) -> str:
    return f"{slug(rel_type)}|{source_id}|{target_id}"
