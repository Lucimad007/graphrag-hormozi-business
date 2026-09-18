from ontology import CORE_ONTOLOGY, OntologyRegistry


def extraction_system_prompt(registry: OntologyRegistry | None = None) -> str:
    ont = registry or CORE_ONTOLOGY
    entities = ", ".join(sorted(ont.entity_types))
    relations = ", ".join(sorted(ont.relation_types))
    edges = "; ".join(
        f"{src} -{rel}-> {dst}" for src, rel, dst in sorted(ont.allowed_edges)
    )
    return (
        "You extract a business-strategy knowledge graph from text. "
        "Use only the provided ontology. Do not invent types. "
        "Return a single JSON object with keys entities and relationships. "
        "entities items: type, name, description, aliases. "
        "relationships items: type, source_type, source_name, target_type, target_name. "
        "Names in relationships must match extracted entity names. "
        f"Entity types: {entities}. "
        f"Relationship types: {relations}. "
        f"Allowed edges: {edges}."
    )


def extraction_user_prompt(text: str, *, source: str | None = None) -> str:
    header = f"Source: {source}\n\n" if source else ""
    return f"{header}Text:\n{text}\n"
