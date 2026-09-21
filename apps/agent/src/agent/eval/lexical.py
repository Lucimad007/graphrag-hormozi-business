from collections.abc import Sequence
from hashlib import sha256


class LexicalEmbeddings:
    """Stable bag-of-tokens vectors for deterministic retrieval evaluation."""

    def __init__(self, dimensions: int = 48) -> None:
        if dimensions < 8:
            raise ValueError("dimensions must be >= 8")
        self.dimensions = dimensions

    def embed(self, texts: Sequence[str], *, input_type: str | None = None) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        values = [0.0] * self.dimensions
        for token in text.lower().split():
            digest = sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:2], "big") % self.dimensions
            values[index] += 1.0
        return values
