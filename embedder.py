# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp

from __future__ import annotations
import logging
import os
import threading
import hashlib
import math
import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

_FAKE_EMBEDDINGS = os.getenv("RAGMAX_FAKE_EMBEDDINGS", "").lower() in {
    "1", "true", "yes", "on"
}

if not _FAKE_EMBEDDINGS:
    from sentence_transformers import SentenceTransformer
else:
    SentenceTransformer = object  # type: ignore[misc,assignment]

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

_model: SentenceTransformer | None = None
_lock = threading.Lock()


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                try:
                    # Avoid a network lookup when the model is already cached;
                    # this keeps offline/public installs deterministic.
                    _model = SentenceTransformer(MODEL_NAME, local_files_only=True)
                except (OSError, ValueError):
                    # First use may legitimately need to populate the cache.
                    _model = SentenceTransformer(MODEL_NAME)
    return _model


def encode(texts: list[str]) -> list[list[float]]:
    """Encode texts to dense vectors. Returns list of float lists."""
    if not texts:
        return []
    if _FAKE_EMBEDDINGS:
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            raw = [((digest[i % len(digest)] / 255.0) * 2.0) - 1.0
                   for i in range(384)]
            norm = math.sqrt(sum(value * value for value in raw)) or 1.0
            vectors.append([value / norm for value in raw])
        return vectors
    model = _get_model()
    vecs = model.encode(texts, normalize_embeddings=True,
                        show_progress_bar=False, convert_to_numpy=True)
    return vecs.tolist()


def encode_one(text: str) -> list[float]:
    return encode([text])[0]
