from functools import lru_cache

from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2


@lru_cache(maxsize=1)
def get_embedding_model():
    return ONNXMiniLM_L6_V2(
        preferred_providers=["CPUExecutionProvider"]
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        raise ValueError("Provide at least one text.")

    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Every text must be a nonempty string.")

    clean_texts = [text.strip() for text in texts]
    vectors = get_embedding_model()(input=clean_texts)

    return [
        [float(value) for value in vector]
        for vector in vectors
    ]