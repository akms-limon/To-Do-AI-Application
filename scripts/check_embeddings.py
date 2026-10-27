from math import sqrt

from app.embeddings import embed_texts


def cosine_similarity(left, right):
    dot_product = sum(a * b for a, b in zip(left, right))
    left_length = sqrt(sum(value * value for value in left))
    right_length = sqrt(sum(value * value for value in right))

    return dot_product / (left_length * right_length)


def main():
    texts = [
        "Write documentation for the API",
        "Update the developer API guide",
        "Buy vegetables for dinner",
    ]

    print("Loading MiniLM and generating embeddings...")
    vectors = embed_texts(texts)

    for text, vector in zip(texts, vectors):
        assert len(vector) == 384
        print(f"{len(vector)} dimensions: {text}")

    related_score = cosine_similarity(vectors[0], vectors[1])
    unrelated_score = cosine_similarity(vectors[0], vectors[2])

    print(f"\nRelated texts:   {related_score:.3f}")
    print(f"Unrelated texts: {unrelated_score:.3f}")

    if related_score > unrelated_score:
        print("The related example ranks higher.")
    else:
        print("Unexpected ranking; investigate before integrating search.")


if __name__ == "__main__":
    main()